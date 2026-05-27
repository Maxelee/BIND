"""High-level orchestration for multi-simulation test-suite runs."""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
import torch

from data import NormStats, N_THERMO, THERMO_KEYS
from train import FlowMatchingLit

from .artifacts import (
    ensure_dirs,
    load_composite,
    load_full_maps,
    load_generated_halos,
    load_halo_catalog,
    load_halo_cutouts,
    load_truth_halos_cube,
    load_truth_thermo_patches,
    resolve_artifact_paths,
    save_composite,
    save_full_maps,
    save_generated_halos,
    save_halo_catalog,
    save_halo_cutouts,
    save_summary_json,
    save_truth_halos_cube,
    save_truth_thermo_patches,
)
from .pipeline import (
    build_bind_composite,
    compute_per_halo_mass_error,
    compute_truth_thermo_patches,
    extract_halo_cutouts,
    extract_halo_cutouts_cube,
    extract_halo_cutouts_cube_from_3d,
    extract_truth_cutouts_cube_from_3d,
    generate_halo_patches,
    load_dmo_particles,
    load_dmo_projection,
    load_halo_catalog as load_halo_catalog_raw,
    load_truth_maps,
    voxelize_dmo_3d,
)
from .schemas import RunConfig, SimulationSpec


def _resolve_device(device_name: str) -> torch.device:
    """Resolve device string to torch.device with auto fallback."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _thermo_patch_metrics(gen_thermo: np.ndarray, truth_thermo: np.ndarray) -> dict:
    """Per-channel log10(gen/truth) bias & scatter over jointly-positive pixels.

    Thermo fields span many orders of magnitude, so a dex (log10) bias/scatter
    is the natural per-pixel error metric.  gen_thermo / truth_thermo are
    (N, N_THERMO, H, W) in THERMO_KEYS order.  Returns a dict keyed by channel
    name with n_pix, log10_bias_median, and log10_scatter_std.
    """
    metrics: dict = {}
    for j, key in enumerate(THERMO_KEYS):
        g = gen_thermo[:, j].ravel()
        t = truth_thermo[:, j].ravel()
        mask = (g > 0) & (t > 0)
        if not mask.any():
            metrics[key] = {"n_pix": 0, "log10_bias_median": None,
                            "log10_scatter_std": None}
            continue
        d = np.log10(g[mask]) - np.log10(t[mask])
        metrics[key] = {
            "n_pix": int(mask.sum()),
            "log10_bias_median": float(np.median(d)),
            "log10_scatter_std": float(np.std(d)),
        }
    return metrics


def load_model_bundle(run_cfg: RunConfig) -> tuple:
    """Load norm stats and model checkpoint once for all simulations.

    Returns (norm_stats, fm, device, param_indices, no_large_scale, predict_thermo).
    """
    device = _resolve_device(run_cfg.device)
    norm_stats = NormStats.load(run_cfg.run_dir / "norm_stats.npz")

    # Optional post-training per-channel calibration.
    # Multiplicative factors in physical space map to additive shifts in log-space.
    if run_cfg.channel_correction is not None:
        corr = np.asarray(run_cfg.channel_correction, dtype=np.float32)
        if corr.shape != (3,):
            raise ValueError(
                f"channel_correction must have shape (3,), got {corr.shape}"
            )
        if not np.all(np.isfinite(corr)) or np.any(corr <= 0):
            raise ValueError("channel_correction must be finite and > 0 for all channels")
        norm_stats.target_mean = norm_stats.target_mean.copy()
        norm_stats.target_mean += np.log10(corr)

    model = FlowMatchingLit.load_from_checkpoint(run_cfg.checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    # Never apply EMA — always use the raw checkpoint weights.
    # fm_two_head was trained before EMA saving was added; its shadow_params are
    # corrupt/zero and would overwrite the valid raw weights.

    # Auto-detect param_indices from hparams so models trained with
    # --exclude_cosmo_params (n_params=31) receive the correct filtered vector.
    _COSMO_INDICES = [0, 1, 7, 8]
    n_params = getattr(model.hparams, "n_params", 35)
    param_indices = (
        np.array([i for i in range(35) if i not in _COSMO_INDICES])
        if n_params < 35
        else None
    )
    no_large_scale = bool(getattr(model.hparams, "no_large_scale", False))
    # predict_thermo is authoritative from norm_stats (the model emits the extra
    # channels iff the stats carry thermo normalization).
    predict_thermo = bool(getattr(norm_stats, "predict_thermo", False))
    return norm_stats, model.fm, device, param_indices, no_large_scale, predict_thermo


def _prepare_data(
    spec: SimulationSpec,
    run_cfg: RunConfig,
    load_truth: bool,
    no_large_scale: bool = False,
    predict_thermo: bool = False,
) -> tuple[dict, object]:
    """Load from cache or prepare DMO/halo/cutout artifacts."""
    paths = resolve_artifact_paths(run_cfg.output_root, spec, run_cfg.model_name)
    ensure_dirs(paths)

    if paths.full_maps_npz.exists() and not run_cfg.regenerate_all:
        dmo_fullbox, truth_maps = load_full_maps(paths.full_maps_npz)
    else:
        dmo_fullbox = load_dmo_projection(spec)
        truth_maps = load_truth_maps(spec) if load_truth else None
        save_full_maps(paths.full_maps_npz, dmo_fullbox, truth_maps)

    if (
        paths.halo_catalog_npz.exists()
        and not run_cfg.regenerate_all
    ):
        halos, halo_masses, halo_r200s, halo_positions = load_halo_catalog(paths.halo_catalog_npz)
        # Old cached catalogs lack R200 data; reload from FOF if needed.
        if run_cfg.r200_factor > 0 and all(h.get("r200", 0.0) == 0.0 for h in halos):
            halos, halo_masses, halo_r200s, halo_positions = load_halo_catalog_raw(spec)
    else:
        halos, halo_masses, halo_r200s, halo_positions = load_halo_catalog_raw(spec)
        save_halo_catalog(paths.halo_catalog_npz, halos, halo_masses, halo_r200s, halo_positions)

    # ── Cutout extraction ────────────────────────────────────────────────────
    if no_large_scale:
        # Cube model: replicate training-data geometry exactly.
        # 1. Voxelize the full DMO box to 1024^3 with MASL CIC.
        # 2. Extract a 128^3 sub-cube centred on each halo voxel (periodic BC).
        # 3. Sum along z → 128×128 DM mass map per halo.
        # This matches how the cube training files were generated.
        cutouts_path = paths.halo_cutouts_cube_npz
        if cutouts_path.exists() and not run_cfg.regenerate_all:
            halo_cutouts = load_halo_cutouts(cutouts_path)
        else:
            dmo_particles, particle_mass = load_dmo_particles(spec)
            field3d = voxelize_dmo_3d(
                dmo_particles, particle_mass, spec.box_size, spec.npix
            )
            del dmo_particles  # free ~several GB before per-halo extraction
            halo_cutouts = extract_halo_cutouts_cube_from_3d(
                field3d,
                halos,
                halo_positions,
                box_size=spec.box_size,
                patch_pix=spec.patch_pix,
            )
            del field3d
            save_halo_cutouts(cutouts_path, halo_cutouts)

        # ── Cube hydro truth patches ─────────────────────────────────────────
        # Voxelize each hydro species (DM, Gas, Stars) to 3D using the same
        # MASL CIC procedure as the DMO condition, then extract 128^3 per halo
        # and sum along z.  This gives truth patches in the exact same geometry
        # as the training targets, enabling direct per-halo comparison.
        truth_halos: np.ndarray | None = None
        if load_truth:
            truth_path = paths.truth_halos_cube_npz
            if truth_path.exists() and not run_cfg.regenerate_all:
                truth_halos = load_truth_halos_cube(truth_path)
            else:
                truth_halos = extract_truth_cutouts_cube_from_3d(
                    spec, halos, halo_positions
                )
                save_truth_halos_cube(truth_path, truth_halos)
    else:
        cutouts_path = paths.halo_cutouts_npz
        if cutouts_path.exists() and not run_cfg.regenerate_all:
            halo_cutouts = load_halo_cutouts(cutouts_path)
        else:
            halo_cutouts = extract_halo_cutouts(
                dmo_fullbox,
                halos,
                box_size=spec.box_size,
                npix=spec.npix,
                patch_pix=spec.patch_pix,
            )
            save_halo_cutouts(cutouts_path, halo_cutouts)
        truth_halos = None  # not computed for standard (large-scale) models

    # ── Per-halo truth thermo patches (snapshot reprojection) ────────────────
    # Reconstructs the 4 gas-thermo fields from the hydro snapshot with the same
    # recipe as the training targets, axis-aligned and cut at each halo center
    # so the patches register with the generated ones. Only needed for models
    # that predict thermo. Not done in prep_only (predict_thermo is unknown
    # without the model/norm_stats); it is computed lazily on the first run.
    truth_thermo: np.ndarray | None = None
    if load_truth and predict_thermo:
        thermo_path = paths.truth_thermo_patches_npz
        if thermo_path.exists() and not run_cfg.regenerate_all:
            truth_thermo = load_truth_thermo_patches(thermo_path)
        else:
            truth_thermo = compute_truth_thermo_patches(spec, halos)
            save_truth_thermo_patches(thermo_path, truth_thermo)

    payload = {
        "dmo_fullbox": dmo_fullbox,
        "truth_maps": truth_maps,
        "truth_halos": truth_halos,
        "truth_thermo": truth_thermo,
        "halos": halos,
        "halo_masses": halo_masses,
        "halo_positions": halo_positions,
        "halo_cutouts": halo_cutouts,
        "artifact_paths": paths,
    }
    return payload, paths


def run_single_simulation(
    spec: SimulationSpec,
    run_cfg: RunConfig,
    norm_stats: NormStats | None,
    fm,
    device: torch.device | None,
    load_truth: bool,
    param_indices: np.ndarray | None = None,
    no_large_scale: bool = False,
    predict_thermo: bool = False,
) -> dict:
    """Run one simulation through prepare/generate/paste stages."""
    prepared, paths = _prepare_data(spec, run_cfg, load_truth=load_truth,
                                    no_large_scale=no_large_scale,
                                    predict_thermo=predict_thermo)

    dmo_fullbox = prepared["dmo_fullbox"]
    truth_maps = prepared["truth_maps"]
    halos = prepared["halos"]
    halo_cutouts = prepared["halo_cutouts"]

    summary = {
        "suite": spec.suite,
        "sim_id": spec.sim_id,
        "snapshot": spec.snapshot,
        "n_halos": int(len(halos)),
        "dmo_total_mass": float(dmo_fullbox.sum()),
        "truth_total_mass": float(truth_maps.sum()) if truth_maps is not None else None,
        "run_config": asdict(run_cfg),
    }

    if run_cfg.prep_only:
        save_summary_json(paths.summary_json, summary)
        return summary

    assert norm_stats is not None and fm is not None and device is not None

    if paths.generated_halos_npz.exists() and not (run_cfg.regenerate or run_cfg.regenerate_all):
        generated_halos = load_generated_halos(paths.generated_halos_npz)
    else:
        generated_halos = generate_halo_patches(
            halo_cutouts,
            norm_stats,
            spec.params,
            fm,
            device,
            n_steps=run_cfg.n_steps,
            batch_size=run_cfg.batch_size,
            use_amp=run_cfg.use_amp,
            param_indices=param_indices,
            no_large_scale=no_large_scale,
        )
        save_generated_halos(paths.generated_halos_npz, generated_halos)

    if paths.composite_npz.exists() and not (run_cfg.repaste or run_cfg.regenerate or run_cfg.regenerate_all):
        composite_loaded = load_composite(paths.composite_npz)
        bind_composite = composite_loaded["composite"]
        rel_err = composite_loaded["mass_rel_err"]
        if rel_err.size > 0:
            mass_mean = float(np.mean(rel_err) * 100.0)
            mass_std = float(np.std(rel_err) * 100.0)
            mass_median = float(np.median(rel_err) * 100.0)
        else:
            mass_mean = mass_std = mass_median = 0.0
        summary.update(
            {
                "coverage_pct": float(composite_loaded["coverage_pct"]),
                "scale_global": float(composite_loaded["scale_global"]),
                "bind_total_mass": float(bind_composite.sum()),
                "mass_error_mean_pct": mass_mean,
                "mass_error_std_pct": mass_std,
                "mass_error_median_pct": mass_median,
            }
        )
    else:
        composite_bundle = build_bind_composite(
            dmo_fullbox,
            halos,
            generated_halos[:, :3],  # mass channels only; thermo is not composited
            halo_cutouts,
            box_size=spec.box_size,
            npix=spec.npix,
            patch_pix=spec.patch_pix,
            patch_mass_match=run_cfg.patch_mass_match,
            taper_frac=run_cfg.taper_frac,
            r200_factor=run_cfg.r200_factor,
        )
        mass_stats = compute_per_halo_mass_error(
            dmo_fullbox,
            composite_bundle["composite"],
            halos,
            box_size=spec.box_size,
            npix=spec.npix,
            patch_pix=spec.patch_pix,
        )
        save_composite(paths.composite_npz, composite_bundle, mass_stats)

        summary.update(
            {
                "coverage_pct": float(composite_bundle["coverage_pct"]),
                "scale_global": float(composite_bundle["scale_global"]),
                "bind_total_mass": float(composite_bundle["composite"].sum()),
                "mass_error_mean_pct": float(mass_stats["mean_pct"]),
                "mass_error_std_pct": float(mass_stats["std_pct"]),
                "mass_error_median_pct": float(mass_stats["median_pct"]),
            }
        )

    # ── Per-halo thermo evaluation (dex bias/scatter vs truth) ───────────────
    # Thermo fields live in the trailing channels of generated_halos and are
    # evaluated per-halo (not composited): compton_y is extensive while
    # temperature/pressure/entropy are intensive mass-weighted means, so the
    # mass-conservation composite does not apply to them.
    if predict_thermo:
        truth_thermo = prepared.get("truth_thermo")
        has_thermo_channels = generated_halos.shape[1] >= 3 + N_THERMO
        gen_thermo = (
            generated_halos[:, 3:3 + N_THERMO] if has_thermo_channels else None
        )
        if (
            truth_thermo is not None and gen_thermo is not None
            and len(truth_thermo) == len(gen_thermo) and len(gen_thermo) > 0
        ):
            summary["thermo_metrics"] = _thermo_patch_metrics(gen_thermo, truth_thermo)

    save_summary_json(paths.summary_json, summary)
    return summary


def run_suite(
    specs: list[SimulationSpec],
    run_cfg: RunConfig,
    load_truth: bool = True,
    max_workers: int = 1,
) -> list[dict]:
    """Execute suite processing for all selected simulations."""
    if not specs:
        return []

    results: list[dict] = []

    if run_cfg.prep_only and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_simulation, spec, run_cfg, None, None, None, load_truth): spec
                for spec in specs
            }
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    summary = fut.result()
                    results.append(summary)
                    print(f"[prep] {spec.sim_label}: {summary['n_halos']} halos")
                except Exception as exc:
                    print(f"[prep] {spec.sim_label}: failed with {exc}")
                    traceback.print_exc()
        return results

    norm_stats = fm = device = None
    param_indices = None
    no_large_scale = False
    predict_thermo = False
    if not run_cfg.prep_only:
        (norm_stats, fm, device, param_indices,
         no_large_scale, predict_thermo) = load_model_bundle(run_cfg)

    for spec in specs:
        try:
            summary = run_single_simulation(spec, run_cfg, norm_stats, fm, device, load_truth,
                                            param_indices=param_indices,
                                            no_large_scale=no_large_scale,
                                            predict_thermo=predict_thermo)
            results.append(summary)
            print(
                f"[{spec.sim_label}] halos={summary['n_halos']} "
                f"bind_total={summary.get('bind_total_mass', 0.0):.3e}"
            )
        except Exception as exc:
            print(f"[{spec.sim_label}] failed with {exc}")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results
