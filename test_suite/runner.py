"""High-level orchestration for multi-simulation test-suite runs."""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data import NormStats
from train import FlowMatchingLit

from .artifacts import (
    ensure_dirs,
    load_composite,
    load_full_maps,
    load_generated_halos,
    load_halo_catalog,
    load_halo_cutouts,
    resolve_artifact_paths,
    save_composite,
    save_full_maps,
    save_generated_halos,
    save_halo_catalog,
    save_halo_cutouts,
    save_summary_json,
)
from .pipeline import (
    build_bind_composite,
    compute_per_halo_mass_error,
    extract_halo_cutouts,
    generate_halo_patches,
    load_dmo_projection,
    load_halo_catalog as load_halo_catalog_raw,
    load_truth_maps,
)
from .schemas import RunConfig, SimulationSpec


def _resolve_device(device_name: str) -> torch.device:
    """Resolve device string to torch.device with auto fallback."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def load_model_bundle(run_cfg: RunConfig) -> tuple[NormStats, object, torch.device]:
    """Load norm stats and model checkpoint once for all simulations."""
    device = _resolve_device(run_cfg.device)
    norm_stats = NormStats.load(run_cfg.run_dir / "norm_stats.npz")
    model = FlowMatchingLit.load_from_checkpoint(run_cfg.checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return norm_stats, model.fm, device


def _prepare_data(
    spec: SimulationSpec,
    run_cfg: RunConfig,
    load_truth: bool,
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
        and paths.halo_cutouts_npz.exists()
        and not run_cfg.regenerate_all
    ):
        halos, halo_masses, halo_positions = load_halo_catalog(paths.halo_catalog_npz)
        halo_cutouts = load_halo_cutouts(paths.halo_cutouts_npz)
    else:
        halos, halo_masses, halo_positions = load_halo_catalog_raw(spec)
        halo_cutouts = extract_halo_cutouts(
            dmo_fullbox,
            halos,
            box_size=spec.box_size,
            npix=spec.npix,
            patch_pix=spec.patch_pix,
        )
        save_halo_catalog(paths.halo_catalog_npz, halos, halo_masses, halo_positions)
        save_halo_cutouts(paths.halo_cutouts_npz, halo_cutouts)

    payload = {
        "dmo_fullbox": dmo_fullbox,
        "truth_maps": truth_maps,
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
) -> dict:
    """Run one simulation through prepare/generate/paste stages."""
    prepared, paths = _prepare_data(spec, run_cfg, load_truth=load_truth)

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
            generated_halos,
            halo_cutouts,
            box_size=spec.box_size,
            npix=spec.npix,
            patch_pix=spec.patch_pix,
            patch_mass_match=run_cfg.patch_mass_match,
            taper_frac=run_cfg.taper_frac,
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
    if not run_cfg.prep_only:
        norm_stats, fm, device = load_model_bundle(run_cfg)

    for spec in specs:
        try:
            summary = run_single_simulation(spec, run_cfg, norm_stats, fm, device, load_truth)
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
