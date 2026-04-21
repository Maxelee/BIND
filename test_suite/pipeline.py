"""Notebook-equivalent DMO->hydro pipeline primitives."""

from __future__ import annotations

import glob
from contextlib import nullcontext
from pathlib import Path

import h5py
import MAS_library as MASL
import numpy as np
import torch
from tqdm import tqdm

from data import NormStats, log_transform
from .schemas import SimulationSpec


def pixelize_z_projection(
    positions: np.ndarray,
    masses: np.ndarray,
    box_size: float,
    npix: int,
) -> np.ndarray:
    """Project particle masses onto a 2D grid with CIC assignment."""
    pos_ = np.ascontiguousarray(positions.astype(np.float32))[:, [0, 1]]
    mass_ = np.ascontiguousarray(masses.astype(np.float32))
    field = np.zeros((npix, npix), dtype=np.float32)
    MASL.MA(pos_, field, box_size, MAS="CIC", W=mass_, verbose=False)
    return field


def _nbody_snapshot_file(nbody_path: Path, snapshot: int) -> Path:
    single = nbody_path / f"snap_{snapshot:03d}.hdf5"
    if single.exists():
        return single
    chunk0 = nbody_path / f"snapdir_{snapshot:03d}" / f"snap_{snapshot:03d}.0.hdf5"
    if chunk0.exists():
        return chunk0
    raise FileNotFoundError(f"Could not find DMO snapshot for {nbody_path} snapshot {snapshot}")


def load_dmo_projection(spec: SimulationSpec) -> np.ndarray:
    """Load DMO particles and project to 2D full-box map."""
    nbody_snap = _nbody_snapshot_file(spec.nbody_path, spec.snapshot)
    with h5py.File(nbody_snap, "r") as handle:
        dmo_pos = handle["PartType1/Coordinates"][:] / 1000.0
        mt = handle["Header"].attrs["MassTable"]
        dmo_mass_arr = np.full(len(dmo_pos), mt[1] * 1e10, dtype=np.float32)

    if spec.proj_frac < 1.0:
        mask = dmo_pos[:, 2] < spec.box_size * spec.proj_frac
        dmo_pos = dmo_pos[mask]
        dmo_mass_arr = dmo_mass_arr[mask]

    return pixelize_z_projection(dmo_pos, dmo_mass_arr, spec.box_size, spec.npix)


def load_halo_catalog(spec: SimulationSpec) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load FoF group catalog, apply halo mass cut, and build halo list."""
    patt = spec.group_catalog / f"fof_subhalo_tab_{spec.snapshot:03d}.*.hdf5"
    files = sorted(glob.glob(str(patt)))
    if not files:
        raise FileNotFoundError(f"No FoF group files found with pattern {patt}")

    all_masses: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []
    for fname in files:
        with h5py.File(fname, "r") as handle:
            if "Group/GroupMass" not in handle:
                continue
            all_masses.append(handle["Group/GroupMass"][:])
            all_positions.append(handle["Group/GroupPos"][:])

    if not all_masses:
        raise RuntimeError(f"Group catalog found but no GroupMass datasets in {spec.group_catalog}")

    masses = np.concatenate(all_masses) * 1e10
    positions = np.concatenate(all_positions) / 1e3

    mask = masses > spec.halo_mass_min
    halo_masses = masses[mask].astype(np.float32)
    halo_positions = positions[mask].astype(np.float32)

    halos = [
        {
            "halo_center": pos[:2],
            "halo_mass": float(mass),
            "params": spec.params,
        }
        for pos, mass in zip(halo_positions, halo_masses)
    ]
    return halos, halo_masses, halo_positions


def extract_periodic_cutout(field: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """Extract square cutout with periodic boundaries."""
    n = field.shape[0]
    half = size // 2
    ix = (cx - half + np.arange(size)) % n
    iy = (cy - half + np.arange(size)) % n
    return field[np.ix_(ix, iy)]


def extract_multiscale(dmo_map: np.ndarray, cx_pix: int, cy_pix: int, target_res: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract condition patch and three large-scale context patches."""
    full_res = dmo_map.shape[0]
    scales_pix = [target_res, target_res * 2, target_res * 4, full_res]
    result = np.zeros((4, target_res, target_res), dtype=np.float32)

    for i, spx in enumerate(scales_pix):
        cutout = extract_periodic_cutout(dmo_map, cx_pix, cy_pix, spx)
        if spx == target_res:
            result[i] = cutout
            continue

        factor = spx // target_res
        result[i] = cutout.reshape(target_res, factor, target_res, factor).mean(axis=(1, 3))

    return result[0], result[1:]


def extract_halo_cutouts(
    dmo_fullbox: np.ndarray,
    halos: list[dict],
    box_size: float,
    npix: int,
    patch_pix: int,
) -> list[dict]:
    """Extract all multiscale DMO cutouts at halo centers."""
    pixels_per_mpc = npix / box_size
    halo_cutouts: list[dict] = []
    for halo in tqdm(halos, desc="Extracting DMO cutouts"):
        cx = int(halo["halo_center"][0] * pixels_per_mpc) % npix
        cy = int(halo["halo_center"][1] * pixels_per_mpc) % npix
        cond_cut, ls_cut = extract_multiscale(dmo_fullbox, cx, cy, target_res=patch_pix)
        halo_cutouts.append({"condition": cond_cut, "large_scale": ls_cut})
    return halo_cutouts


def normalize_cutout(hc: dict, ns: NormStats, sim_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize one halo cutout and associated parameter vector."""
    condition = log_transform(hc["condition"])[None]
    condition = (condition - ns.cond_mean) / (ns.cond_std + 1e-8)

    large_scale = log_transform(hc["large_scale"])
    large_scale = (large_scale - ns.ls_mean[:, None, None]) / (ns.ls_std[:, None, None] + 1e-8)

    rang = ns.param_max - ns.param_min
    params = (sim_params.astype(np.float32) - ns.param_min) / (rang + 1e-8)
    return condition, large_scale, params


def generate_halo_patches(
    halo_cutouts: list[dict],
    norm_stats: NormStats,
    sim_params: np.ndarray,
    fm,
    device: torch.device,
    n_steps: int,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    """Run model inference on all halo cutouts and denormalize to physical space."""
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(halo_cutouts), batch_size), desc="Generating hydro"):
            batch = halo_cutouts[start : start + batch_size]
            conds, lss, params = zip(*[normalize_cutout(hc, norm_stats, sim_params) for hc in batch])

            cond_t = torch.from_numpy(np.stack(conds).astype(np.float32)).to(device)
            ls_t = torch.from_numpy(np.stack(lss).astype(np.float32)).to(device)
            params_t = torch.from_numpy(np.stack(params).astype(np.float32)).to(device)

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                gen = fm.sample(cond_t, ls_t, params_t, n_steps=n_steps)

            gen_np = gen.float().cpu().numpy().astype(np.float32)
            for ch in range(3):
                gen_np[:, ch] = gen_np[:, ch] * norm_stats.target_std[ch] + norm_stats.target_mean[ch]
                gen_np[:, ch] = 10.0 ** gen_np[:, ch] - 1.0
            gen_np = np.clip(gen_np, 0, None)
            outputs.append(gen_np)

    if not outputs:
        return np.zeros((0, 3, 0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def square_taper_weight(patch_size: int, taper_frac: float = 0.15) -> np.ndarray:
    """2D separable Hann taper to blend edges of square patches."""
    t = max(1, int(patch_size * taper_frac))
    hann = 0.5 * (1 - np.cos(np.pi * np.arange(t) / t)).astype(np.float32)
    w1d = np.ones(patch_size, dtype=np.float32)
    w1d[:t] = hann
    w1d[-t:] = hann[::-1]
    return np.outer(w1d, w1d)


def paste_halos_2d(
    canvas_res: int,
    box_size: float,
    halos: list[dict],
    patches: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paste halo patches onto full box with overlap-aware weighted blending."""
    canvas = np.zeros((3, canvas_res, canvas_res), dtype=np.float32)
    w_accum = np.zeros((canvas_res, canvas_res), dtype=np.float32)

    pixels_per_mpc = canvas_res / box_size
    half = weight.shape[0] // 2

    for halo, patch in zip(halos, patches):
        cx = int(halo["halo_center"][0] * pixels_per_mpc) % canvas_res
        cy = int(halo["halo_center"][1] * pixels_per_mpc) % canvas_res
        ix = (cx - half + np.arange(weight.shape[0])) % canvas_res
        iy = (cy - half + np.arange(weight.shape[0])) % canvas_res

        for ch in range(3):
            canvas[ch][np.ix_(ix, iy)] += patch[ch] * weight
        w_accum[np.ix_(ix, iy)] += weight

    safe_w = np.where(w_accum > 0, w_accum, 1.0)
    canvas /= safe_w[None]
    return canvas, w_accum


def build_bind_composite(
    dmo_fullbox: np.ndarray,
    halos: list[dict],
    generated_patches: np.ndarray,
    halo_cutouts: list[dict],
    box_size: float,
    npix: int,
    patch_pix: int,
    patch_mass_match: bool,
    taper_frac: float,
) -> dict:
    """Construct BIND composite map using notebook-consistent blending logic."""
    patches = []
    patch_scales = []

    for patch, hc in zip(generated_patches, halo_cutouts):
        p = patch.copy()
        if patch_mass_match:
            m_pred = float(p.sum())
            m_dmo = float(hc["condition"].sum())
            s = m_dmo / (m_pred + 1e-30)
            p *= s
            patch_scales.append(s)
        patches.append(p)

    patches_np = np.asarray(patches, dtype=np.float32)
    taper = square_taper_weight(patch_pix, taper_frac=taper_frac)
    hydro_canvas, hydro_weights = paste_halos_2d(npix, box_size, halos, patches_np, taper)

    alpha = np.clip(hydro_weights, 0.0, 1.0)
    bind_composite = np.zeros((3, npix, npix), dtype=np.float32)
    bind_composite[0] = (1 - alpha) * dmo_fullbox + alpha * hydro_canvas[0]
    bind_composite[1] = alpha * hydro_canvas[1]
    bind_composite[2] = alpha * hydro_canvas[2]

    scale_global = float(dmo_fullbox.sum() / (bind_composite.sum() + 1e-30))
    bind_composite *= scale_global
    coverage = float((alpha > 0.01).mean() * 100.0)

    return {
        "composite": bind_composite,
        "alpha": alpha,
        "hydro_canvas": hydro_canvas,
        "hydro_weights": hydro_weights,
        "patch_scales": np.asarray(patch_scales, dtype=np.float64),
        "scale_global": scale_global,
        "coverage_pct": coverage,
    }


def compute_per_halo_mass_error(
    dmo_fullbox: np.ndarray,
    bind_composite: np.ndarray,
    halos: list[dict],
    box_size: float,
    npix: int,
    patch_pix: int,
) -> dict:
    """Compute per-halo total mass conservation diagnostics."""
    if not halos:
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "dmo_halo_mass": empty,
            "bind_halo_mass": empty,
            "rel_err": empty,
            "mean_pct": 0.0,
            "std_pct": 0.0,
            "median_pct": 0.0,
        }

    pixels_per_mpc = npix / box_size
    half = patch_pix // 2

    dmo_halo_mass = []
    bind_halo_mass = []
    for halo in halos:
        cx = int(halo["halo_center"][0] * pixels_per_mpc) % npix
        cy = int(halo["halo_center"][1] * pixels_per_mpc) % npix
        ix = (cx - half + np.arange(patch_pix)) % npix
        iy = (cy - half + np.arange(patch_pix)) % npix

        m_dmo = float(dmo_fullbox[np.ix_(ix, iy)].sum())
        m_bind = float(sum(bind_composite[ch][np.ix_(ix, iy)].sum() for ch in range(3)))
        dmo_halo_mass.append(m_dmo)
        bind_halo_mass.append(m_bind)

    dmo_halo_mass_np = np.asarray(dmo_halo_mass, dtype=np.float64)
    bind_halo_mass_np = np.asarray(bind_halo_mass, dtype=np.float64)
    rel_err = (bind_halo_mass_np - dmo_halo_mass_np) / (dmo_halo_mass_np + 1e-30)

    return {
        "dmo_halo_mass": dmo_halo_mass_np,
        "bind_halo_mass": bind_halo_mass_np,
        "rel_err": rel_err,
        "mean_pct": float(rel_err.mean() * 100.0),
        "std_pct": float(rel_err.std() * 100.0),
        "median_pct": float(np.median(rel_err) * 100.0),
    }


def _project_species(pos_list: list[np.ndarray], mass_list: list[np.ndarray], box_size: float, npix: int) -> np.ndarray:
    if not pos_list:
        return np.zeros((npix, npix), dtype=np.float32)
    pos = np.concatenate(pos_list, axis=0) / 1000.0
    mass = np.concatenate(mass_list, axis=0) * 1e10
    return pixelize_z_projection(pos, mass.astype(np.float32), box_size, npix)


def load_truth_maps(spec: SimulationSpec) -> np.ndarray:
    """Load hydro species from snapshot chunks and project to 2D maps."""
    pattern = spec.hydro_snapdir / f"snap_{spec.snapshot:03d}.*.hdf5"
    snap_files = sorted(glob.glob(str(pattern)))
    if not snap_files:
        single = spec.hydro_snapdir / f"snap_{spec.snapshot:03d}.hdf5"
        if single.exists():
            snap_files = [str(single)]

    if not snap_files:
        raise FileNotFoundError(f"No hydro snapshots found with pattern {pattern}")

    hydro_dm_pos: list[np.ndarray] = []
    hydro_dm_mass: list[np.ndarray] = []
    gas_pos: list[np.ndarray] = []
    gas_mass: list[np.ndarray] = []
    star_pos: list[np.ndarray] = []
    star_mass: list[np.ndarray] = []

    for fname in snap_files:
        with h5py.File(fname, "r") as handle:
            if "PartType1/Coordinates" in handle:
                hydro_dm_pos.append(handle["PartType1/Coordinates"][:])
                mt = handle["Header"].attrs["MassTable"]
                n = len(hydro_dm_pos[-1])
                hydro_dm_mass.append(
                    handle["PartType1/Masses"][:] if "PartType1/Masses" in handle else np.full(n, mt[1], dtype=np.float32)
                )

            if "PartType0/Coordinates" in handle:
                gas_pos.append(handle["PartType0/Coordinates"][:])
                gas_mass.append(handle["PartType0/Masses"][:])

            if "PartType4/Coordinates" in handle:
                star_pos.append(handle["PartType4/Coordinates"][:])
                star_mass.append(handle["PartType4/Masses"][:])

    truth_dm = _project_species(hydro_dm_pos, hydro_dm_mass, spec.box_size, spec.npix)
    truth_gas = _project_species(gas_pos, gas_mass, spec.box_size, spec.npix)
    truth_star = _project_species(star_pos, star_mass, spec.box_size, spec.npix)
    return np.stack([truth_dm, truth_gas, truth_star]).astype(np.float32)
