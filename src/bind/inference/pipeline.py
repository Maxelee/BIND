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

from bind.data import NormStats, log_transform
from .schemas import SimulationSpec


def pixelize_z_projection(
    positions: np.ndarray,
    masses: np.ndarray,
    box_size: float,
    npix: int,
) -> np.ndarray:
    """Project particle masses onto a 2D grid with CIC assignment via Pylians."""
    pos_ = np.ascontiguousarray(positions.astype(np.float32))[:, [0, 1]]
    mass_ = np.ascontiguousarray(masses.astype(np.float32))
    field = np.zeros((npix, npix), dtype=np.float32)
    MASL.MA(pos_, field, box_size, MAS="CIC", W=mass_, verbose=False)
    return field


def _dmo_snapshot_files(nbody_path: Path, snapshot: int) -> list[str]:
    """Return sorted list of DMO snapshot files (single-file or multi-chunk)."""
    single = nbody_path / f"snap_{snapshot:03d}.hdf5"
    if single.exists():
        return [str(single)]
    pattern = nbody_path / f"snapdir_{snapshot:03d}" / f"snap_{snapshot:03d}.*.hdf5"
    files = sorted(glob.glob(str(pattern)))
    if files:
        return files
    raise FileNotFoundError(f"Could not find DMO snapshot for {nbody_path} snapshot {snapshot}")


def load_dmo_projection(spec: SimulationSpec) -> np.ndarray:
    """Load DMO particles and project to 2D full-box map.

    For suites without a separate N-body run (e.g. SB35), nbody_path should
    point to the hydro simulation root — PartType1 (DM) is read from there.
    Multi-chunk snapshots are fully concatenated before projection.
    """
    snap_files = _dmo_snapshot_files(spec.nbody_path, spec.snapshot)

    pos_chunks: list[np.ndarray] = []
    dm_particle_mass: float | None = None
    for fname in snap_files:
        with h5py.File(fname, "r") as handle:
            pos_chunks.append(handle["PartType1/Coordinates"][:])
            if dm_particle_mass is None:
                dm_particle_mass = float(handle["Header"].attrs["MassTable"][1]) * 1e10

    dmo_pos = np.concatenate(pos_chunks) / 1000.0
    dmo_mass_arr = np.full(len(dmo_pos), dm_particle_mass, dtype=np.float32)

    if spec.proj_frac < 1.0:
        mask = dmo_pos[:, 2] < spec.box_size * spec.proj_frac
        dmo_pos = dmo_pos[mask]
        dmo_mass_arr = dmo_mass_arr[mask]

    return pixelize_z_projection(dmo_pos, dmo_mass_arr, spec.box_size, spec.npix)


def load_dmo_particles(spec: SimulationSpec) -> tuple[np.ndarray, float]:
    """Load raw DMO particle positions (Mpc/h) and uniform particle mass (Msun/h).

    Unlike load_dmo_projection, this keeps the full 3D positions so that
    per-halo z-slabs can be selected for the cube-model condition patches.
    """
    snap_files = _dmo_snapshot_files(spec.nbody_path, spec.snapshot)
    pos_chunks: list[np.ndarray] = []
    dm_particle_mass: float | None = None
    for fname in snap_files:
        with h5py.File(fname, "r") as handle:
            pos_chunks.append(handle["PartType1/Coordinates"][:])
            if dm_particle_mass is None:
                dm_particle_mass = float(handle["Header"].attrs["MassTable"][1]) * 1e10
    positions = np.concatenate(pos_chunks) / 1000.0  # kpc/h → Mpc/h
    return positions, float(dm_particle_mass)


def _project_cube_patch(
    positions: np.ndarray,  # (N, 3) Mpc/h full-box particles
    particle_mass: float,   # Msun/h, uniform
    halo_xyz: np.ndarray,   # (3,) Mpc/h  [x, y, z]
    box_size: float,
    patch_pix: int,
    slab_depth: float,
) -> np.ndarray:
    """Project the DMO particles inside a cube of side `slab_depth` centred on
    `halo_xyz` onto a `patch_pix × patch_pix` 2D map.

    All three axes use periodic wrapping so halos near the box boundary are
    handled correctly.  The projection axis is z (matching the training data
    convention where each cube file is a z-projection of a 6.25 Mpc/h³ cube).
    """
    half = slab_depth / 2.0
    xh, yh, zh = float(halo_xyz[0]), float(halo_xyz[1]), float(halo_xyz[2])

    # Periodic displacements from halo centre
    dx = ((positions[:, 0] - xh + box_size / 2) % box_size) - box_size / 2
    dy = ((positions[:, 1] - yh + box_size / 2) % box_size) - box_size / 2
    dz = ((positions[:, 2] - zh + box_size / 2) % box_size) - box_size / 2

    mask = (np.abs(dx) < half) & (np.abs(dy) < half) & (np.abs(dz) < half)
    if not mask.any():
        return np.zeros((patch_pix, patch_pix), dtype=np.float32)

    # Translate to [0, slab_depth) so MASL treats the patch as its own box
    px = (dx[mask] + half).astype(np.float32)
    py = (dy[mask] + half).astype(np.float32)
    patch_pos = np.ascontiguousarray(np.stack([px, py], axis=1))
    masses = np.full(int(mask.sum()), particle_mass, dtype=np.float32)

    field = np.zeros((patch_pix, patch_pix), dtype=np.float32)
    MASL.MA(patch_pos, field, slab_depth, MAS="CIC", W=masses, verbose=False)
    return field


def extract_halo_cutouts_cube(
    positions: np.ndarray,       # (N, 3) DMO particle positions Mpc/h
    particle_mass: float,
    halos: list[dict],
    halo_positions: np.ndarray,  # (M, 3) full 3D halo positions Mpc/h
    box_size: float,
    patch_pix: int,
    slab_depth: float,
) -> list[dict]:
    """Extract per-halo DMO condition patches using a z-slab of depth `slab_depth`.

    This replicates the training-data geometry: each cube file was created by
    projecting a `slab_depth`-deep slice (6.25 Mpc/h for the default CV/SB35
    setup) along z, so inference must use the same projection depth rather than
    the full box depth used by the standard extract_halo_cutouts.

    Returns list of dicts with keys:
      condition   – (patch_pix, patch_pix) projected DM mass map
      large_scale – zeros (3, patch_pix, patch_pix), ignored by the cube model
    """
    dummy_ls = np.zeros((3, patch_pix, patch_pix), dtype=np.float32)
    cutouts = []
    for halo, hpos in tqdm(
        zip(halos, halo_positions), total=len(halos), desc="Extracting cube DMO cutouts"
    ):
        cond = _project_cube_patch(
            positions, particle_mass, hpos, box_size, patch_pix, slab_depth
        )
        cutouts.append({"condition": cond, "large_scale": dummy_ls})
    return cutouts


def voxelize_dmo_3d(
    positions: np.ndarray,
    particle_mass: float,
    box_size: float,
    npix: int,
) -> np.ndarray:
    """Voxelize DMO particles into a (npix, npix, npix) 3D CIC mass grid.

    Matches the training-data procedure for the cube model: the full periodic
    box is voxelized at the specified resolution so that sub-cube extraction
    via array indexing reproduces the exact DM maps used during training.
    """
    pos_ = np.ascontiguousarray(positions.astype(np.float32))
    masses = np.full(len(pos_), particle_mass, dtype=np.float32)
    field = np.zeros((npix, npix, npix), dtype=np.float32)
    MASL.MA(pos_, field, box_size, MAS="CIC", W=masses, verbose=False)
    return field


def _extract_cube_patch_project(
    field3d: np.ndarray,   # (npix3d, npix3d, npix3d)
    halo_xyz: np.ndarray,  # (3,) Mpc/h
    box_size: float,
    patch_pix: int,
) -> np.ndarray:
    """Extract a patch_pix^3 sub-cube from the 3D field and project along z.

    Converts the halo position to the nearest voxel index, extracts a
    patch_pix-wide cube with periodic boundary conditions, then sums along
    axis-2 (z) to produce a (patch_pix, patch_pix) 2D map.

    This exactly replicates the training-data generation:
        full-box 3D CIC voxelization → 128^3 sub-cube extraction → z-sum.
    """
    npix3d = field3d.shape[0]
    ppm = npix3d / box_size  # pixels per Mpc/h
    cx = int(round(float(halo_xyz[0]) * ppm)) % npix3d
    cy = int(round(float(halo_xyz[1]) * ppm)) % npix3d
    cz = int(round(float(halo_xyz[2]) * ppm)) % npix3d

    half = patch_pix // 2
    ix = (cx - half + np.arange(patch_pix)) % npix3d
    iy = (cy - half + np.arange(patch_pix)) % npix3d
    iz = (cz - half + np.arange(patch_pix)) % npix3d

    cube = field3d[np.ix_(ix, iy, iz)]  # (patch_pix, patch_pix, patch_pix)
    return cube.sum(axis=2).astype(np.float32)  # project along z


def extract_halo_cutouts_cube_from_3d(
    field3d: np.ndarray,
    halos: list[dict],
    halo_positions: np.ndarray,  # (M, 3) Mpc/h
    box_size: float,
    patch_pix: int,
) -> list[dict]:
    """Extract per-halo 2D z-projections from a pre-computed 3D CIC voxel grid.

    Replicates the training-data geometry for the cube model:
      1. Full-box CIC voxelization at 1024^3 (done externally via voxelize_dmo_3d)
      2. Extract 128^3 sub-cube centred on the halo voxel (periodic BC)
      3. Sum along z → 128×128 DM mass map

    Returns list of dicts with keys:
      condition   – (patch_pix, patch_pix) projected DM mass map
      large_scale – zeros (3, patch_pix, patch_pix), ignored by cube model
    """
    dummy_ls = np.zeros((3, patch_pix, patch_pix), dtype=np.float32)
    cutouts = []
    for halo, hpos in tqdm(
        zip(halos, halo_positions), total=len(halos), desc="Extracting cube cutouts (3D)"
    ):
        cond = _extract_cube_patch_project(field3d, hpos, box_size, patch_pix)
        cutouts.append({"condition": cond, "large_scale": dummy_ls})
    return cutouts


def load_halo_catalog(spec: SimulationSpec) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load FoF group catalog, apply halo mass cut, and build halo list.

    Uses Group_M_Crit200 (M200c) for the mass cut and stored masses,
    consistent with BIND's load_halo_catalog. GroupMass (total FoF mass)
    was previously used here, causing ~10/55 extra halos per sim that
    exceed the FoF threshold but fall below M200c.
    """
    # Try multi-chunk layout first (CV/1P hydro groups), then single-file (SB35 DM FoF)
    patt = spec.group_catalog / f"fof_subhalo_tab_{spec.snapshot:03d}.*.hdf5"
    files = sorted(glob.glob(str(patt)))
    if not files:
        single = spec.group_catalog / f"fof_subhalo_tab_{spec.snapshot:03d}.hdf5"
        if single.exists():
            files = [str(single)]
    if not files:
        raise FileNotFoundError(f"No FoF group files found in {spec.group_catalog}")

    all_masses: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []
    all_r200s: list[np.ndarray] = []
    for fname in files:
        with h5py.File(fname, "r") as handle:
            if "Group/Group_M_Crit200" not in handle:
                continue
            m200 = handle["Group/Group_M_Crit200"][:]
            all_masses.append(m200)
            all_positions.append(handle["Group/GroupPos"][:])
            if "Group/Group_R_Crit200" in handle:
                all_r200s.append(handle["Group/Group_R_Crit200"][:].astype(np.float32))
            else:
                all_r200s.append(np.zeros(len(m200), dtype=np.float32))

    if not all_masses:
        raise RuntimeError(f"Group catalog found but no Group_M_Crit200 datasets in {spec.group_catalog}")

    masses = np.concatenate(all_masses) * 1e10
    positions = np.concatenate(all_positions) / 1e3
    r200s = np.concatenate(all_r200s) / 1e3  # kpc/h -> Mpc/h

    mask = masses > spec.halo_mass_min
    halo_masses = masses[mask].astype(np.float32)
    halo_positions = positions[mask].astype(np.float32)
    halo_r200s = r200s[mask].astype(np.float32)

    halos = [
        {
            "halo_center": pos[:2],
            "halo_mass": float(mass),
            "r200": float(r200),
            "params": spec.params,
        }
        for pos, mass, r200 in zip(halo_positions, halo_masses, halo_r200s)
    ]
    return halos, halo_masses, halo_r200s, halo_positions


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
    """Normalize one halo cutout and associated parameter vector.

    Parameters with `ns.param_log_flag == 1` are log10-transformed before
    min/max scaling — `ns.param_min`/`ns.param_max` are already in log10
    space for those entries (see `data.NormStats`).
    """
    condition = log_transform(hc["condition"])[None]
    condition = (condition - ns.cond_mean) / (ns.cond_std + 1e-8)

    large_scale = log_transform(hc["large_scale"])
    large_scale = (large_scale - ns.ls_mean[:, None, None]) / (ns.ls_std[:, None, None] + 1e-8)

    p = sim_params.astype(np.float64)
    p = np.where(ns.param_log_flag == 1, np.log10(np.maximum(p, 1e-30)), p)
    rang = ns.param_max - ns.param_min
    params = ((p - ns.param_min) / (rang + 1e-8)).astype(np.float32)
    return condition, large_scale, params


def _denormalize_to_physical(
    gen_np: np.ndarray, norm_stats: NormStats
) -> np.ndarray:
    """Take raw model output (B, C, H, W) in normalized space and return
    physical-space (B, 3, H, W) [DM_hydro, Gas, Stars].

    Single-head (C=3): standard inverse standardize → 10^x - 1 per channel.
    Two-head    (C=4): channels 0/1 are DM_hydro/Gas as usual; channels 2/3
        are recombined via a soft multiplier into Stars:
            occ_raw     = clip(gen[2] * stars_occ_std + stars_occ_mean, 0, 1)
            density_log = gen[3] * stars_cond_std + stars_cond_mean
            stars       = occ_raw * (10 ** density_log - 1)
    """
    if norm_stats.stars_two_head:
        if gen_np.shape[1] != 4:
            raise ValueError(
                f"norm_stats.stars_two_head=True but model produced "
                f"{gen_np.shape[1]} channels (expected 4)"
            )
        out = np.zeros((gen_np.shape[0], 3) + gen_np.shape[2:], dtype=np.float32)
        # DM_hydro and Gas: same standard inverse as single-head
        for ch in range(2):
            x = gen_np[:, ch] * norm_stats.target_std[ch] + norm_stats.target_mean[ch]
            out[:, ch] = 10.0 ** x - 1.0
        # Stars: hard binary gate on occupancy × conditional density.
        # occ_prob is near-bimodal (≈0 or ≈1 with negligible ambiguous mass);
        # a soft multiply lets the density head leak through on "empty" pixels
        # (occ_prob~0.05 × large_density >> threshold), inflating occupancy by
        # ~55 pp. Thresholding at 0.5 reduces that error to <0.5 pp.
        occ_raw = gen_np[:, 2] * norm_stats.stars_occ_std + norm_stats.stars_occ_mean
        occ_gate = (occ_raw > 0.5).astype(np.float32)
        density_log = (
            gen_np[:, 3] * norm_stats.stars_cond_std + norm_stats.stars_cond_mean
        )
        density_phys = 10.0 ** density_log - 1.0
        out[:, 2] = occ_gate * density_phys
        return np.clip(out, 0, None).astype(np.float32)

    # Single-head path (legacy)
    if gen_np.shape[1] != 3:
        raise ValueError(
            f"norm_stats.stars_two_head=False but model produced "
            f"{gen_np.shape[1]} channels (expected 3)"
        )
    out = gen_np.astype(np.float32, copy=True)
    for ch in range(3):
        out[:, ch] = out[:, ch] * norm_stats.target_std[ch] + norm_stats.target_mean[ch]
        out[:, ch] = 10.0 ** out[:, ch] - 1.0
    return np.clip(out, 0, None)


def generate_halo_patches(
    halo_cutouts: list[dict],
    norm_stats: NormStats,
    sim_params: np.ndarray,
    fm,
    device: torch.device,
    n_steps: int,
    batch_size: int,
    use_amp: bool,
    param_indices: np.ndarray | None = None,
    no_large_scale: bool = False,
) -> np.ndarray:
    """Run model inference on all halo cutouts and denormalize to physical space.

    Always returns (N, 3, H, W) [DM_hydro, Gas, Stars] regardless of whether
    the model uses single-head or two-head Stars internally.

    param_indices: optional array of indices into the 35-param vector to pass
        to the model.  Use when the model was trained with --exclude_cosmo_params
        (or any other subset).  None means pass all 35 params.
    no_large_scale: when True (cube model), large-scale context is not fed to
        the model (large_scale=None).  The cutout dict may still contain a
        'large_scale' key; it is simply ignored.
    """
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(halo_cutouts), batch_size), desc="Generating hydro"):
            batch = halo_cutouts[start : start + batch_size]
            conds, lss, params = zip(*[normalize_cutout(hc, norm_stats, sim_params) for hc in batch])

            cond_t = torch.from_numpy(np.stack(conds).astype(np.float32)).to(device)
            ls_t = (
                None if no_large_scale
                else torch.from_numpy(np.stack(lss).astype(np.float32)).to(device)
            )
            params_np = np.stack(params).astype(np.float32)
            if param_indices is not None:
                params_np = params_np[:, param_indices]
            params_t = torch.from_numpy(params_np).to(device)

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                gen = fm.sample(cond_t, ls_t, params_t, n_steps=n_steps)

            gen_np = gen.float().cpu().numpy().astype(np.float32)
            outputs.append(_denormalize_to_physical(gen_np, norm_stats))

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


def circular_taper_weight(patch_pix: int, r_pix: float, taper_frac: float = 0.15) -> np.ndarray:
    """2D circular Hann-tapered weight centred at the patch centre.

    Weight is 1 inside (1-taper_frac)*r_pix, smoothly tapers to 0 at r_pix,
    and is 0 outside.  r_pix is clamped to patch_pix//2 so the weight never
    exceeds the patch boundary.  Falls back to square_taper_weight when
    r_pix <= 0.
    """
    half = patch_pix // 2
    r_max = min(float(r_pix), float(half))
    if r_max <= 0:
        return square_taper_weight(patch_pix, taper_frac)

    yy, xx = np.mgrid[:patch_pix, :patch_pix] - half
    r = np.sqrt(xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2)
    r_inner = r_max * (1.0 - taper_frac)

    w = np.zeros((patch_pix, patch_pix), dtype=np.float32)
    w[r <= r_inner] = 1.0
    taper_zone = (r > r_inner) & (r <= r_max)
    t_norm = (r[taper_zone] - r_inner) / max(r_max - r_inner, 1e-6)
    w[taper_zone] = (0.5 * (1.0 + np.cos(np.pi * t_norm))).astype(np.float32)
    return w


def paste_halos_2d(
    canvas_res: int,
    box_size: float,
    halos: list[dict],
    patches: np.ndarray,
    weight: np.ndarray,
    weights_list: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Paste halo patches onto full box with overlap-aware weighted blending.

    If ``weights_list`` is provided each halo uses its own (patch_pix, patch_pix)
    weight (e.g. a per-halo circular mask); otherwise all halos share ``weight``.
    """
    canvas = np.zeros((3, canvas_res, canvas_res), dtype=np.float32)
    w_accum = np.zeros((canvas_res, canvas_res), dtype=np.float32)

    pixels_per_mpc = canvas_res / box_size

    for hi, (halo, patch) in enumerate(zip(halos, patches)):
        w = weights_list[hi] if weights_list is not None else weight
        w_half = w.shape[0] // 2
        cx = int(halo["halo_center"][0] * pixels_per_mpc) % canvas_res
        cy = int(halo["halo_center"][1] * pixels_per_mpc) % canvas_res
        ix = (cx - w_half + np.arange(w.shape[0])) % canvas_res
        iy = (cy - w_half + np.arange(w.shape[0])) % canvas_res

        for ch in range(3):
            canvas[ch][np.ix_(ix, iy)] += patch[ch] * w
        w_accum[np.ix_(ix, iy)] += w

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
    r200_factor: float = 0.0,
) -> dict:
    """Construct BIND composite map using notebook-consistent blending logic.

    When ``r200_factor > 0`` each halo patch is blended with a circular
    Hann-tapered weight of radius ``r200_factor * R200c`` (pixels), confining
    the generated baryonic content to a physically motivated aperture.  The
    square taper is used when ``r200_factor == 0`` (default, legacy behaviour)
    or when R200c data is unavailable for a halo.
    """
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
    square_taper = square_taper_weight(patch_pix, taper_frac=taper_frac)

    if r200_factor > 0:
        pixels_per_mpc = npix / box_size
        weights_list = [
            circular_taper_weight(
                patch_pix,
                r_pix=halo.get("r200", 0.0) * pixels_per_mpc * r200_factor,
                taper_frac=taper_frac,
            )
            for halo in halos
        ]
        hydro_canvas, hydro_weights = paste_halos_2d(
            npix, box_size, halos, patches_np, square_taper, weights_list=weights_list
        )
    else:
        hydro_canvas, hydro_weights = paste_halos_2d(npix, box_size, halos, patches_np, square_taper)

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


def load_hydro_particles(
    spec: SimulationSpec,
) -> tuple[
    tuple[np.ndarray, np.ndarray],  # (dm_pos_kpch, dm_mass_1e10)
    tuple[np.ndarray, np.ndarray],  # (gas_pos_kpch, gas_mass_1e10)
    tuple[np.ndarray, np.ndarray],  # (star_pos_kpch, star_mass_1e10)
]:
    """Load raw particle data for all three hydro species from the snapshot.

    Returns three (positions_kpc_h, masses_1e10_Msun_h) tuples for:
      DM hydro (PartType1), Gas (PartType0), Stars (PartType4).
    Positions are in kpc/h; masses are in units of 1e10 Msun/h — caller
    is responsible for applying the 1/1000 and ×1e10 conversions.
    """
    pattern = spec.hydro_snapdir / f"snap_{spec.snapshot:03d}.*.hdf5"
    snap_files = sorted(glob.glob(str(pattern)))
    if not snap_files:
        single = spec.hydro_snapdir / f"snap_{spec.snapshot:03d}.hdf5"
        if single.exists():
            snap_files = [str(single)]
    if not snap_files:
        raise FileNotFoundError(f"No hydro snapshots found for {spec.hydro_snapdir}")

    dm_pos_chunks:   list[np.ndarray] = []
    dm_mass_chunks:  list[np.ndarray] = []
    gas_pos_chunks:  list[np.ndarray] = []
    gas_mass_chunks: list[np.ndarray] = []
    star_pos_chunks: list[np.ndarray] = []
    star_mass_chunks: list[np.ndarray] = []

    for fname in snap_files:
        with h5py.File(fname, "r") as handle:
            mt = handle["Header"].attrs["MassTable"]
            if "PartType1/Coordinates" in handle:
                pos = handle["PartType1/Coordinates"][:]
                n = len(pos)
                mass = (
                    handle["PartType1/Masses"][:] if "PartType1/Masses" in handle
                    else np.full(n, mt[1], dtype=np.float32)
                )
                dm_pos_chunks.append(pos)
                dm_mass_chunks.append(mass.astype(np.float32))
            if "PartType0/Coordinates" in handle:
                gas_pos_chunks.append(handle["PartType0/Coordinates"][:])
                gas_mass_chunks.append(handle["PartType0/Masses"][:].astype(np.float32))
            if "PartType4/Coordinates" in handle:
                star_pos_chunks.append(handle["PartType4/Coordinates"][:])
                star_mass_chunks.append(handle["PartType4/Masses"][:].astype(np.float32))

    def _cat(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float32)

    def _cat1d(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    return (
        (_cat(dm_pos_chunks),   _cat1d(dm_mass_chunks)),
        (_cat(gas_pos_chunks),  _cat1d(gas_mass_chunks)),
        (_cat(star_pos_chunks), _cat1d(star_mass_chunks)),
    )


def extract_truth_cutouts_cube_from_3d(
    spec: SimulationSpec,
    halos: list[dict],
    halo_positions: np.ndarray,  # (M, 3) Mpc/h
) -> np.ndarray:
    """Voxelize hydro species to 3D and extract per-halo truth patches.

    Replicates the training-data geometry for all three hydro species:
      1. Voxelize each species (DM, Gas, Stars) to a (npix, npix, npix) 3D CIC
         grid over the full periodic box, identical to how the cube training data
         target maps were built from the hydro snapshots.
      2. Extract a patch_pix^3 sub-cube centred on each halo voxel (periodic BC).
      3. Sum along z → patch_pix×patch_pix 2D map per species.

    Each species grid is built and freed sequentially to keep peak memory usage
    at ~one 1024^3 float32 grid (~4.3 GB) rather than all three simultaneously.

    Returns (N_halos, 3, patch_pix, patch_pix) float32 [DM_hydro, Gas, Stars].
    """
    npix      = spec.npix
    box_size  = spec.box_size
    patch_pix = spec.patch_pix
    n_halos   = len(halos)

    (dm_pos_kpch,   dm_mass_1e10), \
    (gas_pos_kpch,  gas_mass_1e10), \
    (star_pos_kpch, star_mass_1e10) = load_hydro_particles(spec)

    truth = np.zeros((n_halos, 3, patch_pix, patch_pix), dtype=np.float32)

    species = [
        (dm_pos_kpch,   dm_mass_1e10,   "DM_hydro"),
        (gas_pos_kpch,  gas_mass_1e10,  "Gas"),
        (star_pos_kpch, star_mass_1e10, "Stars"),
    ]
    for ch_idx, (pos_kpch, mass_1e10, label) in enumerate(species):
        if len(pos_kpch) == 0:
            continue
        pos_mpch     = np.ascontiguousarray((pos_kpch / 1000.0).astype(np.float32))
        masses_msunh = (mass_1e10 * 1e10).astype(np.float32)
        field3d = np.zeros((npix, npix, npix), dtype=np.float32)
        MASL.MA(pos_mpch, field3d, box_size, MAS="CIC", W=masses_msunh, verbose=False)
        for hi, hpos in enumerate(tqdm(
            halo_positions, desc=f"Extracting truth cutouts ({label})", leave=False
        )):
            truth[hi, ch_idx] = _extract_cube_patch_project(
                field3d, hpos, box_size, patch_pix
            )
        del field3d

    return truth


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
