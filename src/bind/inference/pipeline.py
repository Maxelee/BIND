"""Notebook-equivalent DMO->hydro pipeline primitives."""

from __future__ import annotations

import glob
import hashlib
from contextlib import nullcontext
from pathlib import Path

import h5py
import MAS_library as MASL
import numpy as np
import torch
from tqdm import tqdm

from bind.data import (
    N_OBS,
    N_THERMO,
    THERMO_KEYS,
    NormStats,
    compute_observables,
    log_transform,
    thermo_forward,
    thermo_inverse,
)

from .schemas import SimulationSpec

# ---------------------------------------------------------------------------
# Gas thermodynamics — per-particle physics + projection, ported verbatim from
# make_train_data/add_gas_thermo_maps.py so that test-suite truth thermo maps
# are generated with the exact same recipe as the training targets.
# ---------------------------------------------------------------------------

GAMMA       = 5.0 / 3.0
X_H         = 0.76
M_PROTON_KG = 1.6726219e-27        # kg
K_B_J_PER_K = 1.380649e-23         # J/K
SIGMA_T_M2  = 6.6524587e-29        # m^2
M_E_C2_J    = 8.187105776e-14      # J
KPC_IN_M    = 3.085677581e19       # m per kpc
MSUN_KG     = 1.989e30             # kg
KEV_IN_J    = 1.602176634e-16      # J per keV
MPC_IN_M    = KPC_IN_M * 1e3       # m per Mpc


def gas_temperature_K(internal_energy_code, electron_abundance):
    """Gas temperature [K] from code-unit InternalEnergy [(km/s)^2]."""
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * electron_abundance)
    u_SI = internal_energy_code * 1e6   # (km/s)^2 -> (m/s)^2
    return (GAMMA - 1.0) * u_SI * mu * M_PROTON_KG / K_B_J_PER_K


def compton_y_integrand_per_particle(internal_energy_code, electron_abundance,
                                     mass_code, mass_code_to_kg):
    """Per-particle Compton-y integrand [m^2]; sum over a pixel -> y * pixel_area.

    float64 throughout to avoid subnormal flush-to-zero; caller normalises by
    pixel_area before casting to float32 (see :func:`project_thermo_fullbox`).
    """
    T = gas_temperature_K(internal_energy_code, electron_abundance)
    mass_kg = mass_code.astype(np.float64) * mass_code_to_kg
    n_e_V = electron_abundance.astype(np.float64) * X_H * mass_kg / M_PROTON_KG
    return SIGMA_T_M2 * K_B_J_PER_K * T.astype(np.float64) * n_e_V / M_E_C2_J


def _safe_divide(numerator, denominator):
    """Mass-weighted mean numerator/denominator, zero where denom ~ 0."""
    dmax = denominator.max() if denominator.size else 0.0
    thresh = 1e-12 * max(float(dmax), 1e-30)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > thresh)


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


def load_halo_catalog(
    spec: SimulationSpec,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
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


def _downsample_square(cutout: np.ndarray, target_res: int) -> np.ndarray:
    """Downsample a square cutout to ``target_res × target_res``.

    Uses exact block-mean when the size is an integer multiple of ``target_res``
    (the training / CAMELS-suite case — kept bit-for-bit). Falls back to
    area-averaging interpolation when it is not, which happens for the full-box
    context scale of a box whose ``npix`` is not a multiple of ``target_res``
    (e.g. TNG300: ``npix = round(205 / 0.0488) = 4198``, ``target_res = 128``).
    ``mode="area"`` is exactly average pooling, so it matches block-mean for
    integer factors and stays mass-preserving for fractional ones.
    """
    spx = cutout.shape[0]
    if spx == target_res:
        return cutout.astype(np.float32)
    if spx % target_res == 0:
        factor = spx // target_res
        return (cutout.reshape(target_res, factor, target_res, factor)
                .mean(axis=(1, 3)).astype(np.float32))
    # Non-divisible size: area-average to downsample, bilinear to upsample
    # (matches the training generator's scipy-zoom order=1 upsample branch).
    t = torch.from_numpy(np.ascontiguousarray(cutout, dtype=np.float32))[None, None]
    if spx > target_res:
        out = torch.nn.functional.interpolate(t, size=(target_res, target_res), mode="area")
    else:
        out = torch.nn.functional.interpolate(
            t, size=(target_res, target_res), mode="bilinear", align_corners=False)
    return out[0, 0].numpy().astype(np.float32)


# Fixed physical scales (Mpc/h) of the condition + 3 large-scale context channels,
# matching the training data generator (data_generation/process_simulations2_cpu.py,
# extract_multiscale_cutouts: scales_mpc = [6.25, 12.5, 25.0, 50.0]). The model was
# trained with the largest context channel = a 50 Mpc/h window, so inference MUST use
# the same physical scales regardless of the full-box size — not [128,256,512,full_res]
# pixels, which makes the 4th channel span the whole box (e.g. 205 Mpc/h for TNG300)
# and feeds the network out-of-distribution context.
MULTISCALE_MPC = (6.25, 12.5, 25.0, 50.0)


def extract_multiscale(
    dmo_map: np.ndarray, cx_pix: int, cy_pix: int, target_res: int,
    mpc_per_pix: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the condition patch and three large-scale context patches.

    With ``mpc_per_pix`` (Mpc/h per pixel of ``dmo_map``) the four channels are
    taken at the fixed *physical* scales :data:`MULTISCALE_MPC` =
    (6.25, 12.5, 25, 50) Mpc/h — matching the training data — each capped at the
    box and resampled to ``target_res``. For a native 50 Mpc/h / 1024-pixel
    projection (training + CAMELS suites) this gives exactly [128,256,512,1024]
    px, identical to the legacy behaviour. When ``mpc_per_pix is None`` the
    legacy pixel-doubling scales ``[target_res, 2x, 4x, full_res]`` are used.
    """
    full_res = dmo_map.shape[0]
    if mpc_per_pix is None:
        scales_pix = [target_res, target_res * 2, target_res * 4, full_res]
    else:
        scales_pix = [min(int(round(s / mpc_per_pix)), full_res) for s in MULTISCALE_MPC]

    result = np.zeros((4, target_res, target_res), dtype=np.float32)
    for i, spx in enumerate(scales_pix):
        cutout = extract_periodic_cutout(dmo_map, cx_pix, cy_pix, spx)
        result[i] = _downsample_square(cutout, target_res)

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
    mpc_per_pix = box_size / npix
    halo_cutouts: list[dict] = []
    for halo in tqdm(halos, desc="Extracting DMO cutouts"):
        cx = int(halo["halo_center"][0] * pixels_per_mpc) % npix
        cy = int(halo["halo_center"][1] * pixels_per_mpc) % npix
        cond_cut, ls_cut = extract_multiscale(
            dmo_fullbox, cx, cy, target_res=patch_pix, mpc_per_pix=mpc_per_pix)
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


def _denormalize_thermo(gen_thermo: np.ndarray, norm_stats: NormStats) -> np.ndarray:
    """Inverse-transform the N_THERMO model channels to physical units.

    gen_thermo: (B, N_THERMO, H, W) in normalized space, THERMO_KEYS order.
    Returns (B, N_THERMO, H, W) >= 0 via :func:`thermo_inverse` (10**(t*std+mean)).
    """
    mean = norm_stats.thermo_mean[None, :, None, None]
    std = norm_stats.thermo_std[None, :, None, None]
    return np.clip(thermo_inverse(gen_thermo, mean, std), 0, None).astype(np.float32)


def _denormalize_to_physical(
    gen_np: np.ndarray, norm_stats: NormStats
) -> np.ndarray:
    """Take raw model output (B, C, H, W) in normalized space and return
    physical-space (B, 3 [+N_THERMO], H, W).

    The first 3 returned channels are always [DM_hydro, Gas, Stars]:
      Single-head: mass channels 0..2 → 10**x - 1 per channel.
      Two-head:    channels 0/1 are DM_hydro/Gas; channels 2/3 recombine into
                   Stars via a hard occupancy gate × conditional density.
    When ``norm_stats.predict_thermo`` the trailing N_THERMO channels
    (compton_y, temperature, entropy, pressure) are inverse-log10'd and
    appended, so the return is ``(B, 3 + N_THERMO, H, W)``. Thermo channels
    are intensive/extensive physical quantities — they are NOT mass and must
    not enter the composite.
    """
    base_out = 4 if norm_stats.stars_two_head else 3
    n_thermo = N_THERMO if norm_stats.predict_thermo else 0
    expected = base_out + n_thermo
    if gen_np.shape[1] != expected:
        raise ValueError(
            f"model produced {gen_np.shape[1]} channels but norm_stats implies "
            f"{expected} (stars_two_head={norm_stats.stars_two_head}, "
            f"predict_thermo={norm_stats.predict_thermo})"
        )

    mass = np.zeros((gen_np.shape[0], 3) + gen_np.shape[2:], dtype=np.float32)
    if norm_stats.stars_two_head:
        # DM_hydro and Gas: standard inverse standardize.
        for ch in range(2):
            x = gen_np[:, ch] * norm_stats.target_std[ch] + norm_stats.target_mean[ch]
            mass[:, ch] = 10.0 ** x - 1.0
        # Stars: hard binary gate on occupancy × conditional density.
        # occ_prob is near-bimodal (≈0 or ≈1); a soft multiply lets the density
        # head leak through on "empty" pixels, inflating occupancy by ~55 pp.
        # Thresholding at 0.5 reduces that error to <0.5 pp.
        occ_raw = gen_np[:, 2] * norm_stats.stars_occ_std + norm_stats.stars_occ_mean
        occ_gate = (occ_raw > 0.5).astype(np.float32)
        density_log = (
            gen_np[:, 3] * norm_stats.stars_cond_std + norm_stats.stars_cond_mean
        )
        mass[:, 2] = occ_gate * (10.0 ** density_log - 1.0)
    else:
        for ch in range(3):
            x = gen_np[:, ch] * norm_stats.target_std[ch] + norm_stats.target_mean[ch]
            mass[:, ch] = 10.0 ** x - 1.0
    mass = np.clip(mass, 0, None).astype(np.float32)

    if n_thermo == 0:
        return mass
    thermo = _denormalize_thermo(gen_np[:, base_out:base_out + n_thermo], norm_stats)
    return np.concatenate([mass, thermo], axis=1)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
# BIND is generative: every run draws fresh noise, so an unseeded run cannot be
# reproduced.  Seeding is opt-in and strictly local — `make_generator` returns a
# private torch.Generator and nothing here ever calls torch.manual_seed(), so
# (a) an unseeded run is bit-for-bit identical to the historical behaviour and
# (b) a seeded run does not perturb any other RNG consumer in the process.


def derive_seed(seed: int | None, key: str | int) -> int | None:
    """Derive a stable sub-seed from a run seed and a key (sim label, slab index).

    Sub-runs must not share one noise stream, or the very same realization would
    be drawn for every simulation / slab.  The derivation is SHA-256 over
    ``f"{seed}:{key}"`` truncated to 63 bits, so it is stable across processes,
    machines and Python versions (unlike the salted builtin ``hash()``).
    Returns ``None`` when ``seed`` is ``None`` (i.e. stays unseeded).
    """
    if seed is None:
        return None
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def make_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    """Return a local ``torch.Generator`` on ``device`` seeded with ``seed``.

    ``None`` in, ``None`` out — callers then omit the ``generator`` kwarg
    entirely and the sampler draws from the global RNG exactly as before.
    """
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    return generator


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
    cond_vectors: np.ndarray | None = None,
    scale_factor: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Run model inference on all halo cutouts and denormalize to physical space.

    Returns (N, 3, H, W) [DM_hydro, Gas, Stars] regardless of whether the model
    uses single-head or two-head Stars internally.  When the model also predicts
    thermo fields (``norm_stats.predict_thermo``), the return is
    ``(N, 3 + N_THERMO, H, W)`` with the trailing channels in THERMO_KEYS order.

    param_indices: optional array of indices into the 35-param vector to pass
        to the model.  Use when the model was trained with --exclude_cosmo_params
        (or any other subset).  None means pass all 35 params.
    no_large_scale: when True (cube model), large-scale context is not fed to
        the model (large_scale=None).  The cutout dict may still contain a
        'large_scale' key; it is simply ignored.
    cond_vectors: optional (N, n_cond) array of *already-normalized* per-halo
        conditioning vectors that replace the per-sim ``sim_params`` path. Used
        for observable-conditioned models (n_cond = N_OBS), where the vector is
        per-halo rather than per-sim — see :func:`build_observable_vectors`.
        ``param_indices`` is ignored when this is given.
    seed: optional integer seed for the sampler's initial noise.  ``None`` (the
        default) leaves the draw on the global RNG, i.e. bit-identical to the
        historical behaviour.  With a seed, one local generator is created for
        this call and consumed batch by batch, so a rerun reproduces the output
        only when ``halo_cutouts`` order and ``batch_size`` also match (both are
        recorded in the run provenance).  On GPU, bitwise reproducibility further
        assumes the same device and deterministic conv kernels.
    """
    outputs: list[np.ndarray] = []
    generator = make_generator(seed, device)
    gen_kw = {} if generator is None else {"generator": generator}
    if cond_vectors is not None and len(cond_vectors) != len(halo_cutouts):
        raise ValueError(
            f"cond_vectors has {len(cond_vectors)} rows but there are "
            f"{len(halo_cutouts)} halo cutouts"
        )

    with torch.no_grad():
        for start in tqdm(range(0, len(halo_cutouts), batch_size), desc="Generating hydro"):
            batch = halo_cutouts[start : start + batch_size]
            conds, lss, params = zip(*[normalize_cutout(hc, norm_stats, sim_params) for hc in batch])

            cond_t = torch.from_numpy(np.stack(conds).astype(np.float32)).to(device)
            ls_t = (
                None if no_large_scale
                else torch.from_numpy(np.stack(lss).astype(np.float32)).to(device)
            )
            if cond_vectors is not None:
                # Per-halo observable conditioning (already normalized).
                params_np = np.asarray(
                    cond_vectors[start : start + batch_size], dtype=np.float32
                )
            else:
                params_np = np.stack(params).astype(np.float32)
                if param_indices is not None:
                    params_np = params_np[:, param_indices]
            params_t = torch.from_numpy(params_np).to(device)

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            sf_kw = {}
            if scale_factor is not None:
                sf_kw["scale_factor"] = torch.full(
                    (cond_t.shape[0],), float(scale_factor),
                    dtype=torch.float32, device=device,
                )
            with amp_ctx:
                gen = fm.sample(cond_t, ls_t, params_t, n_steps=n_steps,
                                **sf_kw, **gen_kw)

            gen_np = gen.float().cpu().numpy().astype(np.float32)
            outputs.append(_denormalize_to_physical(gen_np, norm_stats))

    if not outputs:
        n_out = 3 + (N_THERMO if norm_stats.predict_thermo else 0)
        return np.zeros((0, n_out, 0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def extract_truth_mass_patches(
    truth_maps: np.ndarray,
    halos: list[dict],
    box_size: float,
    npix: int,
    patch_pix: int,
) -> np.ndarray:
    """Per-halo [DM_hydro, Gas, Stars] 6.25 Mpc/h truth patches from the full-box
    truth maps, registered to the same halo-center convention as the DMO cutouts
    and truth thermo patches. Returns (N_halos, 3, patch_pix, patch_pix)."""
    pixels_per_mpc = npix / box_size
    out = np.zeros((len(halos), 3, patch_pix, patch_pix), dtype=np.float32)
    for i, halo in enumerate(halos):
        cx = int(halo["halo_center"][0] * pixels_per_mpc) % npix
        cy = int(halo["halo_center"][1] * pixels_per_mpc) % npix
        for ch in range(3):
            out[i, ch] = extract_periodic_cutout(truth_maps[ch], cx, cy, patch_pix)
    return out


def build_observable_vectors(
    truth_mass_patches: np.ndarray,    # (N, 3, H, W)  [DM, Gas, Stars]
    truth_thermo_patches: np.ndarray,  # (N, N_THERMO, H, W)  THERMO_KEYS order
    halos: list[dict],
    norm_stats: NormStats,
) -> np.ndarray:
    """Normalized per-halo observable conditioning matrix (N, N_OBS).

    Recomputes the OBSERVABLE_KEYS aperture-integrated observables within R200
    from per-halo truth mass + thermo patches (reusing
    :func:`bind.data.compute_observables`, so the definition matches training
    exactly), then log/standardizes them with the run's obs_* stats. R200 is
    taken from each halo's catalog ``r200`` when > 0, else derived from M200c.

    NOTE: suite truth patches are axis-aligned (z-projection) whereas the
    training observables were measured on randomly-rotated cutouts; the
    aperture-integrated R200 quantities are fairly rotation-robust, but small
    differences are expected.
    """
    n = len(halos)
    if not (len(truth_mass_patches) == len(truth_thermo_patches) == n):
        raise ValueError("mass patches, thermo patches and halos must align in length")
    obs = np.zeros((n, N_OBS), dtype=np.float64)
    for i, halo in enumerate(halos):
        sample = {
            "target": truth_mass_patches[i],
            "halo_mass": float(halo["halo_mass"]),
        }
        for j, key in enumerate(THERMO_KEYS):
            sample[key] = truth_thermo_patches[i, j]
        r200 = float(halo.get("r200", 0.0) or 0.0)
        if r200 > 0:
            sample["r200"] = r200      # else compute_observables derives from M200c
        obs[i] = compute_observables(sample)
    obs_norm = thermo_forward(
        obs, norm_stats.obs_mean[None, :], norm_stats.obs_std[None, :],
        norm_stats.obs_floor[None, :],
    ).astype(np.float32)
    if norm_stats.mask_observables:
        # Masked model expects 2*N_OBS [obs*mask, mask]; full-obs eval = all-ones
        # mask. Pass a partial mask here to condition on a subset instead.
        mask = np.ones((n, N_OBS), dtype=np.float32)
        obs_norm = np.concatenate([obs_norm * mask, mask], axis=1)
    return obs_norm


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


def share_overlap_content(
    halos: list[dict],
    patches: np.ndarray,
    box_size: float,
    npix: int,
    patch_pix: int,
    r200_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy set-cover content sharing ('adoption') for overlapping pastes.

    Patches blended by weighted averaging must contain the *same* realization
    wherever they overlap: averaging N independent generative samples of the
    same region keeps their conditional mean but divides their stochastic
    small-scale variance by ~N, which suppresses the composite's high-k power
    wherever paste apertures overlap (a hydro-replaced control cannot detect
    this — overlapping truth patches are identical pixels, so the average is a
    no-op for truth content but lossy for generated content).

    Walking halos in descending mass, each not-yet-covered halo keeps its own
    patch and becomes a host; every other not-yet-covered halo whose paste
    aperture (``r200_factor * R200c`` pixels) fits inside the host's patch
    footprint adopts the host's realization, rolled to its own frame. The fit
    criterion guarantees an adopted halo's tapered paste disk never reads the
    rolled patch's wrapped edges. Halos without a positive R200c keep their own
    patch (their square-taper paste spans the full footprint).

    Returns ``(contents, host)``: per-halo content ``(N, C, patch_pix,
    patch_pix)`` float32 and the index of the halo whose realization each halo
    carries (``host[i] == i`` for hosts / non-adopted halos).
    """
    n = len(halos)
    pixels_per_mpc = npix / box_size
    half = patch_pix // 2
    masses = np.asarray([h["halo_mass"] for h in halos], dtype=np.float64)
    ap_pix = np.asarray(
        [min(h.get("r200", 0.0) * pixels_per_mpc * r200_factor, half - 2.0) for h in halos]
    )
    px = np.asarray([int(h["halo_center"][0] * pixels_per_mpc) % npix for h in halos])
    py = np.asarray([int(h["halo_center"][1] * pixels_per_mpc) % npix for h in halos])

    host = np.arange(n)
    covered = np.zeros(n, dtype=bool)
    for oi in np.argsort(-masses):
        if covered[oi]:
            continue
        covered[oi] = True
        dx = (px - px[oi] + npix // 2) % npix - npix // 2
        dy = (py - py[oi] + npix // 2) % npix - npix // 2
        fits = (
            (~covered)
            & (ap_pix > 0)
            & (np.abs(dx) + ap_pix < half - 1)
            & (np.abs(dy) + ap_pix < half - 1)
        )
        host[fits] = oi
        covered[fits] = True

    contents = np.empty_like(np.asarray(patches, dtype=np.float32))
    for j in range(n):
        h = int(host[j])
        if h == j:
            contents[j] = patches[j]
        else:
            dx = (px[j] - px[h] + npix // 2) % npix - npix // 2
            dy = (py[j] - py[h] + npix // 2) % npix - npix // 2
            contents[j] = np.roll(patches[h], shift=(-dx, -dy), axis=(1, 2))
    return contents, host


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
    r200_factor: float = 4.0,
    paste_mode: str = "shared",
) -> dict:
    """Construct BIND composite map using notebook-consistent blending logic.

    When ``r200_factor > 0`` (the standard, default 4.0) each halo patch is
    blended with a circular Hann-tapered weight of radius ``r200_factor * R200c``
    (pixels), confining the generated baryonic content to a physically motivated
    aperture.  This recovers the small-scale total-matter power that the legacy
    square taper (``r200_factor == 0``) smears away — see docs/circular_aperture.md.
    The square taper is also used as a per-halo fallback when R200c is unavailable.

    ``paste_mode`` controls how overlapping apertures are populated:

    - ``"shared"`` (default, standard): overlapping halos share one realization
      via :func:`share_overlap_content` before the weighted-average blend, and
      ``patch_mass_match`` rescales each paste *aperture-locally* (its weighted
      content mass matched to the weighted DMO mass in its footprint). Without
      sharing, averaging independent generative realizations in overlaps
      destroys their stochastic small-scale power (≈−10% total-matter P(k) at
      k≈40–70 h/Mpc for a ≥1e12 Msun/h halo population).
    - ``"average"`` (legacy): every halo pastes its own realization and
      ``patch_mass_match`` rescales whole patches against their DMO condition
      cutout. Kept for reproducing pre-fix composites.

    ``r200_factor <= 0`` (legacy square taper) always uses the legacy path.
    """
    if paste_mode not in ("shared", "average"):
        raise ValueError(f"Unknown paste_mode {paste_mode!r} (use 'shared' or 'average')")
    shared = paste_mode == "shared" and r200_factor > 0

    square_taper = square_taper_weight(patch_pix, taper_frac=taper_frac)
    patch_scales: list[float] = []
    host_idx = None

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

    if shared:
        patches_np, host_idx = share_overlap_content(
            halos, generated_patches, box_size, npix, patch_pix, r200_factor
        )
        if patch_mass_match:
            # Aperture-local match: adopted content is rolled, so whole-patch
            # totals no longer correspond to the halo's own condition cutout.
            half = patch_pix // 2
            ar = np.arange(patch_pix)
            for i, (halo, w) in enumerate(zip(halos, weights_list)):
                cx = int(halo["halo_center"][0] * pixels_per_mpc) % npix
                cy = int(halo["halo_center"][1] * pixels_per_mpc) % npix
                footprint = np.ix_((cx - half + ar) % npix, (cy - half + ar) % npix)
                m_dmo = float((dmo_fullbox[footprint] * w).sum())
                m_patch = float((patches_np[i].sum(0) * w).sum())
                s = m_dmo / (m_patch + 1e-30)
                patches_np[i] *= s
                patch_scales.append(s)
        hydro_canvas, hydro_weights = paste_halos_2d(
            npix, box_size, halos, patches_np, square_taper, weights_list=weights_list
        )
    else:
        patches = []
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

        if r200_factor > 0:
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
        "paste_mode": paste_mode,
        "host_idx": host_idx,
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


def _resolve_hydro_snap_files(spec: SimulationSpec) -> list[str]:
    """Resolve the hydro snapshot chunk(s) in ``spec.hydro_snapdir``.

    Accepts both the CAMELS ``snap_NNN`` and the IllustrisTNG/Arepo
    ``snapshot_NNN`` file-name conventions (multi-chunk ``*.N.hdf5`` first,
    then a single ``.hdf5``), so the same loader works across data layouts.
    """
    for prefix in ("snap", "snapshot"):
        files = sorted(glob.glob(str(spec.hydro_snapdir / f"{prefix}_{spec.snapshot:03d}.*.hdf5")))
        if files:
            return files
        single = spec.hydro_snapdir / f"{prefix}_{spec.snapshot:03d}.hdf5"
        if single.exists():
            return [str(single)]
    return []


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
    snap_files = _resolve_hydro_snap_files(spec)
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
    snap_files = _resolve_hydro_snap_files(spec)
    if not snap_files:
        raise FileNotFoundError(
            f"No hydro snapshots (snap_/snapshot_{spec.snapshot:03d}) in {spec.hydro_snapdir}"
        )

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


# ---------------------------------------------------------------------------
# Truth thermo maps (snapshot reprojection)
# ---------------------------------------------------------------------------

def load_gas_thermo_particles(spec: SimulationSpec) -> tuple[float, dict]:
    """Load gas (PartType0) fields for thermo maps plus HubbleParam.

    Returns (h, gas) where gas has keys: pos_mpc (N,3) Mpc/h, mass (N,) code
    units [1e10 Msun/h], density (N,) code units, u (N,) InternalEnergy
    [(km/s)^2], xe (N,) ElectronAbundance, sfr (N,) StarFormationRate.
    Star-forming gas is kept here; the caller applies the SFR>0 cut.
    """
    snap_files = _resolve_hydro_snap_files(spec)
    if not snap_files:
        raise FileNotFoundError(f"No hydro snapshots found for {spec.hydro_snapdir}")

    pos_l, mass_l, dens_l, u_l, xe_l, sfr_l = [], [], [], [], [], []
    h: float | None = None
    for fname in snap_files:
        with h5py.File(fname, "r") as f:
            if h is None:
                h = float(f["Header"].attrs["HubbleParam"])
            if "PartType0" not in f:
                continue
            g = f["PartType0"]
            pos_l.append(g["Coordinates"][:])
            mass_l.append(g["Masses"][:])
            dens_l.append(g["Density"][:])
            u_l.append(g["InternalEnergy"][:])
            xe_l.append(g["ElectronAbundance"][:])
            sfr_l.append(g["StarFormationRate"][:])

    if not pos_l:
        raise RuntimeError(f"No gas particles (PartType0) found in {spec.hydro_snapdir}")

    pos_mpc = np.concatenate(pos_l, axis=0).astype(np.float32) / 1000.0  # kpc/h -> Mpc/h
    gas = dict(
        pos_mpc=pos_mpc,
        mass=np.concatenate(mass_l).astype(np.float32),
        density=np.concatenate(dens_l).astype(np.float32),
        u=np.concatenate(u_l).astype(np.float32),
        xe=np.concatenate(xe_l).astype(np.float32),
        sfr=np.concatenate(sfr_l).astype(np.float32),
    )
    return float(h), gas


def project_thermo_fullbox(spec: SimulationSpec) -> np.ndarray:
    """Axis-aligned full-box projection of the 4 thermo fields (THERMO_KEYS order).

    Mirrors the per-particle physics and weighting of
    make_train_data/add_gas_thermo_maps.py (star-forming gas excluded;
    temperature/pressure/entropy are mass-weighted means; compton_y is the
    line-of-sight sum divided by pixel area).  Unlike the training pipeline
    this projects the whole box axis-aligned (no per-halo rotation), matching
    the test-suite cutout convention used for the mass truth maps so generated
    and truth patches share a frame.

    Returns (N_THERMO, npix, npix) float32 in THERMO_KEYS order.
    """
    h, gas = load_gas_thermo_particles(spec)
    npix, box = spec.npix, spec.box_size

    density_to_kg_m3 = 1e10 * MSUN_KG * h ** 2 / KPC_IN_M ** 3   # code density -> kg/m^3
    mass_code_to_kg = 1e10 * MSUN_KG / h                          # 1e10 Msun/h -> kg
    pixel_side_m = (box / npix) / h * MPC_IN_M                     # comoving pixel side [m]
    pixel_area_m2 = pixel_side_m ** 2

    hot = gas["sfr"] <= 0.0
    pos = gas["pos_mpc"][hot]
    mass = gas["mass"][hot]
    dens = gas["density"][hot]
    u = gas["u"][hot]
    xe = gas["xe"][hot]

    T = gas_temperature_K(u, xe)                                  # [K]
    rho_phys = dens.astype(np.float64) * density_to_kg_m3         # [kg/m^3]
    u_phys = u.astype(np.float64) * 1e6                           # [m/s]^2
    P = ((GAMMA - 1.0) * rho_phys * u_phys).astype(np.float32)    # [Pa]
    n_e = (xe.astype(np.float64) * X_H * rho_phys / M_PROTON_KG * 1e-6).astype(np.float32)  # [cm^-3]
    n_e_safe = np.where(n_e > 0, n_e, 1.0)
    K = ((K_B_J_PER_K * T / KEV_IN_J) / n_e_safe ** (2.0 / 3.0)).astype(np.float32)  # [keV cm^2]
    K[n_e <= 0] = 0.0
    y_int = compton_y_integrand_per_particle(u, xe, mass, mass_code_to_kg)

    def _proj(w: np.ndarray) -> np.ndarray:
        return pixelize_z_projection(pos, w, box, npix)

    m_map = _proj(mass.astype(np.float32))
    Tm = _proj((T * mass).astype(np.float32))
    Pm = _proj((P * mass).astype(np.float32))
    Km = _proj((K * mass).astype(np.float32))
    y_map = _proj((y_int / pixel_area_m2).astype(np.float32))

    temperature = _safe_divide(Tm, m_map)
    pressure = _safe_divide(Pm, m_map)
    entropy = _safe_divide(Km, m_map)
    compton_y = y_map  # already divided by pixel_area before CIC

    return np.stack([compton_y, temperature, entropy, pressure]).astype(np.float32)


def extract_halo_thermo_cutouts(
    thermo_fullbox: np.ndarray,
    halos: list[dict],
    box_size: float,
    npix: int,
    patch_pix: int,
) -> np.ndarray:
    """Per-halo periodic cutouts of the full-box thermo maps.

    Returns (N_halos, N_THERMO, patch_pix, patch_pix) float32, aligned to the
    same halo-center pixel convention as :func:`extract_halo_cutouts` so
    generated and truth patches are spatially registered.
    """
    pixels_per_mpc = npix / box_size
    out = np.zeros((len(halos), N_THERMO, patch_pix, patch_pix), dtype=np.float32)
    for i, halo in enumerate(halos):
        cx = int(halo["halo_center"][0] * pixels_per_mpc) % npix
        cy = int(halo["halo_center"][1] * pixels_per_mpc) % npix
        for ch in range(N_THERMO):
            out[i, ch] = extract_periodic_cutout(thermo_fullbox[ch], cx, cy, patch_pix)
    return out


def compute_truth_thermo_patches(spec: SimulationSpec, halos: list[dict]) -> np.ndarray:
    """Reconstruct per-halo truth thermo patches from the hydro snapshot.

    Returns (N_halos, N_THERMO, patch_pix, patch_pix) float32 in THERMO_KEYS
    order (empty when there are no halos).
    """
    if not halos:
        return np.zeros((0, N_THERMO, spec.patch_pix, spec.patch_pix), dtype=np.float32)
    thermo_fullbox = project_thermo_fullbox(spec)
    return extract_halo_thermo_cutouts(
        thermo_fullbox, halos, spec.box_size, spec.npix, spec.patch_pix
    )
