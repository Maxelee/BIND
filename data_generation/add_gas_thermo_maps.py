"""
add_gas_thermo_maps.py
======================
Append four new 128×128 gas-derived maps to existing npz training files.

New keys added to each file
---------------------------
compton_y   : (128,128) float32, dimensionless Compton-y parameter
temperature : (128,128) float32, mass-weighted gas temperature [K]
entropy     : (128,128) float32, mass-weighted gas entropy [keV cm²]
pressure    : (128,128) float32, mass-weighted thermal pressure [Pa]

All maps are spatially aligned with the existing target[1] (all-gas mass map).
Star-forming gas (StarFormationRate > 0) is excluded before projection.

Usage
-----
Single-rank smoke test (2 sims):
    python add_gas_thermo_maps.py --only_sims 0,1

MPI full run (via sbatch):
    srun -n 128 python add_gas_thermo_maps.py

Flags
-----
--output_base  : root containing train/ and test/ sub-directories (default: ceph path)
--hydro_base   : base path to IllustrisTNG snapshots
--limit_sims N : only process first N sims (useful for testing)
--only_sims    : comma-separated sim IDs to process (overrides limit_sims)
--total_sims   : total number of simulations (default 1024)
--test_frac    : fraction used for test split (must match original, default 0.1)
--seed         : RNG seed for train/test split (must match original, default 1993)
"""

import numpy as np
import h5py
import os
import argparse
import random
import sys
import traceback
from scipy.spatial.transform import Rotation
import MAS_library as MASL
import mpi4py.MPI as MPI

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description='Append thermo maps to existing npz files.')
parser.add_argument('--output_base', type=str,
                    default='/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu')
parser.add_argument('--hydro_base', type=str,
                    default='/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35')
parser.add_argument('--total_sims', type=int, default=1024)
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1993)
parser.add_argument('--limit_sims', type=int, default=None,
                    help='Only process first N sims (for testing).')
parser.add_argument('--only_sims', type=str, default=None,
                    help='Comma-separated sim IDs to process, e.g. "0,1,2".')
args = parser.parse_args()

# ============================================================
# MPI
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================
# Train/test split (must match process_simulations2_cpu.py)
# ============================================================
total_sims = args.total_sims
test_size  = int(args.test_frac * total_sims)
random.seed(args.seed)
_all_sims  = list(range(total_sims))
random.shuffle(_all_sims)
_test_sims = set(_all_sims[:test_size])

def split_for_sim(sim_id):
    return 'test' if sim_id in _test_sims else 'train'

# ============================================================
# Physical constants (verbatim from process_simulations_halo.py)
# ============================================================
GAMMA           = 5.0 / 3.0
X_H             = 0.76
M_PROTON_KG     = 1.6726219e-27        # kg
K_B_J_PER_K     = 1.380649e-23         # J/K
SIGMA_T_M2      = 6.6524587e-29        # m²
M_E_C2_J        = 8.187105776e-14      # J
KPC_IN_M        = 3.085677581e19       # m per kpc
MSUN_KG         = 1.989e30             # kg
KEV_IN_J        = 1.602176634e-16      # J per keV
MPC_IN_M        = KPC_IN_M * 1e3       # 3.085677581e22 m per Mpc

BOX_SIZE = 50.0    # Mpc/h
NPIX     = 1024    # full CIC grid
# Central 128×128 cutout: pixels [448:576, 448:576] of the 1024² projection
CENTER_PIX  = NPIX // 2        # 512
STRETCH     = 128  // 2        # 64
SLICE_LO    = CENTER_PIX - STRETCH   # 448
SLICE_HI    = CENTER_PIX + STRETCH   # 576

NEW_KEYS = ('compton_y', 'temperature', 'entropy', 'pressure')
EXPECTED_KEYS = {'condition', 'target', 'large_scale', 'params', 'halo_mass', 'halo_center'}

# ============================================================
# Per-particle physics (verbatim from process_simulations_halo.py)
# ============================================================

def gas_temperature_K(internal_energy_code, electron_abundance):
    """Gas temperature [K] from code-unit InternalEnergy [(km/s)²]."""
    mu   = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * electron_abundance)
    u_SI = internal_energy_code * 1e6   # (km/s)² → (m/s)²
    return (GAMMA - 1.0) * u_SI * mu * M_PROTON_KG / K_B_J_PER_K


def compton_y_integrand_per_particle(density_code, internal_energy_code,
                                     electron_abundance, mass_code,
                                     mass_code_to_kg):
    """
    Per-particle Compton-y integrand [m²].
    Sum over particles in a pixel → y × pixel_area_m².

    Uses float64 to avoid subnormal flush-to-zero in FTZ mode.
    """
    T       = gas_temperature_K(internal_energy_code, electron_abundance)
    mass_kg = mass_code.astype(np.float64) * mass_code_to_kg
    n_e_V   = electron_abundance.astype(np.float64) * X_H * mass_kg / M_PROTON_KG
    # Return float64: per-particle values can reach ~1e33-1e35 m², which exceeds the
    # float32 dynamic range when accumulated in a dense cluster pixel.  Caller is
    # responsible for normalising (e.g. dividing by pixel_area_m2) before float32.
    return SIGMA_T_M2 * K_B_J_PER_K * T.astype(np.float64) * n_e_V / M_E_C2_J


# ============================================================
# Snapshot I/O
# ============================================================

def read_hubble_param(hydro_path, sim_id):
    """Read HubbleParam h from the first available snapshot chunk."""
    snap_dir = os.path.join(hydro_path, f'SB35_{sim_id}', 'snapdir_090')
    for i in range(16):
        fname = os.path.join(snap_dir, f'snap_090.{i}.hdf5')
        if os.path.exists(fname):
            with h5py.File(fname, 'r') as f:
                return float(f['Header'].attrs['HubbleParam'])
    raise RuntimeError(f'No snapshot chunk found for sim {sim_id} in {snap_dir}')


def load_gas_fields(hydro_path, sim_id):
    """
    Load gas PartType0 fields from all 16 snapshot chunks.

    Returns dict with keys:
        pos_mpc   : (N,3) float32, positions in Mpc/h
        mass      : (N,)  float32, masses in 1e10 Msun/h (code units)
        density   : (N,)  float32, density in code units
        u         : (N,)  float32, InternalEnergy in (km/s)²
        xe        : (N,)  float32, ElectronAbundance
        sfr       : (N,)  float32, StarFormationRate [Msun/yr]
    """
    snap_dir = os.path.join(hydro_path, f'SB35_{sim_id}', 'snapdir_090')
    pos_l, mass_l, dens_l, u_l, xe_l, sfr_l = [], [], [], [], [], []
    for i in range(16):
        fname = os.path.join(snap_dir, f'snap_090.{i}.hdf5')
        if not os.path.exists(fname):
            continue
        with h5py.File(fname, 'r') as f:
            pt = 'PartType0'
            if pt not in f:
                continue
            pos_l.append(f[f'{pt}/Coordinates'][:])
            mass_l.append(f[f'{pt}/Masses'][:])
            dens_l.append(f[f'{pt}/Density'][:])
            u_l.append(f[f'{pt}/InternalEnergy'][:])
            xe_l.append(f[f'{pt}/ElectronAbundance'][:])
            sfr_l.append(f[f'{pt}/StarFormationRate'][:])

    if not pos_l:
        raise RuntimeError(f'No gas particles found for sim {sim_id}')

    pos   = np.concatenate(pos_l,  axis=0).astype(np.float32)
    mass  = np.concatenate(mass_l, axis=0).astype(np.float32)
    dens  = np.concatenate(dens_l, axis=0).astype(np.float32)
    u     = np.concatenate(u_l,    axis=0).astype(np.float32)
    xe    = np.concatenate(xe_l,   axis=0).astype(np.float32)
    sfr   = np.concatenate(sfr_l,  axis=0).astype(np.float32)

    pos /= 1000.0   # kpc/h → Mpc/h; mass stays as 1e10 Msun/h (code units)

    return dict(pos_mpc=pos, mass=mass, density=dens, u=u, xe=xe, sfr=sfr)


# ============================================================
# Projection pipeline (mirrors process_simulations2_cpu.py exactly)
# ============================================================

def apply_periodic_boundary_minimum_image(positions, halo_center, box_size=50.0):
    delta = positions - halo_center
    delta = delta - box_size * np.round(delta / box_size)
    return delta


def create_periodic_copies_with_index(positions, idx, box_size=50.0, buffer=10.0):
    """
    Tile particles near edges (as in process_simulations2_cpu.py) and carry
    a particle-index array alongside positions so any per-particle weight
    can be recovered as weight[idx_out].

    positions : (N,3) centered at origin
    idx       : (N,)  integer indices into the in_cube subset
    Returns tiled_pos (M,3), tiled_idx (M,)
    """
    all_positions = [positions]
    all_idx       = [idx]

    half_box       = box_size / 2.0
    edge_threshold = half_box - buffer

    for axis in range(3):
        near_pos_edge = positions[:, axis] > edge_threshold
        if np.any(near_pos_edge):
            cp = positions[near_pos_edge].copy()
            cp[:, axis] -= box_size
            all_positions.append(cp)
            all_idx.append(idx[near_pos_edge])

        near_neg_edge = positions[:, axis] < -edge_threshold
        if np.any(near_neg_edge):
            cp = positions[near_neg_edge].copy()
            cp[:, axis] += box_size
            all_positions.append(cp)
            all_idx.append(idx[near_neg_edge])

    return np.vstack(all_positions), np.concatenate(all_idx)


def pixelize_z_projection(positions, weights, box_size=50.0, npix=1024):
    """CIC deposit along z → 2-D map."""
    pos_  = np.ascontiguousarray(positions.astype(np.float32))[:, [0, 1]]
    w_    = np.ascontiguousarray(weights.astype(np.float32))
    field = np.zeros((npix, npix), dtype=np.float32)
    MASL.MA(pos_, field, box_size, MAS='CIC', W=w_, verbose=False)
    return field


def project_thermo_maps(gas_pos_mpc, w_m, w_Tm, w_Pm, w_Km, w_y, halo_center, seed):
    """
    Replicate the exact rotation/projection pipeline from process_simulations2_cpu.py
    and return the five central 128² maps.

    Parameters
    ----------
    gas_pos_mpc : (N,3) all non-SFR gas positions [Mpc/h]
    w_m, w_Tm, w_Pm, w_Km, w_y : per-particle weight arrays for mass, T*m, P*m, K*m, y-integrand

    Returns
    -------
    m_c, Tm_c, Pm_c, Km_c, y_c : each (128,128) float32
    """
    # --- 1. Center on halo (minimum image) ---
    centered = apply_periodic_boundary_minimum_image(gas_pos_mpc, halo_center, BOX_SIZE)

    # --- 2. Particles within margin=BOX_SIZE cube ---
    in_cube = np.all(np.abs(centered) < BOX_SIZE, axis=1)
    pos_ic  = centered[in_cube]
    n_ic    = in_cube.sum()
    idx_ic  = np.arange(n_ic, dtype=np.int32)

    # --- 3. Tile near-edge particles (carry index) ---
    tiled_pos, tiled_idx = create_periodic_copies_with_index(
        pos_ic, idx_ic, box_size=BOX_SIZE, buffer=10.0
    )

    # --- 4. Rotation: SINGLE draw from seeded RNG (must match original exactly) ---
    np.random.seed(seed)
    rot = Rotation.from_euler('xyz', np.random.uniform(0, 2 * np.pi, 3))
    rotated = tiled_pos @ rot.as_matrix().T

    # --- 5. Shift + clip ---
    shifted = rotated + BOX_SIZE / 2.0
    in_box  = np.all((shifted >= 0) & (shifted < BOX_SIZE), axis=1)
    final_pos = shifted[in_box]
    final_idx = tiled_idx[in_box]   # maps into in_cube subset

    # --- 6. CIC deposit for each quantity ---
    # Weights at in_cube level, indexed by final_idx
    def _proj(w):
        return pixelize_z_projection(final_pos, w[in_cube][final_idx], BOX_SIZE, NPIX)

    full_m  = _proj(w_m)
    full_Tm = _proj(w_Tm)
    full_Pm = _proj(w_Pm)
    full_Km = _proj(w_Km)
    full_y  = _proj(w_y)

    # --- 7. Central 128² cutout ---
    cut = np.s_[SLICE_LO:SLICE_HI, SLICE_LO:SLICE_HI]
    return full_m[cut], full_Tm[cut], full_Pm[cut], full_Km[cut], full_y[cut]


def safe_divide(numerator, denominator):
    """Mass-weighted mean: numerator/denominator, zero where denom ≈ 0."""
    dmax = denominator.max()
    thresh = 1e-12 * max(float(dmax), 1e-30)
    return np.divide(numerator, denominator,
                     out=np.zeros_like(numerator),
                     where=denominator > thresh)


# ============================================================
# Crash-safe atomic write
# ============================================================

def atomic_append(path, **new_arrays):
    """
    Load existing npz, merge new arrays, write to tmp then os.replace.
    Idempotent: no-op if all NEW_KEYS already present.
    """
    with np.load(path, allow_pickle=False) as d:
        merged = {k: d[k] for k in d.files}

    assert EXPECTED_KEYS.issubset(merged), (
        f'Unexpected/corrupt original: {path} has keys {set(merged)}'
    )
    for k in NEW_KEYS:
        v = new_arrays[k]
        assert v.shape == (128, 128), f'{k} shape {v.shape} != (128,128)'
        assert np.isfinite(v).all(),  f'{k} contains non-finite values in {path}'

    merged.update({k: new_arrays[k].astype(np.float32) for k in NEW_KEYS})

    # np.savez_compressed appends '.npz' when the path doesn't already end in '.npz'.
    # Make the tmp name end in '.npz' to avoid that and keep os.replace consistent.
    assert path.endswith('.npz')
    tmp = path[:-4] + f'.tmp.{os.getpid()}.npz'
    try:
        np.savez_compressed(tmp, **merged)
        os.replace(tmp, path)   # atomic on POSIX / Ceph
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def file_needs_update(path):
    """True if at least one of the four new keys is missing from the file."""
    try:
        with np.load(path, allow_pickle=False) as d:
            return not all(k in d.files for k in NEW_KEYS)
    except Exception:
        return True   # treat unreadable file as needing update


def sim_needs_update(sim_dir_path):
    """True if ANY npz in the sim directory is missing new keys."""
    npz_files = sorted(
        f for f in os.listdir(sim_dir_path) if f.endswith('.npz') and f.startswith('sim_')
    )
    if not npz_files:
        return False
    return any(file_needs_update(os.path.join(sim_dir_path, f)) for f in npz_files)


# ============================================================
# Per-simulation processing
# ============================================================

def process_sim(sim_id, sim_dir_path, hydro_path):
    """Load gas once, then append maps to every npz in sim_dir_path."""

    # ----- idempotent sim-level skip -----
    if not sim_needs_update(sim_dir_path):
        print(f'[rank {rank}] Sim {sim_id}: already complete, skipping.', flush=True)
        return

    print(f'[rank {rank}] Sim {sim_id}: loading gas fields...', flush=True)

    # ----- read h and set per-sim unit factors -----
    try:
        h = read_hubble_param(hydro_path, sim_id)
    except RuntimeError as exc:
        print(f'[rank {rank}] Sim {sim_id}: SKIP (no snapshot): {exc}', flush=True)
        return

    density_to_kg_m3 = 1e10 * MSUN_KG * h**2 / KPC_IN_M**3   # code density → kg/m³
    mass_code_to_kg  = 1e10 * MSUN_KG / h                      # 1e10 Msun/h → kg
    pixel_side_m     = (BOX_SIZE / NPIX) / h * MPC_IN_M        # comoving pixel side [m] (z=0)
    pixel_area_m2    = pixel_side_m ** 2

    # ----- load gas fields -----
    try:
        gas = load_gas_fields(hydro_path, sim_id)
    except Exception as exc:
        print(f'[rank {rank}] Sim {sim_id}: SKIP (gas load error): {exc}', flush=True)
        return

    # ----- exclude star-forming gas -----
    hot_mask = gas['sfr'] <= 0.0
    g_pos  = gas['pos_mpc'][hot_mask]
    g_mass = gas['mass'][hot_mask]
    g_dens = gas['density'][hot_mask]
    g_u    = gas['u'][hot_mask]
    g_xe   = gas['xe'][hot_mask]

    # ----- precompute per-particle quantities -----
    T      = gas_temperature_K(g_u, g_xe)                              # [K]
    rho_phys = g_dens.astype(np.float64) * density_to_kg_m3            # [kg/m³]
    u_phys   = g_u.astype(np.float64) * 1e6                            # [m/s]²
    P      = ((GAMMA - 1.0) * rho_phys * u_phys).astype(np.float32)    # [Pa]
    n_e    = (g_xe.astype(np.float64) * X_H * rho_phys / M_PROTON_KG * 1e-6).astype(np.float32)  # [cm⁻³]
    # entropy: K = k_B T [keV] / n_e^(2/3)  [keV cm²]
    n_e_safe = np.where(n_e > 0, n_e, 1.0)
    K      = ((K_B_J_PER_K * T / KEV_IN_J) / n_e_safe ** (2.0 / 3.0)).astype(np.float32)
    K[n_e <= 0] = 0.0

    y_int  = compton_y_integrand_per_particle(g_dens, g_u, g_xe, g_mass, mass_code_to_kg)

    w_m   = g_mass.astype(np.float32)
    w_Tm  = (T * g_mass).astype(np.float32)
    w_Pm  = (P * g_mass).astype(np.float32)
    w_Km  = (K * g_mass).astype(np.float32)
    # Normalise by pixel area NOW (in float64) so the CIC weights are ~1e-10,
    # safely representable as float32 and immune to per-pixel accumulation overflow.
    # After CIC deposit, y_c already equals Compton-y directly (no further division).
    w_y   = (y_int / pixel_area_m2).astype(np.float32)

    # free large temporaries
    del gas, T, rho_phys, u_phys, P, n_e, n_e_safe, K, y_int, g_dens, g_u, g_xe

    # ----- iterate over npz files -----
    npz_files = sorted(
        f for f in os.listdir(sim_dir_path) if f.endswith('.npz') and f.startswith('sim_')
    )
    n_files = len(npz_files)
    n_done  = 0

    for fname in npz_files:
        fpath = os.path.join(sim_dir_path, fname)

        if not file_needs_update(fpath):
            n_done += 1
            continue

        # parse halo_idx and rot_idx from filename sim_{sim_id}_halo_{hi}_rot_{ri}.npz
        try:
            parts    = fname.replace('.npz', '').split('_')
            halo_idx = int(parts[3])
            rot_idx  = int(parts[5])
        except (IndexError, ValueError) as exc:
            print(f'[rank {rank}] Sim {sim_id}: cannot parse filename {fname}: {exc}', flush=True)
            continue

        seed = sim_id * 1000 + halo_idx * 100 + rot_idx

        # read stored halo_center (do NOT reload FOF catalog)
        try:
            with np.load(fpath, allow_pickle=False) as d:
                halo_center = d['halo_center'].astype(np.float32)
        except Exception as exc:
            print(f'[rank {rank}] Sim {sim_id}: cannot read {fname}: {exc}', flush=True)
            continue

        try:
            m_c, Tm_c, Pm_c, Km_c, y_c = project_thermo_maps(
                g_pos, w_m, w_Tm, w_Pm, w_Km, w_y, halo_center, seed
            )
        except Exception as exc:
            print(f'[rank {rank}] Sim {sim_id}: projection error for {fname}: {exc}', flush=True)
            traceback.print_exc()
            continue

        temperature = safe_divide(Tm_c, m_c)
        pressure    = safe_divide(Pm_c, m_c)
        entropy     = safe_divide(Km_c, m_c)
        compton_y   = y_c  # w_y was already divided by pixel_area_m2 before CIC

        try:
            atomic_append(fpath,
                          compton_y=compton_y,
                          temperature=temperature,
                          entropy=entropy,
                          pressure=pressure)
        except Exception as exc:
            print(f'[rank {rank}] Sim {sim_id}: write error for {fname}: {exc}', flush=True)
            traceback.print_exc()
            continue

        n_done += 1

    print(f'[rank {rank}] Sim {sim_id}: done ({n_done}/{n_files} files updated).', flush=True)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    hydro_path = args.hydro_base

    # Build list of (sim_id, split) on rank 0, broadcast
    if rank == 0:
        if args.only_sims is not None:
            sim_ids = [int(x) for x in args.only_sims.split(',')]
        else:
            sim_ids = list(range(total_sims))
            if args.limit_sims is not None:
                sim_ids = sim_ids[:args.limit_sims]

        # Filter to sims that actually have output directories
        sim_entries = []
        for sid in sim_ids:
            split = split_for_sim(sid)
            sim_dir_path = os.path.join(args.output_base, split, f'sim_{sid}')
            if os.path.isdir(sim_dir_path):
                sim_entries.append((sid, sim_dir_path))
            else:
                print(f'[rank 0] Sim {sid}: output dir not found, skipping ({sim_dir_path})',
                      flush=True)

        print(f'[rank 0] Found {len(sim_entries)} sim directories to process '
              f'across {size} ranks.', flush=True)
    else:
        sim_entries = None

    sim_entries = comm.bcast(sim_entries, root=0)
    comm.Barrier()

    # Distribute: rank i gets sim_entries[i::size]
    my_entries = sim_entries[rank::size]
    print(f'[rank {rank}] Assigned {len(my_entries)} sims.', flush=True)

    for sim_id, sim_dir_path in my_entries:
        try:
            process_sim(sim_id, sim_dir_path, hydro_path)
        except Exception as exc:
            print(f'[rank {rank}] Sim {sim_id}: UNHANDLED ERROR: {exc}', flush=True)
            traceback.print_exc()

    comm.Barrier()
    if rank == 0:
        print('All ranks finished.', flush=True)
