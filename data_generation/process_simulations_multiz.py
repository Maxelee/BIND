"""
process_simulations_multiz.py
=============================
Multi-redshift training-data generator: a single MPI pass that, for every
(simulation, snapshot, halo), projects BOTH the mass maps and the gas-thermo
maps in one shot, with **one rotation per halo**.

This merges process_simulations2_cpu.py (mass maps) and add_gas_thermo_maps.py
(gas-thermo maps) into one script, and loops over a set of snapshots
(redshifts) instead of just z=0 (snap 090).

Output layout (note the extra snap_<NNN> level vs the single-z dataset):

    <output_base>/train_data_multiz_128_cpu/
        train/sim_<i>/snap_<NNN>/sim_<i>_halo_<k>_rot_0.npz
        test/ ...

Each .npz holds the usual mass keys (condition, target [DM_hydro,Gas,Stars],
large_scale, params, halo_mass, halo_center) PLUS:
    redshift      : scalar z of this snapshot
    scale_factor  : scalar a = 1/(1+z)
    compton_y, temperature, entropy, pressure : (128,128) gas-thermo maps

The train/test split is by simulation (same seed as the single-z pipeline), so a
sim is train-or-test across ALL its snapshots — no redshift leakage.

⚠ REDSHIFT-DEPENDENT THERMO PHYSICS — VALIDATE BEFORE TRUSTING z>0 MAPS.
The original thermo code was z=0-only. Here the comoving→physical conversions
carry the scale factor a (read from the snapshot header): physical density
∝ a^-3 (→ pressure, entropy), physical pixel area ∝ a^2 (→ Compton-y), while
temperature is a-independent. These are the textbook factors but have not been
validated against an independent z>0 reference — check a couple of snapshots.

Environment: Flatiron rusty MPI (module load openmpi python python-mpi hdf5;
gen_train_data venv with mpi4py + Pylians MAS_library). CAMELS paths are
argparse defaults — override for your own data copy.
"""

import argparse
import os
import random
import traceback

import h5py
import numpy as np
import pandas as pd
import MAS_library as MASL
from scipy.spatial.transform import Rotation
import mpi4py.MPI as MPI


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description='Multi-redshift mass + thermo map generator (MPI).')
parser.add_argument('--resolution', type=int, default=128)
parser.add_argument('--total_sims', type=int, default=1024)
parser.add_argument('--start_sim', type=int, default=0)
parser.add_argument('--end_sim', type=int, default=None)
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1993,
                    help='Train/test split seed — MUST match the single-z pipeline.')
parser.add_argument('--num_rotations', type=int, default=1,
                    help='Rotations per halo (1 for the multi-redshift dataset).')
parser.add_argument('--snapshots', type=str, default='90,82,74,60,52,44,32,24',
                    help='Comma-separated snapshot numbers to process.')
parser.add_argument('--mass_threshold', type=float, default=1e13)
parser.add_argument('--output_base_root', type=str, default='/mnt/home/mlee1/ceph')
parser.add_argument('--output_name', type=str, default='train_data_multiz_128_cpu')
parser.add_argument('--hydro_base', type=str, default='/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35')
parser.add_argument('--nbody_base', type=str, default='/mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35')
parser.add_argument('--fof_nbody_base', type=str, default='/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/SB35')
parser.add_argument('--param_file', type=str, default='/mnt/home/mlee1/50Mpc_boxes/data/param_df.csv')
args = parser.parse_args()

SNAPSHOTS = [int(s) for s in args.snapshots.split(',')]

# ============================================================
# MPI + train/test split
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

metadata = pd.read_csv(args.param_file)

resolution = args.resolution
total_sims = args.total_sims
test_size = int(args.test_frac * total_sims)
num_rotations = args.num_rotations
BOX_SIZE = 50.0  # Mpc/h
NPIX = int(resolution * BOX_SIZE / 6.25)  # 1024 full-box CIC grid

start_sim = args.start_sim
end_sim = args.end_sim if args.end_sim is not None else total_sims

random.seed(args.seed)
_all_sims = list(range(total_sims))
random.shuffle(_all_sims)
test_sims = set(_all_sims[:test_size])

# Central 128² cutout of the NPIX² projection.
CENTER_PIX = NPIX // 2
STRETCH = resolution // 2
SLICE = np.s_[CENTER_PIX - STRETCH:CENTER_PIX + STRETCH,
              CENTER_PIX - STRETCH:CENTER_PIX + STRETCH]

# ============================================================
# Physical constants (from process_simulations_halo.py)
# ============================================================
GAMMA = 5.0 / 3.0
X_H = 0.76
M_PROTON_KG = 1.6726219e-27
K_B_J_PER_K = 1.380649e-23
SIGMA_T_M2 = 6.6524587e-29
M_E_C2_J = 8.187105776e-14
KPC_IN_M = 3.085677581e19
MSUN_KG = 1.989e30
KEV_IN_J = 1.602176634e-16
MPC_IN_M = KPC_IN_M * 1e3

THERMO_KEYS = ('compton_y', 'temperature', 'entropy', 'pressure')


# ============================================================
# Snapshot-number-aware paths
# ============================================================

def nbody_snap_file(sim_id, snap):
    return os.path.join(args.nbody_base, f'SB35_{sim_id}', f'snap_{snap:03d}.hdf5')


def hydro_snapdir(sim_id, snap):
    return os.path.join(args.hydro_base, f'SB35_{sim_id}', f'snapdir_{snap:03d}')


def fof_file(sim_id, snap):
    return os.path.join(args.fof_nbody_base, f'SB35_{sim_id}', f'fof_subhalo_tab_{snap:03d}.hdf5')


def read_header(sim_id, snap):
    """Return (redshift, scale_factor a, HubbleParam h) from the snapshot header."""
    snapdir = hydro_snapdir(sim_id, snap)
    for i in range(16):
        fname = os.path.join(snapdir, f'snap_{snap:03d}.{i}.hdf5')
        if os.path.exists(fname):
            with h5py.File(fname, 'r') as f:
                hdr = f['Header'].attrs
                z = float(hdr['Redshift'])
                return z, 1.0 / (1.0 + z), float(hdr['HubbleParam'])
    raise RuntimeError(f'No snapshot chunk for sim {sim_id} snap {snap} in {snapdir}')


# ============================================================
# Loaders (snapshot-parameterized versions of the single-z loaders)
# ============================================================

def load_simulation(sim_id, snap):
    """DM (nbody) + hydro DM/gas/star positions & masses, in Mpc/h and Msun."""
    dm_pos, dm_mass = [], []
    with h5py.File(nbody_snap_file(sim_id, snap), 'r') as f:
        dm_pos.append(f['PartType1/Coordinates'][:])
        dm_particle_mass = f['Header'].attrs['MassTable'][1]
        dm_mass.append(np.full(len(f['PartType1/Coordinates'][:]), dm_particle_mass))
    dm_pos = np.concatenate(dm_pos) / 1000.0
    dm_mass = np.concatenate(dm_mass) * 1e10

    hydro_dm_pos, hydro_dm_mass = [], []
    star_pos, star_mass = [], []
    snapdir = hydro_snapdir(sim_id, snap)
    for i in range(16):
        fname = os.path.join(snapdir, f'snap_{snap:03d}.{i}.hdf5')
        if not os.path.exists(fname):
            continue
        with h5py.File(fname, 'r') as f:
            if 'PartType1/Coordinates' in f:
                hydro_dm_pos.append(f['PartType1/Coordinates'][:])
                if 'PartType1/Masses' in f:
                    hydro_dm_mass.append(f['PartType1/Masses'][:])
                else:
                    m = f['Header'].attrs['MassTable'][1]
                    hydro_dm_mass.append(np.full(len(f['PartType1/Coordinates'][:]), m))
            if 'PartType4/Coordinates' in f:
                star_pos.append(f['PartType4/Coordinates'][:])
                star_mass.append(f['PartType4/Masses'][:])

    def _cat(lst):
        return np.concatenate(lst) if lst else np.array([])

    hydro_dm_pos, hydro_dm_mass = _cat(hydro_dm_pos), _cat(hydro_dm_mass)
    star_pos, star_mass = _cat(star_pos), _cat(star_mass)
    if len(hydro_dm_pos) > 0:
        hydro_dm_pos /= 1000.0
        hydro_dm_mass *= 1e10
    if len(star_pos) > 0:
        star_pos /= 1000.0
        star_mass *= 1e10
    return dm_pos, dm_mass, hydro_dm_pos, hydro_dm_mass, star_pos, star_mass


def load_gas_fields(sim_id, snap):
    """Gas PartType0 fields for both the Gas mass target and the thermo maps."""
    snapdir = hydro_snapdir(sim_id, snap)
    pos_l, mass_l, dens_l, u_l, xe_l, sfr_l = [], [], [], [], [], []
    for i in range(16):
        fname = os.path.join(snapdir, f'snap_{snap:03d}.{i}.hdf5')
        if not os.path.exists(fname):
            continue
        with h5py.File(fname, 'r') as f:
            if 'PartType0' not in f:
                continue
            pos_l.append(f['PartType0/Coordinates'][:])
            mass_l.append(f['PartType0/Masses'][:])
            dens_l.append(f['PartType0/Density'][:])
            u_l.append(f['PartType0/InternalEnergy'][:])
            xe_l.append(f['PartType0/ElectronAbundance'][:])
            sfr_l.append(f['PartType0/StarFormationRate'][:])
    if not pos_l:
        return None
    return dict(
        pos=np.concatenate(pos_l).astype(np.float32) / 1000.0,   # Mpc/h
        mass=np.concatenate(mass_l).astype(np.float32),          # 1e10 Msun/h code
        density=np.concatenate(dens_l).astype(np.float32),       # code (comoving)
        u=np.concatenate(u_l).astype(np.float32),                # (km/s)^2
        xe=np.concatenate(xe_l).astype(np.float32),
        sfr=np.concatenate(sfr_l).astype(np.float32),
    )


def load_halos(sim_id, snap, mass_threshold):
    f = fof_file(sim_id, snap)
    halo_pos, halo_mass = [], []
    if os.path.exists(f):
        with h5py.File(f, 'r') as fh:
            if 'Group/GroupPos' in fh:
                halo_pos = fh['Group/GroupPos'][:]
            if 'Group/Group_M_Crit200' in fh:
                halo_mass = fh['Group/Group_M_Crit200'][:]
    if len(halo_mass) > 0:
        halo_pos = np.array(halo_pos) / 1000.0
        halo_mass = np.array(halo_mass) * 1e10
        mask = halo_mass > mass_threshold
        return halo_pos[mask], halo_mass[mask]
    return np.array([]), np.array([])


# ============================================================
# Projection pipeline (shared by mass + thermo so they align exactly)
# ============================================================

def _minimum_image(positions, halo_center):
    delta = positions - halo_center
    return delta - BOX_SIZE * np.round(delta / BOX_SIZE)


def _tile_edges_with_index(positions, idx, buffer=10.0):
    """Periodic copies near edges, carrying an index array (for weights)."""
    all_pos, all_idx = [positions], [idx]
    edge = BOX_SIZE / 2.0 - buffer
    for axis in range(3):
        near_hi = positions[:, axis] > edge
        if np.any(near_hi):
            cp = positions[near_hi].copy()
            cp[:, axis] -= BOX_SIZE
            all_pos.append(cp)
            all_idx.append(idx[near_hi])
        near_lo = positions[:, axis] < -edge
        if np.any(near_lo):
            cp = positions[near_lo].copy()
            cp[:, axis] += BOX_SIZE
            all_pos.append(cp)
            all_idx.append(idx[near_lo])
    return np.vstack(all_pos), np.concatenate(all_idx)


def _cic(positions, weights):
    pos_ = np.ascontiguousarray(positions.astype(np.float32))[:, [0, 1]]
    w_ = np.ascontiguousarray(weights.astype(np.float32))
    field = np.zeros((NPIX, NPIX), dtype=np.float32)
    MASL.MA(pos_, field, BOX_SIZE, MAS='CIC', W=w_, verbose=False)
    return field


def project_weighted(positions, weight_list, halo_center, seed):
    """Rotate (seeded) and CIC-project every weight array in weight_list.

    Returns a list of full NPIX² maps, one per weight. All weights share the
    SAME rotation (seed), so mass and thermo maps are pixel-aligned.
    """
    centered = _minimum_image(positions, halo_center)
    in_cube = np.all(np.abs(centered) < BOX_SIZE, axis=1)
    pos_ic = centered[in_cube]
    idx_ic = np.arange(in_cube.sum(), dtype=np.int64)

    tiled_pos, tiled_idx = _tile_edges_with_index(pos_ic, idx_ic, buffer=10.0)

    np.random.seed(seed)
    rot = Rotation.from_euler('xyz', np.random.uniform(0, 2 * np.pi, 3))
    rotated = tiled_pos @ rot.as_matrix().T

    shifted = rotated + BOX_SIZE / 2.0
    in_box = np.all((shifted >= 0) & (shifted < BOX_SIZE), axis=1)
    final_pos = shifted[in_box]
    final_idx = tiled_idx[in_box]

    return [_cic(final_pos, w[in_cube][final_idx]) for w in weight_list]


def extract_multiscale(field_full, target_res):
    """4 nested DM scales [6.25, 12.5, 25, 50] Mpc/h at target_res."""
    scales = [6.25, 12.5, 25.0, 50.0]
    out = np.zeros((4, target_res, target_res), dtype=np.float32)
    full_res = field_full.shape[0]
    pix = BOX_SIZE / full_res
    c = full_res // 2
    for i, s in enumerate(scales):
        if s >= BOX_SIZE:
            factor = full_res // target_res
            out[i] = (field_full.reshape(target_res, factor, target_res, factor).mean(axis=(1, 3))
                      if factor > 1 else field_full)
        else:
            hp = int(s / (2 * pix))
            cut = field_full[c - hp:c + hp, c - hp:c + hp]
            n = cut.shape[0]
            if n == target_res:
                out[i] = cut
            elif n > target_res:
                factor = n // target_res
                out[i] = cut.reshape(target_res, factor, target_res, factor).mean(axis=(1, 3))
            else:
                from scipy.ndimage import zoom
                out[i] = zoom(cut, target_res / n, order=1)
    return out


def safe_divide(num, den):
    thresh = 1e-12 * max(float(den.max()), 1e-30)
    return np.divide(num, den, out=np.zeros_like(num), where=den > thresh)


# ============================================================
# Per (sim, snapshot) processing
# ============================================================

def gas_temperature_K(u_code, xe):
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)
    return (GAMMA - 1.0) * (u_code * 1e6) * mu * M_PROTON_KG / K_B_J_PER_K


def output_path(sim_id, snap):
    split = 'test' if sim_id in test_sims else 'train'
    base = os.path.join(args.output_base_root, args.output_name, split,
                        f'sim_{sim_id}', f'snap_{snap:03d}')
    return base


def halo_done(out_dir, sim_id, halo_idx):
    f = os.path.join(out_dir, f'sim_{sim_id}_halo_{halo_idx}_rot_0.npz')
    if not os.path.exists(f):
        return False
    try:
        with np.load(f, allow_pickle=False) as d:
            return all(k in d.files for k in THERMO_KEYS)
    except Exception:
        return False


def process_sim_snap(sim_id, snap):
    out_dir = output_path(sim_id, snap)
    try:
        z, a, h = read_header(sim_id, snap)
    except RuntimeError as exc:
        print(f'[rank {rank}] sim {sim_id} snap {snap}: SKIP ({exc})', flush=True)
        return
    halo_pos, halo_mass = load_halos(sim_id, snap, args.mass_threshold)
    if len(halo_pos) == 0:
        print(f'[rank {rank}] sim {sim_id} snap {snap} (z={z:.2f}): no halos > '
              f'{args.mass_threshold:.0e}, skipping', flush=True)
        return

    # Resume: skip if every halo file is already complete.
    if all(halo_done(out_dir, sim_id, hi) for hi in range(len(halo_pos))):
        print(f'[rank {rank}] sim {sim_id} snap {snap}: already complete', flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)

    dm_pos, dm_mass, hdm_pos, hdm_mass, star_pos, star_mass = load_simulation(sim_id, snap)
    gas = load_gas_fields(sim_id, snap)
    if gas is None:
        print(f'[rank {rank}] sim {sim_id} snap {snap}: no gas, skipping', flush=True)
        return
    params = np.array(list(metadata.iloc[sim_id].to_dict().values()))

    # --- redshift-dependent comoving->physical factors (a from header) ---
    # physical density [kg/m^3]: comoving code density * (h^2 / a^3) unit chain.
    density_to_kg_m3 = 1e10 * MSUN_KG * h**2 / KPC_IN_M**3 / a**3
    mass_code_to_kg = 1e10 * MSUN_KG / h
    # physical (proper) pixel side [m]: comoving Mpc/h pixel * a.
    pixel_side_m = (BOX_SIZE / NPIX) / h * MPC_IN_M * a
    pixel_area_m2 = pixel_side_m**2

    # --- non-star-forming gas -> per-particle thermo weights ---
    hot = gas['sfr'] <= 0.0
    g_pos, g_mass = gas['pos'][hot], gas['mass'][hot]
    g_dens, g_u, g_xe = gas['density'][hot], gas['u'][hot], gas['xe'][hot]
    T = gas_temperature_K(g_u, g_xe)                                     # [K], a-independent
    rho_phys = g_dens.astype(np.float64) * density_to_kg_m3             # [kg/m^3]
    P = ((GAMMA - 1.0) * rho_phys * g_u.astype(np.float64) * 1e6).astype(np.float32)  # [Pa]
    n_e = (g_xe.astype(np.float64) * X_H * rho_phys / M_PROTON_KG * 1e-6).astype(np.float32)  # [cm^-3]
    n_e_safe = np.where(n_e > 0, n_e, 1.0)
    K = ((K_B_J_PER_K * T / KEV_IN_J) / n_e_safe**(2.0 / 3.0)).astype(np.float32)  # [keV cm^2]
    K[n_e <= 0] = 0.0
    # Compton-y per-particle integrand / pixel_area (float64 -> float32 safe).
    n_e_count = g_xe.astype(np.float64) * X_H * (g_mass.astype(np.float64) * mass_code_to_kg) / M_PROTON_KG
    y_int = SIGMA_T_M2 * K_B_J_PER_K * T.astype(np.float64) * n_e_count / M_E_C2_J
    w_thermo = [g_mass.astype(np.float32),
                (T * g_mass).astype(np.float32),
                (P * g_mass).astype(np.float32),
                (K * g_mass).astype(np.float32),
                (y_int / pixel_area_m2).astype(np.float32)]

    n_done = 0
    for halo_idx in range(len(halo_pos)):
        if halo_done(out_dir, sim_id, halo_idx):
            n_done += 1
            continue
        halo_center = halo_pos[halo_idx]
        seed = sim_id * 1000 + halo_idx * 100  # rot 0 (matches single-z seed at rot=0)

        # Mass maps (DM condition+large_scale, hydro DM, Gas[all], Stars).
        nbody_full, = project_weighted(dm_pos, [dm_mass], halo_center, seed)
        hdm_full, = project_weighted(hdm_pos, [hdm_mass], halo_center, seed)
        # gas['mass'] is in code units (1e10 Msun/h); the mass TARGET map must be
        # physical Msun like DM_hydro/Stars/condition (load_gas_fields keeps code
        # units because the thermo weights/mass_code_to_kg need them).
        gas_full, = project_weighted(gas['pos'], [gas['mass'] * 1e10], halo_center, seed)
        star_full, = project_weighted(star_pos, [star_mass], halo_center, seed)

        nbody = extract_multiscale(nbody_full, resolution)
        target = np.stack([hdm_full[SLICE], gas_full[SLICE], star_full[SLICE]], axis=0)

        # Thermo maps (same rotation/seed -> pixel-aligned).
        m_c, Tm_c, Pm_c, Km_c, y_c = (mp[SLICE] for mp in
                                      project_weighted(g_pos, w_thermo, halo_center, seed))

        np.savez_compressed(
            os.path.join(out_dir, f'sim_{sim_id}_halo_{halo_idx}_rot_0.npz'),
            condition=nbody[0],
            target=target.astype(np.float32),
            large_scale=nbody[1:],
            params=params,
            halo_mass=halo_mass[halo_idx],
            halo_center=halo_center,
            redshift=np.float32(z),
            scale_factor=np.float32(a),
            compton_y=y_c.astype(np.float32),
            temperature=safe_divide(Tm_c, m_c).astype(np.float32),
            entropy=safe_divide(Km_c, m_c).astype(np.float32),
            pressure=safe_divide(Pm_c, m_c).astype(np.float32),
        )
        n_done += 1
    print(f'[rank {rank}] sim {sim_id} snap {snap} (z={z:.2f}): '
          f'{n_done}/{len(halo_pos)} halos', flush=True)


# ============================================================
# Main: distribute (sim, snap) work items across ranks
# ============================================================

if __name__ == '__main__':
    if rank == 0:
        work = [(sid, snap) for sid in range(start_sim, end_sim) for snap in SNAPSHOTS]
        # Shuffle so each rank (work[rank::size]) gets a representative MIX of
        # snapshots/redshifts rather than being pinned to one. Without this, when
        # size is a multiple of len(SNAPSHOTS) every rank lands on a single
        # snapshot, so the high-z (≈no-halo) ranks finish in minutes and idle
        # while the z≈0 ranks carry all the work. A dedicated RNG keeps the
        # train/test split (global RNG, already drawn) untouched and reproducible.
        random.Random(args.seed).shuffle(work)
        print(f'[rank 0] {len(work)} (sim,snap) items over {size} ranks (shuffled); '
              f'snapshots={SNAPSHOTS}; rotations={num_rotations}', flush=True)
        out_root = os.path.join(args.output_base_root, args.output_name)
        os.makedirs(os.path.join(out_root, 'train'), exist_ok=True)
        os.makedirs(os.path.join(out_root, 'test'), exist_ok=True)
    else:
        work = None
    work = comm.bcast(work, root=0)

    for sim_id, snap in work[rank::size]:
        try:
            process_sim_snap(sim_id, snap)
        except Exception as exc:
            print(f'[rank {rank}] sim {sim_id} snap {snap}: UNHANDLED {exc}', flush=True)
            traceback.print_exc()

    comm.Barrier()
    if rank == 0:
        print('All ranks finished.', flush=True)
