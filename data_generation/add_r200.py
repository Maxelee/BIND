"""
add_r200.py
===========
Append a per-halo ``r200`` scalar (R200c, comoving Mpc/h) to existing npz
training files, in place.

This is OPTIONAL. R200c is a pure function of the already-saved ``halo_mass``
(M200c) at z=0, so :func:`bind.data.r200_from_sample` derives it on the fly when
the key is absent — training does not require this script. Run it only if you
want R200 *persisted* in the data for provenance / external use (observable
conditioning, profile extraction, etc.).

New key added to each file
--------------------------
r200 : float32 scalar, R200c [comoving Mpc/h]

Derivation (h cancels in h-units; matches the FOF Group_R_Crit200 by definition):
    M200c = (4/3) pi R200c^3 * 200 * rho_crit,0,   rho_crit,0 = 2.7754e11 h^2 Msun/Mpc^3
    => R200c[Mpc/h] = ( M200c[Msun/h] / (4/3 pi * 200 * rho_crit,0) )^(1/3)

Usage
-----
Serial (whole data root):
    python add_r200.py --output_base /path/to/train_data_rotated2_128_cpu
Smoke test a couple of sims:
    python add_r200.py --only_sims 0,1
MPI (rank-sharded over files, e.g. via sbatch/srun):
    srun -n 128 python add_r200.py
"""

import argparse
import os
import sys
import numpy as np

# Critical density today, h-units (see module docstring).
RHO_CRIT_0 = 2.7754e11          # h^2 Msun / Mpc^3


def m200c_to_r200c(m200c):
    """R200c [Mpc/h] from M200c [Msun/h] at z=0."""
    return float(np.cbrt(float(m200c) / (4.0 / 3.0 * np.pi * 200.0 * RHO_CRIT_0)))


def atomic_append_r200(path):
    """Load npz, append r200 (idempotent), write tmp then os.replace.

    Returns True if the file was updated, False if it already had r200 or could
    not be processed.
    """
    try:
        with np.load(path, allow_pickle=False) as d:
            if 'r200' in d.files:
                return False
            if 'halo_mass' not in d.files:
                print(f'SKIP (no halo_mass): {path}', flush=True)
                return False
            merged = {k: d[k] for k in d.files}
    except Exception as exc:
        print(f'SKIP (unreadable): {path}: {exc}', flush=True)
        return False

    merged['r200'] = np.float32(m200c_to_r200c(merged['halo_mass']))

    assert path.endswith('.npz')
    tmp = path[:-4] + f'.tmp.{os.getpid()}.npz'
    try:
        np.savez_compressed(tmp, **merged)
        os.replace(tmp, path)        # atomic on POSIX / Ceph
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


def gather_files(output_base, only_sims=None):
    """Collect every sim_*/*.npz under output_base/{train,test}/."""
    files = []
    for split in ('train', 'test'):
        split_dir = os.path.join(output_base, split)
        if not os.path.isdir(split_dir):
            continue
        for sim_name in sorted(os.listdir(split_dir)):
            if not sim_name.startswith('sim_'):
                continue
            if only_sims is not None and int(sim_name.split('_')[1]) not in only_sims:
                continue
            sim_dir = os.path.join(split_dir, sim_name)
            if not os.path.isdir(sim_dir):
                continue
            for f in sorted(os.listdir(sim_dir)):
                if f.endswith('.npz') and f.startswith('sim_'):
                    files.append(os.path.join(sim_dir, f))
    return files


def main():
    p = argparse.ArgumentParser(description='Append r200 (R200c) to npz files.')
    p.add_argument('--output_base', type=str,
                   default='/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu')
    p.add_argument('--only_sims', type=str, default=None,
                   help='Comma-separated sim IDs to process, e.g. "0,1,2".')
    args = p.parse_args()

    only = (set(int(x) for x in args.only_sims.split(','))
            if args.only_sims else None)

    # Optional MPI sharding over the flat file list.
    try:
        import mpi4py.MPI as MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        comm, rank, size = None, 0, 1

    if rank == 0:
        files = gather_files(args.output_base, only)
        print(f'[rank 0] {len(files)} npz files across {size} rank(s).', flush=True)
    else:
        files = None
    if comm is not None:
        files = comm.bcast(files, root=0)

    my_files = files[rank::size]
    n_updated = sum(atomic_append_r200(f) for f in my_files)
    print(f'[rank {rank}] updated {n_updated}/{len(my_files)} files.', flush=True)

    if comm is not None:
        comm.Barrier()
        if rank == 0:
            print('All ranks finished.', flush=True)


if __name__ == '__main__':
    sys.exit(main())
