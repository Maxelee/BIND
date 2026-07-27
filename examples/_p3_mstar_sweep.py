#!/usr/bin/env python3
"""
P3 prep: per-halo painted central stellar-mass sweep over 256 SB35 Sobol runs.

Output: /mnt/ceph/bind_science/ksz_confront/lightcone/catalogs/mstar_central/run_NNNN_snapXXX.npz
with per-halo keys: slab, idx_in_slab, halo_center, logM200, r200, log_mstar_central

Replicates central-M* convention from ksz_tau_gnfw.py::central_mstar():
  M_central = sum(stars_patch[r_kpc/h < 50])
  log_M_central = log10(max(M_central, 1.0))

Idempotent + parallelizable via --start/--stop args.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
RUNS = CEPH / "bind_sb35/runs"
OUT_DIR = CEPH / "bind_science/ksz_confront/lightcone/catalogs/mstar_central"

# Constants from ksz_tau_gnfw.py
PIX_MPCH = 6.25 / 128.0                   # comoving patch pixel [Mpc/h]
MSTAR_AP_KPCH = 50.0                      # central aperture [kpc/h]
MSTAR_CUTS = np.array([11.0, 11.25])      # log10 M* thresholds
REF_CUTS = MSTAR_CUTS                     # validation reference


def central_mstar(stars_patch, rr_kpch):
    """Central-galaxy M* proxy = Stars mass within MSTAR_AP_KPCH of centre."""
    return stars_patch[rr_kpch < MSTAR_AP_KPCH].sum()


def process_run(run_id, snap, start=0, stop=256):
    """Process one run (4 slabs), output one shard. Idempotent."""
    if run_id < start or run_id >= stop:
        return

    rd = RUNS / f"run_{run_id:04d}" / f"snap_{snap:03d}"
    shard_path = OUT_DIR / f"run_{run_id:04d}_snap{snap:03d}.npz"

    if shard_path.exists():
        return  # Already exists, skip

    # Collect per-halo data across all 4 slabs
    slabs_data = []

    for slab_idx in range(4):
        slab_file = rd / f"composite_slab{slab_idx:02d}.npz"
        if not slab_file.exists():
            continue

        d = np.load(slab_file)
        if int(d["n_halos"]) == 0:
            continue

        gp = d["generated_patches"]
        stars = gp[:, 2]                             # [n, 128, 128] Msun/h per pixel
        M = d["halo_masses"]                          # [n] Msun/h
        r200 = d["halo_r200"]                         # [n] Mpc/h
        centers = d["halo_centers"]                   # [n, 2] Mpc/h

        P = stars.shape[-1]
        cc = P // 2
        yy, xx = np.mgrid[0:P, 0:P]
        rr = np.hypot(xx - cc, yy - cc) * PIX_MPCH   # Mpc/h comoving
        rr_kpch = rr * 1e3                            # kpc/h

        n_halos = len(M)
        for h in range(n_halos):
            mstar_central = central_mstar(stars[h], rr_kpch)
            log_mstar = np.log10(max(mstar_central, 1.0))

            slabs_data.append({
                'slab': np.int8(slab_idx),
                'idx_in_slab': np.int32(h),
                'halo_center': centers[h].astype(np.float32),
                'logM200': np.log10(M[h]).astype(np.float32),
                'r200': r200[h].astype(np.float32),
                'log_mstar_central': log_mstar.astype(np.float32),
            })

    if not slabs_data:
        return  # No halos found

    # Stack into arrays
    n_tot = len(slabs_data)
    slab_arr = np.array([d['slab'] for d in slabs_data], dtype=np.int8)
    idx_arr = np.array([d['idx_in_slab'] for d in slabs_data], dtype=np.int32)
    center_arr = np.array([d['halo_center'] for d in slabs_data], dtype=np.float32)
    logm200_arr = np.array([d['logM200'] for d in slabs_data], dtype=np.float32)
    r200_arr = np.array([d['r200'] for d in slabs_data], dtype=np.float32)
    logmstar_arr = np.array([d['log_mstar_central'] for d in slabs_data], dtype=np.float32)

    # Save shard
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(shard_path,
             slab=slab_arr,
             idx_in_slab=idx_arr,
             halo_center=center_arr,
             logM200=logm200_arr,
             r200=r200_arr,
             log_mstar_central=logmstar_arr)


def sweep(snap=85, start=0, stop=256, rank=0, size=1):
    """Process runs [start:stop], optionally MPI-parallelized."""
    n_runs = stop - start
    runs = list(range(start, stop))
    mine = runs[rank::size]

    for k, run_id in enumerate(mine):
        process_run(run_id, snap, start, stop)
        if rank == 0 and (k + 1) % 16 == 0:
            print(f"[mstar] rank0 {k+1}/{len(mine)} runs (snap {snap})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snap", type=int, default=85, help="Snapshot (default 85)")
    parser.add_argument("--start", type=int, default=0, help="Start run (default 0)")
    parser.add_argument("--stop", type=int, default=256, help="Stop run (default 256)")
    parser.add_argument("--mpi", action="store_true", help="Use MPI parallelization")
    args = parser.parse_args()

    if args.mpi:
        try:
            from mpi4py import MPI
            rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
        except ImportError:
            rank, size = 0, 1
    else:
        rank, size = 0, 1

    sweep(snap=args.snap, start=args.start, stop=args.stop, rank=rank, size=size)
