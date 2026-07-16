"""Prepare a WP-A2 fiducial paint run tree (CPU, seconds). OPTIONAL — not on the
critical path.

The core truth validation does NOT need this: the fiducial paint already exists
at ``/mnt/home/mlee1/ceph/bind_lightcone_tng`` (per-snapshot composites +
co-located stage1 conditions), and the driver reuses it by default.

This helper is only for a *re-paint* scenario — chiefly if the cosmology of the
existing paint (CAMELS-CV: Omega_m=0.3, sigma8=0.8) is judged too far from
TNG300's Planck-2015 (0.3089/0.8159) and Max wants the 4-snap fiducial re-painted
at the exact TNG cosmology. It copies a chosen 35-dim parameter vector into a
paint run dir; combine with a 4-snap paint loop (mirror run_wp2_fiducial_paint.sh
but over snaps 096/071/067/063) and point the validation's --painted-root at it.

    python analysis/paper3a/scripts/make_wp2_fiducial_bundle.py \
        --fiducial-params /mnt/home/mlee1/ceph/bind_science/runs/fiducial/run_0000/params.npy
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

FID_SRC = Path("/mnt/home/mlee1/ceph/bind_science/runs/fiducial/run_0000/params.npy")
SNAPS = [96, 71, 67, 63]  # z ~ 0.03 / 0.42 / 0.50 / 0.60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("/mnt/home/mlee1/ceph/paper3/A/wp2_fiducial"))
    ap.add_argument("--fiducial-params", type=Path, default=FID_SRC)
    args = ap.parse_args()

    params = np.load(args.fiducial_params)
    if params.shape != (35,):
        raise SystemExit(f"expected a (35,) fiducial param vector, got {params.shape} from {args.fiducial_params}")

    run_dir = args.run_root / "run_0000"
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / "params.npy"
    shutil.copy2(args.fiducial_params, dst)
    print(f"fiducial params ({params.shape}) -> {dst}")
    print(f"cosmology (Omega_m, sigma8, ...): {params[:5]}")
    print(f"paint snapshots: {SNAPS} (from conditions bind_portable_twobound/conditions/snap_NNN)")
    print("next (⛔ Max submits): sbatch analysis/paper3a/scripts/run_wp2_fiducial_paint.sh")


if __name__ == "__main__":
    main()
