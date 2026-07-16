"""Prepare the WP-A2 fiducial paint run tree (CPU, seconds).

The truth validation needs BIND painted at the *fiducial* TNG parameters (the
cosmology/astro at which TNG300-hydro *is* the ground truth). The DMO conditions
are shared across theta, so the fiducial paint reuses the existing
``bind_portable_twobound`` conditions + weights already on Popeye — only the
35-dim fiducial parameter vector differs. This copies that vector (from the
Paper-2 fiducial run) into the paint run dir the Slurm paint script expects.

    python analysis/paper3a/scripts/make_wp2_fiducial_bundle.py

Then ⛔ Max submits run_wp2_fiducial_paint.sh.
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
