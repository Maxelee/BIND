"""WP-B4 Sobol extension: backfill per-run halo_scaling.npz for the SB35 suite.

The manifold-coverage extension (wp5 REPORT ladder item 8) runs the B4 chain
over the 253 raytraced SB35 Sobol units. Assembly needs each unit's
`halo_scaling.npz` for the (Delta ln M_gas, Delta ln T) coordinates — present
for the twobound/bind/truth units but never computed for SB35.

This computes them with the IDENTICAL code that produced the existing files:
`bind.inference.stats.halo_scaling` imported from the live checkout at
/mnt/home/mlee1/BIND (the June stats vintage; its git commit is recorded in
each output npz). Composite slab paths are built in the same sorted order as
stage 5, so the concatenated halo skeleton matches the fiducial's ordering
and the assembly's exact-match assertion stays meaningful.

Skip-if-exists; stride over an array-task index (see the .sh wrapper).
IO-bound: ~13 GB of composites per run, ~10-15 min each.

    python -m analysis.paper3b.scripts.run_b4_sobol_halo_scaling --task-id 0 --n-tasks 16
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

BIND_LIVE = Path("/mnt/home/mlee1/BIND")
SB35_RUNS = Path("/mnt/home/mlee1/ceph/bind_sb35/runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="explicit run names (default: all with kappa_maps)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(BIND_LIVE / "src"))
    from bind.inference import stats as S  # noqa: E402  (live-checkout import)

    commit = subprocess.run(["git", "-C", str(BIND_LIVE), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"[sb35-hs] bind.inference.stats from {BIND_LIVE} @ {commit}", flush=True)

    if args.runs:
        runs = list(args.runs)
    else:
        runs = sorted(p.name for p in SB35_RUNS.iterdir()
                      if p.is_dir() and (p / "kappa_maps.npz").exists())
    runs = runs[args.task_id::args.n_tasks]
    print(f"[sb35-hs] task {args.task_id}/{args.n_tasks}: {len(runs)} runs", flush=True)

    for run in runs:
        rd = SB35_RUNS / run
        out = rd / "halo_scaling.npz"
        if out.exists() and not args.overwrite:
            print(f"[sb35-hs] skip {run} (exists)", flush=True)
            continue
        paths = sorted(rd.glob("snap_*/composite_slab*.npz"))
        if not paths:
            print(f"[sb35-hs] {run}: no composites — skipping", flush=True)
            continue
        hs = S.halo_scaling(paths)
        np.savez(out, **hs, bind_commit=commit, n_slab_files=len(paths))
        print(f"[sb35-hs] {run}: {len(hs['halo_mass'])} halos "
              f"from {len(paths)} slabs", flush=True)


if __name__ == "__main__":
    main()
