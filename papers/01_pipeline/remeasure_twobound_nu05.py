"""Remeasure the twobound suite's nu-domain statistics on the CANONICAL
nu05 grid (22 centers, dnu=0.5, -2.75..7.75) -- the same grid and the same
engine (`nu_grid.compute`, i.e. bind.inference.stats) that produced the
Sobol suite's emulator_dataset_nu05 statistics.

Motivation (2026-08-12, author): the original twobound reduction stored the
Minkowski functionals only on a 29-point nu=-3..4 grid (the
`nongaussian_stats` default), capping the family-basis MF model at
nu<=3.75 while every other figure runs to nu~8. The kappa maps
(kappa_maps.npz, (50, 5, 1024, 1024)) are on disk, so this is a
recomputation, not a re-trace.

Writes per run:  <run_dir>/nu05_stats.npz
  nu (22,), peak_counts / minima_counts / pdf / mf_v0 / mf_v1 / mf_v2
  (5, 22) realization means, + peak_counts_err / minima_counts_err.
The original nongaussian_stats.npz / paired_stats.npz are NOT touched.

Run:
    /mnt/home/mlee1/venvs/BIND_env/bin/python remeasure_twobound_nu05.py [n_workers]
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline")

TB = Path("/mnt/home/mlee1/ceph/bind_science/runs/twobound")
OUT_NAME = "nu05_stats.npz"


def one_run(r: int) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from nu_grid import compute
    d = TB / f"run_{r:04d}"
    out = d / OUT_NAME
    if out.exists():
        return f"run_{r:04d}: exists, skipped"
    t0 = time.time()
    res = compute(d / "kappa_maps.npz")
    np.savez_compressed(out, **{k: v for k, v in res.items()
                                if not k.endswith("_real")})
    return f"run_{r:04d}: done in {time.time() - t0:.0f}s"


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    runs = [r for r in range(60) if (TB / f"run_{r:04d}" / "kappa_maps.npz").exists()]
    print(f"{len(runs)} runs, {nw} workers")
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for msg in ex.map(one_run, runs):
            print(msg, flush=True)
    print("ALL DONE")
