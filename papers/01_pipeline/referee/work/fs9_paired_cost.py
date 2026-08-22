#!/usr/bin/env python3
"""fidswap -- MEASURE the per-run cost of the paired_stats regeneration.

The one job in the fidswap critical path that does not fit on this node is
re-deriving all 60 twobound paired_stats.npz against the new fiducial.  This
times the dominant term -- bind.cli.paired_stats._perreal on one 50-realization
kappa cube -- so the author gets a measured wall-clock per task instead of an
estimate.  Nothing is written into any campaign tree; the cube is discarded.

    python referee/work/fs9_paired_cost.py [--run 49]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/src")
from bind.cli.paired_stats import _perreal, _sigma0  # noqa: E402

SCI = Path("/mnt/home/mlee1/ceph/bind_science")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=49)
    ap.add_argument("--n_real", type=int, default=50)
    a = ap.parse_args()
    p = SCI / f"runs/twobound/run_{a.run:04d}/kappa_maps.npz"
    t0 = time.time()
    K = np.load(p)["kappa"][:a.n_real]
    t_load = time.time() - t0
    t0 = time.time()
    sig0 = _sigma0(K, 5.0)
    t_sig = time.time() - t0
    t0 = time.time()
    _perreal(K, 5.0, sig0)
    t_pr = time.time() - t0
    print(f"run_{a.run:04d}  n_real={a.n_real}  load {t_load:.1f}s  sigma0 {t_sig:.1f}s  "
          f"_perreal {t_pr:.1f}s   TOTAL per twobound task ~ {t_load+t_sig+t_pr:.0f}s")
    print(f"60 tasks serial on 1 core: {(t_load+t_sig+t_pr)*60/60:.0f} min "
          f"({(t_load+t_sig+t_pr)*60/3600:.1f} h)")


if __name__ == "__main__":
    main()
