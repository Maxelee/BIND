#!/usr/bin/env python3
"""fidswap step 2 -- rebuild the fiducial halo atlas from the new fiducial paint.

reduce_file() is papers/01_pipeline/build_atlas_200c.py::reduce_file VERBATIM
(same apertures, same background annulus, same 500c factor, thermo at BOTH
apertures).  Output is an apples-to-apples drop-in for
bind_science/halo_atlas/fid_snapNNN.npz: same keys, same halo ordering, with the
z/snap manifest scalars carried over from the shipped file.

INTEGRITY GATE (run first, --gate): re-reduce runs/truth/run_0000 and require
every shipped key of halo_atlas/truth_snapNNN.npz to be reproduced bit-for-bit
(max |rel diff| == 0) -- the refuse-don't-guess convention of build_atlas_200c.

Writes ONLY into /mnt/home/mlee1/ceph/referee_work/fidswap/halo_atlas/.

    python referee/work/fs2_atlas.py --gate
    python referee/work/fs2_atlas.py --run 49 --snaps all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_atlas_200c import reduce_file  # noqa: E402  (verbatim shipped reducer)

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
ATLAS = SCI / "halo_atlas"
TWOBOUND = SCI / "runs/twobound"
OUT = CEPH / "referee_work/fidswap/halo_atlas"

SNAPS = [29, 31, 33, 35, 38, 41, 43, 46, 49, 52, 56, 59, 63, 67, 71, 76, 80, 85, 90, 96]


def reduce_run(root: Path, snap: int) -> dict:
    slabs = sorted((root / f"snap_{snap:03d}").glob("composite_slab*.npz"))
    assert slabs, f"no composite slabs under {root}/snap_{snap:03d}"
    parts = [r for r in (reduce_file(s) for s in slabs) if r is not None]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _relmax(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.max(np.abs(np.where(a != 0, b / a - 1.0, b - a))))


def gate(snap=96):
    cat = reduce_run(SCI / "runs/truth/run_0000", snap)
    ref = np.load(ATLAS / f"truth_snap{snap:03d}.npz")
    worst = {}
    for k in ref.files:
        if k in ("z", "snap"):
            continue
        assert k in cat, f"missing {k}"
        worst[k] = _relmax(ref[k], cat[k])
    bad = {k: v for k, v in worst.items() if v != 0.0}
    print(f"GATE truth_snap{snap:03d}: {len(worst)} shipped keys, "
          f"max|rel diff| = {max(worst.values()):.3e}  "
          + ("PASS (bit-for-bit)" if not bad else f"FAIL {bad}"))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=49)
    ap.add_argument("--snaps", default="96", help="'all', or space/comma list")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    if a.gate:
        gate()
        return
    snaps = SNAPS if a.snaps == "all" else [int(s) for s in a.snaps.replace(",", " ").split()]
    OUT.mkdir(parents=True, exist_ok=True)
    root = TWOBOUND / f"run_{a.run:04d}"
    for sn in snaps:
        t0 = time.time()
        cat = reduce_run(root, sn)
        ref = np.load(ATLAS / f"fid_snap{sn:03d}.npz")
        cat["z"], cat["snap"] = ref["z"], ref["snap"]      # manifest lookups
        miss = sorted(set(ref.files) - set(cat))
        assert not miss, f"rebuilt atlas is missing shipped keys {miss}"
        out = OUT / f"fidtb{a.run:02d}_snap{sn:03d}.npz"
        np.savez_compressed(out, **cat)
        n = len(cat["M_fof"])
        print(f"snap{sn:03d}  {time.time()-t0:5.1f}s  n={n:5d}  z={float(cat['z']):.4f}  "
              f"med Y200/Y500={np.median(cat['Y_200c']/cat['Y_500c']):.3f}  -> {out.name}",
              flush=True)


if __name__ == "__main__":
    main()
