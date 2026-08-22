#!/usr/bin/env python3
"""fidswap -- extended-nu Minkowski functionals (nu in [-3,8]) for the NEW fiducial.

mf_cache.py::compute with the 'bind' side re-pointed at a twobound replica.  The
truth legs are unaffected and are copied verbatim from the shipped cache, so the
output is a drop-in for bind_science/mf_cache/mf_nu8_snap096.npz.

GATE: recompute one shipped shard (old fiducial, plane 1) and diff it.

    python referee/work/fs5_mfcache.py --gate
    python referee/work/fs5_mfcache.py --run 49
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mf_cache as mfc  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
OUT = CEPH / "referee_work/fidswap/mf_cache"
KEYS = mfc.KEYS


def compute(kappa_path: Path, plane: int) -> dict:
    from bind.inference.stats import nongaussian_stats
    a = np.load(kappa_path)
    K = a["kappa"][:, plane].astype(np.float32)
    del a
    o = nongaussian_stats(K[:, None], fov_deg=mfc.FOV_DEG,
                          smoothing_scales_arcmin=(mfc.SMOOTH_ARCMIN,),
                          mf_thresholds=mfc.MF_NU8, return_realizations=True)
    r = {k: np.asarray(o[f"{k}_real"])[:, 0] for k in KEYS}
    r.update({f"{k}_mean": np.asarray(o[k])[0] for k in KEYS})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=49)
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    ship = np.load(SCI / "mf_cache/mf_nu8_snap096.npz")
    if a.gate:
        t0 = time.time()
        r = compute(SCI / "runs/bind/run_0000/kappa_maps.npz", 1)
        for k in KEYS:
            ref = ship[f"bind_{k}_mean"][1]
            rel = float(np.nanmax(np.abs(np.where(ref != 0, r[f"{k}_mean"] / ref - 1, 0))))
            print(f"GATE bind_{k}_mean z_s=1: max|rel diff| = {rel:.3e} "
                  + ("PASS" if rel < 1e-12 else "FAIL"))
        print(f"({time.time()-t0:.1f}s for one plane)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    src = SCI / f"runs/twobound/run_{a.run:04d}/kappa_maps.npz"
    per_real = {k: [] for k in KEYS}
    per_mean = {k: [] for k in KEYS}
    for zi in range(mfc.N_PLANES):
        t0 = time.time()
        r = compute(src, zi)
        for k in KEYS:
            per_real[k].append(r[k])
            per_mean[k].append(r[f"{k}_mean"])
        print(f"plane {zi}  {time.time()-t0:6.1f}s", flush=True)
    store = {k: ship[k] for k in ship.files if k.startswith("truth_")}
    store["mf_nu"] = mfc.MF_NU8
    for k in KEYS:
        store[f"bind_{k}"] = np.stack(per_real[k], axis=1)
        store[f"bind_{k}_mean"] = np.stack(per_mean[k], axis=0)
    store["fid_run"] = np.array(f"twobound/run_{a.run:04d}")
    out = OUT / f"mf_nu8_snap096_fidtb{a.run:02d}.npz"
    np.savez_compressed(out, **store)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
