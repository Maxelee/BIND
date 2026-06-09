#!/usr/bin/env python
"""Build the refiner training set: matched (BIND patch, truth patch) pairs.

The refiner is a small CNN that maps a BIND-generated halo patch -> the true
hydro patch, per channel, to restore the high-k stellar/gas core concentration
the flow under-delivers (bind_vs_truth_patches.py: T_stars~0.55-0.75).  It is a
SECOND emulator with the same contract as BIND -- trained on truth pairs,
deployed with NO truth -- so legitimacy rests on generalizing to HELD-OUT theta.
We therefore split by simulation (= by Sobol theta): disjoint sim sets for
train/val, so the val score is a true held-out-parameter generalization test.

Inputs are the CACHED BIND patches (fm_two_head/generated_halos.npz, last.ckpt)
+ truth cutouts from full_maps.npz -- pure CPU cache read, no flow inference.
Writes to a fresh dir (default ceph/refiner_two_head/), never touching any
training run or released weights.

Run:  python refiner_data.py --model fm_two_head --val_frac 0.3
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from kernel_bcm_feasibility import BOX, NPIX, _periodic_cutout

PATCH_PIX = 128
CHANNELS = ["DM_hydro", "Gas", "Stars"]


def collect(sims, model):
    bind_list, truth_list, sim_ids = [], [], []
    ppm = NPIX / BOX
    for s in sims:
        s = Path(s)
        gp = s / "mass_threshold_1p000e13" / model / "generated_halos.npz"
        if not gp.exists():
            continue
        gen = np.load(gp)["generated"][:, :3].astype(np.float32)  # (N,3,128,128) physical
        truth = np.load(s / "full_maps.npz")["truth_maps"].astype(np.float32)
        centers = np.load(s / "mass_threshold_1p000e13" / "halo_catalog.npz")["centers"]
        for i, c in enumerate(centers):
            cx = int(c[0] * ppm) % NPIX; cy = int(c[1] * ppm) % NPIX
            tp = np.stack([_periodic_cutout(truth[ch], cx, cy, PATCH_PIX) for ch in range(3)])
            bind_list.append(gen[i]); truth_list.append(tp.astype(np.float32))
            sim_ids.append(s.parent.name)
    return (np.asarray(bind_list), np.asarray(truth_list), np.asarray(sim_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--model", default="fm_two_head")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--out_dir", default="/mnt/home/mlee1/ceph/refiner_two_head")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sims = sorted(glob.glob(f"{args.suite_root}/Test/sim_SB35_*/snap_090"))
    # split by SIM (=theta): disjoint sets -> held-out-parameter validation
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(sims))
    n_val = int(len(sims) * args.val_frac)
    val_sims = [sims[i] for i in order[:n_val]]
    train_sims = [sims[i] for i in order[n_val:]]
    print(f"[refiner-data] {len(train_sims)} train sims / {len(val_sims)} val sims (theta-disjoint)")

    Xtr, Ytr, Str = collect(train_sims, args.model)
    Xva, Yva, Sva = collect(val_sims, args.model)
    print(f"[refiner-data] train pairs {Xtr.shape}  val pairs {Xva.shape}")

    # per-channel normalization stats from TRAIN truth, in log10(1+x) space
    def logn(x):
        return np.log10(1.0 + np.clip(x, 0, None))
    mean = np.array([logn(Ytr[:, ch]).mean() for ch in range(3)], dtype=np.float32)
    std = np.array([logn(Ytr[:, ch]).std() + 1e-6 for ch in range(3)], dtype=np.float32)
    print(f"[refiner-data] log10(1+x) norm  mean={mean}  std={std}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "train.npz", bind=Xtr, truth=Ytr, sims=Str)
    np.savez(out / "val.npz", bind=Xva, truth=Yva, sims=Sva)
    np.savez(out / "norm.npz", mean=mean, std=std, channels=np.array(CHANNELS))
    print(f"[refiner-data] wrote {out}/ (train.npz, val.npz, norm.npz)")


if __name__ == "__main__":
    main()
