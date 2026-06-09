#!/usr/bin/env python
"""Compare fm_two_head checkpoints on the metric that matters: high-k transfer T(k).

We use last.ckpt for all generations, but FM val_loss only weakly tracks sample
quality (project_fm_checkpoint_selection) and the saved ladder's val losses are
nearly identical (0.2135-0.2139).  This generates patches with EACH checkpoint on
a few HELD-OUT sims and measures the per-channel transfer function
T_c(k) = sqrt(<P_gen>/<P_truth>) (esp. Stars, the high-k driver) so we can pick by
physics, not val_loss.  If all checkpoints look the same at high k, the stellar
deficit is systematic -> a refiner/retrain is needed, not a checkpoint swap.

Reads checkpoints + norm_stats read-only; writes only to --out (a fresh npz).
GPU.  Submit via run_eval_ckpt.sh.

Run:  python eval_ckpt_highk.py --run_dir /mnt/home/mlee1/ceph/fm_runs/fm_two_head \
          --n_sims 4 --out ceph/fm_diag/ckpt_highk/two_head.npz
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from bind.data import NormStats
from bind.train import FlowMatchingLit
from bind.inference.pipeline import generate_halo_patches
from bind.inference.artifacts import load_halo_cutouts
from kernel_bcm_feasibility import BOX, NPIX, _periodic_cutout
from bind_vs_truth_patches import PATCH_PIX, PATCH_BOX, radial_bins

CHANNELS = ["DM_hydro", "Gas", "Stars"]


def transfer_for_ckpt(ckpt, norm_stats, sims, device, idx, nb, n_steps, batch_size):
    model = FlowMatchingLit.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)
    # fm_two_head: raw weights only (its EMA shadow_params are corrupt), 35 params.
    n_params = getattr(model.hparams, "n_params", 35)
    param_indices = (np.array([i for i in range(35) if i not in (0, 1, 7, 8)])
                     if n_params < 35 else None)
    no_large_scale = bool(getattr(model.hparams, "no_large_scale", False))

    SA = np.zeros((4, nb)); SB = np.zeros((4, nb))  # DM, Gas, Stars, Total
    ppm = NPIX / BOX
    for s in sims:
        s = Path(s)
        mt = s / "mass_threshold_1p000e13"
        cutouts = load_halo_cutouts(mt / "halo_cutouts.npz")
        cat = np.load(mt / "halo_catalog.npz")
        sim_params = cat["params"][0].astype(np.float32)
        truth = np.load(s / "full_maps.npz")["truth_maps"].astype(np.float64)
        gen = generate_halo_patches(
            cutouts, norm_stats, sim_params, model.fm, device,
            n_steps=n_steps, batch_size=batch_size, use_amp=True,
            param_indices=param_indices, no_large_scale=no_large_scale)[:, :3].astype(np.float64)
        for i, c in enumerate(cat["centers"]):
            cx = int(c[0] * ppm) % NPIX; cy = int(c[1] * ppm) % NPIX
            gtot = np.zeros((PATCH_PIX, PATCH_PIX)); ttot = np.zeros((PATCH_PIX, PATCH_PIX))
            for ch in range(3):
                a = gen[i, ch]; b = _periodic_cutout(truth[ch], cx, cy, PATCH_PIX)
                gtot += a; ttot += b
                Fa = np.fft.fft2(a - a.mean()); Fb = np.fft.fft2(b - b.mean())
                SA[ch] += np.bincount(idx, weights=(np.abs(Fa) ** 2).ravel(), minlength=nb)
                SB[ch] += np.bincount(idx, weights=(np.abs(Fb) ** 2).ravel(), minlength=nb)
            Fa = np.fft.fft2(gtot - gtot.mean()); Fb = np.fft.fft2(ttot - ttot.mean())
            SA[3] += np.bincount(idx, weights=(np.abs(Fa) ** 2).ravel(), minlength=nb)
            SB[3] += np.bincount(idx, weights=(np.abs(Fb) ** 2).ravel(), minlength=nb)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.sqrt(SA / np.where(SB > 0, SB, 1))  # (4, nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--checkpoints", default="all",
                    help="'all' = every *.ckpt in run_dir/checkpoints, or comma list")
    ap.add_argument("--n_sims", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default="ceph/fm_diag/ckpt_highk/two_head.npz")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.checkpoints == "all":
        ckpts = sorted(glob.glob(str(run_dir / "checkpoints" / "*.ckpt")))
    else:
        ckpts = [str(run_dir / "checkpoints" / c) for c in args.checkpoints.split(",")]
    sims = sorted(glob.glob(f"{args.suite_root}/Test/sim_SB35_*/snap_090"))[:args.n_sims]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_stats = NormStats.load(run_dir / "norm_stats.npz")
    idx, kcent, nb = radial_bins(PATCH_PIX, PATCH_BOX)
    print(f"[ckpt-eval] {len(ckpts)} checkpoints x {len(sims)} sims on {device}")

    results = {}
    for ckpt in ckpts:
        name = Path(ckpt).stem
        T = transfer_for_ckpt(ckpt, norm_stats, sims, device, idx, nb,
                              args.n_steps, args.batch_size)
        results[name] = T
        kf = lambda q: T[:, np.argmin(np.abs(kcent - q))]
        print(f"\n{name}")
        print(f"  {'chan':>9} {'T(10)':>7} {'T(30)':>7} {'T(50)':>7}")
        for ci, cn in enumerate(CHANNELS + ["Total"]):
            print(f"  {cn:>9} {T[ci][np.argmin(abs(kcent-10))]:7.3f} "
                  f"{T[ci][np.argmin(abs(kcent-30))]:7.3f} {T[ci][np.argmin(abs(kcent-50))]:7.3f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, k=kcent, **{n: results[n] for n in results})
    print(f"\n[ckpt-eval] wrote {out}")

    # pick winner by Total T(k=50) closest to truth's high-k (we want T high, ~1)
    j = np.argmin(np.abs(kcent - 50))
    ranking = sorted(results.items(), key=lambda kv: -kv[1][3][j])
    print("\nbest -> worst by Total T(k=50):")
    for n, T in ranking:
        print(f"  {n}: Total T(50)={T[3][j]:.3f}  Stars T(50)={T[2][j]:.3f}")


if __name__ == "__main__":
    main()
