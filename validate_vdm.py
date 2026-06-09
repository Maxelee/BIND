#!/usr/bin/env python
"""Compare the in-progress VDM checkpoint vs flow-matching, per channel, vs truth.

Generates patches on held-out test patches with both models and reports the three
things that decide whether the VDM is the way forward:
  1. integrated mass per channel (gen/truth) — is the stellar mass biased?
  2. radial (BCG-centered) profile per channel
  3. power spectrum per channel — does VDM recover small-scale power FM smooths away?

VDM needs MANY more sampling steps than FM (--vdm_steps default 250 vs --fm_steps 50).
Each model is generated with ITS OWN norm_stats (VDM single-head, FM two-head) and
denormalized to standard 3-channel physical via _denormalize_to_physical, so the
comparison is against a shared physical truth.

Safe to run repeatedly while the VDM trains (reads the latest last.ckpt).
Run:  python validate_vdm.py --vdm_steps 250 ; submit with run_validate_vdm.sh
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bind.data import load_file_list, AstroDataset, NormStats
from bind.train import FlowMatchingLit
from bind.inference.pipeline import _denormalize_to_physical
from bind.metrics import radial_profile, CHANNEL_NAMES

PATCH_BOX = 6.25  # Mpc/h per 128px halo patch


def radial_bins(npix, box, nbins=22):
    """Azimuthal k-bins (h/Mpc) for a npix x npix patch of physical size box."""
    kf = np.fft.fftfreq(npix) * npix * (2 * np.pi / box)
    kx, ky = np.meshgrid(kf, kf, indexing="ij")
    kk = np.sqrt(kx ** 2 + ky ** 2)
    edges = np.geomspace(2 * np.pi / box, kk.max(), nbins + 1)
    idx = np.clip(np.digitize(kk.ravel(), edges) - 1, 0, nbins - 1)
    cent = np.sqrt(edges[:-1] * edges[1:])
    return idx, cent, nbins


def load_ckpt(run_dir, ckpt, device):
    p = Path(run_dir) / "checkpoints" / ckpt
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found (training far enough?): {p}")
    model = FlowMatchingLit.load_from_checkpoint(str(p), map_location=device).eval().to(device)
    ns = NormStats.load(Path(run_dir) / "norm_stats.npz")
    return model, ns


@torch.no_grad()
def generate(model, ns, files, device, n_steps, batch, workers):
    loader = DataLoader(AstroDataset(files, ns), batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True)
    real, gen, losses = [], [], []
    for b in loader:
        cond, ls, pr = b["condition"].to(device), b["large_scale"].to(device), b["params"].to(device)
        tgt = b["target"].to(device)
        # training-objective loss on held-out data: VDM eps-MSE / FM velocity-MSE.
        # eps-MSE ~ 1.0 == predicting eps=0 (learned nothing); << 1 == learning.
        losses.append(float(model.fm.loss(tgt, cond, ls, pr)))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            g = model.fm.sample(cond, ls, pr, n_steps=n_steps)
        real.append(_denormalize_to_physical(b["target"].numpy().copy(), ns))
        gen.append(_denormalize_to_physical(g.float().cpu().numpy(), ns))
    return np.concatenate(real), np.concatenate(gen), float(np.mean(losses))


def stacked_pk(fields, idx, nb):
    S = np.zeros(nb)
    for f in fields:
        F = np.fft.fft2(f - f.mean())
        S += np.bincount(idx, weights=(np.abs(F) ** 2).ravel(), minlength=nb)
    return S / len(fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu")
    ap.add_argument("--runs_dir", default="/mnt/home/mlee1/ceph/fm_runs")
    ap.add_argument("--fm_run", default="fm_two_head")
    ap.add_argument("--fm_ckpt", default="last.ckpt")
    ap.add_argument("--vdm_run", default="vdm")
    ap.add_argument("--vdm_ckpt", default="last.ckpt")
    ap.add_argument("--fm_steps", type=int, default=50)
    ap.add_argument("--vdm_steps", type=int, default=250, help="VDM needs many more steps than FM")
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_show", type=int, default=4, help="example halos to render (truth/FM/VDM)")
    ap.add_argument("--out", default="ceph/fm_diag/vdm_vs_fm.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)
    fall = load_file_list(args.data_root, "test")
    files = [fall[i] for i in rng.choice(len(fall), min(args.n_test, len(fall)), replace=False)]
    idx, kcent, nb = radial_bins(128, PATCH_BOX)
    print(f"[vdm-vs-fm] {len(files)} test patches | FM {args.fm_steps} steps, VDM {args.vdm_steps} steps | {device}")

    gens = {}; train_loss = {}
    truth = None
    for tag, run, ckpt, steps in [("FM", args.fm_run, args.fm_ckpt, args.fm_steps),
                                  ("VDM", args.vdm_run, args.vdm_ckpt, args.vdm_steps)]:
        model, ns = load_ckpt(Path(args.runs_dir) / run, ckpt, device)
        real, gen, loss = generate(model, ns, files, device, steps, args.batch, args.workers)
        gens[tag] = gen; train_loss[tag] = loss; truth = real if truth is None else truth
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {tag}: {gen.shape}  train-objective loss={loss:.4f}")

    # ---- decisive diagnostics: is the VDM undertrained, or is sampling broken? ----
    # (1) VDM eps-MSE: ~1.0 = learned nothing; well below 1 = learning (training OK).
    # (2) log-space (decoupled from the exp-denorm): mean offset + variance ratio.
    #     If the log-space variance ratio ~1 but physical is orders low -> it's the
    #     denorm exponential amplifying a tiny offset (calibration). If log-space
    #     variance is itself far below 1 -> genuinely under-dispersed (undertrained
    #     or sampler). This separates "undertrained" from "sampler bug".
    print(f"\n=== VDM health: eps-MSE={train_loss['VDM']:.4f} (1.0=untrained)  "
          f"FM loss={train_loss['FM']:.4f} ===")
    print(f"{'channel':>9} | {'logVAR VDM/truth':>16} {'log mean off VDM':>16} | "
          f"{'logVAR FM/truth':>15}")
    for ci, cn in enumerate(CHANNEL_NAMES):
        lt = np.log10(1 + np.clip(truth[:, ci], 0, None))
        lv = np.log10(1 + np.clip(gens["VDM"][:, ci], 0, None))
        lf = np.log10(1 + np.clip(gens["FM"][:, ci], 0, None))
        print(f"{cn:>9} | {lv.var()/lt.var():16.3f} {lv.mean()-lt.mean():16.3f} | "
              f"{lf.var()/lt.var():15.3f}")

    COL = {"truth": "k", "FM": "tab:orange", "VDM": "tab:blue"}

    # ---- numeric scorecard ----
    print(f"\n{'channel':>9} | {'mass FM':>9} {'mass VDM':>9} | {'Pk/Pt FM@k50':>12} {'Pk/Pt VDM@k50':>13}")
    k50 = np.argmin(np.abs(kcent - 50))
    for ci, cn in enumerate(CHANNEL_NAMES):
        mt = truth[:, ci].sum((1, 2))
        rfm = np.median(gens["FM"][:, ci].sum((1, 2)) / np.clip(mt, 1e-30, None))
        rvd = np.median(gens["VDM"][:, ci].sum((1, 2)) / np.clip(mt, 1e-30, None))
        pt = stacked_pk(truth[:, ci], idx, nb)
        pfm = stacked_pk(gens["FM"][:, ci], idx, nb) / np.where(pt > 0, pt, 1)
        pvd = stacked_pk(gens["VDM"][:, ci], idx, nb) / np.where(pt > 0, pt, 1)
        print(f"{cn:>9} | {rfm:9.3f} {rvd:9.3f} | {pfm[k50]:12.3f} {pvd[k50]:13.3f}")

    # ---- figure: rows = channels, cols = (mass, profile, P(k)) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    rr = None
    for ci, cn in enumerate(CHANNEL_NAMES):
        mt = truth[:, ci].sum((1, 2))
        # col 0: per-patch integrated mass, gen vs truth
        for tag in ["FM", "VDM"]:
            mg = gens[tag][:, ci].sum((1, 2))
            med = np.median(mg / np.clip(mt, 1e-30, None))
            ax[ci, 0].scatter(mt, mg, s=4, alpha=0.3, color=COL[tag], label=f"{tag} (×{med:.2f})")
        lim = [mt[mt > 0].min(), mt.max()]; ax[ci, 0].plot(lim, lim, "k--", lw=1)
        ax[ci, 0].set_xscale("log"); ax[ci, 0].set_yscale("log")
        ax[ci, 0].set_ylabel(f"{cn}\ngen mass"); ax[ci, 0].legend(fontsize=8)
        if ci == 0: ax[ci, 0].set_title("1. integrated mass")
        # col 1: radial (BCG) profile
        for tag, arr in [("truth", truth), ("FM", gens["FM"]), ("VDM", gens["VDM"])]:
            profs = [radial_profile(f, n_bins=40, logspace=True)[1] for f in arr[:, ci]]
            rr, _ = radial_profile(truth[0, ci], n_bins=40, logspace=True)
            ax[ci, 1].plot(rr * (PATCH_BOX / 128), np.mean(profs, 0), color=COL[tag], label=tag, lw=1.6)
        ax[ci, 1].set_xscale("log"); ax[ci, 1].set_yscale("log")
        ax[ci, 1].set_xlabel("r [Mpc/h]"); ax[ci, 1].legend(fontsize=8)
        if ci == 0: ax[ci, 1].set_title("2. radial profile")
        # col 2: power spectrum
        pt = stacked_pk(truth[:, ci], idx, nb)
        for tag, arr in [("truth", truth), ("FM", gens["FM"]), ("VDM", gens["VDM"])]:
            ax[ci, 2].loglog(kcent, stacked_pk(arr[:, ci], idx, nb), color=COL[tag], label=tag, lw=1.6)
        ax[ci, 2].set_xlabel("k [h/Mpc]"); ax[ci, 2].legend(fontsize=8)
        if ci == 0: ax[ci, 2].set_title("3. power spectrum")
    fig.suptitle(f"VDM ({args.vdm_steps} steps) vs FM ({args.fm_steps} steps) vs truth — {len(files)} patches",
                 fontsize=13)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[vdm-vs-fm] wrote {out}")

    # ---- example halos: generated channels, truth / FM / VDM ----
    show = np.argsort(-truth[:, 2].sum((1, 2)))[:args.n_show]   # most massive stellar
    nrow = args.n_show
    fig2, ax2 = plt.subplots(nrow, 9, figsize=(18, 2.1 * nrow))
    ax2 = np.atleast_2d(ax2)
    for r, i in enumerate(show):
        col = 0
        for ci in range(3):
            vmax = float(np.log10(1 + truth[i, ci]).max())
            for tag, arr in [("truth", truth), ("FM", gens["FM"]), ("VDM", gens["VDM"])]:
                a = ax2[r, col]
                a.imshow(np.log10(1 + np.clip(arr[i, ci], 0, None)), vmin=0, vmax=vmax, cmap="magma")
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(f"{CHANNEL_NAMES[ci][:4]}\n{tag}", fontsize=8)
                if col == 0:
                    a.set_ylabel(f"halo {i}", fontsize=8)
                col += 1
    fig2.suptitle("Generated channels per halo — log10(1+x), shared vmax per (halo, channel)", fontsize=12)
    fig2.tight_layout()
    out2 = out.with_name(out.stem + "_patches.png")
    fig2.savefig(out2, dpi=120)
    print(f"[vdm-vs-fm] wrote {out2}")


if __name__ == "__main__":
    main()
