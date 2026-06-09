#!/usr/bin/env python
"""Validate the core-transplant on the FULL held-out-theta set (reliable numbers).

Applies the saved core codebook to BIND patches on all 30 theta-disjoint val sims,
over several random seeds (so the closure has an error bar, not a noise-floor point),
and checks the five goals:
  1. integrated mass      -- per-patch mass change (mass_mode=bind => ~0)
  4. parameter sensitivity-- per-channel integrated mass preserved (carries theta dep.)
  2. profiles             -- stacked core radial profile vs truth
  3. fields               -- per-channel patch power T(k) vs truth
  5. composite P(k)       -- S(k) closure vs truth, with truth-patch ceiling

Run:  python core_transplant_validate.py --n_seeds 4
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from core_transplant import CoreCodebook, truth_patches, RGRID, PPM
from kernel_bcm_feasibility import BOX, NPIX
from bind.inference.pipeline import build_bind_composite
from bind.inference.artifacts import load_halo_cutouts
from bind.metrics import power_spectrum_pylians_2d as PK
from bind_vs_truth_patches import radial_bins, PATCH_BOX

KW = dict(box_size=BOX, npix=NPIX, patch_pix=128, patch_mass_match=True,
          taper_frac=0.15, r200_factor=0.0)


def sk(field, dmo):
    k, p, _ = PK(field, box_size=BOX, MAS="None"); _, pd, _ = PK(dmo, box_size=BOX, MAS="None")
    return k, p / pd


def patch_power(patches, idx, nb):
    """Stacked per-channel + total |F(patch-mean)|^2 over a (N,3,128,128) array."""
    S = np.zeros((4, nb))
    for p in patches:
        tot = p.sum(0)
        for ch, f in enumerate(list(p) + [tot]):
            F = np.fft.fft2(f - f.mean())
            S[ch] += np.bincount(idx, weights=(np.abs(F) ** 2).ravel(), minlength=nb)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--model", default="fm_two_head")
    ap.add_argument("--codebook", default="/mnt/home/mlee1/ceph/refiner_two_head/core_codebook.npz")
    ap.add_argument("--R", type=float, default=12.0)
    ap.add_argument("--k_near", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ceph/fm_diag/core_transplant_validate.png")
    args = ap.parse_args()

    sims = sorted(glob.glob(f"{args.suite_root}/Test/sim_SB35_*/snap_090"))
    order = np.random.default_rng(args.seed).permutation(len(sims))
    val = [sims[i] for i in order[:int(len(sims) * args.val_frac)]]
    cb = CoreCodebook(args.codebook)
    idx, kcent, nb = radial_bins(128, PATCH_BOX)
    rbin = np.clip((RGRID).astype(int), 0, 40)               # 1px radial bins for the core profile
    rcnt = np.bincount(rbin.ravel(), minlength=41)
    print(f"[validate] {len(val)} held-out sims, {args.n_seeds} seeds, R={args.R} k_near={args.k_near}")

    s_truth, s_bind, s_ceil = [], [], []
    s_tr = [[] for _ in range(args.n_seeds)]
    Pp = {k: np.zeros((4, nb)) for k in ["BIND", "trans", "truth"]}
    prof = {k: np.zeros((3, 41)) for k in ["BIND", "trans", "truth"]}
    mass_chg, mass_ch_pre = [], []
    kref = None
    for si, s in enumerate(val):
        s = Path(s); mt = s / "mass_threshold_1p000e13"
        cat = np.load(mt / "halo_catalog.npz")
        if len(cat["masses"]) == 0:
            continue
        gen = np.load(mt / args.model / "generated_halos.npz")["generated"][:, :3].astype(np.float64)
        fm = np.load(s / "full_maps.npz"); dmo = fm["dmo_fullbox"].astype(np.float64)
        truth = fm["truth_maps"].astype(np.float64)
        tp = truth_patches(truth, cat["centers"])
        halos = [{"halo_center": cat["centers"][i, :2], "halo_mass": float(cat["masses"][i]),
                  "r200": float(cat["radii"][i]) / 1e3, "params": cat["params"][i]}
                 for i in range(len(cat["masses"]))]
        cut = load_halo_cutouts(mt / "halo_cutouts.npz")

        def comp(g):
            return build_bind_composite(dmo, halos, g, cut, **KW)["composite"].sum(0)

        k, st = sk(truth.sum(0), dmo); _, sb = sk(comp(gen), dmo); _, sc = sk(comp(tp), dmo)
        kref = k; s_truth.append(st); s_bind.append(sb); s_ceil.append(sc)
        Pp["BIND"] += patch_power(gen, idx, nb); Pp["truth"] += patch_power(tp, idx, nb)
        for nm, P in [("BIND", gen), ("truth", tp)]:
            for ch in range(3):
                prof[nm][ch] += np.bincount(rbin.ravel(),
                                            weights=P[:, ch].sum(0).ravel(), minlength=41)
        for j in range(args.n_seeds):
            tr = cb.transplant(gen, cat["masses"], R=args.R, k_near=args.k_near,
                               rng=np.random.default_rng(100 + j))
            _, srt = sk(comp(tr), dmo); s_tr[j].append(srt)
            if j == 0:
                Pp["trans"] += patch_power(tr, idx, nb)
                for ch in range(3):
                    prof["trans"][ch] += np.bincount(rbin.ravel(),
                                                     weights=tr[:, ch].sum(0).ravel(), minlength=41)
                mass_chg.append(abs(tr.sum() - gen.sum()) / gen.sum())
                mass_ch_pre.append(np.abs(tr.sum((0, 2, 3)) - gen.sum((0, 2, 3)))
                                   / gen.sum((0, 2, 3)))
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(val)} sims")

    s_truth = np.array(s_truth); s_bind = np.array(s_bind); s_ceil = np.array(s_ceil)
    s_tr = np.array([np.array(x) for x in s_tr])             # (seeds, sims, k)
    for kk in prof:
        prof[kk] /= np.where(rcnt > 0, rcnt, 1)[None]

    def atk(a, q):
        return a.mean(0)[np.argmin(np.abs(kref - q))] if a.ndim == 2 else a[np.argmin(np.abs(kref - q))]

    print("\n=== GOAL 1/4: mass conservation (transplant vs BIND) ===")
    print(f"  total per-patch mass change: {np.mean(mass_chg)*100:.4f}%  "
          f"per-channel: {np.mean(mass_ch_pre, 0)*100} %  (=> integrated mass + its theta-dep preserved)")

    print("\n=== GOAL 5: composite S(k) closure (mean over sims; +-std over seeds) ===")
    print(f"{'':>12} {'k=20':>7} {'k=30':>7} {'k=50':>7}")
    for lab, a in [("truth_full", s_truth), ("ceiling", s_ceil), ("BIND", s_bind)]:
        print(f"{lab:>12} {atk(a,20):7.3f} {atk(a,30):7.3f} {atk(a,50):7.3f}")
    tr_mean = s_tr.mean(1)                                   # (seeds, k)
    print(f"{'transplant':>12} " + " ".join(
        f"{tr_mean.mean(0)[np.argmin(abs(kref-q))]:.3f}" for q in [20, 30, 50]))
    for q in [30, 50]:
        gap = atk(s_truth, q) - atk(s_bind, q)
        cl = tr_mean[:, np.argmin(abs(kref - q))] - atk(s_bind, q)
        print(f"  k={q}: closed {100*cl.mean()/gap:+.0f}% +- {100*cl.std()/abs(gap):.0f}% of BIND->truth gap")

    # ---- plots: S(k), core profile (Stars), patch T(k) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, c, l in [(s_truth, "k", "truth"), (s_ceil, "0.5", "ceiling"), (s_bind, "C1", "BIND")]:
        ax[0].plot(kref, a.mean(0), c, label=l, lw=1.6, ls="--" if l == "ceiling" else "-")
    ax[0].plot(kref, tr_mean.mean(0), "C0", lw=2, label="BIND+transplant")
    ax[0].fill_between(kref, tr_mean.min(0), tr_mean.max(0), color="C0", alpha=0.25)
    ax[0].axhline(1, color="gray", ls=":"); ax[0].set_xscale("log"); ax[0].legend()
    ax[0].set_xlabel("k [h/Mpc]"); ax[0].set_ylabel("P/P_DMO"); ax[0].set_title("composite S(k)")
    rr = np.arange(41) * (BOX / NPIX)
    for nm, c in [("truth", "k"), ("BIND", "C1"), ("trans", "C0")]:
        ax[1].plot(rr[:25], prof[nm][2][:25], c, label=nm, lw=1.8)
    ax[1].set_yscale("log"); ax[1].set_xlabel("r [Mpc/h]"); ax[1].set_ylabel("Stars core profile")
    ax[1].set_title("GOAL 2: stellar core profile"); ax[1].legend()
    Tb = np.sqrt(Pp["BIND"] / Pp["truth"]); Tt = np.sqrt(Pp["trans"] / Pp["truth"])
    for ch, nm in [(2, "Stars"), (3, "Total")]:
        ax[2].plot(kcent, Tb[ch], "--", label=f"{nm} BIND")
        ax[2].plot(kcent, Tt[ch], "-", label=f"{nm} transplant")
    ax[2].axhline(1, color="gray", ls=":"); ax[2].set_xscale("log"); ax[2].legend()
    ax[2].set_xlabel("k [h/Mpc]"); ax[2].set_ylabel("T(k)"); ax[2].set_title("GOAL 3: patch power")
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[validate] wrote {out}")


if __name__ == "__main__":
    main()
