#!/usr/bin/env python
"""Post-hoc fix for BIND's high-k P(k) deficit: per-channel spectral boost.

bind_vs_truth_patches.py showed the P(k) deficit is an AMPLITUDE problem
(per-channel transfer function T_c(k) < 1, dominated by Stars ~0.75 at all k),
NOT a coherence problem (low high-k coherence is intrinsic DMO->baryon
stochasticity and doesn't affect P(k)).  So dividing each generated patch
channel by T_c(k) in Fourier space should restore the power spectrum with no
retrain.  T_c is fit on a TRAIN set of sims and applied to HELD-OUT sims (no
leakage); we then rebuild the composite and compare total-matter S(k) to truth
and to the uncorrected BIND composite (the cached one = the paper plot).

Run:  python bind_highk_fix.py --model fm_thermo_ema --n_train 30 --n_test 6
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from kernel_bcm_feasibility import BOX, NPIX, _periodic_cutout
from bind_vs_truth_patches import PATCH_PIX, PATCH_BOX, radial_bins
from bind.inference.pipeline import build_bind_composite
from bind.inference.artifacts import load_halo_cutouts
from bind.metrics import power_spectrum_pylians_2d


def fit_transfer(train_sims, model, idx, kcent, nb):
    """Per-channel radial transfer function T_c(k) = sqrt(<P_BIND>/<P_truth>)."""
    SA = np.zeros((3, nb)); SB = np.zeros((3, nb))
    ppm = NPIX / BOX
    for s in train_sims:
        s = Path(s)
        gp = s / "mass_threshold_1p000e13" / model / "generated_halos.npz"
        if not gp.exists():
            continue
        gen = np.load(gp)["generated"][:, :3].astype(np.float64)
        truth = np.load(s / "full_maps.npz")["truth_maps"].astype(np.float64)
        centers = np.load(s / "mass_threshold_1p000e13" / "halo_catalog.npz")["centers"]
        for i, c in enumerate(centers):
            cx = int(c[0] * ppm) % NPIX; cy = int(c[1] * ppm) % NPIX
            for ch in range(3):
                a = gen[i, ch]; b = _periodic_cutout(truth[ch], cx, cy, PATCH_PIX)
                Fa = np.fft.fft2(a - a.mean()); Fb = np.fft.fft2(b - b.mean())
                SA[ch] += np.bincount(idx, weights=(np.abs(Fa) ** 2).ravel(), minlength=nb)
                SB[ch] += np.bincount(idx, weights=(np.abs(Fb) ** 2).ravel(), minlength=nb)
    return np.sqrt(SA / np.where(SB > 0, SB, 1))  # (3, nb)


def boost_maps(Tc, kcent, tmin=0.5):
    """2D per-channel boost = 1/T_c(|k|), clamped; DC and T>=1 left unboosted."""
    kf = np.fft.fftfreq(PATCH_PIX) * PATCH_PIX * (2 * np.pi / PATCH_BOX)
    kk = np.sqrt(kf[:, None] ** 2 + kf[None, :] ** 2)
    maps = []
    for ch in range(3):
        t = np.interp(kk.ravel(), kcent, Tc[ch],
                      left=Tc[ch][0], right=Tc[ch][-1]).reshape(kk.shape)
        t = np.clip(t, tmin, 1.0)          # never down-weight (cap boost at 1/tmin)
        bm = 1.0 / t
        bm[0, 0] = 1.0                      # preserve DC -> per-patch mass
        maps.append(bm)
    return maps


def apply_boost(gen3, bmaps):
    """Spectral boost per channel, preserving each channel's total mass so the
    downstream patch_mass_match is a no-op (clipping boost-induced negatives
    would otherwise raise the mass and make mass_match scale the patch down)."""
    out = np.empty_like(gen3)
    for ch in range(3):
        s0 = gen3[ch].sum()
        x = np.clip(np.real(np.fft.ifft2(np.fft.fft2(gen3[ch]) * bmaps[ch])), 0.0, None)
        s1 = x.sum()
        out[ch] = x * (s0 / s1) if s1 > 0 else x
    return out


def sk(total, dmo):
    k, p, _ = power_spectrum_pylians_2d(total, box_size=BOX, MAS="None")
    _, pd, _ = power_spectrum_pylians_2d(dmo, box_size=BOX, MAS="None")
    return k, p / pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--model", default="fm_thermo_ema")
    ap.add_argument("--n_train", type=int, default=30)
    ap.add_argument("--n_test", type=int, default=6)
    ap.add_argument("--out", default="ceph/fm_diag/bind_highk_fix.png")
    args = ap.parse_args()

    sims = sorted(glob.glob(f"{args.suite_root}/Test/sim_SB35_*/snap_090"))
    train, test = sims[:args.n_train], sims[args.n_train:args.n_train + args.n_test]
    idx, kcent, nb = radial_bins(PATCH_PIX, PATCH_BOX)
    print(f"[fix] fitting T_c on {len(train)} sims ...")
    Tc = fit_transfer(train, args.model, idx, kcent, nb)
    bmaps = boost_maps(Tc, kcent)
    print(f"[fix] stellar boost 1/T at k=10/30/50: "
          f"{1/np.clip(np.interp([10,30,50],kcent,Tc[2]),0.5,1)}")

    s_truth, s_bind, s_fix = [], [], []
    kref = None
    for s in test:
        s = Path(s)
        mt = s / "mass_threshold_1p000e13"
        gen = np.load(mt / args.model / "generated_halos.npz")["generated"].astype(np.float64)
        comp_cached = np.load(mt / args.model / "composite.npz")["composite"].astype(np.float64)
        fm = np.load(s / "full_maps.npz")
        dmo = fm["dmo_fullbox"].astype(np.float64)
        truth_total = fm["truth_maps"].astype(np.float64).sum(0)
        cat = np.load(mt / "halo_catalog.npz")
        cutouts = load_halo_cutouts(mt / "halo_cutouts.npz")
        halos = [{"halo_center": cat["centers"][i, :2], "halo_mass": float(cat["masses"][i]),
                  "r200": float(cat["radii"][i]) / 1e3, "params": cat["params"][i]}
                 for i in range(len(cat["masses"]))]

        boosted = np.stack([apply_boost(gen[i, :3], bmaps) for i in range(gen.shape[0])])
        comp_fix = build_bind_composite(
            dmo, halos, boosted, cutouts, box_size=BOX, npix=NPIX, patch_pix=PATCH_PIX,
            patch_mass_match=True, taper_frac=0.15, r200_factor=0.0)["composite"]

        k, st = sk(truth_total, dmo)
        _, sb = sk(comp_cached.sum(0), dmo)
        _, sf = sk(comp_fix.sum(0), dmo)
        kref = k
        s_truth.append(st); s_bind.append(sb); s_fix.append(sf)
        print(f"  {s.parent.name}: done")

    s_truth = np.array(s_truth); s_bind = np.array(s_bind); s_fix = np.array(s_fix)

    def at(a, kq):
        return a.mean(0)[np.argmin(np.abs(kref - kq))]
    print(f"\n{'':>10} {'k=10':>7} {'k=30':>7} {'k=50':>7}")
    for lab, a in [("truth", s_truth), ("BIND", s_bind), ("BIND+fix", s_fix)]:
        print(f"{lab:>10} {at(a,10):7.3f} {at(a,30):7.3f} {at(a,50):7.3f}")
    print(f"\nP/Ptruth at k=50:  BIND={ (s_bind/s_truth).mean(0)[np.argmin(abs(kref-50))]:.3f}"
          f"  BIND+fix={ (s_fix/s_truth).mean(0)[np.argmin(abs(kref-50))]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, col, lab in [(s_truth, "k", "Truth/DMO"), (s_bind, "C1", "BIND/DMO"),
                        (s_fix, "C0", "BIND+fix/DMO")]:
        ax[0].plot(kref, a.mean(0), col, label=lab, lw=1.8)
    ax[0].axhline(1, color="gray", ls=":"); ax[0].set_xscale("log")
    ax[0].set_xlabel("k [h/Mpc]"); ax[0].set_ylabel("P/P_DMO"); ax[0].legend()
    ax[0].set_title(f"total-matter suppression ({len(test)} held-out sims)")
    for a, col, lab in [(s_bind, "C1", "BIND/Truth"), (s_fix, "C0", "BIND+fix/Truth")]:
        ax[1].plot(kref, (a / s_truth).mean(0), col, label=lab)
    ax[1].axhline(1, color="k", lw=0.7); ax[1].fill_between(kref, 0.95, 1.05, color="g", alpha=0.1)
    ax[1].set_xscale("log"); ax[1].set_xlabel("k [h/Mpc]"); ax[1].set_ylabel("P/P_truth")
    ax[1].set_ylim(0.8, 1.15); ax[1].legend(); ax[1].set_title("closure to truth")
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[fix] wrote {out}")


if __name__ == "__main__":
    main()
