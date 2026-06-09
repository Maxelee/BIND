#!/usr/bin/env python
"""Diagnose BIND's high-k power deficit: BIND vs truth patches, per channel.

The "hydro-replaced" control proved the total-matter high-k deficit lives in the
*content* of BIND's painted patches, and the truth channel decomposition pinned
the high-k excess to stellar condensation (point-like cores).  The decision
"post-hoc fix vs retrain" hinges on ONE measurement on matched patches:

  * transfer function   T_c(k)  = sqrt(<|F_BIND|^2> / <|F_truth|^2>)   (amplitude deficit)
  * coherence           r_c(k)  = <Re F_BIND F_truth*> / sqrt(<|F_BIND|^2><|F_truth|^2>)

If r_c(k) stays high (>~0.9) into k~50 with T_c(k)<1, BIND puts the cores in the
right place but too weak -> a deterministic per-channel Fourier boost (divide out
T_c) recovers BOTH P(k) and the field; post-hoc wins.  If r_c(k) falls at high-k,
BIND isn't generating coherent cores -> amplitude boost just amplifies decorrelated
noise -> retrain (spectral/adversarial) or a learned refiner.

Also dumps the per-channel pixel PDF (gen vs truth) to expose any peak-value deficit
a linear spectral boost wouldn't fix (histogram matching territory).

Runs entirely from cached generated_halos.npz + full_maps.npz + halo_catalog.npz.

Run:  python bind_vs_truth_patches.py --model fm_thermo_ema --n_sims 20
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from kernel_bcm_feasibility import BOX, NPIX, _periodic_cutout

PATCH_PIX = 128
PATCH_BOX = BOX / NPIX * PATCH_PIX  # 6.25 Mpc/h
CHANNELS = ["DM_hydro", "Gas", "Stars"]


def radial_bins(npix, box, nbins=22):
    kf = np.fft.fftfreq(npix) * npix * (2 * np.pi / box)  # h/Mpc
    kx, ky = np.meshgrid(kf, kf, indexing="ij")
    kk = np.sqrt(kx ** 2 + ky ** 2)
    kmin = 2 * np.pi / box
    kmax = kk.max()
    edges = np.geomspace(kmin, kmax, nbins + 1)
    idx = np.clip(np.digitize(kk.ravel(), edges) - 1, 0, nbins - 1)
    cent = np.sqrt(edges[:-1] * edges[1:])
    return idx, cent, nbins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--model", default="fm_thermo_ema")
    ap.add_argument("--n_sims", type=int, default=20)
    ap.add_argument("--out", default="ceph/fm_diag/bind_vs_truth_patches.png")
    args = ap.parse_args()

    sims = sorted(glob.glob(f"{args.suite_root}/Test/sim_SB35_*/snap_090"))[:args.n_sims]
    print(f"[patch-diag] model={args.model}  {len(sims)} sims")

    idx, kcent, nb = radial_bins(PATCH_PIX, PATCH_BOX)
    # per-channel (+ total) accumulators of stacked spectra
    keys = CHANNELS + ["Total"]
    SA = {k: np.zeros(nb) for k in keys}   # sum |F_BIND|^2
    SB = {k: np.zeros(nb) for k in keys}   # sum |F_truth|^2
    SX = {k: np.zeros(nb) for k in keys}   # sum Re F_BIND F_truth*
    # pixel-value samples for PDFs (subsample to keep memory sane)
    pdf_bind = {k: [] for k in CHANNELS}
    pdf_truth = {k: [] for k in CHANNELS}

    pix_per_mpc = NPIX / BOX
    n_halos = 0
    for s in sims:
        s = Path(s)
        gpath = s / "mass_threshold_1p000e13" / args.model / "generated_halos.npz"
        if not gpath.exists():
            print(f"  skip {s.parent.name}: no {args.model} patches")
            continue
        gen = np.load(gpath)["generated"][:, :3].astype(np.float64)  # (N,3,128,128)
        truth_maps = np.load(s / "full_maps.npz")["truth_maps"].astype(np.float64)
        centers = np.load(s / "mass_threshold_1p000e13" / "halo_catalog.npz")["centers"]

        for i, c in enumerate(centers):
            cx = int(c[0] * pix_per_mpc) % NPIX
            cy = int(c[1] * pix_per_mpc) % NPIX
            gen_tot = np.zeros((PATCH_PIX, PATCH_PIX))
            tru_tot = np.zeros((PATCH_PIX, PATCH_PIX))
            for ch in range(3):
                a = gen[i, ch]
                b = _periodic_cutout(truth_maps[ch], cx, cy, PATCH_PIX)
                gen_tot += a; tru_tot += b
                Fa = np.fft.fft2(a - a.mean()); Fb = np.fft.fft2(b - b.mean())
                pa = (np.abs(Fa) ** 2).ravel(); pb = (np.abs(Fb) ** 2).ravel()
                px = np.real(Fa * np.conj(Fb)).ravel()
                name = CHANNELS[ch]
                SA[name] += np.bincount(idx, weights=pa, minlength=nb)
                SB[name] += np.bincount(idx, weights=pb, minlength=nb)
                SX[name] += np.bincount(idx, weights=px, minlength=nb)
                if i % 3 == 0:  # subsample pixels for PDF
                    pdf_bind[name].append(a.ravel()[::8])
                    pdf_truth[name].append(b.ravel()[::8])
            Fa = np.fft.fft2(gen_tot - gen_tot.mean()); Fb = np.fft.fft2(tru_tot - tru_tot.mean())
            SA["Total"] += np.bincount(idx, weights=(np.abs(Fa) ** 2).ravel(), minlength=nb)
            SB["Total"] += np.bincount(idx, weights=(np.abs(Fb) ** 2).ravel(), minlength=nb)
            SX["Total"] += np.bincount(idx, weights=np.real(Fa * np.conj(Fb)).ravel(), minlength=nb)
            n_halos += 1

    print(f"[patch-diag] stacked {n_halos} halos")
    T = {k: np.sqrt(SA[k] / np.where(SB[k] > 0, SB[k], 1)) for k in keys}
    r = {k: SX[k] / np.where(np.sqrt(SA[k] * SB[k]) > 0, np.sqrt(SA[k] * SB[k]), 1) for k in keys}

    def at(arr, kq):
        return arr[np.argmin(np.abs(kcent - kq))]

    print(f"\n{'channel':>9} | {'T(k=10)':>8} {'T(k=30)':>8} {'T(k=50)':>8} | "
          f"{'r(k=10)':>8} {'r(k=30)':>8} {'r(k=50)':>8}")
    for k in keys:
        print(f"{k:>9} | {at(T[k],10):8.3f} {at(T[k],30):8.3f} {at(T[k],50):8.3f} | "
              f"{at(r[k],10):8.3f} {at(r[k],30):8.3f} {at(r[k],50):8.3f}")

    print("\nVERDICT:")
    rt_hi = at(r["Total"], 50)
    if rt_hi > 0.9:
        print(f"  total coherence r(k=50)={rt_hi:.3f} HIGH -> deterministic per-channel"
              f" Fourier boost (divide T_c) should recover power AND field. POST-HOC.")
    elif rt_hi > 0.7:
        print(f"  total coherence r(k=50)={rt_hi:.3f} MODERATE -> partial post-hoc; a learned"
              f" refiner likely needed for full recovery.")
    else:
        print(f"  total coherence r(k=50)={rt_hi:.3f} LOW -> boosting amplifies decorrelated"
              f" structure; needs retrain (spectral/adversarial) or refiner.")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for k, col in zip(keys, ["C0", "C1", "C2", "k"]):
        ax[0, 0].plot(kcent, T[k], col, label=k)
        ax[0, 1].plot(kcent, r[k], col, label=k)
    ax[0, 0].axhline(1, color="gray", ls=":"); ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("k [h/Mpc]"); ax[0, 0].set_ylabel("T(k)=sqrt(P_BIND/P_truth)")
    ax[0, 0].set_title("transfer function (1 = matches truth power)"); ax[0, 0].legend()
    ax[0, 1].axhline(1, color="gray", ls=":"); ax[0, 1].axhline(0.9, color="r", ls=":", lw=0.7)
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("k [h/Mpc]"); ax[0, 1].set_ylabel("coherence r(k)")
    ax[0, 1].set_title("coherence (>0.9 -> post-hoc boost recovers field)")
    ax[0, 1].set_ylim(0, 1.05); ax[0, 1].legend()
    # PDFs (log10(1+x)) -- Stars (the high-k driver) and DM_hydro (secondary)
    for j, ch in enumerate(["Stars", "DM_hydro"]):
        axx = ax[1, j]
        b = np.log10(1 + np.concatenate(pdf_bind[ch]).clip(0))
        t = np.log10(1 + np.concatenate(pdf_truth[ch]).clip(0))
        bins = np.linspace(0, max(b.max(), t.max()), 80)
        axx.hist(t, bins=bins, histtype="step", color="k", label="truth", density=True, log=True)
        axx.hist(b, bins=bins, histtype="step", color="C3", label="BIND", density=True, log=True)
        axx.set_xlabel(f"log10(1+{ch})"); axx.set_title(f"{ch} pixel PDF (peak deficit?)")
        axx.legend()
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[patch-diag] wrote {out}")


if __name__ == "__main__":
    main()
