#!/usr/bin/env python
"""Phase-1 BCM-with-flow-amplitudes outpainting: fit the kernel + validate S(k).

Phase 0 (kernel_bcm_feasibility.py) established that, for TOTAL matter, the
per-halo baryonic correction delta_b = rho_hydro - rho_DMO is a central
*suppression* plus a compensating *positive overshoot* at intermediate R
(mass-conserving), and that an in-aperture scalar predicts the far field.

This script:
  1. FIT a mass-conserving difference-of-Gaussians (DoG) kernel
       K(R) = a_w * exp(-R^2/2 sw^2) - a_n * exp(-R^2/2 sn^2),   sn < sw
     to the stacked truth delta_b(R) profiles in mass bins, giving smooth
     a_w, sw, a_n, sn as functions of log M (theta dependence is a later
     refinement -- here we marginalize over the Sobol theta).
  2. RECONSTRUCT each held-out test sim's delta_b field by FFT halo-stacking
     (one periodic convolution per mass bin -- O(N_bins * N_pix log N_pix)).
  3. VALIDATE the total-matter suppression S(k) = P(DMO+delta_b)/P(DMO)
     against the truth S(k) = P(rho_hydro)/P(DMO), plus field-level fidelity.

Amplitude modes (``--amplitude``):
  * ``mass``  -- every halo in a mass bin uses that bin's mean kernel (the
                 ceiling test: can the kernel FAMILY + halo positions
                 reproduce S(k)?).
  * ``patch`` -- per-halo amplitude scaled by the in-aperture scalar
                 s_in,i / <s_in>_bin (the deployable, flow-driven version;
                 here s_in is taken from the TRUTH patch as a stand-in for the
                 flow output, which the model is trained to reproduce).

Runs on CPU from cached full_maps.npz + halo_catalog.npz only.

Run:  python kernel_bcm_build.py --n_train_sb35 40 --n_test 8 --amplitude mass
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from kernel_bcm_feasibility import (
    BOX, MPC_PER_PIX, NPIX, _discover, _load_sim, _periodic_cutout,
)
from bind.metrics import power_spectrum_pylians_2d

MASS_EDGES = np.array([13.0, 13.15, 13.3, 13.45, 13.6, 13.8, 14.0, 14.6])


def dog(R, a_w, sw, a_n, sn):
    """Difference of Gaussians: broad positive minus narrow negative."""
    return a_w * np.exp(-R ** 2 / (2 * sw ** 2)) - a_n * np.exp(-R ** 2 / (2 * sn ** 2))


def dog_conserving(R, a_w, sw, sn):
    """Mass-conserving DoG: the narrow (suppression) amplitude is pinned by
    2D mass conservation a_n*sn^2 = a_w*sw^2, so the net integral is exactly 0
    and the central value a_w*(1-(sw/sn)^2) < 0 (suppression) whenever sw > sn.
    Three free params removes the a_w<->a_n degeneracy of the free 4-param fit.
    """
    a_n = a_w * (sw / sn) ** 2
    return dog(R, a_w, sw, a_n, sn)


# ----------------------------------------------------------------------------
# Pass 1: collect stacked profiles + per-halo scalars from training sims
# ----------------------------------------------------------------------------
def collect_profiles(snap_dirs, r_edges, cut, aperture_r200):
    half = cut // 2
    yy, xx = np.mgrid[:cut, :cut].astype(np.float64) - half
    rgrid = (np.sqrt(xx ** 2 + yy ** 2) * MPC_PER_PIX).ravel()
    bin_idx = np.clip(np.digitize(rgrid, r_edges) - 1, 0, len(r_edges) - 2)
    in_range = (rgrid >= r_edges[0]) & (rgrid < r_edges[-1])
    counts = np.bincount(bin_idx[in_range], minlength=len(r_edges) - 1).astype(np.float64)
    pix_per_mpc = NPIX / BOX

    profs, logms, s_ins = [], [], []
    for snap_dir in snap_dirs:
        try:
            delta_b, centers, logm, r200 = _load_sim(snap_dir)
        except (FileNotFoundError, OSError, KeyError):
            continue
        for c, lm, r2 in zip(centers, logm, r200):
            cx = int(c[0] * pix_per_mpc) % NPIX
            cy = int(c[1] * pix_per_mpc) % NPIX
            patch = _periodic_cutout(delta_b, cx, cy, cut).ravel()
            sums = np.bincount(bin_idx[in_range], weights=patch[in_range],
                               minlength=len(r_edges) - 1)
            profs.append(sums / np.where(counts > 0, counts, 1.0))
            logms.append(lm)
            s_ins.append(patch[rgrid < aperture_r200 * r2].sum())
    return np.array(profs), np.array(logms), np.array(s_ins), counts


# ----------------------------------------------------------------------------
# Fit DoG per mass bin -> smooth params(logM)
# ----------------------------------------------------------------------------
def fit_kernel(profs, logms, r_cent, area_w):
    """Fit the mass-conserving DoG per mass bin, area-weighted (annulus pixel
    counts) so the well-sampled outer profile drives the fit instead of the
    handful of central pixels.  Returns 4-param (a_w, sw, a_n, sn) rows with
    a_n derived from conservation, for a uniform downstream interface.
    """
    bin_centers, fit_params, bin_counts = [], [], []
    sigma = 1.0 / np.sqrt(np.maximum(area_w, 1.0))  # weight ~ sqrt(pixels)
    for lo, hi in zip(MASS_EDGES[:-1], MASS_EDGES[1:]):
        sel = (logms >= lo) & (logms < hi)
        if sel.sum() < 30:
            continue
        m = profs[sel].mean(0)
        scale = np.abs(m).max()
        bounds = ([0, 0.15, 0.04], [1e4 * scale, 4.0, 0.5])
        popt = None
        for p0 in ([scale * 0.3, 0.6, 0.12], [scale * 0.1, 0.3, 0.08],
                   [scale, 1.0, 0.15]):  # retry sparse/noisy (high-mass) bins
            try:
                popt, _ = curve_fit(
                    dog_conserving, r_cent, m, p0=p0, sigma=sigma,
                    absolute_sigma=False, maxfev=60000, bounds=bounds)
                break
            except RuntimeError:
                continue
        if popt is None:
            print(f"  WARN: DoG fit failed for logM~{0.5*(lo+hi):.2f} (n={sel.sum()})")
            continue
        a_w, sw, sn = popt
        a_n = a_w * (sw / sn) ** 2
        bin_centers.append(0.5 * (lo + hi))
        fit_params.append([a_w, sw, a_n, sn])
        bin_counts.append(int(sel.sum()))
    bin_centers = np.array(bin_centers)
    fit_params = np.array(fit_params)  # (Nbin, 4): a_w, sw, a_n, sn

    def kernel_params(logm):
        """Interpolate (a_w, sw, a_n, sn) at given log M, clamped at bin ends."""
        return np.array([np.interp(logm, bin_centers, fit_params[:, j]) for j in range(4)])

    return kernel_params, bin_centers, fit_params, bin_counts


# ----------------------------------------------------------------------------
# Pass 2: reconstruct delta_b for one sim via FFT halo-stacking
# ----------------------------------------------------------------------------
def _radial_kernel_stamp(params, rmax_mpc):
    """2D kernel centred at pixel (0,0) (fft-convention), truncated at rmax."""
    ky = np.fft.fftfreq(NPIX, d=1.0 / NPIX)
    R = np.sqrt(ky[:, None] ** 2 + ky[None, :] ** 2) * MPC_PER_PIX
    K = dog(R, *params)
    K[R > rmax_mpc] = 0.0
    return K.astype(np.float64)


def reconstruct(snap_dir, kernel_params, rmax_mpc, amplitude, sin_ref):
    """Return (delta_model, dmo, truth_total) for a sim."""
    delta_b_truth, centers, logm, r200 = _load_sim(snap_dir)
    fm = np.load(snap_dir / "full_maps.npz")
    dmo = fm["dmo_fullbox"].astype(np.float64)
    truth_total = fm["truth_maps"].astype(np.float64).sum(0)
    pix_per_mpc = NPIX / BOX

    # per-halo in-aperture scalar (for patch-amplitude mode)
    if amplitude == "patch":
        cut = int(round(2 * 12.0 / MPC_PER_PIX)); cut += cut % 2
        half = cut // 2
        yy, xx = np.mgrid[:cut, :cut].astype(np.float64) - half
        rg = (np.sqrt(xx ** 2 + yy ** 2) * MPC_PER_PIX).ravel()
        s_in = np.array([
            _periodic_cutout(delta_b_truth, int(c[0] * pix_per_mpc) % NPIX,
                             int(c[1] * pix_per_mpc) % NPIX, cut).ravel()[rg < 2 * r2].sum()
            for c, r2 in zip(centers, r200)
        ])

    delta_model = np.zeros((NPIX, NPIX), dtype=np.float64)
    for lo, hi in zip(MASS_EDGES[:-1], MASS_EDGES[1:]):
        sel = (logm >= lo) & (logm < hi)
        if sel.sum() == 0:
            continue
        lm_mid = 0.5 * (lo + hi)
        params = kernel_params(lm_mid)
        source = np.zeros((NPIX, NPIX), dtype=np.float64)
        for i in np.where(sel)[0]:
            cx = int(centers[i, 0] * pix_per_mpc) % NPIX
            cy = int(centers[i, 1] * pix_per_mpc) % NPIX
            if amplitude == "patch":
                ref = sin_ref.get(lm_mid, 1.0)
                w = s_in[i] / ref if ref != 0 else 1.0
            else:
                w = 1.0
            source[cx, cy] += w
        K2d = _radial_kernel_stamp(params, rmax_mpc)
        delta_model += np.fft.irfft2(np.fft.rfft2(source) * np.fft.rfft2(K2d), s=(NPIX, NPIX))

    return delta_model, dmo, truth_total, delta_b_truth


def sk(field, dmo):
    k, p, _ = power_spectrum_pylians_2d(field, box_size=BOX, MAS="None")
    k2, pd, _ = power_spectrum_pylians_2d(dmo, box_size=BOX, MAS="None")
    return k, p / pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--n_cv", type=int, default=27)
    ap.add_argument("--n_train_sb35", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=8, help="held-out SB35 sims for S(k)")
    ap.add_argument("--rmax_mpc", type=float, default=8.0)
    ap.add_argument("--amplitude", choices=["mass", "patch"], default="mass")
    ap.add_argument("--out", default="ceph/fm_diag/kernel_build.png")
    args = ap.parse_args()

    root = Path(args.suite_root)
    sb35 = _discover(root, "Test", None)
    train = _discover(root, "CV", args.n_cv) + sb35[:args.n_train_sb35]
    test = sb35[args.n_train_sb35:args.n_train_sb35 + args.n_test]
    print(f"[build] train: {len(train)} sims  | test (held-out SB35): {len(test)} sims")

    r_edges = np.geomspace(2 * MPC_PER_PIX, 10.0, 24)
    r_cent = np.sqrt(r_edges[:-1] * r_edges[1:])
    cut = int(round(2 * 12.0 / MPC_PER_PIX)); cut += cut % 2

    print("[build] collecting training profiles ...")
    profs, logms, s_ins, area_w = collect_profiles(train, r_edges, cut, aperture_r200=2.0)
    print(f"[build] {len(logms)} training halos")

    kernel_params, bin_centers, fit_params, bin_counts = fit_kernel(
        profs, logms, r_cent, area_w)
    print("\n=== DoG kernel fits per mass bin (a_w, sw, a_n, sn) ===")
    for bc, fp, nc in zip(bin_centers, fit_params, bin_counts):
        integ = 2 * np.pi * (fp[0] * fp[1] ** 2 - fp[2] * fp[3] ** 2)
        print(f"  logM~{bc:.2f} (n={nc:4d}): a_w={fp[0]:.2e} sw={fp[1]:.2f}  "
              f"a_n={fp[2]:.2e} sn={fp[3]:.2f}  net_integral={integ:+.2e}")

    # reference in-aperture scalar per mass-bin mid (for patch amplitude)
    sin_ref = {}
    for lo, hi in zip(MASS_EDGES[:-1], MASS_EDGES[1:]):
        sel = (logms >= lo) & (logms < hi)
        if sel.sum() >= 30:
            sin_ref[0.5 * (lo + hi)] = float(s_ins[sel].mean())

    # validate S(k) on held-out sims
    print(f"\n=== S(k) validation (amplitude={args.amplitude}) ===")
    sk_truth, sk_model, field_corr = [], [], []
    k_ref = None
    for snap_dir in test:
        dm, dmo, truth_total, db_truth = reconstruct(
            snap_dir, kernel_params, args.rmax_mpc, args.amplitude, sin_ref)
        total_model = dmo + dm
        k, st = sk(truth_total, dmo)
        _, sm = sk(total_model, dmo)
        k_ref = k
        sk_truth.append(st); sk_model.append(sm)
        # Raw per-pixel corr is dominated by sub-Mpc per-halo structure the mean
        # kernel cannot reproduce; the kernel targets the *large-scale*
        # correction, so also report corr after smoothing both to ~1 Mpc/h.
        from scipy.ndimage import gaussian_filter
        sig_pix = 1.0 / MPC_PER_PIX
        corr = float(np.corrcoef(dm.ravel(), db_truth.ravel())[0, 1])
        cs = float(np.corrcoef(
            gaussian_filter(dm, sig_pix, mode="wrap").ravel(),
            gaussian_filter(db_truth, sig_pix, mode="wrap").ravel())[0, 1])
        field_corr.append(cs)
        print(f"  {snap_dir.parent.name}: field corr raw={corr:+.3f} "
              f"smoothed(1Mpc/h)={cs:+.3f}")

    sk_truth = np.array(sk_truth); sk_model = np.array(sk_model)
    field_corr = np.array(field_corr)
    print(f"\n  mean field-level corr = {field_corr.mean():+.3f}")
    # S(k) agreement: median |model-truth| of the suppression dip
    resid = np.abs(sk_model - sk_truth)
    print(f"  median |S_model - S_truth| over k = {np.median(resid):.4f}")
    print(f"  at k<5 h/Mpc: {np.median(resid[:, k_ref < 5]):.4f}; "
          f"k>20: {np.median(resid[:, k_ref > 20]):.4f}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    # (a) kernel fits
    cmap = plt.cm.viridis(np.linspace(0, 1, len(bin_centers)))
    for lo, hi, col, bc, fp in zip(MASS_EDGES[:-1], MASS_EDGES[1:], cmap,
                                   bin_centers, fit_params):
        sel = (logms >= lo) & (logms < hi)
        if sel.sum() < 30:
            continue
        ax[0, 0].plot(r_cent, profs[sel].mean(0), "o", ms=3, color=col, alpha=0.6)
        rr = np.geomspace(r_cent[0], r_cent[-1], 200)
        ax[0, 0].plot(rr, dog(rr, *fp), "-", color=col, label=f"logM~{bc:.2f}")
    ax[0, 0].axhline(0, color="k", lw=0.7); ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("R [Mpc/h]"); ax[0, 0].set_ylabel("delta_b/pixel")
    ax[0, 0].set_title("DoG kernel fits (points=truth, lines=fit)")
    ax[0, 0].legend(fontsize=7)

    # (b) kernel param trends
    ax[0, 1].plot(bin_centers, fit_params[:, 1], "o-", label="sigma_wide")
    ax[0, 1].plot(bin_centers, fit_params[:, 3], "s-", label="sigma_narrow")
    ax[0, 1].set_xlabel("log M"); ax[0, 1].set_ylabel("sigma [Mpc/h]")
    ax[0, 1].set_title("kernel widths vs mass"); ax[0, 1].legend()

    # (c) S(k) model vs truth
    for st, sm in zip(sk_truth, sk_model):
        ax[1, 0].plot(k_ref, st, color="k", alpha=0.3, lw=1)
        ax[1, 0].plot(k_ref, sm, color="C3", alpha=0.4, lw=1)
    ax[1, 0].plot([], [], "k", label="truth S(k)")
    ax[1, 0].plot([], [], "C3", label=f"BCM model ({args.amplitude})")
    ax[1, 0].axhline(1, color="b", lw=0.6, ls=":")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_xlabel("k [h/Mpc]")
    ax[1, 0].set_ylabel("P/P_DMO"); ax[1, 0].set_title("total-matter suppression")
    ax[1, 0].legend()

    # (d) mean S(k) ratio model/truth
    ax[1, 1].plot(k_ref, (sk_model / sk_truth).mean(0), "C3")
    ax[1, 1].fill_between(k_ref, *np.percentile(sk_model / sk_truth, [16, 84], axis=0),
                          color="C3", alpha=0.2)
    ax[1, 1].axhline(1, color="k", lw=0.7)
    ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("k [h/Mpc]")
    ax[1, 1].set_ylabel("S_model / S_truth")
    ax[1, 1].set_title(f"closure (field corr {field_corr.mean():.2f})")

    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[build] wrote {out}")


if __name__ == "__main__":
    main()
