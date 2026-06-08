#!/usr/bin/env python
"""Pivot: per-halo kernel for the GAS field, head-to-head vs M2 (f_b*smooth(DMO)).

Phase 2 showed the kernel is redundant for *total-matter* S(k) (BIND's 6.25 Mpc/h
patches already carry the whole correction).  The kernel's remaining niche is the
GAS field: total matter is DM-dominated so the DMO background is already right, but
the inter-halo *gas* background is non-zero and the patches leave it at 0.  M2
(f_b*gaussian_filter(DMO, sigma)) fills it train-free and nails the diffuse,
DMO-tracing part (corr~1.0, logRMSE~0.11).  But M2 has NO feedback information --
it cannot know about AGN-ejected gas piled up at 1-5 Mpc/h, which does NOT trace DMO.

This script tests whether a per-halo kernel fit to the M2 *residual*
    Delta_gas(R) = gas(R) - f_b*smooth(DMO)(R)
(i) has coherent, mass/feedback-dependent halo-centric structure (feasibility), and
(ii) reduces the inter-halo error when added on top of M2 (M2 vs M2+kernel vs truth).

Amplitude can be the mass-bin mean or the per-halo in-aperture residual scalar
(``--amplitude {mass,patch}``, the deployable flow-driven analog).  CPU, from cache.

Run:  python kernel_bcm_gas.py --n_test 8 --amplitude patch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from kernel_bcm_feasibility import BOX, MPC_PER_PIX, NPIX, _discover, _periodic_cutout

MASS_EDGES = np.array([13.0, 13.3, 13.6, 14.6])
M2_SIGMA_PIX = 2.5  # the validated M2 IGM gas-filtering scale (~120 kpc/h)


def gauss2(R, a1, s1, a2, s2):
    return a1 * np.exp(-R ** 2 / (2 * s1 ** 2)) + a2 * np.exp(-R ** 2 / (2 * s2 ** 2))


def load_gas_sim(snap_dir):
    fm = np.load(snap_dir / "full_maps.npz")
    dmo = fm["dmo_fullbox"].astype(np.float64)
    gas = fm["truth_maps"].astype(np.float64)[1]  # Gas channel
    cat = np.load(snap_dir / "mass_threshold_1p000e13" / "halo_catalog.npz")
    centers = cat["centers"][:, :2].astype(np.float64)
    logm = np.log10(cat["masses"].astype(np.float64))
    r200 = cat["radii"].astype(np.float64) / 1e3
    params = cat["params"][0].astype(np.float64)
    f_b = float(params[6] / params[0])  # OmegaBaryon / Omega0
    m2 = f_b * gaussian_filter(dmo, M2_SIGMA_PIX, mode="wrap")
    return dmo, gas, m2, centers, logm, r200, f_b


def inter_halo_mask(centers, r200, factor=2.0):
    """Boolean mask True OUTSIDE factor*r200 of every halo (the background region)."""
    mask = np.ones((NPIX, NPIX), dtype=bool)
    pix_per_mpc = NPIX / BOX
    yy, xx = np.ogrid[:NPIX, :NPIX]
    for c, r2 in zip(centers, r200):
        cx = int(c[0] * pix_per_mpc) % NPIX
        cy = int(c[1] * pix_per_mpc) % NPIX
        rpix = max(factor * r2 * pix_per_mpc, 2.0)
        dx = np.minimum((xx - cx) % NPIX, (cx - xx) % NPIX)
        dy = np.minimum((yy - cy) % NPIX, (cy - yy) % NPIX)
        mask &= (dx ** 2 + dy ** 2) > rpix ** 2
    return mask


def collect_residual_profiles(snap_dirs, r_edges, cut, aperture_r200):
    half = cut // 2
    yy, xx = np.mgrid[:cut, :cut].astype(np.float64) - half
    rgrid = (np.sqrt(xx ** 2 + yy ** 2) * MPC_PER_PIX).ravel()
    bidx = np.clip(np.digitize(rgrid, r_edges) - 1, 0, len(r_edges) - 2)
    inr = (rgrid >= r_edges[0]) & (rgrid < r_edges[-1])
    counts = np.bincount(bidx[inr], minlength=len(r_edges) - 1).astype(np.float64)
    ppm = NPIX / BOX

    profs, logms, s_ins = [], [], []
    for snap_dir in snap_dirs:
        try:
            _, gas, m2, centers, logm, r200, _ = load_gas_sim(snap_dir)
        except (FileNotFoundError, OSError, KeyError):
            continue
        resid = gas - m2
        for c, lm, r2 in zip(centers, logm, r200):
            cx = int(c[0] * ppm) % NPIX
            cy = int(c[1] * ppm) % NPIX
            patch = _periodic_cutout(resid, cx, cy, cut).ravel()
            sums = np.bincount(bidx[inr], weights=patch[inr], minlength=len(r_edges) - 1)
            profs.append(sums / np.where(counts > 0, counts, 1.0))
            logms.append(lm)
            s_ins.append(patch[rgrid < aperture_r200 * r2].sum())
    return np.array(profs), np.array(logms), np.array(s_ins), counts


def fit_gas_kernel(profs, logms, r_cent, area_w):
    sigma = 1.0 / np.sqrt(np.maximum(area_w, 1.0))
    bin_centers, params = [], []
    for lo, hi in zip(MASS_EDGES[:-1], MASS_EDGES[1:]):
        sel = (logms >= lo) & (logms < hi)
        if sel.sum() < 30:
            continue
        m = profs[sel].mean(0)
        sc = np.abs(m).max()
        popt = None
        for p0 in ([m[0], 0.2, -0.1 * sc, 1.0], [sc, 0.3, sc, 1.5], [-sc, 0.2, sc, 1.0]):
            try:
                popt, _ = curve_fit(gauss2, r_cent, m, p0=p0, sigma=sigma,
                                    maxfev=40000,
                                    bounds=([-1e4 * sc, 0.05, -1e4 * sc, 0.05],
                                            [1e4 * sc, 4.0, 1e4 * sc, 4.0]))
                break
            except RuntimeError:
                continue
        if popt is None:
            continue
        bin_centers.append(0.5 * (lo + hi))
        params.append(popt)
    bin_centers = np.array(bin_centers)
    params = np.array(params)

    def kp(logm):
        return np.array([np.interp(logm, bin_centers, params[:, j]) for j in range(4)])

    return kp, bin_centers, params


def kernel_field(centers, logm, r200, s_in, kp, sin_ref, rmax, amplitude):
    ky = np.fft.fftfreq(NPIX, d=1.0 / NPIX)
    R = np.sqrt(ky[:, None] ** 2 + ky[None, :] ** 2) * MPC_PER_PIX
    ppm = NPIX / BOX
    out = np.zeros((NPIX, NPIX), dtype=np.float64)
    for lo, hi in zip(MASS_EDGES[:-1], MASS_EDGES[1:]):
        sel = (logm >= lo) & (logm < hi)
        if sel.sum() == 0:
            continue
        mid = 0.5 * (lo + hi)
        K = gauss2(R, *kp(mid)); K[R > rmax] = 0.0
        src = np.zeros((NPIX, NPIX), dtype=np.float64)
        for i in np.where(sel)[0]:
            cx = int(centers[i, 0] * ppm) % NPIX
            cy = int(centers[i, 1] * ppm) % NPIX
            w = (s_in[i] / sin_ref[mid] if amplitude == "patch" and sin_ref.get(mid) else 1.0)
            src[cx, cy] += w
        out += np.fft.irfft2(np.fft.rfft2(src) * np.fft.rfft2(K), s=(NPIX, NPIX))
    return out


def patch_scalars(resid, centers, r200, cut):
    """Per-halo in-aperture (R < 2*r200) integral of the residual field."""
    half = cut // 2
    yy, xx = np.mgrid[:cut, :cut].astype(np.float64) - half
    rg = (np.sqrt(xx ** 2 + yy ** 2) * MPC_PER_PIX).ravel()
    ppm = NPIX / BOX
    out = []
    for c, r2 in zip(centers, r200):
        patch = _periodic_cutout(resid, int(c[0] * ppm) % NPIX,
                                 int(c[1] * ppm) % NPIX, cut).ravel()
        out.append(patch[rg < 2 * r2].sum())
    return np.array(out)


def log_rmse(pred, truth, mask, floor):
    p = np.log10(np.clip(pred[mask], floor, None))
    t = np.log10(np.clip(truth[mask], floor, None))
    return float(np.sqrt(np.mean((p - t) ** 2))), float(np.corrcoef(p, t)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--n_cv", type=int, default=27)
    ap.add_argument("--n_train_sb35", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=8)
    ap.add_argument("--rmax_mpc", type=float, default=6.0)
    ap.add_argument("--amplitude", choices=["mass", "patch"], default="patch")
    ap.add_argument("--out", default="ceph/fm_diag/kernel_gas.png")
    args = ap.parse_args()

    root = Path(args.suite_root)
    sb35 = _discover(root, "Test", None)
    train = _discover(root, "CV", args.n_cv) + sb35[:args.n_train_sb35]
    test = sb35[args.n_train_sb35:args.n_train_sb35 + args.n_test]
    print(f"[gas] train {len(train)} | test {len(test)} | amplitude={args.amplitude}")

    r_edges = np.geomspace(2 * MPC_PER_PIX, 8.0, 22)
    r_cent = np.sqrt(r_edges[:-1] * r_edges[1:])
    cut = int(round(2 * 10.0 / MPC_PER_PIX)); cut += cut % 2

    profs, logms, s_ins, area_w = collect_residual_profiles(train, r_edges, cut, 2.0)
    print(f"[gas] {len(logms)} training halos")

    # feasibility: is the M2 residual coherently halo-centric?
    print("\n=== M2-residual Delta_gas(R) stacked by mass ===")
    stacks = []
    for lo, hi in zip(MASS_EDGES[:-1], MASS_EDGES[1:]):
        sel = (logms >= lo) & (logms < hi)
        if sel.sum() < 30:
            continue
        m = profs[sel].mean(0)
        # signal-to-scatter: |mean profile| vs per-halo scatter
        s2n = np.abs(m).max() / (profs[sel].std(0).max() / np.sqrt(sel.sum()) + 1e-30)
        stacks.append((lo, hi, sel.sum(), m))
        print(f"  logM[{lo},{hi}) n={sel.sum():4d}: center={m[0]:+.3e} "
              f"peak|Delta|={np.abs(m).max():.3e}@R={r_cent[np.argmax(np.abs(m))]:.2f} "
              f"S/N={s2n:.1f}")

    kp, bin_centers, kparams = fit_gas_kernel(profs, logms, r_cent, area_w)
    sin_ref = {}
    for lo, hi in zip(MASS_EDGES[:-1], MASS_EDGES[1:]):
        sel = (logms >= lo) & (logms < hi)
        if sel.sum() >= 30:
            sin_ref[0.5 * (lo + hi)] = float(s_ins[sel].mean())

    # head-to-head in the inter-halo region
    print(f"\n=== inter-halo (outside 2*r200) gas error: M2 vs M2+kernel ===")
    print(f"{'sim':>16} | {'M2 logRMSE':>11} {'+ker logRMSE':>12} | {'M2 corr':>8} {'+ker corr':>9}")
    res = []
    for snap_dir in test:
        dmo, gas, m2, centers, logm, r200, f_b = load_gas_sim(snap_dir)
        floor = 1e-3 * gas.mean()
        mask = inter_halo_mask(centers, r200, factor=2.0)
        s_in = (patch_scalars(gas - m2, centers, r200, cut)
                if args.amplitude == "patch" else None)
        kf = kernel_field(centers, logm, r200, s_in, kp, sin_ref,
                          args.rmax_mpc, args.amplitude)
        m2_rmse, m2_corr = log_rmse(m2, gas, mask, floor)
        mk_rmse, mk_corr = log_rmse(m2 + kf, gas, mask, floor)
        res.append((m2_rmse, mk_rmse, m2_corr, mk_corr))
        print(f"{snap_dir.parent.name:>16} | {m2_rmse:11.4f} {mk_rmse:12.4f} | "
              f"{m2_corr:8.4f} {mk_corr:9.4f}")
    res = np.array(res)
    print(f"\n  mean: M2 logRMSE={res[:,0].mean():.4f}  M2+kernel={res[:,1].mean():.4f}  "
          f"(improvement {100*(1-res[:,1].mean()/res[:,0].mean()):+.1f}%)")
    print(f"        M2 corr  ={res[:,2].mean():.4f}  M2+kernel={res[:,3].mean():.4f}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(stacks)))
    for (lo, hi, n, m), col in zip(stacks, cmap):
        ax[0].plot(r_cent, m, "o", ms=3, color=col, alpha=0.6)
        rr = np.geomspace(r_cent[0], r_cent[-1], 200)
        ax[0].plot(rr, gauss2(rr, *kp(0.5 * (lo + hi))), "-", color=col,
                   label=f"logM~{0.5*(lo+hi):.2f} (n={n})")
    ax[0].axhline(0, color="k", lw=0.7); ax[0].set_xscale("log")
    ax[0].set_xlabel("R [Mpc/h]"); ax[0].set_ylabel("gas - f_b*smooth(DMO) per pixel")
    ax[0].set_title("M2 residual (what M2 misses); lines=kernel fit"); ax[0].legend(fontsize=7)
    ax[1].bar([0, 1], [res[:, 0].mean(), res[:, 1].mean()], color=["C0", "C3"])
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["M2", "M2+kernel"])
    ax[1].set_ylabel("inter-halo gas logRMSE")
    ax[1].set_title(f"head-to-head ({args.amplitude} amp, {len(test)} held-out sims)")
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[gas] wrote {out}")


if __name__ == "__main__":
    main()
