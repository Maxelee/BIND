#!/usr/bin/env python
"""Phase-0 feasibility gate for the BCM-with-flow-amplitudes outpainting method.

The proposed method models the *baryonic correction field* delta_b = rho_hydro - rho_DMO
(total matter) as a sum of per-halo radial kernels K(r; M, theta) whose amplitudes are
distilled from BIND's painted halo patches and then extrapolated into the inter-halo
region.  Two assumptions must hold before any kernel is worth building:

  (1) SHAPE -- stacked truth profiles delta_b(R) around halos must show the expected
      double-component structure (central baryonic enhancement, intermediate AGN-driven
      suppression at ~1-10 Mpc/h), and that shape must vary smoothly with halo mass.

  (2) THE CRUX -- a *single scalar per halo* measured inside the painted aperture
      (R < f * r200, where BIND already gives the full field) must predict the
      far-field correction amplitude (the integral of delta_b *outside* the aperture,
      which the kernel has to supply).  If a patch-integrated scalar carries no
      information about the far field beyond what halo mass already tells us, the
      "amplitudes come from the flow model" premise fails and the method is dead.

This script answers both from CACHED suite outputs only -- no sim reruns, no flow
inference.  It loads dmo_fullbox + truth_maps + halo_catalog, subtracts at the field
level, stacks radial profiles, and reports the partial correlation (controlling for
log M) between the in-aperture scalar and the far-field amplitude.

Run:  python kernel_bcm_feasibility.py --n_sb35 50 --out ceph/fm_diag/kernel_feasibility.png
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

BOX = 50.0  # Mpc/h, CAMELS L50
NPIX = 1024
MPC_PER_PIX = BOX / NPIX  # ~0.0488 Mpc/h


def _discover(suite_root: Path, suite: str, limit: int | None) -> list[Path]:
    pat = str(suite_root / suite / "sim_*" / "snap_090")
    dirs = sorted(glob.glob(pat))
    if limit is not None:
        dirs = dirs[:limit]
    return [Path(d) for d in dirs]


def _load_sim(snap_dir: Path):
    """Return (delta_b 2D total-matter correction, centers Mpc/h, logM, r200 Mpc/h)."""
    fm = np.load(snap_dir / "full_maps.npz")
    dmo = fm["dmo_fullbox"].astype(np.float64)
    total = fm["truth_maps"].astype(np.float64).sum(0)
    delta_b = total - dmo

    cat_path = snap_dir / "mass_threshold_1p000e13" / "halo_catalog.npz"
    cat = np.load(cat_path)
    centers = cat["centers"][:, :2].astype(np.float64)        # Mpc/h
    masses = cat["masses"].astype(np.float64)
    logm = np.log10(masses)
    r200 = cat["radii"].astype(np.float64) / 1e3              # kpc/h -> Mpc/h
    return delta_b, centers, logm, r200


def _periodic_cutout(field: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    n = field.shape[0]
    half = size // 2
    ix = (cx - half + np.arange(size)) % n
    iy = (cy - half + np.arange(size)) % n
    return field[np.ix_(ix, iy)]


def _partial_spearman(x, y, z):
    """Spearman correlation of x,y after linearly removing z (rank space)."""
    from scipy.stats import rankdata

    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    ex, ey = resid(rx, rz), resid(ry, rz)
    if ex.std() == 0 or ey.std() == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--n_cv", type=int, default=27, help="number of CV sims (fixed theta)")
    ap.add_argument("--n_sb35", type=int, default=50, help="number of SB35 Sobol-theta sims")
    ap.add_argument("--cutout_mpc", type=float, default=12.0,
                    help="half-width of the per-halo cutout (Mpc/h)")
    ap.add_argument("--rmax_mpc", type=float, default=10.0, help="max radius for profiles")
    ap.add_argument("--aperture_r200", type=float, default=2.0,
                    help="in-aperture scalar uses R < f*r200 (the painted region)")
    ap.add_argument("--farfield_mpc", type=float, default=5.0,
                    help="far-field amplitude integrates aperture < R < this (Mpc/h)")
    ap.add_argument("--out", default="ceph/fm_diag/kernel_feasibility.png")
    args = ap.parse_args()

    suite_root = Path(args.suite_root)
    snap_dirs = (_discover(suite_root, "CV", args.n_cv)
                 + _discover(suite_root, "Test", args.n_sb35))
    print(f"[feasibility] {len(snap_dirs)} sims "
          f"({args.n_cv} CV requested + {args.n_sb35} SB35 requested)")

    cut = int(round(2 * args.cutout_mpc / MPC_PER_PIX))   # full cutout side in pixels
    cut += cut % 2
    half = cut // 2
    yy, xx = np.mgrid[:cut, :cut].astype(np.float64) - half
    rgrid = np.sqrt(xx ** 2 + yy ** 2) * MPC_PER_PIX       # Mpc/h

    # log-spaced radial bins for the stacked profile.  Start at ~2 px so the
    # innermost annuli are actually populated (bins thinner than the pixel
    # spacing, ~0.049 Mpc/h, would be empty and read as a spurious 0).
    r_edges = np.geomspace(2 * MPC_PER_PIX, args.rmax_mpc, 24)
    r_cent = np.sqrt(r_edges[:-1] * r_edges[1:])
    bin_idx = np.clip(np.digitize(rgrid.ravel(), r_edges) - 1, 0, len(r_cent) - 1)
    in_range = (rgrid.ravel() >= r_edges[0]) & (rgrid.ravel() < r_edges[-1])
    counts = np.bincount(bin_idx[in_range], minlength=len(r_cent)).astype(np.float64)

    # accumulators
    prof_rows, logm_all = [], []
    s_in_all, f_out_all = [], []
    pixels_per_mpc = NPIX / BOX

    for k, snap_dir in enumerate(snap_dirs):
        try:
            delta_b, centers, logm, r200 = _load_sim(snap_dir)
        except (FileNotFoundError, OSError, KeyError) as e:
            print(f"  skip {snap_dir.parent.name}: {e}")
            continue

        for c, lm, r2 in zip(centers, logm, r200):
            cx = int(c[0] * pixels_per_mpc) % NPIX
            cy = int(c[1] * pixels_per_mpc) % NPIX
            patch = _periodic_cutout(delta_b, cx, cy, cut).ravel()

            # stacked mean profile (mean delta_b per pixel in each annulus)
            sums = np.bincount(bin_idx[in_range], weights=patch[in_range],
                               minlength=len(r_cent))
            prof_rows.append(sums / np.where(counts > 0, counts, 1.0))
            logm_all.append(lm)

            # crux scalars: in-aperture integral vs far-field integral
            rflat = rgrid.ravel()
            ap_r = args.aperture_r200 * r2
            s_in = patch[rflat < ap_r].sum()
            f_out = patch[(rflat >= ap_r) & (rflat < args.farfield_mpc)].sum()
            s_in_all.append(s_in)
            f_out_all.append(f_out)

        if (k + 1) % 10 == 0:
            print(f"  processed {k + 1}/{len(snap_dirs)} sims, "
                  f"{len(logm_all)} halos so far")

    prof = np.array(prof_rows)          # (N_halo, N_r)
    logm_all = np.array(logm_all)
    s_in_all = np.array(s_in_all)
    f_out_all = np.array(f_out_all)
    print(f"[feasibility] total halos: {len(logm_all)}")

    # ---- ASSUMPTION 1: shape vs mass ----
    mass_edges = [13.0, 13.3, 13.7, 14.6]
    print("\n=== ASSUMPTION 1: stacked delta_b(R) shape by mass ===")
    stacks = []
    for lo, hi in zip(mass_edges[:-1], mass_edges[1:]):
        sel = (logm_all >= lo) & (logm_all < hi)
        if sel.sum() == 0:
            stacks.append(None)
            continue
        m = prof[sel].mean(0)
        stacks.append((lo, hi, sel.sum(), m))
        # For TOTAL matter the structure is: central suppression (min < 0) then a
        # compensating positive overshoot (ejected mass) at intermediate R.
        i_min = int(np.argmin(m))
        r_neg = r_cent[i_min]
        overshoot = float(m[i_min:].max())   # positive bump after the trough
        two_comp = (m.min() < 0) and (overshoot > 0.1 * abs(m.min()))
        print(f"  logM [{lo},{hi}): n={sel.sum():4d}  "
              f"trough@R={r_neg:.2f}Mpc/h ({m.min():+.3e})  "
              f"overshoot={overshoot:+.3e}  two_component={'yes' if two_comp else 'no'}")

    # ---- ASSUMPTION 2 (THE CRUX): in-aperture scalar -> far-field ----
    from scipy.stats import spearmanr
    print("\n=== ASSUMPTION 2 (CRUX): in-aperture scalar predicts far-field ===")
    rho, p = spearmanr(s_in_all, f_out_all)
    print(f"  overall Spearman(s_in, f_out)      = {rho:+.3f} (p={p:.1e})")
    rho_m, _ = spearmanr(logm_all, f_out_all)
    print(f"  Spearman(logM,  f_out)             = {rho_m:+.3f}  (mass is the confounder)")
    pr = _partial_spearman(s_in_all, f_out_all, logm_all)
    print(f"  PARTIAL Spearman(s_in,f_out | logM) = {pr:+.3f}  <-- does the patch add info?")
    for lo, hi in zip(mass_edges[:-1], mass_edges[1:]):
        sel = (logm_all >= lo) & (logm_all < hi)
        if sel.sum() < 20:
            continue
        rr, _ = spearmanr(s_in_all[sel], f_out_all[sel])
        print(f"    within logM [{lo},{hi}): n={sel.sum():4d}  Spearman={rr:+.3f}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    for s in stacks:
        if s is None:
            continue
        lo, hi, n, m = s
        ax[0, 0].plot(r_cent, m, marker="o", ms=3, label=f"logM[{lo},{hi}) n={n}")
        ax[0, 1].plot(r_cent, m / np.abs(m).max(), marker="o", ms=3,
                      label=f"logM[{lo},{hi})")
    for a in ax[0]:
        a.axhline(0, color="k", lw=0.7)
        a.set_xscale("log")
        a.set_xlabel("R [Mpc/h]")
        a.legend(fontsize=8)
    ax[0, 0].set_title("Stacked delta_b(R) by mass (raw)")
    ax[0, 0].set_ylabel("mean delta_b per pixel")
    ax[0, 1].set_title("shape (normalized) -- look for +center, -intermediate")

    sc = ax[1, 0].scatter(s_in_all, f_out_all, c=logm_all, s=6, cmap="viridis")
    ax[1, 0].set_xlabel("s_in (in-aperture integral)")
    ax[1, 0].set_ylabel("f_out (far-field integral)")
    ax[1, 0].set_title(f"CRUX: r={rho:+.2f}  partial|logM={pr:+.2f}")
    plt.colorbar(sc, ax=ax[1, 0], label="logM")

    # residual scatter (rank space, mass removed) to visualize the partial correlation
    from scipy.stats import rankdata
    rz = rankdata(logm_all)
    b1 = np.c_[np.ones_like(rz), rz]
    ex = rankdata(s_in_all) - b1 @ np.linalg.lstsq(b1, rankdata(s_in_all), rcond=None)[0]
    ey = rankdata(f_out_all) - b1 @ np.linalg.lstsq(b1, rankdata(f_out_all), rcond=None)[0]
    ax[1, 1].scatter(ex, ey, s=6, alpha=0.4)
    ax[1, 1].set_xlabel("s_in rank residual | logM")
    ax[1, 1].set_ylabel("f_out rank residual | logM")
    ax[1, 1].set_title(f"partial correlation = {pr:+.2f}")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[feasibility] wrote {out}")

    # verdict.  For TOTAL matter the kernel is a central suppression + an
    # intermediate compensating overshoot (NOT the stars-centric +center the
    # original pitch assumed), and the in-aperture<->far-field link is a strong
    # *negative* correlation (ejection: aperture deficit <-> far-field surplus),
    # so the crux is judged on |partial corr|.
    print("\n=== VERDICT ===")

    def _two_comp(m):
        i = int(np.argmin(m))
        return (m.min() < 0) and (m[i:].max() > 0.1 * abs(m.min()))

    shape_ok = any(s is not None and _two_comp(s[3]) for s in stacks)
    crux_ok = (not np.isnan(pr)) and abs(pr) > 0.3
    print(f"  shape (central suppression + overshoot): {'PASS' if shape_ok else 'FAIL'}")
    print(f"  crux  (|partial corr| > 0.3):            "
          f"{'PASS' if crux_ok else 'FAIL'} ({pr:+.3f})")
    print("  -> build the kernel (negative central + positive intermediate component)"
          if (shape_ok and crux_ok)
          else "  -> reconsider the amplitude source before building")


if __name__ == "__main__":
    main()
