#!/usr/bin/env python
"""fig03_validation_halo -- halo-level closure at snap 96 (z~0.03), BIND vs
TNG300 hydro truth (Sec. 2.4).

Three panels:
  (a) Y_500c-M200c relation with intrinsic (16-84th pct) scatter band, BIND
      vs TNG truth.
  (b) stacked electron-column (tau) BIND/hydro ratio profile, 3 M200c bins.
  (c) projected Compton-y profile y(r)/y(r500c), 4 M200c bins, compared to
      the Arnaud+2010 universal GNFW pressure profile.

Data (cached, no re-derivation):
  - /mnt/home/mlee1/ceph/bind_science/halo_atlas/{fid,truth}_snap096.npz
    (keys M_fof, Y_500c) for panel (a)
  - /mnt/home/mlee1/ceph/bind_science/tau_profiles/tau_profiles_snap096.npz
    (keys r_r500, mass_bins, bind_prof, truth_prof, bind_cnt) for panel (b)
  - /mnt/home/mlee1/ceph/bind_science/profiles/{fid,truth}_snap096.npz
    (keys r_cen, mass_bins, counts, prof_y) for panel (c), fit against a
    hardcoded Arnaud+2010 GNFW (P0=8.403, c500=1.177, gamma=0.3081,
    alpha=1.0510, beta=5.4905)

Source: examples/paper_lightcone_figs.ipynb (worktree wl-tsz-bridge), cell 5
(heading "Sec. 2.4 -- Validation against TNG300 hydro truth").
Placeholder this replaces: figs/fig03_validation_halo.png (md5-identical to
figs_raw/paper_lightcone_figs/cell005_out1.png).

Styling note: panels (b) and (c) both bin by halo mass, so both use the same
mass-ordered sequential colormap (viridis) here for internal consistency
(the original notebook cell used viridis explicitly for panel (c) but the
unstyled default color cycle for panel (b) -- same underlying curves/data,
just a single consistent mass color scale across both mass-binned panels).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

ATLAS = Path("/mnt/home/mlee1/ceph/bind_science/halo_atlas")
TAU = Path("/mnt/home/mlee1/ceph/bind_science/tau_profiles")
PROF = Path("/mnt/home/mlee1/ceph/bind_science/profiles")


def ym_relation(d):
    M, Y = d["M_fof"], d["Y_500c"]
    g = (M > 1e13) & (Y > 0)
    lM, lY = np.log10(M[g]), np.log10(Y[g])
    edg = np.linspace(13, 14.8, 9)
    cen = 0.5 * (edg[1:] + edg[:-1])
    med = np.full(len(cen), np.nan)
    sca = np.full(len(cen), np.nan)
    for k in range(len(cen)):
        s = (lM >= edg[k]) & (lM < edg[k + 1])
        if s.sum() >= 8:
            med[k] = np.median(lY[s])
            sca[k] = 0.5 * (np.percentile(lY[s], 84) - np.percentile(lY[s], 16))
    return cen, med, sca


def main() -> None:
    setup()
    af, at = np.load(ATLAS / "fid_snap096.npz"), np.load(ATLAS / "truth_snap096.npz")
    fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], 2.7))

    # (a) Y-M relation, BIND vs TNG truth, with intrinsic scatter
    cb, mb, sb = ym_relation(af)
    ct, mt, st = ym_relation(at)
    ax[0].fill_between(cb, mb - sb, mb + sb, color=COLORS["bind"], alpha=0.18, lw=0)
    ax[0].plot(cb, mb, color=COLORS["bind"], lw=1.6, label="BIND")
    ax[0].plot(ct, mt, color=COLORS["truth"], ls="--", lw=1.4, label="TNG300 hydro")
    ax[0].set_xlabel(r"$\log_{10} M_{200c}\,[M_\odot/h]$")
    ax[0].set_ylabel(r"$\log_{10} Y_{500c}$")
    ax[0].legend(loc="upper left")
    panel_label(ax[0], "(a)", loc="lower right")

    # (b) stacked tau profile BIND/truth ratio per mass bin
    t = np.load(TAU / "tau_profiles_snap096.npz")
    r5 = t["r_r500"]
    mb_edges = t["mass_bins"]
    nb = t["bind_prof"].shape[0]
    cols_b = plt.cm.viridis(np.linspace(0.15, 0.85, nb))
    for b in range(nb):
        if t["bind_cnt"][b] < 20:
            continue
        ratio = t["bind_prof"][b] / t["truth_prof"][b]
        lab = rf"$10^{{{np.log10(mb_edges[b]):.1f}}}$--$10^{{{np.log10(mb_edges[b + 1]):.1f}}}$"
        ax[1].plot(r5, ratio, lw=1.3, color=cols_b[b], label=lab)
    ax[1].axhspan(0.95, 1.05, color="0.7", alpha=0.25, lw=0)
    ax[1].axhline(1.0, color="k", lw=0.7)
    ax[1].set_xscale("log")
    ax[1].set_ylim(0.7, 1.4)
    ax[1].set_xlabel(r"$r/r_{500c}$")
    ax[1].set_ylabel(r"$\tau_{\rm BIND}/\tau_{\rm hydro}$")
    ax[1].legend(title=r"$M_{200c}\,[M_\odot/h]$", title_fontsize=6, fontsize=6, loc="upper left")
    panel_label(ax[1], "(b)", loc="lower right")

    # (c) projected pressure profile vs Arnaud+10 universal
    pf, pt = np.load(PROF / "fid_snap096.npz"), np.load(PROF / "truth_snap096.npz")
    rc = pf["r_cen"]
    jn = int(np.argmin(np.abs(rc - 1.0)))
    P0, c5, g0, al, be = 8.403, 1.177, 0.3081, 1.0510, 5.4905
    ll = np.linspace(0, 6, 400)
    def p3(x):
        return P0 / ((c5 * x) ** g0 * (1 + (c5 * x) ** al) ** ((be - g0) / al))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    Ar = np.array([trapz(p3(np.sqrt(r ** 2 + ll ** 2)), ll) for r in rc])
    Ar /= Ar[jn]
    rok = (rc >= 0.15) & (rc <= 3.0)
    ax[2].loglog(rc[rok], Ar[rok], color=COLORS["truth"], ls=":", lw=1.6, label="Arnaud+10")

    def sc(p):
        return p / p[jn]

    n_pc = pf["prof_y"].shape[0]
    mb_lab = [rf"$10^{{{np.log10(pf['mass_bins'][b]):.1f}}}$" for b in range(n_pc)]
    cols_c = plt.cm.viridis(np.linspace(0.1, 0.85, n_pc))
    for b in range(n_pc):
        if pf["counts"][b] < 20:
            continue
        ax[2].loglog(rc[rok], sc(pf["prof_y"][b])[rok], color=cols_c[b], lw=1.4, label=mb_lab[b])
    ax[2].set_xlabel(r"$r/r_{500c}$")
    ax[2].set_ylabel(r"$y(r)/y(r_{500c})$")
    ax[2].legend(title=r"$M_{200c}\,[M_\odot/h]$", title_fontsize=6, fontsize=6, loc="lower left")
    panel_label(ax[2], "(c)", loc="upper right")

    fig.tight_layout(w_pad=1.0)
    save(fig, "figs/fig03_validation_halo")


if __name__ == "__main__":
    main()
