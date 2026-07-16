"""fig02_methods_cap_schematic — from DMO to a stacked observable (§2 METHODS).

3 panels:
  (a) fiducial-TNG stacked tau(x) by halo-mass bin (4 bins), with the
      resolution-clean band 0.3<x<1.5 shaded; the BGS science bin is drawn
      with a thicker line.
  (b) stellar-to-halo mass relation (SHMR) 2-D histogram (painted central
      M_star vs M200 at the BGS snapshot), with the two DESI stellar-mass
      cuts (M*>10^11.0, 10^11.25) and their BIND-derived median host mass.
  (c) compensated-aperture (CAP) filter schematic on an illustrative
      analytic profile: disk (+1) minus equal-area ring (-1).

Source: examples/paper_ksz_desi_act.ipynb (branch analysis/ksz_project,
worktree .../wt/ksz-desi-act), cell 6 (heading "§2 METHODS"), built by
examples/_build_ksz_paper_nb.py lines ~348-388.

Data (read-only, cached on disk, no recomputation):
  - /mnt/home/mlee1/ceph/bind_science/ksz_confront/bind_tauy_fiducial_snap085.npz
    keys x, mbins, tau, cnts (panel a)
  - /mnt/home/mlee1/ceph/bind_science/ksz_confront/bind_mstar_xprof_snap085.npz
    key logM200 (panel b, median host mass per M* cut)
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet
    columns snap, M200, M_star_500 (panel b, 2-D histogram)
  Panel (c) is a purely synthetic illustrative profile ((1+(theta/1.5)^2)^-1),
  no data file, matching the original notebook cell exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
PARQUET = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet")

BGS_BIN = 1
SNAP_BGS = 85

setup()

fce = np.load(KS / "bind_tauy_fiducial_snap085.npz")
x = fce["x"]
tauF = fce["tau"]
cntsF = fce["cnts"]
mbe = fce["mbins"]
cm = np.load(KS / "bind_mstar_xprof_snap085.npz")

fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], 3.0), constrained_layout=True)

# (a) fiducial tau profiles by mass bin (the stacking step)
cols = plt.cm.viridis(np.linspace(0, 0.9, 4))
for b in range(4):
    lab = rf"$\log M$=[{mbe[b]:.1f},{mbe[b+1]:.1f}] (N$\sim${int(cntsF[b])})"
    ax[0].plot(
        x, tauF[b], "o-", ms=2.2, color=cols[b], lw=1.2 + (b == BGS_BIN) * 1.4, label=lab,
        zorder=3 + (b == BGS_BIN),
    )
ax[0].axvspan(0.3, 1.5, color="0.85", alpha=0.5, zorder=0, label="clean band $0.3{<}x{<}1.5$")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"$x=R/r_{200c}$")
ax[0].set_ylabel(r"fiducial stacked $\tau(x)$")
ax[0].set_xlim(x.min(), x.max())
ax[0].legend(loc="lower left")
panel_label(ax[0], "(a)")

# (b) SHMR: M_star -> M200, with the two M* cuts
sh = pd.read_parquet(PARQUET, columns=["snap", "M200", "M_star_500"]).query("snap==@SNAP_BGS")
sh = sh[sh.M_star_500 > 0]
lms, lmh = np.log10(sh.M_star_500.values), np.log10(sh.M200.values)
H, xe, ye = np.histogram2d(lms, lmh, bins=[np.linspace(10.3, 12.0, 40), np.linspace(13.0, 14.4, 40)])
im = ax[1].pcolormesh(xe, ye, np.log10(H.T + 1), cmap="Blues", shading="auto", rasterized=True)
for cut, col in [(11.0, "tab:orange"), (11.25, COLORS["highlight"])]:
    mh = np.nanmedian(cm["logM200"][:, 0 if cut == 11.0 else 1])
    ax[1].axvline(cut, color=col, ls="--", lw=1.2)
    ax[1].axhline(mh, color=col, ls=":", lw=1.0)
    ax[1].plot(cut, mh, "*", color=col, ms=12, label=rf"$M_\star{{>}}{cut}\to\log M_{{200}}{{=}}{mh:.2f}$")
ax[1].set_xlabel(r"$\log_{10} M_\star\,[M_\odot]$")
ax[1].set_ylabel(r"$\log_{10} M_{200}\,[M_\odot/h]$")
ax[1].set_xlim(10.3, 12.0)
ax[1].set_ylim(13.0, 14.2)
ax[1].legend(loc="lower right")
cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
cb.set_label(r"$\log_{10}(N+1)$")
panel_label(ax[1], "(b)")

# (c) CAP filter schematic on a model profile (synthetic, illustrative only)
th = np.linspace(0, 8, 400)
prof = (1 + (th / 1.5) ** 2) ** (-1.0)
thd = 2.5
ax[2].plot(th, prof, "k-", lw=1.4)
ax[2].fill_between(
    th, 0, prof, where=(th <= thd), color=COLORS["bind"], alpha=0.45, label=r"disk $+1$ ($\theta<\theta_d$)"
)
ax[2].fill_between(
    th, 0, prof, where=(th > thd) & (th <= np.sqrt(2) * thd), color=COLORS["highlight"], alpha=0.45,
    label=r"ring $-1$ (equal area)",
)
ax[2].axvline(thd, color="0.4", lw=0.8, ls=":")
ax[2].axvline(np.sqrt(2) * thd, color="0.4", lw=0.8, ls=":")
ax[2].annotate(r"$\theta_d$", (thd, 0.92), fontsize=8, ha="center")
ax[2].annotate(r"$\sqrt{2}\,\theta_d$", (np.sqrt(2) * thd, 0.92), fontsize=8, ha="center")
ax[2].set_xlabel(r"$\theta$ [arcmin]")
ax[2].set_ylabel(r"model $\tau(\theta)$")
ax[2].set_xlim(0, 6)
ax[2].set_ylim(0, 1.05)
ax[2].legend(loc="lower right")
panel_label(ax[2], "(c)")

save(fig, "figs/fig02_methods_cap_schematic")
