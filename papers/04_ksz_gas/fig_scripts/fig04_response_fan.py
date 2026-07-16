"""fig04_response_fan — the continuous feedback response fan (§3).

Top row: stacked tau(x) [kSZ] and y(x) [tSZ] of the BGS science bin
(logM200 in [13.4,13.8]) for all 256 Sobol nodes, each curve colored by
that node's gas fraction f_gas,500/(Omega_b/Omega_m), with the fiducial
TNG profile as the heavy black reference line and the resolution-clean
band 0.3<x<1.5 shaded. Bottom row: fractional node spread
(p84-p16)/2/fiducial vs x, showing the feedback response is core-dominated.

Source: examples/paper_ksz_desi_act.ipynb (branch analysis/ksz_project,
worktree .../wt/ksz-desi-act), cell 8 (heading "§3 The feedback
response"), built by examples/_build_ksz_paper_nb.py lines ~417-458.

Data (read-only, cached on disk, no recomputation):
  - /mnt/home/mlee1/ceph/bind_science/ksz_confront/bind_tauy_xprof_snap085.npz
    keys x, tau, y, nodes (256-node profile cube, BGS mass bin index 1)
  - /mnt/home/mlee1/ceph/bind_science/ksz_confront/bind_tauy_fiducial_snap085.npz
    (fiducial-TNG reference profile)
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet
    columns run, snap, M_tot_500, f_gas_500 (per-node gas-fraction color)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import TWO_COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
PARQUET = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet")

F_B = 0.0490 / 0.3089  # Omega_b/Omega_m (Planck/TNG)
BGS_BIN = 1
SNAP_BGS, Z_BGS = 85, 0.18

setup()

c = np.load(KS / "bind_tauy_xprof_snap085.npz")
x, tau, y, nodes = c["x"], c["tau"], c["y"], c["nodes"]
fc = np.load(KS / "bind_tauy_fiducial_snap085.npz")
MB = BGS_BIN
fgdf = pd.read_parquet(PARQUET, columns=["run", "snap", "M_tot_500", "f_gas_500"])
fgdf = fgdf[(fgdf.snap == SNAP_BGS) & (np.log10(fgdf.M_tot_500) > 13.3)]
fg = (fgdf.groupby("run").f_gas_500.median() / F_B).reindex(nodes).values
norm = plt.Normalize(np.nanpercentile(fg, 5), np.nanpercentile(fg, 95))
cmap = plt.cm.RdYlBu_r

fig, axes = plt.subplots(
    2, 2, figsize=(TWO_COL[0], 4.5), sharex="col", gridspec_kw=dict(height_ratios=[3, 1.25]),
    constrained_layout=True,
)
for j, (fld, ffid, lab) in enumerate(
    [(tau, fc["tau"], r"$\tau(R)$ [electron column, kSZ]"), (y, fc["y"], r"$y(R)$ [pressure, tSZ]")]
):
    ax, axr = axes[0, j], axes[1, j]
    p = fld[:, MB, :]
    pf = ffid[MB]
    segs = [np.column_stack([x, p[i]]) for i in range(len(nodes)) if np.all(p[i] > 0)]
    cvals = [fg[i] for i in range(len(nodes)) if np.all(p[i] > 0)]
    lc = LineCollection(segs, cmap=cmap, norm=norm, alpha=0.5, lw=0.6)
    lc.set_array(np.array(cvals))
    ax.add_collection(lc)
    ax.plot(x, pf, "k-", lw=2.0, zorder=5, label="fiducial TNG")
    ax.axvspan(0.3, 1.5, color="0.85", alpha=0.4, zorder=0, label=r"clean band $0.3{<}x{<}1.5$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(np.nanpercentile(p[p > 0], 1), np.nanpercentile(p[p > 0], 99.5))
    ax.set_ylabel(lab)
    ax.legend(loc="lower left", fontsize=6.0)
    lo, hi = np.nanpercentile(p, [16, 84], 0)
    frac = (hi - lo) / (2 * np.where(pf > 0, pf, np.nan))
    axr.plot(x, frac, "-", color="tab:purple", lw=1.5)
    axr.axvspan(0.3, 1.5, color="0.85", alpha=0.4, zorder=0)
    axr.axvline(1.0, color="0.6", ls=":", lw=0.6)
    axr.set_xscale("log")
    axr.set_xlim(x.min(), x.max())
    axr.set_ylim(0, None)
    axr.set_xlabel(r"$R/r_{200c}$")
    axr.set_ylabel("node spread\n/ fiducial", fontsize=6.5)
axes[0, 0].annotate(
    "feedback fans the core", xy=(0.18, 0.5), xytext=(0.40, 0.14), textcoords="axes fraction", fontsize=6,
    color="0.25",
)
cb = fig.colorbar(lc, ax=axes, fraction=0.04, pad=0.02)
cb.set_label(r"$\tilde f_{\rm gas,500}$ (node)", fontsize=8)

save(fig, "figs/fig04_response_fan")
