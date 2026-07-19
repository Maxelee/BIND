"""fig01_peak_catalogs.pdf — the frozen kappa-peak samples: nu distributions of the
Wiener (headline) and GLIMPSE (cross-check) catalogs at the fiducial 2' smoothing,
with the Wiener smoothing-robustness set {1,5,8}' behind.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/paper3/B/wp1_maps/peaks_{wiener,glimpse}_sm2am.npz (key: nu)
  - /mnt/home/mlee1/ceph/paper3/B/wp1_maps/peaks_wiener_sm{1,5,8}am.npz

Original source: WP-B1 peak chain (analysis/paper3b/maps; PEAK_DEFINITION.md);
catalog independently re-derived bit-exactly 2026-07-19 (gap-2 closure).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B/wp1_maps")

setup()
fig, ax = plt.subplots(figsize=ONE_COL)
bins = np.linspace(-2, 9, 45)

for s in (1, 5, 8):
    nu = np.load(B / f"peaks_wiener_sm{s}am.npz")["nu"]
    ax.hist(nu, bins=bins, histtype="step", lw=0.7, color=COLORS["bind"], alpha=0.35)

nu_w = np.load(B / "peaks_wiener_sm2am.npz")["nu"]
nu_g = np.load(B / "peaks_glimpse_sm2am.npz")["nu"]
ax.hist(nu_w, bins=bins, histtype="step", lw=1.5, color=COLORS["bind"],
        label=f"Wiener 2$'$ ({len(nu_w):,})")
ax.hist(nu_g, bins=bins, histtype="step", lw=1.2, color=COLORS["secondary"],
        label=f"GLIMPSE 2$'$ ({len(nu_g):,})")
for e in (1, 2, 3, 4):
    ax.axvline(e, color=COLORS["dmo"], lw=0.5, ls=":", zorder=0)

ax.set_yscale("log")
ax.set_xlabel(r"peak significance $\nu$")
ax.set_ylabel("peaks per bin")
ax.set_xlim(-2, 9)
ax.legend(loc="upper right")
ax.text(0.985, 0.55, r"faint: Wiener $\{1,5,8\}'$", transform=ax.transAxes,
        ha="right", fontsize=6, color=COLORS["bind"], alpha=0.7)

save(fig, "figs/fig01_peak_catalogs")
