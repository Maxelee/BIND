"""fig03_measurement.pdf — the frozen data vector: <Y_CAP(4')> per nu bin for the
Wiener headline and the GLIMPSE cross-check, with total (stat + CIB-sys) errors.
The 0-1 bin is diagnostic-only (open symbol) and never enters the fit.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/paper3/B/wp2_measurement/stack_{wiener,glimpse}_sm2am_fid.npz
    (y_mean (5,5), y_cov (5,5,5) radius-cov per bin, nu_edges, cap_radii_arcmin)
  - /mnt/home/mlee1/ceph/paper3/B/wp3_nulls/sigma_sys_{wiener,glimpse}_decomposed.npz
    (half_band (5,) -> rank-1 CIB Sigma_sys)

Original source: WP-B2 frozen measurement + WP-B3 Sigma_sys (freeze-signed).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B")
J = 2   # 4' fiducial radius index of (2,3,4,6,8)'


def vec(variant):
    d = np.load(B / f"wp2_measurement/stack_{variant}_sm2am_fid.npz", allow_pickle=True)
    s = np.load(B / f"wp3_nulls/sigma_sys_{variant}_decomposed.npz")
    y = d["y_mean"][:, J]
    stat = np.sqrt(d["y_cov"][:, J, J])
    tot = np.sqrt(stat ** 2 + s["half_band"] ** 2)
    return y, tot, d["nu_edges"]


x = np.arange(5)
setup()
fig, ax = plt.subplots(figsize=ONE_COL)
# field convention (Schaan/Liu/Bigwood): data = black; variants = colored open
for variant, color, dx, fmt, mfc, label in (
        ("wiener", "k", -0.07, "o", None, "Wiener"),
        ("glimpse", COLORS["secondary"], +0.07, "s", "none", "GLIMPSE")):
    y, tot, edges = vec(variant)
    ax.errorbar(x[1:] + dx, y[1:], tot[1:], fmt=fmt, ms=3.5, color=color,
                mfc=mfc, capsize=2, lw=0.9, label=label)
    ax.errorbar(x[0] + dx, y[0], tot[0], fmt=fmt, ms=3.5, color=color,
                mfc="none", alpha=0.45, capsize=2, lw=0.9)

labels = [rf"$[{edges[b]:g},{edges[b+1]:g})$" for b in range(5)]
ax.set_xticks(x, labels)
ax.set_yscale("log")
ax.set_xlabel(r"peak significance bin $\nu$")
ax.set_ylabel(r"$\langle Y_{\rm CAP}(4')\rangle\ \ [y\,{\rm arcmin^2}]$")
ax.legend(loc="upper left")
ax.text(0.06, 0.62, "diagnostic\nbin", transform=ax.transAxes, fontsize=6,
        color=COLORS["dmo"], ha="left")

save(fig, "figs/fig03_measurement")
