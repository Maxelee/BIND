"""fig06_radius_ratio.pdf — the deficit decomposed in radius: CAP-radius-resolved
measurement vs the TNG-painted fiducial, one column per detected nu bin; lower
panels show data/model with the unity line (Siegel-style ratio subpanels) and a
dashed line at each bin's OUTER (8') amplitude. Two components: a bin-coherent
~2x suppression present at ALL radii (~4x top bin), plus an ADDITIONAL central
suppression below the dashed line (x0.5-0.7 in bins 2-3, x0.3 top, x0.08 bin 1).

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/paper3/B/wp2_measurement/stack_wiener_sm2am_fid.npz
    (y_mean (5,5), y_cov (5,5,5) per-bin radius covariance, cap_radii_arcmin)
  - /mnt/home/mlee1/ceph/paper3/B/wp4_mocks/grid_tfwiener/bind_run_0000.npz

Original source: WP-B5 systematic hunt, radius diagnostic (wp5 REPORT).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL_TALL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B")
d = np.load(B / "wp2_measurement/stack_wiener_sm2am_fid.npz", allow_pickle=True)
m = np.load(B / "wp4_mocks/grid_tfwiener/bind_run_0000.npz", allow_pickle=True)
r = d["cap_radii_arcmin"]
edges = d["nu_edges"]

setup()
fig, axes = plt.subplots(2, 4, figsize=(TWO_COL_TALL[0], 3.9), sharex=True,
                         height_ratios=[2.1, 1.0], constrained_layout=True)
for k, b in enumerate(range(1, 5)):
    yd = d["y_mean"][b]
    err = np.sqrt(np.diag(d["y_cov"][b]))
    ym = m["y_mean"][b]
    top, bot = axes[0, k], axes[1, k]
    top.plot(r, ym, color=COLORS["bind"], lw=1.3, label="TNG-painted" if k == 0 else None)
    top.errorbar(r, yd, err, fmt="o", ms=3, color="k", capsize=2, lw=0.9,
                 label="data" if k == 0 else None)
    top.set_yscale("log")
    top.text(0.05, 0.95, rf"$\nu\in[{edges[b]:g},{edges[b+1]:g})$",
             transform=top.transAxes, va="top", fontsize=7)
    bot.errorbar(r, yd / ym, err / ym, fmt="o", ms=3, color="k", capsize=2, lw=0.9)
    bot.axhline(1, color=COLORS["dmo"], lw=0.8)
    # the bin-coherent OUTER amplitude: the component a constant rescaling could
    # absorb; the fall below it toward the centre is the additional shape part
    bot.axhline(yd[-1] / ym[-1], color=COLORS["bind"], lw=0.7, ls="--")
    if k == 0:
        bot.text(1.75, yd[-1] / ym[-1] * 1.18, "outer amplitude", fontsize=5,
                 color=COLORS["bind"], ha="left")
    bot.set_yscale("log")
    bot.set_ylim(0.03, 2.5)
    bot.set_xlabel(r"$\theta_d$ [arcmin]")
    for ax in (top, bot):
        ax.axvspan(6.0, 8.6, color=COLORS["dmo"], alpha=0.12, lw=0)
        ax.axvline(1.6, color=COLORS["dmo"], lw=0.6, ls=":")
        ax.set_xlim(1.6, 8.6)
    if k == 0:
        top.set_ylabel(r"$\langle Y_{\rm CAP}(\theta_d)\rangle\ [y\,{\rm arcmin^2}]$")
        bot.set_ylabel("data / model")
        top.legend(loc="upper left", fontsize=6, bbox_to_anchor=(0.02, 0.82))
    else:
        top.set_yticklabels([]); bot.set_yticklabels([])
    top.set_ylim(8e-7, 2.5e-3)

save(fig, "figs/fig06_radius_ratio")
