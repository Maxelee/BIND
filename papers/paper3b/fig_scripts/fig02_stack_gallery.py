"""fig02_stack_gallery.pdf — the measurement, visually: stacked ACT DR6 Compton-y
thumbnails at DES Y3 Wiener kappa-peaks in the four detected nu bins (the 0-1
diagnostic bin is excluded), with the fiducial 4' CAP aperture overdrawn.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/paper3/B/wp2_measurement/stack_wiener_sm2am_fid.npz
    (keys: stacked_thumbs (5,61,61) at 0.5'/px over +-15', nu_edges, n_per_bin)

Original source: WP-B2 frozen measurement (freeze-signed 2026-07-17).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, save, setup  # noqa: E402

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

d = np.load(Path("/mnt/home/mlee1/ceph/paper3/B/wp2_measurement/stack_wiener_sm2am_fid.npz"),
            allow_pickle=True)
thumbs = d["stacked_thumbs"][1:5]          # detected bins only
edges = d["nu_edges"]
n = d["n_per_bin"][1:5].astype(int)
ext = 15.0                                  # arcmin half-width (61 px at 0.5')

setup()
fig, axes = plt.subplots(1, 4, figsize=(TWO_COL[0], 2.2), constrained_layout=True)
vmax = np.percentile(thumbs[3], 99.0) * 1e6   # shared asinh stretch (Coulton-style
norm = mpl.colors.AsinhNorm(linear_width=0.8,  # asymmetric range; bins 1-2 stay
                            vmin=-0.6, vmax=vmax)  # resolved, bin 4 unclipped)
for k, ax in enumerate(axes):
    im = ax.imshow(thumbs[k] * 1e6, origin="lower", cmap="cividis",
                   extent=[-ext, ext, -ext, ext], norm=norm, rasterized=True)
    th = np.linspace(0, 2 * np.pi, 100)
    for r, ls in ((4.0, "-"), (4.0 * np.sqrt(2), "--")):
        ax.plot(r * np.cos(th), r * np.sin(th), ls, color="w", lw=0.6)
    ax.text(0.05, 0.95, rf"$\nu\in[{edges[k+1]:g},{edges[k+2]:g})$",
            transform=ax.transAxes, va="top", color="w", fontsize=7)
    ax.text(0.05, 0.05, f"{n[k]:,} peaks", transform=ax.transAxes,
            color="w", fontsize=6)
    ax.set_xlabel(r"$\Delta\theta_x$ [arcmin]")
    if k == 0:
        ax.set_ylabel(r"$\Delta\theta_y$ [arcmin]")
    else:
        ax.set_yticklabels([])
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
cb.set_label(r"$\langle y\rangle\ [10^{-6}]$")

save(fig, "figs/fig02_stack_gallery")
