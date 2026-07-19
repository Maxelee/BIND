"""fig07_plane_posterior.pdf — the (Delta ln M_gas, Delta ln T) plane: the sampled
model manifold (twobound nodes + 30-dim Sobol cloud), the TNG origin, and the
B5 posterior 68/95% HPD contours piling into the prior-window corner (the
pre-registered exclusion-style tripwire). pg1024 robustness posterior as an
unfilled dashed contour (Zurcher/Pandey robustness convention).

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/paper3/B/wp4_mocks/model_grid_tfwiener.npz (nodes)
  - /mnt/home/mlee1/ceph/paper3/B/wp4_mocks/sobol/model_grid_tfwiener_sb35.npz
  - /mnt/home/mlee1/ceph/paper3/B/wp5_inference/b5_posterior_wiener.npz (mg, dt, posterior)
  - /mnt/home/mlee1/ceph/paper3/B/wp5_inference/b5_posterior_wiener_pg1024.npz

The posterior is prior-window-truncated (edge mass 0.62-0.78); the contours are
an exclusion statement, not parameter estimates — caption carries this.
Original source: WP-B5 Task 3 fit + ladder items 1/8.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL_SQ, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B")


def hpd_levels(p, fracs=(0.68, 0.95)):
    q = np.sort(p.ravel())[::-1]
    c = np.cumsum(q)
    return [float(q[np.searchsorted(c, f)]) for f in fracs]


tg = np.load(B / "wp4_mocks/model_grid_tfwiener.npz", allow_pickle=True)
sb = np.load(B / "wp4_mocks/sobol/model_grid_tfwiener_sb35.npz", allow_pickle=True)
po = np.load(B / "wp5_inference/b5_posterior_wiener.npz")
pg = np.load(B / "wp5_inference/b5_posterior_wiener_pg1024.npz")

setup()
fig, ax = plt.subplots(figsize=ONE_COL_SQ)

ax.scatter(sb["delta_ln_mgas"], sb["delta_ln_t"], s=3, c=COLORS["dmo"],
           alpha=0.55, lw=0, label="Sobol box (253)")
ax.scatter(tg["delta_ln_mgas"], tg["delta_ln_t"], s=14, marker="s",
           facecolors="none", edgecolors=COLORS["bind"], lw=0.7,
           label="twobound grid (60)")
ax.plot(0, 0, marker="*", ms=10, color=COLORS["bind"], mec="k", mew=0.4,
        ls="none", label="TNG fiducial", zorder=5)

MG, DT = np.meshgrid(po["mg"], po["dt"], indexing="ij")
p = po["posterior"]
ax.contourf(MG, DT, p, levels=[*hpd_levels(p)[::-1], p.max()],
            colors=[COLORS["highlight"], COLORS["highlight"]], alpha=0.35)
ax.contour(MG, DT, p, levels=hpd_levels(p)[::-1], colors=COLORS["highlight"],
           linewidths=0.8)
pp = pg["posterior"]
ax.contour(MG, DT, pp, levels=hpd_levels(pp)[::-1], colors=COLORS["secondary"],
           linewidths=0.8, linestyles="--")
# proxy legend handles for the contours
ax.plot([], [], color=COLORS["highlight"], lw=1.5, label="posterior 68/95%")
ax.plot([], [], color=COLORS["secondary"], lw=1, ls="--", label="pg1024 variant")

ax.text(-0.352, 0.004, "posterior piles into the\nprior corner (exclusion)",
        fontsize=5.5, color=COLORS["highlight"], ha="left")
ax.set_xlim(-0.36, 0.16)
ax.set_ylim(-0.032, 0.062)
ax.set_xlabel(r"$\Delta\ln M_{\rm gas}$")
ax.set_ylabel(r"$\Delta\ln T$")
ax.legend(loc="upper right", fontsize=6)

save(fig, "figs/fig07_plane_posterior")
