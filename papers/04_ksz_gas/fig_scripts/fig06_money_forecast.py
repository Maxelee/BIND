"""fig06_money_forecast.py

The paper's "MONEY PLOT" (Sec. 8): forecast corner plot of the two
feedback directions constrained by a future Simons Observatory/CMB-S4 x
DESI joint kSZ (density) + tSZ (pressure) CAP analysis, compared to their
joint combination. The two probes are complementary (each alone leaves one
direction weakly constrained) while the joint chain pins both -- and those
two directions coincide with the 2-D feedback latent of fig08.

Data (pre-cached, read-only, pure re-plot -- no re-sampling):
  - KS/ksz_posterior_multiprobe_future.npz  (193 MB)
      keys: params, log_params, prior_lo, prior_hi,
            chain_ksz, chain_tsz, chain_joint,
            ratio_ksz, ratio_tsz, ratio_joint
  KS = /mnt/home/mlee1/ceph/bind_science/ksz_confront

Source: examples/paper_ksz_desi_act.ipynb (branch analysis/ksz_project,
worktree ksz-desi-act), cell 28, built by
examples/_build_ksz_paper_nb.py lines ~1307-1330. The pre-computed MCMC
chains for all three forecast scenarios are already in the npz; this
script only whitens by the prior, eigendecomposes cov(chain_joint) to find
the 2 best-constrained directions, projects all three chains onto them, and
draws the corner plot -- identical to the notebook cell's plotting body.
The diagnostic print statements in the original cell (fraction of the
forecast direction captured by the Sec-4 response latent, loaded from a
/tmp scratch file written by an earlier cell) are prose-only asides not
part of the figure and are not reproduced here.
"""
import sys
from pathlib import Path

import corner
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, save, COLORS

# corner's default 2-D layout needs more room than ONE_COL_SQ (3.5in) to keep
# the per-axis loading labels legible without shrinking fonts (FIGURE_STYLE
# rule 7: resize the figure, don't shrink fonts); this is a 3-populated-panel
# triangular layout, so a modest custom square is used instead of ONE_COL_SQ.
CORNER_SQ = (4.6, 4.6)

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")

setup()

d = np.load(KS / "ksz_posterior_multiprobe_future.npz", allow_pickle=True)
params = list(d["params"])
lo, hi = d["prior_lo"], d["prior_hi"]
pmean, psd = 0.5 * (lo + hi), (hi - lo) / np.sqrt(12)


def white(ch):
    return (ch - pmean) / psd


Uj = white(d["chain_joint"])
ev, evec = np.linalg.eigh(np.cov(Uj.T))
V = evec[:, :2]


def topload(v, n=3):
    i = np.argsort(np.abs(v))[::-1][:n]
    return " ".join(f"{'+' if v[j] > 0 else '-'}{params[j][:9]}" for j in i)


labs = [f"dir{k + 1} ($\\nu$={ev[k]:.2f}) [prior $\\sigma$]\n{topload(V[:, k])}" for k in range(2)]
Ak = white(d["chain_ksz"]) @ V
At = white(d["chain_tsz"]) @ V
Aj = Uj @ V

ck = dict(plot_datapoints=False, fill_contours=True, levels=(0.68, 0.95), smooth=1.2,
          bins=26, range=[(-2.2, 2.2)] * 2, label_kwargs=dict(fontsize=7))
fig = plt.figure(figsize=CORNER_SQ)
corner.corner(Ak, color=COLORS["bind"], labels=labs, fig=fig, **ck)       # kSZ (density) -- widest first
corner.corner(At, color=COLORS["secondary"], fig=fig, **ck)               # tSZ (pressure)
corner.corner(Aj, color=COLORS["highlight"], fig=fig, **ck)               # kSZ+tSZ joint -- tightest, drawn last

fig.legend(
    handles=[
        plt.Line2D([], [], color=COLORS["bind"], label="kSZ only"),
        plt.Line2D([], [], color=COLORS["secondary"], label="tSZ only"),
        plt.Line2D([], [], color=COLORS["highlight"], label="kSZ+tSZ joint"),
    ],
    loc="upper right", fontsize=6.5, frameon=False,
)
save(fig, "figs/fig06_money_forecast")
