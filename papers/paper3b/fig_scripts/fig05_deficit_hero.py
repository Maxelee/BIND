"""fig05_deficit_hero.pdf — THE result: the frozen <Y_CAP(4')> data vector against
the entire reachable model manifold. The 60-unit twobound grid is drawn as thin
curves color-coded by Delta ln M_gas (Zurcher-style parameter colorbar); the
253-run 30-dim Sobol box as a grey envelope behind (Bigwood-style hypercube);
the TNG-painted fiducial and the best-achievable unit called out with their
chi2 in the legend (Siegel convention). Data: black points, sigma_total.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/paper3/B/wp2_measurement/stack_wiener_sm2am_fid.npz
  - /mnt/home/mlee1/ceph/paper3/B/wp3_nulls/sigma_sys_wiener_decomposed.npz
  - /mnt/home/mlee1/ceph/paper3/B/wp4_mocks/model_grid_tfwiener.npz
  - /mnt/home/mlee1/ceph/paper3/B/wp4_mocks/sobol/model_grid_tfwiener_sb35.npz
  - /mnt/home/mlee1/ceph/paper3/B/wp4_mocks/grid_tfwiener/bind_run_0000.npz
  - chi2 provenance: wp5_inference/b5_chi2_supplement.json + wp4_mocks/sobol/
    b5_sobol_verdict.json (quoted, not recomputed)

Original source: WP-B5 Task 3 + ladder item 8 (wp5 REPORT).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL_SQ, save, setup  # noqa: E402

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B")
J = 2

d = np.load(B / "wp2_measurement/stack_wiener_sm2am_fid.npz", allow_pickle=True)
s = np.load(B / "wp3_nulls/sigma_sys_wiener_decomposed.npz")
y_data = d["y_mean"][1:5, J]
tot = np.sqrt(d["y_cov"][1:5, J, J] + s["half_band"][1:5] ** 2)

tg = np.load(B / "wp4_mocks/model_grid_tfwiener.npz", allow_pickle=True)
sb = np.load(B / "wp4_mocks/sobol/model_grid_tfwiener_sb35.npz", allow_pickle=True)
fid = np.load(B / "wp4_mocks/grid_tfwiener/bind_run_0000.npz", allow_pickle=True)
y_tb = np.asarray(tg["y_mean"])[:, 1:5, J]
mg_tb = np.asarray(tg["delta_ln_mgas"])
y_sb = np.asarray(sb["y_mean"])[:, 1:5, J]
y_fid = fid["y_mean"][1:5, J]
i_best = int(np.argmin(np.sum((y_data - y_tb) ** 2, axis=1)))  # display pick only

x = np.arange(4)
setup()
fig, ax = plt.subplots(figsize=ONE_COL_SQ)

ax.fill_between(x, y_sb.min(axis=0), y_sb.max(axis=0), color=COLORS["dmo"],
                alpha=0.30, lw=0, label=r"Sobol 30-dim box (253), $\chi^2_{\min}=340/4$")

norm = mpl.colors.Normalize(mg_tb.min(), mg_tb.max())
cmap = mpl.cm.cividis
for i in range(len(mg_tb)):
    ax.plot(x, y_tb[i], color=cmap(norm(mg_tb[i])), lw=0.5, alpha=0.6, zorder=2)
ax.plot(x, y_fid, color=COLORS["bind"], lw=1.6, marker="s", ms=3, zorder=4,
        label=r"TNG-painted fiducial, $\chi^2=444/4$")
ax.plot(x, y_tb[i_best], color=COLORS["highlight"], lw=1.1, ls="--", zorder=4,
        label=r"best grid unit, $\chi^2=313/4$")
ax.errorbar(x, y_data, tot, fmt="o", ms=4.5, color="k", capsize=2.5, lw=1.1,
            zorder=6, label="DES Y3 peaks $\\times$ ACT DR6 $y$")

edges = d["nu_edges"]
ax.set_xticks(x, [rf"$[{edges[b]:g},{edges[b+1]:g})$" for b in range(1, 5)])
ax.set_yscale("log")
ax.set_xlabel(r"peak significance bin $\nu$")
ax.set_ylabel(r"$\langle Y_{\rm CAP}(4')\rangle\ \ [y\,{\rm arcmin^2}]$")
ax.legend(loc="upper left", fontsize=6)

cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                  shrink=0.8, pad=0.02)
cb.set_label(r"grid $\Delta\ln M_{\rm gas}$")

save(fig, "figs/fig05_deficit_hero")
