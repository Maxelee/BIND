"""fig08_sobol_chi2.pdf — ladder item 8, the pre-registered manifold-coverage
verdict: per-unit diagonal chi2 of the frozen data vector against every
twobound grid unit and all 253 Sobol-box units, vs each unit's Delta ln M_gas.
The pre-registered thresholds (100 = full-box rejection; 30 = material
weakening) are drawn; no unit approaches either.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/paper3/B/wp4_mocks/sobol/b5_sobol_verdict.json (per-unit chi2)
  - /mnt/home/mlee1/ceph/paper3/B/wp4_mocks/sobol/model_grid_tfwiener_sb35.npz (coords)
  - /mnt/home/mlee1/ceph/paper3/B/wp4_mocks/model_grid_tfwiener.npz +
    wp2_measurement / wp3_nulls frozen vectors (twobound chi2, registered recipe)

Original source: wp5 REPORT ladder item 8 (criterion pre-registered before any
Sobol y_mean look); registered recipe = diagonal, fit bins at 4',
sigma = sqrt(diag(Sigma_stat + Sigma_sys)).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B")
J = 2

v = json.loads((B / "wp4_mocks/sobol/b5_sobol_verdict.json").read_text())
sb = np.load(B / "wp4_mocks/sobol/model_grid_tfwiener_sb35.npz", allow_pickle=True)
names = [str(n) for n in sb["run_names"]]
chi2_sb = np.array([v["per_unit_chi2"][n] for n in names])
mg_sb = np.asarray(sb["delta_ln_mgas"])

d = np.load(B / "wp2_measurement/stack_wiener_sm2am_fid.npz", allow_pickle=True)
s = np.load(B / "wp3_nulls/sigma_sys_wiener_decomposed.npz")
y = d["y_mean"][1:5, J]
sig = np.sqrt(d["y_cov"][1:5, J, J] + s["half_band"][1:5] ** 2)
tg = np.load(B / "wp4_mocks/model_grid_tfwiener.npz", allow_pickle=True)
chi2_tb = np.sum((y[None] - np.asarray(tg["y_mean"])[:, 1:5, J]) ** 2 / sig[None] ** 2,
                 axis=1)

setup()
fig, ax = plt.subplots(figsize=ONE_COL)
ax.scatter(mg_sb, chi2_sb, s=5, c=COLORS["dmo"], lw=0, alpha=0.8,
           label="Sobol box (253)")
ax.scatter(tg["delta_ln_mgas"], chi2_tb, s=12, marker="s", facecolors="none",
           edgecolors=COLORS["bind"], lw=0.7, label="twobound grid (60)")
k = int(np.argmin(chi2_sb))
ax.plot(mg_sb[k], chi2_sb[k], marker="*", ms=9, color=COLORS["highlight"],
        mec="k", mew=0.3, ls="none", zorder=5, label=r"$\chi^2_{\min}=340/4$")

ax.axhline(100, color=COLORS["highlight"], lw=0.8, ls="--")
ax.axhline(30, color=COLORS["secondary"], lw=0.8, ls="--")
ax.text(0.148, 100, "full-box rejection ", ha="right", va="bottom", fontsize=6,
        color=COLORS["highlight"])
ax.text(-0.345, 30, " material weakening", ha="left", va="bottom", fontsize=6,
        color=COLORS["secondary"])
ax.set_yscale("log")
ax.set_ylim(20, 1500)
ax.set_xlabel(r"$\Delta\ln M_{\rm gas}$")
ax.set_ylabel(r"$\chi^2$ vs frozen data (4 dof)")
ax.legend(loc="lower right", fontsize=6)

save(fig, "figs/fig08_sobol_chi2")
