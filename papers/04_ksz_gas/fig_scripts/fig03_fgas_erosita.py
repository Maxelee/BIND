"""fig03_fgas_erosita — headline "0% coverage" figure (§5 Confrontation I).

BIND f_gas,500(M_500c) for BGS-redshift halos (0.08<z<0.45): the 16-84%
spread over the 30-d Sobol feedback space, the median, and the
2.5%-strongest-feedback edge, overlaid on the Eckert+19 X-ray and
Popesso+24 eROSITA strong-feedback f_gas(M) relations. Result: 0% of the
feedback space reaches the eROSITA strong-feedback band.

Source: examples/paper_ksz_desi_act.ipynb (branch analysis/ksz_project,
worktree .../wt/ksz-desi-act), cell 12 (heading "§5 Confrontation I"),
built by examples/_build_ksz_paper_nb.py lines ~612-634. main.tex places
this cell as figure 3 (early, headline result) though it is notebook
cell 12 / dossier position 5 (narrative order, not notebook cell order).

Data (read-only, cached on disk, no recomputation):
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet
    columns run, snap, z, M_tot_500, f_gas_500
  eckert_fgas / popesso_fgas are hardcoded analytic fits from the notebook
  (Eckert et al. 2019; Popesso et al. 2024) — no cache file needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

PARQUET = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet")

F_B = 0.0490 / 0.3089  # Omega_b/Omega_m (Planck/TNG)


def eckert_fgas(M):
    return 0.131 * (M / 2e14) ** 0.21


def popesso_fgas(M):
    return 2.23e-7 * M**0.39


setup()

df = pd.read_parquet(PARQUET, columns=["run", "snap", "z", "M_tot_500", "f_gas_500"])
sel = df[(df.z > 0.08) & (df.z < 0.45) & (df.f_gas_500 > 0)]
mb = np.linspace(13.0, 14.4, 8)
cM = 0.5 * (mb[:-1] + mb[1:])
Mc = 10**cM
sel = sel.assign(b=np.digitize(np.log10(sel.M_tot_500), mb) - 1)
g = sel.groupby(["run", "b"]).f_gas_500.median()
runs = np.sort(sel.run.unique())
stack = np.full((len(runs), len(cM)), np.nan)
for i, r in enumerate(runs):
    for b in range(len(cM)):
        if (r, b) in g.index:
            stack[i, b] = g.loc[(r, b)]
med = np.nanmedian(stack, 0)
lo, hi = np.nanpercentile(stack, [16, 84], 0)
strong = np.nanpercentile(stack, 2.5, 0)

fig, ax = plt.subplots(figsize=ONE_COL)
ax.fill_between(Mc, lo, hi, color=COLORS["bind"], alpha=0.25, label="BIND Sobol 16-84%")
ax.plot(Mc, med, "-", color=COLORS["bind"], lw=1.6, label="BIND median")
ax.plot(Mc, strong, ":", color="navy", lw=1.3, label="BIND strongest-fb (2.5%)")
ax.plot(Mc, eckert_fgas(Mc), color=COLORS["secondary"], lw=1.4, label="Eckert+19 (X-ray)")
ax.plot(Mc, popesso_fgas(Mc), color=COLORS["highlight"], lw=1.4, label="eROSITA strong-fb")
ax.axhline(F_B, color="k", ls="--", lw=0.8, label=r"$\Omega_b/\Omega_m$")
ax.set_xscale("log")
ax.set_xlabel(r"$M_{500c}\,[M_\odot/h]$")
ax.set_ylabel(r"$f_{\rm gas,500}$")
ax.set_ylim(0, F_B * 1.12)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.03), fontsize=6.3, borderaxespad=0)

save(fig, "figs/fig03_fgas_erosita")
