"""fig13_shmr_scatter_budget.png -- SHMR residual-variance budget on the SB35 Sobol suite.

Balanced variance decomposition of the stellar-halo-mass-relation (SHMR)
residual r = log10(M_star) - trend(log10(M200c)) at fixed halo mass, over
the 256-point SB35 Sobol feedback design (same shared snap-096 halos painted
under 256 different 30-d astrophysics draws):
  - "feedback (Sobol prior)": within-halo variance across the Sobol prior
    (r.var(axis=0).mean()) -- dominates, ~97% of the total.
  - "halo-to-halo": variance of the Sobol-mean residual across halos
    (r.mean(axis=0).var()) -- the leftover halo-to-halo scatter at fixed
    mass, ~3%.
Dashed reference line: the twobound one-at-a-time single-knob EXTREME-bound
feedback variance (marginal, others held at fiducial, NOT an upper bound on
the joint Sobol spread) computed on the same trend/halos.

Data (read-only, cached on disk -- no recomputation):
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz
      keys: M_fof (2933,), sobol_valid (256,), sobol_m_star_500c (256,2933),
            tb_valid (60,), tb_m_star_500c (60,2933)

Source: sobol-sb35/examples/shmr_accretion_scatter_sb35.ipynb (branch
analysis/2d, worktree wt/sobol-sb35), cell 9 (builds `trend` via a quadratic
np.polyfit(log10 M200c, Sobol-mean log10 M_star) mass-trend removal -> the
residual `r`) feeding cell 11 (the scatter-budget bar chart itself).
Manifest confirms: figs_raw/shmr_accretion_scatter_sb35/manifest.json,
cell011_out1.png. No accretion-history file needed for this figure (only
S4/S5 of the notebook, not S6-S8).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ATLAS = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
APERTURE = "500c"
field = f"m_star_{APERTURE}"

cube = np.load(ATLAS, allow_pickle=True)
M = cube["M_fof"].astype(float)  # M200c [Msun/h], fixed per halo (2933,)
valid = cube["sobol_valid"]  # (256,) True where the Sobol run is complete
Mstar = cube[f"sobol_{field}"][valid].astype(float)  # (n_sobol, 2933)

# ---- cell 9 body: quadratic mass-trend removal -> residuals at fixed M ----
lM = np.log10(M)
lS = np.log10(Mstar + 1.0)
coef = np.polyfit(lM, lS.mean(0), 2)
trend = np.polyval(coef, lM)
r = lS - trend[None, :]  # (n_sobol, N) residuals at fixed M

# ---- cell 11 body: balanced variance decomposition ----
var_tot = r.var()
feedback = r.var(axis=0).mean()  # within-halo, across the Sobol prior
halo_mu = r.mean(axis=0)  # param-averaged residual per halo
halo_var = halo_mu.var()  # halo-to-halo at fixed mass

print(f"sigma_tot     = {np.sqrt(var_tot):.4f} dex")
print(f"  feedback    = {np.sqrt(feedback):.4f} dex   ({feedback / var_tot * 100:4.1f}% of variance)")
print(f"  halo-to-halo= {np.sqrt(halo_var):.4f} dex   ({halo_var / var_tot * 100:4.1f}% of variance)")

# contrast: twobound single-knob EXTREME feedback variance (marginal, same halos/trend)
tb_valid = cube["tb_valid"]
tb = cube[f"tb_{field}"][tb_valid].astype(float)
r_tb = np.log10(tb + 1.0) - trend[None, :]
fb_tb = r_tb.var(axis=0).mean()
print(f"  twobound one-at-a-time (marginal) = {np.sqrt(fb_tb):.4f} dex")

# ---- plot ----
setup()
fig, ax = plt.subplots(figsize=ONE_COL)
ax.bar(
    ["feedback\n(Sobol prior)", "halo-to-halo"],
    [feedback, halo_var],
    color=[COLORS["highlight"], COLORS["secondary"]],
)
ax.axhline(fb_tb, color=COLORS["truth"], ls="--", lw=1.3, label="twobound one-at-a-time\n(marginal)")
ax.set_ylabel(r"variance of $\log_{10}M_\star$ at fixed $M$ [dex$^2$]")
ax.legend(loc="upper right")

save(fig, "figs/fig13_shmr_scatter_budget")
