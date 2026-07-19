"""p8 figure: the nu-trend of the tSZ deficit as mass-ordered feedback efficiency.

House style (paper_style). 3 panels + ratio subpanel a la Siegel fig06:
 (a) <Y_CAP>(nu) at 4' : data black points (JK+CIB), TNG solid truth line,
     twobound+Sobol grid family as a colorbar band (color = Delta ln M_gas);
     ratio subpanel D(nu) at 4' (filled) & 8' (open), line at 1, cosmology band.
 (b) feedback plane: implied f_gas,data vs M_eff, Popesso/Eckert/Sun + TNG f_gas-M.
 (c) decomposition: dln(f_gas T) split into dln f_gas + residual dln<T_e>.

Run from the paper dir with the BIND venv python:
  cd /mnt/home/mlee1/BIND-paper3b/papers/paper3b
  /mnt/home/mlee1/venvs/BIND_env/bin/python fig_scripts/fig_nu_trend_feedback_mass.py
Reads the frozen-analysis npz produced by p8work/analyze.py (post-processing only).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, save, panel_label, COLORS, TWO_COL_TALL  # noqa: E402

D_NPZ = ("/tmp/claude-2107/-mnt-home-mlee1-bind-paper3-plans/"
         "7cba0d61-0362-4dfc-a3ac-91ddd96da7b1/scratchpad/p8work/fig_data.npz")
d = np.load(D_NPZ, allow_pickle=True)

SCI = d["SCI"]                       # [1,2,3,4]
R4, R8 = 2, 4
xnu = np.arange(1, 5)                # x positions for the 4 science bins
nu_ticklab = ["[1,2)", "[2,3)", "[3,4)", "[4,12)"]

yd = d["yd"]; ym = d["ym"]; D = d["D"]; sig_lnD = d["sig_lnD"]
yd4 = yd[SCI, R4]; ym4 = ym[SCI, R4]
yerr4 = yd4 * sig_lnD[SCI, R4]                       # y_data error at 4'
D4 = D[SCI, R4]; D8 = d["D8"]
D4err = D4 * d["sig_lnD_4"]; D8err = D8 * d["sig_lnD_8"]
grid_y = d["grid_y"]; grid_dlm = d["grid_dlm"]      # (60,5,5),(60)
sob_y = d["sob_y"]; sob_dlm = d["sob_dlm"]          # (253,5,5),(253)

M_eff = d["M_eff"]
implied_fgas = d["implied_fgas"]; a_prod = d["a_prod_sci"]
fg_TNG = d["fg_TNG"]; fg_Pop = d["fg_Pop"]; fg_Eck = d["fg_Eck"]; fg_Sun = d["fg_Sun"]
dln_fgasT = d["dln_fgasT"]; dln_fgas = d["dln_fgas"]; dln_T = d["dln_T_resid"]
s_f = float(d["s_f"]); c_f = float(d["c_f"]); fbc = float(d["fbcosmic"])
hm_sel = d["hm_sel"]; fg_sel = d["fg_sel"]
suppr = d["desy3_suppr"]                              # DES-Y3 model_suppression_r [floor,top]
break_sig = float(d["break_sig4"])

setup()
fig = plt.figure(figsize=TWO_COL_TALL)
# top block: panel (a) + its ratio subpanel (tightly stacked)
gs_top = fig.add_gridspec(2, 1, height_ratios=[2.7, 1.05], hspace=0.06,
                          left=0.085, right=0.895, top=0.985, bottom=0.47)
ax_a = fig.add_subplot(gs_top[0])
ax_r = fig.add_subplot(gs_top[1], sharex=ax_a)
# bottom block: panels (b) and (c), separated by a gap from the ratio subpanel
gs_bot = fig.add_gridspec(1, 2, wspace=0.30,
                          left=0.085, right=0.895, top=0.37, bottom=0.075)
ax_b = fig.add_subplot(gs_bot[0])
ax_c = fig.add_subplot(gs_bot[1])

# ----------------------------------------------------------------- panel (a)
cmap = plt.get_cmap("cividis")
vmin, vmax = float(sob_dlm.min()), float(sob_dlm.max())
norm = plt.Normalize(vmin, vmax)
# Sobol family (faint, wide box) then twobound family (slightly stronger)
for u in range(sob_y.shape[0]):
    ax_a.plot(xnu, sob_y[u, SCI, R4], color=cmap(norm(sob_dlm[u])), lw=0.35, alpha=0.28, zorder=1)
for u in range(grid_y.shape[0]):
    ax_a.plot(xnu, grid_y[u, SCI, R4], color=cmap(norm(grid_dlm[u])), lw=0.5, alpha=0.55, zorder=2)
ax_a.plot(xnu, ym4, "-", color=COLORS["truth"], lw=1.8, zorder=5, label="TNG fiducial")
ax_a.errorbar(xnu, yd4, yerr=yerr4, fmt="o", ms=5, color="k", mfc="k",
              ecolor="k", elinewidth=1.1, capsize=2.5, zorder=6, label="data (JK+CIB)")
ax_a.set_yscale("log")
ax_a.set_ylabel(r"$\langle Y_{\rm CAP}\rangle$ at $4'$  [dimensionless]")
ax_a.set_ylim(2e-6, 1.1e-3)
ax_a.tick_params(labelbottom=False)
panel_label(ax_a, "(a)", "upper left")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_a, ax_r], pad=0.012, fraction=0.03, aspect=28)
cbar.set_label(r"$\Delta\ln M_{\rm gas}$ (grid family)")
leg_a = [Line2D([], [], color="k", marker="o", ls="none", ms=5, label="data (JK+CIB)"),
         Line2D([], [], color=COLORS["truth"], lw=1.8, label="TNG fiducial"),
         Line2D([], [], color=cmap(0.5), lw=1.0, label="twobound+Sobol grid")]
ax_a.legend(handles=leg_a, loc="lower right", fontsize=6.5, ncol=1)

# ----------------------------------------------------------------- ratio subpanel
ax_r.axhline(1.0, color="0.5", lw=0.8, ls="--")
# cosmology + sigma_pos nuisance: the deficit could be lifted by 1/suppression_r
# floor band drawn across all bins between D and D/suppr (DES-Y3 s8=0.776)
for i, (xx, dd) in enumerate(zip(xnu, D4)):
    hi = dd / suppr[0]                      # cosmology-absorbed upper edge (floor r)
    ax_r.plot([xx, xx], [dd, hi], color=COLORS["bind"], lw=4, alpha=0.20, solid_capstyle="butt", zorder=1)
ax_r.errorbar(xnu, D4, yerr=D4err, fmt="o", ms=4.5, color="k", mfc="k", capsize=2, zorder=5,
              label=r"$4'$")
ax_r.errorbar(xnu, D8, yerr=D8err, fmt="o", ms=4.5, color="k", mfc="none", capsize=2, zorder=5,
              label=r"$8'$")
ax_r.set_ylabel(r"$D=Y_{\rm data}/Y_{\rm TNG}$")
ax_r.set_ylim(0.05, 0.75)
ax_r.set_xticks(xnu); ax_r.set_xticklabels(nu_ticklab)
ax_r.set_xlabel(r"peak-significance bin  $\nu=\kappa/\sigma_\kappa$")
ax_r.set_xlim(0.6, 4.4)
ax_r.annotate("group floor  $\\sim$3.2$\\times$", xy=(2.0, 0.315), xytext=(1.15, 0.60),
              fontsize=6.5, color="0.25",
              arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))
ax_r.annotate(f"top-bin break\n({break_sig:.1f}$\\sigma$, suggestive)", xy=(4.0, 0.147),
              xytext=(3.0, 0.10), fontsize=6.5, color=COLORS["highlight"],
              arrowprops=dict(arrowstyle="->", color=COLORS["highlight"], lw=0.7))
ax_r.legend(loc="upper right", fontsize=6.5, ncol=2, handletextpad=0.3, columnspacing=0.8)
panel_label(ax_r, "", "upper left")

# ----------------------------------------------------------------- panel (b) feedback plane
Mline = np.logspace(13.3, 14.8, 60)
ax_b.plot(Mline, 2.23e-7 * Mline ** 0.39, "-", color=COLORS["highlight"], lw=1.3, label="Popesso+26")
ax_b.plot(Mline, 0.079 * (Mline / 1e14) ** 0.22, "--", color=COLORS["secondary"], lw=1.1, label="Eckert+21")
ax_b.plot(Mline, 0.0616 * (Mline / 1e13) ** 0.135, ":", color=COLORS["secondary"], lw=1.1, label="Sun+09")
ax_b.plot(Mline, 10 ** (c_f + s_f * np.log10(Mline)), "-", color=COLORS["bind"], lw=1.3, label="TNG frozen")
ax_b.axhline(fbc, color="0.4", ls="-.", lw=0.9)
ax_b.text(1.45e14, fbc * 1.03, r"$\Omega_b/\Omega_m$", fontsize=6.5, color="0.4", va="bottom")
# implied f_gas,data (from measured deficit); nu1-3 filled, nu4 flagged open
xerr = np.vstack([M_eff - 10 ** (np.log10(M_eff) - 0.3), 10 ** (np.log10(M_eff) + 0.3) - M_eff])
fgerr = implied_fgas * d["sig_lnD_4"] / a_prod
ax_b.errorbar(M_eff[:3], implied_fgas[:3], yerr=fgerr[:3], xerr=xerr[:, :3], fmt="o", ms=5,
              color="k", mfc="k", capsize=2, zorder=6, label=r"implied $f_{\rm gas,data}$ ($\nu$1-3)")
ax_b.errorbar(M_eff[3], implied_fgas[3], yerr=fgerr[3], xerr=xerr[:, 3:4], fmt="s", ms=5,
              color="k", mfc="none", capsize=2, zorder=6, label=r"$\nu$4 (flagged)")
ax_b.set_xscale("log"); ax_b.set_yscale("log")
ax_b.set_xlabel(r"$M_{\rm 500c}^{\rm eff}(\nu)$  [$M_\odot$, $\pm0.3$ dex]")
ax_b.set_ylabel(r"$f_{\rm gas,500c}$")
ax_b.set_xlim(2.5e13, 6e14); ax_b.set_ylim(3e-3, 0.2)
ax_b.legend(loc="lower left", fontsize=5.6, handletextpad=0.3, borderaxespad=0.3)
panel_label(ax_b, "(b)", "upper right")

# ----------------------------------------------------------------- panel (c) decomposition
xm = np.log10(M_eff)
mt_band = 0.13 * np.log(10)                            # +-0.13 dex M-T scatter -> ln
ax_c.axhspan(-mt_band, mt_band, color="0.6", alpha=0.18, zorder=0, label=r"$\pm0.13$ dex $M$-$T$")
ax_c.axhline(0.0, color="0.5", lw=0.7, ls="--")
ax_c.plot(xm[:3], dln_fgasT[:3], "o-", color="k", lw=1.2, ms=5, zorder=5,
          label=r"$\Delta\ln(f_{\rm gas}T)=\ln D/a$")
ax_c.plot(xm[:3], dln_fgas[:3], "s--", color=COLORS["secondary"], lw=1.2, ms=4.5, zorder=4,
          label=r"$\Delta\ln f_{\rm gas}$ (Popesso anchor)")
ax_c.plot(xm[:3], dln_T[:3], "^:", color=COLORS["highlight"], lw=1.2, ms=5, zorder=4,
          label=r"residual $\Delta\ln\langle T_e\rangle$")
# nu4 flagged: only dln_fgas (external anchor) is well-defined; the product and
# residual-T are off-scale (a_T collapse) -> show the anchor point + a label
ax_c.plot(xm[3], dln_fgas[3], "s", color=COLORS["secondary"], mfc="none", ms=4.5, zorder=4)
ax_c.annotate(r"$\nu$4 flagged" + "\n" + r"($a_T$ collapse)",
              xy=(xm[3], dln_fgas[3]), xytext=(14.18, -0.78), fontsize=5.8, color="0.3",
              arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6), ha="left", va="center")
ax_c.set_xlabel(r"$\log_{10} M_{\rm 500c}^{\rm eff}(\nu)$  [$M_\odot$]")
ax_c.set_ylabel(r"$\Delta\ln$ (data $-$ TNG)")
ax_c.set_ylim(-1.75, 0.35)
ax_c.set_xlim(13.4, 14.72)
ax_c.legend(loc="upper left", fontsize=5.6, handletextpad=0.3, borderaxespad=0.4,
            framealpha=0.9, facecolor="white", edgecolor="none").set_zorder(20)
panel_label(ax_c, "(c)", "upper right")

save(fig, "figs/fig17_nu_trend_feedback_mass")
