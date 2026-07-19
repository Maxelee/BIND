"""fig11_ykappa_specific_energy.pdf -- the group tSZ deficit read as a
specific-thermal-energy (Y/kappa ~ f_gas <T_e>) deficit via the MODEL grid's
fixed-kappa response.

(a) grid response a_g (solid) / a_T (dashed) vs CAP radius, one pair per science
    bin: a_g~a_T~1 at 4' validates Y = M_gas * T for our CAP geometry (top bin
    faded -- a_T collapses, flagged).
(b) the measurement at 4': <Y_CAP>(nu) data (black) vs TNG fiducial (line) and
    the whole twobound family (thin, colored by dln(f_gas T)); ratio subpanel
    D=data/model sits far below the model manifold band (the corner the
    posterior piles into) -- the chi2~340 rejection in physical units.
(c) f_gas/T split: residual <T_e>_data/<T_e>_TNG per nu bin for the anchor
    bracket [0.33 filled, 0.50 open], with the near-self-similar M-T scatter
    band; nu1-3 close at near-self-similar T, top bin breaks (highlight).
(d) two-component decomposition: outer f_gas floor A=D(8') vs extra-central
    contrast C=D(2')/D(8') -- opposite mass trends.

Data (frozen; no engine re-run): figure-data npz written by p1_analysis.py from
  wp2 stack / wp4 twobound grid + TNG fiducial / wp3 sigma_sys.
Artifact: B/wp5_inference/p1_ykappa_specific_energy.json
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL_TALL, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B")
fd = np.load(B / "wp5_inference/p1_ykappa_figdata.npz", allow_pickle=True)

RAD = fd["radii_arcmin"]
NU = fd["nu_bins"]                 # [1,2,3,4]
ag, aT, r2 = fd["ag"], fd["aT"], fd["r2"]
y_data, y_TNG, y_grid = fd["y_data"], fd["y_TNG"], fd["y_grid"]
dg, dt = fd["dg"], fd["dt"]
dln_grid = fd["dln_fgasT_grid"]    # dg+dt per twobound unit
sig_tot_4 = fd["sig_tot_4"]
D, lnD, siglnD = fd["D"], fd["lnD"], fd["siglnD"]
Tr_c, Tr_cons, sTr_c = fd["Tratio_c"], fd["Tratio_cons"], fd["sig_Tratio_c"]
A_outer, C_central, siglnC = fd["A_outer"], fd["C_central"], fd["siglnC"]
RFID, I8 = 2, 4

setup()
fig = plt.figure(figsize=TWO_COL_TALL)
outer = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)
ax_a = fig.add_subplot(outer[0, 0])
inner_b = outer[0, 1].subgridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.06)
ax_bt = fig.add_subplot(inner_b[0])
ax_bb = fig.add_subplot(inner_b[1], sharex=ax_bt)
ax_c = fig.add_subplot(outer[1, 0])
ax_d = fig.add_subplot(outer[1, 1])

nu_colors = [COLORS["bind"], COLORS["secondary"], COLORS["highlight"], COLORS["dmo"]]

# ---- (a) response validation ------------------------------------------------
for k, b in enumerate(range(1, 5)):
    faded = (b == 4)
    col = nu_colors[k]
    alpha = 0.35 if faded else 1.0
    ax_a.plot(RAD, ag[b], "-", color=col, lw=1.4, alpha=alpha,
              label=rf"$\nu_{b}$" if not faded else rf"$\nu_4$ (flagged)")
    ax_a.plot(RAD, aT[b], "--", color=col, lw=1.4, alpha=alpha)
ax_a.axhline(1.0, color=COLORS["dmo"], lw=0.8, ls=":")
ax_a.axvline(4.0, color=COLORS["dmo"], lw=0.6, ls=":", alpha=0.7)
ax_a.annotate(r"$a_g$", xy=(6.0, ag[1, 3]), xytext=(6.3, ag[1, 3] + 0.18),
              fontsize=7, color=COLORS["bind"])
ax_a.annotate(r"$a_T$", xy=(6.0, aT[1, 3]), xytext=(6.3, aT[1, 3] - 0.30),
              fontsize=7, color=COLORS["bind"])
ax_a.set_xlabel(r"$\theta\ [{\rm arcmin}]$")
ax_a.set_ylabel(r"$\partial\ln\langle Y_{\rm CAP}\rangle/\partial\Delta\ln X$")
ax_a.set_ylim(-0.1, 2.7)
ax_a.legend(loc="upper right", fontsize=6, ncol=2, handlelength=1.3)
panel_label(ax_a, "(a)")

# ---- (b) measurement + ratio subpanel --------------------------------------
xnu = NU.astype(float)
norm = Normalize(vmin=dln_grid.min(), vmax=dln_grid.max())
cmap = plt.get_cmap("cividis")
for u in range(y_grid.shape[0]):
    ax_bt.plot(xnu, y_grid[u, 1:5, RFID], "-", lw=0.4,
               color=cmap(norm(dln_grid[u])), alpha=0.55, zorder=1)
ax_bt.plot(xnu, y_TNG[1:5, RFID], "-", color=COLORS["truth"], lw=1.6,
           zorder=3, label="TNG fiducial")
ax_bt.errorbar(xnu, y_data[1:5, RFID], sig_tot_4[1:5], fmt="o", ms=4,
               color="k", capsize=2, lw=1.0, zorder=4, label="data")
ax_bt.set_yscale("log")
ax_bt.set_ylabel(r"$\langle Y_{\rm CAP}\rangle\ [{\rm arcmin^2}]$")
ax_bt.legend(loc="lower right", fontsize=6, bbox_to_anchor=(1.0, 0.02))
ax_bt.tick_params(labelbottom=False)
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax_bt, pad=0.02, fraction=0.05, aspect=14)
cb.set_label(r"$\Delta\ln(f_{\rm gas}T)$ grid", fontsize=6)
cb.ax.tick_params(labelsize=5)
panel_label(ax_bt, "(b)")

# ratio subpanel: D and the model-manifold band
grid_ratio = y_grid[:, 1:5, RFID] / y_TNG[1:5, RFID]
gr_lo, gr_hi = grid_ratio.min(axis=0), grid_ratio.max(axis=0)
ax_bb.fill_between(xnu, gr_lo, gr_hi, color=COLORS["dmo"], alpha=0.25, lw=0,
                   label="model manifold")
ax_bb.errorbar(xnu, D[1:5, RFID], D[1:5, RFID] * siglnD[1:5, RFID], fmt="o",
               ms=4, color="k", capsize=2, lw=1.0, zorder=4)
ax_bb.axhline(1.0, color=COLORS["dmo"], lw=0.8, ls=":")
ax_bb.set_yscale("log")
ax_bb.set_ylim(0.10, 1.5)
ax_bb.set_ylabel("data / model")
ax_bb.set_xlabel(r"$\nu$ bin")
ax_bb.set_xticks([1, 2, 3, 4])
ax_bb.annotate(r"$\Delta\ln(f_{\rm gas}T)\!\approx\!-1.2$ to $-1.9$",
               xy=(2.5, 0.52), fontsize=5.5, ha="center", color="k")

# ---- (c) f_gas/T split ------------------------------------------------------
ax_c.axhspan(np.exp(-0.30), np.exp(0.30), color=COLORS["dmo"], alpha=0.22, lw=0,
             label=r"M-T scatter ($\pm0.13$ dex)")
ax_c.axhline(1.0, color=COLORS["dmo"], lw=0.8, ls=":")
# nu1-3 vs top bin coloring for anchor 0.33
for k, b in enumerate(range(1, 5)):
    hl = (b == 4)
    mc = COLORS["highlight"] if hl else "k"
    ax_c.errorbar(xnu[k], Tr_c[b, RFID], sTr_c[b, RFID], fmt="o", ms=4.5,
                  color=mc, capsize=2, lw=1.0, zorder=4)
# anchor 0.50 as open markers
ax_c.plot(xnu, Tr_cons[1:5, RFID], "o", ms=4.5, mfc="none", mec="0.35",
          mew=1.0, zorder=3)
ax_c.plot([], [], "o", ms=4.5, color="k", label=r"anchor $f_{\rm gas}=0.33$")
ax_c.plot([], [], "o", ms=4.5, mfc="none", mec="0.35", label=r"anchor $0.50$")
ax_c.set_xlabel(r"$\nu$ bin")
ax_c.set_ylabel(r"$\langle T_e\rangle_{\rm data}/\langle T_e\rangle_{\rm TNG}$")
ax_c.set_xticks([1, 2, 3, 4])
ax_c.set_ylim(0.15, 1.35)
ax_c.legend(loc="lower left", fontsize=5.5)
panel_label(ax_c, "(c)")

# ---- (d) two-component decomposition ---------------------------------------
ax_d.plot(xnu, A_outer[1:5], "-o", ms=4, color=COLORS["bind"], lw=1.3,
          label=r"$A=D(8')$ (outer $f_{\rm gas}$ floor)")
ax_d.plot(xnu, C_central[1:5], "-s", ms=4, color=COLORS["secondary"], lw=1.3,
          label=r"$C=D(2')/D(8')$ (extra central)")
ax_d.axhline(1.0, color=COLORS["dmo"], lw=0.8, ls=":")
ax_d.set_xlabel(r"$\nu$ bin")
ax_d.set_ylabel("deficit component")
ax_d.set_xticks([1, 2, 3, 4])
ax_d.set_ylim(0.0, 1.05)
ax_d.legend(loc="upper right", fontsize=5.5)
panel_label(ax_d, "(d)")

save(fig, "figs/fig11_ykappa_specific_energy")
