#!/usr/bin/env python
"""p4 Discussion figure -- self-similar / entropy baseline & specific-energy deficit.

Four panels (Deploy-as sec4 of projectB/discussion-plans/p4-selfsimilar-entropy-baseline.md):
  (a) Y-M plane with data/TNG ratio subpanel (twobound grid family colored by dln Mgas)
  (b) specific-energy deficit dln(f_gas*T) + f_gas/<T_e> split
  (c) non-thermal radial test (data/model rises with theta; predicted <f_th> falls)
  (d) entropy central-contrast C_extra(nu)

Loads (READ-ONLY): the frozen p4 JSON artifact + the frozen twobound grid npz.
No engine, no re-fit. Run: /mnt/home/mlee1/venvs/BIND_env/bin/python fig_scripts/p4_selfsimilar_entropy_baseline.py
"""
import sys, json
import numpy as np
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, save, panel_label, COLORS
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

B = "/mnt/home/mlee1/ceph/paper3/B"
ART = f"{B}/wp5_inference/p4_selfsimilar_entropy_baseline.json"
OUT = "/mnt/home/mlee1/BIND-paper3b/papers/paper3b/figs/fig12_selfsimilar_entropy_baseline"

art = json.load(open(ART))
mg = np.load(f"{B}/wp4_mocks/model_grid_tfwiener.npz")

BINS = [1, 2, 3, 4]
radii = np.array(art["conventions"]["cap_radii_arcmin"])      # [2,3,4,6,8]
R4 = 2
D = np.array(art["deficit_matrix"]["D"])                       # (5,5)
sigD = np.array(art["deficit_matrix"]["sigma_D"])
Meff = np.array(art["M_eff_nu"]["nominal_fallback"], float)    # [1e13,3e13,6e13,1e14]
Ydata = np.array(art["selfsimilar_curves"]["Y_data_4am_per_nu"])   # per-nu (5)
Ytng = np.array(art["selfsimilar_curves"]["Y_tng_4am_per_nu"])
dfgt = np.array([art["required_dln_fgasT"]["per_bin_4arcmin"][str(b)] for b in BINS])
dfgt_sig = np.array([art["required_dln_fgasT"]["sigma"][str(b)] for b in BINS])
Cextra = np.array([art["entropy_floor"]["C_extra"][str(b)] for b in BINS])
fth_rep = np.array(art["nonthermal_radial_test"]["predicted_fth_representative_M3e13"])
# f_gas anchor band (data/TNG): Popesso 0.30 .. Eckert 0.45
anchor_lo, anchor_hi = np.log(0.30), np.log(0.45)
anchor_ctr = np.log(np.sqrt(0.30 * 0.45))
dlnTe_ctr = dfgt - anchor_ctr                                  # residual <T_e> at central anchor

setup()
fig = plt.figure(figsize=(7.2, 6.1))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)

# ---------------------------------------------------------------- (a) Y-M plane + ratio
gsa = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
axa = fig.add_subplot(gsa[0]); axr = fig.add_subplot(gsa[1], sharex=axa)
# twobound grid family: Y at 4' per bin, colored by dln Mgas (viridis, plan override)
yg = mg["y_mean"][:, :, R4]                                    # (60,5)
dg = mg["delta_ln_mgas"]
norm = Normalize(dg.min(), dg.max()); cmap = plt.get_cmap("viridis")
for b in BINS:
    xj = Meff[b - 1] * np.exp(np.random.default_rng(b).normal(0, 0.03, len(dg)))
    axa.scatter(xj, yg[:, b] * 1e6, c=dg, cmap=cmap, norm=norm, s=5, alpha=0.55,
                linewidths=0, rasterized=True)
axa.plot(Meff, Ytng[1:] * 1e6, "-", color=COLORS["dmo"], lw=1.3, zorder=4, label="TNG")
axa.plot(Meff, Ydata[1:] * 1e6, "o-", color=COLORS["truth"], ms=4, lw=1.3, zorder=6, label="data")
# self-similar 5/3 & Battaglia 1.72 anchored at TNG top point
Mr = np.array([Meff[0] * 0.8, Meff[-1] * 1.25])
axa.plot(Mr, Ytng[4] * 1e6 * (Mr / Meff[-1]) ** (5 / 3), ":", color=COLORS["highlight"],
         lw=1.0, label=r"$M^{5/3}$")
axa.plot(Mr, Ytng[4] * 1e6 * (Mr / Meff[-1]) ** 1.72, "--", color=COLORS["secondary"],
         lw=0.9, label=r"$M^{1.72}$")
axa.set_xscale("log"); axa.set_yscale("log")
axa.set_ylabel(r"$Y_{\rm CAP}(4')\ [10^{-6}]$")
axa.legend(loc="lower right", fontsize=5.5, ncol=2, handlelength=1.4, columnspacing=0.8)
plt.setp(axa.get_xticklabels(), visible=False)
panel_label(axa, "(a)")
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=axa, pad=0.02, fraction=0.05, aspect=18)
cb.set_label(r"$\Delta\ln M_{\rm gas}$", fontsize=6); cb.ax.tick_params(labelsize=5)
# ratio subpanel: data/TNG @4' (filled) and @8' outer amplitude (open)
axr.plot(Meff, D[1:, R4], "o-", color=COLORS["truth"], ms=3.5, lw=1.0, label=r"$4'$")
axr.plot(Meff, D[1:, 4], "s--", mfc="none", color=COLORS["truth"], ms=3.5, lw=0.8, label=r"$8'$")
axr.axhline(1.0, color="0.6", lw=0.5)
axr.set_xscale("log"); axr.set_ylim(0, 0.75)
axr.set_xlabel(r"$M_{\rm eff}\ [M_\odot]$ (prior-cond.)"); axr.set_ylabel(r"data/TNG", fontsize=6)
axr.legend(loc="upper left", fontsize=5.2, ncol=2, handlelength=1.1, columnspacing=0.7)

# ---------------------------------------------------------------- (b) deficit & split
axb = fig.add_subplot(gs[0, 1])
axb.axhspan(anchor_lo, anchor_hi, color=COLORS["bind"], alpha=0.18, lw=0,
            label=r"$\Delta\ln f_{\rm gas}$ (0.30-0.45)")
axb.axhspan(-0.30, 0.30, color=COLORS["secondary"], alpha=0.13, lw=0,
            label=r"$M$-$T$ scatter $\pm0.30$")
axb.errorbar(Meff, dfgt, yerr=dfgt_sig, fmt="o-", color=COLORS["truth"], ms=4, lw=1.2,
             capsize=2, label=r"$\Delta\ln(f_{\rm gas}T)$")
axb.plot(Meff, dlnTe_ctr, "D--", color=COLORS["highlight"], ms=3.5, lw=1.0,
         label=r"residual $\Delta\ln\langle T_e\rangle$")
axb.axhline(0, color="0.6", lw=0.5)
axb.set_xscale("log"); axb.set_xlabel(r"$M_{\rm eff}\ [M_\odot]$ (prior-cond.)")
axb.set_ylabel(r"$\Delta\ln$ (dex)")
axb.legend(loc="lower left", fontsize=5.2, handlelength=1.3)
panel_label(axb, "(b)")

# ---------------------------------------------------------------- (c) non-thermal radial test
axc = fig.add_subplot(gs[1, 0])
shades = ["0.68", "0.48", "0.28", "0.0"]
for i, b in enumerate(BINS):
    axc.errorbar(radii, D[b], yerr=sigD[b], fmt="o-", color=shades[i], ms=3.2, lw=0.9,
                 capsize=1.5, label=fr"$\nu\in[{int(art['conventions']['nu_edges'][b])},"
                                    fr"{int(art['conventions']['nu_edges'][b+1])})$")
axc.plot(radii, fth_rep, "s--", color=COLORS["highlight"], ms=3.5, lw=1.3,
         label=r"non-thermal $\langle f_{\rm th}\rangle$")
axc.annotate("prediction falls", xy=(6, fth_rep[3]), xytext=(4.2, 0.90),
             fontsize=5.6, color=COLORS["highlight"],
             arrowprops=dict(arrowstyle="->", color=COLORS["highlight"], lw=0.8))
axc.annotate("data rise", xy=(7, D[3, 4]), xytext=(4.3, 0.30), fontsize=5.6,
             color=COLORS["truth"], arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
axc.set_xlabel(r"CAP radius $\theta$ [arcmin]"); axc.set_ylabel(r"data/model")
axc.set_ylim(0, 1.0)
axc.legend(loc="upper left", fontsize=5.0, ncol=2, handlelength=1.2, columnspacing=0.7)
panel_label(axc, "(c)", loc="lower right")

# ---------------------------------------------------------------- (d) entropy central-contrast
axd = fig.add_subplot(gs[1, 1])
# bins 2-4 trusted (black), bin1 flagged (grey)
axd.plot(Meff[1:], Cextra[1:], "o-", color=COLORS["truth"], ms=4.5, lw=1.2, zorder=5)
axd.plot(Meff[0], Cextra[0], "o", mfc="none", color=COLORS["dmo"], ms=5.5, zorder=5)
axd.annotate(r"bin1 flagged" "\n" r"(miscentering)", xy=(Meff[0], Cextra[0]),
             xytext=(Meff[0] * 1.05, 0.30), fontsize=5.4, color=COLORS["dmo"],
             arrowprops=dict(arrowstyle="->", color=COLORS["dmo"], lw=0.7))
axd.axhline(1.0, color="0.6", lw=0.5)
axd.text(0.50, 0.90, "deeper core-evacuation", transform=axd.transAxes, fontsize=5.4, ha="center")
axd.annotate("", xy=(0.62, 0.10), xytext=(0.62, 0.55), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
axd.set_xscale("log"); axd.set_ylim(0, 1.12)
axd.set_xlabel(r"$M_{\rm eff}\ [M_\odot]$ (prior-cond.)")
axd.set_ylabel(r"$C_{\rm extra}=D(2')/D(8')$")
panel_label(axd, "(d)", loc="upper left")

save(fig, OUT)
