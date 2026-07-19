"""fig15_mt_plane_reconstruction.pdf -- placing the kappa-peak-selected group
stack on the observed f_gas-M500 and M-T planes via the model-relative
reconstruction (plan p2).

Task override honoured: build the RELATIVE placement (RATIOS); absolute keV/f_gas
are PRIOR-CONDITIONAL and de-scoped to nu>=3; Toptun/Lovisari M-T overlaid as
prior-conditional context, flagged.

(a) f_gas-M500 plane: TNG running f_gas (line) vs Popesso+26 / Eckert+21 (lines)
    and cosmic f_b; data f_gas,data(nu>=3) = f_gas,TNG(M_eff) x anchor[0.33,0.50]
    lands on the eROSITA/Popesso ~0.3x-cosmic locus. nu1,nu2 railed (n_pk exceeds
    the 33678-halo catalog -> M_eff below the 10^13 floor); nu4 flagged.
(b) M-T plane: TNG running kT (line), Toptun+25 + 0.13dex + self-similar T~M^2/3
    (PRIOR-CONDITIONAL context); data kT_data(nu3)=kT_TNG(M_eff) x (T_data/T_TNG)
    near-self-similar; nu4 T EXCLUDED (a_T=0.37). Absolute keV is prior-conditional.
(c) anchor-free displacement plane: the required dln(f_gas.T) puts every science
    bin ~7-11x beyond the twobound ejection corner (grid window box).
(d) abundance-match construction: peak cumulative n(>nu) vs halo n(>M)/A_lc; nu>=3
    intersect the catalog (recoverable M_eff), nu1,nu2 exceed it (de-scoped).

Data (frozen; no engine re-run): B/wp5_inference/p2_mt_figdata.npz written by
  scratch p2_analysis.py from wp2 stack / wp4 grids+fiducial+halo_scaling / p1 JSON.
Artifact: B/wp5_inference/p2_mt_reconstruction.json (+ p2_Meff_zeff_nu.json)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B")
fd = np.load(B / "wp5_inference/p2_mt_figdata.npz")

NU = fd["nu"]
railed = fd["railed"]
logMeff = fd["logM_eff"]           # nan for nu1,nu2
Meff = 10.0 ** logMeff
logMfloor = float(fd["logMfloor"])
fgasTNG, kTTNG = fd["fgasTNG"], fd["kTTNG"]
fgas_data_c, fgas_data_cons = fd["fgas_data_c"], fd["fgas_data_cons"]
foc_c, foc_cons = fd["fgas_over_cos_c"], fd["fgas_over_cos_cons"]
kT_data_033, kT_data_050 = fd["kT_data_033"], fd["kT_data_050"]
Tr033, Tr050 = fd["Tratio_033"], fd["Tratio_050"]
D4 = fd["D4"]
y_data_4, y_TNG_4, sig_tot_4 = fd["y_data_4"], fd["y_TNG_4"], fd["sig_tot_4"]
dln_fgasT, factor_beyond = fd["dln_fgasT"], fd["factor_beyond"]
dlnTe_033 = fd["dlnTe_033"]
dln_fgas_c, dln_fgas_cons = float(fd["dln_fgas_c"]), float(fd["dln_fgas_cons"])
gwm, gwt = fd["grid_win_mgas"], fd["grid_win_t"]
Mgrid = fd["Mgrid"]
cpop, geck, ctop = fd["curve_popesso"], fd["curve_eckert"], fd["curve_toptun"]
Mc, fgas_run, kT_run = fd["Mc"], fd["fgas_run"], fd["kT_run"]
COSMIC_FB = float(fd["cosmic_fb"])
tscat = float(fd["toptun_scatter_dex"])
n_pk_cum, N_req, Nhalo = fd["n_pk_cum"], fd["N_req"], int(fd["Nhalo"])
A_lc = float(fd["A_lc"])
halo_logM = fd["halo_logM"]

# discrete nu colours (match fig11 family)
NUC = [COLORS["bind"], COLORS["secondary"], COLORS["highlight"], "0.45"]
FLOOR = logMfloor

setup()
fig = plt.figure(figsize=(7.2, 8.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1.18, 1.18, 1.0], hspace=0.30, wspace=0.28)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, :], sharex=axA)
axC = fig.add_subplot(gs[2, 0])
axD = fig.add_subplot(gs[2, 1])

XLO, XHI = 12.85, 14.55

# ============================================================ (a) f_gas-M plane
lgM = np.log10(Mgrid)
axA.plot(lgM, cpop, "--", color=COLORS["truth"], lw=1.3, label="Popesso+26")
axA.plot(lgM, geck, ":", color=COLORS["truth"], lw=1.3, label="Eckert+21")
axA.plot(np.log10(Mc), fgas_run, "-", color=COLORS["bind"], lw=1.8, label="TNG (this box)")
axA.axhline(COSMIC_FB, color="0.55", lw=1.0, ls=(0, (5, 2)))
axA.text(XHI - 0.02, COSMIC_FB * 1.012, r"cosmic $f_b=\Omega_b/\Omega_m$", fontsize=6,
         color="0.4", va="bottom", ha="right")
# eROSITA/Popesso data locus band 0.2-0.45 x cosmic
axA.axhspan(0.2 * COSMIC_FB, 0.45 * COSMIC_FB, color=COLORS["highlight"], alpha=0.08, lw=0)
axA.text(XHI - 0.02, 0.32 * COSMIC_FB, r"eROSITA/Popesso $0.2$–$0.45\,f_b$", fontsize=5.6,
         color=COLORS["highlight"], ha="right", va="center")
# railed region
axA.axvspan(XLO, FLOOR, color="0.85", alpha=0.6, lw=0)
axA.text(FLOOR - 0.02, 0.155, r"$\nu_1,\nu_2$ railed" + "\n" + r"($n_{\rm pk}>$ catalog)",
         fontsize=5.6, color="0.4", ha="right", va="top")
# data points nu3 (filled), nu4 (flagged)
for b, mk, fill in [(2, "o", True), (3, "s", False)]:
    if not np.isfinite(logMeff[b]):
        continue
    lo, hi = fgas_data_c[b], fgas_data_cons[b]
    axA.plot([logMeff[b], logMeff[b]], [lo, hi], "-", color=NUC[b], lw=3.0,
             alpha=0.35, solid_capstyle="round", zorder=3)
    axA.plot(logMeff[b], fgas_data_c[b], mk, ms=7, mfc=(NUC[b] if fill else "none"),
             mec=NUC[b], mew=1.6, zorder=5,
             label=(r"data $\nu_3$" if b == 2 else r"data $\nu_4$ (flagged)"))
axA.set_ylabel(r"$f_{\rm gas,500}$")
axA.set_ylim(0.0, 0.175)
axA.set_xlim(XLO, XHI)
axA.legend(loc="lower right", fontsize=6, ncol=2, handlelength=1.6)
axA.tick_params(labelbottom=False)
panel_label(axA, r"(a)  $f_{\rm gas}$–$M_{500}$", loc="lower left")
axA.text(0.5, 0.05, r"anchor bar $=[0.33,0.50]\times f_{\rm gas,TNG}$ (prior-conditional)",
         transform=axA.transAxes, fontsize=5.6, ha="center", va="bottom", color="0.35")

# ============================================================ (b) M-T plane
axB.plot(lgM, ctop, "-", color=COLORS["truth"], lw=1.5, label="Toptun+25 M–T")
tband = tscat / 1.65    # 0.13 dex M-scatter -> dex in T along the relation
axB.fill_between(lgM, ctop * 10 ** (-tband), ctop * 10 ** (tband), color=COLORS["truth"],
                 alpha=0.13, lw=0, label=r"$\pm0.13$ dex")
# self-similar T ~ M^{2/3} anchored to Toptun at 10^13.5
Manch = 13.5
Tanch = 10 ** ((Manch - 13.38) / 1.65)
ss = Tanch * 10 ** ((2.0 / 3.0) * (lgM - Manch))
axB.plot(lgM, ss, "-.", color="0.5", lw=1.1, label=r"self-similar $T\propto M^{2/3}$")
axB.plot(np.log10(Mc), kT_run, "-", color=COLORS["bind"], lw=1.8, label="TNG (this box)")
axB.axvspan(XLO, FLOOR, color="0.85", alpha=0.6, lw=0)
# data nu3 (filled), nu4 (T EXCLUDED -> crossed)
b = 2
lo, hi = kT_data_050[b], kT_data_033[b]
axB.plot([logMeff[b], logMeff[b]], [lo, hi], "-", color=NUC[b], lw=3.0, alpha=0.35,
         solid_capstyle="round", zorder=3)
axB.plot(logMeff[b], kT_data_033[b], "o", ms=7, mfc=NUC[b], mec=NUC[b], mew=1.6,
         zorder=5, label=r"data $\nu_3$")
b = 3
axB.plot(logMeff[b], kT_data_033[b], "x", ms=7, color=NUC[b], mew=1.8, zorder=5,
         label=r"$\nu_4$: $T$ excluded ($a_T$=0.37)")
axB.set_yscale("log")
axB.set_ylabel(r"$k_BT\ [{\rm keV}]$")
axB.set_ylim(0.35, 3.2)
axB.set_xlabel(r"$\log_{10}(M_{500,\rm eff}/M_\odot)$")
axB.legend(loc="upper left", fontsize=6, handlelength=1.7)
panel_label(axB, "(b)  M–T plane", loc="lower right")
axB.text(0.5, 0.045, "absolute keV PRIOR-CONDITIONAL (open y-map calib guard); lead with ratio",
         transform=axB.transAxes, fontsize=5.4, ha="center", va="bottom", color="0.4")

# ============================================================ (c) displacement plane
axC.add_patch(Rectangle((gwm[0], gwt[0]), gwm[1] - gwm[0], gwt[1] - gwt[0],
                        facecolor=COLORS["dmo"], alpha=0.4, edgecolor=COLORS["truth"],
                        lw=1.0, zorder=2))
axC.text(0.0, gwt[1] + 0.02, "twobound\ngrid window", fontsize=5.6, ha="center",
         va="bottom", color=COLORS["truth"])
# anchor-free product constraint: dln M_gas + dln T = dln(f_gas T)[b]  (diagonal line)
xx = np.array([-2.6, 0.4])
for k, b in enumerate(range(4)):
    yy = dln_fgasT[b] - xx
    axC.plot(xx, yy, "-", color=NUC[k], lw=1.1, alpha=0.85,
             label=rf"$\nu_{b+1}$")
    # 0.33-anchor point (dln f_gas, dln T_e)
    axC.plot(dln_fgas_c, dlnTe_033[b], "o", ms=5, mfc=NUC[k], mec="k", mew=0.5, zorder=5)
axC.axvline(dln_fgas_c, color="0.6", ls=":", lw=0.8)
axC.text(dln_fgas_c, 0.55, r"anchor $f_{\rm gas}{=}0.33$", rotation=90, fontsize=5.4,
         color="0.4", ha="right", va="top")
axC.set_xlim(-2.6, 0.4)
axC.set_ylim(-1.1, 0.7)
axC.set_xlabel(r"$\Delta\ln M_{\rm gas}\ (=\Delta\ln f_{\rm gas})$")
axC.set_ylabel(r"$\Delta\ln T$")
axC.legend(loc="lower left", fontsize=5.6, ncol=2, handlelength=1.2, columnspacing=1.0)
panel_label(axC, "(c)")
axC.text(0.97, 0.96, r"$\sim$7–11$\times$ beyond corner", transform=axC.transAxes,
         fontsize=6, ha="right", va="top", color=COLORS["highlight"])

# ============================================================ (d) abundance match
ms = np.sort(halo_logM)[::-1]
nh = (np.arange(1, len(ms) + 1)) / A_lc           # cumulative n_halo(>M) per deg2
axD.plot(ms, nh, "-", color=COLORS["bind"], lw=1.6, label=r"halo $n(>M)/A_{\rm lc}$")
nh_max = nh.max()
axD.axhspan(nh_max, n_pk_cum.max() * 1.6, color="0.85", alpha=0.6, lw=0)
axD.text(13.05, nh_max * 2.4, "railed:\n$n_{\\rm pk}>$ catalog", fontsize=5.4, color="0.4",
         va="center")
for k in range(4):
    axD.axhline(n_pk_cum[k], color=NUC[k], ls="--", lw=1.0, alpha=0.9)
    if np.isfinite(logMeff[k]):
        axD.plot(np.log10(Meff[k]), n_pk_cum[k], "o", ms=5, mfc=NUC[k], mec="k", mew=0.5,
                 zorder=5)
        axD.annotate(rf"$\nu_{k+1}$", (np.log10(Meff[k]), n_pk_cum[k]), fontsize=6,
                     color=NUC[k], xytext=(3, 3), textcoords="offset points")
    else:
        axD.text(14.05, n_pk_cum[k] * 1.12, rf"$\nu_{k+1}$", fontsize=6, color=NUC[k],
                 va="bottom", ha="center")
axD.set_yscale("log")
axD.set_xlabel(r"$\log_{10}(M_{500}/M_\odot)$")
axD.set_ylabel(r"$n(> \cdot)\ [{\rm deg^{-2}}]$")
axD.set_xlim(13.0, 14.5)
axD.set_ylim(5e-3, n_pk_cum.max() * 1.6)
axD.legend(loc="lower left", fontsize=5.8)
panel_label(axD, "(d)", loc="upper right")

save(fig, "figs/fig15_mt_plane_reconstruction")
