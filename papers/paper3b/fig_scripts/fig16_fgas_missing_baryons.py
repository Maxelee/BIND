"""fig_fgas_missing_baryons.pdf -- the group hot-gas-fraction deficit.

(a) HEADLINE: f_gas,data/f_gas,TNG read at fixed total mass through the grid's
    validated Y_CAP~M_gas<T_e> response (a_g~1). nu1-3 land at ~0.3 on the
    eROSITA/kSZ ~0.33-cosmic locus (Hadzhiyska+25, Ried Guachalla+25) and off
    TNG (=1). 4' filled / 8' open show the ratio is radius-coherent (a
    gas-content rescale, not a profile effect). Top bin nu4 faded/flagged.
(b) ABSOLUTE f_gas-M500c plane (de-scoped to nu>=3, where abundance-matching
    yields a defined M_eff): black data points sit on the Popesso+26 / Eckert+21
    observed-group locus (~0.15-0.3 cosmic), far below the TNG fiducial (grey,
    ~0.7 cosmic). Right axis = f_gas/cosmic; cosmic Omega_b/Omega_m=0.156 dashed.

S(k)/van Daalen A_mod mapping is handed to p5 (not duplicated here).

Data (frozen; no engine re-run): wp5_inference/p3_fgas_figdata.npz written by
  p3_fgas_analysis.py from the FROZEN p1 response + twobound grid + halo_scaling.
Artifact: B/wp5_inference/p3_fgas_missing_baryons.json
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B")
fd = np.load(B / "wp5_inference/p3_fgas_figdata.npz", allow_pickle=True)

NU = fd["nu_labels"].astype(float)          # [1,2,3,4]
r4, s4 = fd["fgas_ratio_4"], fd["sig_fgas_ratio_4"]
r8, s8 = fd["fgas_ratio_8"], fd["sig_fgas_ratio_8"]
Meff = fd["Meff_mean"]                        # nan for bins 1-2
fgas_d, sfg = fd["fgas_abs_data"], fd["sig_fgas_abs"]
fgas_T = fd["fgas_abs_TNG"]
cosmic = float(fd["OMB_OMM"])
hm, hfg = fd["halo_mass"], fd["halo_fgas"]

setup()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(TWO_COL[0], 3.15))
fig.subplots_adjust(wspace=0.42, bottom=0.16, top=0.95, left=0.09, right=0.90)

# ================================================================ Panel (a) ==
# eROSITA/kSZ locus band + reference line
axA.axhspan(0.28, 0.40, color=COLORS["secondary"], alpha=0.16, lw=0,
            label=r"Popesso+26 / kSZ $\sim\!1/3$")
axA.axhline(0.33, color=COLORS["secondary"], lw=1.0, ls="--",
            label=r"eROSITA/kSZ $0.33$")
axA.axhline(1.0, color=COLORS["dmo"], lw=0.9, ls=":", label="TNG (no deficit)")

hb = [0, 1, 2]        # headline nu1-3
# 4' filled, 8' open; nu4 faded
for k in range(4):
    faded = (k == 3)
    a = 0.35 if faded else 1.0
    axA.errorbar(NU[k] - 0.06, r4[k], s4[k], fmt="o", ms=5, color="k",
                 capsize=2, lw=1.0, alpha=a, zorder=5)
    axA.errorbar(NU[k] + 0.06, r8[k], s8[k], fmt="o", ms=5, mfc="none",
                 mec="k", mew=1.1, capsize=2, lw=1.0, ecolor="0.4", alpha=a, zorder=4)
axA.errorbar([], [], [], fmt="o", ms=5, color="k", label=r"$4'$ (fiducial)")
axA.errorbar([], [], [], fmt="o", ms=5, mfc="none", mec="k", label=r"$8'$ (outer)")
axA.text(4.0, r4[3] + 0.11, "nu4\nflagged", fontsize=5.3, color=COLORS["highlight"],
         ha="center", va="bottom")
axA.set_xlim(0.5, 4.6)
axA.set_ylim(0.0, 1.15)
axA.set_xticks([1, 2, 3, 4])
axA.set_xlabel(r"$\nu$ bin  (peak height)")
axA.set_ylabel(r"$f_{\rm gas,\,data}/f_{\rm gas,\,TNG}$  (fixed mass)")
axA.legend(loc="center left", fontsize=5.6, ncol=1, handlelength=1.4,
           bbox_to_anchor=(0.02, 0.60))
panel_label(axA, "(a)", loc="upper right")

# ================================================================ Panel (b) ==
# TNG f_gas(M) running median from the halo catalogue
logM = np.log10(hm)
edges = np.linspace(13.0, 14.6, 17)
cen = 0.5 * (edges[:-1] + edges[1:])
med = np.array([np.median(hfg[(logM >= edges[i]) & (logM < edges[i + 1])])
                if np.any((logM >= edges[i]) & (logM < edges[i + 1])) else np.nan
                for i in range(len(cen))])
Mgrid = 10 ** cen
axB.plot(10 ** cen, med, "-", color=COLORS["dmo"], lw=1.8, zorder=3,
         label="TNG fiducial")

# published observed-group curves
Mx = np.logspace(13.0, 14.5, 100)
axB.plot(Mx, 2.23e-7 * Mx**0.39, color=COLORS["bind"], lw=1.2, ls="-",
         label=r"Popesso+26")
axB.plot(Mx, 0.079 * (Mx / 1e14) ** 0.22, color=COLORS["secondary"], lw=1.2,
         ls="-.", label=r"Eckert+21")
axB.plot(Mx, 0.0616 * (Mx / 1e13) ** 0.135, color="#8B5FBF", lw=1.0, ls=":",
         label=r"Sun+09")
axB.axhline(cosmic, color="k", lw=0.8, ls="--", alpha=0.7)
axB.text(1.05e13, cosmic * 1.02, r"cosmic $\Omega_b/\Omega_m$", fontsize=5.4,
         va="bottom", color="k")

# data points (nu>=3): bin3 filled highlight, bin4 faded/open
lab = {2: r"data $\nu_3$", 3: r"data $\nu_4$ (flagged)"}
for k in (2, 3):
    if np.isnan(Meff[k]):
        continue
    faded = (k == 3)
    axB.errorbar(Meff[k], fgas_d[k], sfg[k], fmt="o", ms=7,
                 color=COLORS["highlight"] if not faded else "0.5",
                 mfc=COLORS["highlight"] if not faded else "none",
                 mec=COLORS["highlight"] if not faded else "0.5",
                 mew=1.3, capsize=3, lw=1.2, zorder=6, label=lab[k],
                 alpha=1.0 if not faded else 0.85)
axB.set_xscale("log")
axB.set_yscale("log")
axB.set_xlim(1e13, 3e14)
YLO, YHI = 0.006, 0.24
axB.set_ylim(YLO, YHI)
axB.set_xlabel(r"$M_{\rm eff}(\nu)\ [M_\odot]$")
axB.set_ylabel(r"$f_{\rm gas,\,500c}$")
axB.legend(loc="lower left", fontsize=5.5, handlelength=1.6,
           bbox_to_anchor=(0.02, 0.02))
panel_label(axB, "(b)", loc="upper right")

# right axis = f_gas/cosmic
axR = axB.twinx()
axR.set_yscale("log")
axR.set_ylim(YLO / cosmic, YHI / cosmic)
axR.set_ylabel(r"$f_{\rm gas}/(\Omega_b/\Omega_m)$", fontsize=7)

save(fig, "figs/fig16_fgas_missing_baryons")
