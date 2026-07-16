"""fig11_lightcone_population.pdf -- validating the single-epoch physical
readout against the actual lightcone halo population.

Panels:
  (a) the z_s=1 lensing-kernel weight W(z) (grey bars) vs. lens redshift,
      with the directly-measured per-halo Born |Delta-kappa| signal from
      the fiducial BIND-vs-DMO maps overlaid (line); the old, low-redshift
      -only proxy epoch (z=0.034) is annotated with its (small) share of
      the total kernel weight.
  (b) single-epoch (z=0.034) <-> full lightcone-weighted proxy fidelity,
      per baryon feature, split into "amplitude" (f_gas, f_star, Y/M,
      T_mw, K_mw) and "redistribution" (gas concentration / ejection
      fraction) features.
  (c) the direct per-halo link: per-halo Born Delta-kappa (normalized) vs.
      halo f_gas(<r500c), for halos with M>1e13.

Data: /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/lightcone_halo_section6.npz
  (single cache, produced by examples/lightcone_halo_{catalog,observables,
  dkappa}.py -- those engines are NOT re-run here, only their cached output
  is read) -> z_shell, w_shell_frac, dk_z_shell, dk_z_frac, feature_names,
  proxy_run_corr, dk_perhalo_dkappa, dk_perhalo_fgas, dk_perhalo_M,
  dk_fgas_r.

Source: sobol-sb35/examples/paper_lightcone_figs2.ipynb cell 32 (Sec. 6.4,
  "fig_lightcone_population"). A single np.load + direct plotting, no
  recomputation. The AMP/RED feature split ("conc"/"eject" keyword match)
  and the two-color family scheme (`FAMC`) are ported from that notebook's
  cell 1 preamble.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup

H6 = np.load(
    "/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/lightcone_halo_section6.npz",
    allow_pickle=True,
)
names = [str(s) for s in H6["feature_names"]]
AMP = [i for i, n in enumerate(names) if "conc" not in n and "eject" not in n]
RED = [i for i, n in enumerate(names) if i not in AMP]

setup()
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], TWO_COL[1] + 0.4), constrained_layout=True)

# (a) where the z_s=1 lensing signal lives in redshift
z_sh, w_sh = H6["z_shell"], H6["w_shell_frac"]
ax[0].bar(z_sh, 100 * w_sh, width=0.045, color=COLORS["dmo"], label=r"kernel weight $W(z)\,M$")
ax[0].plot(H6["dk_z_shell"], 100 * H6["dk_z_frac"], "-o", ms=2.6, lw=1.1,
           color=COLORS["highlight"], label=r"per-halo $|\Delta\kappa|$ (maps)")
ax[0].axvline(0.034, color="k", lw=0.9, ls=":")
ax[0].text(0.97, 0.06, f"old §6 epoch (z=0.034):\n{100 * w_sh[0]:.0f}\\% of signal",
           transform=ax[0].transAxes, fontsize=6.2, va="bottom", ha="right")
ax[0].set_xlabel(r"lens redshift $z$")
ax[0].set_ylabel(r"\% of $z_s{=}1$ signal")
ax[0].set_xlim(0, 1.1)
ax[0].legend(loc="upper right", fontsize=6.5)
# default upper-left tag sits on the z=0.034 reference line; nudge right,
# clear of the line, without colliding with the upper-right legend.
ax[0].text(0.14, 0.96, "(a)", transform=ax[0].transAxes, ha="left", va="top",
           fontsize=8, fontweight="bold")

# (b) single-epoch <-> lightcone-weighted proxy fidelity per feature
pr = H6["proxy_run_corr"]
yy = np.arange(len(names))
ax[1].barh(yy[AMP], pr[AMP], color=COLORS["highlight"], label="amplitude")
ax[1].barh(yy[RED], pr[RED], color=COLORS["secondary"], label="redistribution")
ax[1].axvline(1.0, color="0.7", lw=0.7, ls=":")
ax[1].set_yticks(yy)
ax[1].set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=5.0)
ax[1].set_xlabel(r"$r$(z=0.034, lightcone)")
ax[1].set_xlim(0, 1.05)
ax[1].legend(loc="lower left")
panel_label(ax[1], "(b)")

# (c) per-halo Born Delta-kappa vs halo gas content
dk, fg = H6["dk_perhalo_dkappa"], H6["dk_perhalo_fgas"]
m = np.isfinite(dk) & np.isfinite(fg) & (H6["dk_perhalo_M"] > 1e13)
hb = ax[2].hexbin(fg[m], dk[m] / np.nanstd(dk[m]), gridsize=28, cmap="cividis",
                   bins="log", mincnt=1, rasterized=True)
ax[2].text(0.05, 0.92, rf"$r={float(H6['dk_fgas_r']):+.2f}$", transform=ax[2].transAxes,
           fontsize=8, va="top", color="k")
ax[2].set_xlabel(r"halo $f_{\rm gas}(<r_{500c})$")
ax[2].set_ylabel(r"Born $\Delta\kappa$ (norm.)")
panel_label(ax[2], "(c)")

save(fig, "figs/fig11_lightcone_population")

print(f"z_s=1 lensing weight at z=0.034: {100 * w_sh[0]:.1f}%   "
      f"(peak shell z={z_sh[np.argmax(w_sh)]:.2f})")
print(f"single-epoch proxy fidelity:  amplitude r={np.mean(pr[AMP]):.2f},  "
      f"redistribution r={np.mean(pr[RED]):.2f}")
print(f"per-halo Born Delta-kappa tracks f_gas at r={float(H6['dk_fgas_r']):+.2f}")
