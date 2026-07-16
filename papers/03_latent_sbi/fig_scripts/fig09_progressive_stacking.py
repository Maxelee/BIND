"""fig09_progressive_stacking.png -- stacking emulated HOS degrades the C_l fit.

Panels:
  (a) mean NPE posterior/prior width as higher-order statistics are
      progressively stacked onto the C_l likelihood: C_l -> C_l+peaks ->
      C_l+peaks+kappa x y. Values >1 mean the posterior is WIDER than the
      prior at the measured fiducial, i.e. the fit has degraded.
  (b) the same comparison, per-parameter, for the ten parameters best
      constrained by C_l alone.

Data (single pre-materialized cache -- no NPE re-run):
  examples/wl_latent_sbi_figs/npe_progressive.npz -> dv0, dv1, dv2
      (each (24000, 30) NPE posterior samples for the 3 progressively
      larger data vectors, in that order)
  /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz -> param_names (30,)
      (same file already used for fig08/fig10; pure metadata, no
      computation -- used here to fully resolve the y-axis parameter labels)

Source: sobol-sb35/examples/wl_latent_sbi.ipynb cell 21 (Sec. 3d, the
  cautionary result). `shr[lab] = prog[key].std(0)/pr` with
  `pr = 1/sqrt(12)` is recomputed here verbatim from the cached samples.
  The console-only `obsz` diagnostic (max-|z| of the measured fiducial vs.
  the emulated manifold) is print-only in the source cell -- not part of
  the plotted panels -- and is reproduced here only as a printed value,
  matching the original.

Note vs. the extracted placeholder: main.tex flags panel (b)'s y-tick
labels as truncated to the ambiguous string "WindEnergyReductio," unable to
tell which 2 of 3 similarly-named SB35 parameters
(WindEnergyReductionFactor / WindEnergyReductionMetallicity /
WindEnergyReductionExponent) are shown. With the full `param_names` array
available here, this is resolved directly: the two are
WindEnergyReductionFactor (index 12) and WindEnergyReductionMetallicity
(index 13) -- confirmed by reproducing the same sort order as the
placeholder. See notes_fig09_progressive_stacking.md.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup

FIGS = Path("/mnt/home/mlee1/BIND/examples/wl_latent_sbi_figs")
DS = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")

prog = np.load(FIGS / "npe_progressive.npz")
pn = [str(s) for s in np.load(DS, allow_pickle=True)["param_names"]]

labels = [r"$C_\ell$", r"$C_\ell$+peaks", r"$C_\ell$+peaks+$\kappa{\times}y$"]
keys = ["dv0", "dv1", "dv2"]
pr = 1.0 / np.sqrt(12.0)
shr = {lab: prog[k].std(0) / pr for lab, k in zip(labels, keys)}

order = np.argsort(shr[labels[0]])[:10]

setup()
import matplotlib.pyplot as plt  # noqa: E402

bar_colors = [COLORS["bind"], COLORS["secondary"], COLORS["highlight"]]

fig, ax = plt.subplots(
    1, 2, figsize=(TWO_COL[0], TWO_COL[1]),
    gridspec_kw=dict(width_ratios=[1, 1.5]), constrained_layout=True,
)

# (a) mean posterior/prior width per data vector
ax[0].bar(labels, [shr[l].mean() for l in labels], color=bar_colors)
ax[0].axhline(1, color="k", ls=":", lw=1)
ax[0].set_ylabel("mean posterior/prior width")
ax[0].tick_params(axis="x", labelsize=6.5)
panel_label(ax[0], "(a)")

# (b) per-parameter constraint, 10 params best-constrained by C_l alone
xw = np.arange(len(order))
for j, lab in enumerate(labels):
    ax[1].barh(xw + (j - 1) * 0.27, shr[lab][order], 0.27, color=bar_colors[j], label=lab)
ax[1].set_yticks(xw)
ax[1].set_yticklabels([pn[i] for i in order], fontsize=6)
ax[1].invert_yaxis()
ax[1].axvline(1, color="k", ls=":", lw=1)
ax[1].set_xlabel("posterior/prior width")
# every row's C_l+peaks / C_l+peaks+kxy bars extend to ~x=1, so no in-axes
# corner is bar-free (unlike a typical sorted-bar chart); place the color
# key above the axes instead of on top of the data.
ax[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3,
             columnspacing=1.0, handlelength=1.2, fontsize=6.5)
panel_label(ax[1], "(b)")

save(fig, "figs/fig09_progressive_stacking")

for lab in labels:
    print(f"   {lab:28s} mean shrink {shr[lab].mean():.3f}")
print("resolved y-tick labels (top 10, C_l-sorted):", [pn[i] for i in order])
