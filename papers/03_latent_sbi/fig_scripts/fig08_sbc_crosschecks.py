"""fig08_sbc_crosschecks.png -- cross-checks on the C_l-only NPE posterior.

Panels:
  (a) NPE posterior marginals (solid) vs. an independent explicit-likelihood
      `emcee` cross-check (dashed) for the three best-constrained parameters
      (smallest posterior/prior std ratio "shrink"), in unit-prior units.
  (b) Simulation-based calibration (SBC) rank histogram -- should be
      approximately uniform if the NPE posterior is well calibrated.
  (c) Per-parameter empirical 68% coverage across all 30 parameters, with the
      suite mean and the nominal 0.68 reference line.

Data (all pre-materialized caches -- no emcee/SBC/NPE re-run):
  examples/wl_latent_sbi_figs/npe_cl.npz   -> samples (24000, 30)
  examples/wl_latent_sbi_figs/emcee_cl.npz -> chain   (115200, 30)
  examples/wl_latent_sbi_figs/sbc_cl.npz   -> ranks   (120, 30)
  /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz -> param_names (30,)
      (same dataset file already read for fig10; used here only to label the
      3 best-constrained parameters by name -- pure metadata, no computation)

Source: sobol-sb35/examples/wl_latent_sbi.ipynb cell 23 (Sec. 3, SBC
  calibration), which depends on `samples`/`shrink`/`pn` from cell 16 and
  `pr = 1/sqrt(12)` (flat-prior std) from the same cell. `shrink =
  samples.std(0)/pr` is recomputed here verbatim from the cached samples --
  a one-line derived statistic, not a re-run.

Note vs. the extracted placeholder: the placeholder PNG's panel (a) shows
only the 3 solid NPE curves with no dashed emcee overlay, even though the
source cell code plots both (`ax[0].hist(mc[:,i], ..., ls="--", ...)`).
main.tex flags this as a to-be-regenerated discrepancy. The `emcee_cl.npz`
cache *does* contain a well-mixed, non-degenerate chain (verified: full
[0,1] range, std~0.26-0.27 for all 3 params), so this regeneration restores
the originally-intended dashed emcee overlay -- resolving the caption's
\\todo. See notes_fig08_sbc_crosschecks.md.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import COLORS, TWO_COL, panel_label, save, setup
from param_labels import short_label

FIGS = Path("/mnt/home/mlee1/BIND/examples/wl_latent_sbi_figs")
DS = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")

samples = np.load(FIGS / "npe_cl.npz")["samples"]          # (24000, 30) NPE posterior
mc = np.load(FIGS / "emcee_cl.npz")["chain"]                # (115200, 30) emcee chain
sbc = np.load(FIGS / "sbc_cl.npz")["ranks"]                  # (120, 30) SBC ranks

pn = [str(s) for s in np.load(DS, allow_pickle=True)["param_names"]]

pr = 1.0 / np.sqrt(12.0)                    # flat-prior std on the unit cube
shrink = samples.std(0) / pr                # posterior/prior width per param
inf3 = np.argsort(shrink)[:3]               # 3 best-constrained parameters

cov68 = ((sbc > 0.16) & (sbc < 0.84)).mean(0)

setup()
import matplotlib.pyplot as plt  # noqa: E402

cat_colors = [COLORS["bind"], COLORS["highlight"], COLORS["secondary"]]

fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], TWO_COL[1]), constrained_layout=True)

# (a) NPE (solid) vs emcee (dashed) marginals, 3 best-constrained params
for j, i in enumerate(inf3):
    c = cat_colors[j]
    ax[0].hist(samples[:, i], bins=40, density=True, histtype="step", color=c,
               label=short_label(pn[i]))
    ax[0].hist(mc[:, i], bins=40, density=True, histtype="step", color=c, linestyle="--")
ax[0].axvline(0.5, color="k", ls=":", lw=0.8)
ax[0].set_xlabel(r"$\theta$ (unit-prior)")
ax[0].set_ylabel("posterior density")
ax[0].text(0.97, 0.96, "solid: NPE\ndashed: emcee", transform=ax[0].transAxes,
           ha="right", va="top", fontsize=6)
ax[0].legend(loc="upper left")
# panel tag in the lower-left corner (not upper-left) so it doesn't sit on
# top of the legend's first color swatch/label (cf. fig02b's same fix)
panel_label(ax[0], "(a)", loc="lower left")

# (b) SBC rank histogram
ax[1].hist(sbc.ravel(), bins=20, color=COLORS["bind"], alpha=0.85)
ax[1].axhline(sbc.size / 20, color=COLORS["highlight"], ls="--", lw=1.2)
ax[1].set_xlabel("posterior rank of truth")
ax[1].set_ylabel("count")
panel_label(ax[1], "(b)")

# (c) per-parameter 68% coverage
ax[2].bar(np.arange(30), cov68, color=COLORS["secondary"])
ax[2].axhline(0.68, color=COLORS["highlight"], ls="--", lw=1.2)
ax[2].axhline(cov68.mean(), color="k", ls=":", lw=0.8)
ax[2].text(0.97, 0.05, f"mean = {cov68.mean():.2f}", transform=ax[2].transAxes,
           ha="right", va="bottom", fontsize=6.5)
ax[2].set_xlabel("param index")
ax[2].set_ylabel("68\\% coverage")
panel_label(ax[2], "(c)")

save(fig, "figs/fig08_sbc_crosschecks")

print(f"3 best-constrained params (shrink): {[pn[i] for i in inf3]} = "
      f"{np.round(shrink[inf3], 2).tolist()}")
print(f"mean 68% coverage = {cov68.mean():.2f}")
