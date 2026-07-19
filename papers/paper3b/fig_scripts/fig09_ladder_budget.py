"""fig09_ladder_budget.pdf — the interpretation ladder as a budget: for every
executed rung, the maximum deficit factor it could absorb (bounds drawn as
right-pointing arrows), against the observed 3.2-6.6x deficit band. The two
bounded partial absorbers (sigma_pos, cosmology) even combined fall short of
the band; every measured rung sits at <= 1.1.

Data (cached, no engine re-run; every value quotes its artifact):
  - transfer shape <=1.08:  wp5_inference/transfer_movement.json (worst 4' ratio dev)
  - m-bias = 1 (exact):     wp5_inference/mbias/mbias_summary.json
  - peak-finding convention <=1.11: wp5_inference/b5_fit_wiener{,_pg1024}.json (top-bin
                            ratio 0.150->0.166; other bins <2%)
  - star/PSF + dust:        wp3_nulls/star_dust_summary.json (worst excision 0.71
                            sigma_stat; in deficit-factor units <=1.26 top bin)
  - mask holes <=1.013:     validation_closures/gap6_mask_hole_asymmetry.json
  - sigma_pos <=2.0 (mid):  wp5_inference/posscatter/posscatter_summary.json
  - cosmology <=1.52 (DES-like) / 1.81 (generous):
                            wp5_inference/b5_item6_cosmology_bracket.json
  - deficit band 3.2-6.6:   frozen data/model 0.31..0.15 (wp5 REPORT Task 3)

Original source: wp5 REPORT interpretation ladder, items 1-8 (all executed).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL_SQ, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B")
cb = json.loads((B / "wp5_inference/b5_item6_cosmology_bracket.json").read_text())
cos_des = max(cb["scenarios"]["DES-Y3-like 0.776"]["deficit_factor_absorbed"])
cos_gen = max(cb["scenarios"]["generous 0.76"]["deficit_factor_absorbed"])
g6 = json.loads((B / "validation_closures/gap6_mask_hole_asymmetry.json").read_text())
holes = 1.0 + g6["worst_case_bias_if_excised_carry_3x"]

rows = [  # (label, value, is_bound, note)
    ("shear $m$-bias", 1.0, False, "exact"),
    ("mask holes", holes, True, ""),
    ("transfer shape", 1.08, True, ""),
    ("peak-finding convention", 1.11, False, ""),
    ("star/PSF + dust", 1.26, True, ""),
    ("cosmology ($\\sigma_8$)", cos_des, True, "DES-like"),
    ("position scatter $\\sigma_{\\rm pos}$", 2.0, True, "mid bins"),
    ("$\\sigma_{\\rm pos}\\times\\sigma_8$ combined", 2.0 * cos_des, True, ""),
]

setup()
fig, ax = plt.subplots(figsize=ONE_COL_SQ)
ax.axvspan(3.2, 6.6, color=COLORS["highlight"], alpha=0.18, lw=0)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.text(4.6, len(rows) - 0.75, "observed deficit", color=COLORS["highlight"],
        fontsize=6.5, ha="center")
ax.axvline(1.0, color=COLORS["dmo"], lw=0.7)

yy = np.arange(len(rows))[::-1]
for y0, (label, val, bound, note) in zip(yy, rows):
    ax.plot([1.0, val], [y0, y0], color=COLORS["bind"], lw=1.1)
    if bound:
        ax.plot(val, y0, marker=4, ms=5, color=COLORS["bind"])  # right arrow: <= bound
    else:
        ax.plot(val, y0, "o", ms=3.5, color=COLORS["bind"])
    if note:
        ax.text(val * 1.06, y0, note, fontsize=5.5, va="center", color=COLORS["dmo"])
ax.plot([], [], marker=4, ms=5, color=COLORS["bind"], ls="none", label="upper bound")
ax.plot([], [], "o", ms=3.5, color=COLORS["bind"], ls="none", label="measured")

ax.set_yticks(yy, [r[0] for r in rows], fontsize=6.5)
ax.set_xscale("log")
ax.set_xlim(0.93, 8)
ax.set_xticks([1, 2, 3, 4, 6, 8], ["1", "2", "3", "4", "6", "8"])
ax.set_xlabel("deficit factor the rung can absorb")
ax.legend(loc="lower right", fontsize=6)

save(fig, "figs/fig09_ladder_budget")
