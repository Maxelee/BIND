"""The off-manifold S8 result as one figure: within-family calibration
absorbs 93%, but if the deficit is real the same calibration absorbs only
4-21% and 79-96% of the S8 bias survives.

Reads wp7_cosmology/s8_offmanifold.json. House style (FIGURE_STYLE.md).

Run: python analysis/paper3a/scripts/fig_a7_offmanifold.py
Out: bind-paper3-plans/systematics-hunt/figures/08_offmanifold_s8.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
from analysis.paper3a.style import COL, W_DOUBLE, apply  # noqa: E402

apply()
OUT = Path("/mnt/home/mlee1/bind-paper3-plans/systematics-hunt/figures")
J = json.loads((Path("/mnt/ceph/users/mlee1/paper3/A/wp7_cosmology")
                / "s8_offmanifold.json").read_text())

# bars: within-family, then the four off-manifold (catalog x nu), then the
# two conservative-nuisance nu3 cases
rows = [("within TNG band\n(inject at band edge)",
         J["within_family_reference"]["absorbed_fraction"], COL["model"])]
labels_cat = {"remeasured_d1ce1133": "remeas.", "run0000_15481cba": "run0000"}
for cat, bins in J["per_bin"].items():
    for nu, r in bins.items():
        rows.append((f"off-manifold\n{labels_cat[cat]} {nu}",
                     r["absorbed_fraction_offmanifold"], COL["data"]))
for cat, scen in J["conservative_after_nuisance_nu3"].items():
    r = scen.get("sigma8_DESY3_0p776")
    if r:
        rows.append((f"if $\\sigma_8$ absorbs\n{labels_cat[cat]} $\\nu$3",
                     r["absorbed_fraction"], COL["variant"]))


def main():
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.2))
    x = np.arange(len(rows))
    for i, (lab, v, c) in enumerate(rows):
        ax.bar(i, v * 100, width=0.66, color=c, edgecolor="white", lw=0.8,
               zorder=3)
        ax.text(i, v * 100 + 2, f"{v*100:.0f}%", ha="center", fontsize=7.5)
    ax.axhline(93.4, color=COL["ref"], ls=":", lw=1.2, zorder=2)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=6.8)
    ax.set_ylabel("S8 bias absorbed by\nTNG-calibrated correction  [%]")
    ax.set_ylim(0, 104)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.tick_params(top=False, right=False)

    import matplotlib.patches as mp
    h = [mp.Patch(fc=COL["model"], label="truth inside the TNG band"),
         mp.Patch(fc=COL["data"], label="truth off-manifold (deficit real)"),
         mp.Patch(fc=COL["variant"],
                  label=r"if $\sigma_8$ absorbs part of the deficit")]
    ax.legend(handles=h, loc="center", bbox_to_anchor=(0.53, 0.66),
              fontsize=7, ncol=1, framealpha=0.95)
    ax.text(0.5, 96, "within-family claim (93%)", transform=ax.get_yaxis_transform(),
            fontsize=6.8, ha="left", va="bottom", color="#555555")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"08_offmanifold_s8.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote 08_offmanifold_s8.pdf/.png")


if __name__ == "__main__":
    main()
