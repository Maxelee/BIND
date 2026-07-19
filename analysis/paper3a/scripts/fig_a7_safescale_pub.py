"""Publication figure 6: the cosmology statement — what gas calibration
buys a weak-lensing analyst.

Left  : safe scales. ell_max at the 0.3-sigma criterion for each survey,
        under an uninformed feedback prior vs the two calibrated bands.
Right : the S8 exercise. Recovered Delta_S8 under three treatments of a
        strong-side-injected feedback signal — ignoring feedback,
        calibrating it, or cutting scales.

Grammar: Bigwood Fig.-8 style grouped comparison on the left (one
physical axis, families as labeled colors), Siegel-style point-with-
error comparison on the right, both in the house style kit.

Run: python analysis/paper3a/scripts/fig_a7_safescale_pub.py
Out: wp7_cosmology/figures/a7_safescale_pub.{pdf,png}
     (+ png copy into the plans repo figures_pub/)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.cosmology.safescale import WP7  # noqa: E402
from analysis.paper3a.style import COL, W_DOUBLE, apply  # noqa: E402

PLANS_FIG = Path("/mnt/home/mlee1/bind-paper3-plans/figures_pub")
BANDS = [("prior", "uninformed prior", COL["shade"]),
         ("fgas_post", r"$f_{\rm gas}$-calibrated", COL["aux"]),
         ("joint_post", "joint A+B calibrated", COL["model"])]
TREATS = [("uncorrected", "ignore\nfeedback"),
          ("calibrated", "gas-\ncalibrated"),
          ("scale_cut", "scale\ncut")]


def main() -> None:
    apply()
    tbl = json.loads((WP7 / "safescale_tables.json").read_text())
    s8 = json.loads((WP7 / "s8_exercise.json").read_text())
    surveys = list(tbl["two_point_clkk"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.1))

    # ---- left: safe scales ------------------------------------------
    x = np.arange(len(surveys))
    w = 0.26
    for i, (key, lab, c) in enumerate(BANDS):
        v = [tbl["two_point_clkk"][s][key]["lmax_safe"] for s in surveys]
        axL.bar(x + (i - 1) * w, v, w, color=c, label=lab,
                edgecolor="k", linewidth=0.4)
    axL.set_xticks(x)
    axL.set_xticklabels(surveys, fontsize=8)
    axL.set_ylabel(r"safe $\ell_{\max}$   ($|b|/\sigma < 0.3$)")
    axL.legend(frameon=False, fontsize=7.5, loc="upper left")
    axL.set_ylim(0, 1500)

    # ---- right: the S8 exercise -------------------------------------
    cols = [COL["data"], COL["truth"], COL["variant"]]
    for j, s in enumerate(surveys):
        r = s8["surveys"][s]
        y = [r[k]["delta_S8"] for k, _ in TREATS]
        e = [r[k]["sigma_S8"] for k, _ in TREATS]
        axR.errorbar(np.arange(3) + (j - 1) * 0.13, y, yerr=e, fmt="o",
                     ms=5, capsize=2.5, color=cols[j], label=s, lw=1.2)
    axR.axhline(0.0, color=COL["ref"], ls=":", lw=1)
    axR.annotate("unbiased", xy=(2.42, 0.0), fontsize=7,
                 ha="right", va="bottom", color=COL["ref"])
    axR.set_xticks(np.arange(3))
    axR.set_xticklabels([lab for _, lab in TREATS], fontsize=8)
    axR.set_ylabel(r"recovered $\Delta S_8$")
    axR.legend(frameon=False, fontsize=7.5, loc="lower right")

    fig.tight_layout()
    out = WP7 / "figures" / "a7_safescale_pub"
    (WP7 / "figures").mkdir(exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    if PLANS_FIG.is_dir():
        shutil.copy(f"{out}.png", PLANS_FIG / "a7_safescale_pub.png")
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    main()
