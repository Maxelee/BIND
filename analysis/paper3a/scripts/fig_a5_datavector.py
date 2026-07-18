"""Publication-style A5 data-vector figure (style-kit demo + paper draft).

Left: f_gas(M500) — SB35 posterior-predictive band + MAP + TNG fiducial
vs the eRASS1 binned medians. Right: the R1 boundary diagnostic for the
three edge-piling parameters. Uses the house style kit end-to-end.

Run: python analysis/paper3a/scripts/fig_a5_datavector.py
Out: wp5_chains/figures/a5_datavector_pub.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.style import (  # noqa: E402
    COL, apply, condition_tag, data_points, fiducial_line, fig_double,
)

EDGE_PARAMS = ["RadioFeedbackReiorientationFactor", "IMFslope", "QuasarThresholdPower"]


def main() -> None:
    apply()
    blk = FgasBlock()
    chains = [np.load(CHAINS / f"a5_final_seed{k}.npz")["chain"].astype(float)
              for k in range(4)]
    flat = np.concatenate([c.reshape(-1, 30) for c in chains])
    rng = np.random.default_rng(0)
    sub = flat[rng.integers(0, len(flat), 3000)]
    logp = np.concatenate([np.load(CHAINS / f"a5_final_seed{k}.npz")["logp"].astype(float).ravel()
                           for k in range(4)])
    u_map = flat[int(np.argmax(logp))]

    fig, (ax, ax2) = fig_double(height=2.7, ncols=2)

    x = blk.data.bin_centers
    pp = blk.predict(sub)
    lo, hi = np.percentile(pp, [16, 84], axis=0)
    lo2, hi2 = np.percentile(pp, [2.5, 97.5], axis=0)
    ax.fill_between(x, lo2, hi2, color=COL["model"], alpha=0.12, lw=0,
                    label="posterior 95%")
    ax.fill_between(x, lo, hi, color=COL["model"], alpha=0.30, lw=0,
                    label="posterior 68%")
    ax.plot(x, blk.predict(u_map)[0], color=COL["model"], lw=1.6, ls="--",
            label=r"MAP ($\chi^2 = 10.8/5$)")
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    fiducial_line(ax, x, blk.predict(u_fid)[0], label="TNG fiducial")
    data_points(ax, x, blk.data.values, yerr=blk.data.stat_err,
                label="eRASS1 (z < 0.2)")
    ax.set_xlabel(r"$\log_{10} M_{500}\ [M_\odot/h]$")
    ax.set_ylabel(r"$f_{\rm gas,sph}(<R_{500})$")
    condition_tag(ax, r"$z \simeq 0$ · X-ray")
    ax.legend(loc="lower right", fontsize=6.5)

    idx = [pm.ASTRO_NAMES.index(n) for n in EDGE_PARAMS]
    for j, (i, name) in enumerate(zip(idx, EDGE_PARAMS)):
        h, e = np.histogram(flat[:, i], bins=40, range=(0, 1), density=True)
        ax2.stairs(h, e, color=COL["model"], alpha=0.35 + 0.3 * j, lw=1.4,
                   label=name.replace("Factor", ""))
    ax2.axhline(1.0, color=COL["ref"], lw=0.9, ls=":")
    ax2.annotate("uniform prior", (0.02, 1.02), xycoords=("axes fraction", "data"),
                 fontsize=7, va="bottom")
    ax2.set_xlabel("parameter (unit cube)")
    ax2.set_ylabel("posterior density")
    ax2.legend(loc="upper left", fontsize=6.5, title="edge-piling (R1)", title_fontsize=7)

    out = CHAINS / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"a5_datavector_pub.{ext}")
    print(f"wrote {out}/a5_datavector_pub.pdf/.png")


if __name__ == "__main__":
    main()
