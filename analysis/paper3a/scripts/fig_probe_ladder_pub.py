"""Publication figure: the per-probe evacuation ladder — Delta ln M_gas
demanded by each probe, one row per likelihood (forest-plot grammar,
Bigwood+25 Fig. 8), with the B-side grid support drawn as the caveat
region it is.

Run: python analysis/paper3a/scripts/fig_probe_ladder_pub.py
Out: wp6_propagation/figures/probe_ladder_pub.{pdf,png} + plans copy
     wp6_propagation/probe_ladder.json (the quoted numbers)
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

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.scripts.fig_a5_tension import load_flat  # noqa: E402
from analysis.paper3a.scripts.run_ab_gate import (  # noqa: E402
    GRID, bin_weights, delta_coords)
from analysis.paper3a.style import (  # noqa: E402
    COL, W_SINGLE, apply, caveat_region)

N_MAP = 3000
PLANS_FIG = Path("/mnt/home/mlee1/bind-paper3-plans/figures_AB")

ROWS = [  # (label, chain pattern or None, color)
    ("X-ray f$_{\\rm gas}$", "a5_final_seed{k}.npz", COL["variant"]),
    ("kSZ", "a5_subset_kszonly_seed{k}.npz", COL["aux"]),
    ("X-ray + kSZ", "a5_subset_joint_seed{k}.npz", COL["variant"]),
    ("joint (+ $\\kappa$-peaks $\\times$ y)", "joint_ab_seed{k}.npz",
     COL["model"]),
]


def main() -> None:
    apply()
    emu = StatsEmulator.load()
    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz",
                  allow_pickle=False)
    edges = np.asarray(json.loads(str(raw["manifest"]))["meta"]["mass_bins"],
                       float)
    w = bin_weights(edges)
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    rng = np.random.default_rng(41)

    quotes = {}
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.4))
    for i, (label, pattern, col) in enumerate(ROWS):
        if pattern == "joint_ab_seed{k}.npz":
            flat = np.concatenate(
                [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
                 .astype(float).reshape(-1, 32) for k in range(4)])
        else:
            flat = load_flat(pattern)
        sub = flat[rng.choice(len(flat), N_MAP, replace=False)][:, :30]
        m = delta_coords(emu, sub, u_fid, w)["dln_mgas"]
        p16, p50, p84 = np.percentile(m, [16, 50, 84])
        y = len(ROWS) - 1 - i
        ax.errorbar([p50], [y], xerr=[[p50 - p16], [p84 - p50]],
                    fmt="o", ms=4.5, color=col, elinewidth=1.3,
                    capsize=2.5, zorder=5)
        quotes[label] = {"p16": float(p16), "p50": float(p50),
                         "p84": float(p84)}

    # B-side: every grid node is rejected (min chi2 485/4 at -0.10 in
    # this frame) -> the kappa-peak likelihood demands evacuation beyond
    # its sampled support; draw as a left-pointing demand arrow.
    gate = json.loads((WP6 / "ab_gate.json").read_text())
    off = gate["reference_offset"]["dln_mgas"]["value"]
    g = np.load(GRID, allow_pickle=True)
    lo = float(np.min(np.asarray(g["delta_ln_mgas"], float) - off))
    hi = float(np.max(np.asarray(g["delta_ln_mgas"], float) - off))
    caveat_region(ax, lo, hi)
    ax.annotate("$\\kappa$-peak grid\nsupport (all\nnodes rejected)",
                (0.5 * (lo + hi), 1.45), fontsize=6.2,
                ha="center", va="center", color="#777777")
    b_edge = lo
    ax.annotate("", xytext=(b_edge, -0.55), xy=(b_edge - 0.18, -0.55),
                arrowprops=dict(arrowstyle="-|>", color=COL["data"],
                                lw=1.4))
    ax.annotate("$\\kappa$-peaks $\\times$ y demand",
                (b_edge - 0.02, -0.38), fontsize=6.5, ha="right",
                color=COL["data"])

    ax.axvline(0, color=COL["ref"], lw=0.6, ls=":")
    ax.annotate("TNG", (0.012, len(ROWS) - 0.62), fontsize=7)
    ax.set_yticks(range(len(ROWS)),
                  [r[0] for r in reversed(ROWS)], fontsize=8)
    ax.set_xlabel(r"$\Delta \ln M_{\rm gas}$ demanded"
                  r"  ($M_{200c} > 10^{13.5}\,{\rm M_\odot}$)")
    ax.set_xlim(-0.85, 0.20)
    ax.set_ylim(-0.75, len(ROWS) - 0.25)

    fig.tight_layout()
    out = WP6 / "figures" / "probe_ladder_pub"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    (WP6 / "probe_ladder.json").write_text(json.dumps(quotes, indent=2))
    if PLANS_FIG.exists():
        shutil.copy(f"{out}.png", PLANS_FIG / "probe_ladder_pub.png")
    print(json.dumps(quotes, indent=2))
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    main()
