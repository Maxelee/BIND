"""Publication-style WP-A6 suppression-envelope figure (style-kit).

The A5-posterior WL suppression envelope from statsemu, shown in the
paper-relevant vs-TNG-fiducial frame (ratio of the emulated painted/DMO
suppression to its own fiducial prediction — the same definition as the
wlemu task-6 numbers), for three source redshifts, with the INDEPENDENT
wlemu prediction overlaid at z_s = 1 as the cross-emulator validation.

Inputs (all frozen artifacts):
- wp6_propagation/a6_posterior_stats.npz  (suppression__q, suppression__fid)
- wp6_propagation/a6_posterior_stats_summary.json (corrected cross-check)
- sb35_stats/emulator_dataset.npz (the ell axis)

Run: python analysis/paper3a/scripts/fig_a6_suppression_pub.py
Out: wp6_propagation/figures/a6_suppression_pub.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

import matplotlib

matplotlib.use("Agg")

from analysis.paper3a.emulator.statsemu import WP6  # noqa: E402
from analysis.paper3a.style import COL, apply, condition_tag, fig_single, refline  # noqa: E402

Z_SHOW = (0.5, 1.0, 2.0)
ELL_MAX = 8000.0        # the paper's WL-statistic range; beyond is paint-noise


def main() -> None:
    apply()
    arr = np.load(WP6 / "a6_posterior_stats.npz", allow_pickle=False)
    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz", allow_pickle=False)
    summ = json.loads((WP6 / "a6_posterior_stats_summary.json").read_text())

    ell = np.asarray(raw["a__suppression__ell"], float)
    zs = list(np.asarray(arr["source_redshifts"], float))
    q = arr["suppression__q"].astype(float)              # (5q, 5z, L)
    fid = arr["suppression__fid"].astype(float)          # (5z, L)
    sel = ell <= ELL_MAX

    fig, ax = fig_single(height=3.0)
    shades = {0.5: 0.18, 1.0: 0.30, 2.0: 0.24}
    for z in Z_SHOW:
        iz = zs.index(z)
        r_med = q[2, iz] / fid[iz]
        lo, hi = q[1, iz] / fid[iz], q[3, iz] / fid[iz]
        ax.plot(ell[sel], r_med[sel], color=COL["model"],
                lw=2.0 if z == 1.0 else 1.1,
                ls="-" if z == 1.0 else "--",
                label=rf"$z_s = {z:g}$" + (" (median + 68%)" if z == 1.0 else ""))
        if z == 1.0:
            ax.fill_between(ell[sel], lo[sel], hi[sel], color=COL["model"],
                            alpha=shades[z], lw=0)

    rows = summ.get("wlemu_crosscheck_zs1", [])
    if rows:
        le = np.array([r["ell_wlemu"] for r in rows])
        m = np.array([r["wlemu_median"] for r in rows])
        e_lo = m - np.array([r["wlemu_68"][0] for r in rows])
        e_hi = np.array([r["wlemu_68"][1] for r in rows]) - m
        ok = le <= ELL_MAX
        ax.errorbar(le[ok], m[ok], yerr=[e_lo[ok], e_hi[ok]], fmt="o", ms=3.5,
                    color=COL["truth"], elinewidth=1.0, capsize=2,
                    label=r"wlemu (independent), $z_s$=1", zorder=6)

    refline(ax, y=1.0)
    ax.set_xscale("log")
    ax.set_xlim(80, ELL_MAX)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$R(\ell) = C_\ell^{\kappa\kappa}(\theta)\,/\,C_\ell^{\kappa\kappa}(\theta_{\rm fid})$")
    condition_tag(ax, "A5 posterior (eRASS1 f$_{gas}$-conditioned)")
    ax.legend(loc="lower left", fontsize=6.5)

    outdir = WP6 / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"a6_suppression_pub.{ext}")
    print(f"wrote {outdir}/a6_suppression_pub.pdf/.png")


if __name__ == "__main__":
    main()
