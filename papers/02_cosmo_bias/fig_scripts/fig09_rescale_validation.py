#!/usr/bin/env python
"""Fig 9 (Appendix A) -- real-data validation of Angulo & White (2010)
cosmology rescaling on TNG300-3-Dark (L205n625TNG_DM, 625^3 particles),
against three SB35 target cosmologies (mild, low-Omega_m, extreme corner).

3x2 grid, one row per target: left column is the halofit-normalized P(k)
ratio, right column the Tinker08(200m)-normalized cumulative halo mass
function ratio. In both columns, blue is the CONTROL (unrescaled
simulation vs its own original-cosmology theory) and red is the RESCALED
simulation (box relabeled by the AW10 length/mass factors) vs the target
cosmology's theory at the target redshift z'=0. The gap between blue and
red at fixed x is the (uncancelled) rescaling error.

Data (already-cached; this is the plot-only half of the source script --
the AW10 solve + P(k)/HMF measurement against the real TNG300-3-Dark
snapshot already ran once and is fully saved here, no simulation re-run):
  /mnt/home/mlee1/BIND/examples/rescale_validation.npz -- per target,
  prefix {mild,low-Om,corner-hi}_: s, z_star, rms, mass_factor,
  k_ctrl/ratio_ctrl, k_resc/ratio_resc, M_ctrl/hmf_ctrl, M_resc/hmf_resc.

Source: examples/rescale_validation.py (cosmo-rescale worktree), _plot()
lines ~191-217.
Placeholder this replaces: figs/fig09_rescale_validation.png
(md5-identical to examples/rescale_validation.png).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

NPZ = Path("/mnt/home/mlee1/BIND/examples/rescale_validation.npz")

TARGETS = [
    ("mild", r"mild"),
    ("low-Om", r"low-$\Omega_{\rm m}$"),
    ("corner-hi", r"extreme corner"),
]
LETTERS = "abc"


def main() -> None:
    setup()
    d = np.load(NPZ, allow_pickle=True)

    fig, axes = plt.subplots(3, 2, figsize=(TWO_COL[0], 6.6), sharex="col")

    for i, (name, label) in enumerate(TARGETS):
        s = float(d[f"{name}_s"])
        z_star = float(d[f"{name}_z_star"])

        ax = axes[i, 0]
        ax.semilogx(d[f"{name}_k_ctrl"], d[f"{name}_ratio_ctrl"],
                     color=COLORS["bind"], lw=0.9,
                     label="control" if i == 0 else None)
        ax.semilogx(d[f"{name}_k_resc"], d[f"{name}_ratio_resc"],
                     color=COLORS["highlight"], lw=0.9,
                     label="rescaled" if i == 0 else None)
        ax.axhline(1, color="k", lw=0.5)
        ax.set_ylim(0.75, 1.25)
        ax.set_ylabel(r"$P(k)\,/\,{\rm halofit}$")
        ax.text(0.97, 0.06, rf"$s$={s:.3f}, $z_*$={z_star:.2f}",
                transform=ax.transAxes, ha="right", color="0.35")
        panel_label(ax, f"({LETTERS[i]}) {label}", loc="upper right")
        if i == 0:
            ax.legend(loc="upper left")

        ax = axes[i, 1]
        ax.semilogx(d[f"{name}_M_ctrl"], d[f"{name}_hmf_ctrl"],
                     color=COLORS["bind"], lw=0.9)
        ax.semilogx(d[f"{name}_M_resc"], d[f"{name}_hmf_resc"],
                     color=COLORS["highlight"], lw=0.9)
        ax.axhline(1, color="k", lw=0.5)
        ax.set_ylim(0.5, 1.5)
        ax.set_ylabel(r"$N({>}M)\,/\,{\rm Tinker08}$")

    axes[-1, 0].set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    axes[-1, 1].set_xlabel(r"$M_{200\rm m}\,[M_\odot/h]$")

    fig.tight_layout()
    save(fig, "figs/fig09_rescale_validation")


if __name__ == "__main__":
    main()
