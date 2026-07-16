#!/usr/bin/env python
"""Fig 4 -- headline result: the baryon-template nuisance-parameter ladder
(216-run SB35 Sobol suite; also backs Table 1).

Adding N SVD-mode "baryon template" nuisance amplitudes to the (Omega_m, S8)
Fisher matrix, marginalising over their span, tests necessity/sufficiency of
a low-dimensional feedback model for removing the unmodelled-baryon cosmic-
shear bias. (a) residual |dS8|/sigma_S8 vs N (median solid, 95th pct dashed;
one color per ell_max cut), N=2 marked. (b) sigma_S8 inflation vs N relative
to the no-baryon-nuisance floor (cost of over-modelling).

Data (already-cached; no re-derivation):
  - examples/figures_lightcone/cosmo_bias_capstone.npz (in the main BIND
    checkout) -- ell_maxes(3,), n_templates(7,)=[0,1,2,3,4,6,10],
    ratio_{ellmax}_{N}(216,), sig_{ellmax}_{N}(), sig0_{ellmax}(),
    evr_{ellmax}(6,). This single file also backs every Table 1 entry.

Source: examples/cosmo_bias_capstone.py (analysis/wl-cosmo-bias branch),
main(), lines ~121-153.
Placeholder this replaces: figs/fig04_template_ladder.png (md5-identical to
examples/figures_lightcone/cosmo_bias_capstone.png).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import TWO_COL, COLORS, panel_label, save, setup  # noqa: E402

CAPSTONE = Path("/mnt/home/mlee1/BIND/examples/figures_lightcone/cosmo_bias_capstone.npz")


def main() -> None:
    setup()
    d = np.load(CAPSTONE, allow_pickle=True)
    ell_maxes = [int(e) for e in d["ell_maxes"]]
    n_templates = d["n_templates"]

    fig, axes = plt.subplots(1, 2, figsize=TWO_COL)
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(ell_maxes)))

    for c, e in zip(cols, ell_maxes):
        med = [np.median(d[f"ratio_{e}_{N}"]) for N in n_templates]
        p95 = [np.percentile(d[f"ratio_{e}_{N}"], 95) for N in n_templates]
        axes[0].plot(n_templates, med, "-o", ms=3, lw=1.3, color=c,
                     label=rf"$\ell_{{\max}}={e}$")
        axes[0].plot(n_templates, p95, "--", lw=1.0, color=c, alpha=0.6)
        cost = [float(d[f"sig_{e}_{N}"]) / float(d[f"sig0_{e}"]) for N in n_templates]
        axes[1].plot(n_templates, cost, "-o", ms=3, lw=1.3, color=c,
                     label=rf"$\ell_{{\max}}={e}$")

    axes[0].axhline(1, color="0.4", ls=":", lw=0.8)
    axes[0].axvline(2, color=COLORS["highlight"], lw=1.0)
    axes[0].set_xlabel("N baryon-template nuisance params")
    axes[0].set_ylabel(r"residual $|\Delta S_8|/\sigma_{S_8}$ (LSST-Y10)")
    axes[0].set_yscale("log")
    axes[0].text(0.97, 0.94, "solid: median\ndashed: 95th pct",
                 transform=axes[0].transAxes, ha="right", va="top")
    axes[0].legend(loc="lower left")
    panel_label(axes[0], "(a)")

    axes[1].axvline(2, color=COLORS["highlight"], lw=1.0)
    axes[1].set_xlabel("N baryon-template nuisance params")
    axes[1].set_ylabel(r"$\sigma_{S_8}$ inflation (vs no-baryon)")
    axes[1].legend(loc="upper left")
    panel_label(axes[1], "(b)", loc="lower right")

    fig.subplots_adjust(wspace=0.35)
    save(fig, "figs/fig04_template_ladder")


if __name__ == "__main__":
    main()
