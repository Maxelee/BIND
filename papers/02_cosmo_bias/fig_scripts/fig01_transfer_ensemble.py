#!/usr/bin/env python
"""Fig 1 -- baryonic WL transfer-function ensemble across the SB35 Sobol suite.

For each of 99 completed SB35 Sobol lightcone realizations,

    S_i(ell, z_s) = C_ell^{kappa kappa}(theta_astro^i) / C_ell^{kappa kappa, DMO}

per tomographic source plane z_s (5 planes). One panel per source plane;
each thin line is one Sobol run, colored by the group-scale
(1e13-1e14.5 Msun) baryon mass-weighted gas fraction f_gas (the physical
mediator of the suppression amplitude/sign). Black solid = ensemble median,
black dashed = 95% envelope (2.5/97.5 pct).

Data (already-cached; no re-derivation):
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/transfer_Sell.npz
      ell(724,), S(99,5,724), run_idx(99,), source_redshifts(5,)
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz
      M_fof(2933,), sobol_m_{gas,dm,star}_500c(256,2933) -- feeds the f_gas
      color axis via group_fgas() (reproduced verbatim below).

Source: examples/lightcone_transfer.py (analysis/wl-cosmo-bias branch),
main(), "Fig 1" block (~lines 92-120); build_cube()/group_fgas() are pure
numpy over the already-cached arrays -- no cosmology library, no re-run.
Placeholder this replaces: figs/fig01_transfer_ensemble.png (md5-identical
to examples/figures_lightcone/transfer_function_ensemble.png).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import TWO_COL, COLORS, panel_label, save, setup  # noqa: E402

CACHE = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache")
TRANSFER = CACHE / "transfer_Sell.npz"
ATLAS = CACHE / "atlas_cubes" / "atlas_cube_snap096.npz"
ELL_MAX_TRUST = 1.5e4  # above this CIC aliasing dominates the run-to-run scatter


def group_fgas(run_idx: np.ndarray, logM_lo: float = 13.0,
                logM_hi: float = 14.5) -> np.ndarray:
    """Per-run mass-weighted gas fraction of group-scale halos (the mediator)."""
    a = np.load(ATLAS)
    M = a["M_fof"].astype(np.float64)
    sel = (np.log10(M) >= logM_lo) & (np.log10(M) < logM_hi)
    mg = a["sobol_m_gas_500c"][run_idx][:, sel].sum(1)
    mt = (a["sobol_m_dm_500c"] + a["sobol_m_gas_500c"]
          + a["sobol_m_star_500c"])[run_idx][:, sel].sum(1)
    return mg / np.where(mt > 0, mt, np.nan)


def main() -> None:
    setup()
    d = np.load(TRANSFER, allow_pickle=True)
    ell, S = d["ell"], d["S"]
    zs, idx = d["source_redshifts"], d["run_idx"]
    fgas = group_fgas(idx)

    m = ell <= ELL_MAX_TRUST
    L = ell[m]
    norm = Normalize(np.nanpercentile(fgas, 2), np.nanpercentile(fgas, 98))
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    order = np.argsort(fgas)  # draw gas-poor (strongest suppression) on top

    fig, axes = plt.subplots(1, 5, figsize=(TWO_COL[0], 2.3), sharey=True)
    for i, ax in enumerate(axes):
        for r in order:
            ax.plot(L, S[r, i, m], lw=0.25, alpha=0.4,
                    color=sm.to_rgba(fgas[r]), rasterized=True)
        med = np.median(S[:, i, m], 0)
        lo, hi = np.percentile(S[:, i, m], [2.5, 97.5], 0)
        ax.plot(L, med, color=COLORS["truth"], lw=1.1, zorder=5)
        ax.plot(L, lo, color=COLORS["truth"], lw=0.6, ls="--", zorder=5)
        ax.plot(L, hi, color=COLORS["truth"], lw=0.6, ls="--", zorder=5)
        ax.axhline(1.0, color="0.5", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(L.min(), ELL_MAX_TRUST)
        ax.set_ylim(0.78, 1.18)
        ax.set_xlabel(r"$\ell$")
        panel_label(ax, rf"$z_s={zs[i]:.2f}$", loc="upper left")
    axes[0].set_ylabel(r"$S(\ell)=C_\ell^{\kappa\kappa}/C_\ell^{\kappa\kappa,\rm DMO}$")

    cb = fig.colorbar(sm, ax=list(axes), fraction=0.02, pad=0.012)
    cb.set_label(r"group $f_{\rm gas}$ ($10^{13}\!-\!10^{14.5}\,M_\odot$)")

    save(fig, "figs/fig01_transfer_ensemble")


if __name__ == "__main__":
    main()
