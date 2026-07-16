#!/usr/bin/env python
"""Fig 2 -- redshift tomography of the baryonic WL transfer function.

(a) Ensemble median S(ell) per source plane z_s (16-84 pct band across the 99
SB35 Sobol runs). (b) Run-to-run scatter sigma[S(ell)] vs ell, same z_s color
coding as (a) (legend shown once, in (a), to avoid repeating five identical
entries).

Data (already-cached; no re-derivation):
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/transfer_Sell.npz
      ell(724,), S(99,5,724), source_redshifts(5,)

Source: examples/lightcone_transfer.py (analysis/wl-cosmo-bias branch),
main(), "Fig 2" block (~lines 124-148); same cached array as fig01, no
f_gas/atlas cube needed for this pair of panels.
Placeholder this replaces: figs/fig02_transfer_tomography.png (md5-identical
to examples/figures_lightcone/transfer_function_tomography.png).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import TWO_COL, panel_label, save, setup  # noqa: E402

CACHE = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache")
TRANSFER = CACHE / "transfer_Sell.npz"
ELL_MAX_TRUST = 1.5e4


def main() -> None:
    setup()
    d = np.load(TRANSFER, allow_pickle=True)
    ell, S, zs = d["ell"], d["S"], d["source_redshifts"]

    m = ell <= ELL_MAX_TRUST
    L = ell[m]
    cols = cm.plasma(np.linspace(0.1, 0.85, len(zs)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=TWO_COL)

    for i in range(len(zs)):
        med = np.median(S[:, i, m], 0)
        lo, hi = np.percentile(S[:, i, m], [16, 84], 0)
        a1.plot(L, med, color=cols[i], lw=1.4, label=rf"$z_s={zs[i]:.2f}$")
        a1.fill_between(L, lo, hi, color=cols[i], alpha=0.15, lw=0)
    a1.axhline(1, color="0.5", lw=0.6, ls=":")
    a1.set_xscale("log")
    a1.set_xlim(L.min(), ELL_MAX_TRUST)
    a1.set_xlabel(r"$\ell$")
    a1.set_ylabel(r"median $S(\ell)$ (16-84\% band)")
    a1.legend(loc="lower left")
    panel_label(a1, "(a)")

    for i in range(len(zs)):
        a2.plot(L, S[:, i, m].std(0), color=cols[i], lw=1.4)
    a2.set_xscale("log")
    a2.set_xlim(L.min(), ELL_MAX_TRUST)
    a2.set_xlabel(r"$\ell$")
    a2.set_ylabel(r"run-to-run $\sigma[S(\ell)]$")
    panel_label(a2, "(b)")

    save(fig, "figs/fig02_transfer_tomography")


if __name__ == "__main__":
    main()
