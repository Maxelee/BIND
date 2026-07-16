#!/usr/bin/env python
"""Fig 3 -- effective (Omega_m, S8) cosmology bias from the unmodelled baryon
signal, per SB35 Sobol realization.

(a) (dOmega_m, dS8) bias cloud at ell_max=3000 (99 runs), colored by group
f_gas; the 1-sigma LSST-Y10 S8 precision is shown as a vertical reference
bar at the origin. (b) Histogram of dS8/sigma_S8 at three ell_max cuts
(2000/3000/5000).

Data (already-cached; no re-derivation):
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/cosmo_bias.npz --
    run_idx(99,), fgas(99,), ell_maxes(3,), dS8_{2000,3000,5000}(99,),
    dOm_{2000,3000,5000}(99,), sigS8_{2000,3000,5000}(99,).

Source: examples/lightcone_cosmo_bias.py (analysis/wl-cosmo-bias branch),
main(), lines ~176-209. The original figure also draws a horizontal
1-sigma(Omega_m) bar on the reference cross using `sigOm`, computed from the
same per-run Fisher matrix as sigS8 but NOT written to cosmo_bias.npz (only
dS8/dOm/sigS8 were cached -- verified by inspecting the file's key list).
Reproducing the pyccl Fisher run to recover sigOm is out of scope (forbidden
re-derivation), so panel (a) keeps only the vertical (S8) component of the
LSST-Y10 reference bar; every plotted data point (dOm, dS8, fgas) is
unchanged. See fig_scripts/notes_fig03_bias_scatter.md.
Placeholder this replaces: figs/fig03_bias_scatter.png (md5-identical to
examples/figures_lightcone/cosmo_bias.png).
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
BIAS = CACHE / "cosmo_bias.npz"
ELL_MAXES = (2000, 3000, 5000)
ELL_HIGHLIGHT = 3000


def main() -> None:
    setup()
    d = np.load(BIAS, allow_pickle=True)
    fgas = d["fgas"]
    dOm0 = d[f"dOm_{ELL_HIGHLIGHT}"]
    dS80 = d[f"dS8_{ELL_HIGHLIGHT}"]
    sigS80 = d[f"sigS8_{ELL_HIGHLIGHT}"]
    nrun = dOm0.shape[0]

    norm = Normalize(np.nanpercentile(fgas, 2), np.nanpercentile(fgas, 98))
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=TWO_COL)

    ax.scatter(dOm0, dS80, c=fgas, cmap="viridis", norm=norm,
               s=14, edgecolor="k", lw=0.2, zorder=3)
    ax.errorbar(0, 0, yerr=np.median(sigS80), color=COLORS["highlight"],
                lw=1.4, capsize=3, zorder=4, label=r"$1\sigma\,S_8$ LSST-Y10")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\Delta\Omega_m$")
    ax.set_ylabel(r"$\Delta S_8$")
    ax.legend(loc="upper left")
    panel_label(ax, rf"(a) $\ell_{{\max}}={ELL_HIGHLIGHT}$", loc="lower right")
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"group $f_{\rm gas}$")

    cols = cm.viridis(np.linspace(0.15, 0.85, len(ELL_MAXES)))
    for k, e in enumerate(ELL_MAXES):
        ratio = d[f"dS8_{e}"] / d[f"sigS8_{e}"]
        ax2.hist(ratio, bins=22, histtype="step", lw=1.4, color=cols[k],
                  label=rf"$\ell_{{\max}}={e}$")
    ax2.axvline(0, color="0.5", lw=0.6)
    for s in (-5, -2, 2):
        ax2.axvline(s, color="0.7", ls=":", lw=0.6)
    ax2.set_xlabel(r"$\Delta S_8/\sigma_{S_8}$ (LSST-Y10)")
    ax2.set_ylabel(f"runs (of {nrun})")
    ax2.legend(loc="upper right")
    panel_label(ax2, "(b)")

    fig.subplots_adjust(wspace=0.4)
    save(fig, "figs/fig03_bias_scatter")


if __name__ == "__main__":
    main()
