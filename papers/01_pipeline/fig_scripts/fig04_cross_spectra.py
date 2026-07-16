#!/usr/bin/env python
"""fig04_cross_spectra -- WL x tSZ cross-correlation, fiducial BIND
lightcone (Sec. "1a", tSZ x WL cross-correlation).

Left: ell(ell+1)|C_ell^{kappa y}|/2pi at the 5 lux tomographic source
planes (z_s = 0.5, 1.0, 1.5, 2.0, 2.44), cross-correlated against the
cumulative Compton-y field to z=2.44. Right: the tSZ auto-spectrum
ell(ell+1) C_ell^{yy}/2pi.

Data (cached, no re-derivation): the pre-computed equivalent of the
notebook's on-the-fly `S.cl_kappa_y(...)` call (same statistic, cached
per-run rather than recomputed from the 100 raw lux realizations):
  - /mnt/home/mlee1/ceph/bind_science/runs/bind/run_0000/Cl_kappa_y.npz
    keys: ell (724,), cl_ky (5,724), cl_ky_err, cl_yy (724,), cl_yy_err

Source: examples/fiducial_lightcone_stats.ipynb (worktree wl-tsz-bridge),
cell 13 (heading "6. tSZ x WL cross-correlation (1a)").
Placeholder this replaces: figs/fig04_cross_spectra.png (md5-identical to
figs_raw/fiducial_lightcone_stats/cell013_out0.png).

Styling note: z_s is an ordered physical quantity, so the 5 cross-spectrum
curves are colored with a sequential (viridis) colormap here rather than the
placeholder's unordered qualitative tab10 cycle -- same 5 curves/data, more
informative color choice per the FIGURE_STYLE semantic-color convention.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

CL_KY = Path("/mnt/home/mlee1/ceph/bind_science/runs/bind/run_0000/Cl_kappa_y.npz")
ZS = np.array([0.5, 1.0, 1.5, 2.0, 2.44])


def main() -> None:
    setup()
    d = np.load(CL_KY)
    ell = d["ell"]
    f = ell * (ell + 1) / (2 * np.pi)
    cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(ZS)))

    fig, ax = plt.subplots(1, 2, figsize=(TWO_COL[0], 2.7))
    for i, zs in enumerate(ZS):
        ax[0].loglog(ell, np.abs(f * d["cl_ky"][i]), color=cols[i], lw=1.2, label=f"$z_s={zs:g}$")
    ax[0].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(r"$\ell(\ell+1)|C_\ell^{\kappa y}|/2\pi$")
    ax[0].legend(loc="lower left", fontsize=6, ncol=1)
    ax[0].set_xlim(1e2, 2e4)
    panel_label(ax[0], "(a)")

    ax[1].loglog(ell, f * d["cl_yy"], color=COLORS["highlight"], lw=1.6)
    ax[1].set_xlabel(r"$\ell$")
    ax[1].set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/2\pi$")
    ax[1].set_xlim(1e2, 2e4)
    panel_label(ax[1], "(b)")

    fig.tight_layout(w_pad=1.0)
    save(fig, "figs/fig04_cross_spectra")


if __name__ == "__main__":
    main()
