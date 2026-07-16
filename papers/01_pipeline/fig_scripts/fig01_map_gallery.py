#!/usr/bin/env python
"""fig01_map_gallery -- the fiducial 5x5 deg lightcone sky: DMO kappa, BIND
kappa, BIND Compton-y (Sec. 2.3, lightcone construction).

Visual proof that the BIND-painted convergence is a distinct, physically
structured field (not a rescaled DMO map), plus the self-consistent tSZ
Compton-y sky produced by the same painted patches.

Data (cached, no re-derivation):
  - /mnt/home/mlee1/ceph/bind_science/demo_maps.npz
      kappa_bind (1024,1024), kappa_dmo (1024,1024), y (1024,1024),
      fov_deg (scalar, =5.0), z_source (scalar)

Source: examples/paper_lightcone_figs.ipynb (worktree wl-tsz-bridge), cell 3
(heading "Sec. 2.3 -- Lightcone construction"), built by
examples/_build_paper_nb.py. Reproduces the exact transform used there: a
1-px Gaussian cosmetic smoothing (does not alter the underlying arrays used
for statistics elsewhere in the paper, only this display panel), cividis for
the two kappa panels with a shared color scale set by the 1st/99th
percentile of BIND kappa, inferno + LogNorm for the y panel.

Placeholder this replaces: figs/fig01_map_gallery.png (md5-identical to
/mnt/home/mlee1/BIND/figs/demo_maps.png, a simpler un-styled direct render of
the same npz -- this script reproduces the properly-styled publication
version of the same three arrays).
"""
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import TWO_COL, save, setup  # noqa: E402

DEMO_MAPS = "/mnt/home/mlee1/ceph/bind_science/demo_maps.npz"
SMOOTH_PX = 1.0  # light cosmetic smoothing, matches source cell


def main() -> None:
    setup()
    d = np.load(DEMO_MAPS)
    kdmo, kbind, y = d["kappa_dmo"], d["kappa_bind"], d["y"]
    fov = float(d["fov_deg"])
    ext = [0, fov, 0, fov]

    kd = gaussian_filter(kdmo, SMOOTH_PX)
    kb = gaussian_filter(kbind, SMOOTH_PX)
    ys = gaussian_filter(y, SMOOTH_PX)

    fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], 2.6))
    vk = np.percentile(kb, [1, 99])
    for a, m, lab in ((ax[0], kd, r"DMO $\kappa$"), (ax[1], kb, r"BIND $\kappa$")):
        im = a.imshow(m, origin="lower", extent=ext, cmap="cividis",
                       vmin=vk[0], vmax=vk[1], rasterized=True)
        a.text(0.04, 0.94, lab, transform=a.transAxes, color="w", fontsize=8, va="top")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
    cb.set_label(r"$\kappa$")

    im2 = ax[2].imshow(
        ys, origin="lower", extent=ext, cmap="inferno", rasterized=True,
        norm=mpl.colors.LogNorm(vmin=np.percentile(ys[ys > 0], 50), vmax=np.percentile(ys, 99.8)),
    )
    ax[2].text(0.04, 0.94, r"BIND $y$", transform=ax[2].transAxes, color="w", fontsize=8, va="top")
    cb2 = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02)
    cb2.set_label(r"$y$")

    for a in ax:
        a.set_xlabel(r"$\theta_x$ [deg]")
        a.set_xticks([0, 2.5, 5])
    ax[0].set_ylabel(r"$\theta_y$ [deg]")
    ax[0].set_yticks([0, 2.5, 5])
    for a in ax[1:]:
        a.set_yticks([0, 2.5, 5])
        a.set_yticklabels([])

    fig.tight_layout(w_pad=0.6)
    save(fig, "figs/fig01_map_gallery")


if __name__ == "__main__":
    main()
