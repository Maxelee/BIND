"""fig01_maps.png -- example convergence maps: DMO, spherical BCM (mono
control), BIND (full), and the difference kappa_bind - kappa_mono.

Data (cached, unmodified):
    /mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/dmo/kappa_maps.npz
    /mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/mono/kappa_maps.npz
    /mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/bind/kappa_maps.npz
  -> key 'kappa', shape (8 realizations, 4 source-z, 1024, 1024).
  Index [0, isrc=1] = realization 0, z_s=1.0 (source_redshifts=[0.5,1,1.5,2]).

GOTCHA (see fig_scripts/DATA_MAP.md): the original notebook cell read
fid/sph/ for the "spherical BCM" panel, but that directory now holds the
faithful radial-BCM-warp control (redefined after the notebook cell was
rendered). fid/mono/ is the directory that reproduces the placeholder's
printed diagnostics exactly (rms dmo=2.34e-02 mono=2.32e-02 bind=2.29e-02,
residual rms = 16.0% of bind rms) -- verified numerically below. So this
script reads fid/mono/, not the current fid/sph/.

Original source: examples/wl_anisotropy_paper.ipynb, cell 20
("Field 1 -- the maps"), worktree copy at
/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/wt/wl-anisotropy/examples/wl_anisotropy_paper.ipynb
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib.pyplot as plt  # noqa: E402

from paper_style import TWO_COL, save, setup  # noqa: E402

CEPH = pathlib.Path("/mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid")
ISRC = 1  # source_redshifts=[0.5,1.0,1.5,2.0] -> z_s=1.0
IREAL = 0
FOV_DEG = 5.0


def load(field):
    return np.load(CEPH / field / "kappa_maps.npz")["kappa"][IREAL, ISRC]


def main():
    setup()

    dmo = load("dmo")
    mono = load("mono")  # "spherical BCM" panel -- see GOTCHA above
    bind = load("bind")
    aniso = bind - mono

    print(
        f"rms kappa  dmo={dmo.std():.2e}  mono={mono.std():.2e}  bind={bind.std():.2e}"
        f"  |  anisotropy residual rms = {aniso.std():.2e} "
        f"({100 * aniso.std() / bind.std():.1f}% of bind rms)"
    )

    vk = np.percentile(bind, 99)
    va = np.percentile(np.abs(aniso), 99)

    panels = [
        ("(a) DMO", dmo, "cividis", 0, vk),
        ("(b) spherical BCM", mono, "cividis", 0, vk),
        ("(c) BIND (full)", bind, "cividis", 0, vk),
        (r"(d) BIND$-$mono", aniso, "RdBu_r", -va, va),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL[0], TWO_COL[0] / 4 + 0.75))
    for i, (ax, (lab, m, cmap, lo, hi)) in enumerate(zip(axes, panels)):
        im = ax.imshow(
            m,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            origin="lower",
            extent=[0, FOV_DEG, 0, FOV_DEG],
            rasterized=True,
        )
        ax.text(
            0.06,
            0.95,
            lab,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color="k",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2),
        )
        cb_label = r"$\kappa$" if cmap == "cividis" else r"$\Delta\kappa$"
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=cb_label)
        ax.set_xlabel(r"$\theta_x$ [deg]")
        if i == 0:
            ax.set_ylabel(r"$\theta_y$ [deg]")
        else:
            ax.set_yticklabels([])
            ax.set_xticks([0, 2, 4])

    fig.subplots_adjust(wspace=0.55)
    save(fig, "figs/fig01_maps")


if __name__ == "__main__":
    main()
