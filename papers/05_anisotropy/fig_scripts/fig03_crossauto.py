"""fig03_crossauto.png -- exact cross/auto decomposition of
Delta P = P_bind - P_sph (Methods): the power-spectrum difference between
the full BIND field and the spherical control splits exactly into a
first-order cross term (correlation of the anisotropic correction with the
halo's own DMO field) and a second-order auto term.

    cross = 2 Re < kappa_dmo* F(kappa_bind - kappa_sph) >   (1st order)
    auto  = <|F d_bind|^2> - <|F d_sph|^2>                  (2nd order)
    dP    = cross + auto  (closure, checked to ~machine precision)

Left panel: raw Delta C_ell contributions. Right: fraction of Delta P.

Data (cached, unmodified):
    /mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/{bind,dmo,mono}/kappa_maps.npz
  -> key 'kappa' (8 real, 4 src-z, 1024, 1024), index [:, isrc=1] (z_s=1.0),
  all 8 realizations averaged.

GOTCHA (see fig_scripts/DATA_MAP.md): the notebook cell reads fid/sph/ for
the "spherical" field, but that directory now holds the faithful
radial-BCM-warp control (redefined after the notebook cell was rendered).
fid/mono/ is the directory that exactly reproduces the placeholder's
printed band values (cross/dP=+1.39, auto/dP=-0.39, closure=1.000 over
ell in [3000,20000]) -- verified numerically below; current fid/sph gives
+1.45/-0.45, which does NOT match. So this script reads fid/mono/, not the
current fid/sph/.

Original source: examples/wl_anisotropy_paper.ipynb, cell 24
("Why the anisotropy is first order: the cross/auto decomposition"),
worktree copy at
/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/wt/wl-anisotropy/examples/wl_anisotropy_paper.ipynb
Pure numpy FFT on cached npz arrays -- no engine/model dependency.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib.pyplot as plt  # noqa: E402

from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

CEPH = pathlib.Path("/mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid")
ISRC = 1  # source_redshifts=[0.5,1.0,1.5,2.0] -> z_s=1.0
FOV_DEG = 5.0
ELL_BAND = (3e3, 2e4)  # band used for the printed closure check


def main():
    setup()

    Kb = np.load(CEPH / "bind" / "kappa_maps.npz", mmap_mode="r")["kappa"]
    Ks = np.load(CEPH / "mono" / "kappa_maps.npz", mmap_mode="r")["kappa"]  # see GOTCHA above
    Kd = np.load(CEPH / "dmo" / "kappa_maps.npz", mmap_mode="r")["kappa"]
    nreal, npix = Kb.shape[0], Kb.shape[-1]

    fov = np.deg2rad(FOV_DEG)
    kk = np.fft.fftfreq(npix) * npix * (2 * np.pi / fov)  # |k| index -> multipole ell
    KK = np.hypot(*np.meshgrid(kk, kk))
    bins = np.logspace(np.log10(200), np.log10(3e4), 25)
    nb = len(bins) + 1
    ib = np.clip(np.digitize(KK.ravel(), bins), 0, nb - 1)
    ellc = np.concatenate([[bins[0]], 0.5 * (bins[1:] + bins[:-1]), [bins[-1]]])

    def _binmean(p2d):
        out = np.bincount(ib, weights=p2d.ravel(), minlength=nb)
        cnt = np.bincount(ib, minlength=nb)
        return out / np.maximum(cnt, 1)

    cross = np.zeros(nb)
    auto = np.zeros(nb)
    dP = np.zeros(nb)
    for r in range(nreal):
        d = Kd[r, ISRC].astype(np.float64)
        b = Kb[r, ISRC].astype(np.float64)
        s = Ks[r, ISRC].astype(np.float64)
        Fa = np.fft.fft2(d - d.mean())
        Fdb = np.fft.fft2((b - d) - (b - d).mean())
        Fds = np.fft.fft2((s - d) - (s - d).mean())
        cross += _binmean(2 * np.real(np.conj(Fa) * (Fdb - Fds)))
        auto += _binmean(np.abs(Fdb) ** 2 - np.abs(Fds) ** 2)
        dP += _binmean(np.abs(Fa + Fdb) ** 2 - np.abs(Fa + Fds) ** 2)
    cross, auto, dP = cross / nreal, auto / nreal, dP / nreal

    band = (ellc > ELL_BAND[0]) & (ellc < ELL_BAND[1])
    cf, af = cross[band].sum() / dP[band].sum(), auto[band].sum() / dP[band].sum()
    print(
        f"band ell in [{ELL_BAND[0]:.0f},{ELL_BAND[1]:.0f}]:  cross/dP = {cf:+.2f}  |  "
        f"auto/dP = {af:+.2f}  |  closure (cross+auto)/dP = {cf + af:.3f}"
    )

    fig, ax = plt.subplots(1, 2, figsize=TWO_COL)
    m = (ellc > 1e3) & (ellc < 2.2e4)

    ax[0].semilogx(ellc[m], dP[m], "-", color=COLORS["bind"], lw=1.5, ms=3,
                    label=r"$\Delta P$")
    ax[0].semilogx(ellc[m], cross[m], "-", color=COLORS["highlight"], lw=1.5, ms=3,
                    label="cross (1st ord.)")
    ax[0].semilogx(ellc[m], auto[m], "-", color=COLORS["secondary"], lw=1.5, ms=3,
                    label="auto (2nd ord.)")
    ax[0].axhline(0, color=COLORS["dmo"], lw=0.7)
    ax[0].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(r"$\Delta C_\ell$ contribution")
    ax[0].legend(fontsize=6.5, loc="lower right")
    panel_label(ax[0], "(a)")

    with np.errstate(divide="ignore", invalid="ignore"):
        ax[1].semilogx(ellc[m], (cross / dP)[m], "-", color=COLORS["highlight"], lw=1.5, ms=3,
                        label=r"cross$/\Delta P$")
        ax[1].semilogx(ellc[m], (auto / dP)[m], "-", color=COLORS["secondary"], lw=1.5, ms=3,
                        label=r"auto$/\Delta P$")
    ax[1].axhline(1, color=COLORS["dmo"], lw=0.7, ls=":")
    ax[1].axhline(0, color=COLORS["dmo"], lw=0.7)
    ax[1].set_xlabel(r"$\ell$")
    ax[1].set_ylabel(r"fraction of $\Delta P$")
    ax[1].set_ylim(-1.5, 2.5)
    ax[1].legend(fontsize=6.5, loc="lower right")
    panel_label(ax[1], "(b)")

    fig.tight_layout()
    save(fig, "figs/fig03_crossauto")


if __name__ == "__main__":
    main()
