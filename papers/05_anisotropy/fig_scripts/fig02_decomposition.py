"""fig02_decomposition.png -- headline figure: BIND vs the faithful
radial-BCM-warp control (sph) vs the crude pure-spherical control (mono),
at z_s=1. Left: suppression S(ell)=C_ell/C_ell^dmo. Center: residual-field
power for the total baryon effect and its BCM-miss / triaxiality pieces.
Right: the decomposition fractions f_aniso, f_tri, f_mono=f_aniso+f_tri.

Data (cached, unmodified -- current/faithful sph and mono, no gotcha here):
    /mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/{bind,sph,mono,dmo}/kappa_maps.npz
  -> key 'kappa' (8 real, 4 src-z, 1024, 1024), 'source_redshifts', 'fov_deg'.

Computation reproduces examples/bcm_warp_comparison.py's compare() exactly
(worktree copy at
/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/wt/wl-anisotropy/examples/bcm_warp_comparison.py):
per realization, bind.inference.stats.power_spectrum (Pylians flat-sky FFT)
of the 4 auto fields plus 4 residual-difference fields, averaged over the
8 matched-seed realizations, then

    f_aniso = (C_bind - C_sph)  / (C_bind - C_dmo)   # BCM (sph) misses
    f_tri   = (C_sph  - C_mono) / (C_bind - C_dmo)   # sph captures, mono misses
    f_mono  = (C_bind - C_mono) / (C_bind - C_dmo) = f_aniso + f_tri

This is a real, cheap FFT-based post-processing of cached maps -- not a
re-run of the lightcone/painting engine.

Placeholder figs/fig02_decomposition.png was already byte-identical to
examples/figures_lightcone/bcm_warp_comparison_fid.png (real, current
engine output, not fabricated); this script reproduces that same
computation and restyles it per FIGURE_STYLE.md.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib.pyplot as plt  # noqa: E402

from paper_style import COLORS, panel_label, save, setup  # noqa: E402

from bind.inference.stats import power_spectrum  # noqa: E402

CEPH = pathlib.Path("/mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid")
FIELDS = ("bind", "sph", "mono", "dmo")
ZSRC_TARGET = 1.0
LMIN, LMAX, LFRAC = 300.0, 2.5e4, 2e3  # plot ranges; LFRAC skips 0/0 low ell


def load():
    K = {}
    for f in FIELDS:
        K[f] = np.load(CEPH / f / "kappa_maps.npz")["kappa"]
    meta = np.load(CEPH / "bind" / "kappa_maps.npz")
    zsrc = meta["source_redshifts"]
    fov = float(meta["fov_deg"])
    return K, zsrc, fov


def compare(K, isrc, fov):
    b, s, m, d = (K[f][:, isrc] for f in FIELDS)
    n_real = b.shape[0]
    Cb, Cs, Cm, Cd = [], [], [], []
    Pmiss, Ptri, Pbar, Pmono = [], [], [], []
    ell = None
    for r in range(n_real):
        ell, cb = power_spectrum(b[r], fov_deg=fov)
        _, cs = power_spectrum(s[r], fov_deg=fov)
        _, cm = power_spectrum(m[r], fov_deg=fov)
        _, cd = power_spectrum(d[r], fov_deg=fov)
        _, pmiss = power_spectrum(b[r] - s[r], fov_deg=fov)
        _, ptri = power_spectrum(s[r] - m[r], fov_deg=fov)
        _, pbar = power_spectrum(b[r] - d[r], fov_deg=fov)
        _, pmono = power_spectrum(b[r] - m[r], fov_deg=fov)
        Cb.append(cb); Cs.append(cs); Cm.append(cm); Cd.append(cd)
        Pmiss.append(pmiss); Ptri.append(ptri); Pbar.append(pbar); Pmono.append(pmono)

    Cb, Cs, Cm, Cd = (np.asarray(x).mean(0) for x in (Cb, Cs, Cm, Cd))
    Pmiss, Ptri, Pbar, Pmono = (np.asarray(x).mean(0) for x in (Pmiss, Ptri, Pbar, Pmono))

    denom = Cb - Cd
    f_aniso = (Cb - Cs) / denom
    f_tri = (Cs - Cm) / denom
    f_mono = (Cb - Cm) / denom
    return dict(
        ell=ell, S_bind=Cb / Cd, S_sph=Cs / Cd, S_mono=Cm / Cd,
        P_bar=Pbar, P_mono=Pmono, P_miss=Pmiss, P_tri=Ptri,
        f_aniso=f_aniso, f_tri=f_tri, f_mono=f_mono,
    )


def main():
    setup()
    K, zsrc, fov = load()
    isrc = int(np.argmin(np.abs(zsrc - ZSRC_TARGET)))
    res = compare(K, isrc, fov)

    ell = res["ell"]
    ok = (ell > LMIN) & (ell < LMAX)
    okf = (ell > LFRAC) & (ell < LMAX)

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) suppression
    ax[0].plot(ell[ok], res["S_bind"][ok], color=COLORS["bind"], lw=1.4, label="BIND")
    ax[0].plot(ell[ok], res["S_sph"][ok], color=COLORS["secondary"], lw=1.4, label="radial BCM")
    ax[0].plot(ell[ok], res["S_mono"][ok], color=COLORS["highlight"], lw=1.2, ls="--", label="pure spherical")
    ax[0].axhline(1.0, color=COLORS["dmo"], lw=0.7)
    ax[0].set_xscale("log")
    ax[0].set_xlim(LMIN, LMAX)
    ax[0].set_ylim(0.85, 1.02)
    ax[0].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(r"$S=C_\ell/C_\ell^{\rm dmo}$")
    ax[0].legend(fontsize=6, loc="lower left")
    panel_label(ax[0], "(a)")

    # (b) residual-field power
    ax[1].loglog(ell[ok], res["P_bar"][ok], color=COLORS["dmo"], lw=1.2, label="total baryon")
    ax[1].loglog(ell[ok], res["P_mono"][ok], color=COLORS["highlight"], lw=1.2, label="pure-sph error")
    ax[1].loglog(ell[ok], res["P_miss"][ok], color=COLORS["bind"], lw=1.2, label="BCM misses")
    ax[1].loglog(ell[ok], res["P_tri"][ok], color=COLORS["secondary"], lw=1.2, label="triaxiality")
    ax[1].set_xlim(LMIN, LMAX)
    ax[1].set_xlabel(r"$\ell$")
    ax[1].set_ylabel(r"$C_\ell^{\Delta\Delta}$")
    ax[1].legend(
        fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
        columnspacing=1.0, handlelength=1.4,
    )
    panel_label(ax[1], "(b)")

    # (c) decomposition fractions
    ax[2].plot(ell[okf], res["f_aniso"][okf], color=COLORS["bind"], lw=1.4, label=r"$f_{\rm aniso}$")
    ax[2].plot(ell[okf], res["f_tri"][okf], color=COLORS["secondary"], lw=1.4, label=r"$f_{\rm tri}$")
    ax[2].plot(ell[okf], res["f_mono"][okf], color=COLORS["highlight"], lw=1.2, ls="--", label=r"$f_{\rm mono}$")
    ax[2].axhline(0, color=COLORS["dmo"], lw=0.7)
    ax[2].axhline(1, color=COLORS["dmo"], lw=0.7)
    ax[2].set_xscale("log")
    ax[2].set_xlim(LFRAC, LMAX)
    ax[2].set_ylim(-0.2, 1.2)
    ax[2].set_xlabel(r"$\ell$")
    ax[2].set_ylabel(r"fraction of total baryon $\Delta C_\ell$")
    ax[2].legend(fontsize=6, loc="upper left")
    panel_label(ax[2], "(c)", loc="lower left")

    fig.tight_layout()
    save(fig, "figs/fig02_decomposition")


if __name__ == "__main__":
    main()
