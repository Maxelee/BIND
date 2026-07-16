"""fig12_desact_sheary.pdf -- BIND shear x y vs the REAL DES Y3 x ACT confrontation.

2 panels (DES source bins 3 and 4, z_bar~=0.74 and 0.94): the real-space
shear-tSZ cross-correlation xi_{gamma y}(theta) built by a Hankel transform
of the emulator's cached tomographic C_ell^{kappa y} (n(z)-weighted over the
5 BIND source planes, physical 2.4' ACT DR6 beam applied), shown as the
256-node SB35 Sobol 16-84% band + median, plus the same statistic with NO
beam (dotted, exposes the intrinsic shape), against the real DES Y3 x ACT
data points with their measured covariance. Grey shading = outside the
robust 8'-40' aperture window (beam/resolution- and field-size-limited).
In-panel text reports the robust-window BIND/data amplitude ratio.

Data (read-only, cached on disk -- no recomputation):
  - /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz (DS)
      keys: source_redshifts, t__cl_kappa_y__value/valid, a__cl_kappa_y__ell
  - /mnt/home/mlee1/ceph/bind_science/ksz_confront/desact_data.npz (KS)
      keys: nz_z, nz_bins, covmat, cs_bin1, cs_ang, cs_value

Source: ksz-desi-act/examples/paper_ksz_field.ipynb (branch
analysis/ksz_project, worktree wt/ksz-desi-act), cell 12, section "S5 --
Confrontation -- BIND shear x y vs the REAL DES Y3 x ACT". The Hankel
transform (xi_gy, scipy.special.jv order-2), Gaussian-beam attenuation, and
the F_XPK=1.2016e7 Pylians XPk_plane normalization constant are ported
verbatim from that cell and its cell-2 setup dependency. Placeholder this
replaces is byte-identical to examples/figures_field/g5_desact_confront.pdf
(cross-paper provenance, dossier.md Gaps #2).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import jv

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import BAND_ALPHA, COLORS, TWO_COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
DS = CEPH / "bind_sb35/emulator_dataset.npz"
KS = CEPH / "bind_science/ksz_confront"
F_XPK = 1.2016e7  # Pylians XPk_plane normalization fix (C_l^{kappa y})
Y_BEAM, TLO, THI = 2.4, 8.0, 40.0  # ACT DR6 beam [arcmin]; robust aperture window

# ---- cell-2 setup dependencies (ported verbatim) ----
E = np.load(DS, allow_pickle=True)
ZS = E["source_redshifts"]


def stat(name):
    v = E[f"t__{name}__value"]
    ok = E[f"t__{name}__valid"]
    err = E[f"t__{name}__err"] if f"t__{name}__err" in E.files else None
    return v[ok], (err[ok] if err is not None else None), ok


# ---- cell 12 body (verbatim math) ----
def xi_gy(th_arcmin, ell, cl):
    th = th_arcmin * np.pi / (180 * 60)
    return np.array([np.trapezoid(ell * cl * jv(2, ell * t), ell) / (2 * np.pi) for t in th])


def beam(ell, fwhm):
    s = fwhm * np.pi / (180 * 60) / 2.3548
    return np.exp(-ell * (ell + 1) * s**2 / 2)


ell = E["a__cl_kappa_y__ell"]
ky, _, _ = stat("cl_kappa_y")
ky = ky * F_XPK

d = np.load(KS / "desact_data.npz")
nz_z, nz = d["nz_z"], d["nz_bins"]
cs_err = np.sqrt(np.diag(d["covmat"][-80:, -80:]))
Bl = beam(ell, Y_BEAM)

setup()
fig, axes = plt.subplots(1, 2, figsize=TWO_COL, constrained_layout=True)
for ax, bi, lab, ploc in [
    (axes[0], 2, r"DES bin 3 ($\bar z{\approx}0.74$)", "(a)"),
    (axes[1], 3, r"DES bin 4 ($\bar z{\approx}0.94$)", "(b)"),
]:
    m = d["cs_bin1"] == (bi + 1)
    th = d["cs_ang"][m]
    dat = d["cs_value"][m]
    err = cs_err[bi * 20 : (bi + 1) * 20]
    w = np.interp(ZS, nz_z, nz[bi], left=0, right=0)
    w = w / w.sum()
    cl_nodes = (w[None, :, None] * ky).sum(1) * Bl
    xi = np.array([xi_gy(th, ell, c) for c in cl_nodes])
    med = np.nanmedian(xi, 0)
    lo, hi = np.nanpercentile(xi, [16, 84], 0)
    raw = xi_gy(th, ell, (w[:, None] * np.nanmedian(ky, 0)).sum(0))  # no-beam -> exposes the shape

    ax.axvspan(2.5, TLO, color="0.85", alpha=0.5)
    ax.axvspan(THI, 200, color="0.85", alpha=0.5)
    ax.errorbar(th, dat * 1e9, yerr=err * 1e9, fmt="o", color=COLORS["truth"], ms=3,
                capsize=2, label="DES Y3$\\times$ACT")
    ax.fill_between(th, lo * 1e9, hi * 1e9, color=COLORS["bind"], alpha=BAND_ALPHA,
                     label="BIND Sobol 16-84%")
    ax.plot(th, med * 1e9, "-", color=COLORS["bind"], lw=1.6, label="BIND (beam)")
    ax.plot(th, raw * 1e9, ":", color=COLORS["bind"], lw=1.0, label="BIND (no beam)")

    sel = (th > TLO) & (th < THI) & (dat > 0)
    r = np.nanmedian(med[sel] / dat[sel])
    ax.text(0.05, 0.05, f"robust $8$-$40'$:\nBIND/data $\\approx${r:.1f}$\\times$",
            transform=ax.transAxes, fontsize=6.3)
    ax.text(0.05, 0.86, lab, transform=ax.transAxes, fontsize=7.5, va="top")
    ax.set_xscale("log")
    ax.set_xlim(2.5, 150)
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$\xi_{\gamma y}\times10^{9}$")
    ax.legend(fontsize=5.6, loc="upper right")
    # every in-axes corner is occupied (curve/legend/DES-bin label/robust-
    # window text); put the panel tag just above the axes instead.
    ax.text(0.0, 1.02, ploc, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", clip_on=False)

    print(f"bin{bi + 1}: robust(8-40') BIND/data = {r:.2f}x")

save(fig, "figs/fig12_desact_sheary")
