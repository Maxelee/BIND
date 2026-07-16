"""fig10_field_desY3_act.png -- BIND ray-traced shear x y vs REAL DES Y3 x ACT.

Two panels, DES source bins 3 and 4: the ray-traced BIND shear-y cross
correlation xi_gamma_y(theta) (16-84% band over the 253-node Sobol suite,
2.4'-beam-convolved solid curve, no-beam dotted curve) against the real
DES Y3 x ACT measurement (points with errors from the joint covariance).
Grey bands mark outside the robust 8'-40' window (finite-field low-ell cut
on one side, beam/resolution on the other). In-panel text reports the
robust-window BIND/data amplitude ratio.

Data:
  - /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz (DS) --
    a__cl_kappa_y__ell, t__cl_kappa_y__value/valid.
  - /mnt/home/mlee1/ceph/bind_science/ksz_confront/desact_data.npz (KS/desact_data.npz) --
    cs_bin1, cs_ang, cs_value, covmat, nz_z, nz_bins (real DES Y3 x ACT
    compton_shear cross-correlation + n(z) + joint covariance).

Source: examples/paper_ksz_field.ipynb (worktree branch analysis/ksz_project,
  wt/ksz-desi-act), cell 12, section "Sec 5 - Confrontation - BIND shear x y
  vs the real DES Y3 x ACT". Built by examples/_build_ksz_field_nb.py lines
  ~359-392. Saved originally as g5_desact_confront.pdf. Ports the F_XPK
  Pylians-normalization constant and the stat() accessor from the notebook's
  constants cell (cell 2), plus the in-cell xi_gy Hankel-transform / beam
  helpers (scipy.special.jv), which are NOT cached data -- they are the
  (documented, in-repo) analytic projection/beam formulas applied to the
  cached C_ell^{kappa y}.

Note (style): the original notebook figure carried a fig.suptitle
("~1.5-2x high ... too steep") that the paper's caption itself flags as a
STALE rounding of the actual 2.5x/2.3x values (see main.tex ~ line 770).
FIGURE_STYLE.md forbids titles/suptitles suite-wide, so this regeneration
drops it outright -- which also removes the staleness the caption was
correcting for. The precise per-panel ratios (recomputed here, matching the
body text: bin3 ~2.5x, bin4 ~2.3x) are kept as in-axes annotations.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import jv

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import BAND_ALPHA, COLORS, TWO_COL, panel_label, save, setup

# ---- constants-cell dependencies (ported from paper_ksz_field.ipynb cell 2) ----
CEPH = Path("/mnt/home/mlee1/ceph")
DS = CEPH / "bind_sb35/emulator_dataset.npz"
KS = CEPH / "bind_science/ksz_confront"
F_XPK = 1.2016e7  # Pylians XPk_plane normalization fix (C_l^{kappa y})

E = np.load(DS, allow_pickle=True)
ZS = E["source_redshifts"]


def stat(name):
    v = E[f"t__{name}__value"]
    ok = E[f"t__{name}__valid"]
    err = E[f"t__{name}__err"] if f"t__{name}__err" in E.files else None
    return v[ok], (err[ok] if err is not None else None), ok


# ---- cell 12 body (verbatim math) ----
Y_BEAM, TLO, THI = 2.4, 8.0, 40.0  # ACT DR6 physical beam [arcmin]; robust window


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

# ---- plot ----
setup()
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=TWO_COL, constrained_layout=True)
panels = [(axes[0], 2, r"(a) DES bin 3 ($\bar z{\approx}0.74$)"),
          (axes[1], 3, r"(b) DES bin 4 ($\bar z{\approx}0.94$)")]
for ax, bi, label in panels:
    m = d["cs_bin1"] == (bi + 1)
    th = d["cs_ang"][m]
    dat = d["cs_value"][m]
    err = cs_err[bi * 20:(bi + 1) * 20]
    w = np.interp(ZS, nz_z, nz[bi], left=0, right=0)
    w = w / w.sum()
    cl_nodes = (w[None, :, None] * ky).sum(1) * Bl
    xi = np.array([xi_gy(th, ell, c) for c in cl_nodes])
    med = np.nanmedian(xi, 0)
    lo, hi = np.nanpercentile(xi, [16, 84], 0)
    raw = xi_gy(th, ell, (w[:, None] * np.nanmedian(ky, 0)).sum(0))  # no-beam -> exposes the shape

    ax.axvspan(2.5, TLO, color="0.85", alpha=0.5)
    ax.axvspan(THI, 200, color="0.85", alpha=0.5)
    ax.text(4.2, 0.1, "beam/res", fontsize=7, rotation=90, color="0.5", va="bottom", transform=ax.get_xaxis_transform())
    h_dat = ax.errorbar(th, dat * 1e9, yerr=err * 1e9, fmt="o", color=COLORS["truth"], ms=3, capsize=2, label="DES Y3 $\\times$ ACT")
    h_band = ax.fill_between(th, lo * 1e9, hi * 1e9, color=COLORS["bind"], alpha=BAND_ALPHA, label="BIND Sobol 16–84%")
    (h_med,) = ax.plot(th, med * 1e9, "-", color=COLORS["bind"], lw=1.6, label="BIND ($2.4'$ beam)")
    (h_raw,) = ax.plot(th, raw * 1e9, ":", color=COLORS["bind"], lw=1.0, label="BIND (no beam)")

    sel = (th > TLO) & (th < THI) & (dat > 0)
    r = np.nanmedian(med[sel] / dat[sel])
    ax.text(0.05, 0.05, f"robust $8$–$40'$:\nBIND/data $\\approx${r:.1f}$\\times$", transform=ax.transAxes, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(2.5, 150)
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$\xi_{\gamma y}\times10^{9}$")
    ax.legend(handles=[h_dat, h_band, h_med, h_raw], loc="upper right")
    panel_label(ax, label, loc="upper left")
    print(f"bin{bi + 1}: robust(8-40') BIND/data = {r:.2f}x")

save(fig, "figs/fig10_field_desY3_act")
