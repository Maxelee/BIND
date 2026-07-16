"""fig09_field_1d_latent.png -- the field-level feedback response is ~1-D.

Panels:
  (a) scree plot of the field-level response covariance (tomographic
      log10-suppression S(ell, z_s) stacked over all 5 source redshifts,
      plus the mean-subtracted kappa-y cross C_ell^{kappa y}(ell, z_s), also
      stacked over z_s) -- component 1 already exceeds the 95%-variance
      line, component 2 is essentially flat.
  (b) the dominant "gas-amplitude" latent e1 vs the weak 2nd latent e2,
      points colored by the per-node group/cluster gas fraction f_gas
      (in cosmic units, Omega_b/Omega_m).
  (c) parameter loadings (correlation of each of the 30 astro params with
      the two latent scores) for the top 9 most-loading parameters.

Data: /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz (DS in DATA_MAP.md)
  keys used: source_redshifts, param_names, X_unit,
             t__suppression__value/valid, a__suppression__ell (unused directly;
             the ell grid used for the band-select is a__cl_kappa__ell, which
             is identical to a__suppression__ell -- confirmed by
             np.array_equal on load),
             t__cl_kappa_y__value/valid, a__cl_kappa__ell,
             t__scaling_f_gas__value/valid.

Source: examples/paper_ksz_field.ipynb (worktree branch analysis/ksz_project,
  wt/ksz-desi-act), cell 10, section "Sec 4 - The field response compresses to
  ~1 latent". Built by examples/_build_ksz_field_nb.py lines ~292-330. Saved
  originally as g4_latent.pdf. This script also ports the handful of lines
  from the notebook's constants cell (cell 2) that cell 10 depends on:
  the `stat()` accessor, the F_B baryon-fraction constant, and the per-node
  `fgas` gas-fraction summary. F_XPK (the Pylians XPk_plane normalization
  constant) is ALSO ported per DATA_MAP.md's note, but is not actually
  applied anywhere in cell 10's math -- the SVD/regression below is invariant
  to an overall linear rescaling of the kappa-y cross (it gets standardized
  to unit variance before the SVD), so multiplying cky by F_XPK here would
  not change any of lam, r_fg, or LOAD. Kept only as a documented no-op for
  provenance parity with the source cell.
"""
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import svd

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup

# ---- constants-cell dependencies (ported from paper_ksz_field.ipynb cell 2) ----
DS = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
F_B = 0.0490 / 0.3089
F_XPK = 1.2016e7  # Pylians XPk_plane normalization fix; unused in this figure's math (see docstring)

E = np.load(DS, allow_pickle=True)
ZS = E["source_redshifts"]
PARAMS = [str(p) for p in E["param_names"]]


def stat(name):
    v = E[f"t__{name}__value"]
    ok = E[f"t__{name}__valid"]
    err = E[f"t__{name}__err"] if f"t__{name}__err" in E.files else None
    return v[ok], (err[ok] if err is not None else None), ok


fg_all = E["t__scaling_f_gas__value"]
fgas = np.nanmedian(fg_all[:, 2:5], axis=1) / F_B  # group-cluster gas fraction, cosmic units

# ---- cell 10 body (verbatim math) ----
S, _, _ = stat("suppression")
cky, _, _ = stat("cl_kappa_y")
ell = E["a__cl_kappa__ell"]
band = (ell > 300) & (ell < 8000)

blocks = [np.log10(np.clip(S[:, zj, band], 1e-3, None)) for zj in range(len(ZS))]
blocks += [cky[:, zj, band] - np.nanmean(cky[:, zj, band], 0) for zj in range(len(ZS))]
R = np.hstack(blocks)
good = np.isfinite(R).all(1)
Rs = (R[good] - R[good].mean(0)) / (R[good].std(0) + 1e-12)
U, Sv, Vt = svd(Rs - Rs.mean(0), full_matrices=False)
lam = Sv**2 / np.sum(Sv**2)
Z = U * Sv

fg_g = fgas[good]
mfg = np.isfinite(fg_g)
grad = np.array([np.cov(Z[mfg, k], fg_g[mfg])[0, 1] for k in range(2)])
e1 = grad / np.linalg.norm(grad)
e2 = np.array([-e1[1], e1[0]])
Ze = Z[:, :2] @ np.c_[e1, e2]
r_fg = np.corrcoef(Ze[mfg, 0], fg_g[mfg])[0, 1]

Xp = E["X_unit"][good]
okp = np.isfinite(Xp).all(1)
Xz = (Xp[okp] - Xp[okp].mean(0)) / (Xp[okp].std(0) + 1e-12)
LOAD = np.array(
    [(Xz.T @ ((Ze[okp, k] - Ze[okp, k].mean()) / Ze[okp, k].std())) / okp.sum() for k in range(2)]
)

print(
    f"field variance: {np.round(lam[:5], 3)} | cum@2={np.cumsum(lam)[1]:.3f} "
    f"| gas axis corr(f_gas)={r_fg:+.2f}"
)

# ---- plot ----
setup()
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(
    1, 3, figsize=(TWO_COL[0], TWO_COL[1]), gridspec_kw=dict(width_ratios=[1, 1.15, 1.4]),
    constrained_layout=True,
)

# (a) scree plot
ax[0].bar(np.arange(1, 7), lam[:6], color=COLORS["bind"], alpha=0.8)
ax[0].plot(np.arange(1, 7), np.cumsum(lam[:6]), "o-", color=COLORS["truth"], ms=4, lw=1, label="cumulative")
ax[0].axhline(0.95, color=COLORS["highlight"], ls=":", lw=1)
ax[0].set_ylim(0, 1.05)
ax[0].legend(loc="center right")
ax[0].text(0.5, 0.55, f"$\\lambda_1$={lam[0]:.2f}\n(~1-d)", transform=ax[0].transAxes, ha="center", fontsize=8)
ax[0].set_xlabel("latent component")
ax[0].set_ylabel("response variance fraction")
panel_label(ax[0], "(a)")

# (b) latent plane colored by f_gas
sc = ax[1].scatter(Ze[mfg, 0], Ze[mfg, 1], c=fg_g[mfg], s=15, edgecolor="0.3", lw=0.2)
ax[1].annotate(
    "", xy=(0.78, 0.5), xytext=(0.22, 0.5), xycoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="k", lw=1.4),
)
ax[1].text(0.5, 0.56, r"$f_{\rm gas}\!\uparrow$ (gas amplitude)", transform=ax[1].transAxes, ha="center", fontsize=7)
ax[1].text(0.04, 0.9, f"$r_{{f_{{\\rm gas}}}}{{=}}{r_fg:+.2f}$", transform=ax[1].transAxes, fontsize=7)
ax[1].set_xlabel(r"gas-amplitude latent $\hat e_1$")
ax[1].set_ylabel(rf"weak 2nd latent $\hat e_2$ (~{lam[1] * 100:.0f}%)")
cb = fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.02)
cb.set_label(r"$\tilde f_{\rm gas}\equiv f_{\rm gas}/(\Omega_b/\Omega_m)$", fontsize=7)
panel_label(ax[1], "(b)")

# (c) parameter loadings
top = np.argsort(np.maximum(np.abs(LOAD[0]), np.abs(LOAD[1])))[::-1][:9][::-1]
yp = np.arange(len(top))
ax[2].barh(yp - 0.2, LOAD[0, top], 0.38, color=COLORS["highlight"], label=r"$\hat e_1$ (gas)")
ax[2].barh(yp + 0.2, LOAD[1, top], 0.38, color=COLORS["secondary"], label=r"$\hat e_2$")
ax[2].axvline(0, color="k", lw=0.6)
ax[2].set_yticks(yp)
ax[2].set_yticklabels([PARAMS[i][:18] for i in top])
ax[2].set_xlabel(r"loading corr$(\theta_j, Z)$")
ax[2].legend(loc="lower right")
panel_label(ax[2], "(c)")

save(fig, "figs/fig09_field_1d_latent")
