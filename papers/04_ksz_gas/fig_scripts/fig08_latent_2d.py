"""fig08_latent_2d.py

The paper's central "idea" figure (Sec. 4 / "Results 5"): the 30-dimensional
CAMELS-TNG feedback response in log(tau), log(y) collapses onto a ~2-D
latent manifold (97% of the response variance in 2 components). The 2
latent axes are then rotated to align with physically interpretable
directions and shown to correlate at r=+0.95 with the INNER gas fraction
f_gas(<R500) (axis 1) and the OUTER gas fraction f_gas(R500->R200) (axis 2)
respectively -- i.e. feedback depletes the core and outskirts
semi-independently.

(a) scree plot: 2 latents = 97% of response variance.
(b) parameter loadings of the 2 (sign-fixed, physically-oriented) axes.
(c) the 2-D latent plane colored by inner-gas f_gas(<R500).
(d) the SAME plane colored by outer-gas f_gas(R500->R200).

Data (pre-cached, read-only):
  - KS/bind_tauy_xprof_snap085.npz   (x, mbins, nodes, a, tau, y, cnts)
  - PARQUET columns: run,snap,M_tot_500,f_gas_500 (for `fg`, ported from
    cell 8) and run,snap,M200,M_gas_500,M_gas_200,M_tot_500,M_tot_200 (for
    the inner/outer physical gas-fraction labels, cell 10 itself) and
    run,<30 astro params> (for the loading bars).
  KS = /mnt/home/mlee1/ceph/bind_science/ksz_confront
  PARQUET = /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet

Source: examples/paper_ksz_desi_act.ipynb (branch analysis/ksz_project,
worktree ksz-desi-act), cell 10, built by
examples/_build_ksz_paper_nb.py lines ~508-589. IMPORTANT (per DATA_MAP):
cell 10 depends on in-notebook state (`x, tau, y, nodes, fg`) computed in
cell 8; the ~6 lines that build `x, tau, y, nodes, fg` below are ported
verbatim from cell 8 (see fig04's script for the sibling response-fan
figure that uses the same cell-8 state) before running cell 10's own
SVD/rotation body unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from importlib.resources import files as _ir_files
from numpy.linalg import svd
import matplotlib.pyplot as plt

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, save, panel_label, COLORS, TWO_COL_TALL

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
PARQUET = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet")
F_B = 0.0490 / 0.3089           # Omega_b/Omega_m (Planck/TNG)
BGS_BIN = 1                     # mass-bin index: logM200 in [13.4, 13.8] (BGS hosts)
SNAP_BGS = 85
MB = BGS_BIN

# --- the 30 CAMELS-IllustrisTNG astrophysical parameters (parquet column order),
#     ported verbatim from notebook cell 2 ("Sec 0 setup") ---
PARAMS = ['WindEnergyIn1e51erg', 'RadioFeedbackFactor', 'VariableWindVelFactor',
          'RadioFeedbackReiorientationFactor', 'MaxSfrTimescale', 'FactorForSofterEQS', 'IMFslope',
          'SNII_MinMass_Msun', 'ThermalWindFraction', 'VariableWindSpecMomentum', 'WindFreeTravelDensFac',
          'MinWindVel', 'WindEnergyReductionFactor', 'WindEnergyReductionMetallicity', 'WindEnergyReductionExponent',
          'WindDumpFactor', 'SeedBlackHoleMass', 'BlackHoleAccretionFactor', 'BlackHoleEddingtonFactor',
          'BlackHoleFeedbackFactor', 'BlackHoleRadiativeEfficiency', 'QuasarThreshold', 'QuasarThresholdPower',
          'UVBH0beta', 'UVBH0Deltaz', 'UVBHepbeta', 'UVBHepDeltaz', 'SNIa_Rate_Norm', 'SNIa_Rate_DTD_power',
          'SofteningComovingType01']
PARAMS = [p for p in PARAMS if p in set(pq.read_schema(PARQUET).names)]

setup()

# ============================================================
# Ported from cell 8: x, tau, y, nodes, fg (the response-fan state fig08 needs)
# ============================================================
c = np.load(KS / "bind_tauy_xprof_snap085.npz")
x, tau, y, nodes = c["x"], c["tau"], c["y"], c["nodes"]
fgdf = pd.read_parquet(PARQUET, columns=["run", "snap", "M_tot_500", "f_gas_500"])
fgdf = fgdf[(fgdf.snap == SNAP_BGS) & (np.log10(fgdf.M_tot_500) > 13.3)]
fg = (fgdf.groupby("run").f_gas_500.median() / F_B).reindex(nodes).values

# ============================================================
# cell 10 body (SVD / latent rotation / loadings), unchanged
# ============================================================
band = (x >= 0.3) & (x <= 1.5)
R = np.hstack([np.log10(np.clip(tau[:, MB, band], 1e-30, None)),
               np.log10(np.clip(y[:, MB, band], 1e-30, None))])
good = np.isfinite(R).all(1)
Rs = (R[good] - R[good].mean(0)) / R[good].std(0)
U, S, Vt = svd(Rs - Rs.mean(0), full_matrices=False)
lam = S ** 2 / np.sum(S ** 2)
Z = U * S
fg_g = fg[good]
mfg = np.isfinite(fg_g)
grad = np.array([np.cov(Z[mfg, k], fg_g[mfg])[0, 1] for k in range(2)])
e1 = grad / np.linalg.norm(grad)
e2 = np.array([-e1[1], e1[0]])

fo = pd.read_parquet(PARQUET, columns=["run", "snap", "M200", "M_gas_500", "M_gas_200", "M_tot_500", "M_tot_200"])
_lo = np.log10(fo.M200.values)
fo = fo[(fo.snap.values == SNAP_BGS) & (_lo > 13.4) & (_lo < 13.8)].copy()
fo["f_in"] = fo.M_gas_500.values / fo.M_tot_500.values / F_B
fo["f_out"] = (fo.M_gas_200.values - fo.M_gas_500.values) / (fo.M_tot_200.values - fo.M_tot_500.values) / F_B
gf = fo.groupby("run").median(numeric_only=True).reindex(nodes[good])
f_in, f_out = gf.f_in.values, gf.f_out.values

Ze = Z[:, :2] @ np.c_[e1, e2]
mo = np.isfinite(f_out) & np.isfinite(Ze[:, 1])
if np.corrcoef(f_out[mo], Ze[mo, 1])[0, 1] < 0:
    e2 = -e2
Ze = Z[:, :2] @ np.c_[e1, e2]
r_ej = np.corrcoef(Ze[mfg, 0], fg_g[mfg])[0, 1]
r_sh = np.corrcoef(Ze[mfg, 1], fg_g[mfg])[0, 1]

Pmat = pd.read_parquet(PARQUET, columns=["run"] + PARAMS).groupby("run").first().reindex(nodes[good]).values
okp = np.isfinite(Pmat).all(1)
Pz = (Pmat[okp] - Pmat[okp].mean(0)) / Pmat[okp].std(0)
LOAD = np.zeros((2, len(PARAMS)))
for k in range(2):
    zk = (Ze[okp, k] - Ze[okp, k].mean()) / Ze[okp, k].std()
    LOAD[k] = Pz.T @ zk / len(zk)

mi = np.isfinite(f_in) & np.isfinite(Ze[:, 0])
mo = np.isfinite(f_out) & np.isfinite(Ze[:, 1])
r_in_e1 = np.corrcoef(f_in[mi], Ze[mi, 0])[0, 1]
r_out_e2 = np.corrcoef(f_out[mo], Ze[mo, 1])[0, 1]
mio = np.isfinite(f_in) & np.isfinite(f_out)
r_io = np.corrcoef(f_in[mio], f_out[mio])[0, 1]


def _R2(yv, X, mm):
    Xc = np.column_stack([np.ones(mm.sum())] + [col[mm] for col in X])
    b = np.linalg.lstsq(Xc, yv[mm], rcond=None)[0]
    return 1 - ((yv[mm] - Xc @ b) ** 2).sum() / ((yv[mm] - yv[mm].mean()) ** 2).sum()


r2_e2 = _R2(Ze[:, 1], [f_in, f_out], mio)
r2_e1 = _R2(Ze[:, 0], [f_in, f_out], mio)

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(2, 2, figsize=TWO_COL_TALL, constrained_layout=True)

# (a) scree
ax[0, 0].bar(np.arange(1, 7), lam[:6], color=COLORS["bind"], alpha=0.8)
ax[0, 0].plot(np.arange(1, 7), np.cumsum(lam[:6]), "ko-", ms=4, lw=1, label="cumulative")
ax[0, 0].axhline(0.95, color=COLORS["highlight"], ls=":", lw=1)
ax[0, 0].text(1.5, 0.965, "95%", color=COLORS["highlight"], fontsize=7)
ax[0, 0].set_xlabel("latent component")
ax[0, 0].set_ylabel("response variance fraction")
ax[0, 0].set_ylim(0, 1.05)
ax[0, 0].legend(loc="center right")
ax[0, 0].text(0.5, 0.5, f"2 latents\n={100 * np.cumsum(lam)[1]:.0f}%", transform=ax[0, 0].transAxes,
              fontsize=8, ha="center")
panel_label(ax[0, 0], "(a)")

# (b) parameter loadings of the two axes
top = np.argsort(np.maximum(np.abs(LOAD[0]), np.abs(LOAD[1])))[::-1][:9][::-1]
yp = np.arange(len(top))
ax[0, 1].barh(yp - 0.2, LOAD[0, top], 0.38, color=COLORS["bind"], label=r"$\hat e_1$ inner gas")
ax[0, 1].barh(yp + 0.2, LOAD[1, top], 0.38, color=COLORS["secondary"], label=r"$\hat e_2$ outer gas")
ax[0, 1].axvline(0, color="k", lw=0.6)
ax[0, 1].set_yticks(yp)
ax[0, 1].set_yticklabels([PARAMS[i][:18] for i in top])
ax[0, 1].set_xlabel(r"loading corr$(\theta_j, Z)$")
ax[0, 1].legend(loc="lower right")
panel_label(ax[0, 1], "(b)")

# (c) plane colored by INNER gas f_gas(<R500)
sc1 = ax[1, 0].scatter(Ze[mi, 0], Ze[mi, 1], c=f_in[mi], cmap="cividis", s=14, edgecolor="0.3", lw=0.2)
ax[1, 0].annotate("", xy=(0.78, 0.5), xytext=(0.22, 0.5), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
ax[1, 0].text(0.5, 0.55, f"inner $f_{{\\rm gas}}\\!\\uparrow$  ($r{{=}}{r_in_e1:+.2f}$)",
              transform=ax[1, 0].transAxes, fontsize=7, ha="center")
ax[1, 0].set_xlabel(r"inner-gas latent $\hat e_1$")
ax[1, 0].set_ylabel(r"outer-gas latent $\hat e_2$")
cb1 = fig.colorbar(sc1, ax=ax[1, 0], fraction=0.046, pad=0.02)
cb1.set_label(r"$\tilde f_{\rm gas}(<R_{500})$", fontsize=7)
panel_label(ax[1, 0], "(c)")

# (d) SAME plane colored by OUTER gas f_gas(R500->R200)
sc2 = ax[1, 1].scatter(Ze[mo, 0], Ze[mo, 1], c=f_out[mo], cmap="cividis", s=14, edgecolor="0.3", lw=0.2)
ax[1, 1].annotate("", xy=(0.5, 0.80), xytext=(0.5, 0.20), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
ax[1, 1].text(0.6, 0.5, f"outer $f_{{\\rm gas}}\\!\\uparrow$  ($r{{=}}{r_out_e2:+.2f}$)",
              transform=ax[1, 1].transAxes, fontsize=7, ha="left", va="center")
ax[1, 1].set_xlabel(r"inner-gas latent $\hat e_1$")
ax[1, 1].set_ylabel(r"outer-gas latent $\hat e_2$")
cb2 = fig.colorbar(sc2, ax=ax[1, 1], fraction=0.046, pad=0.02)
cb2.set_label(r"$\tilde f_{\rm gas}(R_{500}{\to}R_{200})$", fontsize=7)
ax[1, 1].text(0.03, 0.03,
              f"inner/outer corr$={r_io:.2f}$\n$R^2(\\hat e_1){{=}}{r2_e1:.2f}$, $R^2(\\hat e_2){{=}}{r2_e2:.2f}$",
              transform=ax[1, 1].transAxes, fontsize=7, va="bottom", ha="left",
              bbox=dict(boxstyle="round", fc="white", ec="0.7", lw=0.3, alpha=0.85))
panel_label(ax[1, 1], "(d)")

save(fig, "figs/fig08_latent_2d")
