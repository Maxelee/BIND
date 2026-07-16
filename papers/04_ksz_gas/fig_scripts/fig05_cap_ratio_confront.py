"""fig05_cap_ratio_confront.py

The corrected, final kSZ-amplitude confrontation (paper Sec. "Confrontation
II"): the sigma_v-free, M*-matched CAP-ratio gas fraction f~gas(theta) =
CAP_gas/CAP_mat for the BIND fiducial run at the two DESI stellar-mass cuts
(M*>1e11.00, M*>1e11.25), the 256-node SB35 feedback-Sobol 16-84% band, and
the single best-chi2 ("coherent") node, against the real DESI DR2 x ACT DR6
CAP-ratio data points with covariance-derived errors (Zenodo 19160138).
Supersedes the retracted cumulative-aperture CAP numbers discussed in the
paper text.

Data (pre-cached, read-only):
  - KS/fgas_cap_mstar_snap085.npz
      keys: xb, fiducial, sb35, node_ids, logM200, cuts
  - KS/desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm11.00.npz
  - KS/desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm11.25.npz
      keys (both): th, ratio, yerr, prof_kappa_err, cov_ksz
  KS = /mnt/home/mlee1/ceph/bind_science/ksz_confront

Source: examples/paper_ksz_desi_act.ipynb (branch analysis/ksz_project,
worktree ksz-desi-act), cell 14, built by
examples/_build_ksz_paper_nb.py lines ~685-717. Cosmology helpers
(r200phys, DA, ARCMIN) ported verbatim from the notebook's cell 2 (Sec. 0
setup) -- the CAP-ratio arrays themselves are entirely pre-computed in the
npz files; this script only re-derives theta(r200) and the chi2 best-fit
node selection from the cached profiles + real covariance.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, save, ONE_COL

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")

# --- cosmology helpers, ported from notebook cell 2 ("Sec 0 setup") ---
Om, OL, h = 0.3089, 0.6911, 0.6774
C_KMS, H0, ARCMIN = 299792.458, 100 * 0.6774, 180 * 60 / np.pi


def Ez(z):
    return np.sqrt(Om * (1 + z) ** 3 + OL)


def DA(z, n=3000):
    zz = np.linspace(0, z, n)
    return (C_KMS / H0) * np.trapezoid(1 / Ez(zz), zz) / (1 + z)


def r200phys(logM200, z):
    M = 10 ** logM200 / h
    rho = 2.775e11 * h ** 2 * Ez(z) ** 2
    return (3 * M / (4 * np.pi * 200 * rho)) ** (1 / 3)


setup()

fr = np.load(KS / "fgas_cap_mstar_snap085.npz")
xb, fidp, sbp, lmcut = fr["xb"], fr["fiducial"], fr["sb35"], fr["logM200"]
zc = 0.26
ZEN = KS / "desact_zenodo"
DCUT = ["11.00", "11.25"]

fig, ax = plt.subplots(figsize=ONE_COL)
# caption-locked colors: green = M*>11.0 cut, orange = M*>11.25 cut
cuts = [("M*>11.0", "tab:green"), ("M*>11.25", "tab:orange")]
for k, (lbl, col) in enumerate(cuts):
    thx = xb * r200phys(lmcut[k], zc) / DA(zc) * ARCMIN
    thr = r200phys(lmcut[k], zc) / DA(zc) * ARCMIN
    dz = np.load(ZEN / f"Fig8_BGS_BRIGHT-20.2_logm{DCUT[k]}.npz")
    lo, hi = np.nanpercentile(sbp[:, k], [16, 84], 0)
    ax.fill_between(thx, lo, hi, color=col, alpha=0.15)
    ax.plot(thx, fidp[k], "-", color=col, lw=1.6, label=f"BIND fid {lbl}")
    # single coherent best-fit node: chi2 against the real covariance
    m1 = dz["th"] <= 1.4 * thr
    Cinv = np.linalg.inv(dz["cov_ksz"][np.ix_(m1, m1)])
    pn = np.array([np.interp(dz["th"][m1], thx, sbp[i, k]) for i in range(sbp.shape[0])])
    chi2 = np.einsum("ni,ij,nj->n", pn - dz["ratio"][m1], Cinv, pn - dz["ratio"][m1])
    bi = int(np.nanargmin(chi2))
    ax.plot(thx, sbp[bi, k], "-", color=col, lw=0.8, alpha=0.85)
    ax.errorbar(dz["th"], dz["ratio"], yerr=dz["yerr"], fmt="o", color=col, ms=3.5,
                capsize=1.5, alpha=0.9, label=f"data {lbl}")
    ax.axvline(thr, color=col, ls=":", lw=0.6, alpha=0.7)

ax.axhline(1, color="k", ls=":", lw=0.8)
ax.text(8.6, 1.03, "cosmic", fontsize=7)
ax.fill_between([], [], [], color="0.6", alpha=0.25, label="SB35 16-84%")
ax.plot([], [], "-", color="0.45", lw=0.8, label="best-fit node")
ax.set_xlabel(r"$\theta$ [arcmin]  (dotted: $\theta(r_{200})$)")
ax.set_ylabel(r"$\tilde f_{\rm gas}\,(\Omega_m/\Omega_b)$")
ax.set_ylim(0, 1.45)
ax.set_xlim(0, 10.5)
ax.legend(loc="lower right", ncol=1)
save(fig, "figs/fig05_cap_ratio_confront")
