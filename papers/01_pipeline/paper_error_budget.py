#!/usr/bin/env python
"""paper_error_budget.py -- the analytic model's error budget, decomposed
(pfig_s4b_error_budget, 2026-08-06).

The shipped model error sigma_model(ell) (the CV residual, stored in
latent_model_coeffs.npz and used by the predictor and the corner) LUMPS four
physically distinct sources. This script measures each one:

  1. sigma_S,meas(ell): realization noise on a node's measured S -- std over
     the 50 seed-paired fiducial/DMO realization ratios, band-averaged, /sqrt(50).
  2. sigma_lambda -> S: latent measurement noise propagated through the
     kernels, c(ell)^T Sigma_lambda c(ell). Sigma_lambda (4x4, WITH
     cross-latent correlations -- f~_bar and f~_star share halos) is
     estimated deterministically by SPLIT-HALF: each node's bin halos are
     split even/odd by mass rank; Sigma_lambda = cov(median_A - median_B)/4
     across nodes.
  3. kernel-coefficient uncertainty: prediction variance from Cov(beta) =
     sigma_res^2 (A^T A)^-1 at the average leverage, = sigma_CV^2 * 5/N.
  4. sigma_int: what remains of sigma_CV^2 in quadrature -- genuine model
     deficiency + paint stochasticity (the irreducible part).

Also reports the errors-in-variables attenuation (reliability) per latent:
R_i = sigma_cloud,i^2 / (sigma_cloud,i^2 + sigma_lambda,i^2) -- how much the
latent measurement noise biases the fitted kernels toward zero.

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python paper_error_budget.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SB35 = CEPH / "bind_sb35"
SCI = CEPH / "bind_science"
assert Path.cwd().name == "01_pipeline", "run from papers/01_pipeline/"
OB_OM = 0.0486 / 0.3089
ZI = 1

coef = np.load("latent_model_coeffs.npz")
assert int(np.atleast_1d(coef["version"])[0]) == 2
zi = int(np.argmin(np.abs(coef["zs"] - 1.0)))
B = coef["beta"][zi][:, :4]                       # (24, 4)
sig_cv = coef["sig_model"][zi]                    # (24,)
ctr = coef["ell"]
EDG = coef["ell_edges"]

d = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
run_ids = d["run_ids"]
ELL = d["a__suppression__ell"]
n = len(run_ids)
N_FIT = int(coef["n_nodes"])

# ── 1. S measurement noise from the 50 seed-paired realizations ─────────────
clk = np.load(SCI / "runs/bind/run_0000/paired_perreal_fid.npz")["clk"][:, ZI, :]
cld = np.load(SCI / "runs/dmo/run_0000/Cl_kappa_paired.npz")["cl_real"][:, ZI, :]
ratio = clk / cld                                  # (50, 724)
rb = np.stack([ratio[:, (ELL >= EDG[i]) & (ELL < EDG[i + 1])].mean(1)
               for i in range(24)], axis=1)        # (50, 24)
sig_Smeas = rb.std(0, ddof=1) / np.sqrt(rb.shape[0])

# ── 2. latent measurement covariance via deterministic split-half ───────────
cz = np.load(SB35 / "analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
mt = cz["sobol_m_tot_500c_bg"][run_ids]
mg = cz["sobol_m_gas_500c_bg"][run_ids]
g2 = cz["sobol_m_gas_200c_bg"][run_ids]
ms = cz["sobol_m_star_500c"][run_ids]
Tm = cz["sobol_T_mw_500c"][run_ids]
lm = np.log10(np.where(mt > 0, mt, np.nan))

D_half = np.full((n, 4), np.nan)
LATv = np.full((n, 4), np.nan)
for q in range(n):
    s = np.where((lm[q] >= 13.3) & (lm[q] < 13.6) & (g2[q] > 0)
                 & (Tm[q] > 0))[0]
    if len(s) < 10:
        continue
    order = s[np.argsort(mt[q, s])]                # mass-ranked, deterministic
    A_h, B_h = order[0::2], order[1::2]
    per = {
        0: lambda idx: np.median(((mg + ms) / mt)[q, idx]) / OB_OM,
        1: lambda idx: np.median((ms / mt)[q, idx]) / OB_OM,
        2: lambda idx: np.median((mg / g2)[q, idx]),
        3: lambda idx: np.log10(np.median(Tm[q, idx])),
    }
    for i in range(4):
        LATv[q, i] = per[i](order)
        D_half[q, i] = per[i](A_h) - per[i](B_h)
okq = np.isfinite(D_half).all(1) & np.isfinite(LATv).all(1)
Sig_lam = np.cov(D_half[okq].T) / 4.0              # (4, 4)
sig_lam = np.sqrt(np.diag(Sig_lam))
sig_cloud = LATv[okq].std(0)
reliab = sig_cloud**2 / (sig_cloud**2 + sig_lam**2)

# ── 3. propagate + assemble the budget per band ─────────────────────────────
var_lam = np.einsum("bi,ij,bj->b", B, Sig_lam, B)  # c^T Sigma_lambda c
var_beta = sig_cv**2 * 5.0 / N_FIT                 # average-leverage term
var_int = np.clip(sig_cv**2 - var_lam - sig_Smeas**2 - var_beta, 0, None)

LN = ["f~_bar", "f~_star", "c_gas", "logT"]
print("latent measurement errors (split-half, median-of-bin):")
for i in range(4):
    print(f"  {LN[i]:8s}: sigma_lam = {sig_lam[i]:.4f}  (cloud std "
          f"{sig_cloud[i]:.4f}; reliability R = {reliab[i]:.3f})")
print("latent-error correlation matrix (off-diagonals):")
Cl_corr = Sig_lam / np.outer(sig_lam, sig_lam)
for i in range(4):
    print("   " + " ".join(f"{Cl_corr[i, j]:+.2f}" for j in range(4)))

print("\nerror budget for S(ell), per band [sigma units of S]:")
print(f"{'ell':>7s} {'sig_CV':>8s} {'S-meas':>8s} {'lam->S':>8s} "
      f"{'kernel':>8s} {'intrinsic':>9s}")
for ll in (1e3, 5e3, 1.9e4):
    i = int(np.argmin(np.abs(ctr - ll)))
    print(f"{ctr[i]:7.0f} {sig_cv[i]:8.4f} {sig_Smeas[i]:8.4f} "
          f"{np.sqrt(var_lam[i]):8.4f} {np.sqrt(var_beta[i]):8.4f} "
          f"{np.sqrt(var_int[i]):9.4f}")

# ── figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(ONE_COL[0] * 1.35, 3.0))
ax.plot(ctr, sig_cv, color="k", lw=1.8, label=r"total model error $\sigma_{\rm CV}$")
ax.plot(ctr, np.sqrt(var_int), color=COLORS["bind"], lw=1.3, ls="--",
        label="intrinsic (model + paint stoch.)")
ax.plot(ctr, np.sqrt(var_lam), color=COLORS["highlight"], lw=1.3, ls="-.",
        label=r"$\lambda$ measurement $\to$ kernels")
ax.plot(ctr, sig_Smeas, color=COLORS["secondary"], lw=1.3, ls=":",
        label=r"$S$ realization noise (50 paired)")
ax.plot(ctr, np.sqrt(var_beta), color=COLORS["dmo"], lw=1.1, ls=":",
        label="kernel-coefficient uncertainty")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(300, 3e4)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"contribution to $\sigma[S(\ell)]$", fontsize=6.5)
ax.legend(fontsize=5.2, loc="upper left", handletextpad=0.5)
fig.tight_layout()
save(fig, "figs_v2/pfig_s4b_error_budget")
plt.close(fig)
print("\nwrote pfig_s4b_error_budget; note the quadrature identity "
      "sig_CV^2 = S-meas^2 + lam^2 + kernel^2 + intrinsic^2 holds by "
      "construction of the intrinsic term (clipped at 0)")
