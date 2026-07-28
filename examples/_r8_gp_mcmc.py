#!/usr/bin/env python
"""R8 (docs/p4c_referee_hardening_plan.md phase R8): inference upgrade.

Replaces Phase L's ESS~6 importance-sampling pseudo-posterior (weights
exp(-dchi2/2) over the 253 traced Sobol nodes) with a continuous, validated
posterior: GP emulators of the two data-vector predictions over the 30-dim
normalized SB35 design, fed to an emcee MCMC over 32 parameters (30 astro +
f_sat + dlogM nuisances), for three legs (kSZ-only, tSZ-only, joint).

Design choices (each a documented simplification forced by the 1-CPU /
~40-min budget -- flagged, not hidden):

1. **f_sat handling (tSZ leg).** R2's HOD population curves exist at three
   f_sat grid points (0.04/0.08/0.12). Rather than tripling the GP training
   set by adding f_sat as a 31st input (759 points), we emulate at the
   kappa-calibrated center f=0.08 and add a per-node LINEAR f-response
   (slope = (curve_f0.12 - curve_f0.04)/0.08, itself GP-emulated over the
   same 30-dim theta) -- the task brief explicitly sanctions this ("linear
   in f is adequate"), and it keeps the GP input space at 30-dim, which
   matters a lot for ARD optimizer cost (see #2).
2. **Shared-hyperparameter group GPs.** An ARD Matern+White kernel
   optimization in 30-dim costs ~8-10s on this 1-CPU box (benchmarked).
   Refitting independently per fit-range column (6-12 columns per leg)
   would eat the whole time budget. Instead we fit ONE ARD kernel per
   physically-correlated column GROUP (tsz-mean, tsz-slope, ksz-mean,
   gas-fraction) on that group's leading standardized-mean direction, then
   REUSE those fixed hyperparameters per column (GaussianProcessRegressor
   with optimizer=None -- just a linear solve, ~10ms). Columns within a
   leg are the same smooth physical field sampled at neighbouring
   apertures/f_sat values, so sharing length scales is a good
   approximation; per-column mean/std normalization still lets each
   column's amplitude float freely.
3. **GP-uncertainty inflation for MCMC.** T3/R6's BIND-side error term is
   a NODE-SPECIFIC 50-realization covariance (cov_map/n_r, Hartlap
   inflated) -- not defined continuously over theta. R8 substitutes the
   K-fold cross-validation-CALIBRATED GP predictive variance: a FIXED
   per-column floor (temperature x typical GP sigma from the production
   fit), which plays the same statistical role (BIND-side + emulator
   model uncertainty) but exists everywhere in the 30-dim continuum. Since
   this floor is a constant (not evaluated per-walker per-step), the
   MCMC's total leg covariance is a CONSTANT matrix, inverted ONCE outside
   the sampler -- this is what keeps 15k-step chains affordable (only the
   GP MEAN predict, ~2ms/column/step-batch, remains in the hot loop).
4. **Data covariance (fixed, theta-independent).** tSZ replicates T3's
   FULL error budget (Hartlap jackknife + correlated R5 CIB block +
   resample diagonal). kSZ replicates R6's unit-fixed cov_ksz (fix_cov),
   WITHOUT data-side Hartlap -- R6 documents that the data jackknife/
   bootstrap N is external to this repo and cannot be recovered.
5. **f_sat/dlogM/A_2h are tSZ-only nuisances** (R2/R3/R4 are tSZ-derived
   products; the plan's own R8(b) spec is "30 params x {R2 f_sat, R3 mass,
   R4 A_2h, R5 dust}"). All chains sample the SAME 33-dim state for a
   uniform corner/forest comparison; in the kSZ-only chain these three
   carry no likelihood information, so their marginal posterior equals
   their prior by construction -- documented, not a bug.
6. **A_2h (mid-flight addendum, after R4 completed).** R4.json quantifies
   an analytic 2-halo/unpainted-gas CAP-y template at the 6 T3 fit columns
   (16-30% of the data value, exceeds R4's own 10% action gate) -- its
   shape (`cap_2h_analytic`, already in the same y*arcmin^2 units as
   act_ycap_lrg_real.npz's mean_xb_cib17, verified byte-identical to R4's
   own "data_value" field) is added to the tSZ MODEL side with a free
   positive amplitude A_2h ~ N(1.0, 0.3) truncated>=0 (R4's analytic
   prediction is exactly A_2h=1; its own empirical cross-check gives
   1.19+/-0.22). This raises the state to 33 dims. A2h needs no GP (it is
   a fixed template x a free amplitude), so it adds ~zero MCMC cost. A
   shorter secondary "joint_noA2h" chain (same likelihood, A_2h term
   omitted) is run for a with/without comparison -- NOT one of the 3
   primary deliverable chains.

Outputs: LC/r8_posterior.npz, figs/R8_posterior_corner.png,
figs/R8_param_forest.png, verdicts/R8.json.

Usage
-----
    python examples/_r8_gp_mcmc.py                    # one process, full run

Or, driven as bounded/resumable stages (recommended on a shared/contended
1-CPU box -- each call is independently killable/restartable; MCMC state
persists incrementally to an emcee HDFBackend per leg under SCRATCH):
    python examples/_r8_gp_mcmc.py --stage gp
    python examples/_r8_gp_mcmc.py --stage chain --leg ksz          --budget 140
    python examples/_r8_gp_mcmc.py --stage chain --leg tsz          --budget 270
    python examples/_r8_gp_mcmc.py --stage chain --leg joint        --budget 410
    python examples/_r8_gp_mcmc.py --stage chain --leg joint_noA2h  --budget 410
    # (repeat any --stage chain call to add more steps/resume)
    python examples/_r8_gp_mcmc.py --stage finalize
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cloudpickle   # can pickle the closures (loglike functions capture fitted GPs)
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CEPH = Path("/mnt/home/mlee1/ceph")
# Products/bind_sb35 roots (Round-2 T3, docs/paper_improvement_plan.md):
# env-var override only -- deliberately NOT wired into the --stage CLI below
# (this script drives live, long-running MCMC chains; keep the CLI surface
# unchanged). Behavior with neither env var set is byte-identical to before.
# SCRATCH/INGREDIENTS_PATH below are NOT derived from KS and are untouched --
# they are the live chain-checkpoint location and must not move.
KS = Path(os.environ.get("BIND_KSZ_PRODUCTS", str(CEPH / "bind_science/ksz_confront")))
LC = KS / "lightcone"
FIG_DIR = LC / "figs"
VERDICT_DIR = LC / "verdicts"
_SB35_ROOT = Path(os.environ.get("BIND_SB35_RUNS", str(CEPH / "bind_sb35")))
DESIGN = _SB35_ROOT / "design"
PARQUET = _SB35_ROOT / "analysis_cache/integrated.parquet"
FIG_DIR.mkdir(parents=True, exist_ok=True)
VERDICT_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint/cache dir for the multi-process CLI stages (--stage gp/chain/finalize)
# -- NOT a deliverable location, just scratch state so a slow 1-CPU MCMC run can
# be driven as a series of bounded, resumable foreground calls (HDFBackend per
# leg persists incrementally; killing/restarting a --stage chain call resumes
# from the last completed step).
# Relocated to ceph 2026-07-27: the chain-extension sbatch runs on cluster
# nodes where the workstation's /tmp session scratchpad does not exist (the
# first run_r8_chains.sh submission died on exactly this).
SCRATCH = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone/r8_state")
SCRATCH.mkdir(parents=True, exist_ok=True)
INGREDIENTS_PATH = SCRATCH / "r8_ingredients.pkl"

BIND_PIX_AREA = 0.29296875 ** 2
F_EFF0 = 0.08          # R2 kappa-calibrated center f_sat
XB18 = np.linspace(0.3, 3.0, 18)   # lightcone_cap_stack.XB, verbatim
SNAP_BGS, LOGM_LO, LOGM_HI = 85, 13.4, 13.8   # gas_fracs mass bin, verbatim
F_B_GASPLANE = 0.0490 / 0.3089

# R4 two-halo template amplitude (mid-flight addition, docs/p4c_referee_hardening_plan.md
# R8(b): "MCMC over 30 params x {R2 f_sat, R3 mass, R4 A_2h, R5 dust} nuisances" --
# R4.json quantifies an analytic 2-halo/unpainted-gas CAP-y template at the 6 T3 fit
# columns (16-30% of the data value, exceeds the 10% action gate) and its own "next"
# field asks for exactly this: fold A_2h into the R8 MCMC nuisance set.
A2H_LOC, A2H_SCALE, A2H_LO, A2H_HI = 1.0, 0.3, 0.0, 4.0   # truncated at 0, generous soft cap

N_FOLD = 3   # reduced from 5: this 1-CPU box is shared/contended (observed ARD
             # fits running several x slower than an isolated benchmark), and
             # each group-GP ARD optimization costs ~10-30s -- 3-fold keeps
             # --stage gp safely inside one bounded foreground call
GP_SEED = 0
NWALKERS = 72   # must be >= 2*NDIM=66 (emcee red-blue move requirement); 72 for margin
NSTEPS = 15000                  # fallback for run_chain(time_budget_sec=None)
TOTAL_MCMC_BUDGET_SEC = 1500.0  # adaptive wall-clock budget, split across 4 chains
N_INIT_CANDIDATES = 4000
N_NULL_DRAWS = 200
NULL_TARGET_ESS = 6.0

RNG = np.random.default_rng(20260727)

t_start = time.time()


def log(msg):
    print(f"[R8 {time.time()-t_start:7.1f}s] {msg}", flush=True)


# ============================================================================
# 0. prior box (30-dim normalized SB35 design)
# ============================================================================

def load_prior_box():
    from bind.params import PARAM_LOG_FLAG
    sobol = np.load(DESIGN / "astro_params_sobol.npy")            # (256, 35)
    dj = json.load(open(DESIGN / "design.json"))
    names, aidx = dj["astro_param_names"], np.array(dj["astro_param_indices"])
    logflag = (PARAM_LOG_FLAG[aidx] == 1)

    raw_all = sobol[:, aidx]                                       # (256, 30)
    Vn_all = np.where(logflag[None, :], np.log10(np.clip(raw_all, 1e-30, None)), raw_all)
    vmin, vmax = Vn_all.min(0), Vn_all.max(0)
    U_all = (Vn_all - vmin) / (vmax - vmin)                        # (256, 30) in [0,1]
    prior_q = np.percentile(U_all, [16, 50, 84], axis=0).T         # (30, 3), L-convention
    return dict(names=names, aidx=aidx, logflag=logflag, vmin=vmin, vmax=vmax,
                U_all=U_all, prior_q=prior_q, sobol=sobol)


def u_to_physical(u, vmin, vmax, logflag):
    """u in [0,1]^30 -> physical param values (undo log10-normalization)."""
    Vn = vmin[None, :] + u * (vmax[None, :] - vmin[None, :])
    return np.where(logflag[None, :], 10.0 ** Vn, Vn)


# ============================================================================
# 1a. tSZ leg data + fixed covariance (T3's FULL budget, replicated)
# ============================================================================

def load_tsz_data():
    a = np.load(LC / "act_ycap_lrg_real.npz", allow_pickle=True)
    xb = a["xb"]
    iv = np.nonzero(a["valid_cols"])[0]
    d_xb = a["mean_xb_cib17"][iv]                      # resample-corrected
    d_cov_full = a["cov_jk_xb_cib17"]
    corr = a["resample_corr"]                           # 1 + b_resample
    b_resample = corr - 1.0

    f_t2 = np.load(LC / "T2f_lrg_z0406_cib1.7.npz", allow_pickle=True)
    n_jk = int(f_t2["n_jk_cells"])
    dof = len(iv)
    h_data = (n_jk - 1) / max(n_jk - dof - 2, 1)

    r5 = np.load(LC / "R5_cib_cov.npz")
    cov_cib_c = (r5["cov_cib"] / np.outer(corr, corr))[np.ix_(iv, iv)]
    sig_res_only = 0.5 * np.abs(b_resample) * np.abs(a["mean_xb_cib17"])
    cov_fixed = (h_data * d_cov_full[np.ix_(iv, iv)] + cov_cib_c
                 + np.diag(sig_res_only[iv] ** 2))

    r3 = np.load(LC / "R3_mass_template.npz")
    mass_frac_iv = r3["dmodel_frac_per_sigma"][iv]
    mass_ddata_iv = r3["ddata_per_sigma"][iv]

    log(f"tSZ data: dof={dof}, xb={xb[iv].round(3)}, n_jk={n_jk}, h_data={h_data:.3f}")
    return dict(xb_iv=xb[iv], iv=iv, d_xb=d_xb, cov_fixed=cov_fixed,
                mass_frac_iv=mass_frac_iv, mass_ddata_iv=mass_ddata_iv, dof=dof)


def load_tsz_model_targets(node_ids_canonical):
    r2 = np.load(LC / "R2_hod_model_curves.npz")
    assert np.array_equal(r2["node_ids"], node_ids_canonical)
    A = BIND_PIX_AREA
    f004 = np.nanmean(r2["nodes_kcal_f0.04"], axis=1) * A     # (253, 18)
    f008 = np.nanmean(r2["nodes_kcal_f0.08"], axis=1) * A
    f012 = np.nanmean(r2["nodes_kcal_f0.12"], axis=1) * A

    r7 = json.load(open(VERDICT_DIR / "R7.json"))
    r7cols = r7["metrics"]["fit_columns_xb_0.46_1.25"]
    xb_r7 = np.array([c["xb"] for c in r7cols])
    bias_r7 = np.array([c["frac_bias_pct"] for c in r7cols]) / 100.0
    fid_corr18 = 1.0 + np.interp(XB18, xb_r7, bias_r7)

    f004c, f008c, f012c = f004 / fid_corr18, f008 / fid_corr18, f012 / fid_corr18
    slope18 = (f012c - f004c) / (0.12 - 0.04)
    return f008c, slope18     # (253, 18) each; caller slices to iv


def load_r4_twohalo_template(tsz_data):
    """R4.json's analytic 2-halo CAP-y curve at the 6 T3 fit columns, ALREADY
    in the same y*arcmin^2 flux units as act_ycap_lrg_real.npz's mean_xb_cib17
    (verified byte-for-byte: R4's per-column "data_value" equals d_xb at the
    same xb -- no unit conversion needed). Returns T2h aligned to
    tsz_data["xb_iv"]'s own column order."""
    r4 = json.load(open(VERDICT_DIR / "R4.json"))
    cols = r4["metrics"]["fit_columns_xb_0.46_1.25"]
    xb_r4 = np.array([c["xb"] for c in cols])
    t2h_r4 = np.array([c["cap_2h_analytic"] for c in cols])
    xb_iv = tsz_data["xb_iv"]
    assert len(xb_r4) == len(xb_iv) and np.allclose(np.sort(xb_r4), np.sort(xb_iv), atol=1e-6), \
        f"R4 fit-column grid {xb_r4} must match T3's xb_iv {xb_iv}"
    T2h = np.array([t2h_r4[np.argmin(np.abs(xb_r4 - x))] for x in xb_iv])
    log(f"R4 two-halo template (A2h prior N({A2H_LOC},{A2H_SCALE}), truncated>={A2H_LO}): "
        f"T2h={T2h} (analytic prediction A2h=1.0; R4's own empirical cross-check "
        f"A2h={r4['metrics']['A2h_fit_cen_anchor']['A2h']:.3f}"
        f"+/-{r4['metrics']['A2h_fit_cen_anchor']['sigma']:.3f})")
    return T2h


# ============================================================================
# 1b. kSZ leg (bgs110) data + fixed covariance (R6's unit-fix, replicated)
# ============================================================================

def load_ksz_leg(node_ids_canonical):
    from lightcone_m2_ycap_liu import mean_theta200_arcmin
    from _r6_ksz_audit import fix_cov

    ZEN = KS / "desact_zenodo"
    dz = np.load(ZEN / "Fig8_BGS_BRIGHT-20.2_logm11.00.npz")
    th_data, ratio_data, yerr_data, cov_ksz_raw = dz["th"], dz["ratio"], dz["yerr"], dz["cov_ksz"]
    cov_fixed_full = fix_cov(cov_ksz_raw, yerr_data)

    thr = mean_theta200_arcmin("bgs110")
    m1 = th_data <= 1.4 * thr
    ndof = int(m1.sum())
    node_theta = XB18 * thr

    taucap = np.load(LC / "taucap_lightcone.npz", allow_pickle=True)
    capmat = np.load(LC / "capmat_lightcone.npz", allow_pickle=True)
    F_B = float(capmat["F_B"])
    assert np.array_equal(taucap["node_ids"], node_ids_canonical)
    assert np.array_equal(capmat["node_ids"], node_ids_canonical)

    mat_mean_node = capmat["sb35_mean_bgs110"][:, :18].astype(np.float64)
    tau_mean_node = taucap["sb35_mean_bgs110"][:, :18].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        fgas_mean_node = tau_mean_node / mat_mean_node / F_B    # (253, 18)

    target = np.array([np.interp(th_data[m1], node_theta, fgas_mean_node[k])
                        for k in range(len(node_ids_canonical))])   # (253, ndof)

    log(f"kSZ (bgs110) data: ndof={ndof}, theta200_fid={thr:.3f}' , th_data(fit)={th_data[m1].round(2)}")
    return dict(th_fit=th_data[m1], d_ksz=ratio_data[m1],
                cov_fixed=cov_fixed_full[np.ix_(m1, m1)], target=target, ndof=ndof,
                thr=thr)


# ============================================================================
# 1c. gas-plane targets (verbatim lightcone_latent_corner.py::gas_fracs)
# ============================================================================

def load_gasplane_targets(node_ids_canonical):
    from lightcone_latent_corner import gas_fracs
    f_in, f_out = gas_fracs(SNAP_BGS, LOGM_LO, LOGM_HI, node_ids_canonical)
    return f_in, f_out


# ============================================================================
# 2. shared-hyperparameter group GPs
# ============================================================================

def fit_group(X, Y, seed=GP_SEED, n_restarts=0, independent=False):
    """Fit ONE ARD Matern(nu=2.5)+White kernel on Y's standardized mean
    direction, then reuse those hyperparameters (optimizer=None) to solve
    each column's own GP independently (per-column mean/std still free).
    Returns (gps: list[GaussianProcessRegressor], zmean, zstd, kernel_).

    ``independent=True`` instead runs a FULL ARD optimization per column
    (no hyperparameter sharing) -- used for small (<=2 column) groups whose
    columns are not guaranteed to be as tightly correlated as neighbouring
    apertures of the same field (e.g. f_in vs f_out), where a shared
    "mean-direction" kernel risks averaging away a column-specific signal.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    ncol = Y.shape[1]
    zmean = Y.mean(0)
    zstd = Y.std(0) + 1e-12
    Zs = (Y - zmean) / zstd
    d = X.shape[1]

    def _fresh_kernel():
        return (ConstantKernel(1.0, (1e-2, 1e2))
                * Matern(length_scale=np.ones(d), length_scale_bounds=(3e-2, 1e2), nu=2.5)
                + WhiteKernel(1e-2, (1e-6, 1e0)))

    if independent:
        gps = []
        for j in range(ncol):
            gp = GaussianProcessRegressor(kernel=_fresh_kernel(), alpha=1e-8,
                                          n_restarts_optimizer=n_restarts, random_state=seed + j)
            gp.fit(X, Zs[:, j])
            gps.append(gp)
        return gps, zmean, zstd, None

    rep = Zs.mean(1)   # leading shared direction (columns are strongly correlated)
    gp0 = GaussianProcessRegressor(kernel=_fresh_kernel(), alpha=1e-8,
                                   n_restarts_optimizer=n_restarts, random_state=seed)
    gp0.fit(X, rep)
    fixed_kernel = gp0.kernel_

    gps = []
    for j in range(ncol):
        gp = GaussianProcessRegressor(kernel=fixed_kernel, alpha=1e-8, optimizer=None)
        gp.fit(X, Zs[:, j])
        gps.append(gp)
    return gps, zmean, zstd, fixed_kernel


def predict_group(gps, zmean, zstd, X, return_std=False):
    means = []
    stds = []
    for gp in gps:
        if return_std:
            m, s = gp.predict(X, return_std=True)
            stds.append(s * zstd[len(means)])
        else:
            m = gp.predict(X)
        means.append(m)
    mean = np.stack(means, 1) * zstd[None, :] + zmean[None, :]
    if return_std:
        return mean, np.stack(stds, 1)
    return mean


# ============================================================================
# 3. K-fold cross-validation (RMSE, coverage, calibration)
# ============================================================================

def kfold_validate(X, Y, colnames, n_splits=N_FOLD, seed=GP_SEED, independent=False):
    from sklearn.model_selection import KFold
    n, ncol = Y.shape
    good = np.all(np.isfinite(Y), axis=1) & np.all(np.isfinite(X), axis=1)
    Xg, Yg = X[good], Y[good]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred_mean = np.full_like(Yg, np.nan)
    pred_std = np.full_like(Yg, np.nan)
    for tr, te in kf.split(Xg):
        gps, zm, zs, _ = fit_group(Xg[tr], Yg[tr], seed=seed, independent=independent)
        m, s = predict_group(gps, zm, zs, Xg[te], return_std=True)
        pred_mean[te] = m
        pred_std[te] = s
    resid = Yg - pred_mean
    spread = np.std(Yg, axis=0)
    rmse = np.sqrt(np.mean(resid ** 2, axis=0))
    z = resid / np.clip(pred_std, 1e-30, None)
    cov1 = np.mean(np.abs(z) < 1, axis=0)
    cov2 = np.mean(np.abs(z) < 2, axis=0)
    calib = np.sqrt(np.mean(z ** 2, axis=0))     # temperature: multiply GP sigma by this
    # coverage AFTER applying the calibration factor (this is what the MCMC's
    # floor_var actually uses -- calib is defined so mean(z_cal**2)==1 exactly,
    # so cov1/cov2 here are the number that matters for the pass gate, not the
    # raw pre-calibration numbers above, which are reported for transparency)
    z_cal = z / calib[None, :]
    cov1_cal = np.mean(np.abs(z_cal) < 1, axis=0)
    cov2_cal = np.mean(np.abs(z_cal) < 2, axis=0)
    typical_std_oof = np.median(pred_std, axis=0)
    per_col = {}
    for j, name in enumerate(colnames):
        per_col[name] = dict(rmse=float(rmse[j]), node_to_node_spread=float(spread[j]),
                             rmse_over_spread=float(rmse[j] / max(spread[j], 1e-30)),
                             coverage_1sigma_raw=float(cov1[j]), coverage_2sigma_raw=float(cov2[j]),
                             coverage_1sigma=float(cov1_cal[j]), coverage_2sigma=float(cov2_cal[j]),
                             calibration_factor=float(calib[j]),
                             typical_gp_sigma_oof=float(typical_std_oof[j]))
    return per_col, calib, n - good.sum()


# ============================================================================
# 4. production GPs + MCMC likelihood closures
# ============================================================================

def build_tsz_likelihood(prior, node_ids, U253, tsz_data):
    T2h = load_r4_twohalo_template(tsz_data)
    f008c18, slope18 = load_tsz_model_targets(node_ids)
    iv = tsz_data["iv"]
    Y_mean, Y_slope = f008c18[:, iv], slope18[:, iv]
    colnames = [f"tsz_mean_xb{x:.2f}" for x in tsz_data["xb_iv"]]
    colnames_s = [f"tsz_slope_xb{x:.2f}" for x in tsz_data["xb_iv"]]

    log("tSZ: 5-fold CV (mean curve)...")
    cv_mean, calib_mean, ndrop1 = kfold_validate(U253, Y_mean, colnames)
    log("tSZ: 5-fold CV (f_sat slope)...")
    cv_slope, calib_slope, ndrop2 = kfold_validate(U253, Y_slope, colnames_s)

    log("tSZ: production GP fit...")
    good_mean = np.all(np.isfinite(Y_mean), axis=1)
    good_slope = np.all(np.isfinite(Y_slope), axis=1)
    gp_mean, zm_mean, zs_mean, _ = fit_group(U253[good_mean], Y_mean[good_mean], seed=GP_SEED)
    gp_slope, zm_slope, zs_slope, _ = fit_group(U253[good_slope], Y_slope[good_slope], seed=GP_SEED + 1)
    _, std_mean_train = predict_group(gp_mean, zm_mean, zs_mean, U253[good_mean], return_std=True)
    _, std_slope_train = predict_group(gp_slope, zm_slope, zs_slope, U253[good_slope], return_std=True)
    floor_var_mean = (calib_mean * np.median(std_mean_train, axis=0)) ** 2
    floor_var_slope = (calib_slope * np.median(std_slope_train, axis=0)) ** 2

    # The slope-GP's own emulation uncertainty propagates into the model via
    # slope*(fsat-0.08); folded in at a representative |fsat-0.08|=FSAT_SCALE
    # (the prior's 1sigma) so the total floor stays a CONSTANT (preserves the
    # invert-once covariance -- see docstring point 3) while still crediting
    # the slope emulator's own error, not just the mean curve's.
    floor_var_total = floor_var_mean + floor_var_slope * (FSAT_SCALE ** 2)
    cov = tsz_data["cov_fixed"] + np.diag(floor_var_total)
    cov_inv = np.linalg.inv(cov)
    d_xb = tsz_data["d_xb"]
    mass_frac = tsz_data["mass_frac_iv"]
    mass_ddata = tsz_data["mass_ddata_iv"]

    def make_loglike(include_a2h_term):
        def loglike(u, fsat, dlogM, a2h):
            mean0 = predict_group(gp_mean, zm_mean, zs_mean, u)
            slope = predict_group(gp_slope, zm_slope, zs_slope, u)
            model = mean0 + slope * (fsat - F_EFF0)[:, None]
            model = model * (1.0 + (dlogM / 0.1)[:, None] * mass_frac[None, :])
            if include_a2h_term:
                # R4: the DATA contain unpainted 2-halo/diffuse gas that BIND's
                # >=1e13-only painting lacks -- added to the MODEL side, with
                # its own analytic shape (T2h) and a free positive amplitude.
                model = model + a2h[:, None] * T2h[None, :]
            data_shift = d_xb[None, :] + (dlogM / 0.1)[:, None] * mass_ddata[None, :]
            resid = data_shift - model
            return -0.5 * np.einsum("wi,ij,wj->w", resid, cov_inv, resid)
        return loglike

    loglike_with_a2h = make_loglike(True)
    loglike_without_a2h = make_loglike(False)

    validation = dict(mean_curve=cv_mean, fsat_slope=cv_slope,
                      floor_var_mean=[float(x) for x in floor_var_mean],
                      floor_var_slope=[float(x) for x in floor_var_slope],
                      floor_var_total_used_in_mcmc=[float(x) for x in floor_var_total],
                      n_dropped_nonfinite=[int(ndrop1), int(ndrop2)],
                      r4_two_halo_template_y_arcmin2=[float(x) for x in T2h])
    return loglike_with_a2h, loglike_without_a2h, validation


def build_ksz_likelihood(node_ids, U253, ksz_data):
    Y = ksz_data["target"]
    colnames = [f"ksz_bgs110_th{t:.2f}" for t in ksz_data["th_fit"]]
    good = np.all(np.isfinite(Y), axis=1)
    log(f"kSZ: 5-fold CV (bgs110 fgas)... ({good.sum()}/{len(Y)} finite nodes)")
    cv, calib, ndrop = kfold_validate(U253, Y, colnames)

    log("kSZ: production GP fit...")
    Xg, Yg = U253[good], Y[good]
    gp, zm, zs, _ = fit_group(Xg, Yg, seed=GP_SEED + 2)
    _, std_train = predict_group(gp, zm, zs, Xg, return_std=True)
    floor_var = (calib * np.median(std_train, axis=0)) ** 2

    cov = ksz_data["cov_fixed"] + np.diag(floor_var)
    cov_inv = np.linalg.inv(cov)
    d_ksz = ksz_data["d_ksz"]

    def loglike(u):
        model = predict_group(gp, zm, zs, u)
        resid = d_ksz[None, :] - model
        return -0.5 * np.einsum("wi,ij,wj->w", resid, cov_inv, resid)

    validation = dict(fgas_curve=cv, floor_var=[float(x) for x in floor_var],
                      n_dropped_nonfinite=int(ndrop))
    return loglike, validation


def build_gasplane_gp(U253, f_in, f_out):
    good = np.isfinite(f_in) & np.isfinite(f_out)
    Y = np.stack([f_in, f_out], axis=1)
    colnames = ["f_in", "f_out"]
    log(f"gas-plane: 5-fold CV (independent ARD per column: f_in, f_out are "
        f"physically distinct, not neighbouring apertures of one field)... "
        f"({good.sum()}/{len(Y)} finite nodes)")
    cv, calib, ndrop = kfold_validate(U253, Y, colnames, independent=True)
    gp, zm, zs, _ = fit_group(U253[good], Y[good], seed=GP_SEED + 3, independent=True)
    _, std_train = predict_group(gp, zm, zs, U253[good], return_std=True)
    floor_std = calib * np.median(std_train, axis=0)
    validation = dict(gas_frac=cv, floor_std=[float(x) for x in floor_std],
                      n_dropped_nonfinite=int(ndrop))
    return gp, zm, zs, floor_std, validation


def push_through_gasplane(u_samples, gp, zm, zs, floor_std, rng):
    mean, std = predict_group(gp, zm, zs, u_samples, return_std=True)
    tot_std = np.sqrt(std ** 2 + floor_std[None, :] ** 2)
    samp = mean + tot_std * rng.standard_normal(mean.shape)
    return samp[:, 0], samp[:, 1]


# ============================================================================
# 5. emcee MCMC
# ============================================================================

FSAT_LOC, FSAT_SCALE, FSAT_LO, FSAT_HI = 0.08, 0.04, 0.0, 0.2
DLOGM_SCALE, DLOGM_BOUND = 0.1, 0.4


NDIM = 33   # 30 astro (normalized [0,1]) + f_sat + dlogM + A_2h


def log_prior_batch(u, fsat, dlogM, a2h):
    from scipy.stats import truncnorm, norm
    ok = (np.all((u >= 0.0) & (u <= 1.0), axis=1)
          & (fsat >= FSAT_LO) & (fsat <= FSAT_HI)
          & (np.abs(dlogM) <= DLOGM_BOUND)
          & (a2h >= A2H_LO) & (a2h <= A2H_HI))
    lp = np.full(len(u), -np.inf)
    a, b = (FSAT_LO - FSAT_LOC) / FSAT_SCALE, (FSAT_HI - FSAT_LOC) / FSAT_SCALE
    a2, b2 = (A2H_LO - A2H_LOC) / A2H_SCALE, (A2H_HI - A2H_LOC) / A2H_SCALE
    lp[ok] = (truncnorm.logpdf(fsat[ok], a, b, loc=FSAT_LOC, scale=FSAT_SCALE)
              + norm.logpdf(dlogM[ok], 0.0, DLOGM_SCALE)
              + truncnorm.logpdf(a2h[ok], a2, b2, loc=A2H_LOC, scale=A2H_SCALE))
    return lp, ok


def make_log_prob(leg, tsz_loglike=None, ksz_loglike=None):
    """tsz_loglike(u, fsat, dlogM, a2h) -- pass loglike_without_a2h here (with
    a2h still sampled/priored, just not entering the likelihood) to build a
    "no 2-halo term" comparison chain of the SAME 33-dim shape."""
    def log_prob(coords):
        u, fsat, dlogM, a2h = coords[:, :30], coords[:, 30], coords[:, 31], coords[:, 32]
        lp, ok = log_prior_batch(u, fsat, dlogM, a2h)
        if not ok.any():
            return lp
        ll = np.zeros(ok.sum())
        if leg in ("tsz", "joint"):
            ll = ll + tsz_loglike(u[ok], fsat[ok], dlogM[ok], a2h[ok])
        if leg in ("ksz", "joint"):
            ll = ll + ksz_loglike(u[ok])
        lp[ok] += ll
        return lp
    return log_prob


def draw_prior(n, rng):
    from scipy.stats import truncnorm
    u = rng.uniform(0.0, 1.0, size=(n, 30))
    a, b = (FSAT_LO - FSAT_LOC) / FSAT_SCALE, (FSAT_HI - FSAT_LOC) / FSAT_SCALE
    fsat = truncnorm.rvs(a, b, loc=FSAT_LOC, scale=FSAT_SCALE, size=n, random_state=rng)
    dlogM = np.clip(rng.normal(0.0, DLOGM_SCALE, size=n), -DLOGM_BOUND, DLOGM_BOUND)
    a2, b2 = (A2H_LO - A2H_LOC) / A2H_SCALE, (A2H_HI - A2H_LOC) / A2H_SCALE
    a2h = truncnorm.rvs(a2, b2, loc=A2H_LOC, scale=A2H_SCALE, size=n, random_state=rng)
    return np.column_stack([u, fsat, dlogM, a2h])


def warm_start(log_prob_fn, nwalkers, rng, n_candidates=N_INIT_CANDIDATES):
    cand = draw_prior(n_candidates, rng)
    lp = log_prob_fn(cand)
    order = np.argsort(lp)[::-1][:nwalkers]
    p0 = cand[order].copy()
    jitter = rng.normal(scale=1e-3, size=p0.shape)
    p0 = p0 + jitter    # dlogM (col 31) is signed -- do NOT blanket-clip to >=0 here
    p0[:, :30] = np.clip(p0[:, :30], 1e-6, 1 - 1e-6)
    p0[:, 30] = np.clip(p0[:, 30], FSAT_LO + 1e-4, FSAT_HI - 1e-4)
    p0[:, 31] = np.clip(p0[:, 31], -DLOGM_BOUND + 1e-4, DLOGM_BOUND - 1e-4)
    p0[:, 32] = np.clip(p0[:, 32], A2H_LO + 1e-4, A2H_HI - 1e-4)
    return p0, float(lp[order].max())


def backend_path_for(leg):
    return SCRATCH / f"r8_chain_{leg}.h5"


def step_chain_budgeted(leg, log_prob_fn, rng, time_budget_sec, nwalkers=NWALKERS,
                        step_chunk=100):
    """Run (or RESUME) leg's chain against an on-disk emcee HDFBackend for up
    to time_budget_sec of WALL CLOCK, in step_chunk-sized increments (so a
    single call never overruns its external timeout by more than one chunk's
    duration) -- this is what makes the MCMC drivable as a series of bounded,
    independently-killable/restartable foreground calls instead of one long
    background run (see module docstring / WORKLOG for why). HDFBackend
    persists incrementally, so re-invoking this on the same leg picks up
    exactly where the last call left off; iteration=0 means a fresh start."""
    import emcee
    # Default StretchMove showed ~5% acceptance / tau~700 on this problem (a
    # smooth GP likelihood with a few dominant, curved directions embedded in
    # 33 mostly-flat dims -- exactly the regime where affine-invariant
    # stretch moves are known to mix poorly). Switch to the standard
    # DE(+snooker) mix, which tracks the ensemble's own covariance/curvature
    # instead of an isotropic stretch and is the widely-recommended fix for
    # this symptom.
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    backend = emcee.backends.HDFBackend(str(backend_path_for(leg)))
    ndim = NDIM
    fresh = not backend.initialized   # backend.iteration errors if the file doesn't exist yet
    n_before = 0 if fresh else backend.iteration
    t0 = time.time()

    if fresh:
        p0, lp_best = warm_start(log_prob_fn, nwalkers, rng)
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, vectorize=True,
                                        backend=backend, moves=moves)
        state = p0
        log(f"chain[{leg}]: FRESH start, warm-start best logL={lp_best:.2f}, "
            f"budget={time_budget_sec:.0f}s, chunk={step_chunk}...")
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, vectorize=True,
                                        backend=backend, moves=moves)
        state = None   # emcee resumes from the backend's last stored state
        log(f"chain[{leg}]: RESUMING from {n_before} existing steps, "
            f"budget={time_budget_sec:.0f}s, chunk={step_chunk}...")

    while time.time() - t0 < time_budget_sec:
        sampler.run_mcmc(state, step_chunk, progress=False)
        state = None

    n_after = backend.iteration
    log(f"chain[{leg}]: ran {n_after - n_before} steps this call ({time.time()-t0:.1f}s), "
        f"backend total={n_after}, mean accept frac={np.mean(sampler.acceptance_fraction):.3f}")
    return n_after


def process_chain_from_backend(leg):
    """Read the (possibly multi-call-accumulated) HDFBackend for leg and do
    the autocorrelation / discard-burn-in / thin / flatten post-processing --
    the read-only counterpart to step_chain_budgeted's writes.

    Each leg takes ~30 min (full ~20 GB ceph read + autocorr FFTs over ~1M
    steps x 33 dims), so the processed result is cached per leg; a finalize
    that dies partway resumes from the completed legs. The cache is keyed on
    the backend's step count and invalidates itself if the chain is extended."""
    import emcee
    backend = emcee.backends.HDFBackend(str(backend_path_for(leg)), read_only=True)
    ndim = NDIM
    nsteps = backend.iteration
    cache = SCRATCH / f"r8_processed_{leg}.npz"
    if cache.exists():
        c = np.load(cache)
        if int(c["total_steps"]) == nsteps:
            conv = json.loads(str(c["conv_json"]))
            log(f"chain[{leg}]: reusing processed cache ({cache.name}, "
                f"total_steps={nsteps}, kept {int(c['chain'].shape[0])} samples)")
            return c["chain"], c["logp"], conv
        log(f"chain[{leg}]: cache stale ({int(c['total_steps'])} != {nsteps}), reprocessing")
    accept_frac = np.mean(backend.accepted) / max(nsteps, 1)

    try:
        tau = backend.get_autocorr_time(tol=0)
    except Exception as e:
        tau = np.full(ndim, np.nan)
        log(f"chain[{leg}]: autocorr_time failed ({e})")
    tau_max = np.nanmax(tau) if np.isfinite(tau).any() else np.nan
    n_tau = nsteps / tau_max if np.isfinite(tau_max) and tau_max > 0 else np.nan
    discard = int(min(nsteps // 4, 5 * tau_max)) if np.isfinite(tau_max) else nsteps // 4
    thin = max(1, int(tau_max // 3)) if np.isfinite(tau_max) and tau_max > 0 else max(1, nsteps // 2000)

    chain = backend.get_chain(discard=discard, thin=thin, flat=True)
    logp = backend.get_log_prob(discard=discard, thin=thin, flat=True)
    conv = dict(tau=[float(x) if np.isfinite(x) else None for x in tau],
                tau_max=float(tau_max) if np.isfinite(tau_max) else None,
                nsteps_over_tau_max=float(n_tau) if np.isfinite(n_tau) else None,
                converged_50tau=bool(np.isfinite(n_tau) and n_tau >= 50),
                discard=discard, thin=thin, n_samples_kept=int(chain.shape[0]),
                mean_accept_frac=float(accept_frac), total_steps=int(nsteps))
    log(f"chain[{leg}]: total_steps={nsteps}, tau_max={tau_max}, nsteps/tau_max={n_tau}, "
        f"kept {chain.shape[0]} samples (discard={discard}, thin={thin})")
    np.savez(cache, chain=chain, logp=logp, total_steps=nsteps,
             conv_json=json.dumps(conv))
    return chain, logp, conv


# ============================================================================
# 6. look-elsewhere null
# ============================================================================

def dirichlet_alpha_for_ess(n, target_ess):
    # ESS = (n*alpha + 1)/(1 + alpha)  =>  alpha = (target_ess - 1)/(n - target_ess)
    return (target_ess - 1.0) / (n - target_ess)


def wquantile(v, w, qs):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(qs, cw, v[o])


def look_elsewhere_null(U253, prior_q, joint_u_samples, names, rng,
                        n_draws=N_NULL_DRAWS, target_ess=NULL_TARGET_ESS):
    n = U253.shape[0]
    alpha = dirichlet_alpha_for_ess(n, target_ess)
    prior_width = prior_q[:, 2] - prior_q[:, 0]
    null_wr = np.empty((n_draws, U253.shape[1]))
    ess_draws = np.empty(n_draws)
    for d in range(n_draws):
        w = rng.dirichlet(alpha * np.ones(n))
        ess_draws[d] = 1.0 / np.sum(w ** 2)
        for j in range(U253.shape[1]):
            q16, q84 = wquantile(U253[:, j], w, [0.16, 0.84])
            null_wr[d, j] = (q84 - q16) / prior_width[j]

    wr_joint = ((np.percentile(joint_u_samples, 84, axis=0)
                - np.percentile(joint_u_samples, 16, axis=0)) / prior_width)
    null_5th = np.percentile(null_wr, 5, axis=0)
    beats_null = wr_joint < null_5th

    per_param = {names[j]: dict(joint_width_ratio=float(wr_joint[j]),
                                null_5th_pct=float(null_5th[j]),
                                null_median=float(np.median(null_wr[:, j])),
                                beats_null=bool(beats_null[j]))
                for j in range(len(names))}
    result = dict(dirichlet_alpha=float(alpha), mean_ess_achieved=float(ess_draws.mean()),
                 std_ess_achieved=float(ess_draws.std()), n_draws=n_draws,
                 per_param=per_param,
                 beating_params=[names[j] for j in range(len(names)) if beats_null[j]])
    return result


# ============================================================================
# 7. prior-measure sensitivity (flat-linear reweight for LogFlag params)
# ============================================================================

def prior_sensitivity(joint_u_samples, prior, top_names):
    logflag = prior["logflag"]
    vmin, vmax = prior["vmin"], prior["vmax"]
    j_log = np.nonzero(logflag)[0]
    # flat-in-u (== flat-in-log10 V) -> flat-in-linear-V importance weight:
    # w_i propto prod_j dV/du = prod_j V_j * ln(10) * (log10range_j); the
    # constant ln(10)*range cancels in normalization, so w_i propto prod_j V_j.
    logV = vmin[None, j_log] + joint_u_samples[:, j_log] * (vmax[None, j_log] - vmin[None, j_log])
    log_w = np.log(10.0) * logV.sum(1)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))

    names = prior["names"]
    shifts = {}
    for name in top_names:
        j = names.index(name)
        med_flat_log = float(np.median(joint_u_samples[:, j]))
        med_flat_lin = float(wquantile(joint_u_samples[:, j], w, [0.5])[0])
        shifts[name] = dict(median_flatlog_prior=med_flat_log,
                            median_flatlinear_prior=med_flat_lin,
                            shift=med_flat_lin - med_flat_log,
                            is_logflag=bool(logflag[j]))
    return dict(ess_reweighted=ess, n_logflag_dims=int(len(j_log)), per_param=shifts)


# ============================================================================
# CLI stages -- gp (fit+validate, ~minutes) / chain (checkpointed MCMC,
# bounded per call) / finalize (post-process, figures, verdict). Split so a
# slow 1-CPU shared box can drive this as a series of independently-bounded,
# resumable foreground calls rather than one long run. `main()` (no --stage,
# or --stage all) just runs the three in sequence for standalone/documented
# usage (`python examples/_r8_gp_mcmc.py`) when time/resources allow it.
# ============================================================================

def overall_coverage_ok(val_dict):
    cols = []
    for key in val_dict:
        if isinstance(val_dict[key], dict) and "coverage_1sigma" in next(iter(val_dict[key].values()), {}):
            cols.extend(val_dict[key].values())
    if not cols:
        return True
    cov1 = np.array([c["coverage_1sigma"] for c in cols])
    cov2 = np.array([c["coverage_2sigma"] for c in cols])
    # nominal Gaussian 68/95%; accept coverage within [0.5,0.9] and [0.85,0.99]
    return bool(np.all((cov1 > 0.45) & (cov1 < 0.92)) and np.all((cov2 > 0.80)))


def stage_gp():
    """GP fit + K-fold validation for both legs + gas-plane; cloudpickle the
    closures + everything stage_chain/stage_finalize need to INGREDIENTS_PATH.
    Typically ~3-6 minutes; safe to run as a single foreground call."""
    log("loading prior box + design...")
    prior = load_prior_box()

    log("loading tSZ leg...")
    tsz_data = load_tsz_data()
    r2 = np.load(LC / "R2_hod_model_curves.npz")
    node_ids = r2["node_ids"]
    U253 = prior["U_all"][node_ids]

    log("loading kSZ leg...")
    ksz_data = load_ksz_leg(node_ids)

    log("loading gas-plane targets...")
    f_in, f_out = load_gasplane_targets(node_ids)

    tsz_loglike, tsz_loglike_noA2h, tsz_val = build_tsz_likelihood(prior, node_ids, U253, tsz_data)
    ksz_loglike, ksz_val = build_ksz_likelihood(node_ids, U253, ksz_data)
    gp_gas, zm_gas, zs_gas, floor_gas, gas_val = build_gasplane_gp(U253, f_in, f_out)
    gp_coverage_ok = (overall_coverage_ok(tsz_val) and overall_coverage_ok(ksz_val))
    log(f"GP coverage gate: {'PASS' if gp_coverage_ok else 'FAIL'}")

    log_prob_ksz = make_log_prob("ksz", ksz_loglike=ksz_loglike)
    log_prob_tsz = make_log_prob("tsz", tsz_loglike=tsz_loglike)
    log_prob_joint = make_log_prob("joint", tsz_loglike=tsz_loglike, ksz_loglike=ksz_loglike)
    log_prob_joint_noA2h = make_log_prob("joint", tsz_loglike=tsz_loglike_noA2h, ksz_loglike=ksz_loglike)

    payload = dict(
        prior=prior, node_ids=node_ids, U253=U253, f_in=f_in, f_out=f_out,
        tsz_val=tsz_val, ksz_val=ksz_val, gas_val=gas_val, gp_coverage_ok=gp_coverage_ok,
        gp_gas=gp_gas, zm_gas=zm_gas, zs_gas=zs_gas, floor_gas=floor_gas,
        log_prob_ksz=log_prob_ksz, log_prob_tsz=log_prob_tsz,
        log_prob_joint=log_prob_joint, log_prob_joint_noA2h=log_prob_joint_noA2h,
    )
    with open(INGREDIENTS_PATH, "wb") as f:
        cloudpickle.dump(payload, f)
    log(f"wrote {INGREDIENTS_PATH} ({INGREDIENTS_PATH.stat().st_size/1e6:.1f} MB)")
    return payload


LEG_SEEDS = {"ksz": 101, "tsz": 102, "joint": 103, "joint_noA2h": 104}
COL_WEIGHT = {"ksz": 6.0, "tsz": 12.0, "joint": 18.0, "joint_noA2h": 18.0}


def stage_chain(leg, budget_sec, step_chunk=100):
    """Run/resume ONE leg's chain for up to budget_sec of wall clock (see
    step_chain_budgeted). Loads the cloudpickled ingredients from stage_gp;
    safe to call repeatedly (resumes from the on-disk HDFBackend)."""
    with open(INGREDIENTS_PATH, "rb") as f:
        payload = cloudpickle.load(f)
    log_prob_fn = payload[f"log_prob_{leg}"]
    rng = np.random.default_rng(LEG_SEEDS[leg])
    step_chain_budgeted(leg, log_prob_fn, rng, budget_sec, step_chunk=step_chunk)
    # cheap progress read-back (also serves as a live sanity check between calls)
    chain, _, conv = process_chain_from_backend(leg)
    return conv


def stage_finalize():
    with open(INGREDIENTS_PATH, "rb") as f:
        payload = cloudpickle.load(f)
    prior = payload["prior"]
    names = list(prior["names"])
    logflag = prior["logflag"]
    prior_q = prior["prior_q"]
    node_ids = payload["node_ids"]
    U253 = payload["U253"]
    tsz_val, ksz_val, gas_val = payload["tsz_val"], payload["ksz_val"], payload["gas_val"]
    gp_coverage_ok = payload["gp_coverage_ok"]
    gp_gas, zm_gas, zs_gas, floor_gas = (payload["gp_gas"], payload["zm_gas"],
                                         payload["zs_gas"], payload["floor_gas"])

    chains, logps, conv = {}, {}, {}
    for leg in ("ksz", "tsz", "joint", "joint_noA2h"):
        chains[leg], logps[leg], conv[leg] = process_chain_from_backend(leg)

    primary_legs = ("ksz", "tsz", "joint")
    all_converged = all(conv[leg]["converged_50tau"] for leg in primary_legs)

    # ---- A_2h posterior + with/without comparison -----------------------------
    a2h_posterior = {leg: [float(x) for x in np.percentile(chains[leg][:, 32], [16, 50, 84])]
                     for leg in ("tsz", "joint")}
    med_wr_noA2h = ((np.percentile(chains["joint_noA2h"][:, :30], 84, axis=0)
                     - np.percentile(chains["joint_noA2h"][:, :30], 16, axis=0))
                    / (prior_q[:, 2] - prior_q[:, 0]))
    top8_idx_tmp = np.argsort(((np.percentile(chains["joint"][:, :30], 84, axis=0)
                               - np.percentile(chains["joint"][:, :30], 16, axis=0))
                              / (prior_q[:, 2] - prior_q[:, 0])))[:8]
    from scipy.stats import truncnorm as _truncnorm
    a2h_comparison = {
        "a2h_posterior_165084": a2h_posterior,
        "a2h_prior_165084": [float(x) for x in
                            _truncnorm.ppf([0.16, 0.5, 0.84],
                                          (A2H_LO - A2H_LOC) / A2H_SCALE,
                                          (A2H_HI - A2H_LOC) / A2H_SCALE,
                                          loc=A2H_LOC, scale=A2H_SCALE)],
        "joint_vs_joint_noA2h_top8_median_shift": {
            names[j]: dict(
                joint_with_a2h_median=float(np.median(chains["joint"][:, j])),
                joint_no_a2h_median=float(np.median(chains["joint_noA2h"][:, j])),
                shift=float(np.median(chains["joint"][:, j]) - np.median(chains["joint_noA2h"][:, j])),
                joint_with_a2h_width_ratio=float(((np.percentile(chains["joint"][:, j], 84)
                                                  - np.percentile(chains["joint"][:, j], 16))
                                                 / (prior_q[j, 2] - prior_q[j, 0]))),
                joint_no_a2h_width_ratio=float(med_wr_noA2h[j]),
            ) for j in top8_idx_tmp},
        "joint_noA2h_convergence": conv["joint_noA2h"],
        "note": "joint_noA2h is a SHORTER comparison-only chain (same column-weighted "
                "adaptive time budget as joint itself, but with the +A_2h*T2h template "
                "term simply omitted from the tSZ likelihood -- A_2h is still sampled/"
                "priored so the chain stays 33-dim, it just carries no likelihood "
                "information, i.e. this IS the pre-addendum R8 model). Not one of the "
                "3 primary deliverable chains; excluded from the convergence pass gate.",
    }
    log(f"A_2h posterior: tsz={a2h_posterior['tsz']}, joint={a2h_posterior['joint']} "
        f"(prior 16/50/84={a2h_comparison['a2h_prior_165084']})")

    # ---- gas-plane posterior -------------------------------------------------
    gas_stats = {}
    gas_samples = {}
    gasplane_seeds = {"ksz": 201, "tsz": 202, "joint": 203}
    for leg in ("ksz", "tsz", "joint"):
        rng = np.random.default_rng(gasplane_seeds[leg])
        u_samp = chains[leg][:, :30]
        fin_s, fout_s = push_through_gasplane(u_samp, gp_gas, zm_gas, zs_gas, floor_gas, rng)
        gas_samples[leg] = (fin_s, fout_s)
        gas_stats[leg] = dict(
            f_in_165084=[float(x) for x in np.percentile(fin_s, [16, 50, 84])],
            f_out_165084=[float(x) for x in np.percentile(fout_s, [16, 50, 84])],
            n_samples=int(len(fin_s)))

    L_verdict = json.load(open(VERDICT_DIR / "L.json"))
    L_joint = L_verdict["metrics"]["latent_constraints"]["joint"]

    # ---- look-elsewhere null --------------------------------------------------
    rng = np.random.default_rng(7)
    null_result = look_elsewhere_null(U253, prior_q, chains["joint"][:, :30], names, rng)
    log(f"look-elsewhere: {len(null_result['beating_params'])}/30 params beat the null 5th pct: "
        f"{null_result['beating_params']}")

    # ---- prior-measure sensitivity ---------------------------------------------
    joint_u = chains["joint"][:, :30]
    med_wr = ((np.percentile(joint_u, 84, axis=0) - np.percentile(joint_u, 16, axis=0))
              / (prior_q[:, 2] - prior_q[:, 0]))
    top8_idx = np.argsort(med_wr)[:8]
    top8_names = [names[j] for j in top8_idx]
    sens = prior_sensitivity(joint_u, prior, top8_names)
    log(f"prior sensitivity (flat-linear reweight, ESS={sens['ess_reweighted']:.1f}): "
        + ", ".join(f"{n}: {sens['per_param'][n]['shift']:+.3f}" for n in top8_names))

    # ---- figures ---------------------------------------------------------------
    make_corner_figure(chains, gas_samples, gas_stats)
    make_forest_figure(names, logflag, joint_u, chains, prior_q, null_result, med_wr)

    # ---- npz deliverable ---------------------------------------------------------
    npz_kw = dict(node_ids=node_ids, names=np.array(names), logflag=logflag,
                 prior_q=prior_q, U253=U253,
                 chain_columns="[0:30]=astro (prior-normalized), [30]=f_sat, [31]=dlogM, [32]=A_2h")
    for leg in ("ksz", "tsz", "joint"):
        npz_kw[f"chain_{leg}"] = chains[leg].astype(np.float32)
        npz_kw[f"logp_{leg}"] = logps[leg].astype(np.float32)
        npz_kw[f"gas_fin_{leg}"] = gas_samples[leg][0].astype(np.float32)
        npz_kw[f"gas_fout_{leg}"] = gas_samples[leg][1].astype(np.float32)
    # secondary comparison chain (no +A_2h*T2h term; see a2h_comparison in the verdict)
    npz_kw["chain_joint_noA2h"] = chains["joint_noA2h"].astype(np.float32)
    npz_kw["logp_joint_noA2h"] = logps["joint_noA2h"].astype(np.float32)
    np.savez(LC / "r8_posterior.npz", **npz_kw)
    log(f"wrote {LC/'r8_posterior.npz'}")

    # ---- verdict -----------------------------------------------------------------
    pass_gate = bool(gp_coverage_ok and all_converged)
    verdict = {
        "phase": "R8",
        "pass": pass_gate,
        "metrics": {
            "gp_validation": {"tsz": tsz_val, "ksz": ksz_val, "gasplane": gas_val,
                              "coverage_gate_ok": gp_coverage_ok},
            "mcmc_convergence": conv,
            "all_chains_converged_50tau": all_converged,
            "gas_plane": gas_stats,
            "gas_plane_vs_L": {
                "L_joint_f_in_165084": L_joint["f_in_165084"],
                "L_joint_f_out_165084": L_joint["f_out_165084"],
                "R8_joint_f_in_165084": gas_stats["joint"]["f_in_165084"],
                "R8_joint_f_out_165084": gas_stats["joint"]["f_out_165084"],
            },
            "look_elsewhere_null": null_result,
            "prior_sensitivity": sens,
            "top8_joint_width_ratio": [(names[j], float(med_wr[j])) for j in top8_idx],
            "n_traced_nodes": int(len(node_ids)),
            "L_robustness_stable_set_reference": ["RadioFeedbackReiorientationFactor",
                                                   "ThermalWindFraction"],
            "r4_two_halo_a2h_nuisance": a2h_comparison,
        },
        "figs": ["figs/R8_posterior_corner.png", "figs/R8_param_forest.png"],
        "notes": (
            "Replaces Phase L's ESS~6 importance-sampling pseudo-posterior with a "
            "continuous GP+MCMC posterior (see module docstring for the 5 documented "
            "design simplifications: f=0.08 + linear f_sat-slope emulation, "
            "shared-hyperparameter group GPs, CV-calibrated constant GP-variance floor "
            "replacing the node-specific BIND realization covariance, T3/R6 fixed data "
            "covariances, f_sat/dlogM/A_2h as tSZ-only nuisances). Mid-flight addendum: "
            "R4's analytic 2-halo/unpainted-gas CAP-y template (16-30% of the data value, "
            "exceeds the 10% action gate) is folded in as A_2h ~ N(1.0,0.3) truncated>=0 "
            "(33rd dim); see metrics.r4_two_halo_a2h_nuisance for the A_2h posterior and "
            "the joint-vs-joint_noA2h top-param comparison (joint_noA2h is a shorter, "
            "secondary comparison chain, not one of the 3 primary deliverables). "
            f"GP coverage {'PASSED' if gp_coverage_ok else 'FAILED'} the calibration gate; "
            f"the 3 primary chains {'ALL' if all_converged else 'DID NOT ALL'} reach >=50 "
            "tau samples (see mcmc_convergence per leg). "
            f"{len(null_result['beating_params'])}/30 params beat the look-elsewhere null "
            f"5th percentile: {null_result['beating_params']}."
        ),
        "next": "paper integration (replace L's corner/backtrack figures with R8's)",
    }
    with open(VERDICT_DIR / "R8.json", "w") as f:
        json.dump(verdict, f, indent=2, default=float)
    log(f"wrote {VERDICT_DIR/'R8.json'}")
    log(f"PASS={pass_gate}")
    log(f"total wall time: {time.time()-t_start:.1f}s")


def make_corner_figure(chains, gas_samples, gas_stats):
    from scipy.stats import gaussian_kde
    COL = {"ksz": "tab:blue", "tsz": "tab:red", "joint": "k"}
    LAB = {"ksz": "kSZ-only (R8 MCMC)", "tsz": "tSZ-only (R8 MCMC)", "joint": "joint (R8 MCMC)"}

    lat = np.load(LC / "latent_constraints.npz")
    fin_L, fout_L = lat["f_in"], lat["f_out"]
    Wl = {"ksz": lat["w_ksz"], "tsz": lat["w_tsz"], "joint": lat["w_joint"]}

    def kde_levels(dens, dx, dy, fracs=(0.68, 0.95)):
        s = np.sort(dens.ravel())[::-1]
        c = np.cumsum(s) * dx * dy
        return [s[min(np.searchsorted(c, f), len(s) - 1)] for f in fracs]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    all_fin = np.concatenate([gas_samples[t][0] for t in ("ksz", "tsz", "joint")])
    all_fout = np.concatenate([gas_samples[t][1] for t in ("ksz", "tsz", "joint")])
    xlim = np.percentile(all_fin, [1, 99])
    ylim = np.percentile(all_fout, [1, 99])
    g1 = np.linspace(*xlim, 140)
    g2 = np.linspace(*ylim, 140)
    XX, YY = np.meshgrid(g1, g2)

    for tag in ("ksz", "tsz", "joint"):
        fin_s, fout_s = gas_samples[tag]
        m = np.isfinite(fin_s) & np.isfinite(fout_s)
        k = gaussian_kde(np.vstack([fin_s[m], fout_s[m]]), bw_method=0.25)
        D = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        l68, l95 = kde_levels(D, g1[1] - g1[0], g2[1] - g2[0])
        ax.contour(XX, YY, D, levels=[l95, l68], colors=COL[tag],
                  linewidths=(1.0, 2.2 if tag == "joint" else 1.6),
                  linestyles=("--", "-"))
        ax.plot([], [], color=COL[tag], lw=2, label=LAB[tag])

        # overlay L's importance-sampling contours (same fields, old ESS~6 weights)
        mL = np.isfinite(fin_L) & np.isfinite(fout_L)
        kL = gaussian_kde(np.vstack([fin_L[mL], fout_L[mL]]), weights=Wl[tag][mL], bw_method=0.5)
        DL = kL(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        l68L, l95L = kde_levels(DL, g1[1] - g1[0], g2[1] - g2[0])
        ax.contour(XX, YY, DL, levels=[l68L], colors=COL[tag], linewidths=1.0,
                  linestyles=(":",), alpha=0.7)

    ax.plot([], [], color="0.3", lw=1.0, ls=":", label="Phase L (ESS~6 importance sampling, 68%)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\tilde f_{\rm gas}(<R_{500})/F_B$  ($f_{\rm in}$)")
    ax.set_ylabel(r"$\tilde f_{\rm gas}(R_{500}\to R_{200})/F_B$  ($f_{\rm out}$)")
    ax.set_title("R8: gas-plane posterior -- GP+MCMC (solid/dashed) vs\n"
                "Phase L ESS~6 importance sampling (dotted) -- 68/95% contours", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "R8_posterior_corner.png", dpi=140)
    plt.close(fig)
    log(f"wrote {FIG_DIR/'R8_posterior_corner.png'}")


def make_forest_figure(names, logflag, joint_u, chains, prior_q, null_result, med_wr):
    COL = {"ksz": "tab:blue", "tsz": "tab:red", "joint": "k"}
    LAB = {"ksz": "kSZ-only", "tsz": "tSZ-only", "joint": "joint"}
    order = np.argsort(med_wr)
    beats = null_result["per_param"]

    fig, ax = plt.subplots(figsize=(9.5, 12.5))
    for row, j in enumerate(order):
        ax.axhspan(row - 0.42, row + 0.42, color="0.95" if row % 2 else "white", zorder=0)
        for off, tag in [(-0.22, "ksz"), (0.0, "tsz"), (0.22, "joint")]:
            u = chains[tag][:, j]
            q16, q50, q84 = np.percentile(u, [16, 50, 84])
            ax.plot([q16, q84], [row + off] * 2, color=COL[tag], lw=2.2, solid_capstyle="butt")
            ax.plot(q50, row + off, "o", color=COL[tag], ms=4)
        ax.plot([prior_q[j, 0], prior_q[j, 2]], [row - 0.42, row - 0.42], color="0.6", lw=3.5,
               solid_capstyle="butt", alpha=0.6, zorder=0.5)
    ax.axvline(0.5, color="0.7", lw=0.8, zorder=0)

    yticklabels = []
    for j in order:
        name = names[j]
        tag = "*" if beats[name]["beats_null"] else " "
        yticklabels.append(f"{tag} {name}{' (log)' if logflag[j] else ''}   [wr {med_wr[j]:.2f}]")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(yticklabels, fontsize=7.5)
    for row, j in enumerate(order):
        if beats[names[j]]["beats_null"]:
            ax.get_yticklabels()[row].set_color("tab:green")
            ax.get_yticklabels()[row].set_fontweight("bold")
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("prior-normalized parameter value (16-50-84%); grey bar = prior 16-84%")
    handles = [plt.Line2D([], [], color=COL[t], lw=2.2, label=LAB[t]) for t in ("ksz", "tsz", "joint")]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_title("R8: 30-param GP+MCMC posteriors vs prior (sorted by joint width ratio)\n"
                "* green/bold = joint beats the 200-draw ESS~6 look-elsewhere null (5th pct)",
                fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "R8_param_forest.png", dpi=140)
    plt.close(fig)
    log(f"wrote {FIG_DIR/'R8_param_forest.png'}")


def main():
    """Standalone convenience entry point: runs gp -> chain(x4, full adaptive
    budget) -> finalize in one process. For a shared/contended 1-CPU box,
    prefer driving the stages as separate bounded CLI calls instead (see
    module docstring Usage)."""
    stage_gp()
    budget = {leg: TOTAL_MCMC_BUDGET_SEC * w / sum(COL_WEIGHT.values())
             for leg, w in COL_WEIGHT.items()}
    for leg in ("ksz", "tsz", "joint", "joint_noA2h"):
        stage_chain(leg, budget[leg])
    stage_finalize()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["all", "gp", "chain", "process", "finalize"],
                   default="all",
                   help="all (default, one process) | gp (fit+validate GPs) | "
                        "chain (run/resume ONE leg's MCMC for --budget seconds) | "
                        "process (autocorr/thin ONE leg into its cache -- lets the "
                        "4 legs' ~30-min post-processing run as parallel steps) | "
                        "finalize (post-process all 4 chains, figures, verdict)")
    p.add_argument("--leg", choices=list(LEG_SEEDS), default=None,
                   help="required with --stage chain / --stage process")
    p.add_argument("--budget", type=float, default=300.0,
                   help="wall-clock seconds for --stage chain (default 300)")
    args = p.parse_args()

    if args.stage == "all":
        main()
    elif args.stage == "gp":
        stage_gp()
    elif args.stage == "chain":
        if args.leg is None:
            raise SystemExit("--stage chain requires --leg {ksz,tsz,joint,joint_noA2h}")
        stage_chain(args.leg, args.budget)
    elif args.stage == "process":
        if args.leg is None:
            raise SystemExit("--stage process requires --leg {ksz,tsz,joint,joint_noA2h}")
        process_chain_from_backend(args.leg)
    elif args.stage == "finalize":
        stage_finalize()
