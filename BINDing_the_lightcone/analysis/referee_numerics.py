"""Referee-response numerics for Paper II (BINDing the lightcone).

Answers referee points II-1, II-3, II-5, II-6, II-9, II-10, II-11 and the
theta-baseline half of II-2.  The nested cross-validation of II-4 (and the
matched theta SEARCH of II-2) live in referee_nestedcv.py because they take
minutes rather than seconds.

Everything here is read-only against the campaign trees.  Numbers land in
analysis/referee_numbers.json and are echoed to stdout with provenance.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python referee_numerics.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

P1 = Path("/mnt/home/mlee1/BIND/papers/01_pipeline")
CEPH = Path("/mnt/home/mlee1/ceph")
OUT = Path(__file__).resolve().parent
os.chdir(P1)
sys.path.insert(0, str(P1))

from family_model import FamilyModel  # noqa: E402

STATS = ["clk", "pdf", "pk", "mn", "v0", "v1", "v2"]
ZI = 1                      # z_s = 1.0 plane
A2SR = (np.pi / 180 / 60) ** 2
AREA_BOX_SR = 25.0 * (np.pi / 180) ** 2
SURVEYS = {"LSST-Y10": (27.0, 0.26, 0.44), "Euclid-like": (30.0, 0.30, 0.36)}
DLNL = 0.15                 # the band-power width the paper now states explicitly
R: dict = {}


def sec(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


# ─────────────────────────────────────────────────────────────────────────
# shared inputs
# ─────────────────────────────────────────────────────────────────────────
dsn = np.load(CEPH / "bind_sb35/emulator_dataset_nu05.npz", allow_pickle=True)
ELL = np.asarray(dsn["a__suppression__ell"], float)
S_ALL = dsn["t__suppression__value"][:, ZI, :]                 # (256, 724)
THETA = np.asarray(dsn["X_unit"], float)                       # (256, 30) in prior units
PNAMES = [str(x) for x in dsn["param_names"]]
AMPS = np.load(P1 / "figs_preview/amplitude_sets.npz", allow_pickle=True)
SEARCH = np.load(P1 / "figs_preview/agnostic_lambda_results_obs.npz", allow_pickle=True)
NAMES = [str(x) for x in SEARCH["names"]]
X_ALL, X_FID = SEARCH["X_ALL"], SEARCH["X_FID"]
CHOSEN = [NAMES.index(n) for n in [str(x) for x in SEARCH["chosen_names"]]]
fm = FamilyModel()
coef = np.load(P1 / "latent_model_coeffs.npz")
EDGb = coef["ell_edges"]

cl_fid = np.load(CEPH / "bind_science/runs/twobound/run_0049/Cl_kappa.npz")["cl"][ZI, ZI]
cl_dmo = np.load(CEPH / "bind_science/runs/dmo/run_0000/Cl_kappa.npz")["cl"][ZI, ZI]
S_FID = cl_fid / cl_dmo
kk_real = np.load(CEPH / "bind_n1000/analysis/stream_stats_truth_run_0000.npz")["cl"][:, ZI, :]


def sobol_Y(st):
    dk = {"clk": "suppression", "pdf": "pdf", "pk": "peak_counts", "mn": "minima_counts",
          "v0": "mf_v0", "v1": "mf_v1", "v2": "mf_v2"}[st]
    Y = np.asarray(dsn[f"t__{dk}__value"], float)
    Y = Y[:, ZI, :] if Y.ndim == 3 else Y
    if st == "clk":
        Y = np.stack([np.nanmean(Y[:, (ELL >= EDGb[i]) & (ELL < EDGb[i + 1])], 1)
                      for i in range(len(EDGb) - 1)], axis=1)
    return Y[:, :fm.basis(st).shape[1]]


Y_ = {st: sobol_Y(st) for st in STATS}
A_ = {st: AMPS[f"{st}__a_sobol"] for st in STATS}
B_ = {st: AMPS[f"{st}__basis"] for st in STATS}
MEAN_ = {st: AMPS[f"{st}__mean"] for st in STATS}
D_ = {st: Y_[st] - MEAN_[st] for st in STATS}
SST_ = {st: (D_[st] ** 2).sum() for st in STATS}
NROW = len(X_ALL)
MU, SD = X_ALL.mean(0), X_ALL.std(0)
XZ = (X_ALL - MU) / SD
PERM = np.random.default_rng(1).permutation(NROW)
REPORT_FOLDS = [(np.setdiff1d(PERM, te), te) for te in np.array_split(PERM, 5)]


def e2e(X, folds, *, refit_mean=False):
    """per-statistic end-to-end CV R^2 for design matrix X under `folds`."""
    r2, pred = {}, {st: np.empty_like(A_[st]) for st in STATS}
    for tr, te in folds:
        Xtr = np.c_[X[tr] - X[tr].mean(0), np.ones(len(tr))]
        Xte = np.c_[X[te] - X[tr].mean(0), np.ones(len(te))]
        for st in STATS:
            W, *_ = np.linalg.lstsq(Xtr, A_[st][tr], rcond=None)
            pred[st][te] = Xte @ W
    for st in STATS:
        if refit_mean:                       # fold-internal reference statistic (II-3)
            num = den = 0.0
            for tr, te in folds:
                mu = Y_[st][tr].mean(0)
                d_te = Y_[st][te] - mu
                # amplitudes re-projected against the training-fold reference
                a_te = np.linalg.lstsq(B_[st].T, d_te.T, rcond=None)[0].T
                # prediction uses the map fit on the training fold only
                Xtr = np.c_[X[tr] - X[tr].mean(0), np.ones(len(tr))]
                Xte = np.c_[X[te] - X[tr].mean(0), np.ones(len(te))]
                a_tr = np.linalg.lstsq(B_[st].T, (Y_[st][tr] - mu).T, rcond=None)[0].T
                W, *_ = np.linalg.lstsq(Xtr, a_tr, rcond=None)
                num += ((d_te - (Xte @ W) @ B_[st]) ** 2).sum()
                den += (d_te ** 2).sum()
            r2[st] = 1 - num / den
        else:
            r2[st] = 1 - ((D_[st] - pred[st] @ B_[st]) ** 2).sum() / SST_[st]
    return r2


def pooled(r2):
    return float(np.mean([r2[st] for st in STATS]))


# ─────────────────────────────────────────────────────────────────────────
sec("BASELINE — must reproduce audits/agnostic_lambda_search_obs_mf22.log")
base = e2e(XZ[:, CHOSEN], REPORT_FOLDS)
print("  " + "  ".join(f"{st}:{base[st]:.4f}" for st in STATS) + f"   pooled:{pooled(base):.4f}")
REF = {"clk": 0.9632, "pdf": 0.9386, "pk": 0.7872, "mn": 0.8379,
       "v0": 0.9597, "v1": 0.9458, "v2": 0.9638}
dev = max(abs(base[s] - REF[s]) for s in STATS)
print(f"  max |delta| vs the shipped log = {dev:.2e}  {'OK' if dev < 2e-3 else 'MISMATCH'}")
R["baseline_cv"] = {s: round(base[s], 4) for s in STATS}
R["baseline_pooled"] = round(pooled(base), 4)

# ─────────────────────────────────────────────────────────────────────────
sec("II-1 / II-13 — detectability under three binning conventions")


def knox(cl, ng, se, fs, dlnl, ell):
    Nl = se ** 2 * A2SR / ng
    return np.sqrt(2.0 / ((2 * ell + 1) * np.maximum(ell * dlnl, 1.0) * fs)) * (1 + Nl / cl)


def knox_cv(fs, dlnl, ell):
    return np.sqrt(2.0 / ((2 * ell + 1) * np.maximum(ell * dlnl, 1.0) * fs))


edges = np.exp(np.arange(np.log(ELL[0]), np.log(52200) + DLNL, DLNL))
bidx = np.digitize(ELL, edges) - 1
NB = bidx.max() + 1


def rb(a):
    out = np.full(a.shape[:-1] + (NB,), np.nan)
    for b in range(NB):
        m = bidx == b
        if m.sum():
            out[..., b] = np.nanmean(a[..., m], axis=-1)
    return out


ELLb, S_b, Sfid_b, kkb, clfid_b = rb(ELL), rb(S_ALL), rb(S_FID), rb(kk_real), rb(cl_fid)
okb = np.isfinite(ELLb) & (ELLb <= 36864)
dlnl_native = float(np.median(np.diff(np.log(ELL))))
R["ell_grid"] = {"n_native_bins": len(ELL), "dlnl_native": round(dlnl_native, 5),
                 "n_bandpowers": int(okb.sum()), "dlnl_band": DLNL,
                 "width_ratio": round(DLNL / dlnl_native, 1)}
print(f"  native grid: {len(ELL)} bins, dlnl={dlnl_native:.5f}; "
      f"band-powers: {int(okb.sum())} bins, dlnl={DLNL} ({DLNL / dlnl_native:.0f}x wider)")

R["detect"] = {}
for nm, (ng, se, fs) in SURVEYS.items():
    out = {}
    # (a) as shipped: native measured leg + dlnl=0.15 shot leg
    meas_n = np.nanstd(np.log(np.where(kk_real > 0, kk_real, np.nan)), 0) * np.sqrt(AREA_BOX_SR / (fs * 4 * np.pi))
    for tag, dl_shot, ell, meas, clf, Sarr, Sf in (
            ("as_shipped", DLNL, ELL, meas_n, cl_fid, S_ALL, S_FID),
            ("native", dlnl_native, ELL, meas_n, cl_fid, S_ALL, S_FID),
            ("bandpower", DLNL, ELLb,
             np.nanstd(np.log(np.where(kkb > 0, kkb, np.nan)), 0) * np.sqrt(AREA_BOX_SR / (fs * 4 * np.pi)),
             clfid_b, S_b, Sfid_b)):
        shot = knox(clf, ng, se, fs, dl_shot, ell) - knox_cv(fs, dl_shot, ell)
        sig = np.sqrt(np.maximum(meas, knox_cv(fs, dl_shot, ell)) ** 2 + shot ** 2)
        lo, hi = np.nanpercentile(Sarr, [5, 95], axis=0)
        ratio = (hi - lo) / (2 * sig * Sf)
        m = np.isfinite(ratio) & (ell <= 36864)
        g = ell[m & (ratio > 10)]
        z = np.abs(Sarr - Sf) / (sig * Sf)
        mtr = m & (ell >= 300)
        pk = np.nanmax(z[:, mtr], axis=1)
        out[tag] = {"peak": round(float(np.nanmax(np.where(m, ratio, np.nan))), 1),
                    "peak_ell": int(ell[np.nanargmax(np.where(m, ratio, -1))]),
                    "window": [int(g.min()), int(g.max())] if g.size else None,
                    "frac_gt5sig": round(float((pk > 5).mean()), 3),
                    "frac_gt1sig": round(float((pk > 1).mean()), 3),
                    "median_peak": round(float(np.median(pk)), 1)}
        if tag == "bandpower":
            out["sigma_band_pct"] = {int(ell[i]): round(100 * float(sig[i]), 3)
                                     for i in [int(np.nanargmin(np.abs(ell - t)))
                                               for t in (300, 1000, 3000, 6000, 13000, 20000)]}
    out["area_deg2"] = round(fs * 4 * np.pi * (180 / np.pi) ** 2)
    out["N_ell_sr"] = float(f"{se ** 2 * A2SR / ng:.4g}")
    R["detect"][nm] = out
    print(f"  {nm:12s} area={out['area_deg2']} deg^2  N_ell={out['N_ell_sr']:.3g} sr")
    for tag in ("as_shipped", "native", "bandpower"):
        o = out[tag]
        print(f"     {tag:11s} peak {o['peak']:5.1f} @ ell={o['peak_ell']:6d}  "
              f">10x: {o['window']}  nodes>5sig {100 * o['frac_gt5sig']:.0f}%")
i6 = int(np.argmin(np.abs(ELL - 6000)))
R["Cl_kk_6000_zs1"] = float(f"{cl_fid[i6]:.4g}")
R["N_over_C_6000_LSST"] = round(float(SURVEYS["LSST-Y10"][1] ** 2 * A2SR / SURVEYS["LSST-Y10"][0] / cl_fid[i6]), 1)
print(f"  C_l^kk(ell=6000, z_s=1) = {cl_fid[i6]:.3g}  ->  N_l/C_l = {R['N_over_C_6000_LSST']}")

# ─────────────────────────────────────────────────────────────────────────
sec("II-9 — model predictive scatter vs survey precision")
xg = fm.grid("clk")
sp = fm.predictive_sigma("clk")
ng, se, fs = SURVEYS["LSST-Y10"]
shotb = knox(clfid_b, ng, se, fs, DLNL, ELLb) - knox_cv(fs, DLNL, ELLb)
measb = np.nanstd(np.log(np.where(kkb > 0, kkb, np.nan)), 0) * np.sqrt(AREA_BOX_SR / (fs * 4 * np.pi))
sigb = np.sqrt(np.maximum(measb, knox_cv(fs, DLNL, ELLb)) ** 2 + shotb ** 2)
sig_on_grid = np.interp(xg, ELLb[okb], (sigb * Sfid_b)[okb])
ratio_ms = sp / sig_on_grid
cross = xg[ratio_ms > 1]
R["sigma_pred"] = {"ell": [int(v) for v in xg], "sigma_pred": [round(float(v), 5) for v in sp],
                   "sigma_survey": [round(float(v), 5) for v in sig_on_grid]}
R["sigma_pred_exceeds_survey_above_ell"] = int(cross.min()) if cross.size else None
print(f"  sigma_pred exceeds the LSST-Y10 band-power sigma above ell = "
      f"{R['sigma_pred_exceeds_survey_above_ell']}")
for t in (1000, 1500, 5000, 13000, 20000):
    i = int(np.argmin(np.abs(xg - t)))
    print(f"    ell={xg[i]:6.0f}: sigma_pred={sp[i]:.4f}  survey={sig_on_grid[i]:.4f}  "
          f"ratio={ratio_ms[i]:.2f}")

# ─────────────────────────────────────────────────────────────────────────
sec("II-11 — Delta chi^2 replaces the per-bin maximum")
Cb = np.cov(np.log(kkb[:, okb]).T)                       # correlation structure, n1000 rotations
sd = np.sqrt(np.diag(Cb))
corr = Cb / np.outer(sd, sd)
sig_survey = (sigb * Sfid_b)[okb]                        # survey sigma on S, per band-power
Csurv = corr * np.outer(sig_survey, sig_survey)
p = Csurv.shape[0]
N = kkb.shape[0]
h = (N - p - 2) / (N - 1)
Cinv = h * np.linalg.inv(Csurv)
dS = (S_b - Sfid_b)[:, okb]
chi2 = np.einsum("ij,jk,ik->i", dS, Cinv, dS)
R["chi2"] = {"p": int(p), "N": int(N), "hartlap": round(float(h), 3),
             "median": round(float(np.median(chi2)), 1),
             "min": round(float(chi2.min()), 1), "max": round(float(chi2.max()), 1),
             "frac_gt_p": round(float((chi2 > p).mean()), 3),
             "frac_gt_25sig_equiv": round(float((chi2 > 25).mean()), 3)}
print(f"  p={p} band-powers, N={N}, Hartlap h={h:.3f}")
print(f"  Delta chi^2 across the 256 nodes: median {np.median(chi2):.0f}, "
      f"range {chi2.min():.1f}-{chi2.max():.0f}")
print(f"  fraction with Delta chi^2 > p (={p}): {100 * (chi2 > p).mean():.1f}%")

# ─────────────────────────────────────────────────────────────────────────
sec("II-5 — decoy diagnostics (from the saved search, zero new compute)")
step = SEARCH["step_scores"]                              # (n_forward_steps, n_candidates)
dec = np.array([n.startswith("DECOY") for n in NAMES])
path_a = [str(x) for x in SEARCH["path_action"]]
path_i = [int(x) for x in SEARCH["path_index"]]
path_s = [float(x) for x in SEARCH["path_score"]]
prev, rows = 0.0, []
k = 0
for a, i, s in zip(path_a, path_i, path_s):
    if a == "+":
        sc = step[k]
        best_dec = float(np.nanmax(sc[dec]))
        rank = int(np.nansum(sc > sc[i]) + 1)
        dec_rank = int(np.nansum(sc > best_dec) + 1)
        rows.append({"move": len(rows) + 1, "name": NAMES[i], "gain": round(s - prev, 4),
                     "best_decoy_gain": round(best_dec - prev, 4),
                     "decoy_rank": dec_rank, "accepted_rank": rank})
        k += 1
    prev = s
R["decoy_path"] = rows
print(f"  {'move':>4} {'candidate':26s} {'gain':>8} {'best decoy':>11} {'decoy rank':>11}")
for r in rows:
    print(f"  {r['move']:>4} {r['name']:26s} {r['gain']:>+8.4f} {r['best_decoy_gain']:>+11.4f} "
          f"{r['decoy_rank']:>11d}")
marg = SEARCH["marg"]
R["decoy_marginal_best"] = round(float(np.nanmax(marg[dec])), 4)
R["decoy_selected"] = int(sum(dec[i] for i in CHOSEN))
print(f"  best decoy as a marginal: {R['decoy_marginal_best']:.4f} "
      f"(winner {np.nanmax(marg[~dec]):.4f}); decoys selected: {R['decoy_selected']}")

# best-of-N null, rebuilt with 200 decoys along the accepted path
sec("II-5 — best-of-N acceptance null from 200 decoys")
rng = np.random.default_rng(20260814)
src = rng.choice(int((~dec).sum()), 200)
real_idx = np.where(~dec)[0]
DEC200 = np.stack([rng.permutation(X_ALL[:, real_idx[j]]) for j in src], axis=1)
DEC200Z = (DEC200 - DEC200.mean(0)) / DEC200.std(0)
acc_path = [i for a, i in zip(path_a, path_i) if a == "+"]
null_rows = []
cum = []
for step_k in range(len(acc_path)):
    base_idx = acc_path[:step_k]
    Xb = XZ[:, base_idx] if base_idx else np.empty((NROW, 0))
    s0 = pooled(e2e(Xb, REPORT_FOLDS)) if base_idx else 0.0
    gains = np.array([pooled(e2e(np.c_[Xb, DEC200Z[:, j]], REPORT_FOLDS)) - s0
                      for j in range(200)])
    null_rows.append({"k": step_k + 1, "null_mean": round(float(gains.mean()), 5),
                      "null_p95": round(float(np.percentile(gains, 95)), 5),
                      "null_max": round(float(gains.max()), 5)})
    print(f"  k={step_k + 1:2d}: 200-decoy null gain  mean {gains.mean():+.5f}  "
          f"p95 {np.percentile(gains, 95):+.5f}  max {gains.max():+.5f}")
R["decoy_null200"] = null_rows
R["eps_recommended"] = round(float(max(r["null_p95"] for r in null_rows)), 4)
print(f"  => best-of-N (p95) acceptance threshold ~ {R['eps_recommended']:.4f} "
      f"(paper used 0.0020)")

# ─────────────────────────────────────────────────────────────────────────
sec("II-3 — moving the leaky ingredients inside the fold")
r_int = e2e(XZ[:, CHOSEN], REPORT_FOLDS, refit_mean=True)
print("  reference statistic + amplitudes refit per training fold:")
print("  " + "  ".join(f"{st}:{r_int[st]:.4f}" for st in STATS) + f"   pooled:{pooled(r_int):.4f}")
# standardization / lambda-bar recomputed per fold (raw columns, fold-internal scaling)
r_std = {}
pred = {st: np.empty_like(A_[st]) for st in STATS}
for tr, te in REPORT_FOLDS:
    mu, sd_ = X_ALL[tr].mean(0), X_ALL[tr].std(0)
    Xtr = np.c_[(X_ALL[tr][:, CHOSEN] - mu[CHOSEN]) / sd_[CHOSEN], np.ones(len(tr))]
    Xte = np.c_[(X_ALL[te][:, CHOSEN] - mu[CHOSEN]) / sd_[CHOSEN], np.ones(len(te))]
    for st in STATS:
        W, *_ = np.linalg.lstsq(Xtr, A_[st][tr], rcond=None)
        pred[st][te] = Xte @ W
for st in STATS:
    r_std[st] = 1 - ((D_[st] - pred[st] @ B_[st]) ** 2).sum() / SST_[st]
print("  standardization + lambda_bar fold-internal:")
print("  " + "  ".join(f"{st}:{r_std[st]:.4f}" for st in STATS) + f"   pooled:{pooled(r_std):.4f}")
R["cv_fold_internal_mean"] = {s: round(r_int[s], 4) for s in STATS}
R["cv_fold_internal_std"] = {s: round(r_std[s], 4) for s in STATS}
R["cv_shift_mean"] = round(pooled(r_int) - pooled(base), 4)
R["cv_shift_std"] = round(pooled(r_std) - pooled(base), 4)
print(f"  pooled shift: reference {R['cv_shift_mean']:+.4f}, standardization {R['cv_shift_std']:+.4f}")

# ─────────────────────────────────────────────────────────────────────────
sec("II-2 — the theta baseline, all seven statistics")
TZ = (THETA - THETA.mean(0)) / THETA.std(0)
r_theta = e2e(TZ, REPORT_FOLDS)
print("  linear map from all 30 parameters:")
print("  " + "  ".join(f"{st}:{r_theta[st]:.4f}" for st in STATS) + f"   pooled:{pooled(r_theta):.4f}")
R["theta_linear_cv"] = {s: round(r_theta[s], 4) for s in STATS}
R["theta_linear_pooled"] = round(pooled(r_theta), 4)
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    r_gbt, pg = {}, {st: np.empty_like(A_[st]) for st in STATS}
    for tr, te in REPORT_FOLDS:
        for st in STATS:
            for c in range(A_[st].shape[1]):
                g = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                                  random_state=0)
                g.fit(THETA[tr], A_[st][tr][:, c])
                pg[st][te, c] = g.predict(THETA[te])
    for st in STATS:
        r_gbt[st] = 1 - ((D_[st] - pg[st] @ B_[st]) ** 2).sum() / SST_[st]
    print("  gradient-boosted trees on the 30 parameters:")
    print("  " + "  ".join(f"{st}:{r_gbt[st]:.4f}" for st in STATS) + f"   pooled:{pooled(r_gbt):.4f}")
    R["theta_gbt_cv"] = {s: round(r_gbt[s], 4) for s in STATS}
    R["theta_gbt_pooled"] = round(pooled(r_gbt), 4)
except ImportError:
    print("  sklearn unavailable -- GBT baseline skipped")

# ─────────────────────────────────────────────────────────────────────────
sec("II-6 — independent paint replicas at the fiducial (twobound 18/49/53)")
cz = np.load(CEPH / "bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
OB_OM = 0.0486 / 0.3089
BINS = [(13.0, 13.2), (13.2, 13.4), (13.4, 13.6), (13.6, 13.8),
        (13.8, 14.0), (14.0, 14.3), (14.3, 15.0)]
BLAB = [f"{lo:.1f}-{hi:.1f}".replace("-15.0", "+") for lo, hi in BINS]


def latents_from(F):
    """the eight shipped latents, measured with family_model.py's convention."""
    mt5, mg5bg, ms5 = F["m_tot_500c_bg"], F["m_gas_500c_bg"], F["m_star_500c"]
    mg2bg = F["m_gas_200c_bg"]
    T5, Pe5, Y5 = F["T_mw_500c"], F["Pe_mw_500c"], F["Y_500c"]
    lm = np.log10(np.where(mt5 > 0, mt5, np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = {"f_star": ms5 / mt5 / OB_OM, "c_gas": mg5bg / mg2bg}
        L = {"logT": T5, "logPe": Pe5, "logY": Y5,
             "logY_ss": Y5 / np.where(mt5 > 0, mt5, np.nan) ** (5 / 3)}
    out = {}
    for (lo, hi), bl in zip(BINS, BLAB):
        s = (lm >= lo) & (lm < hi)
        for q, arr in Q.items():
            v = arr[s]
            out[f"{q}[{bl}]"] = np.nanmedian(v) if np.isfinite(v).sum() >= 5 else np.nan
        for q, arr in L.items():
            v = np.where(arr[s] > 0, arr[s], np.nan)
            m = np.nanmedian(v) if np.isfinite(v).sum() >= 5 else np.nan
            out[f"{q}[{bl}]"] = np.log10(m) if m > 0 else np.nan
    return np.array([out[n] for n in fm.lat_names])


TBK = ["m_tot_500c_bg", "m_gas_500c_bg", "m_star_500c", "m_gas_200c_bg",
       "T_mw_500c", "Pe_mw_500c", "Y_500c"]
REP = [18, 49, 53]
lam_rep = {r: latents_from({k: np.asarray(cz[f"tb_{k}"], float)[r] for k in TBK}) for r in REP}
S_rep = {}
for r in REP:
    S_rep[r] = np.load(CEPH / f"bind_science/runs/twobound/run_{r:04d}/Cl_kappa.npz")["cl"][ZI, ZI] / cl_dmo
xg = fm.grid("clk")
Sg = {r: np.array([np.nanmean(S_rep[r][(ELL >= EDGb[i]) & (ELL < EDGb[i + 1])])
                   for i in range(len(EDGb) - 1)])[:len(xg)] for r in REP}
lam_spread = np.std(np.array([lam_rep[r] for r in REP]), axis=0, ddof=1)
lam_design = X_ALL[:, CHOSEN].std(0)
print("  per-latent paint-replica scatter vs the across-design spread:")
for i, n in enumerate(fm.lat_names):
    print(f"    {n:22s} sigma_rep/sigma_design = {lam_spread[i] / lam_design[i]:.4f}")
R["replica_lambda_noise_frac"] = {n: round(float(lam_spread[i] / lam_design[i]), 4)
                                  for i, n in enumerate(fm.lat_names)}
own, cross = [], []
for a in REP:
    for b in REP:
        err = float(np.nanmedian(np.abs(fm.predict(lam_rep[a], "clk") - Sg[b]) / np.abs(Sg[b])))
        (own if a == b else cross).append(err)
R["replica_selfpred_median_pct"] = round(100 * float(np.mean(own)), 3)
R["replica_crosspred_median_pct"] = round(100 * float(np.mean(cross)), 3)
R["replica_S_scatter_pct"] = round(100 * float(np.mean(np.std([Sg[r] for r in REP], axis=0, ddof=1)
                                                       / np.mean([Sg[r] for r in REP], axis=0))), 3)
print(f"  predict S(l) of replica B from lambda of replica A:")
print(f"    same replica  (circular): median |dev| = {R['replica_selfpred_median_pct']:.3f}%")
print(f"    cross replica (clean)   : median |dev| = {R['replica_crosspred_median_pct']:.3f}%")
print(f"    replica-to-replica scatter in S(l) itself: {R['replica_S_scatter_pct']:.3f}%")

# ─────────────────────────────────────────────────────────────────────────
sec("II-10 — lightcone geometry")
try:
    from astropy.cosmology import FlatLambdaCDM
    cos = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    hh, fov = 0.6774, 5.0 * np.pi / 180
    geo = {}
    for z in (0.5, 1.0, 1.5, 2.0, 2.44):
        chi = cos.comoving_distance(z).value * hh
        geo[str(z)] = {"chi": round(float(chi), 1), "transverse": round(float(chi * fov), 1),
                       "over_box": round(float(chi * fov / 205.0), 2)}
        print(f"  z_s={z:4.2f}: chi={chi:7.1f}  5deg subtends {chi * fov:6.1f} Mpc/h  "
              f"({chi * fov / 205.0:.2f} x box)")
    zg = np.linspace(1e-3, 1 - 1e-3, 4000)
    chg = cos.comoving_distance(zg).value * hh
    chis = cos.comoving_distance(1.0).value * hh
    ipk = int(np.argmax(chg * (chis - chg) / chis))
    tp = chg[ipk] * fov
    geo["kernel_peak"] = {"z": round(float(zg[ipk]), 3), "chi": round(float(chg[ipk]), 1),
                          "transverse": round(float(tp), 1),
                          "patches_per_face": round(float((205.0 / tp) ** 2), 1)}
    print(f"  z_s=1 kernel peak: z={zg[ipk]:.2f}, 5deg subtends {tp:.0f} Mpc/h "
          f"-> {(205.0 / tp) ** 2:.1f} independent patches per box face")
    R["geometry"] = geo
except ImportError:
    print("  astropy unavailable -- geometry skipped")

# ─────────────────────────────────────────────────────────────────────────
(OUT / "referee_numbers.json").write_text(json.dumps(R, indent=1))
print(f"\nwrote {OUT / 'referee_numbers.json'}")
