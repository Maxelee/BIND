"""scatter/sensitivity.py

Shared, robust global-sensitivity toolkit for the BIND feedback->(f_gas, M_star)
analysis. Operates on the already-generated joint design (NO new sampling):

    outputs/scatter_diagnostics/chunks_joint_cv/joint_part_*.npz
        cube (128 design pts, n_chunk halos, 12 noise draws, 16 obs)
        sub  (128, 30)  Sobol design coords in [0,1] (shared across chunks)
    outputs/scatter_diagnostics/scatter_decomposition_joint_cv.npz : param_names (30,)

Robustness philosophy
---------------------
All halos see the SAME 128-point design, so averaging an observable over halos
(in a mass bin) and noise draws gives a clean response vector Y[design] (128,).
We summarise its dependence on the 30 astro params with THREE estimators that
make different assumptions, and report their agreement:
  * SRC  - standardized regression coefficients (signed, linear). Bootstrapped.
  * dCor - distance correlation (model-free, detects nonlinear dependence).
  * CV R^2 of linear vs GP vs gradient-boosting surrogates, to verify the
    response is predominantly smooth/near-linear (so SRC is accurate, not just
    convenient). On this data linear ~ GP > GBM, gap <= 0.04 in R^2.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHUNK_GLOB = str(ROOT / "outputs/scatter_diagnostics/chunks_joint_cv/joint_part_*.npz")
JOINT_NPZ = ROOT / "outputs/scatter_diagnostics/scatter_decomposition_joint_cv.npz"

# observable column indices in the 16-obs cube
I_MDM, I_MGAS, I_MSTAR, I_FB = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# data
def load_joint_cube():
    """Return Y (128, N_h, 16) [median over K], masses (N_h,), sub (128, 30), names (30,)."""
    fs = sorted(glob.glob(CHUNK_GLOB))
    if not fs:
        raise FileNotFoundError(f"no joint chunks at {CHUNK_GLOB}")
    Ys, Ms, sub = [], [], None
    for f in fs:
        d = np.load(f)
        Ys.append(np.nanmedian(d["cube"], axis=2))   # (128, n_chunk, 16)
        Ms.append(d["masses"])
        if sub is None:
            sub = d["sub"].astype(float)
    Y = np.concatenate(Ys, axis=1)
    masses = np.concatenate(Ms)
    names = [str(s) for s in np.load(JOINT_NPZ)["param_names"]]
    return Y, masses, sub, names


def fgas_cube(Y):
    """(n_design, n_halo) gas fraction M_gas/(M_dm+M_gas+M_star) per design,halo."""
    Mdm, Mg, Ms = Y[:, :, I_MDM], Y[:, :, I_MGAS], Y[:, :, I_MSTAR]
    with np.errstate(divide="ignore", invalid="ignore"):
        return Mg / (Mdm + Mg + Ms)


# ---------------------------------------------------------------------------
# estimators
def src(X, y):
    """Standardized regression coefficients (no intercept term returned) and R^2."""
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    A = np.c_[np.ones(len(ys)), Xs]
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    R2 = 1.0 - np.sum((ys - A @ beta) ** 2) / np.sum(ys ** 2)
    return beta[1:], R2


def src_bootstrap(X, y, n_boot=2000, seed=0):
    """SRC with bootstrap over design points -> beta, R2, lo(16%), hi(84%)."""
    rng = np.random.default_rng(seed)
    beta, R2 = src(X, y)
    n = len(y)
    boots = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b], _ = src(X[idx], y[idx])
    lo, hi = np.percentile(boots, [16, 84], axis=0)
    return beta, R2, lo, hi


def distance_correlation(x, y):
    """Szekely distance correlation in [0,1]; model-free dependence (any shape)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0)[None, :] - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvx = np.sqrt((A * A).mean()); dvy = np.sqrt((B * B).mean())
    denom = dvx * dvy
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom)) if denom > 0 else 0.0


def dcor_all(X, y):
    """Per-parameter distance correlation -> (d,)."""
    return np.array([distance_correlation(X[:, i], y) for i in range(X.shape[1])])


def cv_r2_compare(X, y, seed=0):
    """5-fold CV R^2 for linear / Gaussian-process / gradient-boosting surrogates.

    Used to justify SRC: if the linear (or smooth GP) model is the most accurate
    out-of-sample, the parameter response is near-linear and SRC is a faithful
    summary; tree-based GBM doing worse indicates the response is smooth, not
    piecewise/interaction-dominated.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import r2_score

    cv = KFold(5, shuffle=True, random_state=seed)
    d = X.shape[1]
    models = {
        "linear": LinearRegression(),
        "GP": GaussianProcessRegressor(
            kernel=C(1.0) * RBF([0.3] * d, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-6, 1e1)),
            normalize_y=True, n_restarts_optimizer=3, alpha=1e-8),
        "GBM": GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
            random_state=seed),
    }
    import warnings
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, m in models.items():
            out[k] = float(r2_score(y, cross_val_predict(m, X, y, cv=cv)))
    return out


# ---------------------------------------------------------------------------
# population targets and relation descriptors
def population_targets(Y, masses, mass_bin):
    """Per-design population-median observables in a mass bin: f_gas, logM*, f_b."""
    logm = np.log10(masses)
    sel = (logm >= mass_bin[0]) & (logm < mass_bin[1])
    fg = fgas_cube(Y)[:, sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        lMs = np.log10(Y[:, sel, I_MSTAR])
    return {
        "f_gas": np.nanmedian(fg, axis=1),
        "logM_star": np.nanmedian(lMs, axis=1),
        "f_b": np.nanmedian(Y[:, sel, I_FB], axis=1),
    }, int(sel.sum())


def relation_descriptors(Y, masses, mass_bin):
    """Per-design descriptors of the f_gas-M_star relation among halos in a bin.

    For each design we remove the halo-mass trend (linear in log M_halo) from
    both f_gas and log M_star, then report:
      med_fgas, med_logMstar : normalization (move ALONG the mean relation)
      scat_fgas, scat_logMstar : intrinsic scatter at fixed mass
      coupling : Pearson r of the mass-residuals (the intrinsic f_gas-M_star
                 anticorrelation; how tightly the two co-vary at fixed mass)
    Returns dict[str] -> (n_design,) and the halo count.
    """
    logm = np.log10(masses)
    sel = (logm >= mass_bin[0]) & (logm < mass_bin[1])
    x = logm[sel]
    xd = x - x.mean()
    fg = fgas_cube(Y)[:, sel]                         # (n_design, n_sel)
    with np.errstate(divide="ignore", invalid="ignore"):
        lMs = np.log10(Y[:, sel, I_MSTAR])

    def detrend(row):
        m = np.isfinite(row)
        if m.sum() < 10:
            return np.full_like(row, np.nan)
        b = np.polyfit(xd[m], row[m], 1)
        res = np.full_like(row, np.nan)
        res[m] = row[m] - np.polyval(b, xd[m])
        return res

    nD = Y.shape[0]
    med_fg = np.nanmedian(fg, axis=1)
    med_Ms = np.nanmedian(lMs, axis=1)
    scat_fg = np.full(nD, np.nan); scat_Ms = np.full(nD, np.nan)
    coupling = np.full(nD, np.nan)
    for d in range(nD):
        rg = detrend(fg[d]); rm = detrend(lMs[d])
        scat_fg[d] = np.nanstd(rg); scat_Ms[d] = np.nanstd(rm)
        ok = np.isfinite(rg) & np.isfinite(rm)
        if ok.sum() > 10:
            coupling[d] = np.corrcoef(rg[ok], rm[ok])[0, 1]
    return {
        "med_fgas": med_fg, "med_logMstar": med_Ms,
        "scat_fgas": scat_fg, "scat_logMstar": scat_Ms,
        "coupling": coupling,
    }, int(sel.sum())
