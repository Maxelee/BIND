"""scatter/residual.py
Library for the scatter-residual cross-correlation analysis (BIND paper §3).

Implements:
  - select_residual_observables / OBS_8 / OBS_7
  - lowess_fit, running_mad — local-linear smoother / local MAD (no statsmodels dep)
  - fit_mean_and_scatter — LOWESS mean μ̂(log M) + running σ̂(log M)
  - standardise_residuals — Δ̂_{a,i} = (F_{a,i} − μ̂_a) / σ̂_a
  - residual_correlation_matrix — Spearman/Pearson with bootstrap SE
  - frobenius_null_distribution — split-half bootstrap null for ||C_T − C_G||_F
  - eigen_alignment — eigenvalues + leading-eigenvector angle
  - rho_in_mass_bin — Spearman ρ of ΔM_*, ΔM_gas in mass bins (Farahi & Evrard test)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


# ---------------------------------------------------------------------------
# Observable selection
#
# Brief §1 fixes the 8-observable set. Source data has 16 observables
# (see scatter.measure_scatter.ALL_OBS_NAMES). The mapping is:

# brief name -> (cache key, is_log)
OBS_MAP: dict[str, tuple[str, bool]] = {
    "log10_M_DM":         ("M_dm",        True),
    "log10_M_gas":        ("M_gas",       True),
    "log10_M_star":       ("M_star",      True),
    "log10_Sigma_gas_c":  ("Sigma_gas_c", True),
    "q_DM":               ("q_DM",        False),
    "q_gas":              ("q_gas",       False),
    "q_star":             ("q_star",      False),
    "log10_f_b":          ("f_b",         True),
}

OBS_8: list[str] = list(OBS_MAP.keys())
OBS_7: list[str] = [n for n in OBS_8 if n != "log10_f_b"]


def extract_obs8(
    raw_obs: np.ndarray,
    raw_obs_names: list[str],
) -> np.ndarray:
    """Project (..., N_raw=16) to (..., 8) applying log10 where needed.

    Negative / zero inputs to log10 -> NaN (these halos are excluded downstream).
    """
    out = np.full(raw_obs.shape[:-1] + (8,), np.nan, dtype=np.float64)
    for i, name in enumerate(OBS_8):
        raw_key, is_log = OBS_MAP[name]
        j = raw_obs_names.index(raw_key)
        x = raw_obs[..., j].astype(np.float64)
        if is_log:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[..., i] = np.where(x > 0, np.log10(x), np.nan)
        else:
            out[..., i] = x
    return out


# ---------------------------------------------------------------------------
# Local-linear smoother (LOWESS-style, single pass, tricube kernel)

def _tricube(u: np.ndarray) -> np.ndarray:
    w = np.zeros_like(u)
    m = np.abs(u) < 1.0
    w[m] = (1.0 - np.abs(u[m]) ** 3) ** 3
    return w


def _local_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    frac: float,
) -> np.ndarray:
    """Tricube-weighted local linear regression at each query point.

    Excludes NaN training rows automatically. NaN queries propagate to NaN
    predictions.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    finite = np.isfinite(x_train) & np.isfinite(y_train)
    xt = x_train[finite]
    yt = y_train[finite]
    n = xt.size
    if n < 5:
        return np.full_like(x_query, np.nan, dtype=np.float64)

    k = max(int(np.ceil(frac * n)), 5)
    order = np.argsort(xt)
    xs = xt[order]
    ys = yt[order]

    x_q = np.asarray(x_query, dtype=np.float64).ravel()
    out = np.full_like(x_q, np.nan, dtype=np.float64)

    for i, xq in enumerate(x_q):
        if not np.isfinite(xq):
            continue
        # k nearest neighbours by |x - xq|
        d = np.abs(xs - xq)
        if k >= n:
            idx = np.arange(n)
        else:
            idx = np.argpartition(d, k - 1)[:k]
        h = d[idx].max()
        if h <= 0:
            out[i] = ys[idx].mean()
            continue
        u = (xs[idx] - xq) / h
        w = _tricube(u)
        if w.sum() <= 0:
            continue
        X = np.column_stack([np.ones_like(idx, dtype=np.float64), xs[idx]])
        W = w
        # Solve weighted least squares via the normal equations
        XtW = X.T * W
        try:
            beta = np.linalg.solve(XtW @ X, XtW @ ys[idx])
        except np.linalg.LinAlgError:
            out[i] = np.average(ys[idx], weights=w)
            continue
        out[i] = beta[0] + beta[1] * xq

    return out.reshape(np.shape(x_query))


def lowess_fit(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.4,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable mu(x_query) implementing single-pass local-linear LOWESS."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def mu(x_q):
        return _local_linear(x, y, np.asarray(x_q, dtype=np.float64), frac)

    return mu


def running_mad(
    x: np.ndarray,
    abs_resid: np.ndarray,
    frac: float = 0.4,
) -> Callable[[np.ndarray], np.ndarray]:
    """Local-window median absolute deviation, scaled to a Gaussian σ (×1.4826).

    `abs_resid` should be |y - mu(x)|; the smoother returns the running σ(x).
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(abs_resid, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(a)
    xt = x[finite]
    at = a[finite]
    n = xt.size
    if n < 5:
        return lambda x_q: np.full_like(np.asarray(x_q, dtype=np.float64), np.nan)
    k = max(int(np.ceil(frac * n)), 5)
    order = np.argsort(xt)
    xs = xt[order]
    aa = at[order]

    def sigma(x_q):
        xq = np.asarray(x_q, dtype=np.float64).ravel()
        out = np.full_like(xq, np.nan, dtype=np.float64)
        for i, xqi in enumerate(xq):
            if not np.isfinite(xqi):
                continue
            d = np.abs(xs - xqi)
            if k >= n:
                idx = np.arange(n)
            else:
                idx = np.argpartition(d, k - 1)[:k]
            med = np.median(aa[idx])
            out[i] = 1.4826 * med
        return out.reshape(np.shape(x_q))

    return sigma


# ---------------------------------------------------------------------------
# Mean + scatter fit per observable

@dataclass
class MeanScatter:
    mu: Callable[[np.ndarray], np.ndarray]
    sigma: Callable[[np.ndarray], np.ndarray]
    # Diagnostic cached samples for plotting
    x_pool: np.ndarray
    y_pool: np.ndarray


def fit_mean_and_scatter(
    log_mass_truth: np.ndarray,
    f_truth: np.ndarray,
    log_mass_bind: np.ndarray,
    f_bind_mean: np.ndarray,
    frac: float = 0.4,
    fit_source: str = "combined",
) -> MeanScatter:
    """Fit μ̂(log M_h) and σ̂(log M_h) for one observable.

    log_mass_truth, f_truth   : (N_h,)        — one truth row per halo.
    log_mass_bind, f_bind_mean: (N_h,)        — BIND per-halo sample mean.
    fit_source = "combined" (default), "truth", or "bind".
    """
    if fit_source == "combined":
        x_pool = np.concatenate([log_mass_truth, log_mass_bind])
        y_pool = np.concatenate([f_truth, f_bind_mean])
    elif fit_source == "truth":
        x_pool = log_mass_truth
        y_pool = f_truth
    elif fit_source == "bind":
        x_pool = log_mass_bind
        y_pool = f_bind_mean
    else:
        raise ValueError(f"unknown fit_source {fit_source!r}")

    mu = lowess_fit(x_pool, y_pool, frac=frac)
    mu_pool = mu(x_pool)
    abs_r = np.abs(y_pool - mu_pool)
    sigma = running_mad(x_pool, abs_r, frac=frac)
    return MeanScatter(mu=mu, sigma=sigma, x_pool=x_pool, y_pool=y_pool)


# ---------------------------------------------------------------------------
# Standardised residuals

def standardise_residuals(
    log_mass: np.ndarray,
    f: np.ndarray,
    mu: Callable,
    sigma: Callable,
) -> np.ndarray:
    """Δ̂_i = (F_i − μ̂(log M_i)) / σ̂(log M_i)."""
    log_mass = np.asarray(log_mass, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    mu_v = mu(log_mass)
    sigma_v = sigma(log_mass)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (f - mu_v) / np.where(sigma_v > 0, sigma_v, np.nan)


# ---------------------------------------------------------------------------
# Correlation matrices with bootstrap SE

def _corr_matrix(R: np.ndarray, method: str) -> np.ndarray:
    """Correlation matrix of columns of R, ignoring rows with any NaN."""
    R = np.asarray(R, dtype=np.float64)
    mask = np.all(np.isfinite(R), axis=1)
    Rm = R[mask]
    if Rm.shape[0] < 3:
        n = R.shape[1]
        return np.full((n, n), np.nan)
    if method == "pearson":
        return np.corrcoef(Rm, rowvar=False)
    elif method == "spearman":
        ranks = np.apply_along_axis(rankdata, 0, Rm)
        return np.corrcoef(ranks, rowvar=False)
    else:
        raise ValueError(method)


def residual_correlation_matrix(
    residuals: np.ndarray,
    method: str = "spearman",
    n_boot: int = 2000,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation matrix of column-observables in `residuals` (N_h, n_obs).

    Returns (C, SE) — both (n_obs, n_obs). SE is the bootstrap-over-halos std.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    n_h, n_obs = residuals.shape
    C = _corr_matrix(residuals, method)
    rng = np.random.default_rng(rng_seed)
    boot = np.empty((n_boot, n_obs, n_obs), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_h, size=n_h)
        boot[b] = _corr_matrix(residuals[idx], method)
    SE = np.nanstd(boot, axis=0, ddof=1)
    return C, SE


def per_halo_pearson_diagonal(
    delta_truth: np.ndarray,
    delta_bind_mean: np.ndarray,
    n_boot: int = 2000,
    rng_seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-observable per-halo Pearson(Δ^T_a, Δ^G_a).

    Inputs are (N_h, n_obs). Returns (P_aa, P_aa_se) with bootstrap-over-halos SE.
    """
    delta_truth = np.asarray(delta_truth, dtype=np.float64)
    delta_bind_mean = np.asarray(delta_bind_mean, dtype=np.float64)
    n_h, n_obs = delta_truth.shape

    def _one(dt, dg):
        out = np.full(n_obs, np.nan)
        for a in range(n_obs):
            mask = np.isfinite(dt[:, a]) & np.isfinite(dg[:, a])
            if mask.sum() < 5:
                continue
            out[a] = pearsonr(dt[mask, a], dg[mask, a])[0]
        return out

    P = _one(delta_truth, delta_bind_mean)
    rng = np.random.default_rng(rng_seed)
    boot = np.empty((n_boot, n_obs), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_h, size=n_h)
        boot[b] = _one(delta_truth[idx], delta_bind_mean[idx])
    SE = np.nanstd(boot, axis=0, ddof=1)
    return P, SE


# ---------------------------------------------------------------------------
# Frobenius null distribution (split-half on truth)

def frobenius_norm(M: np.ndarray) -> float:
    return float(np.sqrt(np.nansum(M ** 2)))


def frobenius_null_distribution(
    delta_truth: np.ndarray,
    method: str = "spearman",
    n_boot: int = 2000,
    rng_seed: int = 2,
) -> np.ndarray:
    """Split-half bootstrap null for ||C^(1) − C^(2)||_F on truth.

    delta_truth: (N_h, n_obs).
    Returns an (n_boot,) array of Frobenius distances under H0 (same population).
    """
    delta_truth = np.asarray(delta_truth, dtype=np.float64)
    n_h = delta_truth.shape[0]
    rng = np.random.default_rng(rng_seed)
    out = np.empty(n_boot, dtype=np.float64)
    half = n_h // 2
    for b in range(n_boot):
        perm = rng.permutation(n_h)
        a = perm[:half]
        bset = perm[half:half * 2]
        C1 = _corr_matrix(delta_truth[a], method)
        C2 = _corr_matrix(delta_truth[bset], method)
        out[b] = frobenius_norm(C1 - C2)
    return out


# ---------------------------------------------------------------------------
# Eigen-alignment

def eigen_alignment(C_T: np.ndarray, C_G: np.ndarray) -> dict:
    """Eigenvalue ratios and angle between leading eigenvectors (degrees)."""
    eT, vT = np.linalg.eigh(C_T)
    eG, vG = np.linalg.eigh(C_G)
    # eigh returns ascending; reverse to descending
    eT = eT[::-1]; vT = vT[:, ::-1]
    eG = eG[::-1]; vG = vG[:, ::-1]
    cos_angle = float(np.clip(abs(vT[:, 0] @ vG[:, 0]), 0.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    return {
        "eig_T": eT,
        "eig_G": eG,
        "leading_eigenvector_angle_deg": angle_deg,
        "eig_ratio_top": float(eT[0] / eG[0]) if eG[0] > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Mass-dependence of Δ M_* vs Δ M_gas (Farahi & Evrard 2018)

def rho_in_mass_bin(
    log_mass: np.ndarray,
    delta_mstar: np.ndarray,
    delta_mgas: np.ndarray,
    bin_edges: np.ndarray,
    method: str = "spearman",
    n_boot: int = 2000,
    rng_seed: int = 3,
) -> list[dict]:
    """Per-mass-bin ρ(ΔM_*, ΔM_gas) with bootstrap-over-halos SE.

    bin_edges has length n_bins + 1.
    """
    log_mass = np.asarray(log_mass, dtype=np.float64)
    delta_mstar = np.asarray(delta_mstar, dtype=np.float64)
    delta_mgas = np.asarray(delta_mgas, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)

    results = []
    n_bins = len(bin_edges) - 1
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        m = (log_mass >= lo) & (log_mass < hi)
        x = delta_mstar[m]
        y = delta_mgas[m]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 5:
            results.append(dict(lo=lo, hi=hi, n=n, rho=np.nan, se=np.nan, mid=0.5 * (lo + hi)))
            continue
        if method == "spearman":
            rho = float(spearmanr(x, y)[0])
        else:
            rho = float(pearsonr(x, y)[0])
        # Bootstrap
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            xb, yb = x[idx], y[idx]
            if method == "spearman":
                boots[b] = float(spearmanr(xb, yb)[0])
            else:
                boots[b] = float(pearsonr(xb, yb)[0])
        results.append(dict(lo=float(lo), hi=float(hi), n=int(n),
                            rho=rho, se=float(np.nanstd(boots, ddof=1)),
                            mid=float(0.5 * (lo + hi))))
    return results


def rebalance_to_equal_counts(log_mass: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based bin edges for equal counts; spans the data range."""
    log_mass = np.asarray(log_mass, dtype=np.float64)
    log_mass = log_mass[np.isfinite(log_mass)]
    qs = np.linspace(0, 1, n_bins + 1)
    return np.quantile(log_mass, qs)
