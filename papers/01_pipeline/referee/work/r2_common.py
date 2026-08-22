"""Shared loaders + conventions for the R2 "noisy twin" referee response.

Everything here is READ-ONLY against the campaign trees. Derived products land in
/mnt/home/mlee1/ceph/referee_work/r2/.

The noisy cache (papers/01_pipeline/noisy_grid.py, built 2026-08-05) carries, for
every target, statistics measured on maps that have had LSST-Y10 shape noise added
(Lee+22 Eq. 7) and then been Gaussian-smoothed at theta_G = 1' (Eq. 6, 1/e radius).
This module adds the three things the figures need on top of that cache:

  1. the MEASURED noise-bias power spectrum Nhat(ell) -- pure shape noise pushed
     through the identical smooth() + power_spectrum() code path, so the smoothing
     transfer function W^2(ell) is included exactly (discretisation and all) rather
     than assumed analytic;
  2. covariance machinery: 25 deg^2 sample covariance -> area-scaled LSST-Y10 /
     Euclid covariance, Hartlap-debiased precision matrix, chi^2;
  3. the noiseless counterparts (nu05_shards + field_cache) for the like-for-like
     table, with their convention differences flagged (they are NOT the same
     smoothing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

P1 = Path("/mnt/home/mlee1/BIND/papers/01_pipeline")
sys.path.insert(0, str(P1))

import noisy_grid as ng  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SB35 = CEPH / "bind_sb35"
SCI = CEPH / "bind_science"
OUT = CEPH / "referee_work/r2"
OUT.mkdir(parents=True, exist_ok=True)

NU = ng.NU
NU_KEYS = ("pdf", "peak_counts", "minima_counts", "mf_v0", "mf_v1", "mf_v2")
ZI = 1                       # z_s = 1.0 working plane, as in the paper
ZS = (0.5, 1.0, 1.5, 2.0, 2.44)

# ── survey areas / footprints (paper conventions) ─────────────────────────────
AREA_BOX = 25.0              # deg^2, the 5x5 lightcone footprint
AREA_LSST = 18000.0          # deg^2, main.tex Eq. (area_scaling)
FSKY_LSST = AREA_LSST / 41252.96
# Euclid-like: fig 4/12/23's convention is ngal=30, sigma_e=0.30, f_sky=0.36
FSKY_EUCLID = 0.36
AREA_EUCLID = FSKY_EUCLID * 41252.96          # 14851 deg^2
NGAL_LSST, SIGE_LSST = 27.0, 0.26
NGAL_EUCLID, SIGE_EUCLID = 30.0, 0.30

ARCMIN2_SR = (np.pi / 180.0 / 60.0) ** 2


def shot_noise_cl(ngal: float, sigma_e: float) -> float:
    """Flat shape-noise power N_ell = sigma_e^2 / (2 n_gal) in steradians.

    Same factor-of-2 convention as noisy_grid.NOISE_SIGMA_PIX (Lee+22 Eq. 7):
    sigma_pix^2 = sigma_e^2/(2 n_gal A_pix) => N_ell = sigma_pix^2 A_pix.
    """
    return sigma_e ** 2 / (2.0 * ngal) * ARCMIN2_SR


# ── 1. measured noise-bias spectrum (includes W^2 exactly) ────────────────────
NOISE_CL_CACHE = OUT / "noise_bias_cl.npz"


def measure_noise_bias_cl(n_draw: int = 40, seed: int = 202608) -> dict:
    """Nhat(ell) = <Pk( smooth(pure shape noise) )>, and the smoothing W^2(ell).

    Pushes pure LSST-Y10 shape noise through the SAME add-noise + smooth +
    power_spectrum path noisy_grid.compute() uses, so the returned bias spectrum
    already contains the numerical Gaussian-filter transfer function. `w2` is the
    same thing normalised by the flat unsmoothed shot-noise level, i.e. the
    measured W^2(ell); comparing it to exp(-ell^2 sigma_theta^2) validates the
    theta_G-is-the-1/e-radius reading of Lee+22 Eq. 6.
    """
    if NOISE_CL_CACHE.exists():
        d = np.load(NOISE_CL_CACHE)
        return {k: d[k] for k in d.files}
    from bind.inference.stats import power_spectrum
    rng = np.random.default_rng(seed)
    acc_s = acc_r = None
    ell = None
    for _ in range(n_draw):
        n = rng.normal(0.0, ng.NOISE_SIGMA_PIX, (ng.NPIX, ng.NPIX))
        ell, c_raw = power_spectrum(n, fov_deg=ng.FOV_DEG)
        _, c_sm = power_spectrum(ng.smooth(n), fov_deg=ng.FOV_DEG)
        acc_s = c_sm if acc_s is None else acc_s + c_sm
        acc_r = c_raw if acc_r is None else acc_r + c_raw
    nhat = acc_s / n_draw
    nraw = acc_r / n_draw
    w2 = nhat / nraw
    flat = float(np.median(nraw))
    sigma_theta = ng.SMOOTH_SIGMA_ARCMIN * np.pi / 180.0 / 60.0
    w2_analytic = np.exp(-(ell ** 2) * sigma_theta ** 2)
    out = dict(ell=ell, nhat=nhat, nraw=nraw, w2=w2, w2_analytic=w2_analytic,
               flat_level=np.array(flat), n_draw=np.array(n_draw),
               analytic_flat=np.array(shot_noise_cl(NGAL_LSST, SIGE_LSST)))
    np.savez(NOISE_CL_CACHE, **out)
    return out


# ── 2. covariance machinery ──────────────────────────────────────────────────

def area_scale(area_deg2: float) -> float:
    """C(A) = (A/25 deg^2)^-1 C(25 deg^2)  ->  this is the factor on C."""
    return AREA_BOX / area_deg2


def hartlap(n_real: int, d: int) -> float:
    return (n_real - d - 2.0) / (n_real - 1.0)


def chi2_of(resid: np.ndarray, cov25: np.ndarray, area_deg2: float,
            n_real: int) -> tuple[float, float, int]:
    """Hartlap-debiased chi^2 of `resid` against the area-scaled covariance.

    Returns (chi2, chi2/dof, dof). cov25 is the sample covariance measured on the
    25 deg^2 box; it is scaled to `area_deg2` by main.tex Eq. (area_scaling), and
    its inverse is debiased by (N-d-2)/(N-1) (Eq. inv_cov, Hartlap 2007).
    """
    m = np.isfinite(resid)
    r = resid[m]
    C = cov25[np.ix_(m, m)] * area_scale(area_deg2)
    d = r.size
    h = hartlap(n_real, d)
    if h <= 0:
        return np.nan, np.nan, d
    Cinv = np.linalg.pinv(C) * h
    x2 = float(r @ Cinv @ r)
    return x2, x2 / d, d


def chi2_to_sigma(chi2: float, dof: int) -> float:
    """Gaussian-equivalent two-sided significance of a chi^2 with `dof` dof."""
    from scipy import stats
    if not np.isfinite(chi2):
        return np.nan
    logsf = stats.chi2.logsf(chi2, dof)
    if logsf < -700:                        # far tail: asymptotic Wilson-Hilferty
        return float(np.sqrt(max(0.0, 2 * chi2) - np.sqrt(max(1.0, 2.0 * dof - 1))))
    p = np.exp(logsf)
    return float(stats.norm.isf(max(p, 1e-300) / 2.0))


def sigma_thresholds(dof: int) -> tuple[float, float]:
    """chi^2 values corresponding to 3-sigma and 5-sigma (two-sided Gaussian p)."""
    from scipy import stats
    p3 = 2 * stats.norm.sf(3.0)             # 2.700e-3
    p5 = 2 * stats.norm.sf(5.0)             # 5.733e-7
    return float(stats.chi2.isf(p3, dof)), float(stats.chi2.isf(p5, dof))


# ── 3. loaders ───────────────────────────────────────────────────────────────
SHARDS_N = ng.SHARD_DIR


def load_noisy_target(target: str) -> dict:
    d = np.load(SHARDS_N / f"{target}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_noisy_sobol() -> dict:
    d = np.load(ng.DATASET_OUT, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_noiseless_nu() -> tuple[dict, dict]:
    """nu05_shards sci_bind / sci_truth: 50 realizations, NOISELESS.

    CONVENTION WARNING: these are NOT the same measurement as the noisy shards.
    nu_grid.py uses mixed smoothing (PDF unsmoothed, peaks/minima at 2', MFs at
    1') and a PER-MAP nu normalisation; noisy_grid.py uses one 1' smoothing for
    all six and a FIXED per-plane kappa_rms. So a noisy-vs-noiseless chi^2
    contrast carries both effects; see the memo.
    """
    from nu_grid import SHARD_DIR as SD0
    b = np.load(SD0 / "sci_bind.npz")
    t = np.load(SD0 / "sci_truth.npz")
    return {k: b[k] for k in b.files}, {k: t[k] for k in t.files}


def load_noiseless_fields():
    from field_cache import load as _load
    return _load(n_real=50)


# ── log-ell rebinning (Delta ln ell = 0.15, the cl_relerr band convention) ────

def log_ell_bins(ell: np.ndarray, lo: float, hi: float, dlnl: float = 0.15):
    edges = np.exp(np.arange(np.log(lo), np.log(hi) + dlnl, dlnl))
    idx = np.digitize(ell, edges) - 1
    keep = [b for b in range(len(edges) - 1) if np.sum(idx == b) > 0]
    centers = np.array([ell[idx == b].mean() for b in keep])
    masks = [idx == b for b in keep]
    return centers, masks


def rebin_rows(A: np.ndarray, masks) -> np.ndarray:
    """Average the trailing axis of A within each mask."""
    return np.stack([A[..., m].mean(-1) for m in masks], axis=-1)
