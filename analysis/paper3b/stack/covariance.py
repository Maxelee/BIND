"""WP-B2 spatial-jackknife covariance over HEALPix patches.

Peaks are grouped by ``ang2pix(nside_jk, position)``; the covariance of the
mean is estimated leave-one-patch-out:
``cov = (K-1)/K · Σ_k (m_k − m̄)(m_k − m̄)ᵀ`` over the K occupied patches.
MEASUREMENT_SPEC fixes nside_jk = 8 as the default with {4, 16} as the
patch-size-stability set (acceptance: <~20% drift under area doubling).

Pure numpy given per-object value vectors + patch ids — data/mock agnostic.
"""

from __future__ import annotations

import numpy as np


def patch_ids(ra_deg: np.ndarray, dec_deg: np.ndarray, nside_jk: int) -> np.ndarray:
    """HEALPix RING patch id per object (the jackknife grouping)."""
    import healpy as hp

    return hp.ang2pix(nside_jk, np.radians(90.0 - np.asarray(dec_deg)),
                      np.radians(np.asarray(ra_deg)))


def jackknife_mean_cov(values: np.ndarray, ids: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, int]:
    """(mean, cov of the mean, K) for per-object ``values`` (n_obj, n_dim).

    Leave-one-patch-out over the occupied patches. n_dim may be 1; cov is
    always (n_dim, n_dim). Raises if fewer than 3 occupied patches.
    """
    v = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if v.shape[0] == 1 and np.ndim(values) == 1:
        v = v.T
    ids = np.asarray(ids)
    uniq = np.unique(ids)
    if len(uniq) < 3:
        raise ValueError(f"jackknife needs >=3 occupied patches, got {len(uniq)}")
    total, n = v.sum(axis=0), len(v)
    means = np.empty((len(uniq), v.shape[1]))
    for k, u in enumerate(uniq):
        inpatch = ids == u
        means[k] = (total - v[inpatch].sum(axis=0)) / (n - inpatch.sum())
    mbar = means.mean(axis=0)
    d = means - mbar
    cov = (len(uniq) - 1) / len(uniq) * (d.T @ d)
    return v.mean(axis=0), cov, len(uniq)


def eigenvalue_drift(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    """max |λ_b/λ_a − 1| over sorted eigenvalues — the stability metric."""
    ea = np.sort(np.linalg.eigvalsh(np.atleast_2d(cov_a)))
    eb = np.sort(np.linalg.eigvalsh(np.atleast_2d(cov_b)))
    ok = ea > 0
    return float(np.abs(eb[ok] / ea[ok] - 1.0).max())
