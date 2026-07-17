"""WP-B3 null/systematics battery — config variants of the frozen B2 chain.

Every test here re-uses `run_peak_stack` (or the archived B2 per-peak tables)
unchanged, per plan task 1 ("no forked code") and the pre-registered
`NULL_CRITERIA.md`. Pure numpy/healpy helpers; the MPI-aware orchestration
lives in `scripts/run_b3_battery.py`.
"""

from __future__ import annotations

import numpy as np

from .stacker import FIDUCIAL_RADIUS_INDEX, NU_STACK_EDGES

DETECTED_BINS = slice(1, 5)          # nu >= 1 bins; nu 0-1 is diagnostic-only
J = FIDUCIAL_RADIUS_INDEX


# ------------------------------------------------------------ 2. RA shifts

def shift_catalog_ra(ra_deg: np.ndarray, dec_deg: np.ndarray, delta_deg: float,
                     act_hp: np.ndarray, interior: float = 0.99
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Shift RA by delta, keep positions where the ACT apod map is interior.

    Only the ACT factor is re-cut (NULL_CRITERIA §2 — the DES footprint moves
    with the catalog by construction). Returns (ra', dec', keep mask,
    retention fraction).
    """
    import healpy as hp

    nside = hp.npix2nside(act_hp.size)
    ra2 = (np.asarray(ra_deg, dtype=float) + delta_deg) % 360.0
    pix = hp.ang2pix(nside, np.radians(90.0 - np.asarray(dec_deg)), np.radians(ra2))
    keep = act_hp[pix] >= interior
    return ra2[keep], np.asarray(dec_deg)[keep], keep, float(keep.mean())


# ---------------------------------------------- 5. mask-proximity distances

def distance_to_footprint_edge_deg(ra_deg: np.ndarray, dec_deg: np.ndarray,
                                   binary: np.ndarray) -> np.ndarray:
    """Angular distance [deg] from each position to the nearest OUTSIDE pixel.

    Boundary = outside pixels adjacent to the footprint; exact chord-distance
    KDTree query against the boundary set (fast for ~1e5 boundary px).
    """
    import healpy as hp
    from scipy.spatial import cKDTree

    binary = np.asarray(binary, dtype=bool)
    nside = hp.npix2nside(binary.size)
    inside = np.flatnonzero(binary)
    neigh = hp.get_all_neighbours(nside, inside)          # (8, n_in)
    has_out = ~binary[np.clip(neigh, 0, None)] | (neigh < 0)
    edge_inside = inside[has_out.any(axis=0)]             # inside px touching outside
    # the nearest OUTSIDE point is ~half a pixel beyond the edge-inside pixel;
    # using edge-inside pixels directly is accurate to ~pixel scale (3.4')
    vec_edge = np.array(hp.pix2vec(nside, edge_inside)).T
    tree = cKDTree(vec_edge)
    vec_p = np.array(hp.ang2vec(np.radians(90.0 - np.asarray(dec_deg)),
                                np.radians(np.asarray(ra_deg)))).reshape(-1, 3)
    chord, _ = tree.query(vec_p)
    return np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))


def proximity_tercile_test(per_peak_y: np.ndarray, nu: np.ndarray,
                           dist_deg: np.ndarray,
                           nu_edges: np.ndarray = NU_STACK_EDGES) -> dict:
    """NULL_CRITERIA §5: near-minus-far tercile difference per detected bin."""
    out = {}
    binof = np.digitize(nu, nu_edges) - 1
    for b in range(len(nu_edges) - 1):
        m = binof == b
        if m.sum() < 30:
            out[f"bin{b}"] = {"n": int(m.sum()), "verdict": "too few peaks"}
            continue
        d, y = dist_deg[m], per_peak_y[m][:, J]
        q1, q2 = np.quantile(d, [1 / 3, 2 / 3])
        near, far = y[d <= q1], y[d > q2]
        diff = near.mean() - far.mean()
        err = np.hypot(near.std(ddof=1) / np.sqrt(len(near)),
                       far.std(ddof=1) / np.sqrt(len(far)))
        out[f"bin{b}"] = {
            "n": int(m.sum()), "near_mean": float(near.mean()),
            "far_mean": float(far.mean()), "diff": float(diff),
            "diff_err": float(err), "sigma": float(diff / err),
            "tercile_edges_deg": [float(q1), float(q2)],
        }
    return out


# ------------------------------------------------ 6. interior-threshold cut

def threshold_recut_test(per_peak_y: np.ndarray, nu: np.ndarray,
                         weight_at_peak: np.ndarray, patch8: np.ndarray,
                         y_mean_ref: np.ndarray, y_err_ref: np.ndarray,
                         tight: float = 0.999,
                         nu_edges: np.ndarray = NU_STACK_EDGES) -> dict:
    """NULL_CRITERIA §6: re-cut to ACT >= tight, |delta<Y>| vs 0.5 sigma_stat."""
    from .covariance import jackknife_mean_cov

    keep = weight_at_peak >= tight
    binof = np.digitize(nu, nu_edges) - 1
    out = {"tight_threshold": tight, "kept_fraction": float(keep.mean())}
    for b in range(len(nu_edges) - 1):
        m = (binof == b) & keep
        if m.sum() < 30:
            out[f"bin{b}"] = {"n": int(m.sum()), "verdict": "too few peaks"}
            continue
        mean, cov, _ = jackknife_mean_cov(per_peak_y[m][:, [J]], patch8[m])
        d = float(mean[0] - y_mean_ref[b])
        out[f"bin{b}"] = {
            "n": int(m.sum()), "y_mean_tight": float(mean[0]),
            "delta": d, "delta_over_sigma_stat": d / float(y_err_ref[b]),
            "pass_0p5sig": bool(abs(d) < 0.5 * float(y_err_ref[b])),
        }
    return out


# --------------------------------------------------- 4. CIB band assembly

def cib_band(y_by_variant: dict[str, np.ndarray], sigma_stat: np.ndarray) -> dict:
    """NULL_CRITERIA §4: per-bin max-min band over the variant set + verdicts."""
    arr = np.array(list(y_by_variant.values()))           # (nvar, nbin)
    band = arr.max(axis=0) - arr.min(axis=0)
    ratio = band / np.asarray(sigma_stat)
    verdicts = np.where(ratio < 0.5, "negligible",
                        np.where(ratio <= 1.0, "sigma_sys_term", "exceeds_sigma"))
    return {
        "variants": list(y_by_variant),
        "band": band.tolist(), "band_over_sigma_stat": ratio.tolist(),
        "verdict_per_bin": verdicts.tolist(),
        "escalate": bool((ratio[DETECTED_BINS] > 1.0).all()),
        "sigma_sys_rank1_halfband": (band / 2.0).tolist(),
    }


# ------------------------------------------------- 1. null-ensemble stats

def null_ensemble_stats(means: np.ndarray, jk_sigmas: np.ndarray) -> dict:
    """NULL_CRITERIA §1: (nreal, nbin) realization means + jackknife sigmas."""
    means = np.asarray(means)
    n = means.shape[0]
    ens_mean = means.mean(axis=0)
    ens_scatter = means.std(axis=0, ddof=1)
    return {
        "n_realizations": int(n),
        "ensemble_mean": ens_mean.tolist(),
        "ensemble_scatter": ens_scatter.tolist(),
        "mean_over_2err": (ens_mean / (2 * ens_scatter / np.sqrt(n))).tolist(),
        "jk_validation_ratio": (ens_scatter / np.asarray(jk_sigmas).mean(axis=0)).tolist(),
    }
