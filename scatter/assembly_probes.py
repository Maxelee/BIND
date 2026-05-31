"""scatter/assembly_probes.py — turn 'assembly scatter' into measured halo structure.

The variance decomposition labels the between-halo term 'assembly', but that is just unexplained
halo-to-halo variance. Here we MEASURE structural properties of each halo from its DMO conditioning
field — quantities that are physical assembly proxies — and ask how much of the assembly scatter
they actually explain. This converts 'assembly' from a residual into identified physics, and lets us
see what (if anything) is left over as truly irreducible.

DMO structural proxies (all from the conditioning field; no model needed):
  conc        M_DMO(<0.5 R200) / M_DMO(<R200)   — concentration; tracks formation time (early -> high)
  conc_inner  M_DMO(<0.25 R200) / M_DMO(<R200)  — core concentration
  peak_frac   peak pixel / M_DMO(<R200)         — central cuspiness
  rhalf       half-mass radius / R200           — compactness (anti-concentration)
  q_dmo       projected axis ratio b/a          — shape; relaxation / recent-merger proxy
  logM_dmo    log10 M_DMO(<R200)                — DMO mass within aperture
  environment sum of large-scale context field  — local overdensity (assembly-bias proxy)
"""
from __future__ import annotations

import numpy as np

from scatter.obs_common import aperture_sum, axis_ratio_q, PATCH_PIX, _RR_PIX


def _half_mass_radius_pix(field_2d: np.ndarray) -> float:
    """Radius (pixels) enclosing half the positive mass, via cumulative radial sort."""
    f = np.maximum(field_2d.astype(np.float64), 0.0).ravel()
    r = _RR_PIX.ravel()
    order = np.argsort(r)
    csum = np.cumsum(f[order])
    tot = csum[-1]
    if tot <= 0:
        return np.nan
    k = int(np.searchsorted(csum, 0.5 * tot))
    return float(r[order][min(k, len(r) - 1)])


def dmo_structural_props(cond_2d: np.ndarray, r200_pix: float,
                         ls_3d: np.ndarray | None = None) -> dict:
    """Structural/assembly proxies for one halo from its DMO conditioning field."""
    cond = np.maximum(cond_2d.astype(np.float64), 0.0)
    r = max(min(float(r200_pix), PATCH_PIX / 2 - 2), 4.0)
    M_r = aperture_sum(cond, r)
    M_half = aperture_sum(cond, 0.5 * r)
    M_quart = aperture_sum(cond, 0.25 * r)
    out = {
        "conc":       M_half / M_r if M_r > 0 else np.nan,
        "conc_inner": M_quart / M_r if M_r > 0 else np.nan,
        "peak_frac":  float(cond.max()) / M_r if M_r > 0 else np.nan,
        "rhalf":      _half_mass_radius_pix(cond) / r200_pix if r200_pix > 0 else np.nan,
        "q_dmo":      axis_ratio_q(cond, r),
        "logM_dmo":   np.log10(M_r) if M_r > 0 else np.nan,
    }
    if ls_3d is not None:
        out["environment"] = float(np.maximum(ls_3d.astype(np.float64), 0.0).sum())
    return out


PROP_NAMES = ["conc", "conc_inner", "peak_frac", "rhalf", "q_dmo", "logM_dmo", "environment"]


def compute_props_for_halos(cond_raw: np.ndarray, r200_pix: np.ndarray,
                            ls_raw: np.ndarray | None = None) -> dict:
    """(N,128,128) DMO fields -> {prop: (N,) array}. ls_raw optional (N,3,128,128)."""
    n = len(cond_raw)
    props = {p: np.full(n, np.nan) for p in PROP_NAMES}
    for i in range(n):
        d = dmo_structural_props(cond_raw[i], float(r200_pix[i]),
                                 ls_3d=ls_raw[i] if ls_raw is not None else None)
        for p, v in d.items():
            props[p][i] = v
    return props


def assembly_residual(cube_obs: np.ndarray, log_mass: np.ndarray, is_log: bool) -> np.ndarray:
    """Per-halo offset from the mean mass-relation, averaged over physics and noise.

    cube_obs: (n_theta, N_h, K) for one observable. Returns (N_h,) residual = the halo's
    assembly-driven deviation, with the deterministic mass trend, feedback, and noise removed.
    """
    X = np.log10(np.clip(cube_obs, 1e-30, None)) if is_log else cube_obs.astype(float)
    ybar = np.nanmean(X, axis=(0, 2))                       # (N_h,) over theta & noise
    ok = np.isfinite(ybar) & np.isfinite(log_mass)
    if ok.sum() >= 3:
        b, a = np.polyfit(log_mass[ok], ybar[ok], 1)
        return ybar - (a + b * log_mass)
    return ybar - np.nanmean(ybar)


def intrinsic_per_halo(cube_obs: np.ndarray, is_log: bool) -> np.ndarray:
    """Per-halo intrinsic spread: mean over physics of the within-(halo,theta) std over noise."""
    X = np.log10(np.clip(cube_obs, 1e-30, None)) if is_log else cube_obs.astype(float)
    return np.nanmean(np.nanstd(X, axis=2), axis=0)         # (N_h,)
