"""scatter/obs_common.py — shared, side-effect-free observable definitions.

Single source of truth for the 11 scalar group-scale observables and their
geometry helpers. Extracted verbatim from the PHYSICS_HELPERS block of
gen_obs_notebook.py (where they previously lived only as notebook-source text,
hence un-importable). measure_scatter.py and the figure scripts import from here.

Geometry constants MUST match fd_jacobian_cv.py exactly (PATCH_PIX=128 grid).
This module imports nothing beyond numpy and has no module-level side effects.
"""
from __future__ import annotations

import numpy as np

# ── geometry / cosmology constants — MUST match fd_jacobian_cv.py ────────────
RHO_CRIT          = 2.775e11     # M_sun/h per (Mpc/h)^3
OMEGA_B_FIXED     = 0.049
CLOSURE_THRESHOLD = 0.90

BOX_SIZE   = 50.0
N_PIX_FULL = 1024
PATCH_PIX  = 128
PATCH_BOX  = BOX_SIZE * PATCH_PIX / N_PIX_FULL   # 6.25 Mpc/h
MPC_PER_PIX = PATCH_BOX / PATCH_PIX

_NB = 32
_yy, _xx       = (np.mgrid[:PATCH_PIX, :PATCH_PIX]
                  - np.array([PATCH_PIX / 2, PATCH_PIX / 2])[:, None, None])
_RR_PIX        = np.sqrt(_xx ** 2 + _yy ** 2)
_BIN_EDGES_PIX = np.linspace(0, PATCH_PIX / 2, _NB + 1)
_R_CENTRES_PIX = 0.5 * (_BIN_EDGES_PIX[:-1] + _BIN_EDGES_PIX[1:])
_BIN_MASKS     = [
    (_RR_PIX >= _BIN_EDGES_PIX[k]) & (_RR_PIX < _BIN_EDGES_PIX[k + 1])
    for k in range(_NB)
]
_N_PIX_PER_BIN = np.array([m.sum() for m in _BIN_MASKS], dtype=np.float64)
_R_MPC         = _R_CENTRES_PIX * MPC_PER_PIX     # radial bin centres in Mpc/h

# ── 11 scalar observable keys (order is canonical) ───────────────────────────
OBS_KEYS = [
    'M_dm', 'M_gas', 'M_star',
    'f_b', 'f_b_norm',
    'Rc_over_R200',
    'q_DM', 'q_gas', 'q_star',
    'dq_DM',
    'Sigma_gas_c',
]


def r200c_pix(m200c_msunh):
    """R200c in patch pixels from halo mass [M_sun/h]."""
    r_mpc = (3.0 * m200c_msunh / (4.0 * np.pi * 200.0 * RHO_CRIT)) ** (1.0 / 3.0)
    return r_mpc / MPC_PER_PIX


def aperture_sum(field_2d, r_pix):
    """Sum of positive pixel values within a circular aperture of radius r_pix."""
    return float(np.maximum(field_2d, 0.0)[_RR_PIX < r_pix].sum())


def _radial_profile_2d(field_2d):
    """Azimuthal mean in _NB annular bins from centre to PATCH_PIX/2."""
    return np.array([
        field_2d[m].mean() if c > 0 else 0.0
        for m, c in zip(_BIN_MASKS, _N_PIX_PER_BIN)
    ])


def axis_ratio_q(field_2d, r_aper_pix, max_iter=5, tol=1e-3, min_pixels=8):
    """
    Iterative ellipsoidal moment axis ratio q = b/a in [0, 1].

    Uses mass-weighted 2D quadrupole moments of the positive flux within an
    adaptive elliptical aperture (semi-major axis = r_aper_pix, semi-minor
    axis = q * r_aper_pix).  Returns NaN for empty or insufficiently resolved
    fields.
    """
    H, W   = field_2d.shape
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    f      = field_2d.astype(np.float64)
    q, pa  = 1.0, 0.0
    for _ in range(max_iter):
        c_r, s_r = np.cos(pa), np.sin(pa)
        x_rot = c_r * dx + s_r * dy
        y_rot = -s_r * dx + c_r * dy
        a, b  = r_aper_pix, max(r_aper_pix * q, 1.0)
        mask  = (x_rot / a) ** 2 + (y_rot / b) ** 2 < 1.0
        if mask.sum() < min_pixels:
            return np.nan
        w   = np.where(mask, np.maximum(f, 0.0), 0.0)
        tot = w.sum()
        if tot <= 0:
            return np.nan
        Qxx = (dx * dx * w).sum() / tot
        Qyy = (dy * dy * w).sum() / tot
        Qxy = (dx * dy * w).sum() / tot
        evals, evecs = np.linalg.eigh(np.array([[Qxx, Qxy], [Qxy, Qyy]]))
        lam_min, lam_max = float(evals[0]), float(evals[1])
        if lam_max <= 0 or lam_min < 0:
            return np.nan
        q_new  = float(np.sqrt(lam_min / lam_max))
        pa_new = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
        if abs(q_new - q) < tol:
            return q_new
        q, pa  = q_new, pa_new
    return q


def closure_radius_pix(p_dm, p_gas, p_star, f_b_cosmic,
                       threshold=CLOSURE_THRESHOLD):
    """
    Smallest radius (in pixels) at which the enclosed baryon fraction
    f_b(< r) first reaches `threshold` * f_b_cosmic.

    Computed from cumulative sums of the azimuthal-mean profiles weighted by
    annular pixel area.  Returns NaN when the ratio never exceeds the threshold
    within the patch.
    """
    cum_dm   = np.cumsum(p_dm   * _N_PIX_PER_BIN)
    cum_gas  = np.cumsum(p_gas  * _N_PIX_PER_BIN)
    cum_star = np.cumsum(p_star * _N_PIX_PER_BIN)
    cum_tot  = cum_dm + cum_gas + cum_star
    cum_b    = cum_gas + cum_star
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cum_tot > 0, cum_b / cum_tot, np.nan) / f_b_cosmic
    ok = np.isfinite(ratio) & (ratio >= threshold)
    if not ok.any():
        return np.nan
    k = int(np.argmax(ok))
    if k > 0 and np.isfinite(ratio[k - 1]):
        lo, hi = ratio[k - 1], ratio[k]
        if hi > lo:
            frac = (threshold - lo) / (hi - lo)
            return (1 - frac) * _R_CENTRES_PIX[k - 1] + frac * _R_CENTRES_PIX[k]
    return _R_CENTRES_PIX[k]


def observables_from_phys(phys_3HW, r200_pix, f_b_cosmic, q_DMO_const):
    """
    Compute all 11 scalar observables from a (3, 128, 128) physical-unit map.

    Parameters
    ----------
    phys_3HW   : (3, H, W) array  [DM | Gas | Stars] in M_sun/h per pixel
    r200_pix   : float   R200c in patch pixels
    f_b_cosmic : float   Omega_b / Omega_m for this simulation
    q_DMO_const: float   axis ratio of the DMO condition map (for dq_DM)
    """
    dm, gas, star = phys_3HW[0], phys_3HW[1], phys_3HW[2]
    r_aper = max(min(r200_pix, PATCH_PIX / 2 - 2), 4.0)

    # 1–3. Enclosed masses within R200c
    M_dm   = aperture_sum(dm,   r_aper)
    M_gas  = aperture_sum(gas,  r_aper)
    M_star = aperture_sum(star, r_aper)
    M_tot  = M_dm + M_gas + M_star
    M_b    = M_gas + M_star

    # 4–5. Baryon fractions
    f_b      = M_b / M_tot if M_tot > 0 else np.nan
    f_b_norm = (f_b / f_b_cosmic
                if (M_tot > 0 and np.isfinite(f_b_cosmic)) else np.nan)

    # 6. Closure radius
    p_dm   = _radial_profile_2d(np.maximum(dm,   0.0))
    p_gas  = _radial_profile_2d(np.maximum(gas,  0.0))
    p_star = _radial_profile_2d(np.maximum(star, 0.0))
    Rc_pix = closure_radius_pix(p_dm, p_gas, p_star, f_b_cosmic)
    Rc_over_R200 = Rc_pix / r200_pix if np.isfinite(Rc_pix) else np.nan

    # 7–9. Projected axis ratios
    q_dm   = axis_ratio_q(dm,   r_aper)
    q_gas  = axis_ratio_q(gas,  r_aper)
    q_star = axis_ratio_q(star, r_aper)

    # 10. DM back-reaction in shape relative to DMO
    dq_DM = q_dm - q_DMO_const

    # 11. Central gas surface density within 0.1 R200c
    r_c = max(0.1 * r200_pix, 2.0)
    Sigma_gas_c = float(np.maximum(gas, 0.0)[_RR_PIX < r_c].mean())

    return {
        'M_dm': M_dm, 'M_gas': M_gas, 'M_star': M_star,
        'f_b': f_b, 'f_b_norm': f_b_norm,
        'Rc_over_R200': Rc_over_R200,
        'q_DM': q_dm, 'q_gas': q_gas, 'q_star': q_star,
        'dq_DM': dq_DM, 'Sigma_gas_c': Sigma_gas_c,
    }
