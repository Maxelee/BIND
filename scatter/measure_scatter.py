"""scatter/measure_scatter.py
Scatter measurement engine for the BIND scatter paper.

Public API
----------
measure_scatter(
    model_fm, norm_stats, theta_norm, dmo_conds, ls_conds,
    masses, r200_pix, K=10, n_steps=20, device="cuda", batch_size=32,
    dmo_raw=None, q_dmo=None, omega_m=None, seed=42,
) -> dict

Returns
-------
dict with:
    obs_tensor  : (N_h, K, N_obs) float32 — raw observable values
    obs_names   : list[str] length N_obs
    log_mask    : (N_obs,) bool — True for observables that use log10 variance
    masses      : (N_h,) float64 — pass-through for binning
    sigma_inter : (N_obs,) float64 — std of per-halo means (halo-to-halo scatter)
    sigma_intra : (N_obs,) float64 — mean of per-halo stds (model stochasticity)
    sigma_total : (N_obs,) float64 — pooled total std
    Y_bar       : (N_h, N_obs) float64 — per-halo mean observable in Y-space
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scatter.obs_common import (
    PATCH_PIX,
    OMEGA_B_FIXED,
    OBS_KEYS,
    observables_from_phys,
    axis_ratio_q,
    _RR_PIX,
)
from fd_jacobian_cv import _sample_fixed_noise
from test_suite.pipeline import _denormalize_to_physical

# ---------------------------------------------------------------------------
# Observable metadata

# 5 log-spaced annular fractions of R200c for gas surface density profile
PROFILE_FRACS = np.logspace(-1, 0, 5)  # ≈ [0.10, 0.18, 0.32, 0.56, 1.00]
PROFILE_OBS_NAMES = [f"Sigma_gas_r{i}" for i in range(5)]

# Full observable list: 11 from fd_jacobian_cv + 5 profile bins = 16 total
ALL_OBS_NAMES: list[str] = list(OBS_KEYS) + PROFILE_OBS_NAMES

# Headline subset (used for fig2 / scatter–mean Jacobian plots)
HEADLINE_OBS_NAMES: list[str] = (
    ["M_dm", "M_gas", "M_star", "q_DM", "q_gas", "q_star", "dq_DM"]
    + PROFILE_OBS_NAMES
)

# Which observables use log10 space for variance decomposition
_LOG_SET = {"M_dm", "M_gas", "M_star", "Sigma_gas_c"} | set(PROFILE_OBS_NAMES)
LOG_MASK: np.ndarray = np.array([n in _LOG_SET for n in ALL_OBS_NAMES], dtype=bool)


# ---------------------------------------------------------------------------
# Radial profile helper

def _gas_annular_profile(gas_2d: np.ndarray, r200_pix: float) -> np.ndarray:
    """Mean gas surface density in 5 log-spaced annuli within R200c.

    Annulus i spans [PROFILE_FRACS[i-1]*r200_pix, PROFILE_FRACS[i]*r200_pix).
    The first annulus spans [0, PROFILE_FRACS[0]*r200_pix).
    """
    result = np.full(5, np.nan, dtype=np.float32)
    r_edges = np.concatenate([[0.0], PROFILE_FRACS * r200_pix])
    g = np.maximum(gas_2d, 0.0)
    for i in range(5):
        mask = (_RR_PIX >= r_edges[i]) & (_RR_PIX < r_edges[i + 1])
        if mask.sum() > 0:
            result[i] = float(g[mask].mean())
    return result


def _compute_all_obs(
    phys_3hw: np.ndarray,
    r200_pix: float,
    f_b_cosmic: float,
    q_dmo_val: float,
) -> np.ndarray:
    """All observables for one (3, H, W) physical patch → (N_obs,) float32."""
    obs = observables_from_phys(phys_3hw, r200_pix, f_b_cosmic, q_dmo_val)
    profile = _gas_annular_profile(phys_3hw[1], r200_pix)
    return np.array([obs[k] for k in OBS_KEYS] + list(profile), dtype=np.float32)


# ---------------------------------------------------------------------------
# Main public function

def measure_scatter(
    model_fm,
    norm_stats,
    theta_norm: np.ndarray,
    dmo_conds: np.ndarray,
    ls_conds: np.ndarray,
    masses: np.ndarray,
    r200_pix: np.ndarray,
    K: int = 10,
    n_steps: int = 20,
    device: str = "cuda",
    batch_size: int = 32,
    dmo_raw: Optional[np.ndarray] = None,
    q_dmo: Optional[np.ndarray] = None,
    omega_m: Optional[np.ndarray] = None,
    seed: int = 42,
) -> dict:
    """Measure per-halo scatter via K independent model samples.

    Parameters
    ----------
    model_fm : FlowMatching
        The generative model (from ``FlowMatchingLit.load_from_checkpoint().fm``).
    norm_stats : NormStats
        Normalization statistics used for denormalizing model output.
    theta_norm : (35,) float32
        Normalized parameter vector (after min-max scaling).  All halos use
        the same theta — vary this across calls to probe parameter dependence.
    dmo_conds : (N_h, 1, 128, 128) float32
        Pre-normalized DMO condition patches (log-transformed and z-scored).
    ls_conds : (N_h, 3, 128, 128) float32
        Pre-normalized large-scale context patches.
    masses : (N_h,) float64
        Halo masses [M_sun/h] for downstream mass-bin filtering.
    r200_pix : (N_h,) float64
        R200c in pixels.
    K : int
        Number of independent noise draws per halo.
    n_steps : int
        Number of Euler integration steps.
    device : str
        "cuda" or "cpu".
    batch_size : int
        Number of halos processed per GPU batch (each yields K samples →
        batch_size * K forward passes).
    dmo_raw : (N_h, 128, 128) float32, optional
        Raw (un-normalized) DMO condition patches used to compute q_DMO.
        Ignored if ``q_dmo`` is provided.
    q_dmo : (N_h,) float64, optional
        Pre-computed DMO axis ratio q_DMO per halo.  Takes precedence over
        ``dmo_raw``.  Defaults to 0.0 per halo if neither is provided.
    omega_m : (N_h,) float64, optional
        Omega_m per halo for computing f_b_cosmic = OMEGA_B / Omega_m.
        Defaults to NaN (f_b and Rc_over_R200 will be NaN).
    seed : int
        RNG seed for all noise draws.  Identical seeds with different ``theta_norm``
        produce maximally correlated noise, reducing variance in sigma differences.

    Returns
    -------
    dict
        obs_tensor  : (N_h, K, N_obs) float32
        obs_names   : list[str]  — length N_obs (= 16)
        log_mask    : (N_obs,) bool
        masses      : (N_h,) float64
        sigma_inter : (N_obs,) float64
        sigma_intra : (N_obs,) float64
        sigma_total : (N_obs,) float64
        Y_bar       : (N_h, N_obs) float64
    """
    dev = torch.device(device)
    N_h = len(masses)
    N_obs = len(ALL_OBS_NAMES)

    # Ensure dmo_conds has the channel dimension
    if dmo_conds.ndim == 3:
        dmo_conds = dmo_conds[:, np.newaxis]  # (N_h, 128, 128) → (N_h, 1, 128, 128)

    # Precompute per-halo ancillary quantities
    if q_dmo is not None:
        q_dmo_arr = np.asarray(q_dmo, dtype=np.float64)
    elif dmo_raw is not None:
        q_dmo_arr = np.full(N_h, np.nan, dtype=np.float64)
        for i in range(N_h):
            r_aper = max(min(float(r200_pix[i]), PATCH_PIX / 2 - 2), 4.0)
            q_dmo_arr[i] = axis_ratio_q(np.maximum(dmo_raw[i].astype(np.float64), 0.0), r_aper)
    else:
        q_dmo_arr = np.zeros(N_h, dtype=np.float64)

    if omega_m is not None:
        f_b_arr = OMEGA_B_FIXED / np.where(omega_m > 0, np.asarray(omega_m, dtype=np.float64), np.nan)
    else:
        f_b_arr = np.full(N_h, np.nan, dtype=np.float64)

    # Output storage
    obs_tensor = np.full((N_h, K, N_obs), np.nan, dtype=np.float32)

    # Reproducible RNG
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    out_channels = model_fm.out_channels
    theta_t = torch.tensor(theta_norm, dtype=torch.float32, device=dev)

    with torch.no_grad():
        for start in range(0, N_h, batch_size):
            stop = min(start + batch_size, N_h)
            B = stop - start

            cb = torch.tensor(dmo_conds[start:stop], dtype=torch.float32, device=dev)
            lb = torch.tensor(ls_conds[start:stop], dtype=torch.float32, device=dev)

            # Expand each halo K times → (B*K, C, H, W)
            cb_exp = cb.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, 1, PATCH_PIX, PATCH_PIX)
            lb_exp = lb.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, 3, PATCH_PIX, PATCH_PIX)
            params_exp = theta_t.unsqueeze(0).expand(B * K, -1).contiguous()

            # K independent noise tensors per halo
            noise = torch.randn(
                B * K, out_channels, PATCH_PIX, PATCH_PIX,
                device=dev, generator=gen,
            )

            x_gen = _sample_fixed_noise(model_fm, cb_exp, lb_exp, params_exp, noise, n_steps)

            phys = _denormalize_to_physical(x_gen.cpu().numpy(), norm_stats)
            # phys: (B*K, 3, H, W) → (B, K, 3, H, W)
            phys_bk = phys.reshape(B, K, 3, PATCH_PIX, PATCH_PIX)

            for b in range(B):
                h = start + b
                for k in range(K):
                    obs_tensor[h, k] = _compute_all_obs(
                        phys_bk[b, k],
                        float(r200_pix[h]),
                        float(f_b_arr[h]),
                        float(q_dmo_arr[h]),
                    )

    # -----------------------------------------------------------------------
    # Variance decomposition
    # Y_{h,k}^{(o)} = log10(X) if LOG_MASK[o] else X
    Y = np.full_like(obs_tensor, np.nan, dtype=np.float64)
    for o in range(N_obs):
        x = obs_tensor[:, :, o].astype(np.float64)
        if LOG_MASK[o]:
            with np.errstate(divide="ignore", invalid="ignore"):
                Y[:, :, o] = np.where(x > 0, np.log10(x), np.nan)
        else:
            Y[:, :, o] = x

    # Per-halo mean over K: (N_h, N_obs)
    Y_bar = np.nanmean(Y, axis=1)

    # sigma_intra: mean over halos of within-halo std_k
    per_halo_std = np.nanstd(Y, axis=1, ddof=1)       # (N_h, N_obs)
    sigma_intra = np.nanmean(per_halo_std, axis=0)    # (N_obs,)

    # sigma_inter: std over halo means
    sigma_inter = np.nanstd(Y_bar, axis=0, ddof=1)    # (N_obs,)

    # sigma_total: pooled across (h, k)
    Y_flat = Y.reshape(N_h * K, N_obs)
    sigma_total = np.nanstd(Y_flat, axis=0, ddof=1)   # (N_obs,)

    return {
        "obs_tensor": obs_tensor,
        "obs_names": ALL_OBS_NAMES,
        "log_mask": LOG_MASK,
        "masses": np.asarray(masses, dtype=np.float64),
        "sigma_inter": sigma_inter,
        "sigma_intra": sigma_intra,
        "sigma_total": sigma_total,
        "Y_bar": Y_bar,
    }
