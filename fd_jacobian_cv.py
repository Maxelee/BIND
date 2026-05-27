"""FD Jacobian of scatter.ipynb statistics w.r.t. all 35 CAMELS parameters,
across every CV halo at the fiducial parameter point.

Per-halo statistics (35 scalars, shape J[key] = (N_halos, 35)):
  Masses   : M_dm, M_gas, M_star, M_bar  (aperture sum within R200c)
  Fractions: f_bar, f_gas, f_star         (relative to M_tot)
  Axis ratio at R200c  : q_dm_R, q_gas_R, q_star_R
  Axis ratio at 0.5R   : q_dm_h, q_gas_h, q_star_h
  Ellipticity at R200c : e1_dm, e1_gas, e1_star, e2_dm, e2_gas, e2_star
  Concentration        : conc_dm, conc_gas, conc_star  (M(<0.5R)/M(<R))
  Half-mass radius/R   : rhalf_dm, rhalf_gas, rhalf_star
  Central 4px/M        : cent4_dm, cent4_gas, cent4_star
  Peak pixel/M         : peak_dm, peak_gas, peak_star
  Misalignment [deg]   : dtheta_dg, dtheta_ds, dtheta_gs
  Baryonic spherization: dq_dm  (q_DM_hydro − q_DMO)

Population-level statistics (shape Jpop[key] = (35,)); computed by fitting
scaling relations across all N_halos at each ±eps perturbation:
  Mgas–Mstar  : alpha_MgMs, beta_MgMs, sigma_MgMs
  Mdm–Mstar   : alpha_MdMs, beta_MdMs, sigma_MdMs
  SHMR        : alpha_SHMR,  beta_SHMR,  sigma_SHMR
  GasFrac     : alpha_GasFr, beta_GasFr, sigma_GasFr
  BaryonFrac  : alpha_BarFr, beta_BarFr, sigma_BarFr

Run as a SLURM array job; each task processes a contiguous slice of the 35
parameters. The full 35-param grid is split into --n_chunks shards.

Single-GPU usage:
    python fd_jacobian_cv.py --n_chunks 1 --chunk_id 0 --output cv_fd.npz

Sharded (one chunk per GPU job):
    python fd_jacobian_cv.py --n_chunks 7 --chunk_id 0 \\
        --output analysis_physics_cache/proj6_cv_fd_shard0.npz
    ... (chunks 1..6 in parallel) ...

Merge:
    python fd_jacobian_cv.py --merge \\
        --shard_glob 'analysis_physics_cache/proj6_cv_fd_shard*.npz' \\
        --output analysis_physics_cache/proj6_cv_fd_fm_two_head.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import NormStats, log_transform
from test_suite.pipeline import _denormalize_to_physical
from train import FlowMatchingLit

# ---------------------------------------------------------------------------
# Geometry / physics constants (match project1/project2 conventions)

PATCH_PIX     = 128
N_PARAMS      = 35
RHO_CRIT      = 2.775e11           # M_sun/h per (Mpc/h)^3
BOX_SIZE      = 50.0               # Mpc/h
N_PIX_FULL    = 1024
PATCH_BOX     = BOX_SIZE * PATCH_PIX / N_PIX_FULL
MPC_PER_PIX   = PATCH_BOX / PATCH_PIX
OMEGA_B_FIXED = 0.049
CLOSURE_THRESHOLD = 0.90

# Per-halo observable keys  (J[key] has shape (N_halos, N_params))
PER_HALO_KEYS = [
    # aperture-integrated masses within R200c
    "M_dm", "M_gas", "M_star", "M_bar",
    # baryon/component fractions
    "f_bar", "f_gas", "f_star",
    # axis ratio at R200c and 0.5*R200c
    "q_dm_R", "q_gas_R", "q_star_R",
    "q_dm_h", "q_gas_h", "q_star_h",
    # ellipticity components at R200c
    "e1_dm", "e1_gas", "e1_star",
    "e2_dm", "e2_gas", "e2_star",
    # concentration M(<0.5R200c) / M(<R200c)
    "conc_dm", "conc_gas", "conc_star",
    # half-mass radius / R200c
    "rhalf_dm", "rhalf_gas", "rhalf_star",
    # central 4-px flux / M(<R200c)
    "cent4_dm", "cent4_gas", "cent4_star",
    # peak pixel / M(<R200c)
    "peak_dm", "peak_gas", "peak_star",
    # inter-component misalignment angles [deg]
    "dtheta_dg", "dtheta_ds", "dtheta_gs",
    # baryonic spherization: q_DM_hydro − q_DMO
    "dq_dm",
]

# Population-level observable keys  (Jpop[key] has shape (N_params,))
# Scaling-relation slope (alpha), normalisation (beta), scatter (sigma)
POP_KEYS = [
    "alpha_MgMs", "beta_MgMs", "sigma_MgMs",   # log Mgas = alpha*log Mstar + beta
    "alpha_MdMs", "beta_MdMs", "sigma_MdMs",   # log Mdm  = alpha*log Mstar + beta
    "alpha_SHMR",  "beta_SHMR",  "sigma_SHMR",  # log Mstar = alpha*log M200c + beta
    "alpha_GasFr", "beta_GasFr", "sigma_GasFr", # log Mgas  = alpha*log M200c + beta
    "alpha_BarFr", "beta_BarFr", "sigma_BarFr", # log Mbar  = alpha*log M200c + beta
]

# Annulus geometry (matches metrics.radial_profile, n_bins=32)
_NB = 32
_yy, _xx = np.mgrid[:PATCH_PIX, :PATCH_PIX] - np.array([PATCH_PIX / 2, PATCH_PIX / 2])[:, None, None]
_RR_PIX = np.sqrt(_xx ** 2 + _yy ** 2)
_BIN_EDGES_PIX = np.linspace(0, PATCH_PIX / 2, _NB + 1)
_R_CENTRES_PIX = 0.5 * (_BIN_EDGES_PIX[:-1] + _BIN_EDGES_PIX[1:])
_BIN_MASKS = [(_RR_PIX >= _BIN_EDGES_PIX[k]) & (_RR_PIX < _BIN_EDGES_PIX[k + 1]) for k in range(_NB)]
_N_PIX_PER_BIN = np.array([m.sum() for m in _BIN_MASKS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Physics helpers (copied verbatim from the notebook cells, kept in numpy)

def r200c_mpc_h(m200c_msunh):
    return (3.0 * m200c_msunh / (4.0 * np.pi * 200.0 * RHO_CRIT)) ** (1.0 / 3.0)


def _radial_profile_2d(field_2d):
    return np.array([
        field_2d[m].mean() if c > 0 else 0.0
        for m, c in zip(_BIN_MASKS, _N_PIX_PER_BIN)
    ])


def aperture_sum(field_2d, r_pix):
    mask = _RR_PIX < r_pix
    return float(np.maximum(field_2d, 0.0)[mask].sum())


def shape_moments_2d(field_2d, r_aper_pix, max_iter=10, tol=1e-3, min_pixels=8):
    """Iterative mass-weighted 2D quadrupole in an elliptical aperture.
    Returns (q, pa_rad, e1, e2) or (nan, nan, nan, nan) on failure.
    Matches scatter.ipynb shape_moments() exactly.
    """
    H, W = field_2d.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    f = np.maximum(field_2d.astype(np.float64), 0.0)
    q, pa = 1.0, 0.0
    for _ in range(max_iter):
        c, s = np.cos(pa), np.sin(pa)
        xr =  c * dx + s * dy
        yr = -s * dx + c * dy
        a = r_aper_pix
        b = max(r_aper_pix * q, 1.0)
        mask = (xr / a) ** 2 + (yr / b) ** 2 < 1.0
        if mask.sum() < min_pixels:
            return np.nan, np.nan, np.nan, np.nan
        w = f * mask
        tot = w.sum()
        if tot <= 0:
            return np.nan, np.nan, np.nan, np.nan
        Qxx = (dx * dx * w).sum() / tot
        Qyy = (dy * dy * w).sum() / tot
        Qxy = (dx * dy * w).sum() / tot
        evals, evecs = np.linalg.eigh(np.array([[Qxx, Qxy], [Qxy, Qyy]]))
        lam_min, lam_max = float(evals[0]), float(evals[1])
        if lam_max <= 0 or lam_min < 0:
            return np.nan, np.nan, np.nan, np.nan
        q_new  = float(np.sqrt(lam_min / lam_max))
        pa_new = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
        if abs(q_new - q) < tol:
            q, pa = q_new, pa_new
            break
        q, pa = q_new, pa_new
    eps = (1.0 - q) / (1.0 + q)
    return q, pa, eps * np.cos(2 * pa), eps * np.sin(2 * pa)


# Precompute flattened radius array for half_mass_r_pix
_RR_PIX_FLAT = _RR_PIX.ravel()
_SORT_IDX_R  = np.argsort(_RR_PIX_FLAT)
_SORTED_R    = _RR_PIX_FLAT[_SORT_IDX_R]


def half_mass_r_pix(field_2d):
    """Half-mass radius in pixels from the patch centre."""
    m_sorted = np.maximum(field_2d, 0.0).ravel()[_SORT_IDX_R]
    m_cum = np.cumsum(m_sorted)
    tot = m_cum[-1]
    if tot <= 0:
        return np.nan
    i_half = np.searchsorted(m_cum, 0.5 * tot)
    return float(_SORTED_R[min(i_half, len(_SORTED_R) - 1)])


def misalignment_deg(pa1, pa2):
    """Inter-component misalignment angle in degrees, wrapped to [0, 90]."""
    dpa = abs(pa1 - pa2)
    if dpa > np.pi / 2:
        dpa = np.pi - dpa
    return float(np.degrees(dpa))


def _ols_fit(lx, ly):
    """Simple OLS: ly = slope * lx + intercept."""
    n = len(lx)
    if n < 3:
        return np.nan, np.nan
    mx, my = lx.mean(), ly.mean()
    ssxx = ((lx - mx) ** 2).sum()
    if ssxx <= 0:
        return np.nan, np.nan
    slope = ((lx - mx) * (ly - my)).sum() / ssxx
    intercept = my - slope * mx
    return float(slope), float(intercept)


def _fit_relation(x_arr, y_arr, min_val=1.0):
    """Fit log10(y) = alpha * log10(x) + beta via OLS.
    Returns (alpha, beta, sigma) matching scatter.ipynb fit_mean_relation.
    """
    mask = (x_arr > min_val) & (y_arr > min_val) & np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    lx = np.log10(x_arr[mask])
    ly = np.log10(y_arr[mask])
    alpha, beta = _ols_fit(lx, ly)
    if not np.isfinite(alpha):
        return np.nan, np.nan, np.nan
    sigma = float((ly - (alpha * lx + beta)).std())
    return alpha, beta, sigma


_PATCH_CEN = PATCH_PIX // 2   # = 64


def observables_from_phys(phys_3HW, r200_pix, q_DMO_const):
    """Compute all scatter.ipynb per-halo statistics from a physical (3,H,W) map.

    Args:
        phys_3HW   : ndarray (3, PATCH_PIX, PATCH_PIX) – [DM, Gas, Stars] in
                     physical units (M_sun/h per pixel).
        r200_pix   : R200c in pixels.
        q_DMO_const: DMO axis ratio for this halo (pre-computed); used for dq_dm.

    Returns:
        dict mapping each key in PER_HALO_KEYS to a float.
    """
    dm, gas, star = phys_3HW[0], phys_3HW[1], phys_3HW[2]
    r_aper  = max(min(r200_pix, PATCH_PIX / 2 - 2), 4.0)
    r_half  = max(0.5 * r_aper, 2.0)

    # ── aperture masses ─────────────────────────────────────────────────────
    M_dm   = aperture_sum(dm,   r_aper)
    M_gas  = aperture_sum(gas,  r_aper)
    M_star = aperture_sum(star, r_aper)
    M_bar  = M_gas + M_star
    M_tot  = M_dm + M_bar

    M_dm_h   = aperture_sum(dm,   r_half)
    M_gas_h  = aperture_sum(gas,  r_half)
    M_star_h = aperture_sum(star, r_half)

    # ── fractions ────────────────────────────────────────────────────────────
    f_bar  = M_bar  / M_tot if M_tot > 0 else np.nan
    f_gas  = M_gas  / M_tot if M_tot > 0 else np.nan
    f_star = M_star / M_tot if M_tot > 0 else np.nan

    # ── concentration M(<0.5R) / M(<R) ──────────────────────────────────────
    conc_dm   = M_dm_h   / M_dm   if M_dm   > 0 else np.nan
    conc_gas  = M_gas_h  / M_gas  if M_gas  > 0 else np.nan
    conc_star = M_star_h / M_star if M_star > 0 else np.nan

    # ── half-mass radius / R200c ─────────────────────────────────────────────
    rhalf_dm   = (half_mass_r_pix(dm)   / r200_pix) if r200_pix > 0 else np.nan
    rhalf_gas  = (half_mass_r_pix(gas)  / r200_pix) if r200_pix > 0 else np.nan
    rhalf_star = (half_mass_r_pix(star) / r200_pix) if r200_pix > 0 else np.nan

    # ── central 4-px / M(<R200c) ────────────────────────────────────────────
    cen = _PATCH_CEN
    cent4_dm   = float(np.maximum(dm,   0.0)[cen-1:cen+1, cen-1:cen+1].sum()) / M_dm   if M_dm   > 0 else np.nan
    cent4_gas  = float(np.maximum(gas,  0.0)[cen-1:cen+1, cen-1:cen+1].sum()) / M_gas  if M_gas  > 0 else np.nan
    cent4_star = float(np.maximum(star, 0.0)[cen-1:cen+1, cen-1:cen+1].sum()) / M_star if M_star > 0 else np.nan

    # ── peak pixel / M(<R200c) ───────────────────────────────────────────────
    peak_dm   = float(np.maximum(dm,   0.0).max()) / M_dm   if M_dm   > 0 else np.nan
    peak_gas  = float(np.maximum(gas,  0.0).max()) / M_gas  if M_gas  > 0 else np.nan
    peak_star = float(np.maximum(star, 0.0).max()) / M_star if M_star > 0 else np.nan

    # ── shape moments at R200c (returns q, pa, e1, e2) ─────────────────────
    q_dm_R,   pa_dm,   e1_dm,   e2_dm   = shape_moments_2d(dm,   r_aper)
    q_gas_R,  pa_gas,  e1_gas,  e2_gas  = shape_moments_2d(gas,  r_aper)
    q_star_R, pa_star, e1_star, e2_star = shape_moments_2d(star, r_aper)

    # ── shape moments at 0.5*R200c ──────────────────────────────────────────
    q_dm_h,   *_ = shape_moments_2d(dm,   r_half)
    q_gas_h,  *_ = shape_moments_2d(gas,  r_half)
    q_star_h, *_ = shape_moments_2d(star, r_half)

    # ── misalignment angles [deg] ────────────────────────────────────────────
    dtheta_dg = (misalignment_deg(pa_dm, pa_gas)
                 if (np.isfinite(pa_dm) and np.isfinite(pa_gas))   else np.nan)
    dtheta_ds = (misalignment_deg(pa_dm, pa_star)
                 if (np.isfinite(pa_dm) and np.isfinite(pa_star))  else np.nan)
    dtheta_gs = (misalignment_deg(pa_gas, pa_star)
                 if (np.isfinite(pa_gas) and np.isfinite(pa_star)) else np.nan)

    # ── baryonic spherization ────────────────────────────────────────────────
    dq_dm = (q_dm_R - q_DMO_const
             if (np.isfinite(q_dm_R) and np.isfinite(q_DMO_const)) else np.nan)

    return {
        "M_dm":   M_dm,   "M_gas":  M_gas,  "M_star": M_star, "M_bar":  M_bar,
        "f_bar":  f_bar,  "f_gas":  f_gas,  "f_star": f_star,
        "q_dm_R": q_dm_R, "q_gas_R": q_gas_R, "q_star_R": q_star_R,
        "q_dm_h": q_dm_h, "q_gas_h": q_gas_h, "q_star_h": q_star_h,
        "e1_dm":  e1_dm,  "e1_gas":  e1_gas,  "e1_star": e1_star,
        "e2_dm":  e2_dm,  "e2_gas":  e2_gas,  "e2_star": e2_star,
        "conc_dm":   conc_dm,   "conc_gas":   conc_gas,   "conc_star":   conc_star,
        "rhalf_dm":  rhalf_dm,  "rhalf_gas":  rhalf_gas,  "rhalf_star":  rhalf_star,
        "cent4_dm":  cent4_dm,  "cent4_gas":  cent4_gas,  "cent4_star":  cent4_star,
        "peak_dm":   peak_dm,   "peak_gas":   peak_gas,   "peak_star":   peak_star,
        "dtheta_dg": dtheta_dg, "dtheta_ds":  dtheta_ds,  "dtheta_gs":   dtheta_gs,
        "dq_dm":     dq_dm,
    }


def _sample_fixed_noise(model_fm, cond, ls, params, noise, n_steps):
    x = noise.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((cond.shape[0],), i * dt, device=cond.device)
        if ls is not None:
            inp = torch.cat([x, cond, ls], dim=1)
        else:
            inp = torch.cat([x, cond], dim=1)
        v = model_fm.model(inp, t, params)
        x = x + v * dt
    return x


# ---------------------------------------------------------------------------
# Loaders

def load_cv_halos(cv_root: Path, cube: bool = False):
    sim_dirs = sorted(d for d in cv_root.iterdir() if d.is_dir())
    cond_list, ls_list, mass_list, sid_list, params_list, radii_list = [], [], [], [], [], []
    cut_fname = "halo_cutouts_cube.npz" if cube else "halo_cutouts.npz"
    for d in sim_dirs:
        cat_path = d / "snap_090" / "mass_threshold_1p000e13" / "halo_catalog.npz"
        cut_path = d / "snap_090" / "mass_threshold_1p000e13" / cut_fname
        if not (cat_path.exists() and cut_path.exists()):
            continue
        cat = np.load(cat_path)
        cuts = np.load(cut_path)
        n = len(cat["halo_masses"] if cube else cat["masses"])
        cond_list.append(cuts["condition"])
        ls_list.append(cuts["large_scale"])
        mass_list.append(cat["halo_masses"] if cube else cat["masses"])
        sid_list.append(np.full(n, d.name))
        params_list.append(cat["params"])
        # Cube catalog: r200s in Mpc/h.  Legacy catalog: radii in kpc/h.
        if cube:
            radii_list.append(cat["r200s"].astype(np.float64) / MPC_PER_PIX)
        elif "radii" in cat.files:
            radii_list.append(cat["radii"].astype(np.float64) / 1000.0 / MPC_PER_PIX)
        else:
            radii_list.append(r200c_mpc_h(cat["halo_masses"] if cube else cat["masses"]) / MPC_PER_PIX)
    return {
        "cond_raw":  np.concatenate(cond_list).astype(np.float32),
        "ls_raw":    np.concatenate(ls_list).astype(np.float32),
        "masses":    np.concatenate(mass_list).astype(np.float64),
        "sim_id":    np.concatenate(sid_list),
        "params":    np.concatenate(params_list).astype(np.float32),
        "radii_pix": np.concatenate(radii_list).astype(np.float64),
    }


def normalize_inputs(cv, norm_stats):
    cond = (log_transform(cv["cond_raw"]) - norm_stats.cond_mean) / (norm_stats.cond_std + 1e-8)
    ls = (log_transform(cv["ls_raw"]) - norm_stats.ls_mean[:, None, None]) / (norm_stats.ls_std[:, None, None] + 1e-8)
    return cond, ls


def normalize_params_fid(p_raw, norm_stats):
    _p = np.where(norm_stats.param_log_flag == 1,
                  np.log10(np.maximum(p_raw, 1e-30)), p_raw)
    return ((_p - norm_stats.param_min) / (norm_stats.param_max - norm_stats.param_min + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Compute mode

def run_compute(args):
    # Line-buffered stdout so progress shows up immediately under SLURM
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compute] device = {device}")

    run_dir = Path(args.run_dir)
    ckpt = run_dir / "checkpoints" / "last.ckpt"
    norm_stats = NormStats.load(run_dir / "norm_stats.npz")

    print(f"[compute] loading model from {ckpt}")
    lit = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm
    model_fm.model.eval()
    print(f"[compute] out_channels = {model_fm.out_channels}")

    print(f"[compute] loading CV halos from {args.cv_root}  (cube={args.cube})")
    cv = load_cv_halos(Path(args.cv_root), cube=args.cube)
    # CAMELS-bug correction: the CV simulations were run with p14 = 0 even
    # though the parameter files (and halo_catalog['params']) record 2000.
    # The model was trained with the same overridden label, so we must use
    # 0 here to match what the network associates with the fiducial point.
    cv["params"][:, 14] = 0.0
    N_TOT = len(cv["masses"])
    print(f"[compute] N_TOT = {N_TOT} halos")

    cond_norm, ls_norm = normalize_inputs(cv, norm_stats)
    # Cube model has no large-scale conditioning channel — pass None so
    # _sample_fixed_noise skips the ls concatenation (5 channels, not 8).
    if args.cube:
        ls_norm = None
    p_norm_fid = normalize_params_fid(cv["params"][0], norm_stats)
    print(f"[compute] p_norm_fid[:6] = {p_norm_fid[:6].round(3)}")

    r200_pix = cv["radii_pix"]
    dmo_phys = cv["cond_raw"].astype(np.float64)

    # halo subset (random shuffle for reproducible MAX_HALOS)
    rng = np.random.default_rng(args.subset_seed)
    if args.max_halos is not None and args.max_halos < N_TOT:
        idx_use = np.sort(rng.choice(N_TOT, size=args.max_halos, replace=False))
    else:
        idx_use = np.arange(N_TOT)
    N_USE = len(idx_use)
    print(f"[compute] using {N_USE}/{N_TOT} halos")

    masses_use = cv["masses"][idx_use]   # M200c for scaling-relation x-axis

    # precompute q_DMO once (at R200c, using shape_moments_2d for consistency)
    print("[compute] precomputing q_DMO ...")
    q_DMO = np.full(N_TOT, np.nan)
    for i in range(N_TOT):
        r_aper = max(min(r200_pix[i], PATCH_PIX / 2 - 2), 4.0)
        q_DMO[i], *_ = shape_moments_2d(np.maximum(dmo_phys[i], 0.0), r_aper)

    # which params does this chunk handle?
    if args.params is not None:
        param_idxs = np.array([int(s) for s in args.params.split(",")], dtype=np.int64)
    else:
        # split [0..N_PARAMS) into n_chunks contiguous slices
        edges = np.linspace(0, N_PARAMS, args.n_chunks + 1).astype(int)
        lo, hi = edges[args.chunk_id], edges[args.chunk_id + 1]
        param_idxs = np.arange(lo, hi)
    print(f"[compute] this shard handles params {param_idxs.tolist()}")

    if len(param_idxs) == 0:
        print("[compute] empty shard — nothing to do")
        return

    # build per-halo fixed noise (deterministic, stable across shards)
    torch.manual_seed(args.noise_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.noise_seed)
    z_use = torch.randn(N_USE, model_fm.out_channels, PATCH_PIX, PATCH_PIX, device=device)

    # storage for per-halo Jacobian:  J[key] shape (N_USE, n_params_in_shard)
    J = {k: np.full((N_USE, len(param_idxs)), np.nan, dtype=np.float32)
         for k in PER_HALO_KEYS}
    # storage for population-level Jacobian:  Jpop[key] shape (n_params_in_shard,)
    Jpop = {k: np.full(len(param_idxs), np.nan, dtype=np.float32)
            for k in POP_KEYS}

    p_base = torch.tensor(p_norm_fid, device=device, dtype=torch.float32)

    cond_use = cond_norm[idx_use, np.newaxis]   # (N_USE, 1, H, W)
    ls_use   = ls_norm[idx_use] if ls_norm is not None else None  # None for cube model

    t_start = time.time()
    for jj, j in enumerate(param_idxs):
        p_plus  = p_base.clone(); p_plus[j]  += args.eps
        p_minus = p_base.clone(); p_minus[j] -= args.eps

        F_plus  = {k: np.full(N_USE, np.nan, dtype=np.float32) for k in PER_HALO_KEYS}
        F_minus = {k: np.full(N_USE, np.nan, dtype=np.float32) for k in PER_HALO_KEYS}

        with torch.no_grad():
            for start in range(0, N_USE, args.batch_size):
                stop = min(start + args.batch_size, N_USE)
                B = stop - start
                cb = torch.tensor(cond_use[start:stop], dtype=torch.float32, device=device)
                lb = (torch.tensor(ls_use[start:stop], dtype=torch.float32, device=device)
                      if ls_use is not None else None)
                zb = z_use[start:stop]
                p_plus_b  = p_plus.unsqueeze(0).expand(B, -1).contiguous()
                p_minus_b = p_minus.unsqueeze(0).expand(B, -1).contiguous()

                xp = _sample_fixed_noise(model_fm, cb, lb, p_plus_b,  zb, args.n_steps)
                xm = _sample_fixed_noise(model_fm, cb, lb, p_minus_b, zb, args.n_steps)

                phys_p = _denormalize_to_physical(xp.cpu().numpy(), norm_stats)
                phys_m = _denormalize_to_physical(xm.cpu().numpy(), norm_stats)

                for b, h_local in enumerate(range(start, stop)):
                    h = idx_use[h_local]
                    obs_p = observables_from_phys(phys_p[b], r200_pix[h], q_DMO[h])
                    obs_m = observables_from_phys(phys_m[b], r200_pix[h], q_DMO[h])
                    for k in PER_HALO_KEYS:
                        F_plus[k][h_local]  = obs_p[k]
                        F_minus[k][h_local] = obs_m[k]

        # ── per-halo Jacobians ─────────────────────────────────────────────
        for k in PER_HALO_KEYS:
            J[k][:, jj] = (F_plus[k] - F_minus[k]) / (2 * args.eps)

        # ── population-level Jacobians (fit scaling relations) ─────────────
        Mstar_p = F_plus["M_star"];  Mgas_p = F_plus["M_gas"]
        Mdm_p   = F_plus["M_dm"];    Mbar_p = F_plus["M_bar"]
        Mstar_m = F_minus["M_star"]; Mgas_m = F_minus["M_gas"]
        Mdm_m   = F_minus["M_dm"];   Mbar_m = F_minus["M_bar"]

        rel_defs = [
            ("MgMs", Mstar_p, Mgas_p,    Mstar_m, Mgas_m),    # Mgas–Mstar
            ("MdMs", Mstar_p, Mdm_p,     Mstar_m, Mdm_m),     # Mdm–Mstar
            ("SHMR", masses_use, Mstar_p, masses_use, Mstar_m), # SHMR
            ("GasFr", masses_use, Mgas_p, masses_use, Mgas_m), # GasFrac
            ("BarFr", masses_use, Mbar_p, masses_use, Mbar_m), # BaryonFrac
        ]
        for name, xp, yp, xm, ym in rel_defs:
            a_p, b_p, s_p = _fit_relation(xp, yp)
            a_m, b_m, s_m = _fit_relation(xm, ym)
            Jpop[f"alpha_{name}"][jj] = (a_p - a_m) / (2 * args.eps)
            Jpop[f"beta_{name}"][jj]  = (b_p - b_m) / (2 * args.eps)
            Jpop[f"sigma_{name}"][jj] = (s_p - s_m) / (2 * args.eps)

        elapsed = time.time() - t_start
        eta = elapsed / (jj + 1) * (len(param_idxs) - jj - 1)
        print(f"  param {j:2d} ({jj+1}/{len(param_idxs)}) done   "
              f"({elapsed/60:.1f} min elapsed; ETA {eta/60:.1f} min)", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = {f"J_{k}":    J[k]    for k in PER_HALO_KEYS}
    save.update({f"Jpop_{k}": Jpop[k] for k in POP_KEYS})
    save["param_idxs"] = param_idxs
    save["idx_use"]    = idx_use
    save["sim_id_use"] = cv["sim_id"][idx_use]
    save["masses_use"] = masses_use
    save["q_DMO_use"]  = q_DMO[idx_use]
    save["meta"] = np.array({
        "eps": args.eps, "n_steps": args.n_steps,
        "noise_seed": args.noise_seed, "subset_seed": args.subset_seed,
        "n_total": int(N_TOT), "n_use": int(N_USE),
        "model": str(run_dir),
    }, dtype=object)
    np.savez_compressed(out_path, **save)
    print(f"[compute] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Merge mode

def run_merge(args):
    files = sorted(glob(args.shard_glob))
    if not files:
        print(f"[merge] no files matching {args.shard_glob}")
        return
    print(f"[merge] found {len(files)} shards")

    # all shards must agree on idx_use, sim_id_use, masses_use, q_DMO_use
    first = np.load(files[0], allow_pickle=True)
    idx_use    = first["idx_use"]
    sim_id_use = first["sim_id_use"]
    masses_use = first["masses_use"]
    q_DMO_use  = first["q_DMO_use"]
    meta       = first["meta"].item()

    # per-halo Jacobians: (N_halos, N_PARAMS)
    J_full = {k: np.full((len(idx_use), N_PARAMS), np.nan, dtype=np.float32)
              for k in PER_HALO_KEYS}
    # population Jacobians: (N_PARAMS,)
    Jpop_full = {k: np.full(N_PARAMS, np.nan, dtype=np.float32)
                 for k in POP_KEYS}
    seen_params = np.zeros(N_PARAMS, dtype=bool)

    for f in files:
        z = np.load(f, allow_pickle=True)
        if not np.array_equal(z["idx_use"], idx_use):
            raise RuntimeError(f"{f}: idx_use mismatch — re-run with the same --subset_seed/--max_halos")
        if not np.array_equal(z["sim_id_use"], sim_id_use):
            raise RuntimeError(f"{f}: sim_id_use mismatch")
        for jj, j in enumerate(z["param_idxs"]):
            if seen_params[j]:
                print(f"[merge] WARN: param {j} already filled, overwriting from {f}")
            seen_params[j] = True
            for k in PER_HALO_KEYS:
                J_full[k][:, j] = z[f"J_{k}"][:, jj]
            for k in POP_KEYS:
                Jpop_full[k][j] = z[f"Jpop_{k}"][jj]

    missing = np.where(~seen_params)[0]
    if missing.size:
        print(f"[merge] WARNING: {len(missing)} params missing: {missing.tolist()}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = {f"J_{k}":    J_full[k]    for k in PER_HALO_KEYS}
    save.update({f"Jpop_{k}": Jpop_full[k] for k in POP_KEYS})
    save["idx_use"]    = idx_use
    save["sim_id_use"] = sim_id_use
    save["masses_use"] = masses_use
    save["q_DMO_use"]  = q_DMO_use
    save["meta"]       = np.array(meta, dtype=object)
    np.savez_compressed(out, **save)
    print(f"[merge] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)  "
          f"covering {seen_params.sum()}/{N_PARAMS} params")


# ---------------------------------------------------------------------------
# CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
    ap.add_argument("--cv_root", default="/mnt/home/mlee1/ceph/fm_testsuite/CV")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--params", type=str, default=None,
                    help="Comma-separated explicit param indices (overrides chunk slicing).")
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--max_halos", type=int, default=None)
    ap.add_argument("--noise_seed", type=int, default=42)
    ap.add_argument("--subset_seed", type=int, default=0)
    ap.add_argument("--cube", action="store_true",
                    help="Use cube-model artifacts (halo_cutouts_cube.npz, r200s key).")
    ap.add_argument("--merge", action="store_true",
                    help="Merge mode: read all shard files matching --shard_glob.")
    ap.add_argument("--shard_glob", type=str, default=None)
    args = ap.parse_args()

    if args.merge:
        if not args.shard_glob:
            ap.error("--merge requires --shard_glob")
        run_merge(args)
    else:
        run_compute(args)


if __name__ == "__main__":
    main()
