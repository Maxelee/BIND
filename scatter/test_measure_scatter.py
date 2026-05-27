"""scatter/test_measure_scatter.py
Phase 1 gating check: run measure_scatter on fiducial theta, 20 halos, K=4.
Saves output to scatter/test_fiducial_K4.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import NormStats, log_transform
from fd_jacobian_cv import (
    load_cv_halos, normalize_inputs, normalize_params_fid,
    r200c_mpc_h, MPC_PER_PIX, OMEGA_B_FIXED,
)
from scatter.measure_scatter import measure_scatter, ALL_OBS_NAMES, LOG_MASK
from train import FlowMatchingLit

RUN_DIR   = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
CV_ROOT   = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
OUT_PATH  = Path(__file__).resolve().parent / "test_fiducial_K4.npz"
N_HALOS   = 20
K         = 4
N_STEPS   = 20
BATCH_SZ  = 8


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # Load model
    ckpt = RUN_DIR / "checkpoints" / "last.ckpt"
    norm_stats = NormStats.load(RUN_DIR / "norm_stats.npz")
    lit = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm
    model_fm.model.eval()
    print(f"out_channels = {model_fm.out_channels}")

    # Load CV halos
    cv = load_cv_halos(CV_ROOT)
    cv["params"][:, 14] = 0.0  # CAMELS bug fix
    N_TOT = len(cv["masses"])
    print(f"Total halos: {N_TOT}")

    # Normalize inputs
    cond_norm, ls_norm = normalize_inputs(cv, norm_stats)
    # Add channel dim: (N, H, W) → (N, 1, H, W)
    cond_norm_4d = cond_norm[:, np.newaxis]

    # Fiducial normalized params (use first halo's param set — all CV are fiducial)
    p_norm_fid = normalize_params_fid(cv["params"][0], norm_stats)
    print(f"theta_norm[:6] = {p_norm_fid[:6].round(3)}")

    r200_pix = cv["radii_pix"]
    omega_m  = cv["params"][:, 0].astype(np.float64)

    # Subset to N_HALOS
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(N_TOT, size=N_HALOS, replace=False))

    result = measure_scatter(
        model_fm      = model_fm,
        norm_stats    = norm_stats,
        theta_norm    = p_norm_fid,
        dmo_conds     = cond_norm_4d[idx],
        ls_conds      = ls_norm[idx],
        masses        = cv["masses"][idx],
        r200_pix      = r200_pix[idx],
        K             = K,
        n_steps       = N_STEPS,
        device        = str(device),
        batch_size    = BATCH_SZ,
        dmo_raw       = cv["cond_raw"][idx],
        omega_m       = omega_m[idx],
        seed          = 42,
    )

    print("\n=== Variance decomposition (fiducial, K=4, N_h=20) ===")
    print(f"{'Observable':<22}  {'sigma_inter':>12}  {'sigma_intra':>12}  {'sigma_total':>12}  {'identity_err%':>14}")
    for o, name in enumerate(ALL_OBS_NAMES):
        si  = result["sigma_inter"][o]
        sa  = result["sigma_intra"][o]
        st  = result["sigma_total"][o]
        if np.isfinite(si) and np.isfinite(sa) and np.isfinite(st) and st > 0:
            identity_err = 100 * abs(st**2 - si**2 - sa**2) / (st**2 + 1e-30)
        else:
            identity_err = np.nan
        flag = LOG_MASK[o]
        space = "log" if flag else "lin"
        print(f"  {name:<20} ({space})  {si:12.4f}  {sa:12.4f}  {st:12.4f}  {identity_err:14.1f}%")

    # Gating check: identity must hold within 5% for all finite obs
    errs = []
    for o in range(len(ALL_OBS_NAMES)):
        si, sa, st = result["sigma_inter"][o], result["sigma_intra"][o], result["sigma_total"][o]
        if np.isfinite(si) and np.isfinite(sa) and np.isfinite(st) and st > 0:
            errs.append(100 * abs(st**2 - si**2 - sa**2) / st**2)

    max_err = max(errs) if errs else np.nan
    print(f"\nMax identity error: {max_err:.2f}%  (threshold: 5%)")
    if max_err > 5.0:
        print("WARNING: Identity check FAILED — variance decomposition inconsistent!")
    else:
        print("PASS: Identity check OK")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        obs_tensor  = result["obs_tensor"],
        obs_names   = np.array(result["obs_names"]),
        log_mask    = result["log_mask"],
        masses      = result["masses"],
        sigma_inter = result["sigma_inter"],
        sigma_intra = result["sigma_intra"],
        sigma_total = result["sigma_total"],
        Y_bar       = result["Y_bar"],
        idx_use     = idx,
    )
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
