"""scatter/robustness/check_k_budget.py
Robustness check 1: Does the scatter Jacobian stabilize at K=10?

Computes J_log_sigma for 3 key parameters at K ∈ {5, 10, 20} and overlays.
Saves results to scatter/robustness/k_budget.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import NormStats, log_transform
from fd_jacobian_cv import load_cv_halos, normalize_inputs, normalize_params_fid
from scatter.measure_scatter import measure_scatter, ALL_OBS_NAMES, LOG_MASK
from scatter.scatter_jacobian import PARAM_NAMES
from train import FlowMatchingLit
import torch

RUN_DIR = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
CV_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
OUT_DIR = Path(__file__).resolve().parent

# Key params to test: Omega_m, A_SN1, A_AGN1 (indices 0, 2, 3)
TEST_PARAMS = [0, 2, 3]
K_VALUES    = [5, 10, 20]
EPS         = 0.05
N_HALOS     = 100
N_STEPS     = 20
BATCH_SIZE  = 4


def compute_jac_entry(model_fm, norm_stats, p_norm_fid,
                      cond_use, ls_use, masses_use, r200_pix_use,
                      dmo_raw_use, omega_m_use,
                      j: int, K: int, device: str):
    """Compute J_mean and J_log_sigma for one parameter at one K value."""
    p_plus  = p_norm_fid.copy(); p_plus[j]  += EPS
    p_minus = p_norm_fid.copy(); p_minus[j] -= EPS

    r_plus  = measure_scatter(model_fm, norm_stats, p_plus,  cond_use, ls_use,
                              masses_use, r200_pix_use, K=K, n_steps=N_STEPS,
                              device=device, batch_size=BATCH_SIZE,
                              dmo_raw=dmo_raw_use, omega_m=omega_m_use, seed=42)
    r_minus = measure_scatter(model_fm, norm_stats, p_minus, cond_use, ls_use,
                              masses_use, r200_pix_use, K=K, n_steps=N_STEPS,
                              device=device, batch_size=BATCH_SIZE,
                              dmo_raw=dmo_raw_use, omega_m=omega_m_use, seed=42)

    N_obs = len(ALL_OBS_NAMES)
    J_mean = np.full(N_obs, np.nan)
    J_log_sigma = np.full(N_obs, np.nan)
    for o in range(N_obs):
        diff = r_plus["Y_bar"][:, o] - r_minus["Y_bar"][:, o]
        fm = np.isfinite(diff)
        if fm.sum() >= 2:
            J_mean[o] = np.nanmean(diff) / (2 * EPS)
        si_p = r_plus["sigma_inter"][o]
        si_m = r_minus["sigma_inter"][o]
        if si_p > 0 and si_m > 0:
            J_log_sigma[o] = (np.log(si_p) - np.log(si_m)) / (2 * EPS)
    return J_mean, J_log_sigma


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_stats = NormStats.load(RUN_DIR / "norm_stats.npz")
    lit = FlowMatchingLit.load_from_checkpoint(
        str(RUN_DIR / "checkpoints" / "last.ckpt"), map_location=device)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm
    model_fm.model.eval()

    cv = load_cv_halos(CV_ROOT)
    cv["params"][:, 14] = 0.0
    cond_norm, ls_norm = normalize_inputs(cv, norm_stats)
    cond_4d = cond_norm[:, np.newaxis]
    p_norm_fid = normalize_params_fid(cv["params"][0], norm_stats)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(cv["masses"]), size=N_HALOS, replace=False))

    results = {}  # (j, K) -> (J_mean, J_log_sigma)
    for j in TEST_PARAMS:
        for K in K_VALUES:
            print(f"  param {j} ({PARAM_NAMES[j]}), K={K}", flush=True)
            J_mean, J_log_sigma = compute_jac_entry(
                model_fm, norm_stats, p_norm_fid,
                cond_4d[idx], ls_norm[idx], cv["masses"][idx],
                cv["radii_pix"][idx], cv["cond_raw"][idx],
                cv["params"][idx, 0].astype(np.float64),
                j=j, K=K, device=str(device),
            )
            results[(j, K)] = (J_mean, J_log_sigma)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "k_budget.npz",
        params=np.array(TEST_PARAMS),
        K_values=np.array(K_VALUES),
        obs_names=np.array(ALL_OBS_NAMES),
        **{f"J_mean_p{j}_K{K}": results[(j, K)][0] for j in TEST_PARAMS for K in K_VALUES},
        **{f"J_log_sigma_p{j}_K{K}": results[(j, K)][1] for j in TEST_PARAMS for K in K_VALUES},
    )

    # Plot
    obs_plot = ["M_gas", "M_star", "dq_DM"]
    n_obs_plot = len(obs_plot)
    fig, axes = plt.subplots(len(TEST_PARAMS), n_obs_plot,
                             figsize=(5 * n_obs_plot, 4 * len(TEST_PARAMS)))
    colors_k = {5: "steelblue", 10: "firebrick", 20: "forestgreen"}
    for pi, j in enumerate(TEST_PARAMS):
        for oi, obs_name in enumerate(obs_plot):
            ax = axes[pi, oi]
            o_idx = ALL_OBS_NAMES.index(obs_name)
            for K in K_VALUES:
                jls = results[(j, K)][1][o_idx]
                jm  = results[(j, K)][0][o_idx]
                ax.scatter([K], [jls], color=colors_k[K], s=80, zorder=5, label=f"K={K}")
            ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.4)
            ax.set_xlabel("K")
            ax.set_ylabel(r"$J_{\log\sigma}$")
            ax.set_title(f"param={PARAM_NAMES[j]}, obs={obs_name}")
            ax.legend(fontsize=7)
    fig.suptitle("K convergence check", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k_budget.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "k_budget.png", dpi=150, bbox_inches="tight")
    print("Done. Saved scatter/robustness/k_budget.{npz,pdf,png}")


if __name__ == "__main__":
    main()
