"""scatter/robustness/check_stepsize.py
Robustness check 4: Step-size robustness for 3 high-impact parameters.

Computes J_log_sigma at eps ∈ {0.025, 0.05, 0.10} for params 0, 2, 3.
Verifies that the result is linear in eps (small-perturbation regime).
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

from data import NormStats
from fd_jacobian_cv import load_cv_halos, normalize_inputs, normalize_params_fid
from scatter.measure_scatter import measure_scatter, ALL_OBS_NAMES, LOG_MASK
from scatter.scatter_jacobian import PARAM_NAMES
from train import FlowMatchingLit
import torch

RUN_DIR = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
CV_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
OUT_DIR = Path(__file__).resolve().parent

TEST_PARAMS = [0, 2, 3]  # Omega_m, A_SN1, A_AGN1
EPS_VALUES  = [0.025, 0.05, 0.10]
K           = 10
N_STEPS     = 20
BATCH_SIZE  = 4
N_HALOS     = 100
OBS_NAMES   = ["M_gas", "M_star", "dq_DM"]


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

    results = {}  # (j, eps) -> (J_mean, J_log_sigma)
    for j in TEST_PARAMS:
        for eps in EPS_VALUES:
            print(f"  param {j} ({PARAM_NAMES[j]}), eps={eps}", flush=True)
            p_plus  = p_norm_fid.copy(); p_plus[j]  += eps
            p_minus = p_norm_fid.copy(); p_minus[j] -= eps
            rp = measure_scatter(model_fm, norm_stats, p_plus,
                                 cond_4d[idx], ls_norm[idx], cv["masses"][idx],
                                 cv["radii_pix"][idx], K=K, n_steps=N_STEPS,
                                 device=str(device), batch_size=BATCH_SIZE,
                                 dmo_raw=cv["cond_raw"][idx],
                                 omega_m=cv["params"][idx, 0].astype(np.float64), seed=42)
            rm = measure_scatter(model_fm, norm_stats, p_minus,
                                 cond_4d[idx], ls_norm[idx], cv["masses"][idx],
                                 cv["radii_pix"][idx], K=K, n_steps=N_STEPS,
                                 device=str(device), batch_size=BATCH_SIZE,
                                 dmo_raw=cv["cond_raw"][idx],
                                 omega_m=cv["params"][idx, 0].astype(np.float64), seed=42)
            N_obs = len(ALL_OBS_NAMES)
            J_mean_arr = np.full(N_obs, np.nan)
            J_log_sig  = np.full(N_obs, np.nan)
            for o in range(N_obs):
                diff = rp["Y_bar"][:, o] - rm["Y_bar"][:, o]
                fm = np.isfinite(diff)
                if fm.sum() >= 2:
                    J_mean_arr[o] = np.nanmean(diff) / (2 * eps)
                sp = rp["sigma_inter"][o]; sm = rm["sigma_inter"][o]
                if sp > 0 and sm > 0:
                    J_log_sig[o] = (np.log(sp) - np.log(sm)) / (2 * eps)
            results[(j, eps)] = (J_mean_arr, J_log_sig)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save = {"obs_names": np.array(ALL_OBS_NAMES)}
    for j in TEST_PARAMS:
        for eps in EPS_VALUES:
            eps_str = str(eps).replace(".", "p")
            save[f"J_mean_p{j}_eps{eps_str}"]      = results[(j, eps)][0]
            save[f"J_log_sigma_p{j}_eps{eps_str}"] = results[(j, eps)][1]
    np.savez_compressed(OUT_DIR / "stepsize.npz", **save)

    # Plot
    fig, axes = plt.subplots(len(TEST_PARAMS), len(OBS_NAMES),
                             figsize=(5 * len(OBS_NAMES), 4 * len(TEST_PARAMS)))
    for pi, j in enumerate(TEST_PARAMS):
        for oi, obs_name in enumerate(OBS_NAMES):
            ax = axes[pi, oi]
            o = ALL_OBS_NAMES.index(obs_name)
            jls_vals = [results[(j, eps)][1][o] for eps in EPS_VALUES]
            jm_vals  = [results[(j, eps)][0][o] for eps in EPS_VALUES]
            ax.plot(EPS_VALUES, jls_vals, "o-", color="steelblue", label=r"$J_{\log\sigma}$")
            ax.plot(EPS_VALUES, jm_vals,  "s--", color="firebrick",  label=r"$J_{\rm mean}$")
            ax.axhline(0, color="k", ls=":", lw=0.5, alpha=0.4)
            ax.set_xlabel(r"$\varepsilon$")
            ax.set_ylabel("Jacobian value")
            ax.set_title(f"{PARAM_NAMES[j]} → {obs_name}")
            ax.legend(fontsize=7)
    fig.suptitle("Step-size robustness", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stepsize.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "stepsize.png", dpi=150, bbox_inches="tight")

    # Summary
    print("\nSUMMARY (step-size linearity):")
    all_linear = True
    for j in TEST_PARAMS:
        for obs_name in OBS_NAMES:
            o = ALL_OBS_NAMES.index(obs_name)
            vals = np.array([results[(j, eps)][1][o] for eps in EPS_VALUES])
            if np.all(np.isfinite(vals)):
                spread = np.std(vals)
                flag = "linear" if spread < 0.5 * abs(np.mean(vals)) + 1e-6 else "NONLINEAR"
                if flag == "NONLINEAR":
                    all_linear = False
                print(f"  {PARAM_NAMES[j]:15s} → {obs_name:12s}: spread={spread:.4f} ({flag})")
    if all_linear:
        print("PASS: All checked params/obs are linear in eps")
    else:
        print("WARNING: Some params/obs show nonlinearity in eps")


if __name__ == "__main__":
    main()
