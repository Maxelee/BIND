"""scatter/robustness/check_seed.py
Robustness check 5: Seed robustness for 3 high-impact parameters.

Re-runs the scatter Jacobian for params 0, 2, 3 with a different noise seed
and shows that J_log_sigma shifts by less than its error bar.
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

TEST_PARAMS = [0, 2, 3]
SEEDS       = [42, 123]
EPS         = 0.05
K           = 10
N_STEPS     = 20
BATCH_SIZE  = 4
N_HALOS     = 100
OBS_NAMES   = ["M_gas", "M_star", "dq_DM"]


def compute_jac_entry(model_fm, norm_stats, p_norm_fid,
                      cond_use, ls_use, masses_use, r200_pix_use,
                      dmo_raw_use, omega_m_use,
                      j, seed, device):
    p_plus  = p_norm_fid.copy(); p_plus[j]  += EPS
    p_minus = p_norm_fid.copy(); p_minus[j] -= EPS
    rp = measure_scatter(model_fm, norm_stats, p_plus,
                         cond_use, ls_use, masses_use, r200_pix_use,
                         K=K, n_steps=N_STEPS, device=str(device),
                         batch_size=BATCH_SIZE,
                         dmo_raw=dmo_raw_use, omega_m=omega_m_use, seed=seed)
    rm = measure_scatter(model_fm, norm_stats, p_minus,
                         cond_use, ls_use, masses_use, r200_pix_use,
                         K=K, n_steps=N_STEPS, device=str(device),
                         batch_size=BATCH_SIZE,
                         dmo_raw=dmo_raw_use, omega_m=omega_m_use, seed=seed)
    N_obs = len(ALL_OBS_NAMES)
    J_mean = np.full(N_obs, np.nan)
    J_log_sigma = np.full(N_obs, np.nan)
    J_log_sigma_se = np.full(N_obs, np.nan)
    for o in range(N_obs):
        diff = rp["Y_bar"][:, o] - rm["Y_bar"][:, o]
        fm = np.isfinite(diff)
        if fm.sum() >= 2:
            J_mean[o] = np.nanmean(diff) / (2 * EPS)
        sp = rp["sigma_inter"][o]; sm = rm["sigma_inter"][o]
        if sp > 0 and sm > 0:
            J_log_sigma[o] = (np.log(sp) - np.log(sm)) / (2 * EPS)
            n_h = fm.sum()
            J_log_sigma_se[o] = (1.0 / np.sqrt(2 * max(n_h - 1, 1))) / (2 * EPS)
    return J_mean, J_log_sigma, J_log_sigma_se


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

    results = {}  # (j, seed) -> (J_mean, J_log_sigma, J_log_sigma_se)
    for j in TEST_PARAMS:
        for seed in SEEDS:
            print(f"  param {j} ({PARAM_NAMES[j]}), seed={seed}", flush=True)
            results[(j, seed)] = compute_jac_entry(
                model_fm, norm_stats, p_norm_fid,
                cond_4d[idx], ls_norm[idx], cv["masses"][idx], cv["radii_pix"][idx],
                cv["cond_raw"][idx], cv["params"][idx, 0].astype(np.float64),
                j=j, seed=seed, device=device,
            )

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save = {"obs_names": np.array(ALL_OBS_NAMES)}
    for j in TEST_PARAMS:
        for seed in SEEDS:
            save[f"J_mean_p{j}_s{seed}"]         = results[(j, seed)][0]
            save[f"J_log_sigma_p{j}_s{seed}"]    = results[(j, seed)][1]
            save[f"J_log_sigma_se_p{j}_s{seed}"] = results[(j, seed)][2]
    np.savez_compressed(OUT_DIR / "seed_robustness.npz", **save)

    # Plot
    fig, axes = plt.subplots(len(TEST_PARAMS), len(OBS_NAMES),
                             figsize=(5 * len(OBS_NAMES), 4 * len(TEST_PARAMS)))
    for pi, j in enumerate(TEST_PARAMS):
        for oi, obs_name in enumerate(OBS_NAMES):
            ax = axes[pi, oi]
            o = ALL_OBS_NAMES.index(obs_name)
            colors = {42: "steelblue", 123: "firebrick"}
            for seed in SEEDS:
                jls = results[(j, seed)][1][o]
                se  = results[(j, seed)][2][o]
                ax.errorbar([seed], [jls], yerr=[se if np.isfinite(se) else 0],
                            fmt="o", color=colors[seed], capsize=5, markersize=8,
                            label=f"seed={seed}")
            ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.4)
            ax.set_ylabel(r"$J_{\log\sigma}$")
            ax.set_xticks(SEEDS)
            ax.set_xlabel("Noise seed")
            ax.set_title(f"{PARAM_NAMES[j]} → {obs_name}")
            ax.legend(fontsize=7)
    fig.suptitle("Seed robustness: J_log_sigma", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "seed_robustness.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "seed_robustness.png", dpi=150, bbox_inches="tight")

    # Summary
    print("\nSUMMARY (seed robustness):")
    all_pass = True
    for j in TEST_PARAMS:
        for obs_name in OBS_NAMES:
            o = ALL_OBS_NAMES.index(obs_name)
            vals = [results[(j, s)][1][o] for s in SEEDS]
            ses  = [results[(j, s)][2][o] for s in SEEDS]
            if all(np.isfinite(v) for v in vals) and all(np.isfinite(s) for s in ses):
                shift = abs(vals[1] - vals[0])
                se_mean = np.mean(ses)
                flag = "PASS" if shift < se_mean else "FAIL"
                if flag == "FAIL":
                    all_pass = False
                print(f"  {PARAM_NAMES[j]:15s} → {obs_name:12s}: shift={shift:.4f} SE={se_mean:.4f} ({flag})")
    if all_pass:
        print("PASS: All J_log_sigma shifts < 1 error bar")
    else:
        print("WARNING: Some shifts exceed 1 error bar (may need more K or N_h)")


if __name__ == "__main__":
    main()
