"""scatter/robustness/check_mass_bins.py
Robustness check 2: Repeat scatter Jacobian in 3 separate mass bins.

Computes J_mean and J_log_sigma for all 35 params separately in
3 mass bins and overlays on a 3-panel fig2.
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
from fd_jacobian_cv import load_cv_halos, normalize_inputs, normalize_params_fid, N_PARAMS
from scatter.measure_scatter import measure_scatter, ALL_OBS_NAMES, LOG_MASK
from scatter.scatter_jacobian import PARAM_NAMES
from train import FlowMatchingLit
import torch

RUN_DIR = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
CV_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
OUT_DIR = Path(__file__).resolve().parent

EPS        = 0.05
K          = 10
N_STEPS    = 20
BATCH_SIZE = 4
MIN_HALOS_PER_BIN = 40  # skip params if too few halos in bin

MASS_BINS = [
    (1e13,   10**13.5),
    (10**13.5, 1e14),
    (1e14,   10**14.8),
]
MASS_BIN_LABELS = ["13.0-13.5", "13.5-14.0", "14.0-14.8"]

# Only run 6 key params for speed (full 35 would be ~18 hr per bin)
KEY_PARAMS = [0, 1, 2, 3, 4, 5]  # Omega_m, sigma8, A_SN1, A_AGN1, A_SN2, A_AGN2


def compute_jac_for_subset(
    model_fm, norm_stats, p_norm_fid, device,
    cond_subset, ls_subset, masses_subset, r200_pix_subset,
    dmo_raw_subset, omega_m_subset, param_idxs,
):
    """Compute J for all requested param_idxs over a halo subset."""
    N_obs = len(ALL_OBS_NAMES)
    J_mean_arr      = np.full((N_obs, len(param_idxs)), np.nan)
    J_log_sigma_arr = np.full((N_obs, len(param_idxs)), np.nan)
    for jj, j in enumerate(param_idxs):
        p_plus  = p_norm_fid.copy(); p_plus[j]  += EPS
        p_minus = p_norm_fid.copy(); p_minus[j] -= EPS
        r_p = measure_scatter(model_fm, norm_stats, p_plus,
                              cond_subset, ls_subset, masses_subset, r200_pix_subset,
                              K=K, n_steps=N_STEPS, device=str(device),
                              batch_size=BATCH_SIZE,
                              dmo_raw=dmo_raw_subset, omega_m=omega_m_subset, seed=42)
        r_m = measure_scatter(model_fm, norm_stats, p_minus,
                              cond_subset, ls_subset, masses_subset, r200_pix_subset,
                              K=K, n_steps=N_STEPS, device=str(device),
                              batch_size=BATCH_SIZE,
                              dmo_raw=dmo_raw_subset, omega_m=omega_m_subset, seed=42)
        for o in range(N_obs):
            diff = r_p["Y_bar"][:, o] - r_m["Y_bar"][:, o]
            fm = np.isfinite(diff)
            if fm.sum() >= 2:
                J_mean_arr[o, jj] = np.nanmean(diff) / (2 * EPS)
            si_p = r_p["sigma_inter"][o]
            si_m = r_m["sigma_inter"][o]
            if si_p > 0 and si_m > 0:
                J_log_sigma_arr[o, jj] = (np.log(si_p) - np.log(si_m)) / (2 * EPS)
        print(f"    param {j} ({PARAM_NAMES[j]}) done", flush=True)
    return J_mean_arr, J_log_sigma_arr


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
    masses = cv["masses"]

    results_per_bin = {}
    for bi, ((lo, hi), label) in enumerate(zip(MASS_BINS, MASS_BIN_LABELS)):
        mask = (masses >= lo) & (masses < hi)
        n_bin = mask.sum()
        print(f"\nMass bin {label}: {n_bin} halos", flush=True)
        if n_bin < MIN_HALOS_PER_BIN:
            print(f"  Too few halos ({n_bin} < {MIN_HALOS_PER_BIN}), skipping")
            results_per_bin[label] = None
            continue

        idx_bin = np.where(mask)[0]
        J_mean, J_log_sigma = compute_jac_for_subset(
            model_fm, norm_stats, p_norm_fid, device,
            cond_4d[idx_bin], ls_norm[idx_bin], masses[idx_bin],
            cv["radii_pix"][idx_bin], cv["cond_raw"][idx_bin],
            cv["params"][idx_bin, 0].astype(np.float64),
            KEY_PARAMS,
        )
        results_per_bin[label] = (J_mean, J_log_sigma)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save = {"obs_names": np.array(ALL_OBS_NAMES), "key_params": np.array(KEY_PARAMS)}
    for label, res in results_per_bin.items():
        if res is not None:
            save[f"J_mean_{label}"]      = res[0]
            save[f"J_log_sigma_{label}"] = res[1]
    np.savez_compressed(OUT_DIR / "mass_bins.npz", **save)

    # Plot: 3-panel version of fig2 for M_gas
    obs_name = "M_gas"
    o = ALL_OBS_NAMES.index(obs_name)
    labels_avail = [l for l, r in results_per_bin.items() if r is not None]
    if not labels_avail:
        print("No bins with enough halos")
        return

    fig, axes = plt.subplots(1, len(labels_avail), figsize=(5 * len(labels_avail), 5))
    if len(labels_avail) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels_avail):
        J_mean, J_log_sigma = results_per_bin[label]
        for jj, j in enumerate(KEY_PARAMS):
            ax.scatter([J_mean[o, jj]], [J_log_sigma[o, jj]], s=60,
                       label=PARAM_NAMES[j], zorder=5)
            ax.annotate(PARAM_NAMES[j], (J_mean[o, jj], J_log_sigma[o, jj]),
                        fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", ls="--", lw=0.5, alpha=0.4)
        ax.set_xlabel(r"$J_{\rm mean}$")
        ax.set_ylabel(r"$J_{\log\sigma}$")
        ax.set_title(f"M_gas — bin {label}")
    fig.suptitle("Mass-bin stability of scatter Jacobian", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mass_bins.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "mass_bins.png", dpi=150, bbox_inches="tight")
    print("Done.")


if __name__ == "__main__":
    main()
