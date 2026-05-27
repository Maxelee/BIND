"""scatter/robustness/check_los_contamination.py
Robustness check 3: Re-include LOS-contaminated observables and compare.

Shows that f_b, R_cl/R_200, Sigma_gas_c have similar mean responses but
potentially unreliable scatter responses compared to the headline set.
Uses the existing Jacobian data (requires J_mean_and_scatter.npz).
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

from scatter.measure_scatter import ALL_OBS_NAMES, HEADLINE_OBS_NAMES
from scatter.scatter_jacobian import PARAM_NAMES

JAC_PATH = Path("/mnt/home/mlee1/vdm_bind2/scatter/J_mean_and_scatter.npz")
OUT_DIR  = Path(__file__).resolve().parent

# LOS-contaminated observables (dropped from headline)
LOS_OBS = ["f_b", "f_b_norm", "Rc_over_R200", "Sigma_gas_c"]


def main():
    if not JAC_PATH.exists():
        print(f"Jacobian not found at {JAC_PATH}; run scatter_jacobian.py first")
        return

    d = np.load(JAC_PATH, allow_pickle=True)
    J_mean      = d["J_mean"]        # (N_obs, N_params)
    J_log_sigma = d["J_log_sigma"]
    obs_names   = list(d["obs_names"])

    # Key headline observables for comparison
    compare_obs = ["M_gas", "M_star"] + LOS_OBS
    compare_idxs = [obs_names.index(o) for o in compare_obs if o in obs_names]
    compare_names = [obs_names[i] for i in compare_idxs]

    headline_idxs = [obs_names.index(o) for o in ["M_gas", "M_star"] if o in obs_names]
    los_idxs      = [obs_names.index(o) for o in LOS_OBS if o in obs_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel 1: J_mean scatter plot (all obs)
    ax = axes[0]
    for oi in headline_idxs:
        ax.scatter(J_mean[oi], J_log_sigma[oi], alpha=0.6, s=30,
                   color="steelblue", label=obs_names[oi] if oi == headline_idxs[0] else "")
    for oi in los_idxs:
        ax.scatter(J_mean[oi], J_log_sigma[oi], alpha=0.6, s=30,
                   marker="x", color="firebrick", linewidths=1.5,
                   label=obs_names[oi] if oi == los_idxs[0] else "")
    ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", ls="--", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$J_{\rm mean}$ (over all 35 params)")
    ax.set_ylabel(r"$J_{\log\sigma}$")
    ax.set_title("All observables (blue=headline, red×=LOS-contaminated)")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='steelblue', ls='', label='Headline'),
        Line2D([0], [0], marker='x', color='firebrick', ls='', mew=1.5, label='LOS-contaminated'),
    ]
    ax.legend(handles=handles, fontsize=8)

    # Panel 2: Top-5 param responses for M_gas vs Sigma_gas_c (LOS)
    ax2 = axes[1]
    if "M_gas" in obs_names and "Sigma_gas_c" in obs_names:
        o_mg  = obs_names.index("M_gas")
        o_sgc = obs_names.index("Sigma_gas_c")
        top5 = np.argsort(np.abs(J_mean[o_mg]))[::-1][:10]
        ax2.scatter(J_mean[o_mg, top5], J_log_sigma[o_mg, top5],
                    color="steelblue", s=80, label="M_gas")
        ax2.scatter(J_mean[o_sgc, top5], J_log_sigma[o_sgc, top5],
                    color="firebrick", s=80, marker="x", linewidths=1.5, label="Sigma_gas_c")
        for j in top5:
            ax2.annotate(PARAM_NAMES[j], (J_mean[o_mg, j], J_log_sigma[o_mg, j]),
                         fontsize=6, xytext=(2, 2), textcoords="offset points", color="steelblue")
        ax2.axhline(0, color="k", ls="--", lw=0.5, alpha=0.4)
        ax2.axvline(0, color="k", ls="--", lw=0.5, alpha=0.4)
        ax2.set_xlabel(r"$J_{\rm mean}$")
        ax2.set_ylabel(r"$J_{\log\sigma}$")
        ax2.set_title("M_gas vs Sigma_gas_c: top-10 by |J_mean(M_gas)|")
        ax2.legend(fontsize=8)

    fig.suptitle("LOS contamination robustness check", fontsize=12)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "los_contamination.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "los_contamination.png", dpi=150, bbox_inches="tight")
    print("Done. Saved scatter/robustness/los_contamination.{pdf,png}")

    # One-line summary
    headline_var = np.nanvar(J_log_sigma[headline_idxs, :], axis=1).mean()
    los_var      = np.nanvar(J_log_sigma[los_idxs, :], axis=1).mean()
    print(f"SUMMARY: mean var(J_log_sigma) headline={headline_var:.4f}  LOS={los_var:.4f}")
    if los_var > 2 * headline_var:
        print("LOS observables show elevated scatter variance — supports dropping them.")
    else:
        print("LOS observables show comparable variance — appendix figure supports this.")


if __name__ == "__main__":
    main()
