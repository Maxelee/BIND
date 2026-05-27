"""fig5_mean_vs_scatter_orthogonality.py — Figure 5 of BIND convincingness brief.

For each of 16 observables, compute the angle between the mean Jacobian vector
and the scatter Jacobian vector in 35-parameter space:
  θ_ab = arccos(Ĵ_mean · Ĵ_scatter)  (in degrees)

Two versions per observable:
  (a) All 35 parameters
  (b) Only parameters with R_j < 0.3 for that observable ("decontaminated")

Output:
  figures/scatter_diagnostics/fig_mean_vs_scatter_orthogonality.pdf / .png
  outputs/scatter_diagnostics/fig_mean_vs_scatter_orthogonality.json
"""
from __future__ import annotations

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).parent.parent
JMEAN_FILE = BASE_DIR / "scatter/J_mean_and_scatter.npz"
PHASE1_FILE = BASE_DIR / "outputs/scatter_diagnostics/phase1_intra_jacobian.npz"
FIG_DIR   = BASE_DIR / "figures/scatter_diagnostics"
OUT_DIR   = BASE_DIR / "outputs/scatter_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RJ_CLEAN_THRESHOLD = 0.3  # R_j < this → "clean" parameter


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two vectors; NaN if either is zero."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return np.nan
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    jm = np.load(JMEAN_FILE, allow_pickle=True)
    obs_names   = jm["obs_names"].tolist()
    J_mean      = jm["J_mean"]         # (16, 35) — mean Jacobian
    J_log_sigma = jm["J_log_sigma"]    # (16, 35) — scatter Jacobian

    phase1 = np.load(PHASE1_FILE, allow_pickle=True)
    param_names = phase1["param_names"].tolist()
    R_j         = phase1["contamination_ratio"]   # (16, 35) raw R_j

    N_obs  = len(obs_names)
    N_par  = J_mean.shape[1]

    # ── Compute angles ────────────────────────────────────────────────────────
    angles_all         = np.full(N_obs, np.nan)
    angles_decontam    = np.full(N_obs, np.nan)
    n_clean_params     = np.zeros(N_obs, dtype=int)

    for i in range(N_obs):
        jm_i  = J_mean[i]
        jls_i = J_log_sigma[i]

        # (a) All 35 params
        angles_all[i] = angle_deg(jm_i, jls_i)

        # (b) Only "clean" params: R_j < threshold
        clean = R_j[i] < RJ_CLEAN_THRESHOLD  # (35,)
        n_clean_params[i] = int(clean.sum())
        if clean.sum() >= 2:
            angles_decontam[i] = angle_deg(jm_i[clean], jls_i[clean])

    # Sort by decontaminated angle (NaN last)
    sort_key = np.where(np.isfinite(angles_decontam), angles_decontam, -1)
    order = np.argsort(-sort_key)  # descending: closest to 90° first is highest

    print("=== Mean vs Scatter orthogonality ===")
    print(f"{'obs':20s}  angle_all  angle_decontam  n_clean_params")
    for i in order:
        print(f"  {obs_names[i]:20s}: {angles_all[i]:6.1f}°  "
              f"{angles_decontam[i]:6.1f}°  ({n_clean_params[i]} params)")

    near_90_any = any(
        np.isfinite(angles_decontam[i]) and abs(angles_decontam[i] - 90) < 20
        for i in range(N_obs)
    )
    print(f"\nPass condition (at least one decontam angle near 90°): {near_90_any}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    y_pos = np.arange(N_obs)
    obs_sorted = [obs_names[i] for i in order]
    ang_all_sorted    = angles_all[order]
    ang_decon_sorted  = angles_decontam[order]
    n_clean_sorted    = n_clean_params[order]

    # Panel (a): All 35 params
    ax = axes[0]
    colors_all = ["steelblue" if abs(a - 90) < 20 else "gray" for a in ang_all_sorted]
    bars = ax.barh(y_pos, ang_all_sorted, color=colors_all, alpha=0.8, height=0.7)
    ax.axvline(90, color="red", linewidth=1.5, linestyle="--", label="90° (orthogonal)")
    ax.axvline(60, color="orange", linewidth=0.8, linestyle=":", label="±30° window")
    ax.axvline(120, color="orange", linewidth=0.8, linestyle=":")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(obs_sorted, fontsize=9)
    ax.set_xlabel("Angle (degrees)", fontsize=10)
    ax.set_title("(a) All 35 parameters", fontsize=11)
    ax.set_xlim(0, 185)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.02, "Blue = within 20° of 90°", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="steelblue")

    # Panel (b): Decontaminated (R_j < 0.3)
    ax = axes[1]
    colors_decon = []
    for i, a in enumerate(ang_decon_sorted):
        if np.isnan(a):
            colors_decon.append("lightgray")
        elif abs(a - 90) < 20:
            colors_decon.append("green")
        elif abs(a - 90) < 40:
            colors_decon.append("orange")
        else:
            colors_decon.append("salmon")

    valid = np.isfinite(ang_decon_sorted)
    ax.barh(y_pos[valid], ang_decon_sorted[valid], color=[colors_decon[i] for i in range(N_obs) if valid[i]],
            alpha=0.85, height=0.7)
    # NaN bars
    ax.barh(y_pos[~valid], np.zeros(np.sum(~valid)) + 5, color="lightgray",
            alpha=0.5, height=0.7, label=f"< 2 clean params")
    ax.axvline(90, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(70, color="orange", linewidth=0.8, linestyle=":")
    ax.axvline(110, color="orange", linewidth=0.8, linestyle=":")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(obs_sorted, fontsize=9)
    ax.set_xlabel("Angle (degrees)", fontsize=10)
    ax.set_title(f"(b) Decontaminated ($R_j < {RJ_CLEAN_THRESHOLD}$)", fontsize=11)
    ax.set_xlim(0, 185)

    # n_clean annotation on right side
    for yi, nc in zip(y_pos, n_clean_sorted):
        ax.text(182, yi, f"n={nc}", ha="right", va="center", fontsize=7, color="gray")

    # Legend
    legend_patches = [
        mpatches.Patch(color="green",     label="Near 90° (within 20°)"),
        mpatches.Patch(color="orange",    label="Moderate (within 40°)"),
        mpatches.Patch(color="salmon",    label="Far from 90°"),
        mpatches.Patch(color="lightgray", label="< 2 clean params"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.suptitle(
        r"Mean-Jacobian vs Scatter-Jacobian orthogonality: $\theta = \arccos(\hat J_{\rm mean} \cdot \hat J_{\rm scatter})$"
        "\n(sorted by decontaminated angle; 90° = maximally orthogonal)",
        fontsize=11
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out_path = FIG_DIR / f"fig_mean_vs_scatter_orthogonality.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[fig5] saved {out_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "config": {
            "rj_clean_threshold": RJ_CLEAN_THRESHOLD,
            "n_params_total": N_par,
        },
        "obs_names": obs_names,
        "angles_all_35_params": angles_all.tolist(),
        "angles_decontaminated": angles_decontam.tolist(),
        "n_clean_params_per_obs": n_clean_params.tolist(),
        "sorted_by_decontam_angle": {
            obs_names[i]: {
                "angle_all": float(angles_all[i]),
                "angle_decontam": float(angles_decontam[i]) if np.isfinite(angles_decontam[i]) else None,
                "n_clean": int(n_clean_params[i]),
            }
            for i in order
        },
        "pass_condition": bool(near_90_any),
        "pass_note": (
            "PASS — at least one observable has decontaminated angle near 90°"
            if near_90_any else
            "FAIL — no observable shows near-orthogonal mean/scatter direction"
        ),
    }

    out_json = OUT_DIR / "fig_mean_vs_scatter_orthogonality.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[fig5] wrote {out_json}")
    print(f"\nOverall pass: {'PASS' if near_90_any else 'FAIL'}")


if __name__ == "__main__":
    main()
