"""fig7_parameter_signal_survey.py — Figure 7 of BIND convincingness brief.

For each of 35 parameters j, find the observable a*(j) that maximises:
    score(j, a) = SNR(J_log_sigma_inter, j,a) * I[R_j,a < 0.5]
Score is 0 for contaminated combinations; otherwise the SNR of the inter-Jacobian.

Plot: horizontal bar chart, sorted by score descending.
Side-table: per parameter-class median R_j and best score.

Output:
  figures/scatter_diagnostics/fig_parameter_signal_survey.pdf / .png
  outputs/scatter_diagnostics/fig_parameter_signal_survey.json
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
BASE_DIR    = pathlib.Path(__file__).parent.parent
PHASE1_FILE = BASE_DIR / "outputs/scatter_diagnostics/phase1_intra_jacobian.npz"
FIG_DIR     = BASE_DIR / "figures/scatter_diagnostics"
OUT_DIR     = BASE_DIR / "outputs/scatter_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SNR_DETECT   = 2.0   # detection threshold line
RJ_CLEAN     = 0.3   # R_j < this → green
RJ_MARGINAL  = 0.5   # R_j < this → yellow (still included in score)

# ─────────────────────────────────────────────────────────────────────────────
# Parameter class definitions (using repo param_names)
# ─────────────────────────────────────────────────────────────────────────────
PARAMETER_CLASSES = {
    "cosmology":    ["Omega_m", "sigma8", "Omega_b", "H0", "n_s"],
    "SN_amplitude": ["A_SN1", "A_SN2"],
    "SN_subgrid":   ["MaxSfr", "SoftEQS", "IMFslope", "SNII_MinMass",
                     "ThermalWind", "WindSpecMom", "WindFreeTravelDens",
                     "MinWindVel", "WindEnergyReduction", "WindEnergyReductionZ",
                     "WindEnergyReductionExp", "WindDumpFac"],
    "AGN_amplitude": ["A_AGN1", "A_AGN2"],
    "AGN_subgrid":  ["SeedBHMass", "BHAccretion", "BHEddington", "BHFeedback",
                     "BHRadEff", "QuasarThreshold", "QuasarThreshPow"],
    "cooling_ISM":  ["UVB_H0_beta", "UVB_H0_Dz", "UVB_Hep_beta", "UVB_Hep_Dz",
                     "SNIa_norm", "SNIa_DTD_pow", "SofteningComoving"],
}

CLASS_COLORS = {
    "cosmology":     "#2166ac",
    "SN_amplitude":  "#d73027",
    "SN_subgrid":    "#fc8d59",
    "AGN_amplitude": "#762a83",
    "AGN_subgrid":   "#af8dc3",
    "cooling_ISM":   "#4dac26",
}

def param_class(pname: str) -> str:
    for cls, names in PARAMETER_CLASSES.items():
        if pname in names:
            return cls
    return "other"


def main() -> None:
    # ── Load Phase 1 data ────────────────────────────────────────────────────
    d = np.load(PHASE1_FILE, allow_pickle=True)
    J_inter     = d["J_log_sigma_inter"]       # (16, 35)
    J_inter_se  = d["J_log_sigma_inter_se"]    # (16, 35)
    R_j         = d["contamination_ratio"]     # (16, 35)
    obs_names   = d["obs_names"].tolist()
    param_names = d["param_names"].tolist()
    N_obs, N_par = J_inter.shape

    # ── Compute SNR and scores ───────────────────────────────────────────────
    # Clip SE to avoid div/zero
    safe_se = np.where(J_inter_se > 1e-9, J_inter_se, np.nan)
    SNR = np.abs(J_inter) / safe_se   # (16, 35)

    # Score = SNR * I[R_j < 0.5]; 0 for contaminated
    score_arr = np.where(R_j < RJ_MARGINAL, SNR, 0.0)  # (16, 35)

    # For each parameter: best observable, best score, best R_j, best SNR
    results = []
    for j in range(N_par):
        best_a = int(np.argmax(score_arr[:, j]))
        best_score = float(score_arr[best_a, j])
        best_rj    = float(R_j[best_a, j])
        best_snr   = float(SNR[best_a, j]) if np.isfinite(SNR[best_a, j]) else 0.0
        best_obs   = obs_names[best_a]
        results.append({
            "param":     param_names[j],
            "pidx":      j,
            "class":     param_class(param_names[j]),
            "best_obs":  best_obs,
            "score":     best_score,
            "snr":       best_snr,
            "rj":        best_rj,
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # ── Per-class summary ────────────────────────────────────────────────────
    class_summary = {}
    for cls in PARAMETER_CLASSES:
        cls_rows = [r for r in results if r["class"] == cls]
        if not cls_rows:
            continue
        # Median R_j across all (obs, param) cells in this class
        cls_idxs = [r["pidx"] for r in cls_rows]
        rj_all = R_j[:, cls_idxs]  # (16, n_params_in_class)
        class_summary[cls] = {
            "n_params":      len(cls_rows),
            "median_rj":     float(np.nanmedian(rj_all)),
            "best_score":    float(max(r["score"] for r in cls_rows)),
            "best_param":    max(cls_rows, key=lambda x: x["score"])["param"],
            "best_obs":      max(cls_rows, key=lambda x: x["score"])["best_obs"],
            "n_detectable":  sum(1 for r in cls_rows if r["score"] >= SNR_DETECT),
        }

    print("=== Per-class summary ===")
    for cls, s in class_summary.items():
        print(f"  {cls:15s}: median_R_j={s['median_rj']:.3f}, "
              f"best_score={s['best_score']:.1f} ({s['best_param']} → {s['best_obs']}), "
              f"n_detect={s['n_detectable']}/{s['n_params']}")

    print("\n=== Top 15 parameters by score ===")
    for r in results[:15]:
        print(f"  {r['param']:25s} [{r['class']:15s}]: "
              f"score={r['score']:.2f}  SNR={r['snr']:.2f}  R_j={r['rj']:.3f}  best_obs={r['best_obs']}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 10),
                                   gridspec_kw={"width_ratios": [2.5, 1]})

    # Left panel: sorted horizontal bar chart
    n = len(results)
    y_pos = np.arange(n)

    bar_colors = []
    for r in results:
        if r["score"] == 0.0:
            bar_colors.append("#cccccc")   # grey: contaminated / no signal
        elif r["rj"] < RJ_CLEAN:
            bar_colors.append("#1a9641")   # green: clean
        else:
            bar_colors.append("#fdae61")   # yellow: marginal

    scores = [r["score"] for r in results]
    bars = ax.barh(y_pos, scores, color=bar_colors, edgecolor="white", linewidth=0.4)

    # Annotate with best observable name
    for i, r in enumerate(results):
        ax.text(max(r["score"], 0.05) + 0.05, i, r["best_obs"],
                va="center", ha="left", fontsize=7, color="#333333")

    ax.axvline(SNR_DETECT, color="black", linestyle="--", linewidth=1.0, label=f"SNR = {SNR_DETECT}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["param"] for r in results], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Best-observable SNR  [R_j < 0.5 only]", fontsize=10)
    ax.set_title("Per-parameter best-signal survey", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.3)

    # Colour the y-tick labels by parameter class
    for tick, r in zip(ax.get_yticklabels(), results):
        cls = r["class"]
        tick.set_color(CLASS_COLORS.get(cls, "black"))

    # Legend for bar colours
    legend_patches = [
        mpatches.Patch(color="#1a9641", label=f"R_j < {RJ_CLEAN} (clean)"),
        mpatches.Patch(color="#fdae61", label=f"R_j {RJ_CLEAN}–{RJ_MARGINAL} (marginal)"),
        mpatches.Patch(color="#cccccc", label=f"R_j ≥ {RJ_MARGINAL} or no signal"),
    ]
    for cls, col in CLASS_COLORS.items():
        legend_patches.append(mpatches.Patch(color=col, label=f"[{cls}]"))
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, ncol=1)

    # Right panel: per-class table
    ax2.axis("off")
    cls_order = ["cosmology", "SN_amplitude", "SN_subgrid",
                 "AGN_amplitude", "AGN_subgrid", "cooling_ISM"]
    col_headers = ["Class", "N", "Median\nR_j", "Best\nSNR", "Best\nparam", "N det"]
    rows_data = []
    for cls in cls_order:
        if cls not in class_summary:
            continue
        s = class_summary[cls]
        rows_data.append([
            cls.replace("_", "\n"),
            str(s["n_params"]),
            f"{s['median_rj']:.2f}",
            f"{s['best_score']:.1f}",
            s["best_param"],
            f"{s['n_detectable']}/{s['n_params']}",
        ])

    table = ax2.table(
        cellText=rows_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="upper center",
        bbox=[0, 0.1, 1.0, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    # Color class rows
    for i, cls in enumerate(cls_order):
        if cls not in class_summary:
            continue
        for j in range(len(col_headers)):
            table[(i + 1, j)].set_facecolor(CLASS_COLORS.get(cls, "white") + "44")
    ax2.set_title("Per-class summary", fontsize=10, fontweight="bold", pad=4)

    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = FIG_DIR / f"fig_parameter_signal_survey.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[fig7] saved {path}")
    plt.close(fig)

    # ── JSON sidecar ─────────────────────────────────────────────────────────
    out = {
        "snr_threshold": SNR_DETECT,
        "rj_marginal":   RJ_MARGINAL,
        "rj_clean":      RJ_CLEAN,
        "ranking": results,
        "class_summary": class_summary,
        "top10_above_threshold": [
            r for r in results if r["score"] >= SNR_DETECT
        ][:10],
    }
    json_path = OUT_DIR / "fig_parameter_signal_survey.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[fig7] wrote {json_path}")

    # ── Print findings ────────────────────────────────────────────────────────
    detectable = [r for r in results if r["score"] >= SNR_DETECT]
    print(f"\n=== Summary: {len(detectable)} parameters with clean+detectable signals (score >= {SNR_DETECT}) ===")
    for r in detectable:
        print(f"  {r['param']:20s} [{r['class']:12s}]: SNR={r['snr']:.1f}, R_j={r['rj']:.3f} → {r['best_obs']}")
    if not detectable:
        print("  (none above threshold)")


if __name__ == "__main__":
    main()
