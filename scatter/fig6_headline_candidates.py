"""fig6_headline_candidates.py — Figure 6 of BIND convincingness brief.

4-row × 4-column grid comparing headline candidates:
  Rows: dq_DM|Omega_m (retracted), f_b|A_SN1 (Option A),
        Sigma_gas_r3|A_SN1 (Option B), M_gas|A_SN1 (baseline)
  Columns:
    1. J_log_sigma_inter with SE (scatter sensitivity)
    2. R_j contamination ratio (with threshold lines)
    3. 1P truth J (if available; else "UNAVAILABLE" box)
    4. Verdict: GREEN/YELLOW/RED

Output:
  figures/scatter_diagnostics/fig_headline_candidates.pdf / .png
  outputs/scatter_diagnostics/fig_headline_candidates.json
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
JMEAN_FILE  = BASE_DIR / "scatter/J_mean_and_scatter.npz"
PHASE1_FILE = BASE_DIR / "outputs/scatter_diagnostics/phase1_intra_jacobian.npz"
FIG1_JSON   = BASE_DIR / "outputs/scatter_diagnostics/fig_bind_feedback_mean_response.json"
PHASE4_JSON = BASE_DIR / "outputs/scatter_diagnostics/phase4_1p_truth.json"
FIG_DIR     = BASE_DIR / "figures/scatter_diagnostics"
OUT_DIR     = BASE_DIR / "outputs/scatter_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Candidate definition
# ─────────────────────────────────────────────────────────────────────────────
# (row_label, obs_name, param_idx, param_label, label)
CANDIDATES = [
    ("dq_DM | Ω_m",       "dq_DM",       0, "Omega_m",  "Original (retracted)"),
    ("f_b | A_SN1",        "f_b",         2, "A_SN1",    "Option A"),
    ("Σ_r3 | A_SN1",       "Sigma_gas_r3", 2, "A_SN1",  "Option B"),
    ("M_gas | A_SN1",      "M_gas",        2, "A_SN1",  "Baseline"),
]

# Verdict thresholds
RJ_CONCERN = 0.3     # R_j > this = yellow concern
RJ_CRITICAL = 0.5    # R_j > this = red concern


def load_truth_J(obs_name: str, param_idx: int, obs_names: list) -> tuple[float, float] | None:
    """Try to load truth J from available JSON files. Returns (J, J_se) or None."""
    oi = obs_names.index(obs_name) if obs_name in obs_names else None

    # Phase 4 truth: only has dq_DM for Omega_m and Omega_b
    if PHASE4_JSON.exists():
        p4 = json.loads(PHASE4_JSON.read_text())
        if param_idx == 0 and oi is not None:
            # Look for Omega_m truth in phase 4
            for key in ["1P_p1_n1_vs_1P_p1_1", "Omega_m"]:
                if key in p4:
                    entry = p4[key]
                    if isinstance(entry, dict) and "truth_J_mean" in entry:
                        j_vals = entry["truth_J_mean"]
                        j_se   = entry.get("truth_J_mean_se", [0.0]*len(j_vals))
                        if oi < len(j_vals):
                            return float(j_vals[oi]), float(j_se[oi])

    # Figure 1 truth: has Omega_m vs truth comparison
    if FIG1_JSON.exists():
        f1 = json.loads(FIG1_JSON.read_text())
        if param_idx == 0 and "Omega_m" in f1 and oi is not None:
            entry = f1["Omega_m"]
            if entry.get("has_distinct_truth") and entry.get("truth_J_mean"):
                j_vals = entry["truth_J_mean"]
                j_se   = entry.get("truth_J_mean_se", [0.0]*len(j_vals))
                if oi < len(j_vals):
                    return float(j_vals[oi]), float(j_se[oi])

    return None  # truth unavailable


def make_verdict(jls: float, jls_se: float, rj: float,
                 truth_J: float | None, snr_threshold: float = 2.0) -> tuple[str, str, str]:
    """Return (verdict, color, reason)."""
    snr = abs(jls) / (jls_se + 1e-10)

    if rj > RJ_CRITICAL:
        return "RED", "#FF4444", f"R_j={rj:.2f} > {RJ_CRITICAL} (contaminated)"

    if snr < snr_threshold:
        return "YELLOW", "#FFAA00", f"SNR={snr:.1f} < {snr_threshold} (weak signal)"

    if rj > RJ_CONCERN:
        if truth_J is not None:
            # Has truth — check sign
            if np.sign(jls) == np.sign(truth_J):
                return "YELLOW", "#FFAA00", f"R_j={rj:.2f}>{RJ_CONCERN} but sign✓"
            else:
                return "RED", "#FF4444", f"R_j={rj:.2f}>{RJ_CONCERN} + sign flip"
        return "YELLOW", "#FFAA00", f"R_j={rj:.2f} > {RJ_CONCERN} (moderate)"

    if truth_J is not None:
        if np.sign(jls) == np.sign(truth_J):
            return "GREEN", "#44CC44", f"R_j={rj:.3f}, SNR={snr:.1f}, sign✓"
        else:
            return "RED", "#FF4444", f"R_j={rj:.3f} but sign flip vs truth"

    # Clean but no truth — yellow (unverified)
    return "GREEN", "#44CC44", f"R_j={rj:.3f}, SNR={snr:.1f} (no truth)"


def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    jm = np.load(JMEAN_FILE, allow_pickle=True)
    obs_names   = jm["obs_names"].tolist()
    J_log_sigma = jm["J_log_sigma"]    # (16, 35)
    J_log_sigma_se = jm["J_log_sigma_se"]
    J_log_sigma_intra = jm["J_log_sigma_intra"]
    J_log_sigma_intra_se = jm["J_log_sigma_intra_se"]

    phase1 = np.load(PHASE1_FILE, allow_pickle=True)
    R_j    = phase1["contamination_ratio"]  # (16, 35)

    # Also load J_mean for reference
    J_mean    = jm["J_mean"]
    J_mean_se = jm["J_mean_se"]

    # ── Compute per-candidate values ──────────────────────────────────────────
    rows = []
    for row_label, obs_name, pidx, param_label, cand_label in CANDIDATES:
        oi = obs_names.index(obs_name)
        jls     = float(J_log_sigma[oi, pidx])
        jls_se  = float(J_log_sigma_se[oi, pidx])
        jls_i   = float(J_log_sigma_intra[oi, pidx])
        jls_i_se = float(J_log_sigma_intra_se[oi, pidx])
        rj      = float(R_j[oi, pidx])
        jm_val  = float(J_mean[oi, pidx])
        jm_se   = float(J_mean_se[oi, pidx])

        truth = load_truth_J(obs_name, pidx, obs_names)
        truth_J    = truth[0] if truth else None
        truth_J_se = truth[1] if truth else None

        verdict, verdict_color, reason = make_verdict(jls, jls_se, rj, truth_J)

        rows.append({
            "row_label": row_label,
            "obs_name":  obs_name,
            "pidx":      pidx,
            "param_label": param_label,
            "cand_label": cand_label,
            "J_log_sigma_inter": jls,
            "J_log_sigma_inter_se": jls_se,
            "J_log_sigma_intra": jls_i,
            "J_log_sigma_intra_se": jls_i_se,
            "R_j": rj,
            "J_mean": jm_val,
            "J_mean_se": jm_se,
            "truth_J": truth_J,
            "truth_J_se": truth_J_se,
            "truth_available": truth is not None,
            "verdict": verdict,
            "verdict_color": verdict_color,
            "verdict_reason": reason,
        })

        print(f"\n{row_label} ({cand_label}):")
        print(f"  J_log_sigma_inter = {jls:+.4f} ± {jls_se:.4f}  (SNR={abs(jls)/(jls_se+1e-10):.1f})")
        print(f"  R_j               = {rj:.3f}")
        print(f"  truth_J           = {truth_J}")
        print(f"  verdict           = {verdict}: {reason}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_rows = len(rows)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))

    col_titles = [
        r"$J_{\log\sigma_{\rm inter}}$ ± SE",
        r"$R_j$ contamination",
        r"1P truth $J_{\rm mean}$",
        "Verdict",
    ]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=11, fontweight="bold")

    for ri, row in enumerate(rows):
        # --- Column 0: J_log_sigma_inter bar ---
        ax = axes[ri, 0]
        color = "steelblue" if abs(row["J_log_sigma_inter"]) > 2 * row["J_log_sigma_inter_se"] else "lightblue"
        ax.barh(0, row["J_log_sigma_inter"], color=color, alpha=0.85, height=0.6)
        ax.errorbar(row["J_log_sigma_inter"], 0, xerr=row["J_log_sigma_inter_se"],
                    fmt="none", color="black", capsize=5, linewidth=1.5)
        # Also show J_log_sigma_intra
        ax.barh(-1, row["J_log_sigma_intra"], color="darkorange", alpha=0.6, height=0.6)
        ax.errorbar(row["J_log_sigma_intra"], -1, xerr=row["J_log_sigma_intra_se"],
                    fmt="none", color="black", capsize=4, linewidth=1)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks([0, -1])
        ax.set_yticklabels([r"$\sigma_{\rm inter}$", r"$\sigma_{\rm intra}$"], fontsize=9)
        ax.set_xlabel(row["row_label"], fontsize=9, style="italic")
        snr = abs(row["J_log_sigma_inter"]) / (row["J_log_sigma_inter_se"] + 1e-10)
        ax.text(0.98, 0.95, f"SNR={snr:.1f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="navy")

        # --- Column 1: R_j bar ---
        ax = axes[ri, 1]
        rj_color = "red" if row["R_j"] > RJ_CRITICAL else ("orange" if row["R_j"] > RJ_CONCERN else "green")
        ax.barh(0, row["R_j"], color=rj_color, alpha=0.8, height=0.6)
        ax.axvline(RJ_CONCERN, color="orange", linewidth=1.2, linestyle="--", label=f"R_j={RJ_CONCERN}")
        ax.axvline(RJ_CRITICAL, color="red",    linewidth=1.2, linestyle="--", label=f"R_j={RJ_CRITICAL}")
        ax.set_xlim(0, max(2.5, row["R_j"] * 1.1))
        ax.set_yticks([])
        ax.text(0.98, 0.95, f"R_j={row['R_j']:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="bold", color=rj_color)

        # --- Column 2: truth J ---
        ax = axes[ri, 2]
        if row["truth_available"] and row["truth_J"] is not None:
            sign_ok = (np.sign(row["J_log_sigma_inter"]) == np.sign(row["truth_J"]))
            t_color = "green" if sign_ok else "red"
            ax.barh(0, row["truth_J"], color=t_color, alpha=0.75, height=0.6)
            if row["truth_J_se"]:
                ax.errorbar(row["truth_J"], 0, xerr=row["truth_J_se"],
                            fmt="none", color="black", capsize=4, linewidth=1.2)
            ax.axvline(0, color="black", linewidth=0.8)
            sym = "✓" if sign_ok else "✗"
            ax.text(0.98, 0.95, f"truth {sym}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=10, color=t_color, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "UNAVAILABLE\n(1P sims identical)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))
        ax.set_yticks([])

        # --- Column 3: verdict ---
        ax = axes[ri, 3]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        verdict_txt = row["verdict"]
        reason_txt  = row["verdict_reason"]
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.3), 0.9, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=row["verdict_color"], alpha=0.85, zorder=2
        ))
        ax.text(0.5, 0.5, verdict_txt, ha="center", va="center",
                fontsize=16, fontweight="bold", color="white", zorder=3)
        ax.text(0.5, 0.15, reason_txt, ha="center", va="center",
                fontsize=7.5, color="#333", zorder=3, wrap=True,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
        ax.text(0.5, 0.88, f"({row['cand_label']})", ha="center", va="center",
                fontsize=8, color="#555", style="italic")

    # Column legends
    axes[0, 1].legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Headline candidate comparison\n"
        r"Rows: (1) dq_DM|$\Omega_m$ retracted, (2) $f_b$|$A_{\rm SN1}$ Option A, "
        r"(3) $\Sigma_{r3}$|$A_{\rm SN1}$ Option B, (4) $M_{\rm gas}$|$A_{\rm SN1}$ baseline",
        fontsize=11
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out_path = FIG_DIR / f"fig_headline_candidates.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[fig6] saved {out_path}")

    # ── Pass condition check ──────────────────────────────────────────────────
    green_rows = [r for r in rows if r["verdict"] == "GREEN"]
    pass_cond  = len(green_rows) == 1
    print(f"\n=== Pass condition: exactly 1 GREEN row ===")
    print(f"  Green rows: {[r['row_label'] for r in green_rows]}")
    print(f"  Pass: {pass_cond}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "candidates": [
            {k: v for k, v in r.items() if k != "verdict_color"}
            for r in rows
        ],
        "green_rows": [r["row_label"] for r in green_rows],
        "pass_condition": bool(pass_cond),
        "recommended_headline": green_rows[0]["row_label"] if len(green_rows) == 1 else (
            "AMBIGUOUS" if len(green_rows) > 1 else "NONE"
        ),
    }

    out_json = OUT_DIR / "fig_headline_candidates.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[fig6] wrote {out_json}")
    print(f"\nRecommended headline: {result['recommended_headline']}")


if __name__ == "__main__":
    main()
