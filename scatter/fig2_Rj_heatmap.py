"""fig2_Rj_heatmap.py — Figure 2 of BIND convincingness brief.

R_j = |J_log_sigma_intra| / |J_log_sigma_inter| heatmap across
16 observables × 35 parameters. Columns grouped by parameter class;
rows grouped by observable family. Right panel: median R_j per row
across the 4 headline parameters.

Output:
  figures/scatter_diagnostics/fig_Rj_heatmap.pdf / .png
  outputs/scatter_diagnostics/fig_Rj_heatmap.json
"""
from __future__ import annotations

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent.parent
NPZ_FILE = BASE_DIR / "outputs/scatter_diagnostics/phase1_intra_jacobian.npz"
FIG_DIR  = BASE_DIR / "figures/scatter_diagnostics"
OUT_DIR  = BASE_DIR / "outputs/scatter_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Parameter groupings (by class)
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GROUPS = {
    "Cosmology":   [0, 1, 6, 7, 8],            # Omega_m, sigma8, Omega_b, H0, n_s
    "SN/Wind":     [2, 4, 9, 10, 11, 12, 13,    # A_SN1, A_SN2, MaxSfr, SoftEQS, IMFslope,
                    14, 15, 16, 17, 18, 19, 20], # SNII_MinMass, ThermalWind...WindDumpFac
    "AGN/BH":      [3, 5, 21, 22, 23, 24, 25,   # A_AGN1, A_AGN2, SeedBHMass...
                    26, 27],                     # QuasarThreshold, QuasarThreshPow
    "UV bg":       [28, 29, 30, 31],
    "Other":       [32, 33, 34],
}

# Observable row groupings
OBS_GROUPS = {
    "Masses":  ["M_dm", "M_gas", "M_star", "f_b", "f_b_norm", "Rc_over_R200"],
    "Shapes":  ["q_DM", "q_gas", "q_star", "dq_DM"],
    "Profiles":["Sigma_gas_c", "Sigma_gas_r0", "Sigma_gas_r1",
                "Sigma_gas_r2", "Sigma_gas_r3", "Sigma_gas_r4"],
}

# Four headline parameters (original BIND paper focus)
HEADLINE_PIDX = [0, 2, 3, 4]  # Omega_m, A_SN1, A_AGN1, A_SN2

# ─────────────────────────────────────────────────────────────────────────────
# Custom colormap: white → yellow → orange → deep red
# ─────────────────────────────────────────────────────────────────────────────
CMAP_COLORS = [
    (0.0,  (1.00, 1.00, 1.00)),   # R_j = 0.0: white
    (0.30, (1.00, 0.95, 0.40)),   # R_j = 0.3: yellow
    (0.50, (1.00, 0.55, 0.10)),   # R_j = 0.5: orange
    (1.00, (0.60, 0.05, 0.05)),   # R_j ≥ 1.0: deep red
]

def make_rj_cmap(vmax: float = 1.0) -> LinearSegmentedColormap:
    """Build a white→yellow→orange→deep-red colourmap over [0, vmax]."""
    nodes = [(v / vmax, c) for v, c in CMAP_COLORS if v <= vmax]
    if nodes[-1][0] < 1.0:
        nodes.append((1.0, CMAP_COLORS[-1][1]))
    return LinearSegmentedColormap.from_list(
        "rj_cmap", [(n, c) for n, c in nodes], N=256
    )


def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    d = np.load(NPZ_FILE, allow_pickle=True)
    obs_names_arr  = d["obs_names"]   # (16,)
    param_names_arr = d["param_names"] # (35,)
    rj_raw         = d["contamination_ratio"].astype(float)  # (16, 35)

    obs_names   = obs_names_arr.tolist()
    param_names = param_names_arr.tolist()

    # Cap R_j at 2 for display (values > 1 are all "fully contaminated")
    RJ_DISPLAY_MAX = 2.0
    rj = np.clip(rj_raw, 0, RJ_DISPLAY_MAX)

    # ── Build reordered indices ───────────────────────────────────────────────
    col_order = []
    col_group_edges = []  # (start_col, group_name)
    for gname, pidxs in PARAM_GROUPS.items():
        col_group_edges.append((len(col_order), gname))
        col_order.extend(pidxs)
    assert len(col_order) == 35, f"Expected 35 params, got {len(col_order)}"

    row_order = []
    row_group_edges = []  # (start_row, group_name)
    for gname, onames in OBS_GROUPS.items():
        row_group_edges.append((len(row_order), gname))
        for n in onames:
            if n in obs_names:
                row_order.append(obs_names.index(n))
            else:
                print(f"  WARNING: obs '{n}' not in obs_names, skipping")
    assert len(row_order) == 16, f"Expected 16 obs, got {len(row_order)}"

    # Reorder heatmap
    rj_plot = rj[np.ix_(row_order, col_order)]  # (16, 35)
    obs_labels_plot   = [obs_names[i].replace("Sigma_gas_", "Σ_r").replace("_over_R200", "/R₂₀₀")
                         for i in row_order]
    param_labels_plot = [param_names[j] for j in col_order]

    # Headline param columns in reordered space
    headline_cols = [col_order.index(p) for p in HEADLINE_PIDX if p in col_order]

    # Right panel: median R_j per obs-row across headline params (raw, not capped)
    rj_raw_reordered = rj_raw[np.ix_(row_order, col_order)]
    rj_headline_median = np.median(rj_raw_reordered[:, headline_cols], axis=1)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 8))
    # Two subplots: main heatmap (wide) + right panel (narrow)
    gs = fig.add_gridspec(1, 2, width_ratios=[35, 2.5], wspace=0.05)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    vmax = 1.0  # R_j colormap saturates at 1.0 (even though display cap is 2.0)
    cmap = make_rj_cmap(vmax=vmax)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    im = ax_heat.imshow(rj_plot, aspect="auto", cmap=cmap, norm=norm,
                        interpolation="nearest", origin="upper")

    # ── Grid lines ───────────────────────────────────────────────────────────
    ax_heat.set_xticks(np.arange(35) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(16) - 0.5, minor=True)
    ax_heat.grid(which="minor", color="lightgray", linewidth=0.4)
    ax_heat.tick_params(which="minor", length=0)

    # Group separator lines (thicker)
    for (start_col, _) in col_group_edges[1:]:
        ax_heat.axvline(start_col - 0.5, color="black", linewidth=1.8, zorder=5)
    for (start_row, _) in row_group_edges[1:]:
        ax_heat.axhline(start_row - 0.5, color="black", linewidth=1.8, zorder=5)

    # ── Tick labels ──────────────────────────────────────────────────────────
    ax_heat.set_yticks(np.arange(16))
    ax_heat.set_yticklabels(obs_labels_plot, fontsize=8)
    ax_heat.set_xticks(np.arange(35))
    ax_heat.set_xticklabels(param_labels_plot, rotation=70, ha="right", fontsize=7)

    # ── Cell text labels (R_j > 0.3, using raw values) ───────────────────────
    for row_i in range(16):
        for col_j in range(35):
            raw_val = float(rj_raw_reordered[row_i, col_j])
            if raw_val > 0.3:
                display = f"{min(raw_val, 9.99):.2f}"
                text_color = "white" if min(raw_val, vmax) / vmax > 0.65 else "black"
                ax_heat.text(col_j, row_i, display, ha="center", va="center",
                             fontsize=5.5, color=text_color, fontweight="bold")

    # ── Parameter class strip above columns ──────────────────────────────────
    class_colors = {"Cosmology": "#4472C4", "SN/Wind": "#ED7D31",
                    "AGN/BH": "#FFC000", "UV bg": "#70AD47", "Other": "#9E9E9E"}
    strip_y = 16.0
    strip_h = 0.8
    for (start_col, gname), (end_col, _) in zip(
        col_group_edges,
        col_group_edges[1:] + [(35, "")]
    ):
        mid = (start_col + end_col - 1) / 2
        span = end_col - start_col
        rect = mpatches.FancyBboxPatch(
            (start_col - 0.5, strip_y - 0.05), span, strip_h,
            boxstyle="square,pad=0", facecolor=class_colors.get(gname, "#888"),
            alpha=0.85, transform=ax_heat.transData, clip_on=False, zorder=10
        )
        ax_heat.add_patch(rect)
        ax_heat.text(mid, strip_y + strip_h / 2, gname,
                     ha="center", va="center", fontsize=8, fontweight="bold",
                     color="white", zorder=11, transform=ax_heat.transData, clip_on=False)

    # ── Observable family labels (left margin) ────────────────────────────────
    fam_colors = {"Masses": "#A0C8FF", "Shapes": "#FFD580", "Profiles": "#C8F0C8"}
    for (start_row, gname), (end_row, _) in zip(
        row_group_edges,
        row_group_edges[1:] + [(16, "")]
    ):
        mid = (start_row + end_row - 1) / 2
        ax_heat.text(-1.6, mid, gname, ha="right", va="center",
                     fontsize=9, fontweight="bold", color="#444",
                     transform=ax_heat.transData, rotation=90)

    ax_heat.set_title(
        r"$R_j = |J_{\log\sigma_{\rm intra}}| / |J_{\log\sigma_{\rm inter}}|$"
        "  — contamination ratio\n"
        "(cells with $R_j > 0.3$ labelled; colour saturates at $R_j = 1$)",
        fontsize=11, pad=24
    )

    # ── Highlight headline param columns ─────────────────────────────────────
    for hc in headline_cols:
        ax_heat.axvspan(hc - 0.5, hc + 0.5, color="cyan", alpha=0.12, zorder=0)

    # ── Right panel: median R_j across headline params ────────────────────────
    ax_right.barh(np.arange(16), rj_headline_median[::-1],
                  color="steelblue", alpha=0.75, height=0.7)
    ax_right.axvline(0.3, color="darkorange", linewidth=1.2, linestyle="--", label="0.3")
    ax_right.axvline(0.5, color="red", linewidth=1.2, linestyle="--", label="0.5")
    ax_right.set_yticks([])
    ax_right.set_xlabel("Median $R_j$\n(headline params)", fontsize=8)
    ax_right.set_xlim(0, max(1.5, rj_headline_median.max() * 1.05))
    ax_right.xaxis.set_tick_params(labelsize=7)
    ax_right.set_title("Median\n$R_j$", fontsize=8)
    ax_right.legend(fontsize=7, loc="upper right")

    # ── Colourbar ────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=[ax_heat, ax_right], fraction=0.015, pad=0.01,
                        label=r"$R_j$ (capped at 1 for colour; raw values labelled)")
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out_path = FIG_DIR / f"fig_Rj_heatmap.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[fig2] saved {out_path}")

    # ── Block structure check ─────────────────────────────────────────────────
    cosmo_cols  = [col_order.index(p) for p in PARAM_GROUPS["Cosmology"]]
    sn_cols     = [col_order.index(p) for p in PARAM_GROUPS["SN/Wind"]]
    agn_cols    = [col_order.index(p) for p in PARAM_GROUPS["AGN/BH"]]
    shapes_rows = [row_order.index(obs_names.index(n)) for n in OBS_GROUPS["Shapes"] if n in obs_names]
    profiles_rows = [row_order.index(obs_names.index(n)) for n in OBS_GROUPS["Profiles"] if n in obs_names]
    masses_rows = [row_order.index(obs_names.index(n)) for n in OBS_GROUPS["Masses"] if n in obs_names]

    rj_cosmo_shapes   = float(np.median(rj_raw_reordered[np.ix_(shapes_rows, cosmo_cols)]))
    rj_sn_masses      = float(np.median(rj_raw_reordered[np.ix_(masses_rows, sn_cols)]))
    rj_cosmo_masses   = float(np.median(rj_raw_reordered[np.ix_(masses_rows, cosmo_cols)]))
    rj_sn_shapes      = float(np.median(rj_raw_reordered[np.ix_(shapes_rows, sn_cols)]))

    print("\n=== Block structure check ===")
    print(f"  median R_j(cosmo × shapes)   = {rj_cosmo_shapes:.3f}  (should be HOT)")
    print(f"  median R_j(cosmo × masses)   = {rj_cosmo_masses:.3f}")
    print(f"  median R_j(SN × masses)      = {rj_sn_masses:.3f}")
    print(f"  median R_j(SN × shapes)      = {rj_sn_shapes:.3f}")
    block_ok = (rj_cosmo_shapes > rj_sn_shapes)
    print(f"  Block structure visible: {block_ok}  (cosmo hot on shapes?)")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    headline_names = [param_names[p] for p in HEADLINE_PIDX]
    result = {
        "config": {
            "npz_file": str(NPZ_FILE),
            "headline_params": {p: n for p, n in zip(HEADLINE_PIDX, headline_names)},
            "rj_display_max": RJ_DISPLAY_MAX,
            "rj_colormap_saturate": 1.0,
        },
        "block_structure": {
            "rj_cosmo_x_shapes": rj_cosmo_shapes,
            "rj_sn_x_masses": rj_sn_masses,
            "rj_cosmo_x_masses": rj_cosmo_masses,
            "rj_sn_x_shapes": rj_sn_shapes,
            "block_structure_visible": bool(block_ok),
        },
        "per_row_median_headline": {
            obs_names[row_order[i]]: float(rj_headline_median[i])
            for i in range(16)
        },
        "pass_condition": bool(block_ok),
    }

    out_json = OUT_DIR / "fig_Rj_heatmap.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[fig2] wrote {out_json}")
    print(f"\nOverall pass: {'PASS' if block_ok else 'FAIL — heatmap lacks block structure'}")


if __name__ == "__main__":
    main()
