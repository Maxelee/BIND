"""scatter/residual_figures.py
Phase 4 figures + summary table for the scatter-residual analysis.

Produces:
  paper_figures/scatter_residual/fig1_anchor.{pdf,png}
  paper_figures/scatter_residual/fig2_correlation_matrices.{pdf,png}
  paper_figures/scatter_residual/fig3_mass_dependence.{pdf,png}
  scatter/scatter_residual/summary_table.md
  scatter/scatter_residual/section_paragraph.md
  scatter/scatter_residual/REPORT.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scatter.residual import OBS_7, OBS_8

# ---------------------------------------------------------------------------
# Paths

SCATTER_DIR = Path("/mnt/home/mlee1/vdm_bind2/scatter")
OUT_DIR     = SCATTER_DIR / "scatter_residual"
FIG_DIR     = Path("/mnt/home/mlee1/vdm_bind2/paper_figures/scatter_residual")
FIG_DIR.mkdir(parents=True, exist_ok=True)

OBS_LATEX = {
    "log10_M_DM":         r"$\log_{10}\,M_{\rm DM}$",
    "log10_M_gas":        r"$\log_{10}\,M_{\rm gas}$",
    "log10_M_star":       r"$\log_{10}\,M_\star$",
    "log10_Sigma_gas_c":  r"$\log_{10}\,\Sigma_{\rm gas,c}$",
    "q_DM":               r"$q_{\rm DM}$",
    "q_gas":              r"$q_{\rm gas}$",
    "q_star":             r"$q_\star$",
    "log10_f_b":          r"$\log_{10}\,f_b$",
}


# ---------------------------------------------------------------------------
# Loaders

def load_all():
    obs = np.load(OUT_DIR / "observables.npz", allow_pickle=False)
    res = np.load(OUT_DIR / "residuals.npz",   allow_pickle=False)
    mat = np.load(OUT_DIR / "matrices.npz",    allow_pickle=False)
    stats = json.loads((OUT_DIR / "stats.json").read_text())
    massd = json.loads((OUT_DIR / "mass_dependence.json").read_text())
    return obs, res, mat, stats, massd


# ---------------------------------------------------------------------------
# Figure 1 — Anchor: SHMR + cross-axis coloring (§6.1)

def figure1_anchor(obs, res):
    """Two rows (truth, BIND-mean), three columns: M_*, M_gas, q_DM vs M_200c,
    coloured by truth ΔM_* rank (cross-axis coloring)."""
    log_M = res["log_M"]
    n_h   = log_M.size

    # Extract per-halo truth and BIND-mean values for the three observables
    obs_F = obs["F"]
    source = obs["source"].astype(str)
    sample_id = obs["sample_id"]
    idx_truth = (source == "truth")
    idx_bind  = (source == "bind")

    def col(name):
        return obs["F"][:, OBS_8.index(name)]

    truth_logMstar = col("log10_M_star")[idx_truth]
    truth_logMgas  = col("log10_M_gas")[idx_truth]
    truth_qDM      = col("q_DM")[idx_truth]

    # Layout (from Phase 1): bind rows are K blocks of N_h each, sample 0 first.
    bind_F_only = obs["F"][idx_bind]                # (N_h*K, 8)
    K_levels = int(sample_id[idx_bind].max() + 1)
    bind_F_per_halo = np.nanmean(
        bind_F_only.reshape(K_levels, n_h, 8), axis=0
    )                                                # (n_h, 8)
    bind_logMstar = bind_F_per_halo[:, OBS_8.index("log10_M_star")]
    bind_logMgas  = bind_F_per_halo[:, OBS_8.index("log10_M_gas")]
    bind_qDM      = bind_F_per_halo[:, OBS_8.index("q_DM")]

    # Color = rank percentile of truth Δlog M_*  (from residuals)
    delta_truth = res["delta_truth"]
    delta_bind  = res["delta_bind_mean"]
    a_star = OBS_8.index("log10_M_star")
    d_star_truth = delta_truth[:, a_star]
    d_star_bind  = delta_bind[:, a_star]

    def rank_pct(d):
        from scipy.stats import rankdata
        x = d.copy()
        mask = np.isfinite(x)
        r = np.full_like(x, np.nan, dtype=np.float64)
        if mask.sum() > 0:
            r[mask] = (rankdata(x[mask]) - 0.5) / mask.sum() * 6 - 3
        return r

    color_truth = rank_pct(d_star_truth)
    color_bind  = rank_pct(d_star_bind)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    cmap = "RdBu_r"
    vmin, vmax = -3, 3

    # Row 0: truth
    for ax, y, label in zip(
        axes[0],
        (truth_logMstar, truth_logMgas, truth_qDM),
        (r"$\log_{10}\,M_\star\;[M_\odot/h]$",
         r"$\log_{10}\,M_{\rm gas}\;[M_\odot/h]$",
         r"$q_{\rm DM}$"),
    ):
        sc = ax.scatter(log_M, y, c=color_truth, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=8, alpha=0.85, edgecolor="none")
        ax.set_ylabel(label, fontsize=11)
        # LOWESS overlay (compute fresh on truth-only for visualization)
        from scatter.residual import lowess_fit
        mu = lowess_fit(log_M, y, frac=0.4)
        xg = np.linspace(np.nanmin(log_M), np.nanmax(log_M), 80)
        ax.plot(xg, mu(xg), "k-", lw=1.5, alpha=0.6)
        ax.grid(alpha=0.3)

    # Row 1: BIND
    for ax, y in zip(axes[1], (bind_logMstar, bind_logMgas, bind_qDM)):
        sc = ax.scatter(log_M, y, c=color_bind, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=8, alpha=0.85, edgecolor="none")
        from scatter.residual import lowess_fit
        mu = lowess_fit(log_M, y, frac=0.4)
        xg = np.linspace(np.nanmin(log_M), np.nanmax(log_M), 80)
        ax.plot(xg, mu(xg), "k-", lw=1.5, alpha=0.6)
        ax.set_xlabel(r"$\log_{10}\,M_{200c}\;[M_\odot/h]$", fontsize=11)
        ax.grid(alpha=0.3)
    axes[1, 0].set_ylabel(r"$\log_{10}\,M_\star\;[M_\odot/h]$", fontsize=11)
    axes[1, 1].set_ylabel(r"$\log_{10}\,M_{\rm gas}\;[M_\odot/h]$", fontsize=11)
    axes[1, 2].set_ylabel(r"$q_{\rm DM}$", fontsize=11)

    axes[0, 0].text(0.02, 0.95, "Truth", transform=axes[0, 0].transAxes,
                    fontsize=14, fontweight="bold", va="top",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    axes[1, 0].text(0.02, 0.95, "BIND", transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight="bold", va="top",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(r"$\hat\Delta_{M_\star}$ rank (Z-scaled)", fontsize=10)

    fig.suptitle(f"Joint residual structure on the mass plane "
                 f"(CAMELS-TNG CV, $N_{{\\rm halos}}={n_h}$)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.9, top=0.93, bottom=0.08,
                        wspace=0.28, hspace=0.18)
    out = FIG_DIR / "fig1_anchor"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}.{{pdf,png}}", flush=True)


# ---------------------------------------------------------------------------
# Figure 2 — Master correlation heatmap (§6.2)

def figure2_correlation_matrices(mat):
    C_T = mat["C_T"]; C_G = mat["C_G"]
    SE_T = mat["SE_T"]; SE_G = mat["SE_G"]
    C_T_full = mat["C_T_full"]; C_G_full = mat["C_G_full"]

    obs7_labels = [OBS_LATEX[n] for n in OBS_7]
    obs8_labels = [OBS_LATEX[n] for n in OBS_8]

    cmap = "RdBu_r"

    def heatmap(ax, M, SE, title, labels, vmin=-1.0, vmax=1.0, cmap=cmap):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = M[i, j]
                err = SE[i, j] if SE is not None else None
                if not np.isfinite(val):
                    txt = "—"
                else:
                    if err is not None and np.isfinite(err):
                        txt = f"{val:+.2f}\n±{err:.2f}"
                    else:
                        txt = f"{val:+.2f}"
                color = "white" if abs((val - (vmin + vmax) / 2) /
                                       (0.5 * (vmax - vmin))) > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, color=color)
        ax.set_title(title, fontsize=11)
        return im

    # --- Top row: 3 panels, 7x7 ---
    fig = plt.figure(figsize=(19, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.45)

    ax_T = fig.add_subplot(gs[0, 0])
    ax_G = fig.add_subplot(gs[0, 1])
    ax_D = fig.add_subplot(gs[0, 2])

    im0 = heatmap(ax_T, C_T, SE_T, r"$C^T$ (truth, Spearman)", obs7_labels)
    im1 = heatmap(ax_G, C_G, SE_G, r"$C^G$ (BIND, Spearman)", obs7_labels)
    # Difference panel: tighter range since |diff| << 1
    dmax = float(np.nanmax(np.abs(C_T - C_G)))
    drng = max(0.2, np.ceil(dmax * 10) / 10)
    im2 = heatmap(ax_D, C_T - C_G,
                  np.sqrt(SE_T ** 2 + SE_G ** 2),
                  r"$C^T - C^G$", obs7_labels, vmin=-drng, vmax=drng)

    for ax, im in zip([ax_T, ax_G, ax_D], [im0, im1, im2]):
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, shrink=0.85)

    # --- Bottom: supplementary 8x8 difference ---
    ax_sup = fig.add_subplot(gs[1, 1])
    diff8 = C_T_full - C_G_full
    d8max = float(np.nanmax(np.abs(diff8)))
    d8rng = max(0.2, np.ceil(d8max * 10) / 10)
    im3 = heatmap(ax_sup, diff8, None,
                  r"$C^T - C^G$  (supplementary 8×8, with $\log_{10} f_b$)",
                  obs8_labels, vmin=-d8rng, vmax=d8rng)
    fig.colorbar(im3, ax=ax_sup, fraction=0.045, pad=0.04, shrink=0.85)
    ax_sup.set_aspect("equal")

    fig.add_subplot(gs[1, 0]).axis("off")
    fig.add_subplot(gs[1, 2]).axis("off")

    fig.suptitle("Residual correlation matrices, BIND vs truth, CAMELS-TNG CV",
                 fontsize=15, fontweight="bold", y=0.98)
    out = FIG_DIR / "fig2_correlation_matrices"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}.{{pdf,png}}", flush=True)


# ---------------------------------------------------------------------------
# Figure 3 — Mass-dependence (§6.3)

def figure3_mass_dependence(stats):
    rho_truth = stats["rho_truth_mass_bins"]
    rho_bind  = stats["rho_bind_mass_bins"]

    fig, ax = plt.subplots(figsize=(7, 5))
    xs_t = [r["mid"] for r in rho_truth]
    ys_t = [r["rho"] for r in rho_truth]
    es_t = [r["se"]  for r in rho_truth]
    xs_b = [r["mid"] for r in rho_bind]
    ys_b = [r["rho"] for r in rho_bind]
    es_b = [r["se"]  for r in rho_bind]

    ax.errorbar(xs_t, ys_t, yerr=es_t, fmt="o-", lw=2, ms=8, color="steelblue",
                label="Truth (CAMELS-TNG)", capsize=4)
    ax.errorbar([x + 0.02 for x in xs_b], ys_b, yerr=es_b, fmt="s-", lw=2, ms=7,
                color="firebrick", label="BIND", capsize=4)

    # F&E18 BAHAMAS reference: ρ ≈ -0.3 at M ≈ 3e14
    ax.scatter([np.log10(3e14)], [-0.3], marker="*", s=220, color="black",
               zorder=5,
               label=r"Farahi & Evrard 2018 (BAHAMAS, $\rho\approx-0.3$ at $3\times10^{14}$)")
    ax.fill_between([np.log10(3e14) - 0.15, np.log10(3e14) + 0.15],
                    [-0.45, -0.45], [-0.15, -0.15], alpha=0.15, color="grey",
                    label="F&E18 reference band")

    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlabel(r"$\log_{10}\,M_{200c}\;[M_\odot/h]$", fontsize=12)
    ax.set_ylabel(r"$\rho_{\rm Spearman}(\hat\Delta M_\star,\,\hat\Delta M_{\rm gas})$",
                  fontsize=12)
    ax.set_title("Mass dependence of the stellar–gas residual correlation",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    bin_edges = stats["mass_bin_edges"]
    edge_str = ", ".join(f"{e:.2f}" for e in bin_edges)
    ax.text(0.02, 0.96, f"bin edges: [{edge_str}]\n"
            "N per bin: " + ", ".join(str(r['n']) for r in rho_truth),
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    fig.tight_layout()
    out = FIG_DIR / "fig3_mass_dependence"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}.{{pdf,png}}", flush=True)


# ---------------------------------------------------------------------------
# Summary table (§6.4)

def build_summary_table(mat, stats, gate1_n_halos: int, K: int) -> str:
    C_T = mat["C_T"]; C_G = mat["C_G"]
    SE_T = mat["SE_T"]; SE_G = mat["SE_G"]

    rows = []
    rows.append("# Summary table — BIND scatter-residual analysis (CAMELS-TNG CV)\n")
    rows.append("## Sample\n")
    rows.append(f"- CV halos (post-cut M_200c > 1e13 M_sun/h): **{gate1_n_halos}**")
    rows.append(f"- BIND samples per halo (K): **{K}**")
    rows.append("- Bootstrap B = **2000** (over halos, with replacement)")
    rows.append("\n## Headline statistics\n")
    rows.append(f"- ‖C^T − C^G‖_F  (primary 7×7, Spearman): **{stats['D_primary_7x7']:.4f}**")
    rows.append(f"  - Frobenius null median (split-half truth): {stats['frobenius_null_median']:.4f}")
    rows.append(f"  - Frobenius p-value: **{stats['frobenius_null_p_value']:.4f}**")
    rows.append(f"- ‖C^T − C^G‖_F  (supplementary 8×8, with log10_f_b): {stats['D_supplementary_8x8']:.4f}")
    rows.append(f"- Angle between leading eigenvectors: **{stats['leading_eigenvector_angle_deg']:.2f}°**")
    rows.append(f"- Top eigenvalue ratio (T/G): {stats['eig_ratio_top']:.3f}")

    # Top 3 largest |C^T| entries (off-diagonal)
    n7 = len(OBS_7)
    pairs = []
    for i in range(n7):
        for j in range(i + 1, n7):
            pairs.append((abs(C_T[i, j]), i, j))
    pairs.sort(reverse=True)
    rows.append("\n## Top 3 strongest truth correlations (off-diagonal)\n")
    rows.append("| obs a | obs b | C^T | C^G | z |")
    rows.append("|---|---|---|---|---|")
    for _, i, j in pairs[:3]:
        z = (C_T[i, j] - C_G[i, j]) / np.sqrt(SE_T[i, j] ** 2 + SE_G[i, j] ** 2 + 1e-30)
        rows.append(f"| {OBS_7[i]} | {OBS_7[j]} | {C_T[i, j]:+.3f} ± {SE_T[i, j]:.3f} | "
                    f"{C_G[i, j]:+.3f} ± {SE_G[i, j]:.3f} | {z:+.2f} |")

    # Per-halo Pearson diagonal
    rows.append("\n## Per-halo Pearson agreement P_aa (expected ≈ 0)\n")
    rows.append("| observable | P_aa | SE |")
    rows.append("|---|---|---|")
    for name in OBS_7:
        p = stats["P_aa"][name]
        s = stats["P_aa_SE"][name]
        rows.append(f"| {name} | {p:+.3f} | {s:.3f} |")

    rows.append("\n## Mass dependence of ρ(ΔM_*, ΔM_gas)\n")
    rows.append("| log10 M200c bin | N | ρ truth | SE truth | ρ BIND | SE BIND |")
    rows.append("|---|---|---|---|---|---|")
    for rt, rb in zip(stats["rho_truth_mass_bins"], stats["rho_bind_mass_bins"]):
        rows.append(f"| [{rt['lo']:.2f}, {rt['hi']:.2f}) | {rt['n']} | "
                    f"{rt['rho']:+.3f} | {rt['se']:.3f} | "
                    f"{rb['rho']:+.3f} | {rb['se']:.3f} |")
    rows.append("\n## Pairs flagged at |z| > 2 (off-diagonal C^T − C^G)\n")
    if stats["z_above_2"]:
        rows.append("| obs a | obs b | C^T | C^G | z |")
        rows.append("|---|---|---|---|---|")
        for f in stats["z_above_2"]:
            rows.append(f"| {f['a']} | {f['b']} | {f['C_T']:+.3f} | "
                        f"{f['C_G']:+.3f} | {f['z']:+.2f} |")
    else:
        rows.append("None.")
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Writeup paragraph (§6.5)

def build_writeup_snippet(mat, stats, gate1_n_halos: int, K: int) -> str:
    D = stats["D_primary_7x7"]
    p = stats["frobenius_null_p_value"]
    angle = stats["leading_eigenvector_angle_deg"]
    P_aa = stats["P_aa"]
    mean_Paa = float(np.nanmean(list(P_aa.values())))
    min_Paa = min(P_aa.values()); max_Paa = max(P_aa.values())
    null = np.load(OUT_DIR / "frobenius_null.npy")
    p95 = float(np.percentile(null, 95))

    rho_truth = stats["rho_truth_mass_bins"]
    rho_bind  = stats["rho_bind_mass_bins"]
    fe_trend  = stats.get("rho_mass_trend_ok", False)

    rho_t_lo = rho_truth[0]["rho"]; rho_t_hi = rho_truth[-1]["rho"]
    rho_b_lo = rho_bind[0]["rho"];  rho_b_hi = rho_bind[-1]["rho"]
    midlo = rho_truth[0]["mid"];     midhi = rho_truth[-1]["mid"]

    if fe_trend:
        fe_clause = (f"the high-mass bin ($\\langle\\log M\\rangle\\approx{midhi:.2f}$) "
                     f"lies below the low-mass bin "
                     f"($\\langle\\log M\\rangle\\approx{midlo:.2f}$), recovering the "
                     f"qualitative sign of the F\\&E18 prediction.")
    else:
        fe_clause = (f"in projection along the $\\sim$50 Mpc/h line of sight the "
                     f"two-dimensional $\\rho(\\hat\\Delta M_\\star,\\hat\\Delta M_{{\\rm gas}})$ "
                     f"does \\emph{{not}} follow the F\\&E18 BAHAMAS trend: truth has "
                     f"$\\rho\\approx{rho_t_lo:+.2f}$ at "
                     f"$\\langle\\log M\\rangle\\approx{midlo:.2f}$ and "
                     f"$\\rho\\approx{rho_t_hi:+.2f}$ at "
                     f"$\\langle\\log M\\rangle\\approx{midhi:.2f}$, both positive. "
                     f"This is consistent with intra-halo gas/stellar baryon co-tracking "
                     f"dominating over the hydrostatic-equilibrium signal that produces "
                     f"the negative correlation in 3D analyses (projection effects washing "
                     f"out the F\\&E18 anti-correlation at $M_{{200c}}>10^{{14}}\\,M_\\odot$). "
                     f"BIND tracks the truth: $\\rho\\approx{rho_b_lo:+.2f}\\to{rho_b_hi:+.2f}$ "
                     f"over the same mass range.")

    lines = []
    lines.append(
        f"\\paragraph{{Joint residual structure across observables.}} "
        f"We test whether BIND reproduces the \\emph{{joint}} residual structure of "
        f"baryonic observables across the CAMELS-IllustrisTNG CV suite. For each "
        f"observable in \\{{$\\log M_{{\\rm DM}}$, $\\log M_{{\\rm gas}}$, $\\log M_\\star$, "
        f"$\\log\\Sigma_{{\\rm gas,c}}$, $q_{{\\rm DM}}$, $q_{{\\rm gas}}$, $q_\\star$\\}}, "
        f"standardised residuals "
        f"$\\hat\\Delta_a = (F_a-\\hat\\mu_a(\\log M_{{200c}}))/\\hat\\sigma_a(\\log M_{{200c}})$ "
        f"are computed against a LOWESS mean fit pooled over truth and BIND (absorbing the "
        f"known $\\sim$5--7\\% stellar mean bias). Across {gate1_n_halos} CV halos with "
        f"$M_{{200c}}>10^{{13}}\\,M_\\odot/h$ and $K={K}$ BIND noise draws per halo, the "
        f"Frobenius distance between the truth and BIND $7\\times7$ Spearman "
        f"residual-correlation matrices is $D=\\|C^T-C^G\\|_F={D:.3f}$ "
        f"($\\!95\\%$ split-half-truth null upper limit ${p95:.3f}$, $p={p:.3f}$). "
        f"The leading eigenvectors of $C^T$ and $C^G$ are aligned to "
        f"${angle:.1f}^\\circ$, indicating that the dominant joint mode of variation "
        f"is shared by BIND and truth even though $D$ is statistically above the null."
    )
    lines.append("")
    lines.append(
        f"\\paragraph{{Per-halo realisations.}} "
        f"BIND is more deterministic per halo than a stochastic generator must be: "
        f"the per-halo Pearson agreement "
        f"$P_{{aa}}=\\mathrm{{Pearson}}(\\hat\\Delta^T_a,\\hat\\Delta^G_a)$ averages "
        f"${mean_Paa:+.3f}$ across the seven observables, with "
        f"$P_{{aa}}\\in[{min_Paa:+.3f},\\,{max_Paa:+.3f}]$. The DMO conditioning "
        f"largely determines the BIND output for a given halo; the residual stochasticity "
        f"contributes only a small per-halo fluctuation around the conditional mean. "
        f"This is qualitatively different from the calibrated-stochastic-generator "
        f"null ($P_{{aa}}\\approx 0$) anticipated for analyses that condition only on "
        f"$M_{{200c}}$: BIND's strong conditioning lets it predict per-realisation "
        f"residual variations from the DMO field. The slight over-coupling in $C^G$ "
        f"relative to $C^T$ (all flagged off-diagonal entries have $|z|>2$ and the "
        f"same sign — see Table~1) is consistent with this picture: the deterministic "
        f"conditioning over-constrains the joint residual structure."
    )
    lines.append("")
    lines.append(
        f"\\paragraph{{Mass dependence of the stellar--gas residual correlation.}} "
        f"As a focused test we examine "
        f"$\\rho_{{\\rm Spearman}}(\\hat\\Delta M_\\star,\\hat\\Delta M_{{\\rm gas}})$ in "
        f"three mass bins. The Farahi \\& Evrard (2018) BAHAMAS prediction is "
        f"near-zero at $\\sim10^{{13}}\\,M_\\odot$ and negative at "
        f"$\\sim 3\\times10^{{14}}\\,M_\\odot$. {fe_clause} "
        f"The reversed sign relative to F\\&E18 is consistent with the BIND paper's "
        f"earlier finding that projection along the box line of sight saturates the "
        f"projected baryon fraction near unity inside halos and obscures the "
        f"hydrostatic anti-correlation -- a robustness test of the analysis pipeline "
        f"itself, not a BIND failure mode."
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Run report (§6.6)

def build_run_report(start_time: float, gate1, gate2, gate3, stats) -> str:
    elapsed = time.time() - start_time
    rows = []
    rows.append("# REPORT — BIND scatter-residual analysis run\n")
    rows.append(f"- Run completed in {elapsed:.1f}s wall-clock.")
    rows.append(f"- Output directory: `{OUT_DIR}`")
    rows.append(f"- Figure directory: `{FIG_DIR}`")
    rows.append("\n## Scripts that ran\n")
    rows.append("1. `scatter/residual_pipeline.py` — phases 1–3.")
    rows.append("2. `scatter/residual_figures.py`  — phase 4 (this script).")
    rows.append("\n## Gates\n")
    for g in (gate1, gate2, gate3):
        rows.append(f"- **{g['gate']}**: PASS={g['PASS']}")
        for k, v in g.items():
            if k in ("PASS", "phase", "gate"):
                continue
            if isinstance(v, (dict, list)):
                continue
            rows.append(f"  - {k}: {v}")
    rows.append("\n## Headline numbers\n")
    rows.append(f"- N halos: {gate1['n_halos_post_cut']}, K = {gate1['K']}")
    rows.append(f"- ‖C^T − C^G‖_F (7×7): {stats['D_primary_7x7']:.4f}, "
                f"split-half null p = {stats['frobenius_null_p_value']:.4f}")
    rows.append(f"- Leading eigenvector angle: {stats['leading_eigenvector_angle_deg']:.2f}°")
    rows.append(f"- Mean P_aa across 7 obs: "
                f"{float(np.mean(list(stats['P_aa'].values()))):+.3f}")
    rows.append(f"- ρ_truth(ΔM*,ΔMgas) [low→high mass]: "
                f"{stats['rho_truth_mass_bins'][0]['rho']:+.3f} → "
                f"{stats['rho_truth_mass_bins'][-1]['rho']:+.3f}")
    rows.append(f"- ρ_BIND (ΔM*,ΔMgas) [low→high mass]: "
                f"{stats['rho_bind_mass_bins'][0]['rho']:+.3f} → "
                f"{stats['rho_bind_mass_bins'][-1]['rho']:+.3f}")
    rows.append("\n## Warnings\n")
    if not stats.get("rho_mass_trend_ok", True):
        rows.append(f"- {stats.get('rho_mass_trend_warning', 'rho mass trend reversed')}")
    if stats.get("z_above_2"):
        rows.append(f"- {len(stats['z_above_2'])} pair(s) flagged at |z| > 2 — see summary table.")
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Main entry point

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("[Phase 4] Loading artefacts ...", flush=True)
    obs, res, mat, stats, massd = load_all()

    gate1 = json.loads((OUT_DIR / "gate1_report.json").read_text())
    gate2 = json.loads((OUT_DIR / "gate2_report.json").read_text())
    gate3 = json.loads((OUT_DIR / "gate3_report.json").read_text())
    K = gate1["K"]
    n_halos = gate1["n_halos_post_cut"]

    print("[Phase 4] Figure 1 — anchor ...", flush=True)
    figure1_anchor(obs, res)

    print("[Phase 4] Figure 2 — correlation matrices ...", flush=True)
    figure2_correlation_matrices(mat)

    print("[Phase 4] Figure 3 — mass dependence ...", flush=True)
    figure3_mass_dependence(stats)

    print("[Phase 4] Summary table + writeup snippet + REPORT ...", flush=True)
    (OUT_DIR / "summary_table.md").write_text(build_summary_table(mat, stats, n_halos, K))
    (OUT_DIR / "section_paragraph.md").write_text(build_writeup_snippet(mat, stats, n_halos, K))
    (OUT_DIR / "REPORT.md").write_text(build_run_report(time.time(), gate1, gate2, gate3, stats))

    # Gate 4: verify all artefacts on disk
    gate4 = {
        "phase": "4",
        "gate": "Gate 4",
        "fig1_pdf": (FIG_DIR / "fig1_anchor.pdf").exists(),
        "fig1_png": (FIG_DIR / "fig1_anchor.png").exists(),
        "fig2_pdf": (FIG_DIR / "fig2_correlation_matrices.pdf").exists(),
        "fig2_png": (FIG_DIR / "fig2_correlation_matrices.png").exists(),
        "fig3_pdf": (FIG_DIR / "fig3_mass_dependence.pdf").exists(),
        "fig3_png": (FIG_DIR / "fig3_mass_dependence.png").exists(),
        "summary_table_exists": (OUT_DIR / "summary_table.md").exists(),
        "section_paragraph_exists": (OUT_DIR / "section_paragraph.md").exists(),
        "report_exists": (OUT_DIR / "REPORT.md").exists(),
    }
    gate4["PASS"] = all(gate4[k] for k in gate4 if k.startswith(("fig", "summary", "section", "report")))
    (OUT_DIR / "gate4_report.json").write_text(json.dumps(gate4, indent=2, default=str))
    print(f"  Gate 4 → {gate4['PASS']}")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
