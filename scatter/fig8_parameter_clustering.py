"""fig8_parameter_clustering.py — Figure 8 of BIND convincingness brief.

For each parameter j, take its 16-dim mean-Jacobian vector, normalise to unit
length, and compute pairwise cosine similarity S_jk.

Two-panel figure:
  (a) 35×35 similarity heatmap ordered by hierarchical clustering.
      Dendrogram cut at k=6 clusters; colour strip alongside.
  (b) Same heatmap ordered by physical parameter classes from §6.5.

Outputs:
  figures/scatter_diagnostics/fig_parameter_clustering.pdf / .png
  outputs/scatter_diagnostics/fig_parameter_clustering.json
    — cluster assignments and agreement fraction.
"""
from __future__ import annotations

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent.parent
JMEAN_FILE  = BASE_DIR / "scatter/J_mean_and_scatter.npz"
FIG_DIR     = BASE_DIR / "figures/scatter_diagnostics"
OUT_DIR     = BASE_DIR / "outputs/scatter_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Parameter class definitions
# ─────────────────────────────────────────────────────────────────────────────
PARAMETER_CLASSES = {
    "cosmology":     ["Omega_m", "sigma8", "Omega_b", "H0", "n_s"],
    "SN_amplitude":  ["A_SN1", "A_SN2"],
    "SN_subgrid":    ["MaxSfr", "SoftEQS", "IMFslope", "SNII_MinMass",
                      "ThermalWind", "WindSpecMom", "WindFreeTravelDens",
                      "MinWindVel", "WindEnergyReduction", "WindEnergyReductionZ",
                      "WindEnergyReductionExp", "WindDumpFac"],
    "AGN_amplitude": ["A_AGN1", "A_AGN2"],
    "AGN_subgrid":   ["SeedBHMass", "BHAccretion", "BHEddington", "BHFeedback",
                      "BHRadEff", "QuasarThreshold", "QuasarThreshPow"],
    "cooling_ISM":   ["UVB_H0_beta", "UVB_H0_Dz", "UVB_Hep_beta", "UVB_Hep_Dz",
                      "SNIa_norm", "SNIa_DTD_pow", "SofteningComoving"],
}

# Map class name → integer label (for agreement computation)
CLASS_INT = {c: i for i, c in enumerate(PARAMETER_CLASSES)}

CLASS_COLORS = {
    "cosmology":     "#2166ac",
    "SN_amplitude":  "#d73027",
    "SN_subgrid":    "#fc8d59",
    "AGN_amplitude": "#762a83",
    "AGN_subgrid":   "#af8dc3",
    "cooling_ISM":   "#4dac26",
    "other":         "#888888",
}

# Palette for k=6 data-driven clusters
CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a",
    "#984ea3", "#ff7f00", "#a65628",
]


def param_class(pname: str) -> str:
    for cls, names in PARAMETER_CLASSES.items():
        if pname in names:
            return cls
    return "other"


def cosine_similarity_matrix(M: np.ndarray) -> np.ndarray:
    """M shape (N_obs, N_par). Returns (N_par, N_par) cosine similarity."""
    norms = np.linalg.norm(M, axis=0)          # (N_par,)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    M_norm = M / safe_norms[np.newaxis, :]     # (N_obs, N_par)
    S = M_norm.T @ M_norm                      # (N_par, N_par)
    S = np.clip(S, -1.0, 1.0)
    return S


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    jm = np.load(JMEAN_FILE, allow_pickle=True)
    J_mean    = jm["J_mean"]          # (16, 35)
    obs_names = jm["obs_names"].tolist()

    # param_names from phase1
    ph1 = np.load(
        BASE_DIR / "outputs/scatter_diagnostics/phase1_intra_jacobian.npz",
        allow_pickle=True,
    )
    param_names = ph1["param_names"].tolist()
    N_obs, N_par = J_mean.shape

    # ── Cosine similarity ─────────────────────────────────────────────────────
    S = cosine_similarity_matrix(J_mean)   # (35, 35)

    # ── Hierarchical clustering on dissimilarity 1 - |S| ─────────────────────
    dist_mat = 1.0 - np.abs(S)
    np.fill_diagonal(dist_mat, 0.0)
    dist_vec = squareform(dist_mat, checks=False)
    Z = linkage(dist_vec, method="average")

    # k=6 clusters
    k = 6
    cluster_labels = fcluster(Z, k, criterion="maxclust")   # 1-indexed
    # Order by cluster label (sorted by hierarchical ordering)
    dend = dendrogram(Z, no_plot=True)
    hier_order = dend["leaves"]   # indices of params in dendrogram order

    # Physical class ordering
    class_order = list(PARAMETER_CLASSES.keys())
    phys_order = []
    for cls in class_order:
        cls_idxs = [i for i, p in enumerate(param_names) if param_class(p) == cls]
        phys_order.extend(cls_idxs)
    # add any "other" at the end
    other_idxs = [i for i, p in enumerate(param_names) if param_class(p) == "other"]
    phys_order.extend(other_idxs)

    # ── Cluster–class agreement ───────────────────────────────────────────────
    phys_class_int = np.array([CLASS_INT.get(param_class(p), -1) for p in param_names])
    # For each physical class, find the most common data-driven cluster label
    agreement_per_param = []
    for j in range(N_par):
        pc = phys_class_int[j]
        if pc < 0:
            continue
        # majority cluster within this physical class
        cls_name = list(CLASS_INT.keys())[pc]
        cls_idxs = [i for i, p in enumerate(param_names) if param_class(p) == cls_name]
        cls_clusters = cluster_labels[cls_idxs]
        majority = int(np.bincount(cls_clusters).argmax())
        if majority == 0:
            majority = int(np.bincount(cls_clusters[cls_clusters > 0]).argmax()) if np.any(cls_clusters > 0) else -1
        agreement_per_param.append(cluster_labels[j] == majority)
    agreement_frac = float(np.mean(agreement_per_param)) if agreement_per_param else 0.0

    print(f"Cluster–class agreement fraction: {agreement_frac:.3f}")

    # ── Print cluster assignments ─────────────────────────────────────────────
    print("\n=== Data-driven cluster assignments (k=6) ===")
    cluster_members: dict[int, list[str]] = {}
    for j, cl in enumerate(cluster_labels):
        cluster_members.setdefault(int(cl), []).append(param_names[j])
    for cl in sorted(cluster_members):
        members = cluster_members[cl]
        classes_in_cl = [param_class(p) for p in members]
        from collections import Counter
        dominant = Counter(classes_in_cl).most_common(1)[0][0]
        print(f"  Cluster {cl} (dominant={dominant}, n={len(members)}): {', '.join(members)}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(
        1, 2, figure=fig, left=0.06, right=0.97, top=0.92, bottom=0.08,
        wspace=0.35,
    )

    def _plot_heatmap(ax, order, title, strip_colors, strip_label=""):
        """Draw similarity heatmap for given parameter ordering."""
        S_ord = S[np.ix_(order, order)]
        pnames_ord = [param_names[i] for i in order]

        im = ax.imshow(
            S_ord, cmap="RdBu_r", vmin=-1, vmax=1,
            aspect="auto", interpolation="nearest",
        )
        ax.set_xticks(range(N_par))
        ax.set_yticks(range(N_par))
        ax.set_xticklabels(pnames_ord, rotation=90, fontsize=6.5)
        ax.set_yticklabels(pnames_ord, fontsize=6.5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

        # colour strip on left (y-axis)
        for i, col in enumerate(strip_colors):
            ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 1.0, 1.0,
                                        color=col, clip_on=False,
                                        transform=ax.transData))
        return im

    # Panel (a): hierarchical order
    strip_colors_hier = [CLUSTER_COLORS[cluster_labels[i] - 1] for i in hier_order]
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = _plot_heatmap(
        ax_a, hier_order,
        f"(a) Hierarchical clustering (k={k})",
        strip_colors_hier, "cluster",
    )

    # Panel (b): physical class order
    strip_colors_phys = [CLASS_COLORS.get(param_class(param_names[i]), "#888888")
                         for i in phys_order]
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = _plot_heatmap(
        ax_b, phys_order,
        "(b) Physical class ordering",
        strip_colors_phys, "class",
    )

    # Shared colorbar
    cbar = fig.colorbar(im_b, ax=[ax_a, ax_b], fraction=0.015, pad=0.01)
    cbar.set_label("Cosine similarity", fontsize=9)

    # Legend for panel (a): clusters
    legend_a = [mpatches.Patch(color=CLUSTER_COLORS[c - 1], label=f"Cluster {c}")
                for c in range(1, k + 1)]
    ax_a.legend(handles=legend_a, loc="upper right", fontsize=7, ncol=2,
                bbox_to_anchor=(1.0, 1.14), framealpha=0.85)

    # Legend for panel (b): physical classes
    legend_b = [mpatches.Patch(color=CLASS_COLORS.get(c, "#888"), label=c)
                for c in list(PARAMETER_CLASSES.keys()) + (["other"] if other_idxs else [])]
    ax_b.legend(handles=legend_b, loc="upper right", fontsize=7, ncol=2,
                bbox_to_anchor=(1.0, 1.14), framealpha=0.85)

    # Suptitle with agreement fraction
    fig.suptitle(
        f"Parameter clustering in mean-Jacobian space  "
        f"(cluster–class agreement: {agreement_frac:.1%})",
        fontsize=11,
    )

    for ext in ("pdf", "png"):
        path = FIG_DIR / f"fig_parameter_clustering.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[fig8] saved {path}")
    plt.close(fig)

    # ── JSON sidecar ──────────────────────────────────────────────────────────
    from collections import Counter
    out = {
        "k": k,
        "agreement_fraction": agreement_frac,
        "cluster_assignments": {
            param_names[j]: int(cluster_labels[j]) for j in range(N_par)
        },
        "cluster_members": {
            str(cl): cluster_members[cl] for cl in sorted(cluster_members)
        },
        "cluster_dominant_class": {},
    }
    for cl in sorted(cluster_members):
        classes_in_cl = [param_class(p) for p in cluster_members[cl]]
        dominant = Counter(classes_in_cl).most_common(1)[0][0]
        out["cluster_dominant_class"][str(cl)] = dominant

    json_path = OUT_DIR / "fig_parameter_clustering.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[fig8] wrote {json_path}")
    print(f"\n[fig8] cluster–class agreement: {agreement_frac:.1%}")
    print("[fig8] DONE")


if __name__ == "__main__":
    main()
