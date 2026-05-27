"""scatter/joint_struct_sweep.py — Phase 2 full 6-parameter sweep for §4 of Bind_joint_scatter.md.

Runs the same pipeline as joint_struct_pilot.py across all 6 CAMELS 1P arms:
  p1 = Omega_m,  p2 = sigma8,  p3 = A_SN1 (pilot arm — loaded from cache),
  p4 = A_AGN1,   p5 = A_SN2,   p6 = A_AGN2

For each arm: loads lo/hi sims → BIND inference (K=10) → LOWESS residuals →
Spearman correlation matrices → ΔC_T and ΔC_G.

Outputs:
  outputs/scatter_joint_structure/sweep_results.npz
  outputs/scatter_joint_structure/REPORT.md
  figures/scatter_joint_structure/fig_sweep_headline.pdf / .png
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
RUN_DIR   = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
TEST_1P   = Path("/mnt/home/mlee1/ceph/fm_testsuite/1P")
OUT_DIR   = BASE_DIR / "outputs/scatter_joint_structure"
FIG_DIR   = BASE_DIR / "figures/scatter_joint_structure"
FID_MAT   = BASE_DIR / "scatter/scatter_residual/matrices.npz"
PILOT_NPZ = OUT_DIR / "pilot_matrices.npz"
PROGRESS  = OUT_DIR / "SWEEP_PROGRESS.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SUB_DIR     = "snap_090/mass_threshold_1p000e13"
BOX_SIZE    = 50.0
N_PIX_FULL  = 1024
PATCH_PIX   = 128
MPC_PER_PIX = BOX_SIZE * PATCH_PIX / N_PIX_FULL / PATCH_PIX  # 0.048828125 Mpc/h/pix
NOISE_SEED  = 42
K_SAMPLES   = 10
N_STEPS     = 20
BATCH_SIZE  = 16
BOOT_B      = 2000
FRAC_LOWESS = 0.4

# 6 CAMELS 1P arms: (param_label, sim_lo, sim_hi, arm_idx_for_rng_seed)
ARMS = [
    ("Omega_m", "1P_p1_n2", "1P_p1_2"),
    ("sigma8",  "1P_p2_n2", "1P_p2_2"),
    ("A_SN1",   "1P_p3_n2", "1P_p3_2"),   # pilot arm — loaded from cache
    ("A_AGN1",  "1P_p4_n2", "1P_p4_2"),
    ("A_SN2",   "1P_p5_n2", "1P_p5_2"),
    ("A_AGN2",  "1P_p6_n2", "1P_p6_2"),
]

# ─────────────────────────────────────────────────────────────────────────────
# PYTHONPATH
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))

from data import NormStats, log_transform
from train import FlowMatchingLit
from scatter.measure_scatter import measure_scatter, ALL_OBS_NAMES, _compute_all_obs
from scatter.residual import (
    OBS_7,
    extract_obs8,
    fit_mean_and_scatter,
    standardise_residuals,
    residual_correlation_matrix,
)
from test_suite.pipeline import extract_periodic_cutout

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions (mirrored from joint_struct_pilot.py)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_params_fid(p_raw: np.ndarray, ns: NormStats) -> np.ndarray:
    _p = np.where(
        ns.param_log_flag == 1,
        np.log10(np.maximum(p_raw.astype(float), 1e-30)),
        p_raw.astype(float),
    )
    return ((_p - ns.param_min) / (ns.param_max - ns.param_min + 1e-8)).astype(np.float32)


def load_1p_sim(sim_name: str, ns: NormStats) -> dict | None:
    base = TEST_1P / sim_name / SUB_DIR
    cat_path = base / "halo_catalog.npz"
    cut_path = base / "halo_cutouts.npz"
    if not cat_path.exists() or not cut_path.exists():
        log(f"  MISSING: {base}")
        return None
    cat = np.load(cat_path, allow_pickle=True)
    cut = np.load(cut_path, allow_pickle=True)

    params = cat["params"].copy().astype(np.float32)
    params[:, 14] = 0.0   # CAMELS bug fix

    cond_raw = cut["condition"].astype(np.float32)    # (N, 128, 128)
    ls_raw   = cut["large_scale"].astype(np.float32)  # (N, 3, 128, 128)

    cond_norm = (log_transform(cond_raw) - ns.cond_mean) / (ns.cond_std + 1e-8)
    ls_norm   = (log_transform(ls_raw) - ns.ls_mean[:, None, None]) / (ns.ls_std[:, None, None] + 1e-8)

    radii   = cat["radii"] / 1000.0 / MPC_PER_PIX
    masses  = cat["halo_masses"].astype(np.float64)
    centers = cat["centers"].astype(np.float64)
    omega_m = params[:, 0].astype(np.float64)

    return {
        "cond_raw":  cond_raw,
        "cond_norm": cond_norm[:, np.newaxis],
        "ls_norm":   ls_norm,
        "params":    params,
        "masses":    masses,
        "radii_pix": radii,
        "omega_m":   omega_m,
        "centers":   centers,
        "N":         len(masses),
    }


def extract_truth_obs_1p(sim_name: str, data: dict) -> tuple[np.ndarray, np.ndarray]:
    from scatter.measure_scatter import OMEGA_B_FIXED, axis_ratio_q
    sim_dir  = TEST_1P / sim_name / "snap_090"
    fm_path  = sim_dir / "full_maps.npz"
    if not fm_path.exists():
        raise FileNotFoundError(f"full_maps.npz missing: {fm_path}")

    fm         = np.load(fm_path)
    truth_maps = fm["truth_maps"].astype(np.float32)  # (3, 1024, 1024)

    N      = data["N"]
    masses = data["masses"]
    radii  = data["radii_pix"]
    om     = data["omega_m"]
    centers = data["centers"]
    cond_raw = data["cond_raw"]

    f_b_arr   = OMEGA_B_FIXED / np.where(om > 0, om, np.nan)
    q_dmo_arr = np.full(N, np.nan, dtype=np.float64)
    for i in range(N):
        r_aper = max(min(float(radii[i]), PATCH_PIX / 2 - 2), 4.0)
        q_dmo_arr[i] = axis_ratio_q(np.maximum(cond_raw[i].astype(np.float64), 0.0), r_aper)

    N_obs     = len(ALL_OBS_NAMES)
    truth_obs = np.full((N, N_obs), np.nan, dtype=np.float32)

    for i in range(N):
        cx_pix = int(float(centers[i, 0]) / BOX_SIZE * N_PIX_FULL) % N_PIX_FULL
        cy_pix = int(float(centers[i, 1]) / BOX_SIZE * N_PIX_FULL) % N_PIX_FULL

        patches = np.stack([
            extract_periodic_cutout(truth_maps[ch], cx_pix, cy_pix, PATCH_PIX)
            for ch in range(3)
        ])  # (3, 128, 128)

        truth_obs[i] = _compute_all_obs(
            patches,
            float(radii[i]),
            float(f_b_arr[i]),
            float(q_dmo_arr[i]),
        )

    return truth_obs, q_dmo_arr


def run_bind_endpoint(sim_name: str, data: dict, ns: NormStats, model_fm, device_str: str) -> np.ndarray:
    theta_norm = normalize_params_fid(data["params"][0], ns)
    result = measure_scatter(
        model_fm   = model_fm,
        norm_stats = ns,
        theta_norm = theta_norm,
        dmo_conds  = data["cond_norm"],
        ls_conds   = data["ls_norm"],
        masses     = data["masses"],
        r200_pix   = data["radii_pix"],
        K          = K_SAMPLES,
        n_steps    = N_STEPS,
        device     = device_str,
        batch_size = BATCH_SIZE,
        dmo_raw    = data["cond_raw"],
        omega_m    = data["omega_m"],
        seed       = NOISE_SEED,
    )
    return result["obs_tensor"]  # (N, K, 16)


def compute_endpoint_matrices(
    masses: np.ndarray,
    truth_raw: np.ndarray,
    bind_obs: np.ndarray,
    rng_seed: int = 0,
) -> dict:
    log_mass = np.log10(masses)
    n_obs7   = len(OBS_7)

    obs8_T      = extract_obs8(truth_raw, list(ALL_OBS_NAMES))[:, :n_obs7]
    bind_mean_raw = bind_obs.mean(axis=1)
    obs8_G      = extract_obs8(bind_mean_raw, list(ALL_OBS_NAMES))[:, :n_obs7]

    delta_T = np.full((len(masses), n_obs7), np.nan)
    delta_G = np.full((len(masses), n_obs7), np.nan)

    for a in range(n_obs7):
        f_T = obs8_T[:, a]
        f_G = obs8_G[:, a]

        ms_T = fit_mean_and_scatter(log_mass, f_T, log_mass, f_G, frac=FRAC_LOWESS, fit_source="truth")
        delta_T[:, a] = standardise_residuals(log_mass, f_T, ms_T.mu, ms_T.sigma)

        ms_G = fit_mean_and_scatter(log_mass, f_T, log_mass, f_G, frac=FRAC_LOWESS, fit_source="bind")
        delta_G[:, a] = standardise_residuals(log_mass, f_G, ms_G.mu, ms_G.sigma)

    C_T, SE_T = residual_correlation_matrix(delta_T, method="spearman", n_boot=BOOT_B, rng_seed=rng_seed)
    C_G, SE_G = residual_correlation_matrix(delta_G, method="spearman", n_boot=BOOT_B, rng_seed=rng_seed + 1000)

    return {"C_T": C_T, "SE_T": SE_T, "C_G": C_G, "SE_G": SE_G,
            "delta_T": delta_T, "delta_G": delta_G,
            "obs7_T": obs8_T, "obs7_G": obs8_G}


# ─────────────────────────────────────────────────────────────────────────────
# Headline figure
# ─────────────────────────────────────────────────────────────────────────────
OBS7_SHORT = [r"$M_{\rm DM}$", r"$M_{\rm gas}$", r"$M_\star$",
              r"$\Sigma_{\rm gas,c}$", r"$q_{\rm DM}$", r"$q_{\rm gas}$", r"$q_\star$"]
PARAM_LABELS = [r"$\Omega_m$", r"$\sigma_8$", r"$A_{\rm SN1}$",
                r"$A_{\rm AGN1}$", r"$A_{\rm SN2}$", r"$A_{\rm AGN2}$"]
PARAM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _offdiag_flat(mat: np.ndarray) -> np.ndarray:
    """Return upper-triangle off-diagonal entries as flat array (21 for 7×7)."""
    idx = np.triu_indices(7, k=1)
    return mat[idx]


def make_sweep_figure(arm_results: list[dict], param_names: list[str]) -> None:
    """
    Panel A (top): 6 scatter subplots (one per param), ΔC_T vs ΔC_G, 21 pairs each.
    Panel B (bottom-left): summary scatter ||ΔC_T||_F vs ||ΔC_G||_F per param.
    Panel C (bottom-right): pooled ΔC_G ~ ΔC_T with R².
    """
    n_params = len(arm_results)
    fig = plt.figure(figsize=(16, 10))
    gs_main = gridspec.GridSpec(2, 1, figure=fig, hspace=0.42, height_ratios=[1.2, 1.0])

    # ── Top row: 6 per-param scatter panels ──────────────────────────────────
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_main[0], wspace=0.28)

    # Collect all offdiag entries pooled
    all_DC_T_pool = []
    all_DC_G_pool = []
    frob_T = []
    frob_G = []

    for pi, (arm, pname) in enumerate(zip(arm_results, param_names)):
        DC_T_flat = _offdiag_flat(arm["DC_T"])  # (21,)
        DC_G_flat = _offdiag_flat(arm["DC_G"])  # (21,)

        all_DC_T_pool.append(DC_T_flat)
        all_DC_G_pool.append(DC_G_flat)

        frob_T.append(np.sqrt(np.nansum(arm["DC_T"]**2)))
        frob_G.append(np.sqrt(np.nansum(arm["DC_G"]**2)))

        ax = fig.add_subplot(gs_top[pi])
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.8, alpha=0.5)

        # Colour by obs pair type: mass-mass, mass-shape, shape-shape
        pair_colors = []
        pair_types  = []
        for ia, ib in zip(*np.triu_indices(7, k=1)):
            if ia < 4 and ib < 4:
                pair_colors.append("#1f77b4"); pair_types.append("mass-mass")
            elif ia >= 4 and ib >= 4:
                pair_colors.append("#d62728"); pair_types.append("shape-shape")
            else:
                pair_colors.append("#ff7f0e"); pair_types.append("mass-shape")

        scatter = ax.scatter(DC_T_flat, DC_G_flat, c=pair_colors, s=20, alpha=0.8, zorder=3)

        # R² within-panel
        mask = np.isfinite(DC_T_flat) & np.isfinite(DC_G_flat)
        if mask.sum() >= 4:
            r, p = scipy_stats.pearsonr(DC_T_flat[mask], DC_G_flat[mask])
            r2 = r**2
            ax.text(0.05, 0.93, f"$R^2={r2:.2f}$", transform=ax.transAxes, fontsize=7, va="top")

        ax.set_title(PARAM_LABELS[pi], fontsize=9, pad=3)
        ax.set_xlabel(r"$\Delta C_{\rm truth}$", fontsize=7)
        if pi == 0:
            ax.set_ylabel(r"$\Delta C_{\rm BIND}$", fontsize=7)
        ax.tick_params(labelsize=6)
        lim = max(0.8, np.nanmax(np.abs(np.concatenate([DC_T_flat, DC_G_flat]))) * 1.1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="datalim")

    # Legend for pair types (add to first panel)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=6, label='mass–mass'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=6, label='mass–shape'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=6, label='shape–shape'),
    ]
    fig.axes[0].legend(handles=legend_elements, fontsize=6, loc="lower right")

    # ── Bottom row: Frobenius scatter + pooled scatter ─────────────────────
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], wspace=0.38)

    # Panel B: ||ΔC_T||_F vs ||ΔC_G||_F per param
    ax_b = fig.add_subplot(gs_bot[0])
    frob_T = np.array(frob_T)
    frob_G = np.array(frob_G)
    for pi, (fT, fG) in enumerate(zip(frob_T, frob_G)):
        ax_b.scatter(fT, fG, color=PARAM_COLORS[pi], s=60, zorder=3)
        ax_b.annotate(PARAM_LABELS[pi], (fT, fG), xytext=(4, 3),
                      textcoords="offset points", fontsize=7)
    lim_b = max(frob_T.max(), frob_G.max()) * 1.15
    ax_b.plot([0, lim_b], [0, lim_b], "k--", lw=0.8, alpha=0.5)
    ax_b.set_xlabel(r"$||\Delta C_{\rm truth}||_F$", fontsize=9)
    ax_b.set_ylabel(r"$||\Delta C_{\rm BIND}||_F$", fontsize=9)
    ax_b.set_title("Frobenius norm per parameter", fontsize=9)
    ax_b.tick_params(labelsize=7)
    ax_b.set_xlim(0, lim_b)
    ax_b.set_ylim(0, lim_b)

    # Panel C: pooled ΔC_G ~ ΔC_T, colored by param
    ax_c = fig.add_subplot(gs_bot[1])
    ax_c.axhline(0, color="gray", lw=0.5, ls="--")
    ax_c.axvline(0, color="gray", lw=0.5, ls="--")
    ax_c.plot([-1, 1], [-1, 1], "k--", lw=0.8, alpha=0.5)

    all_T_pooled = np.concatenate(all_DC_T_pool)
    all_G_pooled = np.concatenate(all_DC_G_pool)
    for pi in range(n_params):
        ax_c.scatter(all_DC_T_pool[pi], all_DC_G_pool[pi],
                     color=PARAM_COLORS[pi], s=15, alpha=0.7, label=PARAM_LABELS[pi])

    mask = np.isfinite(all_T_pooled) & np.isfinite(all_G_pooled)
    if mask.sum() >= 10:
        r, p = scipy_stats.pearsonr(all_T_pooled[mask], all_G_pooled[mask])
        slope, intercept, _, _, _ = scipy_stats.linregress(all_T_pooled[mask], all_G_pooled[mask])
        xs = np.linspace(-1.0, 1.0, 100)
        ax_c.plot(xs, slope * xs + intercept, "k-", lw=1.5, alpha=0.7)
        ax_c.text(0.05, 0.93, f"$R^2={r**2:.3f}$, slope={slope:.2f}",
                  transform=ax_c.transAxes, fontsize=8, va="top")

    ax_c.legend(fontsize=6, loc="lower right", ncol=2)
    ax_c.set_xlabel(r"$\Delta C_{\rm truth}$ (all pairs, all params)", fontsize=9)
    ax_c.set_ylabel(r"$\Delta C_{\rm BIND}$", fontsize=9)
    ax_c.set_title("Pooled sweep: BIND tracks truth", fontsize=9)
    ax_c.tick_params(labelsize=7)
    lim_c = max(0.8, np.nanmax(np.abs(np.concatenate([all_T_pooled, all_G_pooled]))) * 1.1)
    ax_c.set_xlim(-lim_c, lim_c)
    ax_c.set_ylim(-lim_c, lim_c)

    fig.suptitle("BIND joint scatter structure: parameter sweep (1P lo→hi endpoints)",
                 fontsize=11, y=0.99)

    out_pdf = FIG_DIR / "fig_sweep_headline.pdf"
    out_png = FIG_DIR / "fig_sweep_headline.png"
    fig.savefig(str(out_pdf), bbox_inches="tight", dpi=150)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)
    log(f"Saved figure: {out_pdf}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def write_report(arm_results: list[dict], param_names: list[str]) -> None:
    all_T = np.concatenate([_offdiag_flat(r["DC_T"]) for r in arm_results])
    all_G = np.concatenate([_offdiag_flat(r["DC_G"]) for r in arm_results])

    mask = np.isfinite(all_T) & np.isfinite(all_G)
    r_pooled, _ = scipy_stats.pearsonr(all_T[mask], all_G[mask])
    slope, intercept, _, _, _ = scipy_stats.linregress(all_T[mask], all_G[mask])

    sign_match  = int(np.sum(np.sign(all_T[mask]) == np.sign(all_G[mask])))
    sign_total  = int(mask.sum())

    lines = ["# REPORT: BIND Joint Scatter Structure Sweep\n",
             f"Generated: {datetime.datetime.now().isoformat()}\n",
             "\n## Summary\n",
             f"- Parameters: {', '.join(param_names)}\n",
             f"- Pairs per param (off-diagonal 7×7): 21\n",
             f"- Total pooled pairs: {sign_total}\n",
             f"- Pooled R²(ΔC_G ~ ΔC_T): {r_pooled**2:.4f}\n",
             f"- Slope: {slope:.3f} (ideal=1.0), intercept={intercept:.3f}\n",
             f"- Sign agreement: {sign_match}/{sign_total} ({100*sign_match/sign_total:.1f}%)\n",
             "\n## Per-parameter results\n",
             "| Param | n_halos_lo | n_halos_hi | ||ΔC_T||_F | ||ΔC_G||_F | ratio | n_sig(|z|>2) |\n",
             "|-------|-----------|-----------|-----------|-----------|-------|---------------|\n"]

    for arm, pname in zip(arm_results, param_names):
        DC_T_flat = _offdiag_flat(arm["DC_T"])
        DC_G_flat = _offdiag_flat(arm["DC_G"])
        SE_T_flat = _offdiag_flat(arm.get("SE_T_lo", arm.get("SE_T", np.zeros((7,7)))))
        SE_T_hi_flat = _offdiag_flat(arm.get("SE_T_hi", arm.get("SE_T", np.zeros((7,7)))))

        fT = np.sqrt(np.nansum(arm["DC_T"]**2))
        fG = np.sqrt(np.nansum(arm["DC_G"]**2))
        ratio = fG / (fT + 1e-8)
        SE_pooled = np.sqrt(SE_T_flat**2 + SE_T_hi_flat**2)
        z = DC_T_flat / (SE_pooled + 1e-6)
        n_sig = int(np.sum(np.abs(z) > 2))

        lines.append(f"| {pname} | {arm.get('n_lo', '?')} | {arm.get('n_hi', '?')} | "
                     f"{fT:.4f} | {fG:.4f} | {ratio:.3f} | {n_sig} |\n")

    lines += ["\n## Verdict\n"]
    r2 = r_pooled**2
    if r2 >= 0.5:
        verdict = "STRONG: BIND faithfully reproduces parameter-driven shifts in joint scatter structure."
    elif r2 >= 0.25:
        verdict = "MODERATE: BIND partially tracks parameter-driven shifts. Some pairs tracked, others not."
    else:
        verdict = "WEAK: BIND does not clearly track parameter-driven shifts in joint scatter structure."

    lines.append(f"R²={r2:.4f} → {verdict}\n\n")
    lines.append("### Significant individual entries across all params:\n")
    lines.append("| Param | obs_a | obs_b | ΔC_T | ΔC_G | z | ratio | sign_ok |\n")
    lines.append("|-------|-------|-------|------|------|---|-------|--------|\n")

    for arm, pname in zip(arm_results, param_names):
        DC_T_flat = _offdiag_flat(arm["DC_T"])
        DC_G_flat = _offdiag_flat(arm["DC_G"])
        SE_T_lo_flat = _offdiag_flat(arm.get("SE_T_lo", arm.get("SE_T", np.zeros((7,7)))))
        SE_T_hi_flat = _offdiag_flat(arm.get("SE_T_hi", arm.get("SE_T", np.zeros((7,7)))))
        SE_pooled = np.sqrt(SE_T_lo_flat**2 + SE_T_hi_flat**2)
        z_flat = DC_T_flat / (SE_pooled + 1e-6)

        pairs = list(zip(*np.triu_indices(7, k=1)))
        for k, (ia, ib) in enumerate(pairs):
            if abs(z_flat[k]) > 2:
                ratio_k = DC_G_flat[k] / (DC_T_flat[k] + 1e-8) if abs(DC_T_flat[k]) > 0.01 else float("nan")
                sign_ok = bool(np.sign(DC_T_flat[k]) == np.sign(DC_G_flat[k]))
                ratio_str = f"{ratio_k:.2f}" if not np.isnan(ratio_k) else "nan"
                lines.append(f"| {pname} | {OBS_7[ia]} | {OBS_7[ib]} | "
                              f"{DC_T_flat[k]:.3f} | {DC_G_flat[k]:.3f} | "
                              f"{z_flat[k]:.2f} | {ratio_str} | "
                              f"{'✓' if sign_ok else '✗'} |\n")

    report_path = OUT_DIR / "REPORT.md"
    report_path.write_text("".join(lines))
    log(f"Saved REPORT.md: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    log("=== joint_struct_sweep.py START ===")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device = {device_str}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = RUN_DIR / "checkpoints/last.ckpt"
    ns   = NormStats.load(RUN_DIR / "norm_stats.npz")
    if "param_log_flag" not in np.load(RUN_DIR / "norm_stats.npz").files:
        ns.param_log_flag = np.zeros(35, dtype=np.float32)
    log(f"Loading model from {ckpt}")
    lit = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device_str)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm
    model_fm.model.eval()
    log("Model loaded")

    # ── Load CV fiducial ─────────────────────────────────────────────────────
    fid = np.load(FID_MAT)
    fid_C_T = fid["C_T"]
    fid_C_G = fid["C_G"]
    fid_SE_T = fid["SE_T"]
    fid_SE_G = fid["SE_G"]
    log(f"CV fiducial matrices loaded")

    # ── Loop over 6 arms ─────────────────────────────────────────────────────
    arm_results = []

    for arm_idx, (pname, sim_lo, sim_hi) in enumerate(ARMS):
        log(f"\n{'='*60}")
        log(f"ARM {arm_idx+1}/6: {pname}  ({sim_lo} → {sim_hi})")
        log(f"{'='*60}")

        # ── Check if A_SN1 pilot already done ─────────────────────────────
        if pname == "A_SN1" and PILOT_NPZ.exists():
            log(f"  Loading A_SN1 from pilot cache: {PILOT_NPZ}")
            pilot = np.load(str(PILOT_NPZ))
            result = {
                "pname": pname,
                "sim_lo": sim_lo,
                "sim_hi": sim_hi,
                "DC_T":   pilot["DC_T"],
                "DC_G":   pilot["DC_G"],
                "C_T_lo": pilot["C_T_lo"],
                "C_T_hi": pilot["C_T_hi"],
                "C_G_lo": pilot["C_G_lo"],
                "C_G_hi": pilot["C_G_hi"],
                "SE_T_lo": pilot["SE_T_lo"],
                "SE_T_hi": pilot["SE_T_hi"],
                "SE_G_lo": pilot["SE_G_lo"],
                "SE_G_hi": pilot["SE_G_hi"],
                "SE_T": pilot["SE_T_lo"],  # alias for report
                "n_lo": 49,
                "n_hi": 49,
            }
            arm_results.append(result)
            log(f"  Loaded from cache: ||ΔC_T||_F={np.sqrt(np.nansum(result['DC_T']**2)):.4f}, "
                f"||ΔC_G||_F={np.sqrt(np.nansum(result['DC_G']**2)):.4f}")
            continue

        # ── Load sims ─────────────────────────────────────────────────────
        log(f"  Loading {sim_lo} ...")
        data_lo = load_1p_sim(sim_lo, ns)
        log(f"  Loading {sim_hi} ...")
        data_hi = load_1p_sim(sim_hi, ns)

        if data_lo is None or data_hi is None:
            log(f"  SKIP: missing sim data")
            continue

        log(f"  {sim_lo}: N={data_lo['N']}, masses=[{data_lo['masses'].min():.2e}, {data_lo['masses'].max():.2e}]")
        log(f"  {sim_hi}: N={data_hi['N']}, masses=[{data_hi['masses'].min():.2e}, {data_hi['masses'].max():.2e}]")

        # ── Extract truth observables ──────────────────────────────────────
        log(f"  Extracting truth observables ...")
        truth_lo, _ = extract_truth_obs_1p(sim_lo, data_lo)
        truth_hi, _ = extract_truth_obs_1p(sim_hi, data_hi)
        log(f"  truth_lo: {truth_lo.shape}, NaN frac: {np.isnan(truth_lo).mean():.3f}")
        log(f"  truth_hi: {truth_hi.shape}, NaN frac: {np.isnan(truth_hi).mean():.3f}")

        # ── BIND inference ─────────────────────────────────────────────────
        log(f"  BIND inference at lo ({sim_lo}) ...")
        bind_lo = run_bind_endpoint(sim_lo, data_lo, ns, model_fm, device_str)
        log(f"    bind_lo: {bind_lo.shape}")

        log(f"  BIND inference at hi ({sim_hi}) ...")
        bind_hi = run_bind_endpoint(sim_hi, data_hi, ns, model_fm, device_str)
        log(f"    bind_hi: {bind_hi.shape}")

        # ── Compute endpoint matrices ──────────────────────────────────────
        log(f"  Computing LOWESS + Spearman matrices (B={BOOT_B}) ...")
        seed_base = arm_idx * 100
        ep_lo = compute_endpoint_matrices(data_lo["masses"], truth_lo, bind_lo, rng_seed=seed_base)
        ep_hi = compute_endpoint_matrices(data_hi["masses"], truth_hi, bind_hi, rng_seed=seed_base + 10)

        DC_T = ep_hi["C_T"] - ep_lo["C_T"]
        DC_G = ep_hi["C_G"] - ep_lo["C_G"]

        log(f"  ||ΔC_T||_F={np.sqrt(np.nansum(DC_T**2)):.4f}, ||ΔC_G||_F={np.sqrt(np.nansum(DC_G**2)):.4f}")

        # Save per-arm npz
        arm_npz = OUT_DIR / f"sweep_arm_{pname}.npz"
        np.savez_compressed(
            str(arm_npz),
            C_T_lo=ep_lo["C_T"], SE_T_lo=ep_lo["SE_T"],
            C_G_lo=ep_lo["C_G"], SE_G_lo=ep_lo["SE_G"],
            C_T_hi=ep_hi["C_T"], SE_T_hi=ep_hi["SE_T"],
            C_G_hi=ep_hi["C_G"], SE_G_hi=ep_hi["SE_G"],
            DC_T=DC_T, DC_G=DC_G,
        )
        log(f"  Saved {arm_npz.name}")

        result = {
            "pname":   pname,
            "sim_lo":  sim_lo,
            "sim_hi":  sim_hi,
            "DC_T":    DC_T,
            "DC_G":    DC_G,
            "C_T_lo":  ep_lo["C_T"],
            "C_T_hi":  ep_hi["C_T"],
            "C_G_lo":  ep_lo["C_G"],
            "C_G_hi":  ep_hi["C_G"],
            "SE_T_lo": ep_lo["SE_T"],
            "SE_T_hi": ep_hi["SE_T"],
            "SE_G_lo": ep_lo["SE_G"],
            "SE_G_hi": ep_hi["SE_G"],
            "SE_T":    ep_lo["SE_T"],  # alias
            "n_lo":    int(data_lo["N"]),
            "n_hi":    int(data_hi["N"]),
        }
        arm_results.append(result)

    if len(arm_results) < 2:
        log("ERROR: fewer than 2 arm results — aborting")
        return

    # ── Save aggregated results ────────────────────────────────────────────
    log("\nSaving sweep_results.npz ...")
    param_names = [r["pname"] for r in arm_results]
    sweep_npz = OUT_DIR / "sweep_results.npz"
    save_dict = {}
    for r in arm_results:
        p = r["pname"]
        save_dict[f"DC_T_{p}"] = r["DC_T"]
        save_dict[f"DC_G_{p}"] = r["DC_G"]
        save_dict[f"C_T_lo_{p}"] = r["C_T_lo"]
        save_dict[f"C_T_hi_{p}"] = r["C_T_hi"]
        save_dict[f"C_G_lo_{p}"] = r["C_G_lo"]
        save_dict[f"C_G_hi_{p}"] = r["C_G_hi"]
        save_dict[f"SE_T_lo_{p}"] = r["SE_T_lo"]
        save_dict[f"SE_T_hi_{p}"] = r["SE_T_hi"]
    save_dict["obs_names"] = np.array(OBS_7)
    save_dict["param_names"] = np.array(param_names)
    save_dict["C_T_fid"] = fid_C_T
    save_dict["C_G_fid"] = fid_C_G
    np.savez_compressed(str(sweep_npz), **save_dict)
    log(f"Saved: {sweep_npz}")

    # ── Headline figure ────────────────────────────────────────────────────
    log("Making headline figure ...")
    make_sweep_figure(arm_results, param_names)

    # ── Report ────────────────────────────────────────────────────────────
    log("Writing REPORT.md ...")
    write_report(arm_results, param_names)

    # ── Quick pooled R² summary ────────────────────────────────────────────
    all_T = np.concatenate([_offdiag_flat(r["DC_T"]) for r in arm_results])
    all_G = np.concatenate([_offdiag_flat(r["DC_G"]) for r in arm_results])
    mask  = np.isfinite(all_T) & np.isfinite(all_G)
    r, _  = scipy_stats.pearsonr(all_T[mask], all_G[mask])
    slope, intercept, _, _, _ = scipy_stats.linregress(all_T[mask], all_G[mask])
    log(f"\n=== SWEEP SUMMARY ===")
    log(f"Parameters: {', '.join(param_names)}")
    log(f"Pooled R²(ΔC_G ~ ΔC_T) = {r**2:.4f}")
    log(f"Slope = {slope:.3f}, intercept = {intercept:.3f}")
    sign_match = int(np.sum(np.sign(all_T[mask]) == np.sign(all_G[mask])))
    log(f"Sign agreement: {sign_match}/{mask.sum()} ({100*sign_match/mask.sum():.1f}%)")
    for r_arm in arm_results:
        fT = np.sqrt(np.nansum(r_arm["DC_T"]**2))
        fG = np.sqrt(np.nansum(r_arm["DC_G"]**2))
        log(f"  {r_arm['pname']:8s}: ||ΔC_T||_F={fT:.3f}, ||ΔC_G||_F={fG:.3f}, ratio={fG/(fT+1e-8):.3f}")

    log(f"=== joint_struct_sweep.py DONE in {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
