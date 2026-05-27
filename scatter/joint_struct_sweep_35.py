"""scatter/joint_struct_sweep_35.py — Full 35-parameter sweep for §4 of Bind_joint_scatter.md.

Extends joint_struct_sweep.py from 6 to all 35 CAMELS SB35 parameters.
Arms p1–p6 are loaded from existing per-arm cache files (already computed).
Arms p7–p35 are run fresh.

Usage:
    python scatter/joint_struct_sweep_35.py [--resume] [--start N]

    --resume    skip arms whose sweep35_arm_{pname}.npz already exists
    --start N   start from arm index N (1-based), for re-running after crash

Outputs:
    outputs/scatter_joint_structure/sweep35_arm_{pname}.npz   (per-arm)
    outputs/scatter_joint_structure/sweep35_results.npz        (all 35 arms)
    outputs/scatter_joint_structure/REPORT35.md
    figures/scatter_joint_structure/fig_sweep35_headline.pdf/.png
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
PROGRESS  = OUT_DIR / "SWEEP35_PROGRESS.log"
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

# ─────────────────────────────────────────────────────────────────────────────
# All 35 arms: (param_name, sim_lo, sim_hi)
# Notes on non-standard endpoints:
#   p15 (WindSpecMom, idx 14): only _1.._4 exist (no n-variants); _1=lo, _4=hi
#       ALSO: must NOT zero params[:,14] for this arm (it IS the varied index)
#   p29 (UVB_H0_beta, idx 28): _3 is hi extreme; _n1 is lo extreme
#   p30 (UVB_H0_Dz, idx 29): only n1, n2 have catalogs; use n2=lo, n1=hi
#   p32 (UVB_Hep_Dz, idx 31): only n-variants; use n4=lo, n1=hi
# ─────────────────────────────────────────────────────────────────────────────
ARMS = [
    # p1-p6: first 6 CAMELS parameters (cached from 6-param sweep)
    ("Omega_m",              "1P_p1_n2",  "1P_p1_2"),   # idx 0
    ("sigma8",               "1P_p2_n2",  "1P_p2_2"),   # idx 1
    ("A_SN1",                "1P_p3_n2",  "1P_p3_2"),   # idx 2 — pilot cache
    ("A_AGN1",               "1P_p4_n2",  "1P_p4_2"),   # idx 3
    ("A_SN2",                "1P_p5_n2",  "1P_p5_2"),   # idx 4
    ("A_AGN2",               "1P_p6_n2",  "1P_p6_2"),   # idx 5
    # p7-p9: remaining cosmological
    ("Omega_b",              "1P_p7_n2",  "1P_p7_2"),   # idx 6
    ("H0",                   "1P_p8_n2",  "1P_p8_2"),   # idx 7
    ("n_s",                  "1P_p9_n2",  "1P_p9_2"),   # idx 8
    # p10-p14: stellar feedback / IMF
    ("MaxSfr",               "1P_p10_n2", "1P_p10_2"),  # idx 9
    ("SoftEQS",              "1P_p11_n2", "1P_p11_2"),  # idx 10
    ("IMFslope",             "1P_p12_n2", "1P_p12_2"),  # idx 11
    ("SNII_MinMass",         "1P_p13_n2", "1P_p13_2"),  # idx 12
    ("ThermalWind",          "1P_p14_n2", "1P_p14_2"),  # idx 13
    ("WindSpecMom",          "1P_p15_1",  "1P_p15_4"),  # idx 14 — SPECIAL
    # p16-p21: wind parameters
    ("WindFreeTravelDens",   "1P_p16_n2", "1P_p16_2"),  # idx 15
    ("MinWindVel",           "1P_p17_n2", "1P_p17_2"),  # idx 16
    ("WindEnergyReduction",  "1P_p18_n2", "1P_p18_2"),  # idx 17
    ("WindEnergyReductionZ", "1P_p19_n2", "1P_p19_2"),  # idx 18
    ("WindEnergyReductionExp","1P_p20_n2","1P_p20_2"),  # idx 19
    ("WindDumpFac",          "1P_p21_n2", "1P_p21_2"),  # idx 20
    # p22-p28: black hole / AGN parameters
    ("SeedBHMass",           "1P_p22_n2", "1P_p22_2"),  # idx 21
    ("BHAccretion",          "1P_p23_n2", "1P_p23_2"),  # idx 22
    ("BHEddington",          "1P_p24_n2", "1P_p24_2"),  # idx 23
    ("BHFeedback",           "1P_p25_n2", "1P_p25_2"),  # idx 24
    ("BHRadEff",             "1P_p26_n2", "1P_p26_2"),  # idx 25
    ("QuasarThreshold",      "1P_p27_n2", "1P_p27_2"),  # idx 26
    ("QuasarThreshPow",      "1P_p28_n2", "1P_p28_2"),  # idx 27
    # p29-p32: UV background parameters (non-standard endpoints)
    ("UVB_H0_beta",          "1P_p29_n1", "1P_p29_3"),  # idx 28
    ("UVB_H0_Dz",            "1P_p30_n2", "1P_p30_n1"), # idx 29 — only neg
    ("UVB_Hep_beta",         "1P_p31_n2", "1P_p31_2"),  # idx 30
    ("UVB_Hep_Dz",           "1P_p32_n4", "1P_p32_n1"), # idx 31 — only neg
    # p33-p35: SN Ia / softening
    ("SNIa_norm",            "1P_p33_n2", "1P_p33_2"),  # idx 32
    ("SNIa_DTD_pow",         "1P_p34_n2", "1P_p34_2"),  # idx 33
    ("SofteningComoving",    "1P_p35_n2", "1P_p35_2"),  # idx 34
]

# Parameters whose idx14 zero-fix must be skipped (they ARE idx 14)
NO_IDX14_ZERO = {"WindSpecMom"}

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
# Utility functions (same as joint_struct_sweep.py)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_params_fid(p_raw: np.ndarray, ns: NormStats) -> np.ndarray:
    _p = np.where(
        ns.param_log_flag == 1,
        np.log10(np.maximum(p_raw.astype(float), 1e-30)),
        p_raw.astype(float),
    )
    return ((_p - ns.param_min) / (ns.param_max - ns.param_min + 1e-8)).astype(np.float32)


def load_1p_sim(sim_name: str, ns: NormStats, pname: str = "") -> dict | None:
    base = TEST_1P / sim_name / SUB_DIR
    cat_path = base / "halo_catalog.npz"
    cut_path = base / "halo_cutouts.npz"
    if not cat_path.exists() or not cut_path.exists():
        log(f"  MISSING: {base}")
        return None
    cat = np.load(cat_path, allow_pickle=True)
    cut = np.load(cut_path, allow_pickle=True)

    params = cat["params"].copy().astype(np.float32)
    if pname not in NO_IDX14_ZERO:
        params[:, 14] = 0.0   # CAMELS bug fix (skip for WindSpecMom arm)

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

    return {"C_T": C_T, "SE_T": SE_T, "C_G": C_G, "SE_G": SE_G}


def _offdiag_flat(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(7, k=1)
    return mat[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Headline figure: Frobenius bar chart + pooled scatter
# ─────────────────────────────────────────────────────────────────────────────
def make_sweep35_figure(arm_results: list[dict], param_names: list[str]) -> None:
    n_arms = len(arm_results)
    fT_vals = [np.sqrt(np.nansum(r["DC_T"]**2)) for r in arm_results]
    fG_vals = [np.sqrt(np.nansum(r["DC_G"]**2)) for r in arm_results]

    # Pooled scatter
    all_T = np.concatenate([_offdiag_flat(r["DC_T"]) for r in arm_results])
    all_G = np.concatenate([_offdiag_flat(r["DC_G"]) for r in arm_results])
    mask  = np.isfinite(all_T) & np.isfinite(all_G)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Frobenius bar chart
    ax = axes[0]
    x = np.arange(n_arms)
    w = 0.35
    ax.bar(x - w/2, fT_vals, width=w, label="||ΔC_T||_F (truth)", color="#2ca02c", alpha=0.8)
    ax.bar(x + w/2, fG_vals, width=w, label="||ΔC_G||_F (BIND)",  color="#1f77b4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=90, fontsize=7)
    ax.set_ylabel("Frobenius norm")
    ax.set_title("Per-parameter Frobenius norm")
    ax.legend(fontsize=8)

    # Right: pooled ΔC_G vs ΔC_T scatter
    ax = axes[1]
    ax.scatter(all_T[mask], all_G[mask], s=4, alpha=0.3, color="steelblue")
    lim = max(0.8, float(np.nanmax(np.abs(np.concatenate([all_T[mask], all_G[mask]])))) * 1.1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axline((0, 0), slope=1.0, color="k", lw=0.8, ls="--", label="y=x")
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)

    if mask.sum() > 2:
        slope, intercept, r, _, _ = scipy_stats.linregress(all_T[mask], all_G[mask])
        x_fit = np.array([-lim, lim])
        ax.plot(x_fit, slope * x_fit + intercept, color="red", lw=1.2,
                label=f"fit: R²={r**2:.3f}, slope={slope:.2f}")

    ax.set_xlabel("ΔC_T (truth)")
    ax.set_ylabel("ΔC_G (BIND)")
    ax.set_title("Pooled ΔC off-diag: BIND vs truth (all 35 params)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"fig_sweep35_headline.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved figure: {FIG_DIR}/fig_sweep35_headline.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def write_report35(arm_results: list[dict], param_names: list[str]) -> None:
    all_T = np.concatenate([_offdiag_flat(r["DC_T"]) for r in arm_results])
    all_G = np.concatenate([_offdiag_flat(r["DC_G"]) for r in arm_results])
    mask  = np.isfinite(all_T) & np.isfinite(all_G)

    r_pooled, _ = scipy_stats.pearsonr(all_T[mask], all_G[mask])
    slope, intercept, _, _, _ = scipy_stats.linregress(all_T[mask], all_G[mask])
    sign_match = int(np.sum(np.sign(all_T[mask]) == np.sign(all_G[mask])))
    sign_total = int(mask.sum())

    lines = ["# REPORT: BIND Joint Scatter Structure Sweep — 35 Parameters\n",
             f"Generated: {datetime.datetime.now().isoformat()}\n",
             "\n## Summary\n",
             f"- Parameters: all 35 CAMELS SB35 params (p1–p35)\n",
             f"- Pairs per param (off-diagonal 7×7): 21\n",
             f"- Total pooled pairs: {sign_total}\n",
             f"- Pooled R²(ΔC_G ~ ΔC_T): {r_pooled**2:.4f}\n",
             f"- Slope: {slope:.3f} (ideal=1.0), intercept={intercept:.3f}\n",
             f"- Sign agreement: {sign_match}/{sign_total} ({100*sign_match/sign_total:.1f}%)\n",
             "\n## Per-parameter results\n",
             "| Param | n_lo | n_hi | ||ΔC_T||_F | ||ΔC_G||_F | ratio | n_sig(|z|>2) |\n",
             "|-------|------|------|-----------|-----------|-------|---------------|\n"]

    for arm, pname in zip(arm_results, param_names):
        DC_T_flat = _offdiag_flat(arm["DC_T"])
        DC_G_flat = _offdiag_flat(arm["DC_G"])
        SE_lo = _offdiag_flat(arm.get("SE_T_lo", arm.get("SE_T", np.zeros((7,7)))))
        SE_hi = _offdiag_flat(arm.get("SE_T_hi", arm.get("SE_T", np.zeros((7,7)))))

        fT = np.sqrt(np.nansum(arm["DC_T"]**2))
        fG = np.sqrt(np.nansum(arm["DC_G"]**2))
        ratio = fG / (fT + 1e-8)
        SE_pooled = np.sqrt(SE_lo**2 + SE_hi**2)
        z = DC_T_flat / (SE_pooled + 1e-6)
        n_sig = int(np.sum(np.abs(z) > 2))

        lines.append(f"| {pname} | {arm.get('n_lo', '?')} | {arm.get('n_hi', '?')} | "
                     f"{fT:.4f} | {fG:.4f} | {ratio:.3f} | {n_sig} |\n")

    r2 = r_pooled**2
    if r2 >= 0.5:
        verdict = "STRONG: BIND faithfully reproduces parameter-driven shifts in joint scatter structure."
    elif r2 >= 0.25:
        verdict = "MODERATE: BIND partially tracks parameter-driven shifts. Some pairs tracked, others not."
    else:
        verdict = "WEAK: BIND does not clearly track parameter-driven shifts in joint scatter structure."

    lines += ["\n## Verdict\n", f"R²={r2:.4f} → {verdict}\n\n"]
    lines.append("### Significant entries (|z|>2) across all 35 params:\n")
    lines.append("| Param | obs_a | obs_b | ΔC_T | ΔC_G | z | ratio | sign_ok |\n")
    lines.append("|-------|-------|-------|------|------|---|-------|--------|\n")

    for arm, pname in zip(arm_results, param_names):
        DC_T_flat = _offdiag_flat(arm["DC_T"])
        DC_G_flat = _offdiag_flat(arm["DC_G"])
        SE_lo = _offdiag_flat(arm.get("SE_T_lo", arm.get("SE_T", np.zeros((7,7)))))
        SE_hi = _offdiag_flat(arm.get("SE_T_hi", arm.get("SE_T", np.zeros((7,7)))))
        SE_pooled = np.sqrt(SE_lo**2 + SE_hi**2)
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

    report_path = OUT_DIR / "REPORT35.md"
    report_path.write_text("".join(lines))
    log(f"Saved REPORT35.md: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_from_cache(pname: str) -> dict | None:
    """Try loading a previously computed arm from sweep35_arm_{pname}.npz,
    then fall back to the 6-arm sweep_arm_{pname}.npz (for p1-p6).
    Returns None if no cache found."""
    for candidate in [
        OUT_DIR / f"sweep35_arm_{pname}.npz",
        OUT_DIR / f"sweep_arm_{pname}.npz",  # 6-param sweep output
    ]:
        if candidate.exists():
            d = np.load(str(candidate), allow_pickle=True)
            result = {
                "pname":   pname,
                "DC_T":    d["DC_T"],
                "DC_G":    d["DC_G"],
                "SE_T_lo": d["SE_T_lo"],
                "SE_T_hi": d["SE_T_hi"],
                "n_lo":    int(d["n_lo"]) if "n_lo" in d.files else "?",
                "n_hi":    int(d["n_hi"]) if "n_hi" in d.files else "?",
            }
            log(f"  Loaded from cache ({candidate.name}): "
                f"||ΔC_T||_F={np.sqrt(np.nansum(result['DC_T']**2)):.4f}, "
                f"||ΔC_G||_F={np.sqrt(np.nansum(result['DC_G']**2)):.4f}")
            return result

    # Special case: A_SN1 pilot
    if pname == "A_SN1" and PILOT_NPZ.exists():
        pilot = np.load(str(PILOT_NPZ))
        result = {
            "pname":   pname,
            "DC_T":    pilot["DC_T"],
            "DC_G":    pilot["DC_G"],
            "SE_T_lo": pilot["SE_T_lo"],
            "SE_T_hi": pilot["SE_T_hi"],
            "n_lo":    49,
            "n_hi":    49,
        }
        log(f"  Loaded A_SN1 from pilot cache: "
            f"||ΔC_T||_F={np.sqrt(np.nansum(result['DC_T']**2)):.4f}")
        return result

    return None


def save_arm_cache(pname: str, result: dict) -> None:
    out = OUT_DIR / f"sweep35_arm_{pname}.npz"
    np.savez(str(out),
             DC_T    = result["DC_T"],
             DC_G    = result["DC_G"],
             C_T_lo  = result.get("C_T_lo", np.full((7,7), np.nan)),
             C_T_hi  = result.get("C_T_hi", np.full((7,7), np.nan)),
             C_G_lo  = result.get("C_G_lo", np.full((7,7), np.nan)),
             C_G_hi  = result.get("C_G_hi", np.full((7,7), np.nan)),
             SE_T_lo = result.get("SE_T_lo", np.full((7,7), np.nan)),
             SE_T_hi = result.get("SE_T_hi", np.full((7,7), np.nan)),
             n_lo    = result.get("n_lo", 0),
             n_hi    = result.get("n_hi", 0))
    log(f"  Saved sweep35_arm_{pname}.npz")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume",  action="store_true", help="Skip arms with existing cache files")
    p.add_argument("--start",   type=int, default=1,  help="Start from arm index (1-based)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    log(f"=== joint_struct_sweep_35.py START (resume={args.resume}, start={args.start}) ===")

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
    log("CV fiducial matrices loaded")

    # ── Loop over all 35 arms ─────────────────────────────────────────────────
    arm_results  = []
    param_names  = []
    n_total      = len(ARMS)

    for arm_idx, (pname, sim_lo, sim_hi) in enumerate(ARMS):
        if arm_idx + 1 < args.start:
            # Need to load results for skipped arms to keep the list complete
            cached = load_from_cache(pname)
            if cached is None:
                log(f"  WARNING: --start skipped arm {pname} but no cache found; inserting NaN arm")
                cached = {
                    "pname": pname,
                    "DC_T": np.full((7,7), np.nan),
                    "DC_G": np.full((7,7), np.nan),
                    "SE_T_lo": np.full((7,7), np.nan),
                    "SE_T_hi": np.full((7,7), np.nan),
                    "n_lo": 0, "n_hi": 0,
                }
            arm_results.append(cached)
            param_names.append(pname)
            continue

        log(f"\n{'='*60}")
        log(f"ARM {arm_idx+1}/{n_total}: {pname}  ({sim_lo} → {sim_hi})")
        log(f"{'='*60}")

        # ── Resume check ──────────────────────────────────────────────────
        if args.resume:
            cached = load_from_cache(pname)
            if cached is not None:
                arm_results.append(cached)
                param_names.append(pname)
                continue

        # ── Load sims ─────────────────────────────────────────────────────
        log(f"  Loading {sim_lo} ...")
        data_lo = load_1p_sim(sim_lo, ns, pname)
        log(f"  Loading {sim_hi} ...")
        data_hi = load_1p_sim(sim_hi, ns, pname)

        if data_lo is None or data_hi is None:
            log(f"  SKIP: missing sim data")
            arm_results.append({
                "pname": pname,
                "DC_T": np.full((7,7), np.nan),
                "DC_G": np.full((7,7), np.nan),
                "SE_T_lo": np.full((7,7), np.nan),
                "SE_T_hi": np.full((7,7), np.nan),
                "n_lo": 0, "n_hi": 0,
            })
            param_names.append(pname)
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

        # ── Compute correlation matrices ───────────────────────────────────
        log(f"  Computing LOWESS + Spearman matrices (B={BOOT_B}) ...")
        seed_offset = arm_idx * 100
        mats_lo = compute_endpoint_matrices(data_lo["masses"], truth_lo, bind_lo, rng_seed=seed_offset)
        mats_hi = compute_endpoint_matrices(data_hi["masses"], truth_hi, bind_hi, rng_seed=seed_offset + 50)

        DC_T = mats_hi["C_T"] - mats_lo["C_T"]
        DC_G = mats_hi["C_G"] - mats_lo["C_G"]

        log(f"  ||ΔC_T||_F={np.sqrt(np.nansum(DC_T**2)):.4f}, ||ΔC_G||_F={np.sqrt(np.nansum(DC_G**2)):.4f}")

        result = {
            "pname":   pname,
            "sim_lo":  sim_lo,
            "sim_hi":  sim_hi,
            "DC_T":    DC_T,
            "DC_G":    DC_G,
            "C_T_lo":  mats_lo["C_T"],
            "C_T_hi":  mats_hi["C_T"],
            "C_G_lo":  mats_lo["C_G"],
            "C_G_hi":  mats_hi["C_G"],
            "SE_T_lo": mats_lo["SE_T"],
            "SE_T_hi": mats_hi["SE_T"],
            "n_lo":    data_lo["N"],
            "n_hi":    data_hi["N"],
        }
        save_arm_cache(pname, result)

        arm_results.append(result)
        param_names.append(pname)

    # ── Aggregate and save ────────────────────────────────────────────────────
    log("\nSaving sweep35_results.npz ...")
    save_dict = {"param_names": np.array(param_names),
                 "C_T_fid": fid["C_T"], "C_G_fid": fid["C_G"],
                 "obs_names": np.array(OBS_7)}
    for r, pname in zip(arm_results, param_names):
        save_dict[f"DC_T_{pname}"]    = r["DC_T"]
        save_dict[f"DC_G_{pname}"]    = r["DC_G"]
        save_dict[f"SE_T_lo_{pname}"] = r.get("SE_T_lo", np.full((7,7), np.nan))
        save_dict[f"SE_T_hi_{pname}"] = r.get("SE_T_hi", np.full((7,7), np.nan))
    np.savez(str(OUT_DIR / "sweep35_results.npz"), **save_dict)
    log(f"Saved: {OUT_DIR}/sweep35_results.npz")

    log("Making headline figure ...")
    make_sweep35_figure(arm_results, param_names)

    log("Writing REPORT35.md ...")
    write_report35(arm_results, param_names)

    elapsed = time.time() - t0
    log(f"\n=== SWEEP SUMMARY ===")
    log(f"Parameters: {', '.join(param_names)}")
    all_T = np.concatenate([_offdiag_flat(r["DC_T"]) for r in arm_results])
    all_G = np.concatenate([_offdiag_flat(r["DC_G"]) for r in arm_results])
    mask  = np.isfinite(all_T) & np.isfinite(all_G)
    if mask.sum() > 2:
        r_pooled, _ = scipy_stats.pearsonr(all_T[mask], all_G[mask])
        slope, intercept, _, _, _ = scipy_stats.linregress(all_T[mask], all_G[mask])
        sign_match = int(np.sum(np.sign(all_T[mask]) == np.sign(all_G[mask])))
        log(f"Pooled R²(ΔC_G ~ ΔC_T) = {r_pooled**2:.4f}")
        log(f"Slope = {slope:.3f}, intercept = {intercept:.3f}")
        log(f"Sign agreement: {sign_match}/{mask.sum()} ({100*sign_match/mask.sum():.1f}%)")
    log(f"=== joint_struct_sweep_35.py DONE in {elapsed:.0f}s ===")


if __name__ == "__main__":
    main()
