"""scatter/joint_struct_pilot.py — Phase 1 pilot for §4 of Bind_joint_scatter.md.

Tests whether BIND reproduces the parameter dependence of residual correlation
matrices across A_SN1 endpoints (1P_p3_n2 and 1P_p3_2).

Outputs:
  outputs/scatter_joint_structure/pilot_matrices.npz
  outputs/scatter_joint_structure/pilot_gate.json
  outputs/scatter_joint_structure/pilot_residuals_p3_n2.npz
  outputs/scatter_joint_structure/pilot_residuals_p3_2.npz
  outputs/scatter_joint_structure/PROGRESS.log
  figures/scatter_joint_structure/fig_pilot_ASN1.pdf / .png

Stop conditions written to:
  outputs/scatter_joint_structure/PILOT_NULL.md   — truth structure doesn't shift
  outputs/scatter_joint_structure/PILOT_FAIL.md   — truth shifts but BIND doesn't track
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
RUN_DIR    = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
TEST_1P    = Path("/mnt/home/mlee1/ceph/fm_testsuite/1P")
OUT_DIR    = BASE_DIR / "outputs/scatter_joint_structure"
FIG_DIR    = BASE_DIR / "figures/scatter_joint_structure"
FID_MAT    = BASE_DIR / "scatter/scatter_residual/matrices.npz"
PROGRESS   = OUT_DIR / "PROGRESS.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SUB_DIR      = "snap_090/mass_threshold_1p000e13"
BOX_SIZE     = 50.0      # Mpc/h
N_PIX_FULL   = 1024
PATCH_PIX    = 128
MPC_PER_PIX  = BOX_SIZE * PATCH_PIX / N_PIX_FULL / PATCH_PIX  # 0.048828125 Mpc/h/pix
NOISE_SEED   = 42
K_SAMPLES    = 10
N_STEPS      = 20
BATCH_SIZE   = 16
BOOT_B       = 2000
FRAC_LOWESS  = 0.4
SIM_LO       = "1P_p3_n2"
SIM_HI       = "1P_p3_2"

# Observable pair indices in OBS_7 for the 3 focal panels
# OBS_7 = ["log10_M_DM","log10_M_gas","log10_M_star","log10_Sigma_gas_c","q_DM","q_gas","q_star"]
FOCUS_PAIRS  = [(2, 1), (2, 4), (1, 4)]   # (M_star,M_gas), (M_star,q_DM), (M_gas,q_DM)
FOCUS_LABELS = [
    (r"$\log_{10}M_\star$", r"$\log_{10}M_{\rm gas}$"),
    (r"$\log_{10}M_\star$", r"$q_{\rm DM}$"),
    (r"$\log_{10}M_{\rm gas}$", r"$q_{\rm DM}$"),
]

# A_SN1 values: fiducial from CV (raw=1.0), lo (raw=0.9), hi (raw=14.4)
ASN1_FID = 1.0
ASN1_LO  = 0.9
ASN1_HI  = 14.4

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

OBS_8_NAMES = list(ALL_OBS_NAMES[:10]) + list(ALL_OBS_NAMES[10:])  # for extract_obs8


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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize_params_fid(p_raw: np.ndarray, ns: NormStats) -> np.ndarray:
    """Raw param vector → normalized [0,1] vector (respects param_log_flag)."""
    _p = np.where(
        ns.param_log_flag == 1,
        np.log10(np.maximum(p_raw.astype(float), 1e-30)),
        p_raw.astype(float),
    )
    return ((_p - ns.param_min) / (ns.param_max - ns.param_min + 1e-8)).astype(np.float32)


def load_1p_sim(sim_name: str, ns: NormStats) -> dict | None:
    """Load + normalize a 1P sim's halo cutouts. Returns None if absent."""
    base = TEST_1P / sim_name / SUB_DIR
    cat_path = base / "halo_catalog.npz"
    cut_path = base / "halo_cutouts.npz"
    if not cat_path.exists() or not cut_path.exists():
        log(f"  MISSING: {base}")
        return None
    cat = np.load(cat_path, allow_pickle=True)
    cut = np.load(cut_path, allow_pickle=True)

    # CAMELS bug: p14 = 0 for 1P/CV runs
    params = cat["params"].copy().astype(np.float32)
    params[:, 14] = 0.0

    cond_raw = cut["condition"].astype(np.float32)    # (N, 128, 128)
    ls_raw   = cut["large_scale"].astype(np.float32)  # (N, 3, 128, 128)

    cond_norm = (log_transform(cond_raw) - ns.cond_mean) / (ns.cond_std + 1e-8)
    ls_norm   = (log_transform(ls_raw) - ns.ls_mean[:, None, None]) / (ns.ls_std[:, None, None] + 1e-8)

    radii  = cat["radii"] / 1000.0 / MPC_PER_PIX   # kpc/h → Mpc/h → pixels
    masses = cat["halo_masses"].astype(np.float64)
    centers = cat["centers"].astype(np.float64)     # (N, 2) in Mpc/h
    omega_m = params[:, 0].astype(np.float64)

    return {
        "cond_raw":  cond_raw,
        "cond_norm": cond_norm[:, np.newaxis],  # (N, 1, 128, 128)
        "ls_norm":   ls_norm,
        "params":    params,
        "masses":    masses,
        "radii_pix": radii,
        "omega_m":   omega_m,
        "centers":   centers,
        "N":         len(masses),
    }


def extract_truth_obs_1p(sim_name: str, data: dict) -> np.ndarray:
    """Extract 16 truth observables (ALL_OBS_NAMES order) from full_maps.npz.

    Returns (N_h, 16) float32, with log-observables still in physical (non-log) space.
    The extract_obs8 function applies log10 later.
    """
    from scatter.measure_scatter import OMEGA_B_FIXED, axis_ratio_q
    sim_dir  = TEST_1P / sim_name / "snap_090"
    fm_path  = sim_dir / "full_maps.npz"
    if not fm_path.exists():
        raise FileNotFoundError(f"full_maps.npz missing: {fm_path}")

    fm          = np.load(fm_path)
    truth_maps  = fm["truth_maps"].astype(np.float32)   # (3, 1024, 1024)

    N      = data["N"]
    masses = data["masses"]
    radii  = data["radii_pix"]
    om     = data["omega_m"]
    centers = data["centers"]
    cond_raw = data["cond_raw"]   # (N, 128, 128) — DMO patches for q_DMO

    # Cosmic baryon fraction per halo
    f_b_arr = OMEGA_B_FIXED / np.where(om > 0, om, np.nan)

    # Pre-compute q_DMO from DMO patches
    q_dmo_arr = np.full(N, np.nan, dtype=np.float64)
    for i in range(N):
        r_aper = max(min(float(radii[i]), PATCH_PIX / 2 - 2), 4.0)
        q_dmo_arr[i] = axis_ratio_q(np.maximum(cond_raw[i].astype(np.float64), 0.0), r_aper)

    N_obs = len(ALL_OBS_NAMES)
    truth_obs = np.full((N, N_obs), np.nan, dtype=np.float32)

    for i in range(N):
        cx_mpc = float(centers[i, 0])
        cy_mpc = float(centers[i, 1])
        cx_pix = int(cx_mpc / BOX_SIZE * N_PIX_FULL) % N_PIX_FULL
        cy_pix = int(cy_mpc / BOX_SIZE * N_PIX_FULL) % N_PIX_FULL

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


def run_bind_endpoint(
    sim_name: str, data: dict, ns: NormStats, model_fm, device_str: str
) -> np.ndarray:
    """Run BIND inference at a single 1P endpoint. Returns obs_tensor (N, K, 16)."""
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
    return result["obs_tensor"]   # (N, K, 16)


def compute_endpoint_matrices(
    masses: np.ndarray,
    truth_raw: np.ndarray,        # (N, 16) — physical space
    bind_obs: np.ndarray,         # (N, K, 16) — physical space
    rng_seed: int = 0,
) -> dict:
    """Compute OBS_7 residuals and Spearman correlation matrices for one endpoint.

    Returns dict with:
      delta_T (N, 7), delta_G (N, 7)
      C_T (7,7), SE_T (7,7), C_G (7,7), SE_G (7,7)
      obs7 (N, 7) truth, bind_mean7 (N, 7) BIND mean
    """
    log_mass = np.log10(masses)
    n_obs7   = len(OBS_7)

    # Project to OBS_8 (8) then take OBS_7 (first 7)
    obs8_T   = extract_obs8(truth_raw, list(ALL_OBS_NAMES))[:, :n_obs7]     # (N, 7) truth log-obs
    # BIND: average K samples in physical space then apply log10
    bind_mean_raw = bind_obs.mean(axis=1)                                    # (N, 16)
    obs8_G   = extract_obs8(bind_mean_raw, list(ALL_OBS_NAMES))[:, :n_obs7] # (N, 7)

    # Per-endpoint, per-source LOWESS fit
    delta_T = np.full((len(masses), n_obs7), np.nan)
    delta_G = np.full((len(masses), n_obs7), np.nan)

    for a in range(n_obs7):
        f_T = obs8_T[:, a]
        f_G = obs8_G[:, a]

        # Fit for truth
        ms_T = fit_mean_and_scatter(
            log_mass, f_T, log_mass, f_G, frac=FRAC_LOWESS, fit_source="truth"
        )
        delta_T[:, a] = standardise_residuals(log_mass, f_T, ms_T.mu, ms_T.sigma)

        # Fit for BIND
        ms_G = fit_mean_and_scatter(
            log_mass, f_T, log_mass, f_G, frac=FRAC_LOWESS, fit_source="bind"
        )
        delta_G[:, a] = standardise_residuals(log_mass, f_G, ms_G.mu, ms_G.sigma)

    # Correlation matrices
    C_T, SE_T = residual_correlation_matrix(delta_T, method="spearman", n_boot=BOOT_B, rng_seed=rng_seed)
    C_G, SE_G = residual_correlation_matrix(delta_G, method="spearman", n_boot=BOOT_B, rng_seed=rng_seed + 1000)

    return {
        "delta_T":   delta_T,
        "delta_G":   delta_G,
        "C_T":       C_T,
        "SE_T":      SE_T,
        "C_G":       C_G,
        "SE_G":      SE_G,
        "obs7_T":    obs8_T,
        "obs7_G":    obs8_G,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
def make_pilot_figure(
    ep_lo: dict,
    ep_hi: dict,
    fid_mat: dict,
    asn1_lo: float,
    asn1_hi: float,
    asn1_fid: float,
    DC_T: np.ndarray,   # (7,7)
    DC_G: np.ndarray,   # (7,7)
    SE_T_lo: np.ndarray,
    SE_T_hi: np.ndarray,
    SE_G_lo: np.ndarray,
    SE_G_hi: np.ndarray,
):
    """3 panels (line plots at 3 x-values) + bar chart."""
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    x_vals   = [asn1_lo, asn1_fid, asn1_hi]
    x_labels = [f"{asn1_lo:.1f}", f"{asn1_fid:.1f}", f"{asn1_hi:.0f}"]
    # OBS_7 labels
    obs7_labels = [r"$\log M_{\rm DM}$", r"$\log M_{\rm gas}$", r"$\log M_\star$",
                   r"$\log\Sigma_{\rm gas,c}$", r"$q_{\rm DM}$", r"$q_{\rm gas}$", r"$q_\star$"]

    for panel_idx, (ia, ib) in enumerate(FOCUS_PAIRS):
        ax = fig.add_subplot(gs[0, panel_idx])
        la, lb = FOCUS_LABELS[panel_idx]

        # Truth: 3 points
        c_T_lo  = ep_lo["C_T"][ia, ib]
        c_T_hi  = ep_hi["C_T"][ia, ib]
        c_T_fid = fid_mat["C_T"][ia, ib]

        # BIND: 3 points
        c_G_lo  = ep_lo["C_G"][ia, ib]
        c_G_hi  = ep_hi["C_G"][ia, ib]
        c_G_fid = fid_mat["C_G"][ia, ib]

        # Bootstrap SE bars
        se_T_lo  = ep_lo["SE_T"][ia, ib]
        se_T_hi  = ep_hi["SE_T"][ia, ib]
        se_T_fid = fid_mat["SE_T"][ia, ib]
        se_G_lo  = ep_lo["SE_G"][ia, ib]
        se_G_hi  = ep_hi["SE_G"][ia, ib]
        se_G_fid = fid_mat["SE_G"][ia, ib]

        # Plot truth
        ax.errorbar(x_vals,
                    [c_T_lo, c_T_fid, c_T_hi],
                    yerr=[se_T_lo, se_T_fid, se_T_hi],
                    color="black", marker="o", ms=6, lw=1.5, capsize=4, label="Truth")
        # Plot BIND
        ax.errorbar(x_vals,
                    [c_G_lo, c_G_fid, c_G_hi],
                    yerr=[se_G_lo, se_G_fid, se_G_hi],
                    color="crimson", marker="s", ms=6, lw=1.5, capsize=4, label="BIND", ls="--")

        ax.set_xscale("log")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel(r"$A_{\rm SN1}$", fontsize=9)
        ax.set_ylabel(r"$\rho_{\rm Sp}$", fontsize=9)
        ax.set_title(f"({la}, {lb})", fontsize=9)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_ylim(-1.05, 1.05)
        if panel_idx == 0:
            ax.legend(fontsize=8, loc="best")

    # ── Bar chart of all 21 off-diagonal ΔC entries ──────────────────────────
    ax_bar = fig.add_subplot(gs[1, :])

    triu_idx = np.triu_indices(7, k=1)  # 21 pairs
    pairs_idx = list(zip(triu_idx[0], triu_idx[1]))

    # Sort by |ΔC_T| descending
    dc_T_vals = np.array([DC_T[ia, ib] for ia, ib in pairs_idx])
    dc_G_vals = np.array([DC_G[ia, ib] for ia, ib in pairs_idx])
    order     = np.argsort(-np.abs(dc_T_vals))
    dc_T_vals = dc_T_vals[order]
    dc_G_vals = dc_G_vals[order]
    pair_lbls = [f"{obs7_labels[pairs_idx[i][0]].replace('$','')}\n{obs7_labels[pairs_idx[i][1]].replace('$','')}"
                 for i in order]

    x_bar = np.arange(21)
    w     = 0.35
    ax_bar.bar(x_bar - w/2, dc_T_vals, width=w, color="black", alpha=0.7, label=r"$\Delta C^T$")
    ax_bar.bar(x_bar + w/2, dc_G_vals, width=w, color="crimson", alpha=0.7, label=r"$\Delta C^G$")

    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(pair_lbls, fontsize=6, rotation=45, ha="right")
    ax_bar.axhline(0, color="gray", lw=0.5)
    ax_bar.set_ylabel(r"$\Delta C_{ab}$", fontsize=10)
    ax_bar.set_title(r"All 21 off-diagonal $\Delta C$ entries: truth (black) vs BIND (red)", fontsize=10)
    ax_bar.legend(fontsize=9)

    fig.suptitle(r"A$_{\rm SN1}$ parameter-dependent residual correlation shift", fontsize=12, y=1.01)

    out_pdf = FIG_DIR / "fig_pilot_ASN1.pdf"
    out_png = FIG_DIR / "fig_pilot_ASN1.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log(f"Saved figure to {out_pdf}")
    return out_pdf


# ─────────────────────────────────────────────────────────────────────────────
# Pilot gate
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_pilot_gate(
    DC_T: np.ndarray,    # (7,7)
    DC_G: np.ndarray,    # (7,7)
    SE_T_lo: np.ndarray,
    SE_T_hi: np.ndarray,
) -> dict:
    """Evaluate pass/fail per §4.5 of brief.

    z = DC_T / sqrt(SE_lo^2 + SE_hi^2) for each off-diagonal entry.
    C1: at least 1 entry with |z| > 2
    C2: for at least 1 C1 entry, |DC_G/DC_T| >= 0.5 AND sign match
    """
    triu_idx = np.triu_indices(7, k=1)
    ias, ibs = triu_idx

    DC_T_flat  = DC_T[ias, ibs]
    DC_G_flat  = DC_G[ias, ibs]
    se_T_lo_fl = SE_T_lo[ias, ibs]
    se_T_hi_fl = SE_T_hi[ias, ibs]

    # z-scores for truth shift
    combined_se = np.sqrt(se_T_lo_fl**2 + se_T_hi_fl**2 + 1e-9)
    z_flat      = DC_T_flat / combined_se

    sig_mask = np.abs(z_flat) > 2.0  # C1 significant entries
    n_sig = int(sig_mask.sum())

    # For significant entries, check BIND tracking (C2)
    tracking_ok = False
    tracking_rows = []
    for k in np.where(sig_mask)[0]:
        ia, ib = int(ias[k]), int(ibs[k])
        ratio = float(DC_G_flat[k] / DC_T_flat[k]) if abs(DC_T_flat[k]) > 1e-9 else np.nan
        sign_match = bool(np.sign(DC_G_flat[k]) == np.sign(DC_T_flat[k]))
        passes_c2  = abs(DC_G_flat[k] / DC_T_flat[k]) >= 0.5 if abs(DC_T_flat[k]) > 1e-9 else False
        passes_c2  = passes_c2 and sign_match
        if passes_c2:
            tracking_ok = True
        tracking_rows.append({
            "obs_a": OBS_7[ia], "obs_b": OBS_7[ib],
            "DC_T": float(DC_T_flat[k]), "DC_G": float(DC_G_flat[k]),
            "z": float(z_flat[k]), "ratio": float(ratio) if ratio is not None and not np.isnan(ratio) else None,
            "sign_match": bool(sign_match), "passes_c2": bool(passes_c2),
        })

    # Verdict
    c1_pass = n_sig >= 1
    c2_pass = tracking_ok if c1_pass else False

    return {
        "n_sig_entries": int(n_sig),
        "c1_pass": bool(c1_pass),
        "c2_pass": bool(c2_pass),
        "overall_pass": bool(c1_pass and c2_pass),
        "significant_entries": tracking_rows,
        "z_scores": z_flat.tolist(),
        "DC_T": DC_T_flat.tolist(),
        "DC_G": DC_G_flat.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    log("=== joint_struct_pilot.py START ===")

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

    # ── Load 1P sims ─────────────────────────────────────────────────────────
    log(f"Loading 1P sims: {SIM_LO}, {SIM_HI}")
    data_lo = load_1p_sim(SIM_LO, ns)
    data_hi = load_1p_sim(SIM_HI, ns)
    if data_lo is None or data_hi is None:
        msg = "PHASE0 FAIL: 1P sim data missing"
        log(msg)
        (OUT_DIR / "STOP_REPORT.md").write_text(f"# STOP\n{msg}\n")
        return

    log(f"  {SIM_LO}: N={data_lo['N']}, A_SN1={data_lo['params'][0,2]:.4f} (raw), "
        f"masses range [{data_lo['masses'].min():.2e}, {data_lo['masses'].max():.2e}]")
    log(f"  {SIM_HI}: N={data_hi['N']}, A_SN1={data_hi['params'][0,2]:.4f} (raw), "
        f"masses range [{data_hi['masses'].min():.2e}, {data_hi['masses'].max():.2e}]")

    # ── Extract truth observables ─────────────────────────────────────────────
    log("Extracting truth observables from full_maps.npz ...")
    truth_lo, q_dmo_lo = extract_truth_obs_1p(SIM_LO, data_lo)
    truth_hi, q_dmo_hi = extract_truth_obs_1p(SIM_HI, data_hi)
    log(f"  truth_lo: {truth_lo.shape}, NaN frac: {np.isnan(truth_lo).mean():.3f}")
    log(f"  truth_hi: {truth_hi.shape}, NaN frac: {np.isnan(truth_hi).mean():.3f}")

    # ── Run BIND inference ────────────────────────────────────────────────────
    log(f"Running BIND inference at lo endpoint ({SIM_LO}), K={K_SAMPLES} ...")
    bind_lo = run_bind_endpoint(SIM_LO, data_lo, ns, model_fm, device_str)
    log(f"  bind_lo: {bind_lo.shape}")

    log(f"Running BIND inference at hi endpoint ({SIM_HI}), K={K_SAMPLES} ...")
    bind_hi = run_bind_endpoint(SIM_HI, data_hi, ns, model_fm, device_str)
    log(f"  bind_hi: {bind_hi.shape}")

    # ── Compute per-endpoint matrices ─────────────────────────────────────────
    log("Computing per-endpoint LOWESS + residuals + Spearman matrices (B=2000) ...")
    ep_lo = compute_endpoint_matrices(data_lo["masses"], truth_lo, bind_lo, rng_seed=0)
    ep_hi = compute_endpoint_matrices(data_hi["masses"], truth_hi, bind_hi, rng_seed=10)
    log(f"  C_T_lo diag: {np.diag(ep_lo['C_T'])}")
    log(f"  C_T_hi diag: {np.diag(ep_hi['C_T'])}")
    log(f"  C_G_lo diag: {np.diag(ep_lo['C_G'])}")
    log(f"  C_G_hi diag: {np.diag(ep_hi['C_G'])}")

    # ── Save residuals ────────────────────────────────────────────────────────
    np.savez_compressed(
        OUT_DIR / "pilot_residuals_p3_n2.npz",
        delta_T=ep_lo["delta_T"], delta_G=ep_lo["delta_G"],
        obs7_T=ep_lo["obs7_T"],   obs7_G=ep_lo["obs7_G"],
        masses=data_lo["masses"], obs_names=np.array(OBS_7),
    )
    np.savez_compressed(
        OUT_DIR / "pilot_residuals_p3_2.npz",
        delta_T=ep_hi["delta_T"], delta_G=ep_hi["delta_G"],
        obs7_T=ep_hi["obs7_T"],   obs7_G=ep_hi["obs7_G"],
        masses=data_hi["masses"], obs_names=np.array(OBS_7),
    )
    log("Saved residuals npz files")

    # ── Load fiducial matrices from morning's CV analysis ─────────────────────
    fid = np.load(FID_MAT)
    fid_mat = {
        "C_T":  fid["C_T"],
        "C_G":  fid["C_G"],
        "SE_T": fid["SE_T"],
        "SE_G": fid["SE_G"],
    }
    log(f"Loaded CV fiducial matrices from {FID_MAT}")

    # ── ΔC_T and ΔC_G ─────────────────────────────────────────────────────────
    DC_T = ep_hi["C_T"] - ep_lo["C_T"]   # (7,7)
    DC_G = ep_hi["C_G"] - ep_lo["C_G"]   # (7,7)

    log(f"ΔC_T off-diag abs-max: {np.abs(DC_T[np.triu_indices(7,k=1)]).max():.4f}")
    log(f"ΔC_G off-diag abs-max: {np.abs(DC_G[np.triu_indices(7,k=1)]).max():.4f}")

    # ── Save pilot_matrices.npz ───────────────────────────────────────────────
    np.savez_compressed(
        OUT_DIR / "pilot_matrices.npz",
        C_T_lo=ep_lo["C_T"],   SE_T_lo=ep_lo["SE_T"],
        C_G_lo=ep_lo["C_G"],   SE_G_lo=ep_lo["SE_G"],
        C_T_hi=ep_hi["C_T"],   SE_T_hi=ep_hi["SE_T"],
        C_G_hi=ep_hi["C_G"],   SE_G_hi=ep_hi["SE_G"],
        DC_T=DC_T,             DC_G=DC_G,
        C_T_fid=fid_mat["C_T"], C_G_fid=fid_mat["C_G"],
        obs_names=np.array(OBS_7),
        sim_lo=np.array(SIM_LO), sim_hi=np.array(SIM_HI),
    )
    log("Saved pilot_matrices.npz")

    # ── Pilot gate ────────────────────────────────────────────────────────────
    gate = evaluate_pilot_gate(DC_T, DC_G, ep_lo["SE_T"], ep_hi["SE_T"])
    gate["sim_lo"] = SIM_LO
    gate["sim_hi"] = SIM_HI
    gate["n_halos_lo"] = int(data_lo["N"])
    gate["n_halos_hi"] = int(data_hi["N"])
    gate["asn1_lo"] = ASN1_LO
    gate["asn1_hi"] = ASN1_HI
    gate["frobenius_DC_T"] = float(np.sqrt(np.nansum(DC_T**2)))
    gate["frobenius_DC_G"] = float(np.sqrt(np.nansum(DC_G**2)))

    (OUT_DIR / "pilot_gate.json").write_text(json.dumps(gate, indent=2))
    log(f"Pilot gate: C1={'PASS' if gate['c1_pass'] else 'FAIL'}, "
        f"C2={'PASS' if gate['c2_pass'] else 'FAIL'}, "
        f"n_sig={gate['n_sig_entries']}")

    # ── Write stop reports if gate fails ─────────────────────────────────────
    if not gate["c1_pass"]:
        msg = (f"# PILOT NULL — Truth correlation structure doesn't shift with A_SN1\n\n"
               f"n_sig_entries = {gate['n_sig_entries']} (need ≥1, |z|>2)\n"
               f"N_halos per endpoint = {data_lo['N']}\n\n"
               f"Possible causes:\n"
               f"- A_SN1 doesn't drive correlation structure at these mass scales\n"
               f"- 49 halos per endpoint too few to detect the shift\n"
               f"- 1P_p3 endpoints (0.9 vs 14.4) are sufficient range but signal is small\n\n"
               f"Recommendation: human review before widening endpoints or changing parameter.\n")
        (OUT_DIR / "PILOT_NULL.md").write_text(msg)
        log("PILOT NULL — stopping as per brief §7.2")
        log(f"=== joint_struct_pilot.py DONE (pilot null) in {time.time()-t0:.0f}s ===")
        return

    if not gate["c2_pass"]:
        msg = (f"# PILOT FAIL — Truth shifts but BIND doesn't track\n\n"
               f"n_sig_entries = {gate['n_sig_entries']} (C1 PASS)\n"
               f"But no entry has |ΔC_G/ΔC_T|≥0.5 AND correct sign (C2 FAIL)\n\n"
               f"Significant entries (|z|>2):\n")
        for row in gate["significant_entries"]:
            msg += f"  ({row['obs_a']}, {row['obs_b']}): ΔC_T={row['DC_T']:.3f}, "
            msg += f"ΔC_G={row['DC_G']:.3f}, ratio={row['ratio']:.3f}, sign_ok={row['sign_match']}\n"
        msg += "\nThis is a meaningful negative result. ESCALATE — human review required.\n"
        (OUT_DIR / "PILOT_FAIL.md").write_text(msg)
        log("PILOT FAIL — stopping and escalating as per brief §7.3")

    # ── Figure ────────────────────────────────────────────────────────────────
    make_pilot_figure(
        ep_lo, ep_hi, fid_mat,
        ASN1_LO, ASN1_HI, ASN1_FID,
        DC_T, DC_G,
        ep_lo["SE_T"], ep_hi["SE_T"],
        ep_lo["SE_G"], ep_hi["SE_G"],
    )

    # ── Summary print ─────────────────────────────────────────────────────────
    log("=== PILOT GATE RESULTS ===")
    log(f"C1 (truth shifts): {'PASS' if gate['c1_pass'] else 'FAIL'} — {gate['n_sig_entries']} entries |z|>2")
    log(f"C2 (BIND tracks): {'PASS' if gate['c2_pass'] else 'FAIL'}")
    log(f"||ΔC_T||_F = {gate['frobenius_DC_T']:.4f}, ||ΔC_G||_F = {gate['frobenius_DC_G']:.4f}")
    if gate["significant_entries"]:
        log("Significant entries:")
        for row in gate["significant_entries"]:
            log(f"  ({row['obs_a']}, {row['obs_b']}): ΔC_T={row['DC_T']:.3f}, "
                f"ΔC_G={row['DC_G']:.3f}, z={row['z']:.2f}, sign_ok={row['sign_match']}, "
                f"passes_c2={row['passes_c2']}")

    if gate["overall_pass"]:
        log("OVERALL: PASS — proceed to Phase 2 (full 6-param sweep)")
    else:
        log("OVERALL: FAIL — see PILOT_NULL.md or PILOT_FAIL.md for details")

    log(f"=== joint_struct_pilot.py DONE in {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
