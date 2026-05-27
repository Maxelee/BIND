"""fig4_truth_scatter_vs_ASN1.py — Figure 4 of BIND convincingness brief.

Show BIND σ_inter prediction vs A_SN1 for clean observables (f_b, Sigma_gas_r3,
M_gas, dq_DM), using a linearized projection from the fiducial sigma + J_log_sigma
Jacobian. Truth σ from 1P_p3 maps (flat baseline — all levels are identical halos).

Data limitation: 1P_p3 sims have IDENTICAL large_scale arrays across all parameter
levels, so the A_SN1 TREND in σ_truth CANNOT be verified. Figure shows:
  (1) BIND predicted σ_inter curve vs A_SN1
  (2) σ_truth at A_SN1 ≈ 1.0 (fiducial) as a horizontal error band
  (3) BIND σ_intra at fiducial for comparison
Outputs:
  figures/scatter_diagnostics/fig_truth_scatter_vs_ASN1.pdf / .png
  outputs/scatter_diagnostics/fig_truth_scatter_vs_ASN1.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from data import NormStats, log_transform
from fd_jacobian_cv import load_cv_halos, normalize_inputs, normalize_params_fid
from train import FlowMatchingLit
from scatter.measure_scatter import (
    measure_scatter, ALL_OBS_NAMES, LOG_MASK, _compute_all_obs, PROFILE_FRACS
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).parent.parent
RUN_DIR   = pathlib.Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
CV_ROOT   = pathlib.Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
TEST_1P   = pathlib.Path("/mnt/home/mlee1/ceph/fm_testsuite/1P")
FIG_DIR   = BASE_DIR / "figures/scatter_diagnostics"
OUT_DIR   = BASE_DIR / "outputs/scatter_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

JMEAN_FILE = BASE_DIR / "scatter/J_mean_and_scatter.npz"
NS_FILE    = RUN_DIR / "norm_stats.npz"
SUB_DIR    = "snap_090/mass_threshold_1p000e13"
MPC_PER_PIX = 0.048828125

NOISE_SEED  = 42
K_SAMPLES   = 5
N_STEPS     = 20
BATCH_SIZE  = 32
MAX_HALOS   = 500   # subset of CV halos for speed

FOCUS_OBS = ["f_b", "Sigma_gas_r3", "M_gas", "dq_DM"]


# ─────────────────────────────────────────────────────────────────────────────
# Truth scatter from 1P truth maps
# ─────────────────────────────────────────────────────────────────────────────
def compute_truth_sigma_from_1p(sim_name: str) -> dict | None:
    """Compute per-observable std (σ_truth) across halos in one 1P sim."""
    base = TEST_1P / sim_name / SUB_DIR
    cat_path = base / "halo_catalog.npz"
    cut_path = base / "halo_cutouts.npz"
    if not cat_path.exists() or not cut_path.exists():
        return None

    cat = np.load(cat_path, allow_pickle=True)
    cut = np.load(cut_path, allow_pickle=True)

    ls = cut["large_scale"].astype(np.float32)   # (N, 3, H, W)
    params = cat["params"].astype(np.float32)
    radii_pix = cat["radii"] / 1000.0 / MPC_PER_PIX
    N = len(radii_pix)

    obs_all = []
    for i in range(N):
        r = float(radii_pix[i])
        om = float(params[i, 0])
        ob = float(params[i, 6])
        f_b_cosmic = ob / om if om > 0 else np.nan
        obs_raw = _compute_all_obs(ls[i], r, f_b_cosmic, q_dmo_val=0.0)  # (16,)
        obs_all.append(obs_raw)

    obs_all = np.array(obs_all)  # (N, 16)
    # Apply log10 to log-scale observables
    obs_log = obs_all.copy()
    for j, log_flag in enumerate(LOG_MASK):
        if log_flag:
            valid = obs_all[:, j] > 0
            obs_log[valid, j] = np.log10(obs_all[valid, j])
            obs_log[~valid, j] = np.nan

    sigma_truth = np.nanstd(obs_log, axis=0, ddof=1)  # (16,)
    sigma_truth_se = sigma_truth / np.sqrt(N - 1)       # rough SE
    return {
        "sigma": sigma_truth.tolist(),
        "sigma_se": sigma_truth_se.tolist(),
        "N": N,
        "sim_name": sim_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fig4] device = {device_str}")

    # ── Load norm stats + Jacobians ──────────────────────────────────────────
    ns = NormStats.load(NS_FILE)
    jm = np.load(JMEAN_FILE, allow_pickle=True)
    obs_names = jm["obs_names"].tolist()
    J_log_sigma_inter    = jm["J_log_sigma"]        # (16, 35)
    J_log_sigma_intra    = jm["J_log_sigma_intra"]  # (16, 35)
    J_log_sigma_inter_se = jm["J_log_sigma_se"]
    J_log_sigma_intra_se = jm["J_log_sigma_intra_se"]

    # A_SN1 is param index 2
    ASN1_PIDX = 2

    # A_SN1 norm range: log10 → linear
    asn1_min = float(ns.param_min[ASN1_PIDX])  # log10 of raw min
    asn1_max = float(ns.param_max[ASN1_PIDX])  # log10 of raw max
    theta_fid_asn1 = 0.0  # fiducial in log10 space; A_SN1=1.0 → log10=0.0
    theta_fid_norm = (theta_fid_asn1 - asn1_min) / (asn1_max - asn1_min)
    print(f"[fig4] A_SN1 param: min={asn1_min:.4f}, max={asn1_max:.4f}")
    print(f"[fig4] theta_fid_norm (A_SN1=1.0) = {theta_fid_norm:.4f}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = RUN_DIR / "checkpoints/last.ckpt"
    lit = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device_str)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm
    model_fm.model.eval()

    # ── Load CV halos ─────────────────────────────────────────────────────────
    print(f"[fig4] loading CV halos from {CV_ROOT}")
    cv = load_cv_halos(CV_ROOT)
    cv["params"][:, 14] = 0.0
    N_TOT = len(cv["masses"])
    print(f"[fig4] N_TOT = {N_TOT}")

    # Subset
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(N_TOT, size=min(MAX_HALOS, N_TOT), replace=False))
    cond_norm, ls_norm = normalize_inputs(cv, ns)

    cond_use    = cond_norm[idx, np.newaxis]
    ls_use      = ls_norm[idx]
    masses_use  = cv["masses"][idx]
    radii_use   = cv["radii_pix"][idx]
    dmo_raw_use = cv["cond_raw"][idx]
    omega_m_use = cv["params"][idx, 0].astype(np.float64)

    # Fiducial param vector (first halo's params as template)
    theta_fid_all = normalize_params_fid(cv["params"][0], ns)
    print(f"[fig4] theta_fid_all[:5] = {theta_fid_all[:5]}")

    # ── Run measure_scatter at fiducial A_SN1 ─────────────────────────────────
    print(f"\n[fig4] running measure_scatter at fiducial A_SN1 (theta_norm={theta_fid_norm:.3f}) ...")
    theta_asn1_fid = theta_fid_all.copy()
    theta_asn1_fid[ASN1_PIDX] = theta_fid_norm

    r_fid = measure_scatter(
        model_fm   = model_fm,
        norm_stats = ns,
        theta_norm = theta_asn1_fid,
        dmo_conds  = cond_use,
        ls_conds   = ls_use,
        masses     = masses_use,
        r200_pix   = radii_use,
        K          = K_SAMPLES,
        n_steps    = N_STEPS,
        device     = device_str,
        batch_size = BATCH_SIZE,
        dmo_raw    = dmo_raw_use,
        omega_m    = omega_m_use,
        seed       = NOISE_SEED,
    )

    sigma_inter_fid = r_fid["sigma_inter"]  # (16,)
    sigma_intra_fid = r_fid["sigma_intra"]  # (16,)
    print(f"[fig4] sigma_inter_fid computed for {len(masses_use)} halos")
    for n in FOCUS_OBS:
        oi = obs_names.index(n)
        print(f"  {n:20s}: sigma_inter={sigma_inter_fid[oi]:.4f}  sigma_intra={sigma_intra_fid[oi]:.4f}")

    # ── Project BIND σ_inter vs A_SN1 ────────────────────────────────────────
    # theta_norm range spanning the 1P levels (roughly n2→2 = 0→1 in normalized space)
    theta_norm_range = np.linspace(0.0, 1.0, 60)
    asn1_raw_range   = 10.0 ** (theta_norm_range * (asn1_max - asn1_min) + asn1_min)
    delta_theta      = theta_norm_range - theta_fid_norm

    sigma_inter_pred = {}  # obs_name → (60,)
    sigma_intra_pred = {}
    for n in FOCUS_OBS:
        oi = obs_names.index(n)
        jls_i = float(J_log_sigma_inter[oi, ASN1_PIDX])
        jls_intra = float(J_log_sigma_intra[oi, ASN1_PIDX])
        sig_fid_i     = float(sigma_inter_fid[oi])
        sig_intra_fid = float(sigma_intra_fid[oi])
        sigma_inter_pred[n] = sig_fid_i * np.exp(jls_i * delta_theta)
        sigma_intra_pred[n] = sig_intra_fid * np.exp(jls_intra * delta_theta)

    # ── Truth sigma from 1P_p3 sims ──────────────────────────────────────────
    print("\n[fig4] computing truth sigma from 1P_p3 sims ...")
    truth_results = {}
    for sim in ["1P_p3_n2", "1P_p3_n1", "1P_p3_0", "1P_p3_1", "1P_p3_2"]:
        res = compute_truth_sigma_from_1p(sim)
        if res:
            print(f"  {sim}: N={res['N']}, sigma_f_b={res['sigma'][obs_names.index('f_b')]:.4f}")
            truth_results[sim] = res
        else:
            print(f"  {sim}: NOT FOUND")

    # Check if truth sigmas are identical (data gap)
    truth_levels = list(truth_results.keys())
    if len(truth_levels) >= 2:
        sigs = np.array([[truth_results[s]["sigma"][obs_names.index(n)] for s in truth_levels]
                         for n in FOCUS_OBS])
        max_diff = float(np.nanmax(np.abs(np.diff(sigs, axis=1))))
        all_identical = max_diff < 1e-6
        print(f"\n  Truth sigma max diff across 1P_p3 levels: {max_diff:.6e}")
        print(f"  All identical: {all_identical} → {'DATA GAP: cannot test A_SN1 trend' if all_identical else 'DISTINCT: trend available'}")
    else:
        all_identical = True
        max_diff = 0.0

    # ── Get A_SN1 values at each 1P level ─────────────────────────────────────
    # Extract actual A_SN1 values from the halo_catalog params
    truth_asn1 = {}
    for sim in truth_levels:
        base = TEST_1P / sim / SUB_DIR / "halo_catalog.npz"
        if base.exists():
            cat = np.load(base, allow_pickle=True)
            asn1_val = float(cat["params"][0, ASN1_PIDX])
            truth_asn1[sim] = asn1_val
            print(f"  {sim}: A_SN1_raw = {asn1_val:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, obs_name in zip(axes, FOCUS_OBS):
        oi = obs_names.index(obs_name)

        # BIND predicted σ_inter curve
        ax.plot(asn1_raw_range, sigma_inter_pred[obs_name],
                color="steelblue", linewidth=2, label=r"BIND $\sigma_{\rm inter}$ (predicted)")
        ax.plot(asn1_raw_range, sigma_intra_pred[obs_name],
                color="steelblue", linewidth=1.5, linestyle="--",
                label=r"BIND $\sigma_{\rm intra}$ (predicted)")

        # Mark fiducial point
        ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.scatter([1.0], [sigma_inter_fid[oi]], color="steelblue", s=60, zorder=5,
                   label="BIND fiducial (measured)")

        # Truth σ at each 1P level (flat if all identical)
        if truth_levels:
            truth_sigs = [truth_results[s]["sigma"][oi] for s in truth_levels]
            truth_sigs_se = [truth_results[s]["sigma_se"][oi] for s in truth_levels]
            asn1_vals = [truth_asn1.get(s, np.nan) for s in truth_levels]

            if all_identical:
                # Use mean as a horizontal band
                mu = float(np.nanmean(truth_sigs))
                se = float(np.nanmean(truth_sigs_se))
                ax.axhline(mu, color="darkorange", linewidth=2, linestyle="-",
                           label=r"$\sigma_{\rm truth}$ (flat — identical cutouts)")
                ax.axhspan(mu - se, mu + se, color="darkorange", alpha=0.2)
            else:
                ax.errorbar(asn1_vals, truth_sigs, yerr=truth_sigs_se,
                            fmt="o", color="darkorange", markersize=6,
                            label=r"$\sigma_{\rm truth}$")

        ax.set_xscale("log")
        ax.set_xlabel("A_SN1 (raw)", fontsize=10)
        ax.set_ylabel(r"$\sigma$ (dex or linear)", fontsize=10)
        ax.set_title(obs_name, fontsize=12)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    if all_identical:
        fig.suptitle(
            "BIND $\\sigma_{\\rm inter}$ vs A_SN1 (linearized from Jacobian)\n"
            r"⚠ Truth trend UNAVAILABLE: 1P_p3 sims have identical hydro cutouts",
            fontsize=11, color="darkred"
        )
    else:
        fig.suptitle(
            "BIND $\\sigma_{\\rm inter}$ vs A_SN1 vs truth",
            fontsize=11
        )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out_path = FIG_DIR / f"fig_truth_scatter_vs_ASN1.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[fig4] saved {out_path}")

    # ── Pass condition ────────────────────────────────────────────────────────
    # Since truth trend is unavailable, check BIND's prediction makes physical sense:
    # J_log_sigma for f_b|A_SN1 should be negative (more SN → less scatter in f_b?)
    focus_pass = {}
    for n in FOCUS_OBS:
        oi = obs_names.index(n)
        jls = float(J_log_sigma_inter[oi, ASN1_PIDX])
        jls_se = float(J_log_sigma_inter_se[oi, ASN1_PIDX])
        sig_fid = float(sigma_inter_fid[oi])
        focus_pass[n] = {
            "J_log_sigma_inter": jls,
            "J_log_sigma_inter_se": jls_se,
            "sigma_inter_fid": sig_fid,
            "sigma_intra_fid": float(sigma_intra_fid[oi]),
        }
        print(f"  {n:20s}: J_log_sigma={jls:+.4f}±{jls_se:.4f}  σ_inter_fid={sig_fid:.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "config": {
            "K": K_SAMPLES, "n_steps": N_STEPS, "seed": NOISE_SEED,
            "max_halos": MAX_HALOS, "N_used": int(len(masses_use)),
            "theta_fid_norm_asn1": float(theta_fid_norm),
            "asn1_raw_fid": 1.0,
        },
        "data_gap_warning": (
            "1P_p3 sims have IDENTICAL large_scale arrays across all parameter "
            "levels. σ_truth trend vs A_SN1 cannot be computed. "
            "Stop condition §7.3/§7.4 applies."
            if all_identical else "Truth data available and distinct."
        ),
        "all_1p_identical": bool(all_identical),
        "truth_sigma_max_diff": float(max_diff),
        "obs_names": obs_names,
        "sigma_inter_fid": sigma_inter_fid.tolist(),
        "sigma_intra_fid": sigma_intra_fid.tolist(),
        "J_log_sigma_inter_asn1": J_log_sigma_inter[:, ASN1_PIDX].tolist(),
        "J_log_sigma_intra_asn1": J_log_sigma_intra[:, ASN1_PIDX].tolist(),
        "focus_obs": focus_pass,
        "truth_levels": truth_levels,
        "truth_sigma_by_level": {
            s: {"sigma": truth_results[s]["sigma"], "N": truth_results[s]["N"]}
            for s in truth_levels
        },
        "pass_condition": not all_identical,
        "pass_note": (
            "FAIL — stop condition §7.3/§7.4: 1P_p3 hydro cutouts identical, "
            "σ_truth trend unavailable" if all_identical else "PASS"
        ),
    }

    out_json = OUT_DIR / "fig_truth_scatter_vs_ASN1.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[fig4] wrote {out_json}")

    if all_identical:
        print(f"\n[fig4] STOP CONDITION §7.3/§7.4 FIRED — writing CONVINCINGNESS_UPDATE.md")
    print("[fig4] DONE")


if __name__ == "__main__":
    main()
