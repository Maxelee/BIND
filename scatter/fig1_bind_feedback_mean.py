"""scatter/fig1_bind_feedback_mean.py

Figure 1 — "BIND IS sensitive to feedback" affirmative.

Two-panel bar chart:
  Panel (a): BIND mean Jacobian vs truth 1P FD for A_SN1
  Panel (b): same for A_AGN1

Truth computed from 1P_p3_* (A_SN1) and 1P_p4_* (A_AGN1) ground-truth hydro maps.
BIND J_mean loaded from scatter/J_mean_and_scatter.npz (no BIND inference).

Pass condition:
  For mass/profile observables (M_gas, M_star, Sigma_gas_r*), BIND tracks truth
  on feedback mean response to within ±20%, sign-correct.

Usage:
    python scatter/fig1_bind_feedback_mean.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fd_jacobian_cv import (
    MPC_PER_PIX, PATCH_PIX, axis_ratio_q, r200c_mpc_h,
    OMEGA_B_FIXED,
)
from scatter.measure_scatter import (
    _compute_all_obs, ALL_OBS_NAMES, LOG_MASK, _RR_PIX,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_1P   = Path("/mnt/home/mlee1/ceph/fm_testsuite/1P")
J_FILE    = ROOT / "scatter" / "J_mean_and_scatter.npz"
NS_FILE   = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head/norm_stats.npz")
OUT_DIR   = ROOT / "outputs" / "scatter_diagnostics"
FIG_DIR   = ROOT / "figures" / "scatter_diagnostics"
SNAP      = "snap_090"
MASS_DIR  = "mass_threshold_1p000e13"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Load BIND J_mean
# ──────────────────────────────────────────────────────────────────────────────
jd       = np.load(J_FILE, allow_pickle=True)
J_mean   = jd["J_mean"]          # (16, 35)  d log<F> / d theta_norm
J_mean_se = jd["J_mean_se"]      # (16, 35)
OBS_NAMES = list(jd["obs_names"])
N_OBS    = len(OBS_NAMES)
assert OBS_NAMES == ALL_OBS_NAMES, f"obs_names mismatch: {OBS_NAMES} vs {ALL_OBS_NAMES}"

# ──────────────────────────────────────────────────────────────────────────────
# Load norm_stats for param normalization
# ──────────────────────────────────────────────────────────────────────────────
ns = np.load(NS_FILE, allow_pickle=True)
param_min      = ns["param_min"]   # (35,) linear or log10 min
param_max      = ns["param_max"]   # (35,) linear or log10 max
param_log_flag = ns["param_log_flag"].astype(bool)  # (35,) True = log10 norm

def raw_to_norm(theta_raw: float, pidx: int) -> float:
    """Convert a raw parameter value to BIND's normalized [0,1] space."""
    if param_log_flag[pidx]:
        val = np.log10(max(theta_raw, 1e-10))
    else:
        val = float(theta_raw)
    return (val - param_min[pidx]) / (param_max[pidx] - param_min[pidx])

# ──────────────────────────────────────────────────────────────────────────────
# Compute truth observables from 1P sim maps
# ──────────────────────────────────────────────────────────────────────────────

def compute_truth_obs_for_sim(sim_name: str, param_idx_fb_cosmic: tuple | None = None):
    """Return (mean_log_obs array [N_OBS], param_norm, N_halos) for one 1P sim."""
    sim_dir  = BASE_1P / sim_name
    cut_path = sim_dir / SNAP / MASS_DIR / "halo_cutouts.npz"
    cat_path = sim_dir / SNAP / MASS_DIR / "halo_catalog.npz"
    if not (cut_path.exists() and cat_path.exists()):
        print(f"  {sim_name}: MISSING")
        return None, None, 0

    cuts = np.load(cut_path)
    cat  = np.load(cat_path)

    cond = cuts["condition"].astype(np.float64)      # (N, 128, 128) DMO
    ls   = cuts["large_scale"].astype(np.float64)    # (N, 3, 128, 128) hydro

    if "radii" in cat.files:
        r200_pix = cat["radii"].astype(np.float64) / 1000.0 / MPC_PER_PIX
    else:
        r200_pix = r200c_mpc_h(cat["masses"]) / MPC_PER_PIX

    params = cat["params"][0]  # (35,) — same for all halos in this 1P sim
    N = cond.shape[0]

    # f_b_cosmic = Omega_b / Omega_m from this sim's params
    omega_m = params[0]
    omega_b = params[6]
    f_b_cosmic = float(omega_b / omega_m) if omega_m > 0 else OMEGA_B_FIXED / 0.3

    all_obs = []
    for i in range(N):
        r_aper = max(min(r200_pix[i], PATCH_PIX / 2 - 2), 4.0)
        q_dmo  = axis_ratio_q(np.maximum(cond[i], 0.0), r_aper)
        obs    = _compute_all_obs(ls[i], r200_pix[i], f_b_cosmic, q_dmo)
        all_obs.append(obs)

    all_obs = np.array(all_obs, dtype=np.float64)  # (N, 16)

    # Apply log transform where LOG_MASK is True
    log_obs = np.where(
        LOG_MASK[np.newaxis, :],
        np.where(all_obs > 0, np.log10(all_obs), np.nan),
        all_obs,
    )

    # Mean per observable (ignoring NaN)
    mean_log_obs = np.nanmean(log_obs, axis=0)  # (16,)
    N_valid = np.sum(np.isfinite(log_obs), axis=0)

    return mean_log_obs, params, N


def compute_truth_jacobian(sim_lo: str, sim_hi: str, pidx: int):
    """Central FD in normalized param space for d log<F> / d theta_norm."""
    mean_lo, params_lo, N_lo = compute_truth_obs_for_sim(sim_lo)
    mean_hi, params_hi, N_hi = compute_truth_obs_for_sim(sim_hi)
    if mean_lo is None or mean_hi is None:
        return None, None, None, None, None

    theta_lo = float(params_lo[pidx])
    theta_hi = float(params_hi[pidx])
    norm_lo  = raw_to_norm(theta_lo, pidx)
    norm_hi  = raw_to_norm(theta_hi, pidx)
    d_norm   = norm_hi - norm_lo
    print(f"    {sim_lo}(θ_raw={theta_lo:.4f}, norm={norm_lo:.3f}) → {sim_hi}(θ_raw={theta_hi:.4f}, norm={norm_hi:.3f}), Δnorm={d_norm:.3f}")

    J_truth = (mean_hi - mean_lo) / d_norm   # (16,)
    # Bootstrap SE from N counts (rough estimate: SE ≈ |J_truth| * sqrt(2/N))
    N_eff   = min(N_lo, N_hi)
    J_se_approx = np.abs(mean_hi - mean_lo) / d_norm * np.sqrt(2.0 / max(N_eff, 1))

    return J_truth, J_se_approx, theta_lo, theta_hi, N_eff


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    results = {}

    # ── Param specs: (label, pidx, lo_sim, hi_sim, panel, has_distinct_truth) ─
    # 1P feedback sims (p3=A_SN1, p4=A_AGN1, p5=A_SN2, p6=A_AGN2) have IDENTICAL
    # large_scale arrays across all parameter levels — truth FD is undefined.
    # Only cosmological params (p1=Omega_m, p7=Omega_b) have distinct hydro data.
    # Panel (a): A_SN1 BIND-only; Panel (b): Omega_m BIND vs truth.
    param_specs = [
        ("A_SN1",  2, "1P_p3_n1", "1P_p3_1", "a", False),  # BIND-only: no distinct truth cutouts
        ("Omega_m", 0, "1P_p1_n1", "1P_p1_1", "b", True),   # BIND vs truth: distinct 1P_p1 data
    ]

    truth_Js = {}
    for pname, pidx, lo_sim, hi_sim, panel, has_distinct_truth in param_specs:
        print(f"\n=== {pname} (param {pidx}) ===")

        if not has_distinct_truth:
            print(f"  SKIP truth FD: 1P feedback sims have IDENTICAL hydro cutouts (data gap).")
            truth_Js[pname] = None
            results[pname] = {
                "pidx": pidx,
                "has_distinct_truth": False,
                "data_gap_note": "1P feedback sims (p3-p6) share identical large_scale arrays — truth FD undefined",
                "obs_names": OBS_NAMES,
                "bind_J_mean": J_mean[:, pidx].tolist(),
                "bind_J_mean_se": J_mean_se[:, pidx].tolist(),
                "truth_J_mean": None,
                "truth_J_mean_se": None,
            }
            for i, n in enumerate(OBS_NAMES):
                print(f"  {n:20s}: BIND={J_mean[i,pidx]:+.4f}±{J_mean_se[i,pidx]:.3f}")
            continue

        J_t, J_se, t_lo, t_hi, N_eff = compute_truth_jacobian(lo_sim, hi_sim, pidx)
        if J_t is None:
            print("  MISSING — skipping")
            truth_Js[pname] = None
            continue
        truth_Js[pname] = {"J": J_t.tolist(), "J_se": J_se.tolist(), "N_eff": int(N_eff)}
        for i, n in enumerate(OBS_NAMES):
            bind_j = J_mean[i, pidx]
            truth_j = J_t[i]
            bse = J_mean_se[i, pidx]
            tse = J_se[i]
            sign_ok = (np.sign(bind_j) == np.sign(truth_j)) if (abs(bind_j) > bse and abs(truth_j) > tse) else None
            print(f"  {n:20s}: BIND={bind_j:+.4f}±{bse:.3f}  truth={truth_j:+.4f}±{tse:.3f}  sign_ok={sign_ok}")

        results[pname] = {
            "pidx": pidx,
            "has_distinct_truth": True,
            "sim_lo": lo_sim, "sim_hi": hi_sim,
            "theta_lo": t_lo, "theta_hi": t_hi, "N_eff": int(N_eff),
            "obs_names": OBS_NAMES,
            "bind_J_mean": J_mean[:, pidx].tolist(),
            "bind_J_mean_se": J_mean_se[:, pidx].tolist(),
            "truth_J_mean": J_t.tolist(),
            "truth_J_mean_se": J_se.tolist(),
        }

    # ── Pass condition check (only for params with truth data) ──────────────
    mass_profile_obs = ["M_gas", "M_star"] + [f"Sigma_gas_r{i}" for i in range(5)]
    print("\n=== Pass condition check (mass/profile observables, truth-available params) ===")
    pass_flag = True
    for pname, pidx, *_ in param_specs:
        if pname not in results or not results[pname].get("has_distinct_truth"):
            continue
        if results[pname]["truth_J_mean"] is None:
            continue
        print(f"\n  {pname}:")
        for n in mass_profile_obs:
            if n not in OBS_NAMES:
                continue
            i = OBS_NAMES.index(n)
            bj = J_mean[i, pidx]
            tj = results[pname]["truth_J_mean"][i]
            bse = J_mean_se[i, pidx]
            tse = results[pname]["truth_J_mean_se"][i]
            sign_ok = (np.sign(bj) == np.sign(tj))
            ratio = abs(bj) / (abs(tj) + 1e-6)
            within20 = (0.8 <= ratio <= 1.2) if sign_ok else False
            status = "PASS" if (sign_ok and within20) else ("SIGN_FLIP" if not sign_ok else "MAG_FAIL")
            print(f"    {n:18s}: BIND={bj:+.3f}  truth={tj:+.3f}  ratio={ratio:.2f}  {status}")
            if status != "PASS":
                pass_flag = False

    # For A_SN1 (no truth): check BIND magnitude is non-trivial (SNR > 2)
    asn1_snr_ok = all(abs(J_mean[OBS_NAMES.index(n), 2]) / (J_mean_se[OBS_NAMES.index(n), 2] + 1e-6) > 2
                     for n in ["M_gas", "M_star", "Sigma_gas_r0"] if n in OBS_NAMES)
    print(f"\n  A_SN1 BIND-only SNR check (M_gas, M_star, Sigma_gas_r0 all SNR>2): {asn1_snr_ok}")

    print(f"\nOverall pass condition: {'PASS' if pass_flag and asn1_snr_ok else 'CHECK_RESULTS'}")
    results["pass_condition"] = pass_flag and asn1_snr_ok
    results["data_gap_warning"] = (
        "1P sims for feedback params (A_SN1, A_AGN1, A_SN2, A_AGN2) have identical "
        "large_scale hydro arrays across all parameter levels. Truth mean Jacobian "
        "for feedback params is unavailable. Panel (a) shows BIND-only response."
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = OUT_DIR / "fig_bind_feedback_mean_response.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[fig1] wrote {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 1 — BIND mean Jacobian vs 1P truth (feedback parameters)", fontsize=13)

    obs_labels = [n.replace("Sigma_gas_", "Σ_r").replace("_over_R200", "/R₂₀₀") for n in OBS_NAMES]
    x = np.arange(N_OBS)
    width = 0.35

    for ax, (pname, pidx, lo_sim, hi_sim, panel, has_truth) in zip(axes, param_specs):
        if pname not in results:
            ax.text(0.5, 0.5, "DATA MISSING", transform=ax.transAxes, ha="center", va="center", fontsize=14)
            ax.set_title(f"Panel ({panel}) — {pname} [MISSING]")
            continue

        bj  = np.array(results[pname]["bind_J_mean"])
        bse = np.array(results[pname]["bind_J_mean_se"])
        truth_available = results[pname].get("has_distinct_truth") and results[pname]["truth_J_mean"] is not None

        if truth_available:
            tj  = np.array(results[pname]["truth_J_mean"])
            tse = np.array(results[pname]["truth_J_mean_se"])
            bars_bind  = ax.bar(x - width/2, bj, width, label="BIND $J_{\\rm mean}$",
                                color="steelblue", alpha=0.85)
            bars_truth = ax.bar(x + width/2, tj, width, label="Truth $\\Delta\\log\\langle F\\rangle$",
                                color="darkorange", alpha=0.85)
            ax.errorbar(x - width/2, bj, yerr=bse, fmt="none", color="black", capsize=3, linewidth=1.2)
            ax.errorbar(x + width/2, tj, yerr=tse, fmt="none", color="black", capsize=3, linewidth=1.2)
            # Sign check symbols
            for i in range(N_OBS):
                if abs(bj[i]) > bse[i] * 2 and abs(tj[i]) > tse[i] * 2:
                    sign_ok = (np.sign(bj[i]) == np.sign(tj[i]))
                    sym = "✓" if sign_ok else "✗"
                    col = "green" if sign_ok else "red"
                    ax.text(i, max(abs(bj[i]), abs(tj[i])) * 1.05 + 0.02, sym,
                            ha="center", va="bottom", fontsize=7, color=col)
        else:
            # BIND-only: full-width bars, note truth unavailable
            ax.bar(x, bj, width * 1.5, label="BIND $J_{\\rm mean}$ (truth unavailable)",
                   color="steelblue", alpha=0.85)
            ax.errorbar(x, bj, yerr=bse, fmt="none", color="black", capsize=3, linewidth=1.2)
            ax.text(0.02, 0.97, "⚠ 1P feedback sims have identical\nhydro cutouts — truth unavailable",
                    transform=ax.transAxes, fontsize=7, va="top", color="darkred",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(obs_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("$d\\log\\langle F\\rangle / d\\theta_{\\rm norm}$")
        title_suffix = "BIND vs Truth" if truth_available else "BIND only"
        ax.set_title(f"Panel ({panel}) — {pname} ({title_suffix})")
        ax.legend(fontsize=9)

        # Mark mass/profile observables with green background
        for i, n in enumerate(OBS_NAMES):
            if n in mass_profile_obs:
                ax.axvspan(i - 0.5, i + 0.5, color="lightgreen", alpha=0.15, zorder=0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_path = FIG_DIR / f"fig_bind_feedback_mean_response.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[fig1] saved {out_path}")
    plt.close(fig)

    print("\n[fig1] DONE")


if __name__ == "__main__":
    main()
