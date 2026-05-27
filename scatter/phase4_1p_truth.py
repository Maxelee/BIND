"""scatter/phase4_1p_truth.py

Phase 4: 1P truth cross-check for dq_DM scatter.

Computes sigma_truth(dq_DM) = std of (q_DM_hydro - q_DMO) from actual simulation
maps across the 1P_p1_* (Omega_m) and 1P_p7_* (Omega_b) suites.

No model inference — purely from ground-truth maps.

Usage:
    python scatter/phase4_1p_truth.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import bootstrap as scipy_bootstrap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fd_jacobian_cv import (
    MPC_PER_PIX, PATCH_PIX,
    axis_ratio_q, r200c_mpc_h,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_1P   = Path("/mnt/home/mlee1/ceph/fm_testsuite/1P")
OUT_DIR   = ROOT / "outputs" / "scatter_diagnostics"
FIG_DIR   = ROOT / "figures" / "scatter_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SNAP      = "snap_090"
MASS_DIR  = "mass_threshold_1p000e13"

# BIND inter-scatter Jacobian at fiducial (from J_mean_and_scatter.npz, K=5 run)
BIND_J_INTER_OM = -0.8642   # d log sigma_inter(dq_DM) / d Omega_m_norm
BIND_J_INTER_OB = +0.7189   # d log sigma_inter(dq_DM) / d Omega_b_norm
SIGMA_INTER_FID  = 0.0355   # fiducial sigma_inter(dq_DM) at Omega_m=0.3

# Param min/max for normalisation (from norm_stats — approximated from 1P range)
# Omega_m: 1P_p1 spans 0.1 to 0.5, fiducial 0.3
# Omega_b: 1P_p7 spans 0.029 to 0.069, fiducial 0.049
# We use the fiducial-centred FD step implicitly via a linear fit in raw units.


# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute dq_DM for all halos in one sim directory
# ──────────────────────────────────────────────────────────────────────────────

def compute_dqDM_truth(sim_dir: Path):
    """Return (dq_DM array, param_vector) for all halos in a 1P sim."""
    cut_path = sim_dir / SNAP / MASS_DIR / "halo_cutouts.npz"
    cat_path = sim_dir / SNAP / MASS_DIR / "halo_catalog.npz"
    if not (cut_path.exists() and cat_path.exists()):
        return None, None

    cuts = np.load(cut_path)
    cat  = np.load(cat_path)

    cond  = cuts["condition"].astype(np.float64)    # (N, 128, 128)  DMO
    ls    = cuts["large_scale"].astype(np.float64)  # (N, 3, 128, 128) hydro

    if "radii" in cat.files:
        r200_pix = cat["radii"].astype(np.float64) / 1000.0 / MPC_PER_PIX
    else:
        r200_pix = r200c_mpc_h(cat["halo_masses"]) / MPC_PER_PIX

    params = cat["params"][0]   # (35,) — same for all halos in this sim
    N = cond.shape[0]

    dq_arr = np.full(N, np.nan)
    for i in range(N):
        r_aper = max(min(r200_pix[i], PATCH_PIX / 2 - 2), 4.0)
        q_dmo  = axis_ratio_q(np.maximum(cond[i], 0.0), r_aper)
        q_dm   = axis_ratio_q(np.maximum(ls[i, 0], 0.0), r_aper)   # DM channel
        dq_arr[i] = q_dm - q_dmo

    return dq_arr, params


def sigma_and_ci(arr, n_boot=2000, ci_level=0.68):
    """Return (sigma_truth, sigma_lo, sigma_hi) via bootstrap."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan, np.nan, np.nan
    sigma = float(np.std(arr, ddof=1))
    if len(arr) < 4:
        return sigma, np.nan, np.nan
    rng = np.random.default_rng(0)
    def _std(x):
        return np.std(x, ddof=1)
    boots = [_std(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * (1 - ci_level) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci_level) / 2))
    return sigma, lo, hi


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    results = {}

    for param_tag, sim_names, param_idx in [
        ("Omega_m", ["1P_p1_n2","1P_p1_n1","1P_p1_0","1P_p1_1","1P_p1_2"], 0),
        ("Omega_b", ["1P_p7_n2","1P_p7_n1","1P_p7_1","1P_p7_2"], 6),
    ]:
        print(f"\n=== {param_tag} 1P truth ===")
        rows = []
        for name in sim_names:
            sim_dir = BASE_1P / name
            dq, params = compute_dqDM_truth(sim_dir)
            if dq is None:
                print(f"  {name}: MISSING")
                continue
            theta_val  = float(params[param_idx])
            N_h        = int(np.isfinite(dq).sum())
            sig, lo, hi = sigma_and_ci(dq)
            mean_dq    = float(np.nanmean(dq))
            print(f"  {name}: {param_tag}={theta_val:.3f}  N={N_h}  "
                  f"sigma={sig:.4f} [{lo:.4f}, {hi:.4f}]  mean_dq={mean_dq:+.4f}")
            rows.append(dict(sim=name, theta=theta_val, N=N_h,
                             sigma_truth=sig, ci_lo=lo, ci_hi=hi,
                             mean_dq=mean_dq))
        results[param_tag] = rows

    # ── Compute truth Jacobian (finite-difference in raw units) ───────────────
    def truth_jacobian_raw(rows, fid_val, param_tag):
        """d log sigma / d theta_raw at fiducial using central FD on closest points.

        Uses a 1% tolerance to treat sims with theta ≈ fid as 'at fiducial' and
        skips them from the central-difference pair selection.
        """
        TOL = 0.01 * abs(fid_val) + 1e-6   # 1% relative + small absolute
        recs = [(r["theta"], r) for r in rows
                if np.isfinite(r.get("sigma_truth", np.nan))
                and abs(r["theta"] - fid_val) > TOL]
        recs.sort(key=lambda x: x[0])
        below = [(v, r) for v, r in recs if v < fid_val]
        above = [(v, r) for v, r in recs if v > fid_val]
        if not below or not above:
            return np.nan
        v_lo, r_lo = below[-1]
        v_hi, r_hi = above[0]
        if r_lo["sigma_truth"] <= 0 or r_hi["sigma_truth"] <= 0:
            return np.nan
        dlog = np.log(r_hi["sigma_truth"]) - np.log(r_lo["sigma_truth"])
        dv   = v_hi - v_lo
        return dlog / dv

    print("\n=== Truth Jacobian (d log sigma_truth / d theta_raw) ===")
    for param_tag, fid_val in [("Omega_m", 0.3), ("Omega_b", 0.049)]:
        rows = results.get(param_tag, [])
        if not rows:
            continue
        j_raw = truth_jacobian_raw(rows, fid_val, param_tag)
        print(f"  {param_tag}: d log sigma_truth(dq_DM) / d {param_tag}_raw = {j_raw:+.3f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = OUT_DIR / "phase4_1p_truth.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[phase4] wrote {out}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Phase 4 — Truth σ(dq_DM) vs parameter (1P sims)", fontsize=12)

    for ax, param_tag, fid_val, bind_J, param_label in [
        (axes[0], "Omega_m", 0.3, BIND_J_INTER_OM, r"$\Omega_m$"),
        (axes[1], "Omega_b", 0.049, BIND_J_INTER_OB, r"$\Omega_b$"),
    ]:
        rows = results.get(param_tag, [])
        if not rows:
            ax.set_visible(False)
            continue
        thetas = np.array([r["theta"] for r in rows])
        sigs   = np.array([r["sigma_truth"] for r in rows])
        lo_arr = np.array([r["ci_lo"] for r in rows])
        hi_arr = np.array([r["ci_hi"] for r in rows])
        Ns     = np.array([r["N"] for r in rows])

        finite = np.isfinite(sigs) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
        if finite.any():
            ax.errorbar(thetas[finite], sigs[finite],
                        yerr=[sigs[finite]-lo_arr[finite], hi_arr[finite]-sigs[finite]],
                        fmt="o-", color="steelblue", capsize=4, label="Truth σ(dq_DM)")
        # mark N values
        for i, r in enumerate(rows):
            if np.isfinite(r["sigma_truth"]):
                ax.annotate(f"N={r['N']}", (r["theta"], r["sigma_truth"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)

        # BIND prediction curve
        if np.isfinite(bind_J):
            # Need to convert raw J to normalised: d log sigma / d theta_norm
            # theta_norm ≈ (theta - fid) / (norm_scale); use approximate scale
            # from 1P range: Omega_m spans 0.1–0.5 so norm_scale ≈ 0.4/2 per eps=0.05 unit
            # We just use the raw-unit truth Jacobian for the overlay
            rows_valid = [r for r in rows if np.isfinite(r["sigma_truth"])]
            if len(rows_valid) >= 2:
                t_range = np.linspace(thetas.min(), thetas.max(), 100)
                # rough normalisation: param range 0.1–0.5 → [0,1], fid=0.3→0.5
                # eps=0.05 in norm space; raw eps = 0.05 * 0.4 = 0.02 for Omega_m
                # -> raw J = bind_J / 0.4 for Omega_m range  (0.069-0.029=0.04 for Ob)
                raw_scale = {"Omega_m": 0.4, "Omega_b": 0.04}[param_tag]
                bind_J_raw = bind_J / raw_scale
                sigma_bind = SIGMA_INTER_FID * np.exp(bind_J_raw * (t_range - fid_val))
                ax.plot(t_range, sigma_bind, "--", color="tomato",
                        label=f"BIND pred (J={bind_J:.3f})")

        ax.axvline(fid_val, color="gray", linestyle=":", alpha=0.5, label="fiducial")
        ax.set_xlabel(param_label)
        ax.set_ylabel(r"$\sigma_{\rm truth}({\rm d}q_{\rm DM})$")
        ax.set_title(param_label)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    pdf_path = FIG_DIR / "fig_dqDM_1ptruth.pdf"
    png_path = FIG_DIR / "fig_dqDM_1ptruth.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"[phase4] wrote {pdf_path}")
    print(f"[phase4] wrote {png_path}")

    # ── PROGRESS log ─────────────────────────────────────────────────────────
    from datetime import datetime
    prog = OUT_DIR / "PROGRESS.log"
    with open(prog, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} | Phase 4 complete."
                f" Results in {out}\n")


if __name__ == "__main__":
    main()
