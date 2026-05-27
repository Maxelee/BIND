"""fig3_cosmo_DMO_decomposition.py — Figure 3 of BIND convincingness brief.

Decompose the Omega_m mean Jacobian into:
  J_cond  — fixed-DMO FD (from J_mean_and_scatter.npz, Phase 1)
  J_DMO   — DMO-structure FD (1P_p1 patches + fiducial params)
  J_full  — total FD (1P_p1 patches + actual params)
  J_truth — 1P truth (from fig_bind_feedback_mean_response.json)

For observables: dq_DM and f_b (primary); also M_dm and M_gas for sanity check.
Plot: 4 bars per panel cell (J_cond, J_DMO, J_full, J_truth) with error bars.

Output:
  figures/scatter_diagnostics/fig_cosmology_DMO_decomposition.pdf / .png
  outputs/scatter_diagnostics/fig_cosmology_DMO_decomposition.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from data import NormStats, log_transform
from train import FlowMatchingLit
from scatter.measure_scatter import measure_scatter, ALL_OBS_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).parent.parent
RUN_DIR   = pathlib.Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
TEST_BASE = pathlib.Path("/mnt/home/mlee1/ceph/fm_testsuite/1P")
FIG_DIR   = BASE_DIR / "figures/scatter_diagnostics"
OUT_DIR   = BASE_DIR / "outputs/scatter_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MPC_PER_PIX = 0.048828125
SUB_DIR     = "snap_090/mass_threshold_1p000e13"

# Phase 1 + figure 1 artefacts
JMEAN_FILE = BASE_DIR / "scatter/J_mean_and_scatter.npz"
FIG1_JSON  = OUT_DIR / "fig_bind_feedback_mean_response.json"

# Reproducibility
NOISE_SEED  = 42
K_SAMPLES   = 5
N_STEPS     = 20
BATCH_SIZE  = 16

# Headline observables for the figure
FOCUS_OBS = ["dq_DM", "f_b", "M_dm", "M_gas"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize_params_fid(p_raw: np.ndarray, ns: NormStats) -> np.ndarray:
    """Raw param vector → normalized [0,1] vector."""
    _p = np.where(ns.param_log_flag == 1,
                  np.log10(np.maximum(p_raw.astype(float), 1e-30)), p_raw.astype(float))
    return ((_p - ns.param_min) / (ns.param_max - ns.param_min + 1e-8)).astype(np.float32)


def load_1p_sim(sim_name: str, ns: NormStats) -> dict | None:
    """Load + normalize a 1P sim's halo cutouts. Returns None if absent."""
    base = TEST_BASE / sim_name / SUB_DIR
    cat_path = base / "halo_catalog.npz"
    cut_path = base / "halo_cutouts.npz"
    if not cat_path.exists() or not cut_path.exists():
        print(f"  MISSING: {base}")
        return None
    cat = np.load(cat_path, allow_pickle=True)
    cut = np.load(cut_path, allow_pickle=True)

    # CAMELS bug: p14 = 0 for CV/1P runs
    params = cat["params"].copy().astype(np.float32)
    params[:, 14] = 0.0

    cond_raw = cut["condition"].astype(np.float32)   # (N, 128, 128)
    ls_raw   = cut["large_scale"].astype(np.float32) # (N, 3, 128, 128)

    cond_norm = (log_transform(cond_raw) - ns.cond_mean) / (ns.cond_std + 1e-8)
    ls_norm   = (log_transform(ls_raw) - ns.ls_mean[:, None, None]) / (ns.ls_std[:, None, None] + 1e-8)

    radii = cat["radii"] / 1000.0 / MPC_PER_PIX  # kpc/h → Mpc/h → pixels
    masses = cat["halo_masses"].astype(np.float64)
    omega_m = params[:, 0].astype(np.float64)

    return {
        "cond_raw":  cond_raw,
        "cond_norm": cond_norm[:, np.newaxis],  # (N, 1, 128, 128)
        "ls_norm":   ls_norm,
        "params":    params,
        "masses":    masses,
        "radii_pix": radii,
        "omega_m":   omega_m,
        "N":         len(masses),
    }


def run_scatter_for_params(
    model_fm, ns, theta_norm, data, device_str
) -> dict:
    """Run measure_scatter for a single theta_norm on a dataset."""
    return measure_scatter(
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


def grand_mean_and_se(r: dict, obs_idx: int) -> tuple[float, float]:
    """Grand mean of Y_bar across all halos, with bootstrap SE."""
    ybar = r["Y_bar"][:, obs_idx]  # (N_h,)
    finite = np.isfinite(ybar)
    n = finite.sum()
    if n < 2:
        return np.nan, np.nan
    m = float(np.mean(ybar[finite]))
    se = float(np.std(ybar[finite], ddof=1) / np.sqrt(n))
    return m, se


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fig3] device = {device_str}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = RUN_DIR / "checkpoints/last.ckpt"
    ns   = NormStats.load(RUN_DIR / "norm_stats.npz")
    print(f"[fig3] loading model from {ckpt}")
    lit = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device_str)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm
    model_fm.model.eval()

    # ── Load Phase 1 J_cond (mean Jacobian at fixed DMO) ──────────────────────
    jm = np.load(JMEAN_FILE, allow_pickle=True)
    obs_names   = jm["obs_names"].tolist()
    # param_names are stored in phase1_intra_jacobian.npz, not in J_mean_and_scatter.npz
    phase1 = np.load(BASE_DIR / "outputs/scatter_diagnostics/phase1_intra_jacobian.npz", allow_pickle=True)
    param_names = phase1["param_names"].tolist()
    J_cond     = jm["J_mean"][:, 0]       # (16,) for Omega_m (pidx=0)
    J_cond_se  = jm["J_mean_se"][:, 0]
    eps        = float(jm["eps"])
    print(f"[fig3] J_cond loaded: eps={eps}, obs={obs_names}")

    # ── Load J_truth from Figure 1 ───────────────────────────────────────────
    fig1 = json.loads(FIG1_JSON.read_text())
    om_data = fig1.get("Omega_m", {})
    if om_data.get("has_distinct_truth") and om_data.get("truth_J_mean"):
        J_truth    = np.array(om_data["truth_J_mean"])
        J_truth_se = np.array(om_data["truth_J_mean_se"])
        theta_lo   = float(om_data["theta_lo"])
        theta_hi   = float(om_data["theta_hi"])
        delta_theta_truth = theta_hi - theta_lo
        print(f"[fig3] J_truth loaded: delta_theta={delta_theta_truth:.3f}")
    else:
        print("[fig3] WARNING: J_truth not found in fig1 JSON, setting to NaN")
        J_truth    = np.full(len(obs_names), np.nan)
        J_truth_se = np.full(len(obs_names), np.nan)
        delta_theta_truth = 0.5

    # ── Load 1P_p1 data ───────────────────────────────────────────────────────
    print("\n[fig3] loading 1P_p1 data ...")
    data_n1 = load_1p_sim("1P_p1_n1", ns)
    data_hi = load_1p_sim("1P_p1_1",  ns)

    if data_n1 is None or data_hi is None:
        print("[fig3] STOP: 1P_p1 data not found — Figure 3 cannot be produced.")
        return

    # Normalized Omega_m values
    om_min, om_max = float(ns.param_min[0]), float(ns.param_max[0])
    theta_n1_norm = float(np.mean(data_n1["params"][:, 0] - om_min) / (om_max - om_min + 1e-8))
    theta_hi_norm = float(np.mean(data_hi["params"][:, 0] - om_min) / (om_max - om_min + 1e-8))
    theta_fid_norm = (0.3 - om_min) / (om_max - om_min + 1e-8)
    delta_theta = theta_hi_norm - theta_n1_norm  # should be ~0.5

    print(f"  n1 Omega_m actual: mean={data_n1['omega_m'].mean():.3f} → norm={theta_n1_norm:.3f}")
    print(f"  hi Omega_m actual: mean={data_hi['omega_m'].mean():.3f} → norm={theta_hi_norm:.3f}")
    print(f"  delta_theta = {delta_theta:.3f}, fiducial norm = {theta_fid_norm:.3f}")
    print(f"  n1 N_halos={data_n1['N']}, hi N_halos={data_hi['N']}")

    # ── Build theta_norm vectors ──────────────────────────────────────────────
    # Use the first halo's params as representative (all halos in a 1P sim
    # have the same params except for the varied parameter)
    theta_actual_n1 = normalize_params_fid(data_n1["params"][0], ns)
    theta_actual_hi = normalize_params_fid(data_hi["params"][0], ns)

    # Fiducial: same as actual_n1 but replace Omega_m with fiducial (0.3)
    theta_fid = theta_actual_n1.copy()
    theta_fid[0] = float((0.3 - om_min) / (om_max - om_min + 1e-8))

    print(f"\n  theta_actual_n1[0:5] = {theta_actual_n1[:5]}")
    print(f"  theta_actual_hi[0:5] = {theta_actual_hi[:5]}")
    print(f"  theta_fid[0:5]       = {theta_fid[:5]}")

    # ── Run 4 inference scenarios ─────────────────────────────────────────────
    # 1. J_DMO: n1 patches + fiducial params
    # 2. J_DMO: hi patches + fiducial params
    # 3. J_full: n1 patches + actual params
    # 4. J_full: hi patches + actual params

    t0 = time.time()
    print("\n[fig3] running scenario 1/4: n1 patches + fiducial params ...")
    r_dmo_n1 = run_scatter_for_params(model_fm, ns, theta_fid, data_n1, device_str)
    print(f"  done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("[fig3] running scenario 2/4: hi patches + fiducial params ...")
    r_dmo_hi = run_scatter_for_params(model_fm, ns, theta_fid, data_hi, device_str)
    print(f"  done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("[fig3] running scenario 3/4: n1 patches + actual params ...")
    r_full_n1 = run_scatter_for_params(model_fm, ns, theta_actual_n1, data_n1, device_str)
    print(f"  done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("[fig3] running scenario 4/4: hi patches + actual params ...")
    r_full_hi = run_scatter_for_params(model_fm, ns, theta_actual_hi, data_hi, device_str)
    print(f"  done ({time.time()-t0:.1f}s)")

    # ── Compute J_DMO and J_full ─────────────────────────────────────────────
    N_obs = len(obs_names)
    J_DMO  = np.full(N_obs, np.nan)
    J_full = np.full(N_obs, np.nan)
    J_DMO_se  = np.full(N_obs, np.nan)
    J_full_se = np.full(N_obs, np.nan)

    for i, oname in enumerate(obs_names):
        m_dmo_n1, se_dmo_n1 = grand_mean_and_se(r_dmo_n1, i)
        m_dmo_hi, se_dmo_hi = grand_mean_and_se(r_dmo_hi, i)
        m_full_n1, se_full_n1 = grand_mean_and_se(r_full_n1, i)
        m_full_hi, se_full_hi = grand_mean_and_se(r_full_hi, i)

        if np.isfinite(m_dmo_n1) and np.isfinite(m_dmo_hi) and delta_theta > 0:
            J_DMO[i] = (m_dmo_hi - m_dmo_n1) / delta_theta
            J_DMO_se[i] = np.sqrt(se_dmo_n1**2 + se_dmo_hi**2) / delta_theta
        if np.isfinite(m_full_n1) and np.isfinite(m_full_hi) and delta_theta > 0:
            J_full[i] = (m_full_hi - m_full_n1) / delta_theta
            J_full_se[i] = np.sqrt(se_full_n1**2 + se_full_hi**2) / delta_theta

    # ── Print decomposition table ─────────────────────────────────────────────
    print("\n=== Decomposition table: J_cond, J_DMO, J_full, J_truth ===")
    print(f"  {'obs':20s}  J_cond      J_DMO       J_full      J_truth    J_cond+J_DMO")
    for i, oname in enumerate(obs_names):
        approx_sum = J_cond[i] + J_DMO[i]
        print(f"  {oname:20s}: {J_cond[i]:+.4f}  {J_DMO[i]:+.4f}  {J_full[i]:+.4f}  {J_truth[i]:+.4f}  {approx_sum:+.4f}")

    # ── Diagnostic outcome ────────────────────────────────────────────────────
    print("\n=== Diagnostic outcome ===")
    focus_idxs = [obs_names.index(n) for n in FOCUS_OBS if n in obs_names]
    for i in focus_idxs:
        n = obs_names[i]
        jc, jd, jf, jt = J_cond[i], J_DMO[i], J_full[i], J_truth[i]
        approx_ok = abs((jc + jd) - jf) < 0.3 * abs(jf) + 0.05 if np.isfinite(jf) and jf != 0 else None
        full_truth_ok = abs(jf - jt) < 0.5 * abs(jt) + 0.05 if np.isfinite(jt) and jt != 0 else None
        dmo_dominates = abs(jd) > abs(jc) if np.isfinite(jd) and np.isfinite(jc) else None
        print(f"  {n}: chain_rule_approx={approx_ok}, full≈truth={full_truth_ok}, DMO_dominates={dmo_dominates}")
        if full_truth_ok and dmo_dominates:
            print(f"    → Outcome A: BIND fine, sign-flip is DMO-structure artefact")
        elif full_truth_ok and not dmo_dominates:
            print(f"    → Outcome B: Both contributions comparable")
        elif not full_truth_ok:
            print(f"    → Outcome C: J_full ≠ J_truth — non-linearity or model failure")

    # ── Plot ──────────────────────────────────────────────────────────────────
    focus_names = [n for n in FOCUS_OBS if n in obs_names]
    n_focus = len(focus_names)

    fig, axes = plt.subplots(1, n_focus, figsize=(5 * n_focus, 5), sharey=False)
    if n_focus == 1:
        axes = [axes]

    bar_labels  = [r"$J^{\rm cond}$", r"$J^{\rm DMO}$",
                   r"$J^{\rm full}$", r"$J^{\rm truth}$",
                   r"$J^{\rm cond}+J^{\rm DMO}$"]
    bar_colors  = ["steelblue", "darkorange", "green", "red", "purple"]
    bar_hatches = ["", "", "", "//", "xx"]

    for ax, obs_name in zip(axes, focus_names):
        oi = obs_names.index(obs_name)
        vals = [J_cond[oi], J_DMO[oi], J_full[oi], J_truth[oi], J_cond[oi] + J_DMO[oi]]
        errs = [J_cond_se[oi], J_DMO_se[oi], J_full_se[oi], J_truth_se[oi],
                np.sqrt(J_cond_se[oi]**2 + J_DMO_se[oi]**2)]

        x = np.arange(len(vals))
        for xi, (v, e, lbl, c, h) in enumerate(zip(vals, errs, bar_labels, bar_colors, bar_hatches)):
            ax.bar(xi, v, color=c, alpha=0.75, hatch=h, label=lbl if ax == axes[0] else "")
            if np.isfinite(e):
                ax.errorbar(xi, v, yerr=e, fmt="none", color="black", capsize=4, linewidth=1.5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{obs_name}", fontsize=12)
        ax.set_ylabel(r"$d\log\langle F\rangle / d\theta_{\rm norm}$", fontsize=10)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        r"Omega_m Jacobian decomposition: $J^{\rm full} \approx J^{\rm cond} + J^{\rm DMO}$?"
        "\n(dashed = chain-rule sum; red = truth 1P FD)",
        fontsize=11
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out_path = FIG_DIR / f"fig_cosmology_DMO_decomposition.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[fig3] saved {out_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "config": {
            "K": K_SAMPLES, "n_steps": N_STEPS, "seed": NOISE_SEED,
            "theta_n1_norm": float(theta_n1_norm),
            "theta_hi_norm": float(theta_hi_norm),
            "delta_theta": float(delta_theta),
            "N_n1": int(data_n1["N"]), "N_hi": int(data_hi["N"]),
        },
        "obs_names": obs_names,
        "J_cond":    J_cond.tolist(),
        "J_cond_se": J_cond_se.tolist(),
        "J_DMO":     J_DMO.tolist(),
        "J_DMO_se":  J_DMO_se.tolist(),
        "J_full":    J_full.tolist(),
        "J_full_se": J_full_se.tolist(),
        "J_truth":    J_truth.tolist(),
        "J_truth_se": J_truth_se.tolist(),
        "J_chain_rule_sum": (J_cond + J_DMO).tolist(),
        "focus_obs_decomposition": {
            n: {
                "J_cond": float(J_cond[obs_names.index(n)]),
                "J_DMO":  float(J_DMO[obs_names.index(n)]),
                "J_full": float(J_full[obs_names.index(n)]),
                "J_truth": float(J_truth[obs_names.index(n)]),
                "J_cond+J_DMO": float(J_cond[obs_names.index(n)] + J_DMO[obs_names.index(n)]),
            }
            for n in FOCUS_OBS if n in obs_names
        },
    }

    out_json = OUT_DIR / "fig_cosmology_DMO_decomposition.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[fig3] wrote {out_json}")
    print("[fig3] DONE")


if __name__ == "__main__":
    main()
