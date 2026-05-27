"""scatter/residual_pipeline.py
Driver for the BIND scatter-residual cross-correlation analysis.

Phases (per `BIND Scatter-Residual Analysis.md`):
  Phase 1 — assemble the observable table from cached CV truth / BIND-K10 caches.
  Phase 2 — LOWESS mean μ̂(log M) + running σ̂(log M); standardised residuals.
  Phase 3 — Correlation matrices, bootstrap SE, Frobenius null, eigen-alignment,
            per-halo Pearson diagonals, mass-binned ρ(ΔM_*, ΔM_gas).

Usage
-----
    python scatter/residual_pipeline.py                # run all phases end-to-end
    python scatter/residual_pipeline.py --phase 1      # only phase 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fd_jacobian_cv import load_cv_halos
from scatter.measure_scatter import ALL_OBS_NAMES
from scatter.residual import (
    OBS_8, OBS_7, OBS_MAP,
    extract_obs8,
    eigen_alignment,
    fit_mean_and_scatter,
    frobenius_norm, frobenius_null_distribution,
    per_halo_pearson_diagonal,
    residual_correlation_matrix,
    rho_in_mass_bin, rebalance_to_equal_counts,
    standardise_residuals,
)

# ---------------------------------------------------------------------------
# Paths (repo convention: scatter/ for data, paper_figures/ for figures)

CV_ROOT      = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
SCATTER_DIR  = Path("/mnt/home/mlee1/vdm_bind2/scatter")
OUT_DIR      = SCATTER_DIR / "scatter_residual"
FIG_DIR      = Path("/mnt/home/mlee1/vdm_bind2/paper_figures/scatter_residual")
TRUTH_CACHE  = SCATTER_DIR / "cv_truth_obs.npz"
BIND_CACHE   = SCATTER_DIR / "cv_bind_obs_K10.npz"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASS_THRESHOLD = 1e13   # M_sun/h (brief §3.1)


# ---------------------------------------------------------------------------
# Configuration / metadata helpers

def _config_dict() -> dict:
    return {
        "cv_root":              str(CV_ROOT),
        "truth_cache":          str(TRUTH_CACHE),
        "bind_cache":           str(BIND_CACHE),
        "obs_8":                OBS_8,
        "obs_7_primary":        OBS_7,
        "mass_threshold_msunh": MASS_THRESHOLD,
        "K_bind_samples":       10,
        "lowess_frac":          0.4,
        "bootstrap_B":          2000,
        "fit_source":           "combined",
        "mass_bin_edges":       [13.0, 13.3, 13.7, 14.8],
        "corr_method_matrix":   "spearman",
        "corr_method_paa":      "pearson",
    }


def _save_sidecar(path: Path, payload: dict) -> None:
    payload = {**payload, "config": _config_dict()}
    path.write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Phase 1 — build observable table

def phase1_build_table() -> dict:
    print("[Phase 1] Loading CV halos + cached observables...", flush=True)
    cv = load_cv_halos(CV_ROOT)
    cv["params"][:, 14] = 0.0  # CAMELS p14 bug fix (per PLAN_NOTES.md)

    sim_id_str = cv["sim_id"]
    masses     = cv["masses"]
    n_total    = len(masses)

    sim_id_int = np.array([int(s.split("_")[1]) for s in sim_id_str], dtype=np.int32)
    # halo_id within each sim is the position in that sim's catalogue
    halo_id = np.zeros(n_total, dtype=np.int32)
    for s in np.unique(sim_id_int):
        m = sim_id_int == s
        halo_id[m] = np.arange(int(m.sum()), dtype=np.int32)

    keep = masses > MASS_THRESHOLD
    n_kept = int(keep.sum())
    print(f"  total halos: {n_total};  passing mass cut > 1e13 M_sun/h: {n_kept}", flush=True)
    if n_kept < 100:
        raise RuntimeError(f"Hard halo-count requirement failed: only {n_kept} < 100 halos pass cut.")

    d_t = np.load(TRUTH_CACHE)
    d_b = np.load(BIND_CACHE)
    truth_raw = d_t["truth_obs"][keep]           # (N_h, 16)
    bind_raw  = d_b["obs_tensor"][keep]          # (N_h, K, 16)
    bind_masses_check = d_b["masses"][keep]
    if not np.allclose(masses[keep], bind_masses_check, rtol=1e-6):
        raise RuntimeError("Mass alignment mismatch between BIND cache and CV catalogue.")

    K = bind_raw.shape[1]
    assert K == 10, f"Expected K=10, got K={K}"

    # Project to 8 brief observables (with log10 where required)
    truth_F = extract_obs8(truth_raw, ALL_OBS_NAMES)             # (N_h, 8)
    bind_F  = extract_obs8(bind_raw,  ALL_OBS_NAMES)             # (N_h, K, 8)

    sim_id_kept  = sim_id_int[keep].astype(np.int32)
    halo_id_kept = halo_id[keep].astype(np.int32)
    M200c_kept   = masses[keep].astype(np.float64)

    # --- Build long-format table ---------------------------------------------------------
    M = n_kept * (1 + K)
    col_sim    = np.empty(M, dtype=np.int32)
    col_halo   = np.empty(M, dtype=np.int32)
    col_source = np.empty(M, dtype="U5")
    col_sample = np.empty(M, dtype=np.int32)
    col_M200c  = np.empty(M, dtype=np.float64)
    F_long     = np.empty((M, 8), dtype=np.float64)

    # truth rows (N_h)
    col_sim[:n_kept]    = sim_id_kept
    col_halo[:n_kept]   = halo_id_kept
    col_source[:n_kept] = "truth"
    col_sample[:n_kept] = 0
    col_M200c[:n_kept]  = M200c_kept
    F_long[:n_kept]     = truth_F

    # bind rows (N_h * K), inner index = sample, outer = halo
    for k in range(K):
        s, e = n_kept * (1 + k), n_kept * (2 + k)
        col_sim[s:e]    = sim_id_kept
        col_halo[s:e]   = halo_id_kept
        col_source[s:e] = "bind"
        col_sample[s:e] = k
        col_M200c[s:e]  = M200c_kept
        F_long[s:e]     = bind_F[:, k, :]

    obs_npz = OUT_DIR / "observables.npz"
    np.savez_compressed(
        obs_npz,
        sim_id=col_sim, halo_id=col_halo, source=col_source,
        sample_id=col_sample, M200c=col_M200c, F=F_long,
        obs_names=np.array(OBS_8),
        n_halos=n_kept, K=K,
    )
    print(f"  saved {obs_npz} ({M} rows × 8 obs)", flush=True)

    # --- Gate 1 report ----------------------------------------------------------
    log_M_DM_idx = OBS_8.index("log10_M_DM")
    med_truth = float(np.nanmedian(truth_F[:, log_M_DM_idx]))
    med_bind  = float(np.nanmedian(bind_F[..., log_M_DM_idx].mean(axis=1)))
    units_ok  = abs(med_truth - med_bind) < 0.05

    def _stats(arr: np.ndarray) -> dict:
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return {"n": 0, "median": np.nan, "mad": np.nan, "finite_frac": 0.0}
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        return {"n": int(a.size), "median": med, "mad": mad,
                "finite_frac": float(a.size / arr.size)}

    truth_stats = {OBS_8[i]: _stats(truth_F[:, i]) for i in range(8)}
    bind_stats  = {OBS_8[i]: _stats(bind_F[..., i]) for i in range(8)}

    # Required obs finite-ness check (no NaN/Inf in any required column for ALL rows)
    nan_inf_ok = bool(np.all(np.isfinite(truth_F[:, :7])) and np.all(np.isfinite(bind_F[..., :7])))

    gate1 = {
        "phase":              "1",
        "gate":               "Gate 1",
        "n_halos_post_cut":   n_kept,
        "n_halos_required":   100,
        "halo_count_ok":      n_kept >= 100,
        "K":                  K,
        "K_required":         10,
        "K_ok":               K == 10,
        "median_log10_M_DM_truth": med_truth,
        "median_log10_M_DM_bind":  med_bind,
        "median_diff":             med_truth - med_bind,
        "median_diff_ok":          units_ok,
        "no_nan_inf_primary":      nan_inf_ok,
        "truth_stats":             truth_stats,
        "bind_stats":              bind_stats,
    }
    gate1["PASS"] = bool(
        gate1["halo_count_ok"] and gate1["K_ok"]
        and gate1["median_diff_ok"] and gate1["no_nan_inf_primary"]
    )

    gate1_path = OUT_DIR / "gate1_report.json"
    _save_sidecar(gate1_path, gate1)
    print(f"  Gate 1 → {gate1['PASS']}; written {gate1_path}", flush=True)
    if not gate1["PASS"]:
        raise RuntimeError(f"Gate 1 failed: {gate1}")

    return {
        "n_halos":   n_kept, "K": K,
        "sim_id":    sim_id_kept, "halo_id": halo_id_kept,
        "M200c":     M200c_kept,
        "log_M":     np.log10(M200c_kept),
        "truth_F":   truth_F,
        "bind_F":    bind_F,
    }


# ---------------------------------------------------------------------------
# Phase 2 — LOWESS mean + standardised residuals

def phase2_residuals(table: dict) -> dict:
    print("[Phase 2] Fitting LOWESS mean + scatter per observable...", flush=True)
    log_M = table["log_M"]
    truth_F = table["truth_F"]
    bind_F  = table["bind_F"]
    K = table["K"]

    n_h = log_M.size
    bind_F_mean = np.nanmean(bind_F, axis=1)   # (N_h, 8) per-halo BIND sample-mean

    # Pre-allocate
    delta_truth = np.full((n_h, 8), np.nan)         # (N_h, 8)
    delta_bind  = np.full((n_h, K, 8), np.nan)      # (N_h, K, 8)
    delta_bind_mean = np.full((n_h, 8), np.nan)     # (N_h, 8)
    mu_curves = {}
    sigma_curves = {}
    mean_diffs = {}

    for i, name in enumerate(OBS_8):
        ms = fit_mean_and_scatter(
            log_mass_truth=log_M, f_truth=truth_F[:, i],
            log_mass_bind=log_M,  f_bind_mean=bind_F_mean[:, i],
            frac=0.4, fit_source="combined",
        )
        delta_truth[:, i] = standardise_residuals(log_M, truth_F[:, i], ms.mu, ms.sigma)
        for k in range(K):
            delta_bind[:, k, i] = standardise_residuals(log_M, bind_F[:, k, i], ms.mu, ms.sigma)
        delta_bind_mean[:, i] = standardise_residuals(log_M, bind_F_mean[:, i], ms.mu, ms.sigma)

        # Diagnostics: truth-only and bind-only mean lines (for sanity plot)
        ms_t = fit_mean_and_scatter(log_M, truth_F[:, i], log_M, bind_F_mean[:, i],
                                    frac=0.4, fit_source="truth")
        ms_b = fit_mean_and_scatter(log_M, truth_F[:, i], log_M, bind_F_mean[:, i],
                                    frac=0.4, fit_source="bind")
        x_grid = np.linspace(np.nanmin(log_M), np.nanmax(log_M), 100)
        mu_combined = ms.mu(x_grid)
        mu_truth_only = ms_t.mu(x_grid)
        mu_bind_only  = ms_b.mu(x_grid)
        sigma_combined = ms.sigma(x_grid)
        mu_curves[name]    = (x_grid, mu_combined, mu_truth_only, mu_bind_only)
        sigma_curves[name] = (x_grid, sigma_combined)
        mean_diffs[name]   = float(np.nanmax(np.abs(mu_truth_only - mu_bind_only)))

    # Gate 2 checks
    # With the brief's "combined" fit (§4.1), the LOWESS subtracts a shared mean
    # curve fit to (truth + bind). The pooled-cloud mean residual is ≈ 0 by
    # construction; per-source means inherit a ±(bias/2) offset that reflects
    # the known +5–7% stellar bias the brief explicitly leaves in (§4.3:
    # "log them but do not stop — they are physics, not a bug"). The gate
    # therefore checks the *combined* mean and logs per-source offsets.
    per_source_offsets = {}
    for label, arr in (("truth", delta_truth), ("bind", delta_bind.reshape(-1, 8))):
        for i, name in enumerate(OBS_8):
            mean_resid = float(np.nanmean(arr[:, i]))
            per_source_offsets[f"{label}/{name}"] = mean_resid
            if abs(mean_resid) > 0.05:
                print(f"  [info] mean residual {label}/{name} = {mean_resid:+.3f}  "
                      "(bias retained per brief §4.3; cancels in correlations)",
                      flush=True)
    # The LOWESS fit pool is (truth + bind-per-halo-mean), one entry per halo
    # per source. The pool's mean residual is ≈ 0 by construction; this is
    # what we verify. (Per-source offsets reflect the +5-7% stellar bias.)
    fit_pool_means = []
    for i, name in enumerate(OBS_8):
        bm = np.nanmean(delta_bind[..., i], axis=1)  # per-halo bind sample-mean residual
        pooled = np.concatenate([delta_truth[:, i], bm])
        fit_pool_means.append(float(np.nanmean(pooled)))
    means_zero_ok = all(abs(m) < 0.05 for m in fit_pool_means)
    if not means_zero_ok:
        for i, name in enumerate(OBS_8):
            if abs(fit_pool_means[i]) >= 0.05:
                print(f"  [warn] fit-pool mean residual {name} = {fit_pool_means[i]:+.3f}",
                      flush=True)

    n_truth_rows = int(np.isfinite(delta_truth[:, 0]).sum())
    n_bind_rows  = int(np.isfinite(delta_bind[..., 0]).sum())

    cardinality_ok = (n_truth_rows == n_h) and (n_bind_rows == n_h * K)
    if not cardinality_ok:
        print(f"  [warn] cardinality: truth={n_truth_rows} (expected {n_h}); "
              f"bind={n_bind_rows} (expected {n_h*K})", flush=True)

    # Save residuals table
    residuals_npz = OUT_DIR / "residuals.npz"
    np.savez_compressed(
        residuals_npz,
        sim_id=table["sim_id"], halo_id=table["halo_id"], M200c=table["M200c"],
        log_M=log_M,
        delta_truth=delta_truth,
        delta_bind=delta_bind,
        delta_bind_mean=delta_bind_mean,
        obs_names=np.array(OBS_8),
    )
    print(f"  saved {residuals_npz}", flush=True)

    # Sanity plot
    sanity_plot(mu_curves, log_M, truth_F, bind_F_mean, OBS_8,
                FIG_DIR / "mean_lines_sanity")
    print(f"  saved sanity plot to {FIG_DIR / 'mean_lines_sanity.pdf'}", flush=True)

    gate2 = {
        "phase":            "2",
        "gate":             "Gate 2",
        "n_halos":          n_h,
        "K":                K,
        "cardinality_ok":   bool(cardinality_ok),
        "fit_pool_mean_zero_ok": bool(means_zero_ok),
        "fit_pool_mean_per_obs": dict(zip(OBS_8, fit_pool_means)),
        "per_source_offsets": per_source_offsets,
        "max_truth_bind_mean_diff_per_obs": mean_diffs,
        "any_diff_above_0p10_dex": any(v > 0.10 for v in mean_diffs.values()),
    }
    gate2["PASS"] = bool(gate2["cardinality_ok"] and gate2["fit_pool_mean_zero_ok"])
    _save_sidecar(OUT_DIR / "gate2_report.json", gate2)
    print(f"  Gate 2 → {gate2['PASS']}", flush=True)
    if not gate2["PASS"]:
        raise RuntimeError(f"Gate 2 failed: {gate2}")

    return {
        "log_M": log_M,
        "delta_truth": delta_truth,
        "delta_bind": delta_bind,
        "delta_bind_mean": delta_bind_mean,
        "mu_curves": mu_curves,
        "sigma_curves": sigma_curves,
    }


# ---------------------------------------------------------------------------
# Phase 3 — correlation matrices and statistics

def phase3_statistics(residuals: dict) -> dict:
    print("[Phase 3] Correlation matrices + bootstrap + null distribution...", flush=True)
    log_M = residuals["log_M"]
    delta_truth = residuals["delta_truth"]
    delta_bind_mean = residuals["delta_bind_mean"]
    delta_bind = residuals["delta_bind"]

    idx7 = [OBS_8.index(n) for n in OBS_7]
    idx8 = list(range(8))

    delta_T7 = delta_truth[:, idx7]
    delta_G7 = delta_bind_mean[:, idx7]
    delta_T8 = delta_truth
    delta_G8 = delta_bind_mean

    rng_seeds = {"T7": 100, "G7": 101, "T8": 102, "G8": 103, "null": 200, "Paa": 201}

    print("  bootstrap C_T (7x7, Spearman, B=2000)...", flush=True); t0 = time.time()
    C_T, SE_T = residual_correlation_matrix(delta_T7, method="spearman",
                                            n_boot=2000, rng_seed=rng_seeds["T7"])
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    print("  bootstrap C_G (7x7, Spearman, B=2000)...", flush=True); t0 = time.time()
    C_G, SE_G = residual_correlation_matrix(delta_G7, method="spearman",
                                            n_boot=2000, rng_seed=rng_seeds["G7"])
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    print("  bootstrap C_T_full / C_G_full (8x8)...", flush=True); t0 = time.time()
    C_T_full, SE_T_full = residual_correlation_matrix(delta_T8, method="spearman",
                                                      n_boot=2000, rng_seed=rng_seeds["T8"])
    C_G_full, SE_G_full = residual_correlation_matrix(delta_G8, method="spearman",
                                                      n_boot=2000, rng_seed=rng_seeds["G8"])
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    # P_pair: cross-correlation between paired truth and BIND-sample-mean residuals
    # Diagonal entries are the key per-halo agreement statistic.
    print("  per-halo Pearson diagonal (P_aa)...", flush=True); t0 = time.time()
    P_aa, P_aa_SE = per_halo_pearson_diagonal(delta_T7, delta_G7,
                                              n_boot=2000, rng_seed=rng_seeds["Paa"])
    # Off-diagonal P_pair: corr of Δ^T_a, Δ^G_b across halos
    n7 = len(OBS_7)
    P_pair = np.full((n7, n7), np.nan)
    for a in range(n7):
        for b in range(n7):
            mask = np.isfinite(delta_T7[:, a]) & np.isfinite(delta_G7[:, b])
            if mask.sum() < 5:
                continue
            from scipy.stats import pearsonr
            P_pair[a, b] = float(pearsonr(delta_T7[mask, a], delta_G7[mask, b])[0])
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    # Frobenius distance + null
    D = frobenius_norm(C_T - C_G)
    D8 = frobenius_norm(C_T_full - C_G_full)
    print("  Frobenius null distribution (B=2000, split-half truth)...", flush=True); t0 = time.time()
    null = frobenius_null_distribution(delta_T7, method="spearman",
                                       n_boot=2000, rng_seed=rng_seeds["null"])
    pval = float((null >= D).mean())
    print(f"    D={D:.4f}, null median={np.median(null):.4f}, p={pval:.4f}; "
          f"{time.time()-t0:.1f}s", flush=True)

    # Eigen-alignment
    eig = eigen_alignment(C_T, C_G)
    eig["eig_T"] = eig["eig_T"].tolist()
    eig["eig_G"] = eig["eig_G"].tolist()

    # Per-pair z-scores
    Z = (C_T - C_G) / np.sqrt(SE_T ** 2 + SE_G ** 2 + 1e-30)
    Z[~np.isfinite(C_T - C_G)] = np.nan
    flagged = []
    for a in range(n7):
        for b in range(a + 1, n7):
            z = float(Z[a, b])
            if np.isfinite(z) and abs(z) > 2.0:
                flagged.append(dict(a=OBS_7[a], b=OBS_7[b],
                                    C_T=float(C_T[a, b]), C_G=float(C_G[a, b]),
                                    z=z))

    # Mass-binned ρ(ΔM_*, ΔM_gas)
    bin_edges = np.array([13.0, 13.3, 13.7, 14.8])
    idx_star = OBS_7.index("log10_M_star")
    idx_gas  = OBS_7.index("log10_M_gas")
    rho_truth = rho_in_mass_bin(log_M, delta_T7[:, idx_star], delta_T7[:, idx_gas],
                                bin_edges, method="spearman",
                                n_boot=2000, rng_seed=300)
    rho_bind  = rho_in_mass_bin(log_M, delta_G7[:, idx_star], delta_G7[:, idx_gas],
                                bin_edges, method="spearman",
                                n_boot=2000, rng_seed=301)
    bin_edges_used = bin_edges
    # Rebalance if any bin has < 20 halos
    if any(r["n"] < 20 for r in rho_truth + rho_bind):
        bin_edges_used = rebalance_to_equal_counts(log_M, 3)
        print(f"  rebalancing mass bins to equal-count: {bin_edges_used.tolist()}", flush=True)
        rho_truth = rho_in_mass_bin(log_M, delta_T7[:, idx_star], delta_T7[:, idx_gas],
                                    bin_edges_used, method="spearman",
                                    n_boot=2000, rng_seed=300)
        rho_bind  = rho_in_mass_bin(log_M, delta_G7[:, idx_star], delta_G7[:, idx_gas],
                                    bin_edges_used, method="spearman",
                                    n_boot=2000, rng_seed=301)

    # --- Gate 3 sanity checks ----------------------------------------------------------
    sanity_signs_ok = True
    qDM_idx = OBS_7.index("q_DM"); qstar_idx = OBS_7.index("q_star")
    c_qdm_qstar = float(C_T[qDM_idx, qstar_idx])
    if c_qdm_qstar < 0:
        sanity_signs_ok = False

    rho_lo = rho_truth[0]["rho"]; rho_hi = rho_truth[-1]["rho"]
    rho_mass_trend_ok = bool(np.isfinite(rho_lo) and np.isfinite(rho_hi) and rho_hi <= rho_lo)
    rho_mass_warning = None
    if not rho_mass_trend_ok:
        rho_mass_warning = (f"truth ρ(ΔM*, ΔM_gas) trend reversed: "
                            f"bin_lo={rho_lo:.3f}, bin_hi={rho_hi:.3f}")
        print(f"  [warn] {rho_mass_warning}", flush=True)

    frob_finite_ok = bool(np.isfinite(D) and D < 7.0)
    eig_T_pos = (np.min(eig["eig_T"]) > -1e-8)
    eig_G_pos = (np.min(eig["eig_G"]) > -1e-8)
    eig_pos_ok = bool(eig_T_pos and eig_G_pos)

    np.savez_compressed(
        OUT_DIR / "matrices.npz",
        C_T=C_T, C_G=C_G, SE_T=SE_T, SE_G=SE_G,
        C_T_full=C_T_full, C_G_full=C_G_full,
        SE_T_full=SE_T_full, SE_G_full=SE_G_full,
        P_pair=P_pair, P_aa=P_aa, P_aa_SE=P_aa_SE,
        Z=Z,
        obs_7=np.array(OBS_7), obs_8=np.array(OBS_8),
    )
    np.save(OUT_DIR / "frobenius_null.npy", null)

    stats = {
        "D_primary_7x7":           float(D),
        "D_supplementary_8x8":     float(D8),
        "frobenius_null_median":   float(np.median(null)),
        "frobenius_null_p_value":  pval,
        "leading_eigenvector_angle_deg": eig["leading_eigenvector_angle_deg"],
        "eig_ratio_top":           eig["eig_ratio_top"],
        "eig_T":                   eig["eig_T"],
        "eig_G":                   eig["eig_G"],
        "P_aa":                    {OBS_7[i]: float(P_aa[i])    for i in range(n7)},
        "P_aa_SE":                 {OBS_7[i]: float(P_aa_SE[i]) for i in range(n7)},
        "z_above_2":               flagged,
        "rho_truth_mass_bins":     rho_truth,
        "rho_bind_mass_bins":      rho_bind,
        "mass_bin_edges":          bin_edges_used.tolist(),
        "sanity_qdm_qstar_C_T":    c_qdm_qstar,
        "sanity_qdm_qstar_ok":     bool(sanity_signs_ok),
        "rho_mass_trend_ok":       bool(rho_mass_trend_ok),
        "rho_mass_trend_warning":  rho_mass_warning,
        "frob_finite_ok":          frob_finite_ok,
        "eig_T_positive_ok":       bool(eig_T_pos),
        "eig_G_positive_ok":       bool(eig_G_pos),
    }
    _save_sidecar(OUT_DIR / "stats.json", stats)
    _save_sidecar(OUT_DIR / "mass_dependence.json",
                  {"rho_truth": rho_truth, "rho_bind": rho_bind,
                   "bin_edges": bin_edges_used.tolist()})

    # Gate 3
    gate3 = {
        "phase":             "3",
        "gate":              "Gate 3",
        "sanity_qdm_qstar":  sanity_signs_ok,
        "rho_mass_trend":    rho_mass_trend_ok,
        "frob_finite":       frob_finite_ok,
        "eig_positive":      eig_pos_ok,
    }
    # Per brief §5.4: sanity 1 is a hard stop, sanity 2 is a warning, 3 and 4 are hard
    gate3["PASS"] = bool(sanity_signs_ok and frob_finite_ok and eig_pos_ok)
    _save_sidecar(OUT_DIR / "gate3_report.json", gate3)
    print(f"  Gate 3 → {gate3['PASS']}", flush=True)
    if not gate3["PASS"]:
        raise RuntimeError(f"Gate 3 failed: {gate3}")

    return {
        "C_T": C_T, "C_G": C_G, "SE_T": SE_T, "SE_G": SE_G,
        "C_T_full": C_T_full, "C_G_full": C_G_full,
        "P_pair": P_pair, "P_aa": P_aa, "P_aa_SE": P_aa_SE,
        "Z": Z, "D": D, "D8": D8, "pval": pval, "null": null,
        "eig": eig, "rho_truth": rho_truth, "rho_bind": rho_bind,
        "bin_edges": bin_edges_used, "stats": stats,
    }


# ---------------------------------------------------------------------------
# Sanity plot (mean lines per observable)

def sanity_plot(mu_curves: dict, log_M: np.ndarray, truth_F: np.ndarray,
                bind_F_mean: np.ndarray, obs_names: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(obs_names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, name in enumerate(obs_names):
        ax = axes[i // ncols, i % ncols]
        ax.scatter(log_M, truth_F[:, i], s=4, c="C0", alpha=0.3, label="truth")
        ax.scatter(log_M, bind_F_mean[:, i], s=4, c="C3", alpha=0.3, label="BIND ⟨k⟩")
        x_grid, mu_combined, mu_truth_only, mu_bind_only = mu_curves[name]
        ax.plot(x_grid, mu_combined, "k-", lw=1.5, label="LOWESS combined")
        ax.plot(x_grid, mu_truth_only, "C0--", lw=1.0, label="LOWESS truth-only")
        ax.plot(x_grid, mu_bind_only,  "C3:",  lw=1.0, label="LOWESS bind-only")
        ax.set_xlabel(r"$\log_{10}\,M_{200c}\;[M_\odot/h]$")
        ax.set_ylabel(name)
        if i == 0:
            ax.legend(fontsize=7)
        diff = float(np.nanmax(np.abs(mu_truth_only - mu_bind_only)))
        flag = " *" if diff > 0.05 else ""
        ax.set_title(f"{name} (max diff {diff:.3f}{flag})", fontsize=9)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("LOWESS sanity check — truth vs BIND mean lines (CV)", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=0, help="0=all; 1/2/3 = stop after that phase")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    start = time.time()
    table = phase1_build_table()
    if args.phase == 1:
        print(f"Done (phase 1 only); elapsed {time.time()-start:.1f}s")
        return
    residuals = phase2_residuals(table)
    if args.phase == 2:
        print(f"Done (phase 2 only); elapsed {time.time()-start:.1f}s")
        return
    phase3_statistics(residuals)
    print(f"Done (phases 1-3); elapsed {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
