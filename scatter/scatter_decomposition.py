"""scatter/scatter_decomposition.py — Program 1: assembly vs. physics vs. intrinsic.

Decompose the variance of group-scale observables into three causally clean
components, using the generative model's defining capability: it conditions on
the *specific* DMO field of each halo, so we can hold a halo fixed and vary only
the subgrid physics (a paired counterfactual no simulation or observation can do).

For observable O and sample O[t,h,k]  (physics theta_t, halo h, noise draw k):

    Var(O) = sigma2_assembly  (between halos, fixed physics)
           + sigma2_physics   (between theta, averaged over halos)
           + sigma2_intrinsic (within a fixed (halo, theta): model stochasticity)
           + interaction

We estimate these with an *unbiased* balanced two-way random-effects ANOVA
(factor A = physics theta, factor B = halo, K replicate noise draws). The naive
np.var of group means is biased upward by the within-cell noise; the EMS
formulas below remove that bias.

Mechanics reuse scatter.measure_scatter (which already does the within-halo /
between-halo split for a single theta). This driver orchestrates it across a
theta-grid with COMMON RANDOM NUMBERS (identical seed every theta), which makes
the between-theta comparison paired and slashes Monte-Carlo noise in
sigma2_physics.

VALIDATION GATE (do before trusting any number): sigma2_intrinsic is the model's
*generative* noise. It must (a) reproduce the true simulation scatter on held-out
sims and (b) not collapse to ~0 (mode collapse) nor blow up (under-resolved
2D-projected conditioning). The smoke test prints it per observable so you can
eyeball both failure modes; cross-check against the truth-side scatter numbers
in the scatter project before interpreting.

Smoke test (default): SN-strength line + AGN-strength line through the fiducial,
fixed halo population from the CAMELS fiducial sim, K=15 draws. If feedback
measurably moves the *residual scatter* (not just the mean) at fixed halo, the
physics component is real and you scale to the full prior ensemble.

Usage:
    python -m scatter.scatter_decomposition                 # smoke test
    python -m scatter.scatter_decomposition --n-halos 50 --k 20 --levels 7
"""
from __future__ import annotations

import argparse
import glob
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
from scatter.measure_scatter import measure_scatter, ALL_OBS_NAMES, LOG_MASK

# ─────────────────────────────────────────────────────────────────────────────
# Paths / conventions (mirror scatter/fig3_cosmo_DMO_decomposition.py)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).parent.parent
RUN_DIR   = pathlib.Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
TEST_BASE = pathlib.Path("/mnt/home/mlee1/ceph/fm_testsuite/1P")
FIG_DIR   = BASE_DIR / "figures/scatter_diagnostics"
OUT_DIR   = BASE_DIR / "outputs/scatter_diagnostics"
SUB_DIR   = "snap_090/mass_threshold_1p000e13"
MPC_PER_PIX = 0.048828125

# Canonical 35-param order (scatter/scatter_jacobian.py); indices we sweep.
PARAM_NAMES = [
    "Omega_m", "sigma8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2", "Omega_b", "H0",
    "n_s", "MaxSfr", "SoftEQS", "IMFslope", "SNII_MinMass", "ThermalWind",
    "WindSpecMom", "WindFreeTravelDens", "MinWindVel", "WindEnergyReduction",
    "WindEnergyReductionZ", "WindEnergyReductionExp", "WindDumpFac", "SeedBHMass",
    "BHAccretion", "BHEddington", "BHFeedback", "BHRadEff", "QuasarThreshold",
    "QuasarThreshPow", "UVB_H0_beta", "UVB_H0_Dz", "UVB_Hep_beta", "UVB_Hep_Dz",
    "SNIa_norm", "SNIa_DTD_pow", "SofteningComoving",
]
IDX_A_SN1, IDX_A_AGN1, IDX_A_SN2, IDX_A_AGN2 = 2, 3, 4, 5

# Physics axes: each maps a sweep level (normalized [0,1]) onto a set of params.
# SN strength co-varies (A_SN1, A_SN2); AGN strength co-varies (A_AGN1, A_AGN2).
PHYSICS_AXES = {
    "SN":  [IDX_A_SN1, IDX_A_SN2],
    "AGN": [IDX_A_AGN1, IDX_A_AGN2],
}

# Parameter partition: 35 = 30 astro + 5 cosmology.
# Cosmology is excluded from fixed-DMO scans because varying it with a fixed DMO field is an
# out-of-distribution counterfactual (the DMO structure should change) — see --include-cosmo.
COSMO_IDX = [0, 1, 6, 7, 8]                 # Omega_m, sigma8, Omega_b, H0, n_s
ASTRO_IDX = [j for j in range(35) if j not in COSMO_IDX]   # 30 astro knobs (incl. p14)
# Note: p14 (WindSpecMom) was fixed at 0 in CV/1P (project memory); if SB35 training also fixed
# it, its Sobol index will come out ~0 here — an empirical check rather than a prior exclusion.

# Observables to feature in the headline figure (skipped if absent from the run).
FOCUS_OBS = ["M_gas", "M_star", "f_b", "q_gas"]

NOISE_SEED = 42  # IDENTICAL across all theta → common random numbers (paired).


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (fixed halo population, one sim)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_params_fid(p_raw: np.ndarray, ns: NormStats) -> np.ndarray:
    """Raw 35-param vector → normalized [0,1] vector (log-flagged params logged first)."""
    _p = np.where(ns.param_log_flag == 1,
                  np.log10(np.maximum(p_raw.astype(float), 1e-30)), p_raw.astype(float))
    return ((_p - ns.param_min) / (ns.param_max - ns.param_min + 1e-8)).astype(np.float32)


CV_BASE = pathlib.Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")   # 27 fiducial CV sims (sim_0..26)


def _load_dir(base_sim: pathlib.Path, ns: NormStats) -> dict | None:
    """Load + normalize ALL halo cutouts under one sim directory. None if absent."""
    base = base_sim / SUB_DIR
    cat_path, cut_path = base / "halo_catalog.npz", base / "halo_cutouts.npz"
    if not cat_path.exists() or not cut_path.exists():
        return None
    cat = np.load(cat_path, allow_pickle=True); cut = np.load(cut_path, allow_pickle=True)
    params = cat["params"].copy().astype(np.float32)
    params[:, 14] = 0.0  # CAMELS bug: p14=0 for CV/1P runs (see project memory).
    cond_raw = cut["condition"].astype(np.float32)
    ls_raw   = cut["large_scale"].astype(np.float32)
    cond_norm = (log_transform(cond_raw) - ns.cond_mean) / (ns.cond_std + 1e-8)
    ls_norm   = (log_transform(ls_raw) - ns.ls_mean[:, None, None]) / (ns.ls_std[:, None, None] + 1e-8)
    masses = cat["halo_masses"].astype(np.float64)
    if "radii" in cat.files:
        radii_pix = cat["radii"] / 1000.0 / MPC_PER_PIX   # kpc/h → Mpc/h → pixels
    else:
        from scatter.obs_common import r200c_pix          # fallback (e.g. CV sim_17 lacks 'radii')
        radii_pix = r200c_pix(masses)
    return {
        "cond_raw": cond_raw, "cond_norm": cond_norm[:, np.newaxis], "ls_norm": ls_norm,
        "params": params, "masses": masses,
        "radii_pix": radii_pix, "omega_m": params[:, 0].astype(np.float64),
        "N": len(masses),
    }


def load_halos(sim_name: str, ns: NormStats, n_halos: int, seed: int) -> dict | None:
    """Fixed halo set from ONE sim (optionally subsampled to n_halos)."""
    d = _load_dir(TEST_BASE / sim_name, ns)
    if d is None:
        print(f"  MISSING: {TEST_BASE / sim_name / SUB_DIR}"); return None
    N = d["N"]
    sel = (np.sort(np.random.default_rng(seed).choice(N, size=n_halos, replace=False))
           if n_halos < N else np.arange(N))
    keys = ["cond_raw", "cond_norm", "ls_norm", "params", "masses", "radii_pix", "omega_m"]
    out = {k: d[k][sel] for k in keys}
    out["sim"] = sim_name; out["N"] = len(sel)
    return out


def load_cv_halos(ns: NormStats, n_sims: int = 27, cap: int | None = None) -> dict | None:
    """Pool ALL halo cutouts across the 27 fiducial CV sims (sim_0..26) -> ~1100 halos."""
    parts = [_load_dir(CV_BASE / f"sim_{i}", ns) for i in range(n_sims)]
    parts = [p for p in parts if p]
    if not parts:
        print(f"  MISSING CV cutouts under {CV_BASE}"); return None
    keys = ["cond_raw", "cond_norm", "ls_norm", "params", "masses", "radii_pix", "omega_m"]
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    if cap is not None and cap < len(merged["masses"]):
        merged = {k: v[:cap] for k, v in merged.items()}
    merged["sim"] = "CV"; merged["N"] = len(merged["masses"])
    print(f"[decomp] pooled {len(parts)} CV sims -> {merged['N']} halos")
    return merged


def slice_data(data: dict, start: int, count: int) -> dict:
    """Halo slice [start:start+count] for chunked (GPU-parallel) generation."""
    sl = slice(start, start + count)
    keys = ["cond_raw", "cond_norm", "ls_norm", "params", "masses", "radii_pix", "omega_m"]
    out = {k: data[k][sl] for k in keys}
    out["sim"] = data["sim"]; out["N"] = len(out["masses"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Theta grid: one physics axis = a set of theta levels (factor A of the ANOVA)
# ─────────────────────────────────────────────────────────────────────────────
def build_axis_grid(theta_fid: np.ndarray, axis_param_idxs: list[int],
                    levels: np.ndarray) -> np.ndarray:
    """(n_levels, 35) array: fiducial theta with the axis params set to each level."""
    grid = np.tile(theta_fid, (len(levels), 1)).astype(np.float32)
    for li, lv in enumerate(levels):
        for j in axis_param_idxs:
            grid[li, j] = lv
    return grid


def build_joint_design(theta_fid: np.ndarray, idxs: list[int], n_design: int,
                       lo: float, hi: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scrambled-Sobol design over the chosen params (others fixed at fiducial).

    Returns (grid (M,35) of full theta vectors, sub (M,len(idxs)) the design coords in [0,1]).
    The same M points are applied to every halo, so (design × halo × K) stays balanced for the
    ANOVA, while the joint sampling lets us attribute the physics variance to individual params.
    """
    from scipy.stats import qmc
    m = int(2 ** np.ceil(np.log2(max(n_design, 2))))  # Sobol prefers powers of two
    sub = qmc.Sobol(d=len(idxs), scramble=True, seed=seed).random(m)
    sub = lo + (hi - lo) * sub                          # scale to [lo, hi]
    grid = np.tile(theta_fid, (m, 1)).astype(np.float32)
    grid[:, idxs] = sub.astype(np.float32)
    return grid, sub


def sobol_first_order(sub: np.ndarray, Y: np.ndarray, n_bins: int = 6) -> np.ndarray:
    """First-order Sobol indices S_i = Var(E[Y|X_i]) / Var(Y) via the binning estimator.

    'Given-data' Sobol on a space-filling design: bin the design along each parameter, take the
    (count-weighted) variance of the per-bin means of Y. Captures each parameter's marginal
    contribution; 1 - sum_i S_i is the interaction remainder. Noisy for many params at small M —
    report with the bootstrap CIs and treat as directional until M is large.
    """
    Y = np.asarray(Y, float)
    ok = np.isfinite(Y)
    Y, sub = Y[ok], sub[ok]
    vY = np.var(Y)
    D = sub.shape[1]
    S = np.zeros(D)
    if vY <= 0 or len(Y) < n_bins * 2:
        return S
    for i in range(D):
        edges = np.quantile(sub[:, i], np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        b = np.clip(np.digitize(sub[:, i], edges[1:-1]), 0, n_bins - 1)
        means, weights = [], []
        for k in range(n_bins):
            m = b == k
            if m.sum() > 0:
                means.append(Y[m].mean()); weights.append(m.sum())
        means, weights = np.array(means), np.array(weights, float)
        cond_mean = np.average(means, weights=weights)
        S[i] = np.average((means - cond_mean) ** 2, weights=weights) / vY
    return S


def run_cube(model_fm, ns, theta_grid, data, device_str, K, n_steps, batch_size) -> np.ndarray:
    """Generate the (n_theta, N_h, K, N_obs) observable cube with common random numbers."""
    n_theta = len(theta_grid)
    cube = None
    for ti, theta in enumerate(theta_grid):
        t0 = time.time()
        r = measure_scatter(
            model_fm=model_fm, norm_stats=ns, theta_norm=theta.astype(np.float32),
            dmo_conds=data["cond_norm"], ls_conds=data["ls_norm"],
            masses=data["masses"], r200_pix=data["radii_pix"],
            K=K, n_steps=n_steps, device=device_str, batch_size=batch_size,
            dmo_raw=data["cond_raw"], omega_m=data["omega_m"], seed=NOISE_SEED,
        )
        ot = r["obs_tensor"]  # (N_h, K, N_obs)
        if cube is None:
            cube = np.full((n_theta,) + ot.shape, np.nan, dtype=np.float64)
        cube[ti] = ot
        print(f"    theta {ti+1}/{n_theta} done ({time.time()-t0:.1f}s)")
    return cube  # (n_theta, N_h, K, N_obs)


# ─────────────────────────────────────────────────────────────────────────────
# Variance decomposition (balanced two-way random-effects ANOVA, unbiased EMS)
# ─────────────────────────────────────────────────────────────────────────────
def two_way_random_effects(X: np.ndarray) -> dict:
    """Unbiased variance components for a balanced (a x b x n) design.

    X[a, b, n] : factor A = physics theta (a levels), B = halo (b levels),
                 n = K replicate noise draws. Must be fully finite & balanced.

    Returns physics (A), assembly (B), interaction (AB), intrinsic (residual)
    variance components, clamped at 0 (standard for negative MoM estimates).
    """
    a, b, n = X.shape
    grand     = X.mean()
    A_means   = X.mean(axis=(1, 2))           # (a,)
    B_means   = X.mean(axis=(0, 2))           # (b,)
    cell_mean = X.mean(axis=2)                # (a, b)

    MS_A  = b * n * np.sum((A_means - grand) ** 2) / max(a - 1, 1)
    MS_B  = a * n * np.sum((B_means - grand) ** 2) / max(b - 1, 1)
    MS_AB = n * np.sum((cell_mean - A_means[:, None] - B_means[None, :] + grand) ** 2) \
        / max((a - 1) * (b - 1), 1)
    MS_E  = np.sum((X - cell_mean[:, :, None]) ** 2) / max(a * b * (n - 1), 1)

    s2_E  = MS_E
    s2_AB = max((MS_AB - MS_E) / n, 0.0)
    s2_A  = max((MS_A - MS_AB) / (b * n), 0.0)   # physics
    s2_B  = max((MS_B - MS_AB) / (a * n), 0.0)   # assembly

    total = s2_A + s2_B + s2_AB + s2_E
    return {
        "physics": float(s2_A), "assembly": float(s2_B),
        "interaction": float(s2_AB), "intrinsic": float(s2_E),
        "total": float(total),
        "frac": {k: (float(v / total) if total > 0 else np.nan)
                 for k, v in [("physics", s2_A), ("assembly", s2_B),
                              ("interaction", s2_AB), ("intrinsic", s2_E)]},
    }


def bootstrap_fracs(Xg: np.ndarray, n_boot: int, seed: int) -> dict:
    """95% CIs on each variance fraction by resampling halos (factor B) with replacement.

    Resampling the levels of the random halo factor propagates halo-population
    sampling uncertainty into the variance components — the dominant uncertainty
    when N_halos is modest. Returns {component: [lo, hi]} (2.5/97.5 percentiles).
    """
    a, b, n = Xg.shape
    rng = np.random.default_rng(seed)
    keys = ["assembly", "physics", "interaction", "intrinsic"]
    samples = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        bi = rng.integers(0, b, size=b)               # resample halos w/ replacement
        fr = two_way_random_effects(Xg[:, bi, :])["frac"]
        for k in keys:
            samples[k][i] = fr[k]
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in samples.items()}


def detrend_mass(cube_obs: np.ndarray, log_mass: np.ndarray) -> np.ndarray:
    """Remove the deterministic O–logM trend so the decomposition reflects *scatter*.

    Without this, 'assembly' is dominated by the halo mass range, not by assembly
    bias. We fit O_h = c0 + c1*logM_h on the per-halo grand mean (averaged over
    theta and noise) and subtract the fit from every (theta, noise) realization,
    leaving the residual structure: assembly = scatter-about-relation from halo
    identity, physics = how feedback shifts that scatter, intrinsic = model noise.

    cube_obs : (a, b, n) for one observable.  log_mass : (b,).
    """
    per_halo_mean = np.nanmean(cube_obs, axis=(0, 2))  # (b,)
    ok = np.isfinite(per_halo_mean) & np.isfinite(log_mass)
    if ok.sum() >= 3:
        c1, c0 = np.polyfit(log_mass[ok], per_halo_mean[ok], 1)
        fit = c0 + c1 * log_mass  # (b,)
    else:
        fit = np.zeros_like(log_mass)
    return cube_obs - fit[None, :, None]


def decompose_axis(cube: np.ndarray, obs_names: list[str], log_mask: np.ndarray,
                   log_mass: np.ndarray, focus: list[str], detrend: bool,
                   n_boot: int = 0, boot_seed: int = 7) -> dict:
    """Run the variance decomposition for every focus observable on one physics axis."""
    out = {}
    for oname in focus:
        if oname not in obs_names:
            continue
        oi = obs_names.index(oname)
        X = cube[:, :, :, oi].copy()                      # (a, b, n)
        if log_mask[oi]:                                  # match measure_scatter's log space
            X = np.log10(np.clip(X, 1e-30, None))
        # Drop halos with any non-finite entry to keep the design balanced.
        good_h = np.all(np.isfinite(X), axis=(0, 2))      # (b,)
        if good_h.sum() < 3:
            print(f"    [skip {oname}] <3 clean halos")
            continue
        Xg = X[:, good_h, :]
        lm = log_mass[good_h]
        if detrend:
            Xg = detrend_mass(Xg, lm)
        out[oname] = two_way_random_effects(Xg)
        out[oname]["n_halos_used"] = int(good_h.sum())
        if n_boot > 0:
            out[oname]["ci"] = bootstrap_fracs(Xg, n_boot, boot_seed + oi)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def print_table(axis_name: str, dec: dict) -> None:
    print(f"\n=== Physics axis: {axis_name} — variance fractions (detrended) ===")
    print(f"  {'obs':10s}  {'assembly':>9s}  {'physics':>9s}  {'interac':>9s}  "
          f"{'intrinsic':>10s}  {'total_var':>10s}  {'Nh':>3s}")
    for oname, d in dec.items():
        f = d["frac"]
        print(f"  {oname:10s}  {f['assembly']:9.3f}  {f['physics']:9.3f}  "
              f"{f['interaction']:9.3f}  {f['intrinsic']:10.3f}  "
              f"{d['total']:10.4g}  {d['n_halos_used']:3d}")


def plot_stacked(results: dict, focus: list[str], detrend: bool, out_stub: pathlib.Path) -> None:
    """One stacked-bar panel per physics axis; bars = focus observables."""
    axes_names = list(results.keys())
    comps  = ["assembly", "physics", "interaction", "intrinsic"]
    colors = {"assembly": "#1565C0", "physics": "#E65100",
              "interaction": "#9E9E9E", "intrinsic": "#6A1B9A"}
    fig, axs = plt.subplots(1, len(axes_names), figsize=(5.5 * len(axes_names), 5),
                            squeeze=False)
    for ax, axis_name in zip(axs[0], axes_names):
        dec = results[axis_name]
        obs = [o for o in focus if o in dec]
        x = np.arange(len(obs))
        bottom = np.zeros(len(obs))
        for c in comps:
            vals = np.array([dec[o]["frac"][c] for o in obs])
            ax.bar(x, vals, bottom=bottom, color=colors[c], label=c, edgecolor="white")
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(obs, rotation=20, ha="right")
        ax.set_ylim(0, 1); ax.set_ylabel("variance fraction")
        ax.set_title(f"Physics axis: {axis_name}")
    axs[0][0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("Scatter decomposition: assembly vs. physics vs. intrinsic"
                 + ("  (mass-detrended)" if detrend else "  (raw)"), fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_stub.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[decomp] saved {p}")


def plot_sensitivity(param_names: list[str], focus: list[str], phys: np.ndarray,
                     out_stub: pathlib.Path) -> None:
    """Heatmap: physics-driven scatter fraction for each (parameter, observable)."""
    fig, ax = plt.subplots(figsize=(1.4 + 0.55 * len(focus), 0.32 * len(param_names) + 1.5))
    im = ax.imshow(phys, aspect="auto", cmap="magma", vmin=0.0,
                   vmax=float(np.nanmax(phys)) if np.isfinite(phys).any() else 1.0)
    ax.set_xticks(np.arange(len(focus))); ax.set_xticklabels(focus, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(param_names))); ax.set_yticklabels(param_names, fontsize=7)
    ax.set_title("Physics-driven scatter fraction\nby parameter (rows) × observable (cols)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="physics fraction")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_stub.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[decomp] saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Assembly vs physics vs intrinsic scatter decomposition.")
    ap.add_argument("--mode", choices=["axes", "per-param", "joint"], default="axes",
                    help="'axes': SN/AGN lines. 'per-param': each astro knob (1P-style, marginal). "
                         "'joint': Sobol design over all astro knobs jointly (interactions).")
    ap.add_argument("--n-design", type=int, default=128,
                    help="joint mode: number of Sobol design points (rounded up to a power of 2).")
    ap.add_argument("--include-cosmo", action="store_true",
                    help="joint/per-param: also vary cosmology (UNPHYSICAL with fixed DMO — see docs).")
    ap.add_argument("--sim", default="1P_p1_0", help="Fixed-halo source sim (default: CAMELS fiducial).")
    ap.add_argument("--n-halos", type=int, default=30)
    ap.add_argument("--k", type=int, default=15, help="Noise draws per (halo, theta).")
    ap.add_argument("--levels", type=int, default=5, help="Theta levels per physics axis.")
    ap.add_argument("--lo", type=float, default=0.15, help="Min normalized level.")
    ap.add_argument("--hi", type=float, default=0.85, help="Max normalized level.")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=200,
                    help="Bootstrap resamples over halos for CIs (0 = off).")
    ap.add_argument("--no-detrend", action="store_true", help="Skip mass-detrending.")
    ap.add_argument("--axes", nargs="+", default=["SN", "AGN"], choices=list(PHYSICS_AXES))
    # base halo population + GPU-parallel map/reduce
    ap.add_argument("--base", choices=["sim", "cv"], default="sim",
                    help="'sim': one sim (--sim). 'cv': pool all 27 fiducial CV sims (~1100 halos).")
    ap.add_argument("--cv-cap", type=int, default=None, help="cap CV halo count (debug).")
    ap.add_argument("--phase", choices=["full", "generate", "reduce"], default="full",
                    help="'generate': one halo chunk -> partial cube (GPU task). 'reduce': combine.")
    ap.add_argument("--halo-start", type=int, default=0, help="generate: first halo index of chunk.")
    ap.add_argument("--halo-count", type=int, default=10**9, help="generate: halos in this chunk.")
    ap.add_argument("--chunk-dir", default=None, help="dir for partial cubes (default per base/mode).")
    args = ap.parse_args()
    detrend = not args.no_detrend
    args.label = "CV" if args.base == "cv" else args.sim
    args.tag = "_cv" if args.base == "cv" else ""
    if args.chunk_dir is None:
        args.chunk_dir = f"outputs/scatter_diagnostics/chunks_{args.mode}{args.tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_idx = sorted(ASTRO_IDX + (COSMO_IDX if args.include_cosmo else []))
    if args.include_cosmo:
        print("[decomp] WARNING: including cosmology with a FIXED DMO field — out-of-distribution.")
    levels = np.linspace(args.lo, args.hi, args.levels)

    # ── reduce: no model / halo loading needed (CPU) ────────────────────────────
    if args.phase == "reduce":
        print(f"[decomp] reduce  mode={args.mode}  base={args.label}")
        phase_reduce(args, scan_idx, detrend); return

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[decomp] mode={args.mode} phase={args.phase} base={args.label} device={device_str} "
          f"K={args.k} n_boot={args.n_boot} detrend={detrend}")

    # ── Model ───────────────────────────────────────────────────────────────────
    ns = NormStats.load(RUN_DIR / "norm_stats.npz")
    lit = FlowMatchingLit.load_from_checkpoint(str(RUN_DIR / "checkpoints/last.ckpt"),
                                               map_location=device_str)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm; model_fm.model.eval()

    # ── Halo population (one sim or pooled CV) ──────────────────────────────────
    data = load_cv_halos(ns, cap=args.cv_cap) if args.base == "cv" \
        else load_halos(args.sim, ns, args.n_halos, seed=NOISE_SEED)
    if data is None:
        print("[decomp] STOP: halo data not found."); return
    theta_fid = normalize_params_fid(data["params"][0], ns)   # CV halos are all fiducial
    if args.phase == "generate":     # slice to this chunk before generating
        data = slice_data(data, args.halo_start, args.halo_count)
        if data["N"] == 0:
            print(f"[decomp] empty chunk (start={args.halo_start}) — nothing to do"); return
    log_mass = np.log10(data["masses"])
    print(f"[decomp] halos: N={data['N']}  logM=[{log_mass.min():.2f}, {log_mass.max():.2f}]")

    if args.phase == "generate":
        phase_generate(args, model_fm, ns, data, theta_fid, levels, scan_idx, device_str); return

    if args.mode == "axes":
        run_axes_mode(args, model_fm, ns, data, theta_fid, log_mass, levels, detrend, device_str)
    elif args.mode == "per-param":
        run_per_param_mode(args, model_fm, ns, data, theta_fid, log_mass, levels, detrend,
                           device_str, scan_idx)
    else:
        run_joint_mode(args, model_fm, ns, data, theta_fid, log_mass, detrend, device_str, scan_idx)

    print("\n[decomp] DONE.  VALIDATION GATE: confirm 'intrinsic' fraction is neither ~0 "
          "(mode collapse) nor implausibly dominant before interpreting.")


def _gen_axes_cubes(args, model_fm, ns, data, theta_fid, levels, device_str) -> dict:
    """Generate the per-axis observable cubes (the GPU work) for a halo set."""
    cubes = {}
    for axis_name in args.axes:
        print(f"\n[decomp] === physics axis: {axis_name} (params "
              f"{[PARAM_NAMES[j] for j in PHYSICS_AXES[axis_name]]}) ===")
        grid = build_axis_grid(theta_fid, PHYSICS_AXES[axis_name], levels)
        cubes[axis_name] = run_cube(model_fm, ns, grid, data, device_str,
                                    args.k, args.n_steps, args.batch_size)
    return cubes


def _analyze_axes(args, cubes, levels, masses, log_mass, detrend) -> None:
    """Decompose + save + plot from per-axis cubes (shared by full and reduce paths)."""
    tag = args.tag; results = {}
    for axis_name, cube in cubes.items():
        dec = decompose_axis(cube, ALL_OBS_NAMES, LOG_MASK, log_mass, FOCUS_OBS,
                             detrend, n_boot=args.n_boot)
        results[axis_name] = dec; print_table(axis_name, dec)
    out_json = OUT_DIR / f"scatter_decomposition{tag}.json"
    out_json.write_text(json.dumps({
        "config": {"mode": "axes", "base": args.label, "n_halos": int(len(masses)), "K": args.k,
                   "levels": levels.tolist(), "n_steps": args.n_steps, "n_boot": args.n_boot,
                   "detrend": detrend, "seed": NOISE_SEED, "axes": list(cubes)},
        "results": results,
    }, indent=2))
    print(f"\n[decomp] wrote {out_json}")
    out_npz = OUT_DIR / f"scatter_decomposition_cube{tag}.npz"
    np.savez_compressed(
        out_npz, obs_names=np.array(ALL_OBS_NAMES), log_mask=LOG_MASK, levels=levels,
        masses=masses, log_mass=log_mass, focus_obs=np.array(FOCUS_OBS),
        axes=np.array(list(cubes)), detrend=detrend, sim=args.label,
        **{f"cube_{ax}": cubes[ax] for ax in cubes},
        **{f"axis_params_{ax}": np.array(PHYSICS_AXES[ax]) for ax in cubes},
    )
    print(f"[decomp] wrote {out_npz}  ({ {ax: cubes[ax].shape for ax in cubes} })")
    plot_stacked(results, FOCUS_OBS, detrend, FIG_DIR / f"scatter_decomposition{tag}")


def run_axes_mode(args, model_fm, ns, data, theta_fid, log_mass, levels, detrend, device_str) -> None:
    cubes = _gen_axes_cubes(args, model_fm, ns, data, theta_fid, levels, device_str)
    _analyze_axes(args, cubes, levels, data["masses"], log_mass, detrend)


def run_per_param_mode(args, model_fm, ns, data, theta_fid, log_mass, levels, detrend, device_str,
                       scan_idx) -> None:
    """Inclusive scan: sweep every astro parameter individually → sensitivity matrix.

    For each of the ~29 astro knobs we vary only that parameter (others fixed at
    fiducial) and decompose. The physics component is then that parameter's
    marginal contribution to each observable's scatter; assembly and intrinsic are
    ~parameter-independent and reported as the per-param mean. The headline product
    is a (parameter × observable) matrix of physics fractions — which physical knob
    drives the scatter of which observable, across the whole astro sector.
    """
    pnames = [PARAM_NAMES[j] for j in scan_idx]
    nP, nO = len(scan_idx), len(FOCUS_OBS)
    phys      = np.full((nP, nO), np.nan)
    phys_lo   = np.full((nP, nO), np.nan)
    phys_hi   = np.full((nP, nO), np.nan)
    assembly  = np.full((nP, nO), np.nan)
    intrinsic = np.full((nP, nO), np.nan)
    full = {}

    for pi, j in enumerate(scan_idx):
        print(f"\n[decomp] === per-param {pi+1}/{nP}: {PARAM_NAMES[j]} (idx {j}) ===")
        grid = build_axis_grid(theta_fid, [j], levels)
        cube = run_cube(model_fm, ns, grid, data, device_str,
                        args.k, args.n_steps, args.batch_size)
        dec = decompose_axis(cube, ALL_OBS_NAMES, LOG_MASK, log_mass, FOCUS_OBS,
                             detrend, n_boot=args.n_boot)
        full[PARAM_NAMES[j]] = dec
        for oi, o in enumerate(FOCUS_OBS):
            d = dec.get(o)
            if d is None:
                continue
            phys[pi, oi]      = d["frac"]["physics"]
            assembly[pi, oi]  = d["frac"]["assembly"]
            intrinsic[pi, oi] = d["frac"]["intrinsic"]
            if "ci" in d:
                phys_lo[pi, oi], phys_hi[pi, oi] = d["ci"]["physics"]
        # progress: top observable for this param
        row = phys[pi]
        if np.isfinite(row).any():
            bo = FOCUS_OBS[int(np.nanargmax(row))]
            print(f"    strongest on {bo} (physics frac={np.nanmax(row):.3f})")

    out_json = OUT_DIR / "scatter_decomposition_perparam.json"
    out_json.write_text(json.dumps({
        "config": {"mode": "per-param", "sim": args.sim, "n_halos": data["N"], "K": args.k,
                   "levels": levels.tolist(), "n_steps": args.n_steps, "n_boot": args.n_boot,
                   "detrend": detrend, "seed": NOISE_SEED,
                   "scan_idx": scan_idx, "scan_params": pnames},
        "results": full,
    }, indent=2))
    print(f"\n[decomp] wrote {out_json}")

    out_npz = OUT_DIR / "scatter_decomposition_perparam.npz"
    np.savez_compressed(
        out_npz, param_names=np.array(pnames), scan_idx=np.array(scan_idx),
        focus_obs=np.array(FOCUS_OBS), levels=levels, detrend=detrend, sim=args.sim,
        phys_frac=phys, phys_ci_lo=phys_lo, phys_ci_hi=phys_hi,
        assembly_frac=assembly, intrinsic_frac=intrinsic,
    )
    print(f"[decomp] wrote {out_npz}  (sensitivity matrix {phys.shape})")
    plot_sensitivity(pnames, FOCUS_OBS, phys, FIG_DIR / "scatter_decomposition_sensitivity")

    # Ranked summary: top drivers per observable.
    print("\n=== Top physics drivers per observable ===")
    for oi, o in enumerate(FOCUS_OBS):
        col = phys[:, oi]
        if not np.isfinite(col).any():
            continue
        order = np.argsort(-np.nan_to_num(col, nan=-1))[:5]
        ranked = ", ".join(f"{pnames[k]}({col[k]:.2f})" for k in order if np.isfinite(col[k]))
        print(f"  {o:10s}: {ranked}")


def run_joint_mode(args, model_fm, ns, data, theta_fid, log_mass, detrend, device_str,
                   scan_idx) -> None:
    """Joint Sobol scan over all scanned params at once → interactions + Sobol indices.

    All halos share the same M-point joint design (factor A = design point), so the
    ANOVA stays balanced: 'physics' is now the TOTAL variance from the full joint prior
    (not a 1-D line). First-order Sobol indices then attribute that joint physics
    variance to individual parameters; 1 - sum_i S_i is the interaction remainder that
    one-at-a-time scans (per-param / 1P) fundamentally cannot see.

    NOTE: cosmology is excluded by default (scan_idx). Varying cosmology here with a
    fixed DMO field is an out-of-distribution counterfactual — see --include-cosmo docs.
    """
    grid, sub = build_joint_design(theta_fid, scan_idx, args.n_design, args.lo, args.hi, NOISE_SEED)
    print(f"[decomp] joint Sobol design: {grid.shape[0]} points over {len(scan_idx)} params")
    cube = run_cube(model_fm, ns, grid, data, device_str, args.k, args.n_steps, args.batch_size)
    _analyze_joint(args, cube, sub, data["masses"], log_mass, detrend, scan_idx)


def _analyze_joint(args, cube, sub, masses, log_mass, detrend, scan_idx) -> None:
    """Decompose + Sobol + save + plot from the joint cube (shared by full and reduce paths)."""
    tag = args.tag; pnames = [PARAM_NAMES[j] for j in scan_idx]
    dec = decompose_axis(cube, ALL_OBS_NAMES, LOG_MASK, log_mass, FOCUS_OBS,
                         detrend, n_boot=args.n_boot)
    print_table("JOINT", dec)
    sobol = np.full((len(scan_idx), len(FOCUS_OBS)), np.nan)
    for oi, o in enumerate(FOCUS_OBS):
        if o not in ALL_OBS_NAMES:
            continue
        k = ALL_OBS_NAMES.index(o)
        X = cube[:, :, :, k].astype(float)
        if LOG_MASK[k]:
            X = np.log10(np.clip(X, 1e-30, None))
        if detrend:
            for m in range(X.shape[0]):
                X[m] = detrend_mass(X[m][None], log_mass)[0]
        Y = np.nanmean(X, axis=(1, 2))
        sobol[:, oi] = sobol_first_order(sub, Y)

    out_json = OUT_DIR / f"scatter_decomposition_joint{tag}.json"
    out_json.write_text(json.dumps({
        "config": {"mode": "joint", "base": args.label, "n_halos": int(len(masses)), "K": args.k,
                   "n_design": int(sub.shape[0]), "n_steps": args.n_steps, "n_boot": args.n_boot,
                   "detrend": detrend, "seed": NOISE_SEED,
                   "scan_idx": scan_idx, "scan_params": pnames, "lo": args.lo, "hi": args.hi},
        "decomposition": dec,
        "sobol_first_order": {o: sobol[:, oi].tolist() for oi, o in enumerate(FOCUS_OBS)},
    }, indent=2))
    print(f"\n[decomp] wrote {out_json}")
    np.savez_compressed(
        OUT_DIR / f"scatter_decomposition_joint{tag}.npz", param_names=np.array(pnames),
        scan_idx=np.array(scan_idx), focus_obs=np.array(FOCUS_OBS), design=sub,
        sobol_first_order=sobol, detrend=detrend, sim=args.label,
    )
    plot_sensitivity(pnames, FOCUS_OBS, sobol, FIG_DIR / f"scatter_decomposition_joint_sobol{tag}")
    print("\n=== Top first-order Sobol drivers per observable (joint prior) ===")
    for oi, o in enumerate(FOCUS_OBS):
        col = sobol[:, oi]
        if not np.isfinite(col).any():
            continue
        order = np.argsort(-np.nan_to_num(col, nan=-1))[:5]
        ranked = ", ".join(f"{pnames[k]}({col[k]:.2f})" for k in order if np.isfinite(col[k]))
        print(f"  {o:10s}: {ranked}   [interaction~{1.0 - np.nansum(np.clip(col, 0, None)):.2f}]")


# ─────────────────────────────────────────────────────────────────────────────
# GPU-parallel map/reduce: each task generates a halo chunk, then reduce combines
# ─────────────────────────────────────────────────────────────────────────────
def phase_generate(args, model_fm, ns, data, theta_fid, levels, scan_idx, device_str) -> None:
    cdir = pathlib.Path(args.chunk_dir); cdir.mkdir(parents=True, exist_ok=True)
    s = args.halo_start
    if args.mode == "axes":
        cubes = _gen_axes_cubes(args, model_fm, ns, data, theta_fid, levels, device_str)
        np.savez_compressed(cdir / f"axes_part_{s:06d}.npz", start=s, masses=data["masses"],
                            levels=levels, axes=np.array(args.axes),
                            **{f"cube_{ax}": cubes[ax] for ax in args.axes})
    else:  # joint
        grid, sub = build_joint_design(theta_fid, scan_idx, args.n_design, args.lo, args.hi, NOISE_SEED)
        cube = run_cube(model_fm, ns, grid, data, device_str, args.k, args.n_steps, args.batch_size)
        np.savez_compressed(cdir / f"joint_part_{s:06d}.npz", start=s, masses=data["masses"],
                            sub=sub, scan_idx=np.array(scan_idx), cube=cube)
    print(f"[decomp] wrote partial (start={s}, N={data['N']}) to {cdir}")


def phase_reduce(args, scan_idx, detrend) -> None:
    cdir = pathlib.Path(args.chunk_dir)
    pre = "axes" if args.mode == "axes" else "joint"
    parts = sorted(glob.glob(f"{cdir}/{pre}_part_*.npz"),
                   key=lambda p: int(np.load(p)["start"]))   # lazy: reads only 'start'
    if not parts:
        print(f"[decomp] no partial cubes in {cdir} — run --phase generate first."); return
    print(f"[decomp] reducing {len(parts)} chunks from {cdir}")
    loaded = [dict(np.load(p)) for p in parts]
    masses = np.concatenate([d["masses"] for d in loaded]); log_mass = np.log10(masses)
    if args.mode == "axes":
        axes = [str(x) for x in loaded[0]["axes"]]
        cubes = {ax: np.concatenate([d[f"cube_{ax}"] for d in loaded], axis=1) for ax in axes}
        _analyze_axes(args, cubes, loaded[0]["levels"], masses, log_mass, detrend)
    else:
        cube = np.concatenate([d["cube"] for d in loaded], axis=1)
        _analyze_joint(args, cube, loaded[0]["sub"], masses, log_mass, detrend, scan_idx)


if __name__ == "__main__":
    main()
