"""scatter/calibration_cv.py
Phase 2: Posterior calibration of BIND scatter against CV ground truth.

For each of the 27 CV simulations:
  1. Extract per-halo truth cutouts from full_maps.npz and compute observables.
  2. Run measure_scatter at fiducial theta with K >= 10.
  3. Mass-bin results and compare BIND sigma to truth sigma.

Outputs
-------
  scatter/cv_calibration.csv       — table per obs × mass-bin
  scatter/fig1_calibration.pdf/png — bar chart
  scatter/cv_calibration_data.npz  — full raw arrays for inspection

Usage
-----
  python scatter/calibration_cv.py [--K 10] [--n_steps 20] [--batch_size 32]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import NormStats, log_transform
from fd_jacobian_cv import (
    load_cv_halos,
    normalize_inputs,
    normalize_params_fid,
    r200c_mpc_h,
    MPC_PER_PIX,
    BOX_SIZE,
    N_PIX_FULL,
    PATCH_PIX,
    OMEGA_B_FIXED,
    OBS_KEYS,
    observables_from_phys,
    axis_ratio_q,
    _RR_PIX,
)
from scatter.measure_scatter import (
    measure_scatter,
    ALL_OBS_NAMES,
    LOG_MASK,
    HEADLINE_OBS_NAMES,
    _compute_all_obs,
    _gas_annular_profile,
)
from test_suite.pipeline import extract_periodic_cutout
from train import FlowMatchingLit

RUN_DIR  = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
CV_ROOT  = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
OUT_DIR  = Path(__file__).resolve().parent

# Mass bins: [lo, hi) in M_sun/h
MASS_BINS = [
    (1e13,   10**13.5),
    (10**13.5, 1e14),
    (1e14,   10**14.8),
]
MASS_BIN_LABELS = [
    r"$10^{13}{-}10^{13.5}$",
    r"$10^{13.5}{-}10^{14}$",
    r"$10^{14}{-}10^{14.8}$",
]


# ---------------------------------------------------------------------------
# Truth observable extraction

def extract_truth_obs(
    cv_root: Path,
    cv: dict,
) -> np.ndarray:
    """Compute truth observables for all halos across all CV sims.

    Returns
    -------
    truth_obs : (N_tot, N_obs) float32
        Same ordering as ALL_OBS_NAMES.
    """
    N_tot = len(cv["masses"])
    N_obs = len(ALL_OBS_NAMES)
    truth_obs = np.full((N_tot, N_obs), np.nan, dtype=np.float32)

    # Build per-sim index map
    sim_ids = np.unique(cv["sim_id"])
    r200_pix = cv["radii_pix"]
    omega_m  = cv["params"][:, 0].astype(np.float64)
    f_b_arr  = OMEGA_B_FIXED / np.where(omega_m > 0, omega_m, np.nan)
    centers  = cv["centers"]          # (N_tot, 2) in Mpc/h

    # Precompute q_DMO from raw DMO condition
    q_dmo_arr = np.full(N_tot, np.nan, dtype=np.float64)
    dmo_raw = cv["cond_raw"].astype(np.float64)
    for i in range(N_tot):
        r_aper = max(min(float(r200_pix[i]), PATCH_PIX / 2 - 2), 4.0)
        q_dmo_arr[i] = axis_ratio_q(np.maximum(dmo_raw[i], 0.0), r_aper)

    global_idx = 0
    for sim_id in sorted(sim_ids):
        sim_dir = cv_root / sim_id
        full_maps_path = sim_dir / "snap_090" / "full_maps.npz"
        if not full_maps_path.exists():
            print(f"  WARNING: no full_maps.npz for {sim_id}, skipping truth")
            # Advance global_idx past this sim's halos
            n_sim = (cv["sim_id"] == sim_id).sum()
            global_idx += n_sim
            continue

        fm = np.load(full_maps_path)
        truth_maps = fm["truth_maps"].astype(np.float32)  # (3, 1024, 1024)

        mask = cv["sim_id"] == sim_id
        idxs = np.where(mask)[0]
        n_sim = len(idxs)
        print(f"  {sim_id}: {n_sim} halos", flush=True)

        for i in idxs:
            cx_mpc = float(centers[i, 0])
            cy_mpc = float(centers[i, 1])
            cx_pix = int(cx_mpc / BOX_SIZE * N_PIX_FULL) % N_PIX_FULL
            cy_pix = int(cy_mpc / BOX_SIZE * N_PIX_FULL) % N_PIX_FULL

            # Extract 128x128 patch from truth maps at each channel
            patches = np.stack([
                extract_periodic_cutout(truth_maps[ch], cx_pix, cy_pix, PATCH_PIX)
                for ch in range(3)
            ])  # (3, 128, 128)

            truth_obs[i] = _compute_all_obs(
                patches,
                float(r200_pix[i]),
                float(f_b_arr[i]),
                float(q_dmo_arr[i]),
            )

        global_idx += n_sim

    return truth_obs, q_dmo_arr


def compute_sigma_per_bin(
    obs: np.ndarray,    # (N, N_obs)
    masses: np.ndarray,  # (N,)
    mass_bins: list[tuple],
) -> dict:
    """Per-mass-bin sigma for each observable.

    Returns: {bin_label: (N_obs,) std array}
    """
    N_obs = obs.shape[1]
    result = {}
    for lo, hi in mass_bins:
        mask = (masses >= lo) & (masses < hi)
        if mask.sum() < 2:
            result[(lo, hi)] = np.full(N_obs, np.nan)
            continue
        Y = np.full((mask.sum(), N_obs), np.nan, dtype=np.float64)
        for o in range(N_obs):
            x = obs[mask, o].astype(np.float64)
            if LOG_MASK[o]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    Y[:, o] = np.where(x > 0, np.log10(x), np.nan)
            else:
                Y[:, o] = x
        result[(lo, hi)] = np.nanstd(Y, axis=0, ddof=1)
    return result


# ---------------------------------------------------------------------------
# Plotting

def make_figure(
    truth_sigma: dict,
    bind_sigma_inter: dict,
    bind_sigma_total: dict,
    out_path: Path,
    headline_only: bool = True,
):
    """Bar chart: BIND vs CV-true scatter per observable per mass bin."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return

    obs_names_plot = HEADLINE_OBS_NAMES if headline_only else ALL_OBS_NAMES
    n_obs = len(obs_names_plot)
    n_bins = len(MASS_BINS)

    fig, axes = plt.subplots(n_bins, 1, figsize=(max(12, n_obs * 0.8), 4 * n_bins), squeeze=False)

    for bi, (lo, hi) in enumerate(MASS_BINS):
        ax = axes[bi, 0]
        key = (lo, hi)
        x = np.arange(n_obs)
        width = 0.25

        y_true  = np.array([truth_sigma[key][ALL_OBS_NAMES.index(n)] for n in obs_names_plot])
        y_inter = np.array([bind_sigma_inter[key][ALL_OBS_NAMES.index(n)] for n in obs_names_plot])
        y_total = np.array([bind_sigma_total[key][ALL_OBS_NAMES.index(n)] for n in obs_names_plot])

        ax.bar(x - width, y_true,  width, label="CV truth", color="steelblue", alpha=0.8)
        ax.bar(x,         y_inter, width, label="BIND inter", color="firebrick", alpha=0.8)
        ax.bar(x + width, y_total, width, label="BIND total", color="darkorange", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(obs_names_plot, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(r"$\sigma$ (log$_{10}$ or linear)")
        ax.set_title(f"Mass bin {MASS_BIN_LABELS[bi]}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("BIND scatter vs CV ground-truth scatter", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main

def main(args):
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ckpt = RUN_DIR / "checkpoints" / "last.ckpt"
    norm_stats = NormStats.load(RUN_DIR / "norm_stats.npz")
    lit = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm
    model_fm.model.eval()

    print("Loading CV halos ...")
    cv = load_cv_halos(CV_ROOT)
    cv["params"][:, 14] = 0.0  # CAMELS bug fix

    # Add centers field from halo_catalog (needed for truth extraction)
    # Reload to get centers (load_cv_halos doesn't expose them); use the
    # same inclusion criterion as load_cv_halos (both cat and cutouts must exist).
    sim_dirs = sorted(d for d in CV_ROOT.iterdir() if d.is_dir())
    centers_list = []
    for d in sim_dirs:
        cat_path  = d / "snap_090" / "mass_threshold_1p000e13" / "halo_catalog.npz"
        cut_path  = d / "snap_090" / "mass_threshold_1p000e13" / "halo_cutouts.npz"
        if not (cat_path.exists() and cut_path.exists()):
            continue
        cat = np.load(cat_path)
        centers_list.append(cat["centers"])
    cv["centers"] = np.concatenate(centers_list).astype(np.float32)

    N_TOT = len(cv["masses"])
    print(f"Total halos: {N_TOT}")

    # --- Truth observables ---
    truth_npz = OUT_DIR / "cv_truth_obs.npz"
    if truth_npz.exists() and not args.recompute_truth:
        print(f"Loading cached truth observables from {truth_npz}")
        d = np.load(truth_npz)
        truth_obs = d["truth_obs"]
        q_dmo_arr = d["q_dmo_arr"]
    else:
        print("Computing truth observables ...")
        truth_obs, q_dmo_arr = extract_truth_obs(CV_ROOT, cv)
        np.savez_compressed(truth_npz, truth_obs=truth_obs, q_dmo_arr=q_dmo_arr)
        print(f"Saved truth obs to {truth_npz}")

    # --- BIND observables ---
    bind_npz = OUT_DIR / f"cv_bind_obs_K{args.K}.npz"
    if bind_npz.exists() and not args.recompute_bind:
        print(f"Loading cached BIND observables from {bind_npz}")
        d = np.load(bind_npz)
        bind_obs_tensor = d["obs_tensor"]
        bind_masses     = d["masses"]
    else:
        print(f"Running measure_scatter (K={args.K}, N_h={N_TOT}) ...")
        cond_norm, ls_norm = normalize_inputs(cv, norm_stats)
        cond_norm_4d = cond_norm[:, np.newaxis]
        p_norm_fid = normalize_params_fid(cv["params"][0], norm_stats)

        result = measure_scatter(
            model_fm   = model_fm,
            norm_stats = norm_stats,
            theta_norm = p_norm_fid,
            dmo_conds  = cond_norm_4d,
            ls_conds   = ls_norm,
            masses     = cv["masses"],
            r200_pix   = cv["radii_pix"],
            K          = args.K,
            n_steps    = args.n_steps,
            device     = str(device),
            batch_size = args.batch_size,
            dmo_raw    = cv["cond_raw"],
            omega_m    = cv["params"][:, 0].astype(np.float64),
            seed       = 42,
        )
        bind_obs_tensor = result["obs_tensor"]
        bind_masses     = result["masses"]
        np.savez_compressed(
            bind_npz,
            obs_tensor = bind_obs_tensor,
            masses     = bind_masses,
            sigma_inter = result["sigma_inter"],
            sigma_intra = result["sigma_intra"],
            sigma_total = result["sigma_total"],
            Y_bar       = result["Y_bar"],
        )
        print(f"Saved BIND obs to {bind_npz}")

    # --- Per-mass-bin sigma computation ---
    masses = cv["masses"]

    truth_sigma = compute_sigma_per_bin(truth_obs, masses, MASS_BINS)

    # BIND inter: sigma of per-halo means
    bind_sigma_inter = {}
    bind_sigma_total = {}
    N_obs = len(ALL_OBS_NAMES)
    for lo, hi in MASS_BINS:
        mask = (masses >= lo) & (masses < hi)
        if mask.sum() < 2:
            bind_sigma_inter[(lo, hi)] = np.full(N_obs, np.nan)
            bind_sigma_total[(lo, hi)] = np.full(N_obs, np.nan)
            continue

        obs_bin = bind_obs_tensor[mask]   # (N_bin, K, N_obs)
        K = obs_bin.shape[1]
        Y_bin = np.full_like(obs_bin, np.nan, dtype=np.float64)
        for o in range(N_obs):
            x = obs_bin[:, :, o].astype(np.float64)
            if LOG_MASK[o]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    Y_bin[:, :, o] = np.where(x > 0, np.log10(x), np.nan)
            else:
                Y_bin[:, :, o] = x
        Y_bar_bin = np.nanmean(Y_bin, axis=1)            # (N_bin, N_obs)
        bind_sigma_inter[(lo, hi)] = np.nanstd(Y_bar_bin, axis=0, ddof=1)
        Y_flat = Y_bin.reshape(-1, N_obs)
        bind_sigma_total[(lo, hi)] = np.nanstd(Y_flat, axis=0, ddof=1)

    # --- Print calibration table ---
    print("\n=== Calibration summary ===")
    print(f"{'Observable':<22}  {'Bin':<25}  {'sigma_truth':>11}  {'BIND_inter':>10}  {'BIND_total':>10}  {'ratio_total':>11}")
    n_pass = 0
    n_finite = 0
    for name in HEADLINE_OBS_NAMES:
        oi = ALL_OBS_NAMES.index(name)
        for (lo, hi), label in zip(MASS_BINS, MASS_BIN_LABELS):
            st = truth_sigma[(lo, hi)][oi]
            si = bind_sigma_inter[(lo, hi)][oi]
            so = bind_sigma_total[(lo, hi)][oi]
            if np.isfinite(st) and st > 0 and np.isfinite(so) and so > 0:
                ratio = abs(so - st) / st
                n_finite += 1
                if ratio < 0.3:
                    n_pass += 1
                flag = "" if ratio < 0.3 else "  <-- FAIL"
            else:
                ratio = np.nan
                flag = "  (nan)"
            print(f"  {name:<20}  {label:<25}  {st:11.4f}  {si:10.4f}  {so:10.4f}  {ratio:11.2%}{flag}")

    print(f"\nGating check: {n_pass}/{n_finite} obs×bin pairs within 30% (need >= 8 / 11 obs avg)")

    # Gating: check per-observable average
    per_obs_pass = []
    for name in HEADLINE_OBS_NAMES[:11]:  # first 11 (non-profile) observables
        oi = ALL_OBS_NAMES.index(name)
        ratios = []
        for (lo, hi) in MASS_BINS:
            st = truth_sigma[(lo, hi)][oi]
            so = bind_sigma_total[(lo, hi)][oi]
            if np.isfinite(st) and st > 0 and np.isfinite(so) and so > 0:
                ratios.append(abs(so - st) / st)
        if ratios:
            per_obs_pass.append(np.mean(ratios) < 0.3)

    n_obs_pass = sum(per_obs_pass)
    print(f"Observables passing (<30% avg error): {n_obs_pass} / {len(per_obs_pass)}")
    if n_obs_pass >= 8:
        print("GATING CHECK: PASSED ✓")
    else:
        print("GATING CHECK: FAILED — see PLAN_NOTES.md for next steps")

    # --- Save CSV ---
    import csv
    csv_path = OUT_DIR / "cv_calibration.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["observable", "mass_bin_lo", "mass_bin_hi", "n_halos",
                         "sigma_truth", "sigma_BIND_inter", "sigma_BIND_total", "rel_err"])
        for name in ALL_OBS_NAMES:
            oi = ALL_OBS_NAMES.index(name)
            for (lo, hi) in MASS_BINS:
                mask = (masses >= lo) & (masses < hi)
                st = truth_sigma[(lo, hi)][oi]
                si = bind_sigma_inter[(lo, hi)][oi]
                so = bind_sigma_total[(lo, hi)][oi]
                ratio = abs(so - st) / st if (np.isfinite(st) and st > 0 and np.isfinite(so)) else np.nan
                writer.writerow([name, f"{lo:.2e}", f"{hi:.2e}", int(mask.sum()),
                                 f"{st:.6f}", f"{si:.6f}", f"{so:.6f}", f"{ratio:.4f}"])
    print(f"Saved {csv_path}")

    # --- Save full data ---
    np.savez_compressed(
        OUT_DIR / "cv_calibration_data.npz",
        truth_obs   = truth_obs,
        masses      = masses,
    )

    # --- Figure ---
    fig_path = Path("/mnt/home/mlee1/vdm_bind2/paper_figures/scatter/fig1_calibration.pdf")
    make_figure(truth_sigma, bind_sigma_inter, bind_sigma_total, fig_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--recompute_truth", action="store_true")
    ap.add_argument("--recompute_bind", action="store_true")
    main(ap.parse_args())
