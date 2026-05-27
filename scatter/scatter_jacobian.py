"""scatter/scatter_jacobian.py
Phase 3: Central FD Jacobian of mean and scatter observables w.r.t. all 35 params.

For each parameter j, perturb theta_fid ± eps and run measure_scatter to get:
  J_mean[o, j]       = (grand_mean_Y^+ - grand_mean_Y^-) / (2*eps)
  J_log_sigma[o, j]  = (log(sigma_inter^+) - log(sigma_inter^-)) / (2*eps)

Uses the SAME random seed for theta+ and theta- calls to maximally correlate
noise draws and minimize variance in the sigma difference estimate.

Run as a 1-GPU job per chunk:
    python scatter/scatter_jacobian.py --n_chunks 7 --chunk_id 0 \
        --output scatter/intermediate/scatter_jac_shard0.npz

Or single-GPU (all 35 params):
    python scatter/scatter_jacobian.py --n_chunks 1 --chunk_id 0 \
        --output scatter/J_mean_and_scatter.npz

Merge shards:
    python scatter/scatter_jacobian.py --merge \
        --shard_glob 'scatter/intermediate/scatter_jac_shard*.npz' \
        --output scatter/J_mean_and_scatter.npz

Optional: restrict to specific params:
    python scatter/scatter_jacobian.py --params 0,1,5 \
        --output scatter/intermediate/scatter_jac_test.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import NormStats, log_transform
from fd_jacobian_cv import (
    load_cv_halos, normalize_inputs, normalize_params_fid,
    N_PARAMS,
)
from scatter.measure_scatter import (
    measure_scatter,
    ALL_OBS_NAMES,
    LOG_MASK,
)
from train import FlowMatchingLit

RUN_DIR = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
CV_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
N_OBS   = len(ALL_OBS_NAMES)

# Parameter names (matches CAMELS IllustrisTNG L50n512 35-param order)
PARAM_NAMES = [
    "Omega_m", "sigma8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2",
    "Omega_b", "H0", "n_s",
    "MaxSfr", "SoftEQS", "IMFslope", "SNII_MinMass",
    "ThermalWind", "WindSpecMom",
    "WindFreeTravelDens", "MinWindVel", "WindEnergyReduction",
    "WindEnergyReductionZ", "WindEnergyReductionExp", "WindDumpFac",
    "SeedBHMass", "BHAccretion", "BHEddington", "BHFeedback",
    "BHRadEff", "QuasarThreshold", "QuasarThreshPow",
    "UVB_H0_beta", "UVB_H0_Dz", "UVB_Hep_beta", "UVB_Hep_Dz",
    "SNIa_norm", "SNIa_DTD_pow",
    "SofteningComoving",
]
assert len(PARAM_NAMES) == 35, f"Expected 35 param names, got {len(PARAM_NAMES)}"


# ---------------------------------------------------------------------------
# Compute mode

def run_compute(args):
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[scatter_jac] device = {device}")

    norm_stats = NormStats.load(RUN_DIR / "norm_stats.npz")
    ckpt = RUN_DIR / "checkpoints" / "last.ckpt"
    print(f"[scatter_jac] loading model from {ckpt}")
    lit = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm
    model_fm.model.eval()

    print(f"[scatter_jac] loading CV halos from {CV_ROOT}")
    cv = load_cv_halos(CV_ROOT)
    cv["params"][:, 14] = 0.0  # CAMELS bug fix
    N_TOT = len(cv["masses"])

    # Subset halos
    rng = np.random.default_rng(args.subset_seed)
    if args.max_halos is not None and args.max_halos < N_TOT:
        idx_use = np.sort(rng.choice(N_TOT, size=args.max_halos, replace=False))
    else:
        idx_use = np.arange(N_TOT)
    N_USE = len(idx_use)
    print(f"[scatter_jac] N_USE = {N_USE}/{N_TOT}")

    # Normalized conditioning arrays
    cond_norm, ls_norm = normalize_inputs(cv, norm_stats)
    cond_4d  = cond_norm[:, np.newaxis]  # (N, 1, H, W)
    p_norm_fid = normalize_params_fid(cv["params"][0], norm_stats)

    # Per-halo ancillary
    omega_m = cv["params"][:, 0].astype(np.float64)
    dmo_raw = cv["cond_raw"]

    # Subset
    cond_use    = cond_4d[idx_use]
    ls_use      = ls_norm[idx_use]
    masses_use  = cv["masses"][idx_use]
    r200_pix_use = cv["radii_pix"][idx_use]
    omega_m_use = omega_m[idx_use]
    dmo_raw_use = dmo_raw[idx_use]

    # Which parameters to process in this shard
    if args.params is not None:
        param_idxs = np.array([int(s) for s in args.params.split(",")], dtype=np.int64)
    else:
        edges = np.linspace(0, N_PARAMS, args.n_chunks + 1).astype(int)
        lo, hi = edges[args.chunk_id], edges[args.chunk_id + 1]
        param_idxs = np.arange(lo, hi)
    print(f"[scatter_jac] processing params: {param_idxs.tolist()}")

    if len(param_idxs) == 0:
        print("[scatter_jac] empty shard — nothing to do")
        return

    # Output arrays: shape (N_obs, n_params_this_shard)
    J_mean      = np.full((N_OBS, len(param_idxs)), np.nan, dtype=np.float64)
    J_log_sigma = np.full((N_OBS, len(param_idxs)), np.nan, dtype=np.float64)
    # Standard errors estimated as half the interquartile range over halos
    J_mean_se      = np.full((N_OBS, len(param_idxs)), np.nan, dtype=np.float64)
    J_log_sigma_se = np.full((N_OBS, len(param_idxs)), np.nan, dtype=np.float64)

    t_start = time.time()

    for jj, j in enumerate(param_idxs):
        pname = PARAM_NAMES[j] if j < len(PARAM_NAMES) else f"p{j}"
        print(f"\n[scatter_jac] param {j} ({pname})  ({jj+1}/{len(param_idxs)})", flush=True)

        p_plus  = p_norm_fid.copy(); p_plus[j]  += args.eps
        p_minus = p_norm_fid.copy(); p_minus[j] -= args.eps

        # Same seed for + and - to correlate noise → smaller variance on difference
        r_plus = measure_scatter(
            model_fm   = model_fm,
            norm_stats = norm_stats,
            theta_norm = p_plus,
            dmo_conds  = cond_use,
            ls_conds   = ls_use,
            masses     = masses_use,
            r200_pix   = r200_pix_use,
            K          = args.K,
            n_steps    = args.n_steps,
            device     = str(device),
            batch_size = args.batch_size,
            dmo_raw    = dmo_raw_use,
            omega_m    = omega_m_use,
            seed       = args.noise_seed,
        )
        r_minus = measure_scatter(
            model_fm   = model_fm,
            norm_stats = norm_stats,
            theta_norm = p_minus,
            dmo_conds  = cond_use,
            ls_conds   = ls_use,
            masses     = masses_use,
            r200_pix   = r200_pix_use,
            K          = args.K,
            n_steps    = args.n_steps,
            device     = str(device),
            batch_size = args.batch_size,
            dmo_raw    = dmo_raw_use,
            omega_m    = omega_m_use,
            seed       = args.noise_seed,
        )

        for o in range(N_OBS):
            # Mean Jacobian: derivative of grand mean Y
            # Y_bar^+ - Y_bar^-: both are (N_h, N_obs), take mean over halos
            ybar_plus  = r_plus["Y_bar"][:, o]   # (N_h,)
            ybar_minus = r_minus["Y_bar"][:, o]  # (N_h,)
            diff_mean  = ybar_plus - ybar_minus   # (N_h,)
            finite_mask = np.isfinite(diff_mean)
            if finite_mask.sum() >= 2:
                J_mean[o, jj] = np.mean(diff_mean[finite_mask]) / (2 * args.eps)
                # SE: std of per-halo differences / sqrt(N) / (2*eps)
                J_mean_se[o, jj] = (np.std(diff_mean[finite_mask], ddof=1)
                                    / np.sqrt(finite_mask.sum()) / (2 * args.eps))

            # Sigma Jacobian: derivative of log(sigma_inter)
            si_plus  = r_plus["sigma_inter"][o]
            si_minus = r_minus["sigma_inter"][o]
            if si_plus > 0 and si_minus > 0:
                J_log_sigma[o, jj] = (np.log(si_plus) - np.log(si_minus)) / (2 * args.eps)
                # Bootstrap SE: jackknife over halos on sigma_inter
                # Use the two-sample estimate via propagation of uncertainty
                # SE(log σ) ≈ SE(σ) / σ; SE(σ) ≈ σ / sqrt(2*(N-1))
                n_h = max(finite_mask.sum(), 2)
                se_log_sigma = 1.0 / np.sqrt(2 * (n_h - 1))
                J_log_sigma_se[o, jj] = se_log_sigma / (2 * args.eps)

        elapsed = time.time() - t_start
        eta = elapsed / (jj + 1) * (len(param_idxs) - jj - 1)
        print(f"  done  ({elapsed/60:.1f} min elapsed; ETA {eta/60:.1f} min)", flush=True)

        # Quick diagnostic for this param
        for o, name in enumerate(ALL_OBS_NAMES[:5]):  # print first 5
            print(f"    {name:20s}  J_mean={J_mean[o, jj]:+.4f}  J_log_sigma={J_log_sigma[o, jj]:+.4f}", flush=True)

    # Save shard
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        J_mean         = J_mean,
        J_log_sigma    = J_log_sigma,
        J_mean_se      = J_mean_se,
        J_log_sigma_se = J_log_sigma_se,
        param_idxs     = param_idxs,
        obs_names      = np.array(ALL_OBS_NAMES),
        log_mask       = LOG_MASK,
        idx_use        = idx_use,
        masses_use     = masses_use,
        eps            = np.float64(args.eps),
        K              = np.int64(args.K),
        n_steps        = np.int64(args.n_steps),
    )
    print(f"\n[scatter_jac] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Merge mode

def run_merge(args):
    files = sorted(glob(args.shard_glob))
    if not files:
        raise FileNotFoundError(f"No files matching: {args.shard_glob}")
    print(f"[merge] found {len(files)} shards")

    first = np.load(files[0], allow_pickle=True)
    obs_names = list(first["obs_names"])
    log_mask  = first["log_mask"]
    idx_use   = first["idx_use"]
    masses_use = first["masses_use"]
    eps       = float(first["eps"])
    K         = int(first["K"])

    J_mean_full         = np.full((N_OBS, N_PARAMS), np.nan)
    J_log_sigma_full    = np.full((N_OBS, N_PARAMS), np.nan)
    J_mean_se_full      = np.full((N_OBS, N_PARAMS), np.nan)
    J_log_sigma_se_full = np.full((N_OBS, N_PARAMS), np.nan)
    seen = np.zeros(N_PARAMS, dtype=bool)

    for f in files:
        d = np.load(f, allow_pickle=True)
        for jj, j in enumerate(d["param_idxs"]):
            if seen[j]:
                print(f"  WARN: param {j} already seen — overwriting from {f}")
            seen[j] = True
            J_mean_full[:, j]         = d["J_mean"][:, jj]
            J_log_sigma_full[:, j]    = d["J_log_sigma"][:, jj]
            J_mean_se_full[:, j]      = d["J_mean_se"][:, jj]
            J_log_sigma_se_full[:, j] = d["J_log_sigma_se"][:, jj]

    missing = np.where(~seen)[0]
    if missing.size:
        print(f"[merge] WARNING: {len(missing)} params not covered: {missing.tolist()}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        J_mean         = J_mean_full,
        J_log_sigma    = J_log_sigma_full,
        J_mean_se      = J_mean_se_full,
        J_log_sigma_se = J_log_sigma_se_full,
        obs_names      = np.array(obs_names),
        log_mask       = log_mask,
        idx_use        = idx_use,
        masses_use     = masses_use,
        eps            = np.float64(eps),
        K              = np.int64(K),
        params_seen    = seen,
    )
    print(f"[merge] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)  "
          f"covering {seen.sum()}/{N_PARAMS} params")


# ---------------------------------------------------------------------------
# Gating check & summary

def print_gating_check(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    J_mean      = d["J_mean"]        # (N_obs, N_params)
    J_log_sigma = d["J_log_sigma"]
    J_mean_se   = d["J_mean_se"]
    J_log_sigma_se = d["J_log_sigma_se"]
    obs_names   = list(d["obs_names"])
    masses_use  = d["masses_use"]
    eps         = float(d["eps"])

    param_names = PARAM_NAMES if len(PARAM_NAMES) == N_PARAMS else [f"p{i}" for i in range(N_PARAMS)]

    print(f"\n=== Scatter Jacobian summary (eps={eps}) ===")

    for o_name in ["M_gas", "M_star", "dq_DM"]:
        if o_name not in obs_names:
            continue
        o = obs_names.index(o_name)
        jm = J_mean[o]
        js = J_log_sigma[o]
        jm_se = J_mean_se[o]
        js_se = J_log_sigma_se[o]

        top_mean = np.argsort(np.abs(jm))[::-1][:5]
        top_sig  = np.argsort(np.abs(js))[::-1][:5]

        print(f"\n  Observable: {o_name}")
        print(f"  Top-5 mean movers:    " +
              ", ".join(f"{param_names[j]}={jm[j]:+.3f}±{jm_se[j]:.3f}" for j in top_mean))
        print(f"  Top-5 scatter movers: " +
              ", ".join(f"{param_names[j]}={js[j]:+.3f}±{js_se[j]:.3f}" for j in top_sig))

    # Sanity: Omega_m should be in top-3 for M_gas mean
    if "M_gas" in obs_names:
        o = obs_names.index("M_gas")
        top3_mean = set(np.argsort(np.abs(J_mean[o]))[::-1][:3])
        om_idx = 0  # Omega_m is param 0 in CAMELS order
        if om_idx in top3_mean:
            print(f"\n  SANITY CHECK: Omega_m (p0) is in top-3 M_gas mean movers ✓")
        else:
            print(f"\n  SANITY CHECK: WARNING — Omega_m (p0) NOT in top-3 M_gas mean movers")
            top3_names = [param_names[j] for j in np.argsort(np.abs(J_mean[o]))[::-1][:3]]
            print(f"  Top-3 are: {top3_names}")

    # SE quality check: for AGN/SN amplitude params (indices 2-7 in CAMELS)
    agn_sn_idxs = list(range(2, 8))
    for o_name in ["M_gas", "M_star"]:
        if o_name not in obs_names:
            continue
        o = obs_names.index(o_name)
        for j in agn_sn_idxs:
            se  = J_log_sigma_se[o, j]
            sig = abs(J_log_sigma[o, j])
            if np.isfinite(se) and np.isfinite(sig) and sig > 0:
                snr = sig / se
                flag = "✓" if snr > 1 / 0.3 else "  <-- LOW SNR"
                print(f"  SE check  {o_name} p{j}: |J|={sig:.3f}  SE={se:.3f}  SNR={snr:.1f} {flag}")


# ---------------------------------------------------------------------------
# CLI

def main():
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(dest="mode")

    # Compute mode (default)
    c = subparsers.add_parser("compute", help="Compute scatter Jacobian shard")
    c.add_argument("--n_chunks", type=int, default=1)
    c.add_argument("--chunk_id", type=int, default=0)
    c.add_argument("--params", type=str, default=None,
                   help="Comma-separated param indices (overrides n_chunks/chunk_id)")
    c.add_argument("--output", required=True)
    c.add_argument("--eps", type=float, default=0.05)
    c.add_argument("--K", type=int, default=10)
    c.add_argument("--n_steps", type=int, default=20)
    c.add_argument("--batch_size", type=int, default=4)
    c.add_argument("--max_halos", type=int, default=None)
    c.add_argument("--noise_seed", type=int, default=42)
    c.add_argument("--subset_seed", type=int, default=0)

    # Merge mode
    m = subparsers.add_parser("merge", help="Merge scatter Jacobian shards")
    m.add_argument("--shard_glob", required=True)
    m.add_argument("--output", required=True)

    # Summary mode
    s = subparsers.add_parser("summary", help="Print gating check for merged npz")
    s.add_argument("--input", required=True)

    # Backward compat: if no subcommand, default to compute
    args, remaining = ap.parse_known_args()
    if args.mode is None:
        # Treat all args as compute args
        ap2 = argparse.ArgumentParser()
        ap2.add_argument("--n_chunks", type=int, default=1)
        ap2.add_argument("--chunk_id", type=int, default=0)
        ap2.add_argument("--params", type=str, default=None)
        ap2.add_argument("--output", required=True)
        ap2.add_argument("--eps", type=float, default=0.05)
        ap2.add_argument("--K", type=int, default=10)
        ap2.add_argument("--n_steps", type=int, default=20)
        ap2.add_argument("--batch_size", type=int, default=4)
        ap2.add_argument("--max_halos", type=int, default=None)
        ap2.add_argument("--noise_seed", type=int, default=42)
        ap2.add_argument("--subset_seed", type=int, default=0)
        ap2.add_argument("--merge", action="store_true")
        ap2.add_argument("--shard_glob", type=str, default=None)
        args = ap2.parse_args()
        if args.merge:
            run_merge(args)
        else:
            run_compute(args)
        return

    if args.mode == "compute":
        run_compute(args)
    elif args.mode == "merge":
        run_merge(args)
    elif args.mode == "summary":
        print_gating_check(args.input)


if __name__ == "__main__":
    main()
