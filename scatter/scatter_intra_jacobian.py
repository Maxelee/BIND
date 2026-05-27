"""scatter/scatter_intra_jacobian.py
Phase 1 diagnostic: Central FD Jacobian of log(σ_intra) w.r.t. all 35 params.

σ_intra,a(θ) = mean over halos of within-halo std across K noise seeds.

For each parameter j:
  J_log_sigma_intra[o, j] = (log σ_intra,a(θ+) - log σ_intra,a(θ-)) / (2ε)

Uses IDENTICAL K, ε, halo subset, and noise seed as the afternoon
J_log_sigma_inter run (scatter_jacobian.py), so the two Jacobians are
directly comparable.

Modes:
  compute  — run one shard (SLURM array element)
  merge    — merge shards → phase1_intra_jacobian.npz + contamination_ratio
  figures  — produce fig_intra_vs_inter_jacobian.pdf

Run as SLURM array:
    sbatch run_scatter_intra_jacobian.sh

Then merge + figure:
    sbatch --dependency=afterok:<JOBID> run_scatter_intra_merge.sh
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
assert len(PARAM_NAMES) == 35


# ---------------------------------------------------------------------------
# Compute mode

def run_compute(args):
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[intra_jac] device = {device}")

    norm_stats = NormStats.load(RUN_DIR / "norm_stats.npz")
    ckpt = RUN_DIR / "checkpoints" / "last.ckpt"
    print(f"[intra_jac] loading model from {ckpt}")
    lit = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm
    model_fm.model.eval()

    print(f"[intra_jac] loading CV halos from {CV_ROOT}")
    cv = load_cv_halos(CV_ROOT)
    cv["params"][:, 14] = 0.0  # CAMELS bug fix
    N_TOT = len(cv["masses"])

    # Subset halos — same seed and max_halos as afternoon run
    rng = np.random.default_rng(args.subset_seed)
    if args.max_halos is not None and args.max_halos < N_TOT:
        idx_use = np.sort(rng.choice(N_TOT, size=args.max_halos, replace=False))
    else:
        idx_use = np.arange(N_TOT)
    N_USE = len(idx_use)
    print(f"[intra_jac] N_USE = {N_USE}/{N_TOT}")

    cond_norm, ls_norm = normalize_inputs(cv, norm_stats)
    cond_4d  = cond_norm[:, np.newaxis]
    p_norm_fid = normalize_params_fid(cv["params"][0], norm_stats)

    omega_m  = cv["params"][:, 0].astype(np.float64)
    dmo_raw  = cv["cond_raw"]

    cond_use     = cond_4d[idx_use]
    ls_use       = ls_norm[idx_use]
    masses_use   = cv["masses"][idx_use]
    r200_pix_use = cv["radii_pix"][idx_use]
    omega_m_use  = omega_m[idx_use]
    dmo_raw_use  = dmo_raw[idx_use]

    # Which parameters this shard processes
    if args.params is not None:
        param_idxs = np.array([int(s) for s in args.params.split(",")], dtype=np.int64)
    else:
        edges = np.linspace(0, N_PARAMS, args.n_chunks + 1).astype(int)
        lo, hi = edges[args.chunk_id], edges[args.chunk_id + 1]
        param_idxs = np.arange(lo, hi)
    print(f"[intra_jac] processing params: {param_idxs.tolist()}")

    if len(param_idxs) == 0:
        print("[intra_jac] empty shard — nothing to do")
        return

    # Output arrays (N_obs × n_params_this_shard)
    J_log_sigma_intra    = np.full((N_OBS, len(param_idxs)), np.nan)
    J_log_sigma_intra_se = np.full((N_OBS, len(param_idxs)), np.nan)

    # Per-param intermediate directory for crash recovery
    INT_DIR = Path(args.int_dir)
    INT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    for jj, j in enumerate(param_idxs):
        pname = PARAM_NAMES[j] if j < len(PARAM_NAMES) else f"p{j}"
        print(f"\n[intra_jac] param {j} ({pname})  ({jj+1}/{len(param_idxs)})", flush=True)

        # Resume from intermediate if it exists
        int_path = INT_DIR / f"param_{j:02d}.npz"
        if int_path.exists():
            d_int = np.load(int_path)
            J_log_sigma_intra[:, jj]    = d_int["J_log_sigma_intra_col"]
            J_log_sigma_intra_se[:, jj] = d_int["J_log_sigma_intra_se_col"]
            print(f"  [RESUMED from {int_path}]", flush=True)
            continue

        p_plus  = p_norm_fid.copy(); p_plus[j]  += args.eps
        p_minus = p_norm_fid.copy(); p_minus[j] -= args.eps

        # Same noise seed for θ+ and θ- to correlate draws → smaller σ on difference
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
            # σ_intra(θ) = mean across halos of within-halo std across K seeds
            # Already aggregated in measure_scatter: r["sigma_intra"][o] is a scalar
            si_plus  = float(r_plus["sigma_intra"][o])
            si_minus = float(r_minus["sigma_intra"][o])
            if si_plus > 0 and si_minus > 0:
                J_log_sigma_intra[o, jj] = (np.log(si_plus) - np.log(si_minus)) / (2 * args.eps)
                # SE: propagation of uncertainty on log σ_intra
                # σ_intra = (1/N) Σ_h std_h, and std_h ~ chi dist → var(std_h) ≈ σ²_intra/(2(K-1))
                # SE(σ_intra) ≈ σ_intra / sqrt(2*(K-1)*N_h)
                # SE(log σ_intra) ≈ 1/sqrt(2*(K-1)*N_h)
                N_h = N_USE
                se_log_intra = 1.0 / np.sqrt(2 * (args.K - 1) * N_h)
                J_log_sigma_intra_se[o, jj] = se_log_intra / (2 * args.eps)

        elapsed = time.time() - t_start
        eta = elapsed / (jj + 1) * (len(param_idxs) - jj - 1)
        print(f"  done  ({elapsed/60:.1f} min elapsed; ETA {eta/60:.1f} min)", flush=True)

        # Quick diagnostic
        for o, name in enumerate(ALL_OBS_NAMES[:5]):
            print(f"    {name:20s}  J_log_sigma_intra={J_log_sigma_intra[o, jj]:+.4f}", flush=True)

        # Save intermediate for crash recovery
        np.savez_compressed(
            int_path,
            J_log_sigma_intra_col    = J_log_sigma_intra[:, jj],
            J_log_sigma_intra_se_col = J_log_sigma_intra_se[:, jj],
            param_idx                = np.int64(j),
        )

    # Save shard
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        J_log_sigma_intra    = J_log_sigma_intra,
        J_log_sigma_intra_se = J_log_sigma_intra_se,
        param_idxs           = param_idxs,
        obs_names            = np.array(ALL_OBS_NAMES),
        log_mask             = LOG_MASK,
        idx_use              = idx_use,
        masses_use           = masses_use,
        eps                  = np.float64(args.eps),
        K                    = np.int64(args.K),
        n_steps              = np.int64(args.n_steps),
    )
    print(f"\n[intra_jac] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Merge mode

def run_merge(args):
    files = sorted(glob(args.shard_glob))
    if not files:
        raise FileNotFoundError(f"No files matching: {args.shard_glob}")
    print(f"[merge] found {len(files)} shards")

    first = np.load(files[0], allow_pickle=True)
    obs_names  = list(first["obs_names"])
    log_mask   = first["log_mask"]
    idx_use    = first["idx_use"]
    masses_use = first["masses_use"]
    eps        = float(first["eps"])
    K          = int(first["K"])

    # Accept shards from the combined scatter_jacobian.py (preferred) or from
    # the standalone scatter_intra_jacobian.py compute mode (fallback).
    combined_mode = "J_log_sigma_intra" in first  # combined shard has both

    J_log_sigma_intra_full    = np.full((N_OBS, N_PARAMS), np.nan)
    J_log_sigma_intra_se_full = np.full((N_OBS, N_PARAMS), np.nan)
    seen = np.zeros(N_PARAMS, dtype=bool)

    for f in files:
        d = np.load(f, allow_pickle=True)
        for jj, j in enumerate(d["param_idxs"]):
            if seen[j]:
                print(f"  WARN: param {j} already seen — overwriting from {f}")
            seen[j] = True
            if combined_mode:
                if "J_log_sigma_intra" in d:
                    J_log_sigma_intra_full[:, j]    = d["J_log_sigma_intra"][:, jj]
                    J_log_sigma_intra_se_full[:, j] = d["J_log_sigma_intra_se"][:, jj]
            else:
                J_log_sigma_intra_full[:, j]    = d["J_log_sigma_intra"][:, jj]
                J_log_sigma_intra_se_full[:, j] = d["J_log_sigma_intra_se"][:, jj]

    missing = np.where(~seen)[0]
    if missing.size:
        print(f"[merge] WARNING: {len(missing)} params not covered: {missing.tolist()}")

    # Load afternoon's J_log_sigma_inter for comparison
    inter_path = Path("scatter/J_mean_and_scatter.npz")
    d_inter = np.load(inter_path, allow_pickle=True)
    J_log_sigma_inter    = d_inter["J_log_sigma"]       # (N_obs, N_params)
    J_log_sigma_inter_se = d_inter["J_log_sigma_se"]

    # Contamination ratio: |J_intra| / |J_inter|, with 1e-8 floor
    contamination_ratio = np.abs(J_log_sigma_intra_full) / (np.abs(J_log_sigma_inter) + 1e-8)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        J_log_sigma_intra    = J_log_sigma_intra_full,
        J_log_sigma_intra_se = J_log_sigma_intra_se_full,
        J_log_sigma_inter    = J_log_sigma_inter,
        J_log_sigma_inter_se = J_log_sigma_inter_se,
        contamination_ratio  = contamination_ratio,
        param_names          = np.array(PARAM_NAMES),
        obs_names            = np.array(obs_names),
        log_mask             = log_mask,
        idx_use              = idx_use,
        masses_use           = masses_use,
        eps                  = np.float64(eps),
        K                    = np.int64(K),
        params_seen          = seen,
    )
    print(f"[merge] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)  "
          f"covering {seen.sum()}/{N_PARAMS} params")

    # Gate 1 report — print headline contamination ratios
    _print_gate1_report(out)


def _print_gate1_report(npz_path):
    """Print Gate 1 contamination ratios for the 4 headline parameters on dq_DM."""
    d = np.load(npz_path, allow_pickle=True)
    obs_names = list(d["obs_names"])
    param_names = list(d["param_names"])
    R = d["contamination_ratio"]
    J_intra = d["J_log_sigma_intra"]
    J_inter = d["J_log_sigma_inter"]

    headline_params = {"Omega_m": 0, "A_SN1": 2, "A_SN2": 4, "Omega_b": 6}

    if "dq_DM" not in obs_names:
        print("[gate1] WARNING: dq_DM not in obs_names — cannot evaluate gate")
        return

    dq_idx = obs_names.index("dq_DM")
    print("\n=== Gate 1 — contamination_ratio R_j = |J_intra| / |J_inter| for dq_DM ===")
    print(f"{'Param':12s}  {'idx':3s}  {'J_inter':8s}  {'J_intra':8s}  {'R_j':6s}  {'Verdict'}")
    print("-" * 68)

    gate_pass = True
    for pname, pidx in headline_params.items():
        j_int = float(J_inter[dq_idx, pidx])
        j_intra = float(J_intra[dq_idx, pidx])
        r = float(R[dq_idx, pidx])
        if r >= 0.7:
            verdict = "ESCALATE (contaminated)"
            gate_pass = False
        elif r >= 0.3:
            verdict = "PARTIAL contamination"
        else:
            verdict = "PASS"
        print(f"{pname:12s}  {pidx:3d}  {j_int:+8.4f}  {j_intra:+8.4f}  {r:6.3f}  {verdict}")

    print()
    if gate_pass:
        print("Gate 1 overall: PASS — all headline R_j < 0.7")
    else:
        print("Gate 1 overall: ESCALATE — one or more headline R_j >= 0.7")

    # Write gate1 report json
    import json
    gate1_data = {
        "headline_contamination": {},
        "gate_verdict": "PASS" if gate_pass else "ESCALATE",
    }
    for pname, pidx in headline_params.items():
        gate1_data["headline_contamination"][pname] = {
            "J_inter": float(J_inter[dq_idx, pidx]),
            "J_intra": float(J_intra[dq_idx, pidx]),
            "R_j": float(R[dq_idx, pidx]),
        }
    gate1_path = Path("outputs/scatter_diagnostics/phase1_gate1_report.json")
    gate1_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gate1_path, "w") as f:
        json.dump(gate1_data, f, indent=2)
    print(f"[gate1] wrote {gate1_path}")


# ---------------------------------------------------------------------------
# Figure mode

def run_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_path = Path(args.input)
    d = np.load(npz_path, allow_pickle=True)
    J_intra  = d["J_log_sigma_intra"]   # (N_obs, N_params)
    J_inter  = d["J_log_sigma_inter"]
    obs_names = list(d["obs_names"])
    param_names = list(d["param_names"])

    # Parameter family colours
    cosmo_idxs  = [0, 1, 6, 7, 8]         # Omega_m, sigma8, Omega_b, H0, n_s
    sn_idxs     = [2, 4]                   # A_SN1, A_SN2
    agn_idxs    = [3, 5]                   # A_AGN1, A_AGN2
    other_idxs  = [i for i in range(35) if i not in cosmo_idxs + sn_idxs + agn_idxs]
    colors = {i: "royalblue" for i in cosmo_idxs}
    colors.update({i: "tomato" for i in sn_idxs})
    colors.update({i: "darkorange" for i in agn_idxs})
    colors.update({i: "gray" for i in other_idxs})

    # Annotation labels for headline params
    annotate = {0: "Ω_m", 2: "A_SN1", 4: "A_SN2", 6: "Ω_b"}

    panels = [n for n in ["M_gas", "M_star", "dq_DM"] if n in obs_names]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, oname in zip(axes, panels):
        oidx = obs_names.index(oname)
        xi = J_inter[oidx]   # (35,)
        yi = J_intra[oidx]

        for j in range(35):
            ax.scatter(xi[j], yi[j], color=colors[j], s=30, alpha=0.7, zorder=3)
            if j in annotate:
                ax.annotate(annotate[j], (xi[j], yi[j]),
                            fontsize=8, textcoords="offset points", xytext=(5, 2))

        # Diagonal y = x
        lim = max(np.nanmax(np.abs(xi)), np.nanmax(np.abs(yi)), 0.1) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5, label="y = x")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$J_{\log\sigma_{\rm inter}}$", fontsize=12)
        ax.set_ylabel(r"$J_{\log\sigma_{\rm intra}}$", fontsize=12)
        ax.set_title(oname, fontsize=13)
        ax.set_aspect("equal")

        # Legend
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue",  markersize=8, label="Cosmology"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",     markersize=8, label="A_SN"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange", markersize=8, label="A_AGN"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",       markersize=8, label="Other"),
            Line2D([0], [0], linestyle="--", color="k", lw=0.8, label="y = x"),
        ]
        ax.legend(handles=legend_elems, fontsize=7, loc="upper left")

    fig.suptitle(r"$J_{\log\sigma_{\rm intra}}$ vs $J_{\log\sigma_{\rm inter}}$"
                 "\n(contamination check: points near y=x → noise-floor drives scatter signal)",
                 fontsize=11)
    fig.tight_layout()

    out_dir = Path("figures/scatter_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "fig_intra_vs_inter_jacobian.pdf"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    out_png = out_dir / "fig_intra_vs_inter_jacobian.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[fig] wrote {out_fig} and {out_png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI

def main():
    ap = argparse.ArgumentParser(description="Phase 1: intra-σ Jacobian diagnostic")
    subparsers = ap.add_subparsers(dest="mode")

    # Compute
    c = subparsers.add_parser("compute")
    c.add_argument("--n_chunks",    type=int,   default=1)
    c.add_argument("--chunk_id",    type=int,   default=0)
    c.add_argument("--params",      type=str,   default=None,
                   help="Comma-separated param indices (overrides n_chunks/chunk_id)")
    c.add_argument("--output",      required=True)
    c.add_argument("--int_dir",     type=str,   default="scatter/intermediate_intra",
                   help="Directory for per-param crash-recovery intermediates")
    c.add_argument("--eps",         type=float, default=0.05)
    c.add_argument("--K",           type=int,   default=5)
    c.add_argument("--n_steps",     type=int,   default=10)
    c.add_argument("--batch_size",  type=int,   default=32)
    c.add_argument("--max_halos",   type=int,   default=None)
    c.add_argument("--noise_seed",  type=int,   default=42)
    c.add_argument("--subset_seed", type=int,   default=0)

    # Merge
    m = subparsers.add_parser("merge")
    m.add_argument("--shard_glob", required=True)
    m.add_argument("--output",     required=True)

    # Figures
    f = subparsers.add_parser("figures")
    f.add_argument("--input", required=True)

    args = ap.parse_args()
    if args.mode == "compute":
        run_compute(args)
    elif args.mode == "merge":
        run_merge(args)
    elif args.mode == "figures":
        run_figures(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
