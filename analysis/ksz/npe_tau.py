"""NPE scaffold for kSZ: P(θ_astro | stacked τ(M)) from the fixed-halo Sobol cube.

Consumes the reduced τ cube from `gen_sobol_taucube.py --reduce` and trains a
neural posterior estimator (sbi) for the 30 astrophysical parameters at fixed
cosmology.  Parameters live in normalized [0,1] (SB35 log-aware bounds), so the
prior is BoxUniform — un-normalize for reporting.

Diagnostics (no real data needed): a held-out split gives per-parameter
constraint (1 − σ_post/σ_prior), 68% coverage, and SBC rank uniformity — the
honest analogue of validation_e, now on a real (nonlinear) posterior.  Pass
``--x_obs`` (an npz with the ACT-derived stacked τ(M), same mass bins) to draw
the actual science posterior and save samples + a marginal figure.

This is a scaffold: it runs end-to-end on a small cube for wiring checks, but
the science run wants thousands of designs (see gen_sobol_taucube.py) and the
ACT observable in --x_obs.

Usage:
    python -m analysis.ksz.npe_tau \\
        --cube analysis_physics_cache/ksz_sobol_taucube_fm_cube_two_head.npz \\
        --observable rich --n_train_steps_max 0 \\
        --out_dir analysis_physics_cache/npe_fm_cube_two_head
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .param_meta import load_param_meta


def _raw_features(tau_med, scat_med, observable):
    """Assemble the (unstandardized) feature matrix + names from per-bin τ stats."""
    with np.errstate(invalid="ignore", divide="ignore"):
        log_tau = np.log10(np.clip(tau_med, 1e-30, None))
    blocks = [log_tau]
    names = [f"logTau_bin{b}" for b in range(log_tau.shape[-1])]
    if observable == "rich":
        blocks.append(np.atleast_2d(scat_med))
        names += [f"scatTau_bin{b}" for b in range(np.atleast_2d(scat_med).shape[-1])]
    return np.hstack(blocks) if log_tau.ndim == 2 else np.concatenate(blocks, axis=-1), names


def _build_xy(cube_path: Path, observable: str):
    """Return x (standardized), theta_norm, scan_idx, labels, feat_names, and the
    standardization transform (keep_col, mu, sd) needed to map x_obs identically."""
    z = np.load(cube_path, allow_pickle=True)
    tau = np.asarray(z["tau_stack"], dtype=np.float64)       # (D, K, nb)
    scat = np.asarray(z["tau_scat"], dtype=np.float64)       # (D, K, nb)
    theta35 = np.asarray(z["theta35"], dtype=np.float64)     # (D, 35)
    scan_idx = np.asarray(z["scan_idx"], dtype=np.int64)

    # median over the K stochastic draws → the BIND-mean observable
    tau_med = np.nanmedian(tau, axis=1)                      # (D, nb)
    scat_med = np.nanmedian(scat, axis=1)
    X, feat_names = _raw_features(tau_med, scat_med, observable)

    meta = load_param_meta()
    theta_norm = meta.normalized(theta35)[:, scan_idx]       # (D, 30) in [0,1]
    labels = [meta.labels[i] for i in scan_idx]

    # standardize: drop cols finite in <90% designs, drop rows with NaN, store moments
    keep_col = np.isfinite(X).mean(axis=0) >= 0.9
    X = X[:, keep_col]
    feat_names = [f for f, k in zip(feat_names, keep_col) if k]
    row_mask = np.isfinite(X).all(axis=1)
    Xc = X[row_mask]
    mu, sd = Xc.mean(0), Xc.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    transform = {"keep_col": keep_col, "mu": mu, "sd": sd}
    return (Xc - mu) / sd, theta_norm[row_mask], scan_idx, labels, feat_names, transform


def _npe_class():
    try:
        from sbi.inference import NPE      # sbi >= 0.23
        return NPE
    except ImportError:
        from sbi.inference import SNPE     # older
        return SNPE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", type=Path, required=True, help="Reduced τ cube npz.")
    ap.add_argument("--observable", choices=["narrow", "rich"], default="rich")
    ap.add_argument("--density_estimator", default="maf")
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--n_post_samples", type=int, default=2000)
    ap.add_argument("--x_obs", type=Path, default=None,
                    help="npz with key 'tau_obs' (nb,) [+ optional 'scat_obs'] = ACT-derived stack.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from sbi.utils import BoxUniform

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    X, theta, scan_idx, labels, feat_names, transform = _build_xy(args.cube, args.observable)
    d_theta = theta.shape[1]
    n = len(theta)
    print(f"[info] {n} designs · x-dim {X.shape[1]} ({args.observable}) · θ-dim {d_theta}")
    if n < 50:
        print("[warn] <50 designs — this is a wiring smoke, not a science posterior.")

    # train / held-out split
    idx = rng.permutation(n)
    n_hold = max(1, int(round(args.holdout_frac * n)))
    hold, train = idx[:n_hold], idx[n_hold:]

    prior = BoxUniform(low=torch.zeros(d_theta), high=torch.ones(d_theta))
    NPE = _npe_class()
    inference = NPE(prior=prior, density_estimator=args.density_estimator)
    theta_t = torch.tensor(theta[train], dtype=torch.float32)
    x_t = torch.tensor(X[train], dtype=torch.float32)
    inference.append_simulations(theta_t, x_t)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "posterior.pkl", "wb") as fh:
        pickle.dump({"posterior": posterior, "scan_idx": scan_idx,
                     "labels": labels, "feat_names": feat_names,
                     "observable": args.observable, "transform": transform}, fh)

    # ── held-out diagnostics: constraint, coverage, SBC ranks ───────────────
    prior_std = 1.0 / np.sqrt(12.0)             # std of Uniform(0,1)
    sbc_ranks = np.full((len(hold), d_theta), np.nan)
    post_std = np.full((len(hold), d_theta), np.nan)
    cover68 = np.zeros((len(hold), d_theta), dtype=bool)
    for r, i in enumerate(hold):
        xo = torch.tensor(X[i], dtype=torch.float32)
        s = posterior.sample((args.n_post_samples,), x=xo, show_progress_bars=False).numpy()
        th = theta[i]
        sbc_ranks[r] = (s < th[None, :]).mean(axis=0)
        post_std[r] = s.std(axis=0)
        lo, hi = np.percentile(s, [16, 84], axis=0)
        cover68[r] = (th >= lo) & (th <= hi)

    constraint = np.clip(1.0 - np.nanmean(post_std, axis=0) / prior_std, 0.0, 1.0)
    coverage = cover68.mean(axis=0)
    from scipy.stats import kstest
    sbc_ks_p = np.array([
        kstest(sbc_ranks[np.isfinite(sbc_ranks[:, j]), j], "uniform").pvalue
        if np.isfinite(sbc_ranks[:, j]).sum() >= 5 else np.nan
        for j in range(d_theta)
    ])

    order = np.argsort(-constraint)
    print(f"\n# NPE held-out diagnostics ({len(hold)} held-out designs)")
    print(f"# {int((constraint>0.1).sum())}/{d_theta} astro params constrained (>0.1)")
    for j in order[:10]:
        print(f"  p{scan_idx[j]:02d} {labels[j]:<22s} constraint={constraint[j]:.2f} "
              f"cover68={coverage[j]:.2f} SBC-KS_p={sbc_ks_p[j]:.2f}")

    np.savez(args.out_dir / "diagnostics.npz",
             scan_idx=scan_idx, labels=np.array(labels),
             constraint=constraint, coverage68=coverage,
             sbc_ranks=sbc_ranks, sbc_ks_p=sbc_ks_p, post_std=post_std)

    # ── optional: real-data posterior ───────────────────────────────────────
    if args.x_obs is not None:
        zo = np.load(args.x_obs, allow_pickle=True)
        tau_obs = np.asarray(zo["tau_obs"], dtype=np.float64)[None, :]     # (1, nb)
        scat_obs = (np.asarray(zo["scat_obs"], dtype=np.float64)[None, :]
                    if "scat_obs" in zo.files else np.zeros_like(tau_obs))
        xo_raw, _ = _raw_features(tau_obs, scat_obs, args.observable)       # (1, F_full)
        # apply the SAME standardization as training (keep_col → (−mu)/sd)
        xo = (xo_raw[:, transform["keep_col"]] - transform["mu"]) / transform["sd"]
        meta = load_param_meta()
        s = posterior.sample((args.n_post_samples * 5,),
                             x=torch.tensor(xo[0], dtype=torch.float32),
                             show_progress_bars=False).numpy()
        # un-normalize samples to raw param units (log-aware) for reporting
        raw = np.empty_like(s)
        lo, hi = meta.minv[scan_idx], meta.maxv[scan_idx]
        logm = meta.logflag[scan_idx].astype(bool)
        llo, lhi = np.log10(np.clip(lo, 1e-30, None)), np.log10(np.clip(hi, 1e-30, None))
        raw[:, ~logm] = lo[~logm] + s[:, ~logm] * (hi[~logm] - lo[~logm])
        raw[:, logm] = 10.0 ** (llo[logm] + s[:, logm] * (lhi[logm] - llo[logm]))
        np.savez(args.out_dir / "posterior_samples_xobs.npz",
                 samples_norm=s, samples_raw=raw, scan_idx=scan_idx, labels=np.array(labels))
        print(f"[save] {args.out_dir/'posterior_samples_xobs.npz'}")

    print(f"[save] {args.out_dir}/  (posterior.pkl, diagnostics.npz)")


if __name__ == "__main__":
    main()
