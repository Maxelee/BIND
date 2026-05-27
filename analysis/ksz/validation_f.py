"""Validation plot F — velocity-field robustness.

We treat the v_los systematic as a multiplicative bias on the τ-stack
observable: x_obs = x_true × (1 + ε_v), ε_v ~ N(0, σ_v²).  σ_v=0 corresponds
to the "τ-only / v_los marginalised" path (§3.2.5 option b in
docs/paper2_ksz_plan.md), larger σ_v emulates an unmodelled multiplicative
v_los systematic.

For each σ_v in --vlos_sigmas we run the same leave-one-out coverage protocol
as validation_e — with the synthetic observation built from the **real
held-out stack** ``x[i]`` (not the emulator's own mean), so the systematic
genuinely perturbs the data the posterior sees.  If coverage is roughly flat
as a function of σ_v, the τ-only path is robust to v_los systematics; if it
degrades quickly, the paper has to flag the systematic.

Caveat: a flat curve is only meaningful for *data-informed* parameters.  When
the stacked observable does not constrain a parameter (posterior ≈ prior), its
coverage is insensitive to σ_v regardless — see validation_e's `constraint`.

Usage:
    python -m analysis.ksz.validation_f \\
        --testsuite_root /mnt/home/mlee1/ceph/fm_testsuite \\
        --model fm_two_head --suites Test \\
        --aperture cap --r_ap_mpc_h 0.5 \\
        --mass_bins 1e13 2e13 5e13 1e14 1e15 \\
        --vlos_sigmas 0 0.05 0.10 0.20 0.30 \\
        --out analysis_physics_cache/ksz_validation_f_fm_two_head.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ._io import find_sim_dirs, load_sim, los_advisory
from .inference import (
    GaussianPosterior,
    StackedEmulator,
    central_credible_contains,
    stack_per_sim,
)
from .tau_utils import per_halo_tau


def _gather_pairs(args, edges):
    thetas: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    banner_shown = False
    for suite in args.suites:
        sims = find_sim_dirs(args.testsuite_root, suite)
        for sd in sims:
            try:
                art = load_sim(
                    sd, suite=suite, model_name=args.model,
                    halo_mass_min=args.halo_mass_min,
                    box_size=args.box_size,
                    patch_size_mpc_h=args.patch_size_mpc_h,
                )
            except Exception as exc:
                print(f"[err]  {suite}/{sd.name}: {exc}")
                continue
            if art is None or art.params.shape[1] == 0:
                continue
            if not banner_shown:
                print(los_advisory(art.truth_source, art.los_depth_mpc_h, args.aperture))
                banner_shown = True
            theta = art.params[0]
            pix_size = args.patch_size_mpc_h / art.patch_pix
            r_ap_pix = args.r_ap_mpc_h / pix_size
            tau_b = per_halo_tau(art.bind_gas, r_ap_pix, pix_size, args.hubble,
                                 estimator=args.aperture)
            x, _ = stack_per_sim(tau_b, art.halo_masses, edges)
            thetas.append(theta)
            xs.append(x)
    if not thetas:
        raise SystemExit("No sims gathered for v_los robustness test.")
    return np.asarray(thetas, dtype=np.float64), np.asarray(xs, dtype=np.float64)


def _loo_coverage(theta, x, *, ridge, prior_std, noise_frac, vlos_sigma,
                  n_realizations, level, rng):
    """Return per-param (coverage, |bias|/σ) across LOO sims × realizations."""
    n, p = theta.shape
    hits = np.zeros(p, dtype=np.int64)
    abs_bias = np.zeros(p, dtype=np.float64)
    n_trials = n * n_realizations
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        emu = StackedEmulator.fit(theta[mask], x[mask], ridge=ridge)
        # Real held-out forward-model stack (not emu.predict) so the v_los
        # systematic perturbs genuine data and the posterior can actually drift.
        x_true = x[i]
        sigma_meas = noise_frac * np.abs(x_true)
        for _ in range(n_realizations):
            eps_v = rng.normal(scale=vlos_sigma) if vlos_sigma > 0 else 0.0
            x_obs = x_true * (1.0 + eps_v) + rng.normal(scale=sigma_meas)
            post = GaussianPosterior.from_observation(
                emu, x_obs, sigma_meas=sigma_meas, prior_std=prior_std,
            )
            hits += central_credible_contains(
                post.mean, post.std, theta[i], level=level
            ).astype(np.int64)
            abs_bias += np.abs(post.mean - theta[i]) / np.maximum(post.std, 1e-12)
    return hits / n_trials, abs_bias / n_trials


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--testsuite_root", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--suites", nargs="+", default=["Test"])
    p.add_argument("--halo_mass_min", type=float, default=1e13)
    p.add_argument("--box_size", type=float, default=50.0)
    p.add_argument("--patch_size_mpc_h", type=float, default=6.25)
    p.add_argument("--hubble", type=float, default=0.6711)
    p.add_argument("--aperture", choices=["disk", "cap"], default="cap")
    p.add_argument("--r_ap_mpc_h", type=float, default=0.5)
    p.add_argument("--mass_bins", nargs="+", type=float,
                   default=[1e13, 2e13, 5e13, 1e14, 1e15])
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--prior_std", type=float, default=3.0)
    p.add_argument("--noise_frac", type=float, default=0.05)
    p.add_argument("--vlos_sigmas", nargs="+", type=float,
                   default=[0.0, 0.05, 0.10, 0.20, 0.30],
                   help="Fractional v_los systematic levels (std of multiplicative bias).")
    p.add_argument("--n_realizations", type=int, default=8)
    p.add_argument("--min_bin_coverage", type=float, default=0.8)
    p.add_argument("--level", type=float, default=0.6827)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    edges = np.asarray(args.mass_bins, dtype=np.float64)
    centers = np.sqrt(edges[:-1] * edges[1:])

    theta, x = _gather_pairs(args, edges)
    bin_ok = (np.isfinite(x).mean(axis=0) >= args.min_bin_coverage)
    x = x[:, bin_ok]
    centers = centers[bin_ok]
    sim_ok = np.isfinite(x).all(axis=1)
    theta = theta[sim_ok]
    x = x[sim_ok]
    n, p_params = theta.shape
    print(f"[info] {n} sims × {x.shape[1]} mass bins after filtering")

    sigmas = np.asarray(args.vlos_sigmas, dtype=np.float64)
    cov = np.zeros((len(sigmas), p_params), dtype=np.float64)
    bias = np.zeros_like(cov)
    rng = np.random.default_rng(args.seed)
    n_trials = n * args.n_realizations
    for s, sv in enumerate(sigmas):
        cov[s], bias[s] = _loo_coverage(
            theta, x, ridge=args.ridge, prior_std=args.prior_std,
            noise_frac=args.noise_frac, vlos_sigma=float(sv),
            n_realizations=args.n_realizations, level=args.level, rng=rng,
        )
        print(f"  σ_v={sv:.2f}: mean coverage = {cov[s].mean():.3f}  "
              f"mean |bias|/σ = {bias[s].mean():.2f}")

    cov_err = np.sqrt(cov * (1.0 - cov) / n_trials)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        vlos_sigmas=sigmas,
        coverage=cov,
        coverage_err=cov_err,
        abs_bias_in_sigma=bias,
        nominal_level=args.level,
        n_sims=n,
        n_realizations=args.n_realizations,
        mass_centers=centers,
        meta=np.array(repr({
            "aperture": args.aperture,
            "r_ap_mpc_h": args.r_ap_mpc_h,
            "model": args.model,
            "suites": args.suites,
            "noise_frac": args.noise_frac,
        })),
    )
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
