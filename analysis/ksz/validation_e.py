"""Validation plot E — SBI coverage on synthetic data.

For each sim in the chosen training pool (default: Test = SB35 sub-sample),
we (1) stack BIND τ(M) into the same mass bins used by validation_d, (2)
treat (θ_sim, x_sim) as a labelled pair, and (3) run a leave-one-out coverage
test of the analytic Gaussian posterior provided by
``analysis.ksz.inference``.

The credibility-level coverage of each of the 35 CAMELS parameters should be
close to the nominal level (default 0.6827) if the inference is well
calibrated.  Parameters whose coverage is far from nominal are either
under-constrained by the stacked observable or biased by emulator
mis-specification — both are honest diagnostics for the §4.E check in
docs/paper2_ksz_plan.md.

Usage:
    python -m analysis.ksz.validation_e \\
        --testsuite_root /mnt/home/mlee1/ceph/fm_testsuite \\
        --model fm_two_head --suites Test \\
        --aperture cap --r_ap_mpc_h 0.5 \\
        --mass_bins 1e13 2e13 5e13 1e14 1e15 \\
        --noise_frac 0.05 --n_realizations 8 --level 0.6827 \\
        --out analysis_physics_cache/ksz_validation_e_fm_two_head.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ._io import find_sim_dirs, load_sim
from .inference import (
    GaussianPosterior,
    StackedEmulator,
    central_credible_contains,
    stack_per_sim,
)
from .tau_utils import (
    aperture_cap_signal,
    aperture_gas_mass,
    gas_mass_to_tau_in_aperture,
    gas_surface_density_to_tau,
)


# --- mirror the per-halo τ definitions from validation_d --------------------


def _per_halo_tau_disk(patches, r_ap_pix, pix_size_mpc_h, hubble):
    mass = aperture_gas_mass(patches, r_ap_pix)
    area = np.pi * (r_ap_pix * pix_size_mpc_h) ** 2
    return gas_mass_to_tau_in_aperture(mass, area, hubble=hubble)


def _per_halo_tau_cap(patches, r_in_pix, pix_size_mpc_h, hubble):
    sig = aperture_cap_signal(patches, r_in_pix)
    area = np.pi * (r_in_pix * pix_size_mpc_h) ** 2
    sigma_msun_h_per_mpc2_h = sig / area
    return gas_surface_density_to_tau(sigma_msun_h_per_mpc2_h, hubble=hubble)


# --- gather per-sim (θ, x) pairs --------------------------------------------


def _gather_pairs(args, edges):
    thetas: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    sim_ids: list[str] = []
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
            theta = art.params[0]                       # all halos share θ
            pix_size = args.patch_size_mpc_h / art.patch_pix
            r_ap_pix = args.r_ap_mpc_h / pix_size
            if args.aperture == "cap":
                tau_b = _per_halo_tau_cap(art.bind_gas, r_ap_pix, pix_size, args.hubble)
            else:
                tau_b = _per_halo_tau_disk(art.bind_gas, r_ap_pix, pix_size, args.hubble)
            x, _ = stack_per_sim(tau_b, art.halo_masses, edges)
            thetas.append(theta)
            xs.append(x)
            sim_ids.append(f"{suite}/{sd.name}")
    if not thetas:
        raise SystemExit("No sims gathered for SBI coverage test.")
    return np.asarray(thetas, dtype=np.float64), np.asarray(xs, dtype=np.float64), sim_ids


# --- main ------------------------------------------------------------------


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
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="Ridge regularisation for the per-bin emulator.")
    p.add_argument("--prior_std", type=float, default=3.0,
                   help="Std of the Gaussian prior on standardised θ.")
    p.add_argument("--noise_frac", type=float, default=0.05,
                   help="Fractional measurement noise on x_obs (σ = noise_frac × |x_true|).")
    p.add_argument("--n_realizations", type=int, default=8,
                   help="Number of noisy x_obs draws per held-out sim.")
    p.add_argument("--min_bin_coverage", type=float, default=0.8,
                   help="Drop mass bins where < this fraction of sims have ≥1 halo.")
    p.add_argument("--level", type=float, default=0.6827,
                   help="Nominal credible level.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    edges = np.asarray(args.mass_bins, dtype=np.float64)
    centers = np.sqrt(edges[:-1] * edges[1:])

    theta, x, sim_ids = _gather_pairs(args, edges)
    n, p_params = theta.shape
    nb = x.shape[1]

    # Drop bins with too many NaNs across sims, then drop sims still with NaNs.
    bin_ok = (np.isfinite(x).mean(axis=0) >= args.min_bin_coverage)
    if not bin_ok.any():
        raise SystemExit("No mass bin satisfies --min_bin_coverage.")
    x = x[:, bin_ok]
    centers = centers[bin_ok]
    sim_ok = np.isfinite(x).all(axis=1)
    if sim_ok.sum() < 5:
        raise SystemExit("Too few sims with complete x after bin filter.")
    theta = theta[sim_ok]
    x = x[sim_ok]
    sim_ids = [sid for sid, keep in zip(sim_ids, sim_ok) if keep]
    n = theta.shape[0]
    print(f"[info] {n} sims × {x.shape[1]} mass bins after filtering")

    rng = np.random.default_rng(args.seed)

    # Leave-one-out coverage
    # For each held-out sim we draw `n_realizations` noisy x_obs and build the
    # posterior; per-param hit-counts accumulate over (sim × realization).
    n_trials = n * args.n_realizations
    hits = np.zeros(p_params, dtype=np.int64)
    abs_bias = np.zeros(p_params, dtype=np.float64)   # mean |μ_post − θ_true| / σ_post
    posterior_widths = np.zeros((n, p_params), dtype=np.float64)
    posterior_means = np.zeros((n, p_params), dtype=np.float64)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        emu = StackedEmulator.fit(theta[mask], x[mask], ridge=args.ridge)
        x_true = emu.predict(theta[i:i+1])[0]         # in-emulator prediction
        sigma_meas = args.noise_frac * np.abs(x_true)
        for _ in range(args.n_realizations):
            x_obs = x_true + rng.normal(scale=sigma_meas)
            post = GaussianPosterior.from_observation(
                emu, x_obs, sigma_meas=sigma_meas, prior_std=args.prior_std,
            )
            ok = central_credible_contains(post.mean, post.std, theta[i], level=args.level)
            hits += ok.astype(np.int64)
            abs_bias += np.abs(post.mean - theta[i]) / np.maximum(post.std, 1e-12)
        # Record the noiseless posterior for diagnostics
        post0 = GaussianPosterior.from_observation(
            emu, x_true, sigma_meas=sigma_meas, prior_std=args.prior_std,
        )
        posterior_widths[i] = post0.std
        posterior_means[i] = post0.mean

    coverage = hits / n_trials
    abs_bias /= n_trials
    # Binomial 1-σ error on coverage
    cov_err = np.sqrt(coverage * (1.0 - coverage) / n_trials)

    # Constraint fraction in std-θ space: 1 - σ_post / σ_prior.
    # 0 → pure prior; 1 → perfectly constrained.  Use the noiseless run.
    sigma_post_std = posterior_widths / np.maximum(theta.std(axis=0), 1e-12)
    constraint = 1.0 - sigma_post_std.mean(axis=0) / args.prior_std
    constraint = np.clip(constraint, 0.0, 1.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        coverage=coverage,
        coverage_err=cov_err,
        constraint=constraint,
        nominal_level=args.level,
        abs_bias_in_sigma=abs_bias,
        posterior_widths=posterior_widths,
        posterior_means=posterior_means,
        theta_truth=theta,
        x_truth=x,
        mass_centers=centers,
        n_sims=n,
        n_realizations=args.n_realizations,
        noise_frac=args.noise_frac,
        ridge=args.ridge,
        prior_std=args.prior_std,
        sim_ids=np.array(sim_ids),
        meta=np.array(repr({
            "aperture": args.aperture,
            "r_ap_mpc_h": args.r_ap_mpc_h,
            "model": args.model,
            "suites": args.suites,
            "hubble": args.hubble,
        })),
    )
    print(f"[save] {args.out}")
    print(f"# Validation E — leave-one-out coverage at level {args.level:.4f}")
    n_constrained = int((constraint > 0.1).sum())
    print(f"# {n_constrained}/{len(constraint)} params have constraint > 0.1 (data-informed)")
    worst = np.argsort(np.abs(coverage - args.level))[::-1][:5]
    for j in worst:
        print(f"  param {j:2d}: cov={coverage[j]:.3f} ± {cov_err[j]:.3f}  "
              f"|bias|/σ={abs_bias[j]:.2f}")


if __name__ == "__main__":
    main()
