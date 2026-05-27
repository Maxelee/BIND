"""Validation plot B — per-halo annular τ profiles, BIND vs simulation truth.

For every halo this builds a τ(r) profile in fixed R/R200 bins by averaging
the gas-mass surface density (then converting to τ) inside concentric annuli
centred on the halo.  The output npz keeps per-halo profiles so downstream
plotters can slice by mass bin or by parameter regime (e.g. extreme-AGN vs
extreme-SN slices called out in docs/paper2_ksz_plan.md §4.B).

Usage:
    python -m analysis.ksz.validation_b \\
        --testsuite_root /mnt/home/mlee1/ceph/fm_testsuite \\
        --model fm_two_head --suites CV 1P Test \\
        --r_edges 0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 \\
        --out analysis_physics_cache/ksz_validation_b_fm_two_head.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ._io import find_sim_dirs, load_sim
from .tau_utils import gas_surface_density_to_tau


def _radius_grid(patch_pix: int) -> np.ndarray:
    c = (patch_pix - 1) / 2.0
    y, x = np.indices((patch_pix, patch_pix))
    return np.sqrt((x - c) ** 2 + (y - c) ** 2)


def _annular_profile(
    patches: np.ndarray,        # (N, P, P) gas-mass-per-pixel [Msun/h / pixel]
    r200_pix: np.ndarray,       # (N,) per-halo R200 in pixels
    r_edges_r200: np.ndarray,   # (K+1,) edges in units of R/R200
    pix_size_mpc_h: float,
    hubble: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-halo annular gas-surface-density and τ profiles.

    Returns
    -------
    sigma_profile : (N, K) Σ_gas [Msun/h / (Mpc/h)^2] averaged in each annulus
    tau_profile   : (N, K) τ (unitless) converted via gas_surface_density_to_tau
    """
    n, p, _ = patches.shape
    k = len(r_edges_r200) - 1
    rgrid = _radius_grid(p)
    pix_area_mpc_h2 = pix_size_mpc_h ** 2

    sigma = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        if r200_pix[i] <= 0:
            sigma[i] = np.nan
            continue
        r_over_r200 = rgrid / r200_pix[i]
        for j in range(k):
            mask = (r_over_r200 >= r_edges_r200[j]) & (r_over_r200 < r_edges_r200[j + 1])
            n_pix = int(mask.sum())
            if n_pix == 0:
                sigma[i, j] = np.nan
                continue
            mass_in_annulus = float(patches[i][mask].sum())
            area = n_pix * pix_area_mpc_h2
            sigma[i, j] = mass_in_annulus / area
    tau = gas_surface_density_to_tau(sigma, hubble=hubble)
    return sigma, tau


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testsuite_root", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--suites", nargs="+", default=["CV", "1P", "Test"])
    p.add_argument("--halo_mass_min", type=float, default=1e13)
    p.add_argument("--box_size", type=float, default=50.0)
    p.add_argument("--patch_size_mpc_h", type=float, default=6.25)
    p.add_argument("--hubble", type=float, default=0.6711)
    p.add_argument(
        "--r_edges",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
        help="Annulus edges in units of R/R200c (≥ 2 values).",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if len(args.r_edges) < 2:
        raise SystemExit("--r_edges needs at least two values.")
    r_edges = np.asarray(args.r_edges, dtype=np.float64)
    pix_size_mpc_h = args.patch_size_mpc_h / 128  # P=128 assumed; recomputed below per sim if different

    chunks: list[dict] = []
    for suite in args.suites:
        sims = find_sim_dirs(args.testsuite_root, suite)
        if not sims:
            print(f"[skip] suite {suite}: no sim dirs")
            continue
        n_ok = 0
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
            if art is None:
                continue

            pix_size_mpc_h = args.patch_size_mpc_h / art.patch_pix
            r200_pix = art.r200_mpc_h / pix_size_mpc_h
            sigma_b, tau_b = _annular_profile(
                art.bind_gas, r200_pix, r_edges, pix_size_mpc_h, args.hubble
            )
            sigma_t, tau_t = _annular_profile(
                art.truth_gas, r200_pix, r_edges, pix_size_mpc_h, args.hubble
            )
            chunks.append({
                "suite": np.array([suite] * len(art.halo_masses)),
                "sim_id": np.array([sd.name] * len(art.halo_masses)),
                "halo_mass_msun_h": art.halo_masses,
                "r200_mpc_h": art.r200_mpc_h,
                "params": art.params,
                "bind_tau_profile": tau_b,
                "truth_tau_profile": tau_t,
                "bind_sigma_profile": sigma_b,
                "truth_sigma_profile": sigma_t,
            })
            n_ok += 1
        print(f"[ok]   suite {suite}: {n_ok}/{len(sims)} sims processed")

    if not chunks:
        raise SystemExit("No simulations produced output; nothing to save.")

    merged: dict[str, np.ndarray] = {}
    for key in chunks[0].keys():
        merged[key] = np.concatenate([c[key] for c in chunks], axis=0)
    merged["r_edges_r200"] = r_edges
    merged["r_centers_r200"] = 0.5 * (r_edges[:-1] + r_edges[1:])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **merged)
    print(f"[save] {args.out}  ({len(merged['halo_mass_msun_h'])} halos, "
          f"{len(r_edges) - 1} annuli)")


if __name__ == "__main__":
    main()
