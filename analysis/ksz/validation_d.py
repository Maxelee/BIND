"""Validation plot D — stacked τ(M) in ACT-DR6-like apertures.

For each halo, integrate τ inside a fixed *physical* aperture (Mpc/h) matched
to the ACT-DR6 stack geometry, then average within mass bins to produce a
stacked τ(M) curve.  Both BIND and the simulation truth are stacked through
the same aperture so the comparison is apples-to-apples.

This is the §4.D check from docs/paper2_ksz_plan.md and the first step toward
the §3.2 mock-observable build-out.  Aperture geometry options:

  --aperture disk          : centred disk of radius R_ap_mpc_h
  --aperture cap           : ACT-style compensated aperture (CAP):
                             disk r_in = R_ap_mpc_h, annulus r_out = √2 r_in,
                             yields τ_disk − ⟨τ⟩_annulus.

The CAP filter is the closest match to what ACT-DR6 reports.

Usage:
    python -m analysis.ksz.validation_d \\
        --testsuite_root /mnt/home/mlee1/ceph/fm_testsuite \\
        --model fm_two_head --suites CV 1P Test \\
        --aperture cap --r_ap_mpc_h 0.5 \\
        --mass_bins 1e13 2e13 5e13 1e14 1e15 \\
        --out analysis_physics_cache/ksz_validation_d_fm_two_head.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ._io import find_sim_dirs, load_sim
from .tau_utils import (
    aperture_cap_signal,
    aperture_gas_mass,
    gas_mass_to_tau_in_aperture,
    gas_surface_density_to_tau,
)


def _per_halo_tau_disk(
    patches: np.ndarray,
    r_ap_pix: float,
    pix_size_mpc_h: float,
    hubble: float,
) -> np.ndarray:
    mass = aperture_gas_mass(patches, r_ap_pix)
    area = np.pi * (r_ap_pix * pix_size_mpc_h) ** 2
    return gas_mass_to_tau_in_aperture(mass, area, hubble=hubble)


def _per_halo_tau_cap(
    patches: np.ndarray,
    r_ap_pix: float,
    pix_size_mpc_h: float,
    hubble: float,
) -> np.ndarray:
    """CAP-filtered τ: aperture sum with weights {+1 in disk, −n_in/n_ann in annulus}.

    The CAP weights have ∑ w = 0, so a uniform background cancels exactly.
    aperture_cap_signal(.) returns ∑ w · M  [Msun/h]; dividing by the disk
    area gives an effective Σ_gas estimate, which we convert to τ.
    """
    cap_mass = aperture_cap_signal(patches, r_ap_pix)
    disk_area = np.pi * (r_ap_pix * pix_size_mpc_h) ** 2  # area of the +1 disk
    sigma_eff = cap_mass / disk_area
    return gas_surface_density_to_tau(sigma_eff, hubble=hubble)


def _stack(values: np.ndarray, mass: np.ndarray, edges: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean, median, std, n) with NaN tolerance."""
    k = len(edges) - 1
    mean = np.full(k, np.nan)
    med = np.full(k, np.nan)
    sem = np.full(k, np.nan)  # standard error on the mean
    n = np.zeros(k, dtype=np.int64)
    bin_idx = np.digitize(mass, edges) - 1
    for b in range(k):
        sel = (bin_idx == b) & np.isfinite(values)
        n[b] = int(sel.sum())
        if n[b] == 0:
            continue
        v = values[sel]
        mean[b] = float(np.mean(v))
        med[b] = float(np.median(v))
        if n[b] > 1:
            sem[b] = float(np.std(v, ddof=1) / np.sqrt(n[b]))
    return mean, med, sem, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testsuite_root", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--suites", nargs="+", default=["CV", "1P", "Test"])
    p.add_argument("--halo_mass_min", type=float, default=1e13)
    p.add_argument("--box_size", type=float, default=50.0)
    p.add_argument("--patch_size_mpc_h", type=float, default=6.25)
    p.add_argument("--hubble", type=float, default=0.6711)
    p.add_argument("--aperture", choices=["disk", "cap"], default="cap",
                   help="Aperture geometry.  'cap' is the ACT-DR6-style filter.")
    p.add_argument("--r_ap_mpc_h", type=float, default=0.5,
                   help="Aperture (or CAP-inner-disk) radius in Mpc/h.")
    p.add_argument("--mass_bins", nargs="+", type=float,
                   default=[1e13, 2e13, 5e13, 1e14, 1e15],
                   help="Halo-mass bin edges [Msun/h] for the stack.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    all_mass: list[np.ndarray] = []
    all_tau_b: list[np.ndarray] = []
    all_tau_t: list[np.ndarray] = []

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

            pix_size = args.patch_size_mpc_h / art.patch_pix
            r_ap_pix = args.r_ap_mpc_h / pix_size
            if r_ap_pix < 1.0:
                print(f"[warn] {sd.name}: R_ap = {r_ap_pix:.2f} pix < 1")
            if args.aperture == "cap":
                # CAP outer radius = √2 r_in pixels; check it fits in the patch
                r_out_pix = np.sqrt(2.0) * r_ap_pix
                if r_out_pix >= art.patch_pix / 2:
                    print(f"[warn] {sd.name}: CAP outer radius {r_out_pix:.1f} px "
                          f">= half-patch ({art.patch_pix // 2}); truncated.")
                tau_b = _per_halo_tau_cap(art.bind_gas, r_ap_pix, pix_size, args.hubble)
                tau_t = _per_halo_tau_cap(art.truth_gas, r_ap_pix, pix_size, args.hubble)
            else:
                tau_b = _per_halo_tau_disk(art.bind_gas, r_ap_pix, pix_size, args.hubble)
                tau_t = _per_halo_tau_disk(art.truth_gas, r_ap_pix, pix_size, args.hubble)

            all_mass.append(art.halo_masses)
            all_tau_b.append(tau_b)
            all_tau_t.append(tau_t)
            n_ok += 1
        print(f"[ok]   suite {suite}: {n_ok}/{len(sims)} sims processed")

    if not all_mass:
        raise SystemExit("Nothing to save.")

    mass = np.concatenate(all_mass)
    tau_b = np.concatenate(all_tau_b)
    tau_t = np.concatenate(all_tau_t)

    edges = np.asarray(args.mass_bins, dtype=np.float64)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean per bin

    bind_mean, bind_med, bind_sem, n_b = _stack(tau_b, mass, edges)
    truth_mean, truth_med, truth_sem, n_t = _stack(tau_t, mass, edges)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mass_edges=edges,
        mass_centers=centers,
        n_per_bin=n_b.astype(np.int64),  # n is the same for both
        bind_tau_mean=bind_mean,
        bind_tau_median=bind_med,
        bind_tau_sem=bind_sem,
        truth_tau_mean=truth_mean,
        truth_tau_median=truth_med,
        truth_tau_sem=truth_sem,
        # also keep per-halo arrays for diagnostics
        halo_mass_msun_h=mass,
        bind_tau_per_halo=tau_b,
        truth_tau_per_halo=tau_t,
        meta=np.array(repr({
            "aperture": args.aperture,
            "r_ap_mpc_h": args.r_ap_mpc_h,
            "model": args.model,
            "suites": args.suites,
            "hubble": args.hubble,
        })),
    )
    print(f"[save] {args.out}  ({len(mass)} halos, {len(centers)} mass bins)")


if __name__ == "__main__":
    main()
