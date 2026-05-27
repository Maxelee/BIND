"""Validation plot C — Spearman τ–parameter sensitivity, BIND vs truth.

For each of the 35 CAMELS parameters, compute the Spearman rank correlation
between the per-halo aperture-integrated τ and that parameter, separately for
BIND and for the truth.  Plotting layer renders a 35-bar comparison.

This is the §4.C check from docs/paper2_ksz_plan.md: "BIND responds to subgrid
parameters *the right way for τ specifically*, not just for integrated mass."

Aggregation is across simulations within each suite (e.g. SB35-holdout sweeps
all 35 dimensions; CV varies only seed; 1P varies one parameter at a time).
By default we use the SB35-style ``Test`` suite, which has independent variation
across all 35 axes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ._io import find_sim_dirs, load_sim
from .tau_utils import (
    aperture_gas_mass,
    gas_mass_to_tau_in_aperture,
)


def _aperture_radius_pix(r200_mpc_h: np.ndarray, r200_factor: float,
                         patch_size_mpc_h: float, patch_pix: int,
                         fixed_r_pix: float) -> np.ndarray:
    r = r200_mpc_h * r200_factor * (patch_pix / patch_size_mpc_h)
    r[r200_mpc_h <= 0] = float(fixed_r_pix)
    return r


def _per_halo_tau(patches: np.ndarray, r_pix: np.ndarray,
                  patch_size_mpc_h: float, patch_pix: int,
                  hubble: float) -> np.ndarray:
    pix_size_mpc_h = patch_size_mpc_h / patch_pix
    masses = np.empty(patches.shape[0], dtype=np.float64)
    areas = np.empty(patches.shape[0], dtype=np.float64)
    for i, r in enumerate(r_pix):
        masses[i] = aperture_gas_mass(patches[i:i + 1], float(r))[0]
        areas[i] = np.pi * (float(r) * pix_size_mpc_h) ** 2
    return gas_mass_to_tau_in_aperture(masses, areas, hubble=hubble)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testsuite_root", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--suite", default="Test",
                   help="Single suite with broad parameter variation (default Test).")
    p.add_argument("--halo_mass_min", type=float, default=1e13)
    p.add_argument("--box_size", type=float, default=50.0)
    p.add_argument("--patch_size_mpc_h", type=float, default=6.25)
    p.add_argument("--hubble", type=float, default=0.6711)
    p.add_argument("--r200_factor", type=float, default=1.0)
    p.add_argument("--fixed_r_pix", type=float, default=8.0)
    p.add_argument("--mass_bins", nargs="+", type=float,
                   default=[1e13, 3e13, 1e14, 1e15],
                   help="Compute one Spearman per mass bin in addition to all-halo.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sims = find_sim_dirs(args.testsuite_root, args.suite)
    if not sims:
        raise SystemExit(f"No sim dirs under {args.testsuite_root / args.suite}")

    all_mass: list[np.ndarray] = []
    all_params: list[np.ndarray] = []
    all_tau_bind: list[np.ndarray] = []
    all_tau_truth: list[np.ndarray] = []

    n_ok = 0
    for sd in sims:
        try:
            art = load_sim(
                sd, suite=args.suite, model_name=args.model,
                halo_mass_min=args.halo_mass_min,
                box_size=args.box_size,
                patch_size_mpc_h=args.patch_size_mpc_h,
            )
        except Exception as exc:
            print(f"[err]  {args.suite}/{sd.name}: {exc}")
            continue
        if art is None or art.params.shape[1] == 0:
            continue

        r_pix = _aperture_radius_pix(
            art.r200_mpc_h, args.r200_factor,
            args.patch_size_mpc_h, art.patch_pix, args.fixed_r_pix,
        )
        tau_b = _per_halo_tau(art.bind_gas, r_pix,
                              args.patch_size_mpc_h, art.patch_pix, args.hubble)
        tau_t = _per_halo_tau(art.truth_gas, r_pix,
                              args.patch_size_mpc_h, art.patch_pix, args.hubble)

        all_mass.append(art.halo_masses)
        all_params.append(art.params)
        all_tau_bind.append(tau_b)
        all_tau_truth.append(tau_t)
        n_ok += 1
    print(f"[ok]   suite {args.suite}: {n_ok}/{len(sims)} sims processed")

    if not all_mass:
        raise SystemExit("Nothing to save.")

    mass = np.concatenate(all_mass)
    params = np.concatenate(all_params, axis=0)
    tau_b = np.concatenate(all_tau_bind)
    tau_t = np.concatenate(all_tau_truth)
    n_p = params.shape[1]

    def _spearman_vs_params(tau, sel):
        rho = np.full(n_p, np.nan)
        pval = np.full(n_p, np.nan)
        if sel.sum() < 5:
            return rho, pval
        good = sel & np.isfinite(tau)
        if good.sum() < 5:
            return rho, pval
        tau_g = tau[good]
        for j in range(n_p):
            x = params[good, j]
            if np.allclose(x, x[0]):
                continue
            r, pv = spearmanr(x, tau_g)
            rho[j] = r
            pval[j] = pv
        return rho, pval

    edges = np.asarray(args.mass_bins, dtype=np.float64)
    bin_idx = np.digitize(mass, edges) - 1

    bins_info: list[dict] = []
    # All-halo entry first
    sel_all = np.ones(len(mass), dtype=bool)
    rho_b, pv_b = _spearman_vs_params(tau_b, sel_all)
    rho_t, pv_t = _spearman_vs_params(tau_t, sel_all)
    bins_info.append({"label": "all", "lo": float(edges[0]), "hi": float(edges[-1]),
                      "n": int(sel_all.sum()),
                      "rho_bind": rho_b, "p_bind": pv_b,
                      "rho_truth": rho_t, "p_truth": pv_t})
    # Per-mass-bin entries
    for b in range(len(edges) - 1):
        sel = bin_idx == b
        rho_b, pv_b = _spearman_vs_params(tau_b, sel)
        rho_t, pv_t = _spearman_vs_params(tau_t, sel)
        bins_info.append({
            "label": f"logM_{np.log10(edges[b]):.2f}_{np.log10(edges[b+1]):.2f}",
            "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "n": int(sel.sum()),
            "rho_bind": rho_b, "p_bind": pv_b,
            "rho_truth": rho_t, "p_truth": pv_t,
        })

    out_arrays = {
        "n_params": np.int32(n_p),
        "param_idx": np.arange(n_p, dtype=np.int32),
        "labels": np.array([b["label"] for b in bins_info]),
        "n_per_bin": np.array([b["n"] for b in bins_info], dtype=np.int32),
        "rho_bind": np.stack([b["rho_bind"] for b in bins_info]),
        "rho_truth": np.stack([b["rho_truth"] for b in bins_info]),
        "p_bind": np.stack([b["p_bind"] for b in bins_info]),
        "p_truth": np.stack([b["p_truth"] for b in bins_info]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out_arrays)
    print(f"[save] {args.out}  (n_params={n_p}, n_bins={len(bins_info)})")


if __name__ == "__main__":
    main()
