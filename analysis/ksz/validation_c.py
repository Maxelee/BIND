"""Validation plot C — Spearman τ–parameter sensitivity, BIND vs truth.

For each of the 35 CAMELS parameters, compute the Spearman rank correlation
between the stacked τ and that parameter, separately for BIND and for the
truth.  Plotting layer renders a 35-bar comparison.

**Sim-level aggregation (no pseudo-replication).**  Within a single SB35 sim
*all halos share the same θ*, so per-halo τ values are not independent samples
of the θ→τ map — pooling them inflates the effective N from N_sims to N_halos
and makes Spearman p-values meaningless, while halo-to-halo scatter dilutes ρ.
We therefore reduce each sim to **one stacked τ per mass bin** (the mean over
that sim's halos in the bin) and compute the Spearman *across sims*.  Doing it
per mass bin also removes the confound that the halo mass function itself
shifts with Ω_m/σ8 across the suite.  ``n`` reported per panel is the number
of contributing sims, and the p-values are now honest.

The τ estimator matches D/E/F (default: ACT-style compensated aperture ``cap``
at a fixed physical radius), so the whole pipeline measures one observable.

By default we use the SB35-style ``Test`` suite, which has independent
variation across all 35 axes (CV varies only seed → zero param variance).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ._io import find_sim_dirs, load_sim
from .tau_utils import per_halo_tau


def _per_sim_binned_means(
    tau: np.ndarray, mass: np.ndarray, edges: np.ndarray
) -> tuple[float, np.ndarray]:
    """Reduce one sim to (all-halo mean τ, per-mass-bin mean τ).  NaN if empty."""
    finite = np.isfinite(tau)
    all_mean = float(np.mean(tau[finite])) if finite.any() else np.nan
    nb = len(edges) - 1
    binmeans = np.full(nb, np.nan, dtype=np.float64)
    idx = np.digitize(mass, edges) - 1
    for k in range(nb):
        sel = (idx == k) & finite
        if sel.any():
            binmeans[k] = float(np.mean(tau[sel]))
    return all_mean, binmeans


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
    p.add_argument("--aperture", choices=["disk", "cap"], default="cap",
                   help="τ estimator; 'cap' is the kSZ-canonical observable (matches D/E/F).")
    p.add_argument("--r_ap_mpc_h", type=float, default=0.5,
                   help="Aperture (or CAP inner-disk) radius in Mpc/h.")
    p.add_argument("--mass_bins", nargs="+", type=float,
                   default=[1e13, 3e13, 1e14, 1e15],
                   help="Compute one Spearman per mass bin in addition to all-halo.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sims = find_sim_dirs(args.testsuite_root, args.suite)
    if not sims:
        raise SystemExit(f"No sim dirs under {args.testsuite_root / args.suite}")

    edges = np.asarray(args.mass_bins, dtype=np.float64)
    n_bins = len(edges) - 1

    theta_rows: list[np.ndarray] = []
    bind_all: list[float] = []
    truth_all: list[float] = []
    bind_bins: list[np.ndarray] = []
    truth_bins: list[np.ndarray] = []

    n_ok = 0
    n_p = 0
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

        pix_size = args.patch_size_mpc_h / art.patch_pix
        r_ap_pix = args.r_ap_mpc_h / pix_size
        tau_b = per_halo_tau(art.bind_gas, r_ap_pix, pix_size, args.hubble,
                             estimator=args.aperture)
        tau_t = per_halo_tau(art.truth_gas, r_ap_pix, pix_size, args.hubble,
                             estimator=args.aperture)

        b_all, b_bins = _per_sim_binned_means(tau_b, art.halo_masses, edges)
        t_all, t_bins = _per_sim_binned_means(tau_t, art.halo_masses, edges)

        theta_rows.append(art.params[0])     # all halos share θ → one row per sim
        bind_all.append(b_all)
        truth_all.append(t_all)
        bind_bins.append(b_bins)
        truth_bins.append(t_bins)
        n_p = art.params.shape[1]
        n_ok += 1
    print(f"[ok]   suite {args.suite}: {n_ok}/{len(sims)} sims processed")

    if n_ok < 5:
        raise SystemExit(f"Only {n_ok} sims — too few for a sim-level Spearman.")

    theta = np.asarray(theta_rows, dtype=np.float64)          # (S, P)
    bind_all_arr = np.asarray(bind_all, dtype=np.float64)     # (S,)
    truth_all_arr = np.asarray(truth_all, dtype=np.float64)
    bind_bins_arr = np.asarray(bind_bins, dtype=np.float64)   # (S, n_bins)
    truth_bins_arr = np.asarray(truth_bins, dtype=np.float64)

    def _spearman_over_sims(tau_sim: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Spearman of each param vs the per-sim stacked τ, across sims."""
        rho = np.full(n_p, np.nan)
        pval = np.full(n_p, np.nan)
        good_sim = np.isfinite(tau_sim)
        n_eff = int(good_sim.sum())
        if n_eff < 5:
            return rho, pval, n_eff
        y = tau_sim[good_sim]
        for j in range(n_p):
            x = theta[good_sim, j]
            if np.allclose(x, x[0]):
                continue
            r, pv = spearmanr(x, y)
            rho[j] = r
            pval[j] = pv
        return rho, pval, n_eff

    bins_info: list[dict] = []
    rho_b, pv_b, n_eff = _spearman_over_sims(bind_all_arr)
    rho_t, pv_t, _ = _spearman_over_sims(truth_all_arr)
    bins_info.append({"label": "all", "n": n_eff,
                      "rho_bind": rho_b, "p_bind": pv_b,
                      "rho_truth": rho_t, "p_truth": pv_t})
    for b in range(n_bins):
        rho_b, pv_b, n_eff = _spearman_over_sims(bind_bins_arr[:, b])
        rho_t, pv_t, _ = _spearman_over_sims(truth_bins_arr[:, b])
        bins_info.append({
            "label": f"logM_{np.log10(edges[b]):.2f}_{np.log10(edges[b+1]):.2f}",
            "n": n_eff,
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
        "meta": np.array(repr({
            "aggregation": "sim-level (one stacked tau per sim per bin)",
            "n_sims": int(n_ok),
            "aperture": args.aperture,
            "r_ap_mpc_h": args.r_ap_mpc_h,
            "model": args.model,
            "suite": args.suite,
        })),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out_arrays)
    print(f"[save] {args.out}  (n_params={n_p}, n_bins={len(bins_info)}, "
          f"n_sims={n_ok})")


if __name__ == "__main__":
    main()
