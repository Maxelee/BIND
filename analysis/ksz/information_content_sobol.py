"""Gate 2+3, fixed-halo Sobol version — the decisive SBI information test.

The SB35 inverse gate (`information_content.py`) was triple-confounded: ~100
sims, *different halos per sim*, and *cosmology varying*.  The cleaner design —
and the one BIND actually enables for SBI — is to paint the **same** CV halos at
many astrophysical-parameter points with cosmology fixed.  That cube already
exists from the scatter project:
``outputs/scatter_diagnostics/chunks_joint_cv/joint_part_*.npz`` —
128 Sobol designs × 1154 CV halos × 12 draws × 16 observables, with the 30
ASTRO params varied (`scan_idx`) and cosmology held fixed.

Here we reuse it as a kSZ pilot: the integrated gas mass M_gas (obs index 1) is
the τ proxy (τ ∝ aperture electron column ∝ gas mass).  We measure the *inverse*
held-out R²(θ_j | x) — can the gas observable recover each astro parameter? —
on this clean paired design, and compare narrow (gas only) vs multi-channel
(gas + stars).  This directly answers "does fixing halos+cosmo and adding
designs rescue identifiability vs the SB35 result?".

Caveat it does *not* dodge: a genuine forward-map degeneracy (two params with
the same gas signature) is not broken by more/cleaner data — only sharpened on
the surviving combination.  This gate measures whether the SB35 nulls were
noise/confound (then SBI is promising) or intrinsic (then it's a low-dim
direction + multi-probe story).  128 designs is a pilot; thousands would tighten
it (and is what NPE training should use).

Usage:
    python -m analysis.ksz.information_content_sobol \\
        --cube_dir outputs/scatter_diagnostics/chunks_joint_cv \\
        --out analysis_physics_cache/ksz_infocontent_sobol_cv.npz \\
        --fig figures/ksz_infocontent_sobol_cv.pdf
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from .information_content import _held_out_r2, _prep
from .param_meta import load_param_meta

# observable indices inside the cube's 16-obs axis (from the scatter project)
OBS_M_DM, OBS_M_GAS, OBS_M_STAR = 0, 1, 2


def _load_cube(cube_dir: Path):
    """Reconstruct (n_design, n_halo, n_obs) after median over draws, plus
    halo masses, the (n_design, 30) astro design, and scan_idx."""
    files = sorted(glob.glob(str(cube_dir / "joint_part_*.npz")))
    if not files:
        raise SystemExit(f"No joint_part_*.npz under {cube_dir}")
    cubes, masses = [], []
    sub = scan_idx = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        c = np.asarray(d["cube"], dtype=np.float64)          # (D, H_chunk, K, O)
        cubes.append(np.median(c, axis=2))                   # median over draws → (D, H_chunk, O)
        masses.append(np.asarray(d["masses"], dtype=np.float64))
        if sub is None:
            sub = np.asarray(d["sub"], dtype=np.float64)     # (D, 30) — shared
            scan_idx = np.asarray(d["scan_idx"], dtype=np.int64)
    cube = np.concatenate(cubes, axis=1)                     # (D, H_total, O)
    masses = np.concatenate(masses)                          # (H_total,)
    return cube, masses, sub, scan_idx


def _stack(cube, masses, edges, obs_idx, log=True):
    """Per-design mean (and fractional scatter) of an observable in mass bins.

    Returns (mean_feat (D, Km), scatter_feat (D, Km)).  mean is log10 of the
    halo-averaged raw obs; scatter is std/mean of the per-halo obs in the bin.
    """
    n_d = cube.shape[0]
    km = len(edges) - 1
    idx = np.digitize(masses, edges) - 1
    mean_feat = np.full((n_d, km), np.nan)
    scat_feat = np.full((n_d, km), np.nan)
    for k in range(km):
        sel = idx == k
        if sel.sum() < 3:
            continue
        vals = cube[:, sel, obs_idx]            # (D, n_in_bin)
        m = vals.mean(axis=1)                   # (D,)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_feat[:, k] = np.log10(np.clip(m, 1e-30, None)) if log else m
            scat_feat[:, k] = vals.std(axis=1) / np.clip(m, 1e-30, None)
    return mean_feat, scat_feat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube_dir", type=Path,
                    default=Path("outputs/scatter_diagnostics/chunks_joint_cv"))
    ap.add_argument("--mass_bins", nargs="+", type=float, default=[1e13, 3e13, 1e14, 1e15])
    ap.add_argument("--r2_thresh", type=float, default=0.1)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, default=None)
    args = ap.parse_args()

    meta = load_param_meta()
    edges = np.asarray(args.mass_bins, dtype=np.float64)

    cube, masses, sub, scan_idx = _load_cube(args.cube_dir)
    n_d, n_h, n_o = cube.shape
    print(f"[info] cube: {n_d} designs × {n_h} halos × {n_o} obs; "
          f"{len(scan_idx)} astro params varied, cosmology fixed")
    labels = [meta.labels[i] for i in scan_idx]
    is_cosmo = meta.is_cosmo[scan_idx]   # all False by construction (astro only)

    # observable blocks
    gas_mean, gas_scat = _stack(cube, masses, edges, OBS_M_GAS)
    star_mean, _ = _stack(cube, masses, edges, OBS_M_STAR)

    observables = {
        "gas_only":   gas_mean,                                  # kSZ-like (τ ∝ M_gas)
        "gas_rich":   np.hstack([gas_mean, gas_scat]),           # + per-bin scatter
        "gas+stars":  np.hstack([gas_mean, star_mean]),          # + stellar channel (multi-probe)
    }

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1)

    results = {}
    Y_full = sub  # (D, 30)
    for name, X in observables.items():
        Xc, row_mask, _ = _prep(X)
        Y = Y_full[row_mask]
        results[name] = {
            "r2_linear": _held_out_r2(Xc, Y, ridge, seed=args.seed),
            "r2_rf": _held_out_r2(Xc, Y, rf, seed=args.seed),
            "n_design": int(row_mask.sum()),
            "dim": X.shape[1],
        }

    def _count(r2):
        return int((r2 >= args.r2_thresh).sum())

    print(f"\n# Fixed-halo Sobol inverse gate — recoverable ASTRO params "
          f"(held-out R² ≥ {args.r2_thresh}) of {len(scan_idx)}:")
    for name in observables:
        nl = _count(results[name]["r2_linear"])
        nr = _count(results[name]["r2_rf"])
        print(f"  {name:10s} (dim {results[name]['dim']:2d})  "
              f"linear: {nl:2d}   RF: {nr:2d}")

    best = results["gas+stars"]["r2_linear"]
    order = np.argsort(-best)
    print("\n# top recoverable astro params (gas+stars, linear inverse):")
    for r in order[:10]:
        print(f"  p{scan_idx[r]:02d} {labels[r]:<24s}  "
              f"R²(gas+stars)={best[r]:+.3f}  "
              f"(gas_only={results['gas_only']['r2_linear'][r]:+.3f}  "
              f"RF={results['gas+stars']['r2_rf'][r]:+.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "scan_idx": scan_idx, "labels": np.array(labels),
        "r2_thresh": args.r2_thresh, "mass_edges": edges, "n_design": n_d,
    }
    for name, res in results.items():
        save[f"{name}_r2_linear"] = res["r2_linear"]
        save[f"{name}_r2_rf"] = res["r2_rf"]
    np.savez(args.out, **save)
    print(f"\n[save] {args.out}")

    if args.fig is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order20 = np.argsort(-np.maximum(results["gas+stars"]["r2_linear"],
                                         results["gas_only"]["r2_linear"]))
        y = np.arange(len(order20))
        fig, axx = plt.subplots(figsize=(8, 0.32 * len(order20) + 1.5))
        axx.barh(y + 0.2, np.clip(results["gas+stars"]["r2_linear"][order20], -0.05, None),
                 height=0.4, color="C0", label="gas+stars (linear)")
        axx.barh(y - 0.2, np.clip(results["gas_only"]["r2_linear"][order20], -0.05, None),
                 height=0.4, color="C3", alpha=0.8, label="gas only (linear)")
        axx.axvline(args.r2_thresh, color="k", ls="--", lw=1, label=f"recoverable ≥ {args.r2_thresh}")
        axx.set_yticks(y)
        axx.set_yticklabels([f"p{scan_idx[r]:02d} {labels[r]}" for r in order20], fontsize=7)
        axx.invert_yaxis()
        axx.set_xlabel(r"held-out $R^2(\theta_j \mid x)$  (fixed-halo Sobol, cosmo fixed)")
        axx.set_title(f"Fixed-halo Sobol inverse gate ({n_d} designs, 1154 CV halos)\n"
                      "does fixing halos+cosmo rescue astro identifiability?")
        axx.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig)
        plt.close(fig)
        print(f"[save] {args.fig}")


if __name__ == "__main__":
    main()
