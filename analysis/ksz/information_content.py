"""Gates 2 + 3 — how much of θ is recoverable from the observable?

Quantifies the *inverse* information content of the τ observable: for each of
the 35 parameters, the held-out (cross-validated) R² of an emulator that
predicts θ_j FROM the stacked observable x.  R²(θ_j | x) > 0 means the
observable constrains θ_j; ≈ 0 means it does not.  This is the honest,
overfit-proof analogue of validation_e's `constraint` and the direct precursor
to SBI: if no feedback param is recoverable, NPE/NLE will only return the prior.

Two knobs, matching the SBI go/no-go question:

  * **Gate 2 (nonlinearity).** Compare a linear Ridge inverse to a nonlinear
    RandomForest inverse on the *same* observable.  If the nonlinear model
    recovers params the linear one cannot, validation_e's linear surrogate was
    the bottleneck (not the observable).
  * **Gate 3 (richer observable).** Compare the narrow observable (stacked CAP
    τ per mass bin) to a rich observable (+ annular τ profiles + per-halo τ
    scatter per mass bin).  If the rich observable lifts feedback params above
    the recoverability floor, build SBI on the rich summary.

Both BIND and truth observables are scored: truth R² is the information ceiling
of the observable; BIND R² is what BIND's (slightly biased) forward map delivers.

Usage:
    python -m analysis.ksz.information_content \\
        --testsuite_root /mnt/home/mlee1/ceph/fm_testsuite_cube \\
        --model fm_cube_two_head --suite Test \\
        --out analysis_physics_cache/ksz_infocontent_fm_cube_two_head.npz \\
        --fig figures/ksz_infocontent_fm_cube_two_head.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

from ._io import find_sim_dirs, load_sim, los_advisory
from .param_meta import load_param_meta
from .tau_utils import per_halo_tau
from .validation_b import _annular_profile


def _per_sim_features(art, args, mass_edges, r_edges, which: str):
    """Build (narrow, rich) feature vectors for one sim from BIND or truth gas.

    narrow = mean CAP τ per mass bin.
    rich   = narrow ++ mean annular τ (mass × annulus) ++ per-bin τ scatter.
    Returns (narrow (Km,), rich (Km + Km*Ka + Km,)).  NaN where a bin is empty.
    """
    patches = art.bind_gas if which == "bind" else art.truth_gas
    pix_size = args.patch_size_mpc_h / art.patch_pix
    r_ap_pix = args.r_ap_mpc_h / pix_size

    tau = per_halo_tau(patches, r_ap_pix, pix_size, args.hubble, estimator=args.aperture)
    r200_pix = np.asarray(art.r200_mpc_h, dtype=np.float64) / pix_size
    _, tau_prof = _annular_profile(patches, r200_pix, r_edges, pix_size, args.hubble)  # (N, Ka)

    idx = np.digitize(art.halo_masses, mass_edges) - 1
    km = len(mass_edges) - 1
    ka = len(r_edges) - 1
    cap = np.full(km, np.nan)
    scat = np.full(km, np.nan)
    prof = np.full((km, ka), np.nan)
    for k in range(km):
        sel = (idx == k) & np.isfinite(tau)
        if sel.any():
            cap[k] = np.mean(tau[sel])
            scat[k] = np.std(tau[sel]) if sel.sum() > 1 else 0.0
            pk = tau_prof[sel]
            with np.errstate(invalid="ignore"):
                prof[k] = np.nanmean(pk, axis=0)
    narrow = cap
    rich = np.concatenate([cap, prof.ravel(), scat])
    return narrow, rich


def _held_out_r2(X, Y, model, n_splits=5, seed=0):
    """Per-target held-out R² via K-fold cross_val_predict."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, Y, cv=kf)
    return np.array([r2_score(Y[:, j], pred[:, j]) for j in range(Y.shape[1])])


def _prep(X):
    """Standardize features; drop columns finite in <90% of sims, then drop
    rows with any remaining NaN.  Returns (X_clean, row_mask)."""
    finite_frac = np.isfinite(X).mean(axis=0)
    keep_col = finite_frac >= 0.9
    X = X[:, keep_col]
    row_mask = np.isfinite(X).all(axis=1)
    Xc = X[row_mask]
    mu, sd = Xc.mean(0), Xc.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xc - mu) / sd, row_mask, keep_col


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testsuite_root", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="Test", help="Broad-variation suite (SB35-like).")
    ap.add_argument("--halo_mass_min", type=float, default=1e13)
    ap.add_argument("--box_size", type=float, default=50.0)
    ap.add_argument("--patch_size_mpc_h", type=float, default=6.25)
    ap.add_argument("--hubble", type=float, default=0.6711)
    ap.add_argument("--aperture", choices=["disk", "cap"], default="cap")
    ap.add_argument("--r_ap_mpc_h", type=float, default=0.5)
    ap.add_argument("--mass_bins", nargs="+", type=float, default=[1e13, 3e13, 1e14, 1e15])
    ap.add_argument("--r_edges", nargs="+", type=float, default=[0.25, 0.5, 1.0, 1.5, 2.0],
                    help="Annulus edges in R/R200 for the rich observable.")
    ap.add_argument("--r2_thresh", type=float, default=0.1,
                    help="Held-out R² above which a param is 'recoverable'.")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, default=None)
    args = ap.parse_args()

    meta = load_param_meta()
    mass_edges = np.asarray(args.mass_bins, dtype=np.float64)
    r_edges = np.asarray(args.r_edges, dtype=np.float64)

    sims = find_sim_dirs(args.testsuite_root, args.suite)
    if not sims:
        raise SystemExit(f"No sim dirs under {args.testsuite_root / args.suite}")

    thetas: list[np.ndarray] = []
    Xn_b, Xr_b, Xn_t, Xr_t = [], [], [], []
    banner = False
    for sd in sims:
        try:
            art = load_sim(sd, suite=args.suite, model_name=args.model,
                           halo_mass_min=args.halo_mass_min, box_size=args.box_size,
                           patch_size_mpc_h=args.patch_size_mpc_h)
        except Exception as exc:
            print(f"[err]  {sd.name}: {exc}")
            continue
        if art is None or art.params.shape[1] == 0:
            continue
        if not banner:
            print(los_advisory(art.truth_source, art.los_depth_mpc_h, args.aperture))
            banner = True
        nb, rb = _per_sim_features(art, args, mass_edges, r_edges, "bind")
        nt, rt = _per_sim_features(art, args, mass_edges, r_edges, "truth")
        thetas.append(art.params[0])
        Xn_b.append(nb); Xr_b.append(rb); Xn_t.append(nt); Xr_t.append(rt)

    theta = np.asarray(thetas, dtype=np.float64)
    n_p = theta.shape[1]
    if len(theta) < 20:
        raise SystemExit(f"Only {len(theta)} sims — too few for cross-validated R².")
    print(f"[info] {len(theta)} sims; narrow dim={len(Xn_b[0])}, rich dim={len(Xr_b[0])}")

    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.seed,
                               n_jobs=-1)

    results = {}
    for name, Xlist in (("narrow_bind", Xn_b), ("rich_bind", Xr_b),
                        ("narrow_truth", Xn_t), ("rich_truth", Xr_t)):
        X = np.asarray(Xlist, dtype=np.float64)
        Xc, row_mask, _ = _prep(X)
        Y = theta[row_mask]
        r2_lin = _held_out_r2(Xc, Y, ridge, seed=args.seed)
        r2_rf = _held_out_r2(Xc, Y, rf, seed=args.seed)
        results[name] = {"r2_linear": r2_lin, "r2_rf": r2_rf, "n_sims": int(row_mask.sum())}

    # ── verdict tables ──────────────────────────────────────────────────────
    def _count(r2):
        rec = r2 >= args.r2_thresh
        return int(rec.sum()), int((rec & ~meta.is_cosmo).sum())

    print(f"\n# Gates 2+3 — recoverable params (held-out R² ≥ {args.r2_thresh}), "
          f"feedback in parentheses:")
    for name in ("narrow_bind", "rich_bind", "narrow_truth", "rich_truth"):
        nl, fl = _count(results[name]["r2_linear"])
        nr, fr = _count(results[name]["r2_rf"])
        print(f"  {name:13s}  linear: {nl:2d} ({fl} fb)   RF: {nr:2d} ({fr} fb)")

    # top recoverable feedback params on the best (rich_bind RF) variant
    rf_rich = results["rich_bind"]["r2_rf"]
    fb_order = [d for d in np.argsort(-rf_rich) if not meta.is_cosmo[d]]
    print("\n# top recoverable FEEDBACK params (rich observable, BIND, RF inverse):")
    for d in fb_order[:8]:
        print(f"  p{d:02d} {meta.labels[d]:<22s}  R²(rich,BIND,RF)={rf_rich[d]:+.3f}  "
              f"(narrow,RF={results['narrow_bind']['r2_rf'][d]:+.3f}  "
              f"truth ceiling={results['rich_truth']['r2_rf'][d]:+.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "param_idx": np.arange(n_p, dtype=np.int32),
        "labels": np.array(meta.labels),
        "is_cosmo": meta.is_cosmo,
        "r2_thresh": args.r2_thresh,
        "mass_edges": mass_edges, "r_edges": r_edges,
    }
    for name, res in results.items():
        save[f"{name}_r2_linear"] = res["r2_linear"]
        save[f"{name}_r2_rf"] = res["r2_rf"]
    np.savez(args.out, **save)
    print(f"\n[save] {args.out}")

    # ── figure: per-param R² for the key variants ───────────────────────────
    if args.fig is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(-np.maximum(results["rich_bind"]["r2_rf"],
                                       results["rich_truth"]["r2_rf"]))
        order = order[:20]
        y = np.arange(len(order))
        fig, axx = plt.subplots(figsize=(8, 0.34 * len(order) + 1.5))
        axx.barh(y + 0.27, np.clip(results["rich_truth"]["r2_rf"][order], -0.05, None),
                 height=0.22, color="0.5", label="rich · truth (ceiling)")
        axx.barh(y + 0.0, np.clip(results["rich_bind"]["r2_rf"][order], -0.05, None),
                 height=0.22, color="C0", label="rich · BIND · RF")
        axx.barh(y - 0.27, np.clip(results["narrow_bind"]["r2_rf"][order], -0.05, None),
                 height=0.22, color="C3", alpha=0.8, label="narrow · BIND · RF")
        axx.axvline(args.r2_thresh, color="k", ls="--", lw=1, label=f"recoverable ≥ {args.r2_thresh}")
        axx.set_yticks(y)
        axx.set_yticklabels([f"p{d:02d} {meta.labels[d]}"
                             + ("  (cosmo)" if meta.is_cosmo[d] else "") for d in order],
                            fontsize=7)
        axx.invert_yaxis()
        axx.set_xlabel(r"held-out $R^2(\theta_j \mid x)$")
        axx.set_title("Gates 2+3 — inverse recoverability of θ from the τ observable")
        axx.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig)
        plt.close(fig)
        print(f"[save] {args.fig}")


if __name__ == "__main__":
    main()
