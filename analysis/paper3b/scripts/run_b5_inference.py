"""WP-B5 driver: LOO validation -> recovery suite -> (--fit-data) the fit.

Serial CPU, ~10-20 min total. Stages gate each other per the plan: the data
fit refuses to run unless the recovery summary passes. Outputs under
~/ceph/paper3/B/wp5_inference/: b5_loo.json, b5_recovery.json,
(then) b5_fit_<variant>.json + posterior npz + figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.paper3b.inference.gridmodel import GridEmulator, GridTable
from analysis.paper3b.inference.likelihood import (
    fit_frozen_data,
    selection_residual_ratios,
)
from analysis.paper3b.inference.recovery import recovery_suite

OUT = Path("/mnt/home/mlee1/ceph/paper3/B/wp5_inference")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-data", action="store_true",
                    help="run the frozen-data fit (requires recovery PASS)")
    ap.add_argument("--variant", default="wiener", choices=["wiener", "glimpse"])
    ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--skip-loo", action="store_true")
    ap.add_argument("--grid-suffix", default="",
                    help="e.g. _pg1024 to fit against an alternate B4 grid "
                         "(model_grid_tfwiener<suffix>.npz etc.); outputs are "
                         "tagged with the same suffix so frozen artifacts are "
                         "never overwritten.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = args.grid_suffix
    WP4 = Path("/mnt/home/mlee1/ceph/paper3/B/wp4_mocks")

    table = GridTable.load(
        grid_npz=WP4 / f"model_grid_tfwiener{suffix}.npz",
        fiducial_unit=WP4 / f"grid_tfwiener{suffix}" / "bind_run_0000.npz")
    em = GridEmulator.fit(table)
    sel = selection_residual_ratios(WP4 / f"b4_summary_tfwiener{suffix}.json")

    if not args.skip_loo:
        print("[b5] leave-one-out over", len(table.names), "points ...", flush=True)
        loo = em.loo_table()
        rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in loo.items()}
        (OUT / f"b5_loo{suffix}.json").write_text(json.dumps(rec, indent=2))
        print("  median |rel err| per bin:", np.round(loo["median_abs_rel"], 4))
        print("  pull std per bin:       ", np.round(loo["pull_std"], 2), flush=True)

    # injection covariance: frozen ERROR BARS only (central values unused)
    from analysis.paper3b.stack.frozen import load_frozen
    cov_inj = load_frozen(args.variant).cov_total

    print("[b5] recovery suite ...", flush=True)
    results, summary = recovery_suite(table, cov_inj, sel_ratios=sel,
                                      n_draws=args.n_draws)
    summary["points"] = [{"name": r.point_name,
                          "true": r.true_coords.tolist(),
                          "mean_pull": r.pulls.mean(axis=0).tolist(),
                          "cov68": float(r.in68.mean()),
                          "cov95": float(r.in95.mean())} for r in results]
    (OUT / f"b5_recovery{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("pull_mean", "pull_std", "coverage_68", "coverage_95",
                       "pass")}, indent=2), flush=True)

    if args.fit_data:
        if not summary["pass"]:
            raise SystemExit("recovery suite FAILED — the data fit is blocked "
                             "(plan acceptance criterion)")
        print(f"[b5] fitting the FROZEN {args.variant} data ...", flush=True)
        post, info = fit_frozen_data(
            em, variant=args.variant,
            sel_summary_json=WP4 / f"b4_summary_tfwiener{suffix}.json")
        np.savez(OUT / f"b5_posterior_{args.variant}{suffix}.npz",
                 mg=post.mg, dt=post.dt, lnlike=post.lnlike,
                 posterior=post.posterior)
        (OUT / f"b5_fit_{args.variant}{suffix}.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
