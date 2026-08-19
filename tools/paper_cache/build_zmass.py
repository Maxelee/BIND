#!/usr/bin/env python
"""Stitch per-snapshot `mass_table.pkl` files into one multi-redshift table.

Fig. R1 (`fig_z_mass_error`) must be measured on the same CV+1P+SB35 population
as the paper's z=0 Fig. 3, otherwise the two quote different stellar accuracies
purely from suite composition (+4.3% over CV+1P+SB35 vs -8.3% over SB35 alone,
for the *same* model). This driver runs the existing `build_metric.py mass`
reducer once per snapshot -- via `PAPER_SNAP`, so nothing about the estimator
changes -- and concatenates the results with a `snapshot`/`z` column.

The R200c aperture is correct at z>0 without any change: `paper_config.
r200_pix_patch` prefers the catalog's stored `r200s` (= FoF `Group_R_Crit200`),
which is already the comoving R200c at that snapshot. Verified against the z=0
formula on a z=1.045 catalog: stored/E(z)-corrected = 1.0001, stored/z=0-formula
= 1.3785.

Usage
-----
    # build partials + reduce for every snapshot, then stitch
    python tools/paper_cache/build_zmass.py --suite_root /mnt/home/mlee1/ceph/fm_redshift_suite \
        --model_subdir fm_redshift --pool 16

    # stitch only (partials already reduced)
    python tools/paper_cache/build_zmass.py --stitch_only
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from bind.data import SNAPSHOT_REDSHIFTS  # noqa: E402

# Matches tools/paper_cache/make_z_figures.py MAIN_SNAPS.
SNAPSHOTS = [90, 82, 74, 60, 52, 44]


def cache_dir_for(tag: str, snap: int) -> Path:
    return Path(f"/mnt/home/mlee1/ceph/paper_cache/{tag}_snap{snap:03d}")


def env_for(suite_root: str, model_subdir: str, tag: str, snap: int,
            mass_dir: str) -> dict:
    env = dict(os.environ)
    env.update(
        PAPER_SUITE_ROOT=suite_root,
        PAPER_MODEL_SUBDIR=model_subdir,
        PAPER_MODEL_TAG=tag,
        PAPER_SNAP=f"snap_{snap:03d}",
        # paper_config defaults this to the 1e12 low-mass study; the redshift
        # eval runs at the paper's trained-regime 1e13 cut, and a mismatch here
        # silently discovers zero simulations.
        PAPER_MASS_DIR=mass_dir,
        PAPER_CACHE_DIR=str(cache_dir_for(tag, snap)),
    )
    return env


def run_metric(env: dict, extra: list[str]) -> int:
    """Run build_metric.py, failing loudly on a non-zero exit.

    Ignoring the return code once let a crash in the parameter-response
    reduction pass unnoticed while the stitched table was written anyway, which
    is exactly the kind of half-built output this driver exists to prevent.
    """
    cmd = [sys.executable, str(HERE / "build_metric.py"), "--metric", "mass", *extra]
    print("  $ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env, cwd=str(REPO))
    if rc != 0:
        raise RuntimeError(f"build_metric.py exited {rc} for {env['PAPER_SNAP']} "
                           f"({' '.join(extra)})")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_redshift_suite")
    ap.add_argument("--model_subdir", default="fm_redshift")
    ap.add_argument("--tag", default="fm_redshift")
    ap.add_argument("--snapshots", type=int, nargs="*", default=SNAPSHOTS)
    ap.add_argument("--mass_dir", default="mass_threshold_1p000e13",
                    help="must match --halo_mass_min of the suite eval")
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--stitch_only", action="store_true")
    ap.add_argument("--out", default=None,
                    help="output pickle (default <cache>/mass_table_multiz.pkl)")
    args = ap.parse_args()

    frames, missing = [], []
    for snap in args.snapshots:
        cdir = cache_dir_for(args.tag, snap)
        env = env_for(args.suite_root, args.model_subdir, args.tag, snap,
                      args.mass_dir)
        if not args.stitch_only:
            print(f"[snap_{snap:03d}] building mass partials -> {cdir}")
            run_metric(env, ["--pool", str(args.pool)])
            run_metric(env, ["--reduce"])

        tbl = cdir / "mass_table.pkl"
        if not tbl.exists():
            missing.append(snap)
            print(f"[snap_{snap:03d}] MISSING {tbl}")
            continue
        df = pickle.load(open(tbl, "rb"))
        df = df.copy()
        df["snapshot"] = snap
        df["z"] = SNAPSHOT_REDSHIFTS[snap]
        frames.append(df)
        print(f"[snap_{snap:03d}] z={SNAPSHOT_REDSHIFTS[snap]:.4f}  {len(df)} halos  "
              f"suites={sorted(df['suite'].unique())}")

    if not frames:
        raise SystemExit("no per-snapshot mass tables found — run without --stitch_only first")

    out_df = pd.concat(frames, ignore_index=True)
    out = Path(args.out) if args.out else cache_dir_for(args.tag, args.snapshots[0]).parent / \
        f"{args.tag}_mass_table_multiz.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(out_df, f)

    print(f"\nwrote {out}  ({len(out_df)} halos)")
    print(out_df.groupby(["snapshot", "suite"]).size().unstack(fill_value=0).to_string())
    if missing:
        # Never let a partially-built table masquerade as complete: the whole
        # point of this table is a population that is identical across epochs.
        print(f"\nWARNING: snapshots with no mass_table.pkl: {missing}")
    # 1P has no hydro snapdir_082, so it is legitimately absent at z=0.209.
    for snap in args.snapshots:
        if snap in missing:
            continue
        sub = out_df[out_df.snapshot == snap]
        if snap != 82 and set(sub["suite"].unique()) != {"CV", "1P", "Test"}:
            print(f"WARNING: snap_{snap:03d} has suites {sorted(sub['suite'].unique())}, "
                  f"expected CV+1P+Test — the population is not matched across z")


if __name__ == "__main__":
    main()
