"""Build the WP-A3 1P paint bundle: run dirs + params.npy for the gate's 1P trajectories.

The rusty SB35 bundle (/mnt/ceph/users/mlee1/bind_sb35) carries 256 painted
Sobol points but no painted 1P runs; the A3 gate wants 1P trajectories (all
5 levels) for the key feedback parameters plus the fiducial. This script
selects those rows from the bundle's own design files (astro_params_1P.npy,
150 x 35 = 30 params x 5 levels, cosmology already fixed at TNG300) and lays
out a run tree that run_wp3_paint_1p.sh (same machinery as
run_sb35_generate.sh) can paint.

Usage (CPU, seconds):
    python analysis/paper3a/scripts/make_wp3_1p_bundle.py [--fiducial-only] [--params NAME [NAME ...]]

2026-07-16 update: Max pointed out the painted 1P *extremes* already exist —
/mnt/home/mlee1/ceph/bind_portable_twobound/runs/ holds all 30 params x 2
bounds (60 runs, 20 snapshots, same npz format incl. thermo). Only the
TNG-fiducial run is unpainted on rusty, so the expected invocation is now
``--fiducial-only`` (1 run; sbatch --array=0-0). The 5-level trajectory
build below is kept for the case where the gate finds two bounds too coarse
along some parameter direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BUNDLE = Path("/mnt/ceph/users/mlee1/bind_sb35")
OUT_ROOT = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate/1p_runs")

DEFAULT_PARAMS = [
    "WindEnergyIn1e51erg",       # SN wind energy (CAMELS ASN1 analog) — ejection
    "VariableWindVelFactor",     # SN wind velocity (ASN2 analog) — ejection
    "RadioFeedbackFactor",       # AGN kinetic-mode energy (AAGN1 analog) — heating
    "QuasarThreshold",           # kinetic/thermal mode threshold (AAGN2 analog)
    "BlackHoleFeedbackFactor",   # AGN feedback coupling — heating
    "BlackHoleAccretionFactor",  # BH growth rate — heating amplitude
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", nargs="+", default=DEFAULT_PARAMS)
    ap.add_argument("--fiducial-only", action="store_true",
                    help="build only the fiducial run (the 1P extremes already exist in bind_portable_twobound)")
    ap.add_argument("--out_root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    if args.fiducial_only:
        args.params = []

    meta = json.loads((BUNDLE / "design" / "astro_params_1P_meta.json").read_text())
    table = np.load(BUNDLE / "design" / "astro_params_1P.npy")
    fiducial = np.load(BUNDLE / "design" / "astro_params_fiducial.npy")
    known = sorted({m["param"] for m in meta})
    for name in args.params:
        if name not in known:
            raise SystemExit(f"unknown param {name!r}; valid names: {known}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    run_idx = 0

    def add_run(params_vec: np.ndarray, label: dict) -> None:
        nonlocal run_idx
        run_dir = args.out_root / f"run_{run_idx:04d}"
        run_dir.mkdir(exist_ok=True)
        np.save(run_dir / "params.npy", params_vec.astype(np.float64))
        manifest.append({"run": run_dir.name, **label})
        run_idx += 1

    add_run(fiducial, {"param": "fiducial", "level": None, "value": None})
    # meta[i] describes table row i (verified exactly: the varied slot holds
    # meta's value and every other slot equals the fiducial, 150/150 rows —
    # WP-A3 prep session 2026-07-16).
    for name in args.params:
        rows = [(i, m) for i, m in enumerate(meta) if m["param"] == name]
        for i, m in sorted(rows, key=lambda im: im[1]["level"]):
            add_run(table[i], {"param": name, "level": m["level"], "value": m["value"]})

    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {run_idx} run dirs under {args.out_root}")
    print(f"array range for sbatch: 0-{run_idx - 1}")
    for m in manifest:
        print(f"  {m['run']}: {m['param']} level={m['level']} value={m['value']}")


if __name__ == "__main__":
    main()
