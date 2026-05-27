"""Validation plot G — HMF coverage / mass-range constraint.

For each suite, report how many halos populate each ACT-DR6-relevant mass bin
across the full simulation set, and per simulation.  Flags mass bins where the
BIND τ(M) curve would be HMF-limited (too few halos for a reliable stack).

Reads only halo_catalog.npz (cheap), so this can be run before any model
generation.

Usage:
    python -m analysis.ksz.validation_g \\
        --testsuite_root /mnt/home/mlee1/ceph/fm_testsuite \\
        --suites CV 1P Test \\
        --mass_bins 1e13 2e13 5e13 1e14 1e15 \\
        --out analysis_physics_cache/ksz_validation_g.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ._io import find_sim_dirs, format_mass_tag


def _load_halo_masses(sim_dir: Path, halo_mass_min: float, snapshot: int | None
                      ) -> tuple[int, np.ndarray] | None:
    snap_dirs = sorted(
        p for p in sim_dir.iterdir() if p.is_dir() and p.name.startswith("snap_")
    )
    if not snap_dirs:
        return None
    snap_dir = snap_dirs[0]
    snap = int(snap_dir.name.removeprefix("snap_"))
    if snapshot is not None and snap != snapshot:
        return None
    cat_path = snap_dir / f"mass_threshold_{format_mass_tag(halo_mass_min)}" / "halo_catalog.npz"
    if not cat_path.exists():
        return None
    cat = np.load(cat_path)
    if "halo_masses" in cat.files:
        m = np.asarray(cat["halo_masses"], dtype=np.float64)
    elif "masses" in cat.files:
        m = np.asarray(cat["masses"], dtype=np.float64)
    else:
        return None
    return snap, m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testsuite_root", type=Path, required=True)
    p.add_argument("--suites", nargs="+", default=["CV", "1P", "Test"])
    p.add_argument("--halo_mass_min", type=float, default=1e13)
    p.add_argument("--snapshot", type=int, default=None,
                   help="If set, restrict to this snapshot.")
    p.add_argument("--mass_bins", nargs="+", type=float,
                   default=[1e13, 2e13, 5e13, 1e14, 1e15])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    edges = np.asarray(args.mass_bins, dtype=np.float64)
    n_bins = len(edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:])

    by_suite: dict[str, np.ndarray] = {}
    per_sim: dict[str, dict] = {}

    for suite in args.suites:
        sims = find_sim_dirs(args.testsuite_root, suite)
        if not sims:
            print(f"[skip] suite {suite}: no sim dirs")
            continue

        suite_counts = np.zeros(n_bins, dtype=np.int64)
        sim_count_rows: list[np.ndarray] = []
        sim_names: list[str] = []
        for sd in sims:
            r = _load_halo_masses(sd, args.halo_mass_min, args.snapshot)
            if r is None:
                continue
            _, m = r
            hist, _ = np.histogram(m, bins=edges)
            suite_counts += hist
            sim_count_rows.append(hist)
            sim_names.append(sd.name)
        by_suite[suite] = suite_counts
        per_sim[suite] = {
            "names": np.array(sim_names),
            "counts": np.stack(sim_count_rows) if sim_count_rows else np.zeros((0, n_bins), dtype=np.int64),
        }
        print(f"[ok]   suite {suite}: {len(sim_names)} sims, "
              f"halos/bin = {suite_counts.tolist()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "mass_edges": edges,
        "mass_centers": centers,
        "suites": np.array(args.suites),
    }
    for suite, counts in by_suite.items():
        save_kwargs[f"counts_{suite}"] = counts
        save_kwargs[f"sim_names_{suite}"] = per_sim[suite]["names"]
        save_kwargs[f"sim_counts_{suite}"] = per_sim[suite]["counts"]
    np.savez(args.out, **save_kwargs)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
