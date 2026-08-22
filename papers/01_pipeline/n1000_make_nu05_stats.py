"""Per-run ``nu05_stats.npz`` for the campaign tree (pure repackaging).

``family_basis_all.py`` / ``refresh_amplitude_sets_mf22.py`` read a small
per-run bundle of the nu-binned statistics.  Everything in it is already
produced by the campaign's stats pass -- ``peak_counts.npz`` (peaks/minima and
their errors) and ``nongaussian_stats.npz`` (pdf, V0/V1/V2) -- both already on
the canonical 22-point axis, so this just re-shapes, it does not recompute.

Refuses to write unless BOTH sources are on ``stats.NU_CANON``: a run left on
the old axis would otherwise be silently mixed into the family basis.

    python3 papers/01_pipeline/n1000_make_nu05_stats.py            # all runs
    python3 papers/01_pipeline/n1000_make_nu05_stats.py --glob 'twobound/run_*'
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bind.inference.stats import NU_CANON

N1K = Path("/mnt/home/mlee1/ceph/bind_n1000")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=N1K)
    ap.add_argument("--glob", default="*/run_*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    made = skipped = bad = 0
    for run in sorted(a.root.glob(a.glob)):
        out = run / "nu05_stats.npz"
        pk_p, ng_p = run / "peak_counts.npz", run / "nongaussian_stats.npz"
        if not (pk_p.exists() and ng_p.exists()):
            continue
        if out.exists() and not a.force:
            skipped += 1
            continue
        try:
            pk, ng = np.load(pk_p), np.load(ng_p)
            for arr, what in ((pk["nu"], "peak_counts"), (ng["mf_nu"], "nongaussian")):
                if arr.shape != NU_CANON.shape or not np.allclose(arr, NU_CANON):
                    raise ValueError(f"{what} is not on NU_CANON")
            np.savez(out, nu=NU_CANON,
                     peak_counts=pk["peak_counts"], peak_counts_err=pk["peak_counts_err"],
                     minima_counts=pk["minima_counts"],
                     minima_counts_err=pk["minima_counts_err"],
                     pdf=ng["pdf"], mf_v0=ng["V0"], mf_v1=ng["V1"], mf_v2=ng["V2"])
            made += 1
        except Exception as e:                                   # noqa: BLE001
            print(f"  [skip] {run.parent.name}/{run.name}: {e}")
            bad += 1
    print(f"nu05_stats.npz: {made} written, {skipped} already present, {bad} unusable")


if __name__ == "__main__":
    main()
