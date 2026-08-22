"""QA the N1000 stat products written by ``bind.cli.lightcone_stats_mpi``.

Two modes, both cheap (only the small npz products are read — never the cubes):

``--compare``
    Diff each run's current products against a baseline copy (by default the
    pre-FFT-smoothing snapshot in ``analysis/stats_preFFT_20260816/``) and print
    the max relative change per statistic.  Expected after a FORCE re-stat:

    ================  =========================================================
    Cl_kappa cl       ~1e-15  (spectra never touched the smoothing kernel)
    nongauss pdf      ~1e-15  (PDF is built from the UNsmoothed map)
    variance, V0-V2   ~2e-4   (scipy 4-sigma-truncated kernel -> exact FFT)
    peak/minima       <1e-3 where populated, up to ~6% in the sparsest bins
    dm_bins, dm_pdf   ~5% / ~12% ONLY if the baseline came from the MPI path
                      before the shared-axis fix; ~0 against a serial baseline
    ================  =========================================================

``--audit``
    Sweep every traced run in the tree and flag products that fail to load, sit
    on the wrong nu axis, disagree with the fiducial sigma0 table, or carry a
    non-finite / non-monotonic DM axis.  Run this once the sweep finishes.

    python examples/n1000_stats_check.py --compare 0 1 2 21
    python examples/n1000_stats_check.py --audit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bind.inference.stats import NU_CANON

ROOT = Path("/mnt/home/mlee1/ceph/bind_n1000")
BASELINE = ROOT / "analysis/stats_preFFT_20260816"
PRODUCTS = ("Cl_kappa.npz", "Cl_kappa_y.npz", "Cl_tau.npz", "dm_stats.npz",
            "peak_counts.npz", "nongaussian_stats.npz")
#: Bin axes and run settings — identical by construction, so they would crowd the
#: "worst change" ranking with zeros.  ``dm_bins`` is deliberately NOT here: it is
#: the axis the shared-axis fix repairs, so it is always reported.
METADATA = frozenset({"ell", "nu", "pdf_bins", "mf_nu", "smoothing_arcmin",
                      "smoothing_scales_arcmin", "sigma_e", "shape_noise_ngal",
                      "TAU_PER_DM", "nu_sigma0_table", "nu_sigma0_unsmoothed_table"})


def run_dir(i: int, root: Path = ROOT) -> Path:
    """Manifest index -> run directory (same map as ``n1000_body.sh``)."""
    if i == 0:
        return root / "bind/run_0000"
    if i == 1:
        return root / "dmo/run_0000"
    if i == 2:
        return root / "truth/run_0000"
    if i <= 62:
        return root / f"twobound/run_{i - 3:04d}"
    return root / f"sb35/run_{i - 63:04d}"


def traced_indices(root: Path = ROOT) -> list[int]:
    return [i for i in range(319) if (run_dir(i, root) / ".trace_complete").exists()]


def max_rel(x, y) -> float:
    """Max relative difference; NaN when the shapes disagree."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape:
        return float("nan")
    s = np.maximum(np.abs(x), np.abs(y))
    return float(np.max(np.where(s > 0, np.abs(x - y) / np.maximum(s, 1e-300), 0.0)))


def compare(idxs: list[int], root: Path, baseline: Path) -> int:
    """Diff current products against the baseline snapshot.  Returns an exit code."""
    missing = 0
    for i in idxs:
        rd = run_dir(i, root)
        cat, name = rd.parent.name, rd.name
        print(f"\n=== task {i}: {cat}/{name} ===")
        for f in PRODUCTS:
            cur, base = rd / f, baseline / cat / f"{name}_{f}"
            if not base.exists():
                continue
            if not cur.exists():
                print(f"  {f:24s} MISSING in the run dir")
                missing += 1
                continue
            with np.load(cur) as a, np.load(base) as b:
                keys = [k for k in a.files if k in b.files
                        and np.asarray(a[k]).dtype.kind in "fiu"
                        and k not in METADATA]
                diffs = sorted(((max_rel(a[k], b[k]), k) for k in keys), reverse=True)
                stamp = "" if cur.stat().st_mtime > base.stat().st_mtime else "  (NOT re-run)"
                shape_changed = [k for v, k in diffs if v != v]
                finite = [(v, k) for v, k in diffs if v == v]
                if finite and finite[0][0] == 0.0:
                    head = "identical"
                else:
                    head = ", ".join(f"{k} {v:.2e}" for v, k in finite[:3])
                    if "dm_bins" in keys and "dm_bins" not in [k for _, k in finite[:3]]:
                        head += f", dm_bins {dict((k, v) for v, k in finite)['dm_bins']:.2e}"
                print(f"  {f:24s} {head}{stamp}"
                      + (f"  shape-changed: {shape_changed}" if shape_changed else ""))
    return 1 if missing else 0


def audit(root: Path, sigma0_table: Path) -> int:
    """Flag unloadable / off-axis / stale products across every traced run."""
    with np.load(sigma0_table) as t:
        want_fid = str(t["source_run"])
    bad: list[str] = []
    n_ok = 0
    idxs = traced_indices(root)
    for i in idxs:
        rd = run_dir(i, root)
        tag = f"{rd.parent.name}/{rd.name}"
        problems: list[str] = []
        has_tau = (rd / "tau_maps.npz").exists()
        want = ["Cl_kappa.npz", "peak_counts.npz", "nongaussian_stats.npz"]
        if has_tau:
            want += ["Cl_tau.npz", "dm_stats.npz"]
        for f in want:
            p = rd / f
            if not p.exists():
                problems.append(f"{f} missing")
                continue
            try:
                with np.load(p) as d:
                    for k in d.files:                       # forces a full read
                        _ = np.asarray(d[k]).shape
                    if f in ("peak_counts.npz", "nongaussian_stats.npz"):
                        if str(d["nu_grid"]) != "canon":
                            problems.append(f"{f} nu_grid={d['nu_grid']}")
                        if str(d["nu_fiducial"]) != want_fid:
                            problems.append(f"{f} fiducial={d['nu_fiducial']}")
                        axes = ([d["nu"]] if f.startswith("peak")
                                else [d["pdf_bins"], d["mf_nu"]])
                        for a in axes:
                            if a.shape != NU_CANON.shape or not np.allclose(a, NU_CANON):
                                problems.append(f"{f} off the canonical nu axis")
                    if f == "dm_stats.npz":
                        b, pdf = np.asarray(d["dm_bins"]), np.asarray(d["dm_pdf"])
                        if not np.all(np.isfinite(b)) or np.any(np.diff(b) <= 0):
                            problems.append("dm_bins not finite/increasing")
                        if not np.all(np.isfinite(pdf)):
                            problems.append("dm_pdf has non-finite entries")
                        else:
                            # np.histogram(density=True) normalises by the counts
                            # INSIDE the range, so a PDF averaged over one shared
                            # axis integrates to exactly 1 per source plane.  The
                            # pre-fix MPI products, whose per-realization axes were
                            # averaged together, land at 0.9984 — hence the tight
                            # tolerance: 5% would have waved that through.
                            norm = pdf.sum(-1) * float(b[1] - b[0])
                            if np.any(np.abs(norm - 1.0) > 1e-4):
                                problems.append("dm_pdf norm "
                                                f"{np.array2string(norm, precision=4)}"
                                                " (mixed per-realization axes?)")
            except Exception as e:                          # noqa: BLE001
                problems.append(f"{f} unreadable ({type(e).__name__})")
        if problems:
            bad.append(f"  {tag:22s} " + "; ".join(problems))
        else:
            n_ok += 1
    print(f"=== audit: {n_ok}/{len(idxs)} traced runs have clean products ===")
    if bad:
        print("\n".join(bad))
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", type=int, nargs="*", metavar="IDX",
                   help="manifest indices to diff against --baseline")
    p.add_argument("--audit", action="store_true", help="check every traced run")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--sigma0_table", type=Path,
                   default=ROOT / "analysis/nu_sigma0_bind.npz")
    args = p.parse_args()
    rc = 0
    if args.compare is not None:
        idxs = args.compare or [0, 1, 2, 21]
        rc |= compare(idxs, args.root, args.baseline)
    if args.audit:
        rc |= audit(args.root, args.sigma0_table)
    if args.compare is None and not args.audit:
        p.error("pass --compare [IDX ...] and/or --audit")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
