"""``paired_stats.npz`` from the ALREADY-COMPUTED per-run statistics (seconds).

What the MPI builder ``n1000_paired_stats.py`` actually computes, for cl/peaks/
minima, is recoverable from the stats npz files that the stats sweep already
wrote -- so re-reading two 21 GB kappa cubes to get it is wasted work.

Both the response and its error are derivable:

    resp = (mean_run - mean_fid) / mean_fid
    err  = sqrt(var_run + var_fid) / n / |mean_fid|
         = sqrt(err_run^2 + err_fid^2) / |mean_fid|      <- stored errs ARE sqrt(var/n)

The second line is exactly the MPI builder's formula (see its ``stat``/``err``
block): it accumulates the run and fiducial arms SEPARATELY and never forms the
difference, so no covariance term enters and the result is the unpaired
combination.  Its docstring claims a seed-paired scatter; the code does not do
that.  Reproducing it here is therefore an identity, not an approximation --
validate with ``--check``.

The ONE thing the MPI builder genuinely adds is V0/V1/V2 errors:
``nongaussian_stats.npz`` stores Minkowski means with no error keys, so the MF
errors cannot be reconstructed and are written as NaN here.  ``family_basis_all``
consumes them via ``snr_resp("V0_resp", "V0_err")``, so the MF S/N rows need the
MPI product; every other consumer is fully served by this script.

    python3 papers/01_pipeline/n1000_paired_stats_fast.py --glob 'twobound/run_*'
    python3 papers/01_pipeline/n1000_paired_stats_fast.py --check      # vs MPI output
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import NG_SCALES_DEFAULT, NU_CANON

N1K = Path("/mnt/home/mlee1/ceph/bind_n1000")


def _resp(rm, re_, fm, fe):
    """Fractional response and its (unpaired) error -- the MPI builder's formula."""
    rm, re_, fm, fe = (np.asarray(x, float) for x in (rm, re_, fm, fe))
    ok = fm != 0
    resp, err = np.zeros_like(fm), np.zeros_like(fm)
    resp[ok] = (rm[ok] - fm[ok]) / fm[ok]
    err[ok] = np.sqrt(re_[ok] ** 2 + fe[ok] ** 2) / np.abs(fm[ok])
    return resp, err


def _diag(a):
    """(5,5,nell) cross matrix -> (5,nell) autos, matching per_real's stacking."""
    a = np.asarray(a, float)
    return np.stack([a[z, z] for z in range(a.shape[0])])


def _n_real(run: Path) -> int:
    with zipfile.ZipFile(run / "kappa_maps.npz") as z, z.open("kappa.npy") as f:
        v = fmt.read_magic(f)
        shp, _, _ = (fmt.read_array_header_1_0(f) if v == (1, 0)
                     else fmt.read_array_header_2_0(f))
    return int(shp[0])


def build(run: Path, fid_cache) -> dict:
    fcl, fpk, fng = fid_cache
    rcl = np.load(run / "Cl_kappa.npz")
    rpk = np.load(run / "peak_counts.npz")
    rng = np.load(run / "nongaussian_stats.npz")
    out = {"ell": fcl["ell"], "nu": NU_CANON, "mf_nu": NU_CANON,
           "mf_scales": np.asarray(NG_SCALES_DEFAULT), "nu_fixed": np.array(1),
           "err_kind": np.str_("unpaired"), "mf_err_available": np.array(0)}
    out["clk_resp"], out["clk_err"] = _resp(_diag(rcl["cl"]), _diag(rcl["cl_err"]),
                                            _diag(fcl["cl"]), _diag(fcl["cl_err"]))
    for nm, a, b in (("pk", "peak_counts", "peak_counts_err"),
                     ("min", "minima_counts", "minima_counts_err")):
        out[f"{nm}_resp"], out[f"{nm}_err"] = _resp(rpk[a], rpk[b], fpk[a], fpk[b])
    for v in ("V0", "V1", "V2"):
        z = np.zeros_like(np.asarray(fng[v], float))
        out[f"{v}_resp"], _ = _resp(rng[v], z, fng[v], z)
        out[f"{v}_err"] = np.full_like(out[f"{v}_resp"], np.nan)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=N1K)
    ap.add_argument("--glob", default="twobound/run_*")
    ap.add_argument("--fid", default="twobound/run_0049")
    ap.add_argument("--out_name", default="paired_stats_fast.npz")
    ap.add_argument("--check", action="store_true",
                    help="compare against existing paired_stats.npz instead of writing")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    fid = a.root / a.fid
    fid_cache = (np.load(fid / "Cl_kappa.npz"), np.load(fid / "peak_counts.npz"),
                 np.load(fid / "nongaussian_stats.npz"))

    if a.check:
        print(f"{'run':10s} {'key':10s} {'max|rel diff|':>14s}")
        for run in sorted(a.root.glob(a.glob)):
            ref_p = run / "paired_stats.npz"
            if not ref_p.exists() or not (run / "Cl_kappa.npz").exists():
                continue
            ref, got = np.load(ref_p), build(run, fid_cache)
            for k in ("clk_resp", "clk_err", "pk_resp", "pk_err",
                      "min_resp", "min_err", "V0_resp", "V1_resp", "V2_resp"):
                x, y = np.asarray(ref[k], float), np.asarray(got[k], float)
                if x.shape != y.shape:
                    print(f"{run.name:10s} {k:10s}  SHAPE {x.shape} vs {y.shape}")
                    continue
                d = np.abs(x - y) / np.maximum(np.abs(x), 1e-30)
                m = np.nanmax(np.where(np.isfinite(d), d, 0.0))
                print(f"{run.name:10s} {k:10s} {m:14.3e}")
        return

    made = skipped = 0
    for run in sorted(a.root.glob(a.glob)):
        out = run / a.out_name
        if out.exists() and not a.force:
            skipped += 1
            continue
        if not (run / "nongaussian_stats.npz").exists():
            continue
        d = build(run, fid_cache)
        d["n_real"] = np.array(_n_real(run))
        np.savez(out, **d)
        made += 1
    print(f"{made} written, {skipped} present  (V0/V1/V2 _err are NaN -- MF S/N needs the MPI product)")


if __name__ == "__main__":
    main()
