"""Streaming per-realization stats for the N1000 campaign (workstation-friendly).

Computes the same statistics as ``bind.cli.lightcone_stats`` but **one
realization at a time**, streamed straight out of the npz zip member, so a
(1000, 5, 1024, 1024) cube never lands in memory (~21 MB working set instead of
21 GB).  Written for the case where the Slurm stats array can't be submitted
(queue at the submit cap) and the reduction has to run locally.

Per realization it stores, for every source plane:
  cl        (n_real, 5, n_ell)   auto angular power (Pylians, fov 5 deg)
  peaks     (n_real, 5, 68)      peak counts vs S/N nu (2 arcmin smoothing)
  minima    (n_real, 5, 68)      minimum counts vs the same nu bins
  V0/V1/V2  (n_real, 5, 29)      Minkowski functionals (1 arcmin, S/N units)
  pdf       (n_real, 5, 41)      kappa PDF on FIXED edges (see --pdf_ref)

Binning is pinned to the production convention so the output is directly
comparable with the 50-real products: nu bins from ``stats.peak_counts``
defaults, MF thresholds from ``stats.nongaussian_stats`` defaults, and PDF edges
taken from an existing ``nongaussian_stats.npz`` (per-realization PDF ranges
would otherwise float with each map's own std).

    python examples/n1000_stats_stream.py --runs truth/run_0000 dmo/run_0000
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import (
    _gaussian_smooth,
    minkowski_functionals,
    peak_counts,
    power_spectrum,
)

N1K = Path("/mnt/home/mlee1/ceph/bind_n1000")
PDF_REF = Path("/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/nongaussian_stats.npz")


def stream_realizations(npz: Path, key: str, n_max: int | None = None):
    """Yield (idx, (n_src, N, N) float32) without decompressing the whole array."""
    with zipfile.ZipFile(npz) as z, z.open(key + ".npy") as f:
        v = fmt.read_magic(f)
        shape, fortran, dt = (fmt.read_array_header_1_0(f) if v == (1, 0)
                              else fmt.read_array_header_2_0(f))
        if fortran:
            raise ValueError("unexpected Fortran-order npy")
        n_real = shape[0] if n_max is None else min(shape[0], n_max)
        per = int(np.prod(shape[1:])) * dt.itemsize
        for i in range(n_real):
            buf = f.read(per)
            if len(buf) < per:
                return
            yield i, np.frombuffer(buf, dtype=dt).reshape(shape[1:])


def pdf_edges(ref: Path, n_bins: int = 41) -> np.ndarray:
    c = np.load(ref)["pdf_bins"]
    d = float(np.diff(c).mean())
    return np.concatenate([c - 0.5 * d, [c[-1] + 0.5 * d]])[: n_bins + 1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="e.g. truth/run_0000")
    p.add_argument("--root", type=Path, default=N1K)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n_real", type=int, default=None, help="cap (benchmarking)")
    p.add_argument("--fov_deg", type=float, default=5.0)
    p.add_argument("--smoothing_arcmin", type=float, default=2.0)
    p.add_argument("--mf_smoothing_arcmin", type=float, default=1.0)
    p.add_argument("--pdf_ref", type=Path, default=PDF_REF)
    args = p.parse_args()
    out_dir = args.out or (args.root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    nu_bins = np.linspace(-5.0, 12.0, 69)
    mf_nu = np.linspace(-3.0, 4.0, 29)
    edges = pdf_edges(args.pdf_ref)

    for run in args.runs:
        src = args.root / run / "kappa_maps.npz"
        if not src.exists():
            print(f"[skip] {run}: no kappa_maps.npz", flush=True)
            continue
        tag = run.replace("/", "_")
        dest = out_dir / f"stream_stats_{tag}.npz"
        t0 = time.time()
        cl_l, pk_l, mn_l, v0_l, v1_l, v2_l, pdf_l = [], [], [], [], [], [], []
        ell = None
        for i, cube in stream_realizations(src, "kappa", args.n_real):
            pc = peak_counts(cube[None], fov_deg=args.fov_deg,
                             smoothing_arcmin=args.smoothing_arcmin,
                             nu_bins=nu_bins, return_realizations=True)
            pk_l.append(pc["peak_counts_real"][0])       # (n_src, 68)
            mn_l.append(pc["minima_counts_real"][0])
            cl_r, v_r, pdf_r = [], [], []
            for s in range(cube.shape[0]):
                m = cube[s].astype(np.float64)
                ell, cl = power_spectrum(m, fov_deg=args.fov_deg)
                cl_r.append(cl)
                sm = _gaussian_smooth(m, args.mf_smoothing_arcmin, args.fov_deg)
                nu = (sm - sm.mean()) / sm.std()
                v_r.append(np.stack(minkowski_functionals(nu, mf_nu)))
                pdf_r.append(np.histogram(m, bins=edges, density=True)[0])
            cl_l.append(np.asarray(cl_r))
            v = np.asarray(v_r)                          # (n_src, 3, n_thr)
            v0_l.append(v[:, 0])
            v1_l.append(v[:, 1])
            v2_l.append(v[:, 2])
            pdf_l.append(np.asarray(pdf_r))
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"[{tag}] {i+1} reals  {el/(i+1):.2f}s/real  elapsed {el/60:.1f} min",
                      flush=True)
                np.savez_compressed(dest, ell=ell, nu=0.5 * (nu_bins[1:] + nu_bins[:-1]),
                                    mf_nu=mf_nu, pdf_edges=edges, n_done=i + 1,
                                    cl=np.asarray(cl_l), peaks=np.asarray(pk_l),
                                    minima=np.asarray(mn_l), V0=np.asarray(v0_l),
                                    V1=np.asarray(v1_l), V2=np.asarray(v2_l),
                                    pdf=np.asarray(pdf_l))
        np.savez_compressed(dest, ell=ell, nu=0.5 * (nu_bins[1:] + nu_bins[:-1]),
                            mf_nu=mf_nu, pdf_edges=edges, n_done=len(cl_l),
                            cl=np.asarray(cl_l), peaks=np.asarray(pk_l),
                            minima=np.asarray(mn_l), V0=np.asarray(v0_l),
                            V1=np.asarray(v1_l), V2=np.asarray(v2_l),
                            pdf=np.asarray(pdf_l))
        print(f"[{tag}] DONE {len(cl_l)} reals in {(time.time()-t0)/60:.1f} min -> {dest}",
              flush=True)


if __name__ == "__main__":
    main()
