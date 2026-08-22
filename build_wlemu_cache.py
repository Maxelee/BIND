#!/usr/bin/env python
"""build_wlemu_cache.py — standalone OpenMPI builder for the WL-emulator κ cache.

Pure numpy + mpi4py.  No ``bind`` import, no CLI framework — just pools the SB35
Sobol κ suite (``run_NNNN/kappa_maps.npz``) into the three files that
``bind.wlemu.KappaCache`` reads:

    <out>/kappa.npy   float16 (N_runs, n_real, n_z, R, R)   raw κ, mean-pooled
    <out>/meta.npz    params_native (N,35), run_ids, source_redshifts, res, fov
    <out>/norm.npz    per-z arcsinh softening λ + standardisation μ, σ

The work is embarrassingly parallel over runs; each rank pools its ``[rank::size]``
runs and writes disjoint rows of the shared memmap.  The κ normalisation is a
two-pass streamed reduction over the memmap, combined across ranks with
``MPI.Allreduce`` — never materialising the full (130 GB at 1024²) array.

**Idempotent / resumable:** if ``kappa.npy`` already holds the full data it skips
the (expensive) fill and only (re)computes ``norm.npz`` — so it also *finishes* a
cache whose pooled array is already written.  Pass ``--force`` to refill.

Run with OpenMPI (see run_wlemu_cache.sh)::

    module load python openmpi python-mpi
    source ~/venvs/BIND_env/bin/activate
    srun python build_wlemu_cache.py --out <dir> --resolution 1024

or serial (single process, for finishing / small caches)::

    python build_wlemu_cache.py --out <dir> --resolution 1024
"""

import argparse
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

RUN_RE = re.compile(r"run_(\d+)")
DEF_RUNS = "/mnt/home/mlee1/ceph/bind_sb35/runs"
DEF_DESIGN = "/mnt/home/mlee1/ceph/bind_sb35/design/astro_params_sobol.npy"


def run_id(path):
    return int(RUN_RE.search(os.path.basename(path)).group(1))


def mean_pool(x, factor):
    if factor == 1:
        return x
    *lead, h, w = x.shape
    n = h // factor
    return x.reshape(*lead, n, factor, n, factor).mean(axis=(-3, -1))


def pool_run(run_dir, factor, n_real):
    d = np.load(os.path.join(run_dir, "kappa_maps.npz"))
    k = d["kappa"]
    if n_real:
        k = k[:n_real]
    pooled = mean_pool(k.astype(np.float32), factor).astype(np.float16)
    return pooled, np.asarray(d["source_redshifts"], float), float(d["fov_deg"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs_dir", default=DEF_RUNS)
    ap.add_argument("--design_file", default=DEF_DESIGN)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--n_real", type=int, default=0, help="0 = all realisations")
    ap.add_argument("--chunk", type=int, default=8, help="rows per norm reduction chunk")
    ap.add_argument("--norm_rows", type=int, default=2000,
                    help="subsample ~this many (run,real) maps for the norm fit (0=all)")
    ap.add_argument("--force", action="store_true", help="refill even if kappa.npy exists")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if 1024 % args.resolution:
        raise SystemExit(f"resolution {args.resolution} must divide 1024")
    factor = 1024 // args.resolution
    n_real = args.n_real or None
    out = Path(args.out)
    kpath = out / "kappa.npy"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Deterministic run list — identical on every rank (race-free [rank::size]).
    run_dirs = sorted(
        (p for p in glob.glob(str(Path(args.runs_dir) / "run_*"))
         if os.path.exists(os.path.join(p, "kappa_maps.npz"))),
        key=run_id)
    if not run_dirs:
        raise SystemExit(f"no run_*/kappa_maps.npz under {args.runs_dir}")
    N = len(run_dirs)
    run_ids = np.array([run_id(p) for p in run_dirs], dtype=np.int64)

    # Need a fill?  Skip if a matching memmap is already on disk.
    need_fill = args.force or not kpath.exists()
    if kpath.exists() and not args.force:
        need_fill = np.load(kpath, mmap_mode="r").shape[0] != N
    if rank == 0:
        print(f"[wlemu-cache] {N} runs · res {args.resolution}² · {size} rank(s) · "
              f"fill={need_fill}", flush=True)

    # ── Phase 1+2: allocate + pool (only if needed) ───────────────────────────
    if need_fill:
        t0 = time.time()
        if rank == 0:
            p0, zs, fov = pool_run(run_dirs[0], factor, n_real)
            nr, nz, R = p0.shape[0], p0.shape[1], p0.shape[-1]
            mm = np.lib.format.open_memmap(
                kpath, mode="w+", dtype=np.float16, shape=(N, nr, nz, R, R))
            mm[0] = p0
            mm.flush()
            del mm
            design = np.load(args.design_file)
            np.savez(out / "meta.npz",
                     params_native=design[run_ids].astype(np.float64),
                     source_redshifts=zs, run_ids=run_ids,
                     resolution=np.int64(R), fov_deg=np.float64(fov),
                     n_real=np.int64(nr), n_runs=np.int64(N))
        comm.Barrier()
        arr = np.load(kpath, mmap_mode="r+")
        for i in range(rank, N, size):
            if i == 0:
                continue                                  # rank 0 wrote it above
            arr[i] = pool_run(run_dirs[i], factor, n_real)[0]
        arr.flush()
        del arr
        comm.Barrier()
        if rank == 0:
            print(f"[wlemu-cache] filled {N} runs in {time.time() - t0:.0f}s", flush=True)

    # ── Phase 3: κ normalisation (per-z λ, μ, σ), reduced across ranks ─────────
    # λ/μ/σ are just normalisation *scales* — std(κ) converges in a few hundred
    # maps, so we estimate them from an evenly-strided subsample of the rows
    # (~--norm_rows total) in float32.  Reading all 13 G pixels twice to fit a
    # scale is pointless; the subsample is identical to many digits.
    t0 = time.time()
    arr = np.load(kpath, mmap_mode="r")
    nz, R = arr.shape[2], arr.shape[-1]
    flat = arr.reshape(-1, nz, R, R)                       # view; merge (run,real)
    M = flat.shape[0]
    rows = np.arange(M)
    if args.norm_rows and M > args.norm_rows:
        rows = rows[:: max(1, M // args.norm_rows)]
    my_rows = rows[rank::size]

    def reduce_pass(fn):
        s1, s2, cnt = np.zeros(nz), np.zeros(nz), 0
        for a in range(0, len(my_rows), args.chunk):
            blk = fn(np.asarray(flat[my_rows[a:a + args.chunk]], dtype=np.float32))
            s1 += blk.sum(axis=(0, 2, 3), dtype=np.float64)
            s2 += np.square(blk, dtype=np.float64).sum(axis=(0, 2, 3))
            cnt += blk.shape[0]
        S1, S2 = np.zeros(nz), np.zeros(nz)
        comm.Allreduce(s1, S1, op=MPI.SUM)
        comm.Allreduce(s2, S2, op=MPI.SUM)
        n = comm.allreduce(cnt, op=MPI.SUM) * R * R
        mean = S1 / n
        return mean, np.sqrt(np.maximum(S2 / n - mean ** 2, 1e-30))

    _, lam = reduce_pass(lambda b: b)                      # λ_z = std(κ)
    lam_b = lam[None, :, None, None].astype(np.float32)
    mu, sigma = reduce_pass(lambda b: np.arcsinh(b / lam_b))   # μ,σ of arcsinh(κ/λ)
    if rank == 0:
        print(f"[wlemu-cache] norm from {len(rows)}/{M} rows", flush=True)

    if rank == 0:
        zs = np.load(out / "meta.npz")["source_redshifts"]
        tmp = out / f"_norm_tmp_{os.getpid()}.npz"      # atomic write (rename)
        np.savez(tmp, lam=lam, mu=mu, sigma=sigma, source_redshifts=zs)
        os.replace(tmp, out / "norm.npz")
        print(f"[wlemu-cache] norm in {time.time() - t0:.0f}s  "
              f"λ={np.round(lam, 4)}", flush=True)
        print(f"[wlemu-cache] DONE → {out}", flush=True)


if __name__ == "__main__":
    main()
