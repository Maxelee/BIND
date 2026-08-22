"""Per-realization field-statistics cache for the N=1000 campaign (MPI).

The 50-real cache (``field_cache.py``) loads whole 21 GB cubes into memory and
reads ``bind_science``; neither works for the campaign tree.  This builder:

* streams realizations out of the compressed npz (never materialising a cube),
* computes EVERY product for a side in ONE pass over the maps (the old builder
  made six separate passes),
* shards by MPI rank -- each rank writes only its own realizations plus their
  global indices -- then rank 0 merges them back into realization order, which
  avoids Gatherv for the ragged per-realization arrays,
* writes the same key names the figure builders expect
  (``kk_bind``/``pk_truth``/``pdf_bins``/``ell``/...), so fig05/fig06 need no
  change beyond pointing FID_FC at the n1000 cache.

Sides: ``bind`` = the paper fiducial replica (twobound/run_0049 by default,
BIND_FID_REPLICA to change), ``truth`` = truth/run_0000.  Unlike the 50-real
release, the campaign's fiducial has its OWN tau trace, so tau is read from the
run itself rather than from bind_lightcone_tng.

    srun -n 48 python3 -u papers/01_pipeline/build_field_cache_n1000.py --side bind
    srun -n 48 python3 -u papers/01_pipeline/build_field_cache_n1000.py --side truth
    python3 papers/01_pipeline/build_field_cache_n1000.py --merge
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import NU_CANON, peak_counts, power_spectrum

CEPH = Path("/mnt/home/mlee1/ceph")
N1K = CEPH / "bind_n1000"
CACHE_DIR = N1K / "field_cache"
SHARD_DIR = CACHE_DIR / "shards"
CACHE = CACHE_DIR / "field_stats_fid_n1000.npz"
N_PLANES, FOV_DEG, SMOOTH_PK = 5, 5.0, 2.0
REPLICA = os.environ.get("BIND_FID_REPLICA", "tb49")
_RUN = {"tb18": "run_0018", "tb49": "run_0049", "tb53": "run_0053"}[REPLICA]


def run_dir(side: str) -> Path:
    return {"bind": N1K / ("twobound/" + _RUN),
            "truth": N1K / "truth/run_0000",
            "dmo": N1K / "dmo/run_0000"}[side]


def stream(side: str, field: str, idx: list[int]):
    """Yield (global_index, (n_src, N, N) float32) for the wanted realizations."""
    path = run_dir(side) / f"{field}_maps.npz"
    want = set(idx)
    with zipfile.ZipFile(path) as z, z.open(f"{field}.npy") as f:
        v = fmt.read_magic(f)
        shape, _, dt = (fmt.read_array_header_1_0(f) if v == (1, 0)
                        else fmt.read_array_header_2_0(f))
        per = int(np.prod(shape[1:])) * dt.itemsize
        for i in range(shape[0]):
            buf = f.read(per)
            if len(buf) < per:
                return
            if i in want:
                yield i, np.frombuffer(buf, dtype=dt).reshape(shape[1:])


def n_real_of(side: str) -> int:
    with zipfile.ZipFile(run_dir(side) / "kappa_maps.npz") as z, z.open("kappa.npy") as f:
        v = fmt.read_magic(f)
        shape, _, _ = (fmt.read_array_header_1_0(f) if v == (1, 0)
                       else fmt.read_array_header_2_0(f))
    return int(shape[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=("bind", "truth", "dmo"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--n_real", type=int, default=None)
    args = ap.parse_args()

    if args.merge:
        store: dict[str, np.ndarray] = {}
        for side in ("bind", "truth", "dmo"):
            shards = sorted(SHARD_DIR.glob(f"{side}_rank*.npz"))
            if not shards:
                if side == "dmo":
                    print("[merge] no dmo shards — skipping (kk_dmo will be absent)")
                    continue
                raise SystemExit(f"no shards for side={side} — run the compute step first")
            parts = [np.load(s) for s in shards]
            idx = np.concatenate([p["idx"] for p in parts])
            order = np.argsort(idx)
            for key in parts[0].files:
                if key in ("idx", "ell", "pdf_bins"):
                    continue
                arr = np.concatenate([p[key] for p in parts])[order]
                store[f"{key}_{side}"] = arr
            store["ell"] = parts[0]["ell"]
            store["pdf_bins"] = parts[0]["pdf_bins"]
            print(f"[merge] {side}: {len(idx)} realizations from {len(shards)} shards")
        # provenance the figure builders print/assert on
        store["fid_run"] = np.str_(f"twobound/{_RUN}")
        store["n_real"] = np.int64(store["kk_bind"].shape[0])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **store)
        print(f"[merge] wrote {CACHE}  keys={sorted(store)}")
        return

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    side = args.side
    nr = n_real_of(side) if args.n_real is None else min(args.n_real, n_real_of(side))
    mine = list(range(rank, nr, size))
    has_tau = (run_dir(side) / "tau_maps.npz").exists()
    has_y = (run_dir(side) / "y_maps.npz").exists()      # dmo has neither

    idx, kk, yy, ky, tt, kt, pk, mn, pdf = [], [], [], [], [], [], [], [], []
    edges = np.concatenate([NU_CANON - 0.25, [NU_CANON[-1] + 0.25]])
    ell = None
    ks = stream(side, "kappa", mine)
    ys = stream(side, "y", mine) if has_y else None
    ts = stream(side, "tau", mine) if has_tau else None
    for i, K in ks:
        Y = next(ys)[1] if ys is not None else None
        T = next(ts)[1] if ts is not None else None
        # keep float32 throughout: power_spectrum casts to float32 anyway, and
        # subtracting the mean in float64 first shifts the result by ~4e-6
        # relative -- enough to trip the figures' cached-vs-recomputed guards.
        Ytot = Y[-1] if Y is not None else None
        Ttot = T[-1] if T is not None else None
        rkk, ryy, rky, rtt, rkt = [], [], [], [], []
        rpk, rmn, rpdf = [], [], []
        for zi in range(N_PLANES):
            k = K[zi]
            ell, a = power_spectrum(k, fov_deg=FOV_DEG)
            rkk.append(a)
            if Ytot is not None:
                ryy.append(power_spectrum(Y[zi], fov_deg=FOV_DEG)[1])
                rky.append(power_spectrum(k, Ytot, fov_deg=FOV_DEG, auto1=a)[1])
            if Ttot is not None:
                rtt.append(power_spectrum(T[zi], fov_deg=FOV_DEG)[1])
                rkt.append(power_spectrum(k, Ttot, fov_deg=FOV_DEG, auto1=a)[1])
            o = peak_counts(k[None, None], fov_deg=FOV_DEG, smoothing_arcmin=SMOOTH_PK,
                            nu_bins=edges, nu_norm="map", return_realizations=True)
            rpk.append(o["peak_counts_real"][0, 0])
            rmn.append(o["minima_counts_real"][0, 0])
            sm = k.astype(np.float64) - k.mean()      # PDF only: accumulate in f64
            rpdf.append(np.histogram(sm / (sm.std() + 1e-30), bins=edges, density=True)[0])
        idx.append(i)
        kk.append(rkk)
        if ryy:
            yy.append(ryy)
            ky.append(rky)
        pk.append(rpk)
        mn.append(rmn)
        pdf.append(rpdf)
        if Ttot is not None:
            tt.append(rtt)
            kt.append(rkt)
        if rank == 0 and len(idx) % 5 == 0:
            print(f"[rank0] {len(idx)}/{len(mine)} reals", flush=True)

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(idx=np.array(idx), kk=np.array(kk), pk=np.array(pk),
               min=np.array(mn), pdf=np.array(pdf), ell=ell, pdf_bins=NU_CANON)
    if yy:
        out.update(yy=np.array(yy), ky=np.array(ky))
    if tt:
        out.update(tt=np.array(tt), kt=np.array(kt))
    np.savez(SHARD_DIR / f"{side}_rank{rank:03d}.npz", **out)
    comm.Barrier()
    if rank == 0:
        print(f"[{side}] {nr} realizations across {size} ranks -> {SHARD_DIR}")


if __name__ == "__main__":
    main()
