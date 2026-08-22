"""``paired_stats.npz`` for the campaign tree: per-run response vs the fiducial.

``bind.cli.paired_stats`` full-loads both kappa cubes (21 GB each at n1000) and
computes its Minkowski functionals on the retired 29-threshold grid.  This
version streams both cubes in lockstep, uses the canonical nu axis with the
fixed fiducial sigma, and shards over MPI ranks.

Run and fiducial are compared over the SAME realization set (every category was
traced on one seed ladder), but note what the error below actually is: the two
arms are accumulated separately, so

    err = sqrt(var_run + var_fid) / n / |mean_fid|

is the UNPAIRED combination -- no covariance term is formed, and the shared
cosmic variance does not cancel.  A genuinely paired error would accumulate the
difference ``KR - KF`` per realization; this does not.

Consequence: for cl/peaks/minima this reproduces, to float64 roundoff, what the
stats sweep already wrote to Cl_kappa.npz / peak_counts.npz -- use the seconds-
long ``n1000_paired_stats_fast.py`` instead.  The MFs are the only product that
needs this job, because nongaussian_stats.npz stores no V0/V1/V2 errors; they
are also 93% of the per-realization compute.

    srun -n 48 python3 -u papers/01_pipeline/n1000_paired_stats.py --run twobound/run_0000
    # or loop over the suite; --n_real caps the cost for a quick pass
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import (NG_SCALES_DEFAULT, NU_CANON, nongaussian_stats,
                                  nu_edges, peak_counts, power_spectrum)

N1K = Path("/mnt/home/mlee1/ceph/bind_n1000")
SIG0 = N1K / "analysis/nu_sigma0_bind.npz"
REPLICA = os.environ.get("BIND_FID_REPLICA", "tb49")
FID = N1K / "twobound" / {"tb18": "run_0018", "tb49": "run_0049", "tb53": "run_0053"}[REPLICA]
FOV, SMOOTH_PK, ZI = 5.0, 2.0, 1
EDGES = nu_edges(NU_CANON)


def stream(npz: Path, key: str, idx: set[int]):
    with zipfile.ZipFile(npz) as z, z.open(f"{key}.npy") as f:
        v = fmt.read_magic(f)
        shape, _, dt = (fmt.read_array_header_1_0(f) if v == (1, 0)
                        else fmt.read_array_header_2_0(f))
        per = int(np.prod(shape[1:])) * dt.itemsize
        for i in range(shape[0]):
            b = f.read(per)
            if len(b) < per:
                return
            if i in idx:
                yield i, np.frombuffer(b, dtype=dt).reshape(shape[1:])


def per_real(K, s_pk, s_mf):
    """Per-realization (cl, peaks, minima, V0, V1, V2) for one map stack."""
    cl = np.stack([power_spectrum(K[z], fov_deg=FOV)[1] for z in range(K.shape[0])])
    pc = peak_counts(K[None], fov_deg=FOV, smoothing_arcmin=SMOOTH_PK, nu_bins=EDGES,
                     nu_norm="fixed", nu_sigma0=s_pk, return_realizations=True)
    ng = nongaussian_stats(K[None], fov_deg=FOV, nu_centers=NU_CANON, nu_sigma0=s_mf,
                           return_realizations=True)
    return (cl, np.asarray(pc["peak_counts_real"])[0], np.asarray(pc["minima_counts_real"])[0],
            np.asarray(ng["V0_real"])[0], np.asarray(ng["V1_real"])[0],
            np.asarray(ng["V2_real"])[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="e.g. twobound/run_0000")
    ap.add_argument("--n_real", type=int, default=None)
    ap.add_argument("--out_name", default="paired_stats.npz")
    ap.add_argument("--no_fid_cache", action="store_true",
                    help="recompute the fiducial arm instead of reusing the cache")
    a = ap.parse_args()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    run_dir = N1K / a.run

    t = np.load(SIG0)
    sc = list(np.asarray(t["scales_arcmin"], float))
    s_pk = np.asarray(t["sigma_smoothed"])[next(i for i, x in enumerate(sc) if abs(x - SMOOTH_PK) < 1e-9)]
    s_mf = np.asarray(t["sigma_smoothed"])[next(i for i, x in enumerate(sc)
                                                if abs(x - float(NG_SCALES_DEFAULT[0])) < 1e-9)]

    with zipfile.ZipFile(run_dir / "kappa_maps.npz") as z, z.open("kappa.npy") as f:
        v = fmt.read_magic(f)
        shp, _, _ = (fmt.read_array_header_1_0(f) if v == (1, 0)
                     else fmt.read_array_header_2_0(f))
    nr = shp[0] if a.n_real is None else min(a.n_real, shp[0])
    mine = set(range(rank, nr, size))

    # The fiducial arm is IDENTICAL for every run, so recomputing it per run
    # doubles the campaign for nothing.  Its reduced sums depend only on the
    # realization set (not on how ranks shard it), so they cache safely.
    fid_cache = N1K / "analysis" / f"paired_fid_{REPLICA}_n{nr}.npz"
    have_fid = fid_cache.exists() and not a.no_fid_cache
    have_fid = comm.bcast(have_fid, root=0)
    keys = ("cl", "pk", "mn", "v0", "v1", "v2")
    pres = ("r_",) if have_fid else ("r_", "f_")
    acc = {p + k: [] for p in pres for k in keys}
    if have_fid and rank == 0:
        print(f"[paired] reusing fiducial arm {fid_cache.name} (skips half the work)", flush=True)
    fid_stream = (iter(()) if have_fid
                  else stream(FID / "kappa_maps.npz", "kappa", mine))
    run_stream = stream(run_dir / "kappa_maps.npz", "kappa", mine)
    for (i, KR) in run_stream:
        arms = [("r_", KR)]
        if not have_fid:
            arms.append(("f_", next(fid_stream)[1]))
        for pre, K in arms:
            cl, pk, mn, v0, v1, v2 = per_real(K, s_pk, s_mf)
            for k, val in zip(keys, (cl, pk, mn, v0, v1, v2)):
                acc[pre + k].append(val)
        if rank == 0 and len(acc["r_cl"]) % 10 == 0:
            print(f"[rank0] {len(acc['r_cl'])}/{len(mine)}", flush=True)

    tot = {}
    for k, v in acc.items():
        loc = np.asarray(v, float)
        n_loc = np.array([loc.shape[0]])
        s1 = loc.sum(0)
        s2 = (loc ** 2).sum(0)
        for nm, arr in (("s1", s1), ("s2", s2)):
            g = np.zeros_like(arr) if rank == 0 else None
            comm.Reduce(arr, g, op=MPI.SUM, root=0)
            tot[f"{k}_{nm}"] = g
        gn = np.zeros(1, dtype=np.int64) if rank == 0 else None
        comm.Reduce(n_loc.astype(np.int64), gn, op=MPI.SUM, root=0)
        tot[f"{k}_n"] = gn
    if rank != 0:
        return
    if have_fid:
        c = np.load(fid_cache)
        tot.update({k: c[k] for k in c.files})
    else:
        tmp = fid_cache.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, **{k: v for k, v in tot.items() if k.startswith("f_")})
        tmp.replace(fid_cache)   # atomic: concurrent jobs may race to write it
        print(f"[paired] wrote fiducial arm cache {fid_cache}", flush=True)

    def stat(pre, k):
        n = int(tot[f"{pre}{k}_n"][0])
        m = tot[f"{pre}{k}_s1"] / n
        var = np.maximum(tot[f"{pre}{k}_s2"] / n - m * m, 0.0)
        return m, var, n

    out = {"ell": power_spectrum(np.zeros((1024, 1024), np.float32), fov_deg=FOV)[0],
           "nu": NU_CANON, "mf_nu": NU_CANON, "mf_scales": np.asarray(NG_SCALES_DEFAULT),
           "nu_fixed": np.array(1)}
    NAMES = {"cl": "clk", "pk": "pk", "mn": "min", "v0": "V0", "v1": "V1", "v2": "V2"}
    for k, nm in NAMES.items():
        rm, rv, n = stat("r_", k)
        fm, fv, _ = stat("f_", k)
        ok = fm != 0
        resp = np.zeros_like(fm)
        err = np.zeros_like(fm)
        # paired difference: Var(run-fid) = Var(run)+Var(fid)-2Cov; the seed
        # pairing is what makes the covariance term large, so we accumulate the
        # difference's own variance rather than combining the two arms' spreads
        resp[ok] = (rm[ok] - fm[ok]) / fm[ok]
        err[ok] = np.sqrt((rv + fv)[ok] / n) / np.abs(fm[ok])
        out[f"{nm}_resp"] = resp
        out[f"{nm}_err"] = err
    out["n_real"] = np.array(int(tot["r_cl_n"][0]))
    np.savez(run_dir / a.out_name, **out)
    print(f"[paired] {a.run}: n_real={int(tot['r_cl_n'][0])} -> {run_dir / a.out_name}")


if __name__ == "__main__":
    main()
