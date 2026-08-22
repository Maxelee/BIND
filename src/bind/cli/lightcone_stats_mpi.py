"""``bind.cli.lightcone_stats_mpi``: MPI-parallel summary statistics.

Same products as :mod:`bind.cli.lightcone_stats`, but the realization loop —
which is embarrassingly parallel, every statistic being a mean over
realizations — is split across MPI ranks.  A 1000-realization run drops from
~3 h on one core to a few minutes on a node.

Design
------
* **Read once, share locally.**  ``kappa/y/tau_maps.npz`` are DEFLATE-compressed,
  so a zip member has no random access (seeking read-and-discards).  Ranks 0/1/2
  therefore decompress one field each, in parallel, into an *uncompressed* .npy
  in node-local scratch; every rank then ``np.load(mmap_mode="r")`` that file and
  reads only the realizations it owns.  One decompress pass, not one per rank.
* **Rank r owns realizations r::size**, processes them ONE at a time (bounded
  memory: ~60 MB of maps resident per rank) and accumulates running sums.
* **Exact combination.**  Each rank accumulates ``sum(x)`` and ``sum(x^2)``;
  rank 0 reduces both with ``MPI.SUM`` and forms
  ``mean = Sx/N`` and ``err = sqrt(Sxx/N - mean^2)/sqrt(N)`` — algebraically the
  same ``mean`` / ``std/sqrt(N)`` the serial code produces (numpy's ``std``
  is population, ddof=0), so outputs match to floating-point noise.
* **peak_cross (R(nu)) is NOT computed** — its outputs are ratios of peak-stacked
  sums that cannot be averaged across ranks, and it is unused downstream.  Use
  the serial ``bind.cli.lightcone_stats`` if it is ever needed.

    srun -n 64 python -m bind.cli.lightcone_stats_mpi --run_dir <run> \
         --nu_sigma0_from <table.npz> --nu_grid canon --nu_norm fixed
"""

from __future__ import annotations

import argparse
import shutil
import socket
import time
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference import stats as S

FIELDS = ("kappa", "y", "tau")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--smoothing_arcmin", type=float, nargs="+", default=[2.0])
    p.add_argument("--nu_grid", choices=["canon", "legacy"], default="canon")
    p.add_argument("--nu_sigma0_from", type=Path, default=None)
    p.add_argument("--nu_norm", choices=["map", "noise", "fixed"], default=None)
    p.add_argument("--shape_noise_ngal", type=float, default=None)
    p.add_argument("--sigma_e", type=float, default=0.26)
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--out_suffix", default="")
    p.add_argument("--no_dm", action="store_true")
    p.add_argument("--n_real", type=int, default=None, help="cap (debug)")
    p.add_argument("--scratch", type=Path, default=Path("/tmp"),
                   help="node-local dir for the decompressed cubes")
    p.add_argument("--keep_scratch", action="store_true")
    return p.parse_args()


def decompress(npz: Path, key: str, dest: Path, n_max: int | None = None) -> tuple[int, ...]:
    """Stream one npz member out to an uncompressed .npy (memmap-able).

    ``n_max`` stages only the leading realizations — a capped debug run should
    not spend minutes and tens of GB decompressing the full cube.
    """
    with zipfile.ZipFile(npz) as z, z.open(f"{key}.npy") as f:
        v = fmt.read_magic(f)
        shape, fortran, dt = (fmt.read_array_header_1_0(f) if v == (1, 0)
                              else fmt.read_array_header_2_0(f))
        assert not fortran
        if n_max is not None:
            shape = (min(int(n_max), shape[0]),) + tuple(shape[1:])
        out = np.lib.format.open_memmap(dest, mode="w+", dtype=dt, shape=shape)
        per = int(np.prod(shape[1:])) * dt.itemsize
        for i in range(shape[0]):
            buf = f.read(per)
            if len(buf) < per:
                break
            out[i] = np.frombuffer(buf, dtype=dt).reshape(shape[1:])
        out.flush()
        del out
    return shape


class Acc:
    """Running sum / sum-of-squares for the statistics of one rank."""

    def __init__(self):
        self.s, self.ss, self.n = {}, {}, 0

    def add(self, key, val, sq=True):
        v = np.asarray(val, dtype=np.float64)
        if key not in self.s:
            self.s[key] = np.zeros_like(v)
            if sq:
                self.ss[key] = np.zeros_like(v)
        self.s[key] += v
        if sq and key in self.ss:
            self.ss[key] += v * v


def main() -> None:  # noqa: PLR0912, PLR0915
    args = parse_args()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    rd = args.run_dir
    t0 = time.time()

    nu_norm = args.nu_norm or ("fixed" if args.nu_sigma0_from
                               else ("noise" if args.shape_noise_ngal else "map"))
    nu_centers = S.NU_CANON if args.nu_grid == "canon" else None
    nu_bins = S.NU_EDGES_CANON if nu_centers is not None else None

    sig0 = sig0_u = sig0_ng = None
    nu_fiducial = "per-map"
    if nu_norm == "fixed":
        t = np.load(args.nu_sigma0_from)
        tab = list(np.asarray(t["scales_arcmin"], float))
        idx = [next(i for i, x in enumerate(tab) if abs(x - sc) < 1e-9)
               for sc in args.smoothing_arcmin]
        sig0 = np.asarray(t["sigma_smoothed"])[idx]
        sig0_u = np.asarray(t["sigma_unsmoothed"])
        ng_i = next(i for i, x in enumerate(tab)
                    if abs(x - float(S.NG_SCALES_DEFAULT[0])) < 1e-9)
        sig0_ng = np.asarray(t["sigma_smoothed"])[[ng_i]]
        nu_fiducial = str(t["source_run"])

    # ── stage the cubes into node-local scratch ───────────────────────────
    # EVERY node needs its own copy: --scratch is node-local (/tmp), so ranks on
    # node B cannot see what node A decompressed.  Split off a per-node
    # communicator and let each node's local ranks 0/1/2 stage one field each,
    # in parallel; nodes stage concurrently, so the cost does not grow with the
    # node count.  The hostname in the path also keeps this safe if --scratch is
    # pointed at a SHARED filesystem (nodes then never clobber each other).
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    scratch = args.scratch / f"n1000stats_{rd.name}_{socket.gethostname()}"
    if node_comm.rank == 0:
        scratch.mkdir(parents=True, exist_ok=True)
    node_comm.Barrier()
    have = {}
    for k, field in enumerate(FIELDS):
        src = rd / f"{field}_maps.npz"
        dest = scratch / f"{field}.npy"
        ok = src.exists()
        if ok and node_comm.rank == (k % node_comm.size):
            decompress(src, field, dest, args.n_real)
        have[field] = ok
    node_comm.Barrier()
    comm.Barrier()
    if rank == 0:
        print(f"[mpi-stats] staged cubes in {time.time()-t0:.0f}s "
              f"({', '.join(f for f in FIELDS if have[f])})", flush=True)

    cubes = {f: np.load(scratch / f"{f}.npy", mmap_mode="r") for f in FIELDS if have[f]}
    n_real = cubes["kappa"].shape[0] if args.n_real is None else min(args.n_real, cubes["kappa"].shape[0])
    fov = float(np.load(rd / "kappa_maps.npz")["fov_deg"])
    mine = list(range(rank, n_real, size))
    acc = Acc()
    noise_kw = dict(shape_noise_ngal=args.shape_noise_ngal, sigma_e=args.sigma_e,
                    noise_seed=args.noise_seed, nu_norm=nu_norm)
    sm = args.smoothing_arcmin[0] if len(args.smoothing_arcmin) == 1 else args.smoothing_arcmin

    # ── shared DM histogram axis ──────────────────────────────────────────
    # dm_stats takes its range from the 0.1/99.9 percentiles of the cube it is
    # handed: serial sees all N realizations, a rank sees ONE at a time, so a
    # per-call axis would differ realization-to-realization and the averaged PDF
    # would mix axes.  Pool a fine histogram over every rank's realizations and
    # hand the same edges to every call — the serial percentiles to within one
    # fine bin.  Two extra streaming passes over the (node-local) tau cube.
    dm_edges = dm_bins = dm_tau_per_dm = None
    if have["tau"] and not args.no_dm:
        from bind.inference.lightcone_maps import TAU_PER_DM
        dm_tau_per_dm = float(TAU_PER_DM)
        lo_l, hi_l = np.inf, -np.inf
        for r in mine:
            d = np.asarray(cubes["tau"][r], dtype=np.float64) / TAU_PER_DM
            lo_l, hi_l = min(lo_l, float(d.min())), max(hi_l, float(d.max()))
        lo_g = comm.allreduce(lo_l, op=MPI.MIN)
        hi_g = comm.allreduce(hi_l, op=MPI.MAX)
        n_fine = 1 << 20          # axis resolved to ~1e-5 of the DM range
        fine = np.zeros(n_fine)
        for r in mine:
            d = np.asarray(cubes["tau"][r], dtype=np.float64) / TAU_PER_DM
            fine += np.histogram(d, bins=n_fine, range=(lo_g, hi_g))[0]
        tot_fine = np.zeros(n_fine)
        comm.Allreduce(fine, tot_fine, op=MPI.SUM)
        cdf = np.cumsum(tot_fine) / tot_fine.sum()
        fine_edges = np.linspace(lo_g, hi_g, n_fine + 1)
        lo, hi = np.interp([0.001, 0.999], cdf, fine_edges[1:])
        dm_edges = np.linspace(float(lo), float(hi), 42)
        dm_bins = 0.5 * (dm_edges[1:] + dm_edges[:-1])
        if rank == 0:
            print(f"[mpi-stats] DM axis {lo:.2f}-{hi:.2f} pc/cm^3 pooled over "
                  f"{n_real} reals ({time.time()-t0:.0f}s)", flush=True)

    for r in mine:
        kap = np.asarray(cubes["kappa"][r])[None]                # (1, n_src, N, N)
        ck = S.cl_kappa(kap, fov_deg=fov)
        acc.add("cl", ck["cl"])
        ell = ck["ell"]
        if have["y"]:
            y = np.asarray(cubes["y"][r])[None]
            cky = S.cl_kappa_y(kap, y[:, -1], fov_deg=fov)
            acc.add("cl_ky", cky["cl_ky"])
            acc.add("cl_yy", cky["cl_yy"])
        if have["tau"]:
            tau = np.asarray(cubes["tau"][r])[None]
            yt = y[:, -1] if have["y"] else None
            ckt = S.cl_kappa_tau(kap, tau[:, -1], y_maps=yt, fov_deg=fov)
            acc.add("cl_kt", ckt["cl_kt"])
            acc.add("cl_tt", ckt["cl_tt"])
            if "cl_yt" in ckt:
                acc.add("cl_yt", ckt["cl_yt"])
            if not args.no_dm:
                dm = S.dm_stats(tau, fov_deg=fov, dm_edges=dm_edges)
                for key in ("dm_pdf", "dm_mean", "sigma_dm", "F", "skewness", "kurtosis"):
                    acc.add(f"dm_{key}", dm[key], sq=False)
        pk = S.peak_counts(kap, fov_deg=fov, smoothing_arcmin=sm, nu_bins=nu_bins,
                           nu_sigma0=(sig0 if len(args.smoothing_arcmin) > 1
                                      else (sig0[0] if sig0 is not None else None)),
                           **noise_kw)
        acc.add("peaks", pk["peak_counts"])
        acc.add("minima", pk["minima_counts"])
        ng = S.nongaussian_stats(kap, fov_deg=fov, nu_centers=nu_centers,
                                 nu_sigma0=sig0_ng, nu_sigma0_unsmoothed=sig0_u)
        for key in ("pdf", "variance", "skewness", "kurtosis", "V0", "V1", "V2"):
            acc.add(f"ng_{key}", ng[key], sq=False)
        acc.n += 1
        if rank == 0 and acc.n % 5 == 0:
            print(f"[mpi-stats] rank0 {acc.n}/{len(mine)} reals "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # ── reduce ────────────────────────────────────────────────────────────
    # rank 0 always owns realization 0, so it defines the key/shape schema; a rank
    # with FEWER realizations than others (or none at all, when n_real < size)
    # must still contribute correctly shaped zeros to every Reduce.
    schema = comm.bcast({k: (v.shape, v.dtype.str, k in acc.ss)
                         for k, v in acc.s.items()} if rank == 0 else None, root=0)
    tot_s, tot_ss = {}, {}
    for k, (shp, dts, has_sq) in schema.items():
        loc = acc.s.get(k)
        if loc is None:
            loc = np.zeros(shp, dtype=dts)
        g = np.zeros(shp, dtype=dts) if rank == 0 else None
        comm.Reduce(loc, g, op=MPI.SUM, root=0)
        tot_s[k] = g
        if has_sq:
            loc2 = acc.ss.get(k)
            if loc2 is None:
                loc2 = np.zeros(shp, dtype=dts)
            g2 = np.zeros(shp, dtype=dts) if rank == 0 else None
            comm.Reduce(loc2, g2, op=MPI.SUM, root=0)
            tot_ss[k] = g2
    N = comm.allreduce(acc.n, op=MPI.SUM)
    if rank != 0:
        # every node's local rank 0 removes that node's scratch copy
        if node_comm.rank == 0 and not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        return

    def mean(k):
        return tot_s[k] / N

    def err(k):
        m = mean(k)
        var = np.maximum(tot_ss[k] / N - m * m, 0.0)
        return np.sqrt(var) / np.sqrt(N)

    prov = dict(nu_grid=str(args.nu_grid), nu_norm_used=str(nu_norm),
                nu_fiducial=str(nu_fiducial),
                nu_sigma0_table=(np.asarray(sig0) if sig0 is not None else np.zeros(0)),
                nu_sigma0_unsmoothed_table=(np.asarray(sig0_u) if sig0_u is not None
                                            else np.zeros(0)))
    sfx = args.out_suffix
    np.savez(rd / "Cl_kappa.npz", ell=ell, cl=mean("cl"), cl_err=err("cl"))
    print(f"[mpi-stats] Cl_kappa.npz cl{mean('cl').shape} (N={N})")
    if have["y"]:
        np.savez(rd / "Cl_kappa_y.npz", ell=ell, cl_ky=mean("cl_ky"), cl_ky_err=err("cl_ky"),
                 cl_yy=mean("cl_yy"), cl_yy_err=err("cl_yy"))
    if have["tau"]:
        kw = dict(ell=ell, cl_kt=mean("cl_kt"), cl_kt_err=err("cl_kt"),
                  cl_tt=mean("cl_tt"), cl_tt_err=err("cl_tt"))
        if "cl_yt" in tot_s:
            kw.update(cl_yt=mean("cl_yt"), cl_yt_err=err("cl_yt"))
        np.savez(rd / "Cl_tau.npz", **kw)
        if not args.no_dm:
            np.savez(rd / "dm_stats.npz", dm_bins=dm_bins, dm_pdf=mean("dm_dm_pdf"),
                     dm_mean=mean("dm_dm_mean"), sigma_dm=mean("dm_sigma_dm"),
                     F=mean("dm_F"), skewness=mean("dm_skewness"),
                     kurtosis=mean("dm_kurtosis"), TAU_PER_DM=dm_tau_per_dm)
    np.savez(rd / f"peak_counts{sfx}.npz", nu=pk["nu"], peak_counts=mean("peaks"),
             peak_counts_err=err("peaks"), minima_counts=mean("minima"),
             minima_counts_err=err("minima"), smoothing_arcmin=sm,
             shape_noise_ngal=args.shape_noise_ngal or 0.0, sigma_e=args.sigma_e,
             nu_norm=nu_norm, **prov)
    np.savez(rd / "nongaussian_stats.npz", pdf=mean("ng_pdf"), pdf_bins=ng["pdf_bins"],
             smoothing_scales_arcmin=ng["smoothing_scales_arcmin"],
             variance=mean("ng_variance"), skewness=mean("ng_skewness"),
             kurtosis=mean("ng_kurtosis"), mf_nu=ng["mf_nu"], V0=mean("ng_V0"),
             V1=mean("ng_V1"), V2=mean("ng_V2"), **prov)
    print(f"[mpi-stats] nongaussian_stats.npz pdf{mean('ng_pdf').shape}")
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)   # this node's copy; others clean their own
    print(f"[mpi-stats] DONE {rd.name}: {N} reals on {size} ranks in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
