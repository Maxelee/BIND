"""nu statistics for the n1000 fig-5 arms, on the CANONICAL axis (MPI).

Supersedes ``n1000_nu05_stream.py``, which used the older "nu05" convention:

    old : centres -2.75 .. 7.75 (edges -3.0 .. 8.0), nu normalised per map
    new : centres -2.5  .. 8.0  (edges -2.75 .. 8.25), nu = kappa / sigma_fid

i.e. the axis is shifted by +0.25 and the normalisation is now ONE fixed sigma
taken from the fiducial run (bind.cli.nu_sigma0), so a parameter variation
cannot absorb its own sigma_kappa response.  Both changes match the convention
the campaign's peak_counts.npz / nongaussian_stats.npz already use, so every
nu-binned product in the tree finally lives on the same axis.

Per realization, per selected plane, on the 22 canonical nu values:
    pdf     UNSMOOTHED kappa / sigma_unsmoothed(fid), density on the bin edges
    peaks   2' smoothing, sigma_fid at 2'
    minima  same
    V0/V1/V2  1' smoothing (nongaussian's first scale), sigma_fid at 1'

    srun -n 48 python3 -u n1000_nu05_canon.py --arm bind
    srun -n 48 python3 -u n1000_nu05_canon.py --arm truth
    python3 n1000_nu05_canon.py --merge
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import NG_SCALES_DEFAULT, NU_CANON, nongaussian_stats, nu_edges, peak_counts

N1K = Path("/mnt/home/mlee1/ceph/bind_n1000")
HERE = Path(__file__).resolve().parent
DEST = HERE / "n1000_nu05_canon.npz"
SHARDS = N1K / "field_cache/nu_canon_shards"
SIG0 = N1K / "analysis/nu_sigma0_bind.npz"
NU, EDGES = NU_CANON, nu_edges(NU_CANON)
FOV, SMOOTH_PK = 5.0, 2.0
REPLICA = os.environ.get("BIND_FID_REPLICA", "tb49")
_RUN = {"tb18": "run_0018", "tb49": "run_0049", "tb53": "run_0053"}[REPLICA]
ARMS = {"bind": (f"twobound/{_RUN}", [0, 1, 2, 3, 4]),
        "truth": ("truth/run_0000", [1])}
KEYS = ("pdf", "peaks", "minima", "v0", "v1", "v2")


def stream(npz: Path, key: str, idx: set[int]):
    with zipfile.ZipFile(npz) as z, z.open(f"{key}.npy") as f:
        v = fmt.read_magic(f)
        shape, _, dt = (fmt.read_array_header_1_0(f) if v == (1, 0)
                        else fmt.read_array_header_2_0(f))
        per = int(np.prod(shape[1:])) * dt.itemsize
        for i in range(shape[0]):
            buf = f.read(per)
            if len(buf) < per:
                return
            if i in idx:
                yield i, np.frombuffer(buf, dtype=dt).reshape(shape[1:])


def sigmas(planes):
    """(peaks sigma, MF sigma, unsmoothed sigma) for the selected planes."""
    t = np.load(SIG0)
    sc = list(np.asarray(t["scales_arcmin"], float))
    i_pk = next(i for i, s in enumerate(sc) if abs(s - SMOOTH_PK) < 1e-9)
    i_mf = next(i for i, s in enumerate(sc) if abs(s - float(NG_SCALES_DEFAULT[0])) < 1e-9)
    S = np.asarray(t["sigma_smoothed"])
    return S[i_pk][planes], S[i_mf][planes], np.asarray(t["sigma_unsmoothed"])[planes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--n_real", type=int, default=None)
    a = ap.parse_args()

    if a.merge:
        out = {"nu": NU, "replica": REPLICA}
        for arm in ARMS:
            sh = sorted(SHARDS.glob(f"{arm}_rank*.npz"))
            if not sh:
                raise SystemExit(f"no shards for arm={arm}")
            parts = [np.load(s) for s in sh]
            idx = np.concatenate([p["idx"] for p in parts])
            order = np.argsort(idx)
            for k in KEYS:
                arr = np.concatenate([p[k] for p in parts])[order]
                out[f"{arm}_{k}"] = arr
            out[f"n_done_{arm}"] = len(idx)
            print(f"[merge] {arm}: {len(idx)} reals, {KEYS[0]} shape {out[f'{arm}_pdf'].shape}")
        np.savez_compressed(DEST, **out)
        print(f"[merge] wrote {DEST}")
        return

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    rel, planes = ARMS[a.arm]
    src = N1K / rel / "kappa_maps.npz"
    with zipfile.ZipFile(src) as z, z.open("kappa.npy") as f:
        v = fmt.read_magic(f)
        shape, _, _ = (fmt.read_array_header_1_0(f) if v == (1, 0)
                       else fmt.read_array_header_2_0(f))
    nr = shape[0] if a.n_real is None else min(a.n_real, shape[0])
    mine = set(range(rank, nr, size))
    s_pk, s_mf, s_u = sigmas(planes)
    acc = {k: [] for k in KEYS}
    idx = []

    for i, cube in stream(src, "kappa", mine):
        P = cube[planes]                                  # (n_pl, N, N)
        pc = peak_counts(P[None], fov_deg=FOV, smoothing_arcmin=SMOOTH_PK,
                         nu_bins=EDGES, nu_norm="fixed", nu_sigma0=s_pk,
                         return_realizations=True)
        ng = nongaussian_stats(P[None], fov_deg=FOV, nu_centers=NU,
                               nu_sigma0=s_mf, nu_sigma0_unsmoothed=s_u,
                               return_realizations=True)
        pdf = np.stack([np.histogram((m.astype(np.float64) - m.mean()) / s, bins=EDGES,
                                     density=True)[0] for m, s in zip(P, s_u)])
        vals = {"pdf": pdf,
                "peaks": np.asarray(pc["peak_counts_real"])[0],
                "minima": np.asarray(pc["minima_counts_real"])[0],
                "v0": np.asarray(ng["V0_real"])[0],
                "v1": np.asarray(ng["V1_real"])[0],
                "v2": np.asarray(ng["V2_real"])[0]}
        for k in KEYS:
            acc[k].append(vals[k] if len(planes) > 1 else vals[k][0])
        idx.append(i)
        if rank == 0 and len(idx) % 10 == 0:
            print(f"[rank0 {a.arm}] {len(idx)}/{len(mine)}", flush=True)

    SHARDS.mkdir(parents=True, exist_ok=True)
    np.savez(SHARDS / f"{a.arm}_rank{rank:03d}.npz", idx=np.array(idx),
             **{k: np.asarray(acc[k]) for k in KEYS})
    comm.Barrier()
    if rank == 0:
        print(f"[{a.arm}] {nr} reals on {size} ranks (replica={REPLICA}) -> {SHARDS}")


if __name__ == "__main__":
    main()
