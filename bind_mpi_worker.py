#!/usr/bin/env python
"""
bind_mpi_worker.py
------------------
Halo aperture reduction for the BIND sb35 Sobol array, redesigned to be
**preempt-safe** (Flatiron rusty `preempt` partition).

Each (run, snap) key is reduced independently and written to its own shard
(``analysis_cache/shards/run_RRRR_snap_SSS.parquet``, or a ``.empty`` marker for
zero-halo keys).  A key counts as done iff its shard exists, so a preempted job
simply resumes -- no work is gathered through rank 0 and nothing is lost on a
kill.  This replaces the old single-final-write design.

Modes
-----
  migrate : seed shards from an existing ``integrated.pkl`` WITHOUT re-reading any
            patch npz (one-time; lets us reuse the 2971 keys already reduced).
  build   : reduce every complete-on-disk key that has no shard yet -> shards.
            MPI-aware (distributes keys round-robin across ranks); falls back to
            serial / multiprocessing when not launched under srun.
  merge   : concatenate all shards, join the Sobol astro params, and write
            ``integrated.parquet`` + ``integrated.pkl`` + ``integrated_keys.pkl``.

Typical preempt workflow:
    python bind_mpi_worker.py migrate          # cheap, no ceph patch reads
    srun  python bind_mpi_worker.py build      # the heavy, restartable step
    python bind_mpi_worker.py merge            # assemble the final table

Env overrides: BIND_BUNDLE, BIND_COND_DIR.
"""

import os, sys, json, glob, pickle, shutil, warnings, zipfile, argparse
from pathlib import Path

import numpy as np
import pandas as pd

import bind  # noqa: F401  (ensures the package imports cleanly on the worker)
from bind.params import PARAM_NAMES, PARAM_LOG_FLAG
from bind.inference.design import ASTRO_PARAM_INDICES

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BUNDLE   = Path(os.environ.get("BIND_BUNDLE", "/mnt/home/mlee1/ceph/bind_sb35"))
RUNS_DIR = BUNDLE / "runs"
CACHE    = BUNDLE / "analysis_cache"
SHARDS   = CACHE / "shards"

# The per-snapshot metadata (pixel_size, z, a, n_slabs) lives in the SHARED
# fiducial stage-1 conditions, NOT in the bundle.  On the original system this
# was symlinked in as BUNDLE/conditions; here it is the fiducial lightcone dir.
# Both layouts are probed; failing that we fall back to a constant (the values
# are uniform across all 20 snaps anyway: pixel_size = 50/1024, n_slabs = 4).
COND_CANDIDATES = [
    Path(os.environ["BIND_COND_DIR"]) if os.environ.get("BIND_COND_DIR") else None,
    Path("/mnt/home/mlee1/ceph/bind_lightcone_tng"),   # fiducial: snap_*/stage1/manifest
    BUNDLE / "conditions",                             # original layout: snap_*/manifest
]

ASTRO_IDX   = list(ASTRO_PARAM_INDICES)
ASTRO_NAMES = [PARAM_NAMES[i] for i in ASTRO_IDX]
SOBOL  = np.load(BUNDLE / "design" / "astro_params_sobol.npy")
N_RUNS = SOBOL.shape[0]

R500_OVER_R200 = 0.659
# Snapshot list (low-z first), matching run_sb35_generate.sh.
SNAP_LIST = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]


def _load_snap_info():
    """snap -> {z, a, pixel_size, n_slabs} from the fiducial stage-1 manifests."""
    for base in COND_CANDIDATES:
        if base is None or not base.exists():
            continue
        info = {}
        for pat in ("snap_*/stage1/stage1_manifest.json", "snap_*/stage1_manifest.json"):
            for mpath in sorted(base.glob(pat)):
                m = json.loads(Path(mpath).read_text())
                # snap dir is the first 'snap_<NNN>' ancestor of the manifest
                sd = next(p for p in Path(mpath).parents if p.name.startswith("snap_"))
                snap = int(sd.name.split("_")[1])
                info[snap] = {
                    "z": float(m["redshift"]),
                    "a": float(m["scale_factor"]),
                    "pixel_size": float(m["pixel_size"]),
                    "n_slabs": int(m["n_slabs"]),
                }
            if info:
                return info, str(base)
    # Constant backstop (uniform across all snaps; z/a filled at merge from cache).
    const = {s: {"z": np.nan, "a": np.nan, "pixel_size": 50.0 / 1024.0, "n_slabs": 4}
             for s in SNAP_LIST}
    return const, "<constant backstop>"


SNAP_INFO, COND_SRC = _load_snap_info()
SNAPS = sorted(SNAP_INFO)

_BAD = (zipfile.BadZipFile, OSError, EOFError, ValueError, KeyError)


def _safe_npz(path):
    try:
        d = np.load(path)
        int(d["n_halos"])  # force the central-directory read; truncated npz raises
        return d
    except _BAD:
        return None


_RR_CACHE = {}
def _radius_grid(P, pixel_size):
    key = (P, float(pixel_size))
    rr = _RR_CACHE.get(key)
    if rr is None:
        cen = P // 2
        yy, xx = np.mgrid[0:P, 0:P]
        rr = (np.hypot(xx - cen, yy - cen) * pixel_size).astype(np.float32)
        _RR_CACHE[key] = rr
    return rr


def _aperture_quantities(d, pixel_size):
    """Per-halo aperture integrals for one composite_slab npz (unchanged math)."""
    gen = d["generated_patches"]
    th  = d["thermo_patches"] if "thermo_patches" in d.files else None
    masses, r200 = d["halo_masses"], d["halo_r200"]
    n, _, P, _ = gen.shape
    rr = _radius_grid(P, pixel_size)
    pix_area = pixel_size ** 2
    dm, gas, star = gen[:, 0], gen[:, 1], gen[:, 2]
    tot = dm + gas + star
    s = lambda a, ap: (a * ap).sum((1, 2), dtype=np.float64)
    rows = {}
    for tag, rad in (("200", r200.astype(np.float32)),
                     ("500", np.float32(R500_OVER_R200) * r200.astype(np.float32))):
        ap = rr[None] <= rad[:, None, None]
        mt = s(tot, ap)
        mgas = s(gas, ap)
        mstar = s(star, ap)
        mdm = s(dm, ap)
        mt_s = np.where(mt > 0, mt, np.nan)
        rows[f"M_tot_{tag}"]  = mt
        rows[f"M_gas_{tag}"]  = mgas
        rows[f"M_star_{tag}"] = mstar
        rows[f"M_dm_{tag}"]   = mdm
        rows[f"f_gas_{tag}"]  = mgas / mt_s
        rows[f"f_star_{tag}"] = mstar / mt_s
        if th is not None:
            rows[f"Y_{tag}"] = s(th[:, 0], ap) * pix_area
            gw_s = np.where(mgas > 0, mgas, np.nan)
            rows[f"T_mw_{tag}"] = s(th[:, 1] * gas, ap) / gw_s
        else:
            rows[f"Y_{tag}"] = np.full(n, np.nan)
            rows[f"T_mw_{tag}"] = np.full(n, np.nan)
    rows["M200"] = masses.astype(np.float64)
    rows["r200"] = r200.astype(np.float64)
    return pd.DataFrame(rows), n


def _shard_path(run, snap):
    return SHARDS / f"run_{run:04d}_snap_{snap:03d}.parquet"


def _empty_path(run, snap):
    return SHARDS / f"run_{run:04d}_snap_{snap:03d}.empty"


def _has_shard(run, snap):
    return _shard_path(run, snap).exists() or _empty_path(run, snap).exists()


def _atomic_write_parquet(df, path):
    # This pyarrow build ships without snappy/zstd/gzip codecs, so write
    # uncompressed parquet (still columnar + fast; shards are small).
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False, compression=None)
    os.replace(tmp, path)


def _reduce_key(run, snap):
    """Reduce one (run, snap); return a DataFrame (no params) or None if zero-halo.

    Returns the string 'incomplete' if the key is not fully generated on disk."""
    info = SNAP_INFO.get(snap, {"pixel_size": 50.0 / 1024.0, "n_slabs": 4})
    snapdir = RUNS_DIR / f"run_{run:04d}" / f"snap_{snap:03d}"
    slabs = sorted(snapdir.glob("composite_slab*.npz"))
    if len(slabs) < info["n_slabs"]:
        return "incomplete"
    parts = []
    for sp in slabs:
        d = _safe_npz(sp)
        if d is None:
            return "incomplete"
        if int(d["n_halos"]) == 0:
            continue
        sub, n = _aperture_quantities(d, info["pixel_size"])
        sub.insert(0, "idx", np.arange(n))
        sub.insert(0, "slab", int(d["slab_idx"]))
        parts.append(sub)
    if not parts:
        return None
    t = pd.concat(parts, ignore_index=True)
    t["run"], t["snap"] = run, snap
    t["z"], t["a"] = info.get("z", np.nan), info.get("a", np.nan)
    t["halo_id"] = t.snap.astype(str) + "_" + t.slab.astype(str) + "_" + t.idx.astype(str)
    return t


def _write_key(run, snap, force=False):
    """Reduce + persist one key as a shard. Returns 'skip'|'ok'|'empty'|'incomplete'."""
    if not force and _has_shard(run, snap):
        return "skip"
    res = _reduce_key(run, snap)
    if isinstance(res, str):          # "incomplete"
        return res
    if res is None:                   # complete on disk but zero halos
        _empty_path(run, snap).touch()
        return "empty"
    _atomic_write_parquet(res, _shard_path(run, snap))
    return "ok"


# ── modes ─────────────────────────────────────────────────────────────────────
def mode_migrate():
    """Seed shards from an existing integrated.pkl (+ keys) without patch reads."""
    SHARDS.mkdir(parents=True, exist_ok=True)
    cache_df = CACHE / "integrated.pkl"
    if not cache_df.exists():
        print(f"[migrate] no {cache_df}; nothing to seed.", flush=True)
        return
    df = pickle.load(open(cache_df, "rb"))
    core = [c for c in df.columns if c not in ASTRO_NAMES]  # drop params; re-added at merge
    n = 0
    for (run, snap), g in df[core].groupby(["run", "snap"], sort=False):
        p = _shard_path(int(run), int(snap))
        if not p.exists():
            _atomic_write_parquet(g.reset_index(drop=True), p)
            n += 1
    # mark zero-halo keys that were 'done' but never produced rows
    keyf = CACHE / "integrated_keys.pkl"
    nz = 0
    if keyf.exists():
        for (run, snap) in pickle.load(open(keyf, "rb")):
            if not _has_shard(int(run), int(snap)):
                _empty_path(int(run), int(snap)).touch(); nz += 1
    print(f"[migrate] seeded {n} parquet shards + {nz} empty markers into {SHARDS}", flush=True)


def _todo_keys():
    """Keys with no shard yet (for status / quick checks; not used by the split)."""
    return [(r, s) for (r, s) in _all_keys() if not _has_shard(r, s)]


def _all_keys():
    """Full, deterministic (run, snap) grid — independent of which shards exist.

    Stable ordering is what makes the srun [rank::size] split race-free: every
    rank slices the *same* list and only skips keys whose shard already exists."""
    runs = sorted(int(p.name.split("_")[1]) for p in RUNS_DIR.glob("run_*"))
    return [(r, s) for r in runs for s in SNAPS]


def _rank_size():
    """(rank, size) for an SPMD launch.

    Prefer srun's env (SLURM_PROCID / SLURM_NTASKS) — robust and independent of
    any MPI library, which sidesteps mpi4py<->launcher mismatches on this system.
    Falls back to a 1-rank serial run (where --nproc can fan out locally)."""
    try:
        size = int(os.environ.get("SLURM_NTASKS", "1"))
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        if size > 1:
            return rank, size
    except ValueError:
        pass
    return 0, 1


def mode_build(rebuild=False, nproc=1, limit=None):
    SHARDS.mkdir(parents=True, exist_ok=True)
    rank, size = _rank_size()

    keys = _all_keys()
    if limit:
        keys = keys[:limit]
    my_keys = keys[rank::size]
    if rank == 0:
        print(f"[build] cond_src={COND_SRC}", flush=True)
        print(f"[build] {len(keys)} keys total across {size} rank(s); "
              f"rank 0 owns {len(my_keys)}", flush=True)

    counts = {"skip": 0, "ok": 0, "empty": 0, "incomplete": 0}
    if size == 1 and nproc > 1:
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            for st in pool.starmap(_write_key, [(r, s, rebuild) for (r, s) in my_keys]):
                counts[st] += 1
    else:
        for i, (run, snap) in enumerate(my_keys):
            counts[_write_key(run, snap, force=rebuild)] += 1
            if rank == 0 and (i + 1) % 25 == 0:
                print(f"[rank 0] {i+1}/{len(my_keys)} "
                      f"(ok={counts['ok']} skip={counts['skip']} "
                      f"empty={counts['empty']} inc={counts['incomplete']})", flush=True)

    print(f"[rank {rank}/{size}] DONE  ok={counts['ok']} skip={counts['skip']} "
          f"empty={counts['empty']} incomplete={counts['incomplete']}", flush=True)


def mode_merge():
    shard_files = sorted(SHARDS.glob("run_*_snap_*.parquet"))
    print(f"[merge] {len(shard_files)} non-empty shards "
          f"(+{len(list(SHARDS.glob('*.empty')))} empty keys)", flush=True)
    if not shard_files:
        print("[merge] nothing to merge; run build first.", flush=True)
        return
    df = pd.concat((pd.read_parquet(f) for f in shard_files), ignore_index=True)

    # join the Sobol astro params (one row per run) onto every halo record
    pmat = pd.DataFrame(SOBOL[:, ASTRO_IDX], columns=ASTRO_NAMES)
    pmat["run"] = np.arange(N_RUNS)
    df = df.drop(columns=[c for c in ASTRO_NAMES if c in df.columns], errors="ignore")
    df = df.merge(pmat, on="run", how="left")

    done = set(map(tuple, df[["run", "snap"]].drop_duplicates().itertuples(index=False)))
    for ef in SHARDS.glob("*.empty"):
        parts = ef.stem.split("_")
        done.add((int(parts[1]), int(parts[3])))

    df.to_parquet(CACHE / "integrated.parquet", index=False, compression=None)
    pickle.dump(df,   open(CACHE / "integrated.pkl", "wb"))
    pickle.dump(done, open(CACHE / "integrated_keys.pkl", "wb"))
    print(f"[merge] wrote {len(df):,} halo records | runs={df.run.nunique()} "
          f"snaps={df.snap.nunique()} | {len(done)} keys -> integrated.parquet/.pkl", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="build",
                   choices=["migrate", "build", "merge", "status"])
    p.add_argument("--rebuild", action="store_true",
                   help="build: recompute every key, ignoring existing shards")
    p.add_argument("--nproc", type=int, default=1,
                   help="build (serial launch only): multiprocessing pool size")
    p.add_argument("--limit", type=int, default=None,
                   help="build: only the first N todo keys (smoke test)")
    args = p.parse_args()

    if args.mode == "migrate":
        mode_migrate()
    elif args.mode == "merge":
        mode_merge()
    elif args.mode == "status":
        done = sum(1 for _ in SHARDS.glob("run_*_snap_*.parquet")) + \
               sum(1 for _ in SHARDS.glob("*.empty"))
        todo = _todo_keys()
        print(f"[status] cond_src={COND_SRC}")
        print(f"[status] shards present: {done} | keys remaining: {len(todo)} "
              f"/ {len(_all_keys())} total")
    else:
        mode_build(rebuild=args.rebuild, nproc=args.nproc, limit=args.limit)
