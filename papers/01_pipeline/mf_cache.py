"""Cached Minkowski functionals on the extended threshold grid nu in [-3, 8].

WHY THIS EXISTS. The released nongaussian_stats.npz caches V0/V1/V2 only on
linspace(-3, 4, 29) — the library default. Paper I's fig 4 needs nu up to 8, so the
functionals must be recomputed. Doing that inline costs ~0.25 s per map x 50
realizations x 5 source planes x 2 sides ~ 2 min of pure compute, plus decompressing
two ~1 GB kappa cubes, on EVERY notebook execution. This module computes them once,
in parallel, and stores the result; the notebook then loads a ~1 MB file.

The cache is keyed by the threshold grid and by the map geometry, so a stale cache
cannot silently be used against a different grid — `load()` returns None on any
mismatch and the notebook falls back to computing inline.

BUILD (parallel, 10 independent tasks = 2 sides x 5 source planes):
    sbatch papers/01_pipeline/run_mf_cache.sbatch
then the shards are merged automatically by the same script. To rebuild one shard:
    python build_mf_cache.py --side bind --plane 2

USE (from the notebook):
    from mf_cache import MF_NU8, load as load_mf
    mf = load_mf()          # dict or None
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
CACHE_DIR = SCI / "mf_cache"
SHARD_DIR = CACHE_DIR / "shards"
CACHE = CACHE_DIR / "mf_nu8_snap096.npz"

# The extended grid. MUST contain the released linspace(-3, 4, 29) as a prefix so the
# recomputation can be validated against the cached V0/V1/V2 before it is trusted.
MF_NU8 = np.linspace(-3.0, 8.0, 45)
SIDES = ("bind", "truth")
N_PLANES = 5
FOV_DEG = 5.0
SMOOTH_ARCMIN = 1.0          # nongaussian_stats uses its FIRST smoothing scale for the MFs
KEYS = ("V0", "V1", "V2")


def shard_path(side: str, plane: int) -> Path:
    return SHARD_DIR / f"mf_{side}_z{plane}.npz"


def compute(side: str, plane: int) -> dict[str, np.ndarray]:
    """V0/V1/V2 for one (side, source plane): per-realization draws + the mean."""
    from bind.inference.stats import nongaussian_stats

    p = SCI / f"runs/{side}/run_0000/kappa_maps.npz"
    a = np.load(p)
    K = a["kappa"][:, plane].astype(np.float32)          # (n_real, N, N)
    del a
    out = nongaussian_stats(K[:, None], fov_deg=FOV_DEG,
                            smoothing_scales_arcmin=(SMOOTH_ARCMIN,),
                            mf_thresholds=MF_NU8, return_realizations=True)
    res = {k: np.asarray(out[f"{k}_real"])[:, 0] for k in KEYS}   # (n_real, 45)
    res.update({f"{k}_mean": np.asarray(out[k])[0] for k in KEYS})
    return res


def merge() -> Path:
    """Collect the 10 shards into one cache file. Raises if any shard is missing."""
    store: dict[str, np.ndarray] = {"mf_nu": MF_NU8}
    for side in SIDES:
        per_real, per_mean = {k: [] for k in KEYS}, {k: [] for k in KEYS}
        for zi in range(N_PLANES):
            sp = shard_path(side, zi)
            if not sp.exists():
                raise FileNotFoundError(f"missing shard {sp} — rerun that task")
            s = np.load(sp)
            for k in KEYS:
                per_real[k].append(s[k])
                per_mean[k].append(s[f"{k}_mean"])
        for k in KEYS:
            store[f"{side}_{k}"] = np.stack(per_real[k], axis=1)      # (n_real, 5, 45)
            store[f"{side}_{k}_mean"] = np.stack(per_mean[k], axis=0)  # (5, 45)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **store)
    return CACHE


def load(n_real: int | None = None):
    """Return the cache as a dict, or None if absent / built on a different grid.

    A mismatch is never an error — the notebook recomputes inline instead, so the
    figure is always reproducible from the released maps alone.
    """
    if not CACHE.exists():
        return None
    d = np.load(CACHE)
    if "mf_nu" not in d or not np.allclose(d["mf_nu"], MF_NU8):
        return None
    need = [f"{s}_{k}" for s in SIDES for k in KEYS]
    if any(k not in d for k in need):
        return None
    if n_real is not None and d[need[0]].shape[0] != n_real:
        return None
    if d[need[0]].shape[1] != N_PLANES or d[need[0]].shape[2] != len(MF_NU8):
        return None
    return {k: d[k] for k in d.files}
