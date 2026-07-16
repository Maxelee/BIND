"""Data layer for the field-level WL emulator: cache + normalisation + dataset.

The 256-point SB35 Sobol κ suite is ~256 GB of compressed npz
(``run_NNNN/kappa_maps.npz`` holds ``kappa`` of shape
``(n_real, n_z, 1024, 1024)``).  Training on that directly is I/O-bound, so we
pre-pool it once into a single memmap-able cache:

    <cache>/kappa.npy   float16 (N_runs, n_real, n_z, R, R)   raw κ, mean-pooled
    <cache>/meta.npz    params_unit (N,30), source_redshifts, run_ids, fov, res
    <cache>/norm.npz    KappaNorm: per-z arcsinh softening λ + standardisation μ,σ

We store **raw** (pooled) κ so the normalisation can be re-tuned without
re-reading the npz; ``KappaCacheDataset`` applies the transform on the fly.

Normalisation (per source redshift, because κ variance grows strongly with
``z_s`` and the tail is heavy — max ≈ 70σ):

    y = arcsinh(κ / λ_z)          # tames the positive (halo) tail, keeps sign
    x = (y - μ_z) / σ_z           # standardise → ~unit-variance FM target

inverse:  κ = λ_z · sinh(σ_z · x + μ_z).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bind.emulator.dataset import ASTRO_PARAM_NAMES, SOURCE_REDSHIFTS, params_to_unit

__all__ = [
    "ASTRO_PARAM_NAMES",
    "SOURCE_REDSHIFTS",
    "KappaNorm",
    "KappaCache",
    "KappaCacheDataset",
    "build_cache",
]

DEFAULT_RUNS = Path("/mnt/home/mlee1/ceph/bind_sb35/runs")
DEFAULT_DESIGN = Path("/mnt/home/mlee1/ceph/bind_sb35/design/astro_params_sobol.npy")

_RUN_RE = re.compile(r"run_(\d+)")


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class KappaNorm:
    """Per-source-redshift arcsinh + standardisation of convergence maps.

    Arrays are length ``n_z`` (one entry per source plane).
    """

    lam: np.ndarray      # arcsinh softening scale λ_z (≈ std of κ_z)
    mu: np.ndarray       # mean of arcsinh(κ/λ)
    sigma: np.ndarray    # std  of arcsinh(κ/λ)
    source_redshifts: np.ndarray

    def forward(self, kappa, z_idx):
        """Raw κ → normalised. ``z_idx`` is an int or array broadcasting to κ."""
        lam = np.take(self.lam, z_idx)
        mu = np.take(self.mu, z_idx)
        sig = np.take(self.sigma, z_idx)
        y = np.arcsinh(kappa / lam)
        return (y - mu) / sig

    def inverse(self, x, z_idx):
        """Normalised → raw κ."""
        lam = np.take(self.lam, z_idx)
        mu = np.take(self.mu, z_idx)
        sig = np.take(self.sigma, z_idx)
        return lam * np.sinh(sig * x + mu)

    # torch-friendly scalar accessors (z_idx is a python int) -------------------
    def scale_factor(self, z_idx):
        """a = 1/(1+z_s) for source plane ``z_idx`` — the redshift conditioner."""
        return 1.0 / (1.0 + float(self.source_redshifts[z_idx]))

    def save(self, path):
        np.savez(
            path,
            lam=self.lam,
            mu=self.mu,
            sigma=self.sigma,
            source_redshifts=self.source_redshifts,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(
            lam=d["lam"],
            mu=d["mu"],
            sigma=d["sigma"],
            source_redshifts=d["source_redshifts"],
        )

    @classmethod
    def from_raw(cls, kappa, source_redshifts, chunk_rows=8):
        """Fit from a raw (pooled) cache array of shape (..., n_z, R, R).

        Streams the array in small row-blocks accumulating per-z (count, Σκ,
        Σκ²) — never materialises the whole cache.  This matters at 1024²: a
        single ``[:, z]`` slice cast to float64 would be ~100 GB.  Two passes
        (λ = std(κ) is needed before the arcsinh standardisation), each reading
        the memmap once in ``chunk_rows``-sized blocks (~hundreds of MB).
        """
        n_z, R = kappa.shape[-3], kappa.shape[-1]
        flat = kappa.reshape(-1, n_z, kappa.shape[-2], R)   # view; no copy
        M = flat.shape[0]
        axes = (0, 2, 3)

        # Pass 1 — per-z mean/std of raw κ → softening λ_z.
        s1, s2 = np.zeros(n_z), np.zeros(n_z)
        for a in range(0, M, chunk_rows):
            blk = np.asarray(flat[a:a + chunk_rows], dtype=np.float64)
            s1 += blk.sum(axis=axes)
            s2 += np.square(blk).sum(axis=axes)
        cnt = M * R * R
        mean = s1 / cnt
        lam = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-30))

        # Pass 2 — per-z mean/std of y = arcsinh(κ/λ_z).
        ys, ys2 = np.zeros(n_z), np.zeros(n_z)
        lam_b = lam[None, :, None, None]
        for a in range(0, M, chunk_rows):
            y = np.arcsinh(np.asarray(flat[a:a + chunk_rows], dtype=np.float64) / lam_b)
            ys += y.sum(axis=axes)
            ys2 += np.square(y).sum(axis=axes)
        mu = ys / cnt
        sigma = np.sqrt(np.maximum(ys2 / cnt - mu ** 2, 1e-30))
        return cls(lam=lam, mu=mu, sigma=sigma,
                   source_redshifts=np.asarray(source_redshifts, dtype=np.float64))


# ──────────────────────────────────────────────────────────────────────────────
# Cache builder
# ──────────────────────────────────────────────────────────────────────────────
def _mean_pool(x, factor):
    """Mean-pool the last two axes by an integer ``factor``."""
    if factor == 1:
        return x
    *lead, h, w = x.shape
    n = h // factor
    return x.reshape(*lead, n, factor, n, factor).mean(axis=(-3, -1))


def _run_id(path: Path) -> int:
    m = _RUN_RE.search(path.name)
    if m is None:
        raise ValueError(f"cannot parse run id from {path}")
    return int(m.group(1))


def _load_pool_one(run_dir: Path, factor: int, n_real: int | None):
    """Read one run's kappa_maps.npz, mean-pool, return (id, pooled, zs, fov)."""
    d = np.load(run_dir / "kappa_maps.npz")
    k = d["kappa"]                                  # (n_real, n_z, 1024, 1024)
    if n_real is not None:
        k = k[:n_real]
    pooled = _mean_pool(k.astype(np.float32), factor).astype(np.float16)
    return _run_id(run_dir), pooled, d["source_redshifts"], float(d["fov_deg"])


def build_cache(out_dir, runs_dir=DEFAULT_RUNS, design_file=DEFAULT_DESIGN,
                resolution: int = 256, n_real: int | None = None,
                n_workers: int = 1, max_runs: int | None = None,
                verbose: bool = True):
    """Single-process cache build (for the notebook / small caches / tests).

    Pools every ``run_NNNN/kappa_maps.npz`` into ``kappa.npy`` + ``meta.npz`` +
    ``norm.npz`` — the same on-disk format the parallel OpenMPI builder
    ``build_wlemu_cache.py`` writes (use that on SLURM for the full 1024² suite).

    Parameters
    ----------
    resolution : target map size; source maps (1024²) are mean-pooled by 1024//R.
    n_real : keep only the first ``n_real`` realisations per run (None = all).
    n_workers : thread workers for reading/pooling the npz.
    max_runs : cap the number of runs (for quick demo/test caches).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if 1024 % resolution != 0:
        raise ValueError(f"resolution {resolution} must divide 1024")
    factor = 1024 // resolution

    run_dirs = sorted(
        (p for p in Path(runs_dir).glob("run_*") if (p / "kappa_maps.npz").exists()),
        key=_run_id)
    if not run_dirs:
        raise FileNotFoundError(f"no run_*/kappa_maps.npz under {runs_dir}")
    if max_runs is not None:
        run_dirs = run_dirs[:max_runs]
    N = len(run_dirs)

    _id0, p0, zs, fov = _load_pool_one(run_dirs[0], factor, n_real)
    n_real_eff, n_z, R, _ = p0.shape
    if verbose:
        gb = N * n_real_eff * n_z * R * R * 2 / 1e9
        print(f"[cache] {N} runs · {n_real_eff} real · {n_z} z · {R}² → {gb:.1f} GB fp16")

    arr = np.lib.format.open_memmap(
        out_dir / "kappa.npy", mode="w+", dtype=np.float16,
        shape=(N, n_real_eff, n_z, R, R))
    arr[0] = p0
    run_ids = np.empty(N, dtype=np.int64)
    run_ids[0] = _id0

    def _do(i):
        rid, pooled, _zs, _fov = _load_pool_one(run_dirs[i], factor, n_real)
        arr[i] = pooled
        return i, rid

    rest = range(1, N)
    if n_workers > 1:
        from joblib import Parallel, delayed
        for i, rid in Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(_do)(i) for i in rest):
            run_ids[i] = rid
    else:
        for i in rest:
            _, run_ids[i] = _do(i)
    arr.flush()

    np.savez(
        out_dir / "meta.npz",
        params_native=np.load(design_file)[run_ids].astype(np.float64),
        source_redshifts=np.asarray(zs, dtype=np.float64),
        run_ids=run_ids, resolution=np.int64(R), fov_deg=np.float64(fov),
        n_real=np.int64(n_real_eff), n_runs=np.int64(N),
    )
    norm = KappaNorm.from_raw(arr, zs)
    norm.save(out_dir / "norm.npz")
    if verbose:
        print(f"[cache] λ_z = {np.array2string(norm.lam, precision=4)}  → {out_dir}")
    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class KappaCache:
    """Read-only handle to a built cache (memmap + metadata + norm)."""

    kappa: np.ndarray            # memmap (N, n_real, n_z, R, R) float16, raw κ
    params_unit: np.ndarray      # (N, 30) float32
    source_redshifts: np.ndarray
    run_ids: np.ndarray
    resolution: int
    fov_deg: float
    n_real: int
    norm: KappaNorm

    @classmethod
    def load(cls, cache_dir, mmap=True):
        cache_dir = Path(cache_dir)
        meta = np.load(cache_dir / "meta.npz")
        kappa = np.load(cache_dir / "kappa.npy", mmap_mode="r" if mmap else None)
        # Accept either the pre-normalised params (single-process build_cache) or
        # the raw native vectors (the standalone MPI build_wlemu_cache.py, which
        # stays bind-free) — normalise the latter to the unit cube here.
        if "params_unit" in meta.files:
            params_unit = meta["params_unit"]
        else:
            params_unit = params_to_unit(meta["params_native"])
        params_unit = np.asarray(params_unit, dtype=np.float32)
        return cls(
            kappa=kappa,
            params_unit=params_unit,
            source_redshifts=meta["source_redshifts"],
            run_ids=meta["run_ids"],
            resolution=int(meta["resolution"]),
            fov_deg=float(meta["fov_deg"]),
            n_real=int(meta["n_real"]),
            norm=KappaNorm.load(cache_dir / "norm.npz"),
        )

    @property
    def n_runs(self):
        return self.kappa.shape[0]

    @property
    def n_z(self):
        return self.kappa.shape[2]

    def run_split(self, val_frac=0.125, seed=0):
        """Split *parameter points* (runs) into (train_idx, val_idx).

        Splitting by run — not by map — makes validation test the actual task:
        generalisation to unseen parameters.  Returns indices into the cache's
        run axis (0..n_runs-1), not raw run ids.
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_runs)
        n_val = max(1, int(round(val_frac * self.n_runs)))
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])


try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover - torch always present in practice
    _TorchDataset = object


class KappaCacheDataset(_TorchDataset):
    """Torch dataset over a :class:`KappaCache`.

    Each item is one (run, realisation, source-plane) triple →
    ``(x[1,R,R], params[30], scale_factor[])`` with ``x`` arcsinh-normalised and
    ``scale_factor = 1/(1+z_s)`` the redshift conditioner.  A single model is
    trained across all source planes.
    """

    def __init__(self, cache: KappaCache, run_idx=None, z_idx=None):
        self.cache = cache
        self.runs = np.arange(cache.n_runs) if run_idx is None else np.asarray(run_idx)
        self.zsel = np.arange(cache.n_z) if z_idx is None else np.asarray(z_idx)
        self.n_real = cache.n_real
        self.scale_factors = (1.0 / (1.0 + cache.source_redshifts)).astype(np.float32)
        self._per_run = self.n_real * len(self.zsel)

    def __len__(self):
        return len(self.runs) * self._per_run

    def __getitem__(self, i):
        r, rem = divmod(i, self._per_run)
        real, zk = divmod(rem, len(self.zsel))
        run = int(self.runs[r])
        z = int(self.zsel[zk])
        kappa = np.asarray(self.cache.kappa[run, real, z], dtype=np.float32)
        x = self.cache.norm.forward(kappa, z).astype(np.float32)
        return (
            torch.from_numpy(x)[None],                                   # (1,R,R)
            torch.from_numpy(self.cache.params_unit[run].copy()),        # (30,)
            torch.tensor(self.scale_factors[z]),                         # ()
        )
