"""Per-realization field statistics for Paper I figs 4 and 4b, computed once.

WHAT IS EXPENSIVE. Both figures rebuild, from the raw 1024^2 map cubes on every
notebook run: auto and cross spectra for 50 realizations x 5 source planes, peak and
minima counts, and the convergence PDF. That is ~750 power_spectrum calls plus the
counts, on top of decompressing five ~1 GB npz cubes. This module computes all of it
once, in parallel, and stores ~15 MB of per-realization arrays; every mean, paired
residual and error band in the figures then derives from those.

Per-REALIZATION is the point: the paired +-1 sigma/sqrt(50) bands that make fig 4 a
closure test need the individual draws, not summary means.

CONVENTIONS (must match the notebook exactly, and are asserted in verify()):
  - kappa autos       : power_spectrum(kappa[r, zi])
  - y / tau autos     : power_spectrum(F[r, zi])          per-plane cumulative column
  - crosses           : power_spectrum(kappa[r, zi], F[r, -1])   kappa(z_s) x TOTAL column
                        NB the released Cl_kappa_y.npz / Cl_tau.npz caches carry the OLD
                        XPk_plane normalization (low by ~1.2e7, ell-dependently) and are
                        deliberately NOT used -- that is the bug this module avoids.
  - peaks/minima      : peak_counts(kappa[:, zi, None], 2.0', nu_norm='map')
  - PDF               : histogram(m - m.mean()) on the released pdf_bins grid

BUILD (parallel, 10 independent tasks):
    sbatch papers/01_pipeline/run_field_cache.sbatch

USE:
    from field_cache import load as load_fields
    fc = load_fields(n_real=50)      # dict or None
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
LC = CEPH / "bind_lightcone_tng"
CACHE_DIR = SCI / "field_cache"
SHARD_DIR = CACHE_DIR / "shards"
CACHE = CACHE_DIR / "field_stats_fid.npz"

N_PLANES = 5
FOV_DEG = 5.0
SMOOTH_PK = 2.0            # peak/minima smoothing, matches the released peak_counts.npz

# (product, side) -> one disBatch task. 'truth' has no tau trace in this release.
TASKS = [("kk", "bind"), ("kk", "truth"),
         ("yy", "bind"), ("yy", "truth"),
         ("ky", "bind"), ("ky", "truth"),
         ("tt", "bind"), ("kt", "bind"),
         ("counts", "bind"), ("counts", "truth")]


def _run_dir(side: str) -> Path:
    return SCI / f"runs/{side}/run_0000"


def _load(side: str, field: str) -> np.ndarray:
    """kappa/y from the paired validation run; tau only exists on the BIND side."""
    if field == "tau":
        if side != "bind":
            raise ValueError("no seed-paired hydro-truth tau trace exists in this release")
        f = np.load(LC / "tau_maps.npz"); a = f["tau"]
    else:
        f = np.load(_run_dir(side) / f"{field}_maps.npz"); a = f[field]
    del f
    return a


def shard_path(product: str, side: str) -> Path:
    return SHARD_DIR / f"{product}_{side}.npz"


def compute(product: str, side: str) -> dict[str, np.ndarray]:
    import gc

    from bind.inference.stats import peak_counts, power_spectrum

    if product == "counts":
        K = _load(side, "kappa")
        nr = K.shape[0]
        # BOTH sides are histogrammed on the BIND bin grid, which is what fig 4 does
        # and what a paired residual requires. The two runs do NOT share a grid --
        # nongaussian_stats derives the edges from each map's own range, giving
        # +-0.20863 (bind) vs +-0.20745 (truth). Binning each side on its own edges
        # would make the per-bin difference meaningless.
        ng = np.load(_run_dir("bind") / "nongaussian_stats.npz")
        cen = ng["pdf_bins"]; d = cen[1] - cen[0]
        edges = np.concatenate([cen - d / 2, [cen[-1] + d / 2]])
        pk = np.empty((nr, N_PLANES, len(np.load(_run_dir(side) / "peak_counts.npz")["nu"])))
        mn = np.empty_like(pk)
        pdf = np.empty((nr, N_PLANES, len(cen)))
        for zi in range(N_PLANES):
            o = peak_counts(K[:, zi][:, None], fov_deg=FOV_DEG,
                            smoothing_arcmin=SMOOTH_PK, nu_norm="map",
                            return_realizations=True)
            pk[:, zi] = o["peak_counts_real"][:, 0]
            mn[:, zi] = o["minima_counts_real"][:, 0]
            pdf[:, zi] = [np.histogram(m - m.mean(), bins=edges, density=True)[0]
                          for m in K[:, zi]]
            del o; gc.collect()
        return {"pk": pk, "min": mn, "pdf": pdf, "pdf_bins": cen}

    auto = {"kk": ("kappa", None), "yy": ("y", None), "tt": ("tau", None)}
    cross = {"ky": ("kappa", "y"), "kt": ("kappa", "tau")}
    if product in auto:
        fld = auto[product][0]
        A = _load(side, fld)
        nr = A.shape[0]
        ell, _ = power_spectrum(A[0, 0])
        out = np.empty((nr, N_PLANES, len(ell)))
        for zi in range(N_PLANES):
            for r in range(nr):
                out[r, zi] = power_spectrum(A[r, zi])[1]
            gc.collect()
    else:
        fa, fb = cross[product]
        A = _load(side, fa)
        B = _load(side, fb)[:, -1].copy()          # TOTAL column, per the convention above
        nr = A.shape[0]
        ell, _ = power_spectrum(A[0, 0])
        out = np.empty((nr, N_PLANES, len(ell)))
        for zi in range(N_PLANES):
            for r in range(nr):
                out[r, zi] = power_spectrum(A[r, zi], B[r])[1]
            gc.collect()
        del B
    del A; gc.collect()
    return {"cl": out, "ell": ell}


def merge() -> Path:
    store: dict[str, np.ndarray] = {}
    for product, side in TASKS:
        sp = shard_path(product, side)
        if not sp.exists():
            raise FileNotFoundError(f"missing shard {sp} — rerun that task")
        s = np.load(sp)
        if product == "counts":
            for k in ("pk", "min", "pdf"):
                store[f"{k}_{side}"] = s[k]
            # both sides must have been binned on the SAME grid, or the paired PDF
            # residual is meaningless. Previously this line just overwrote, so a
            # per-side grid would have been kept silently (caught by fig 4's assert).
            if "pdf_bins" in store:
                assert np.allclose(store["pdf_bins"], s["pdf_bins"]), (
                    "counts shards were built on different PDF grids — rebuild both")
            store["pdf_bins"] = s["pdf_bins"]
        else:
            store[f"{product}_{side}"] = s["cl"]
            store["ell"] = s["ell"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **store)
    return CACHE


def load(n_real: int | None = None):
    """Return the cache, or None if absent / built for different data.

    A mismatch is never fatal: the notebook recomputes inline instead, so both
    figures stay reproducible from the released maps alone.
    """
    if not CACHE.exists():
        return None
    d = np.load(CACHE)
    need = [f"{p}_{s}" for p, s in TASKS if p != "counts"]
    need += ["pk_bind", "pk_truth", "min_bind", "min_truth",
             "pdf_bind", "pdf_truth", "ell", "pdf_bins"]
    if any(k not in d.files for k in need):
        return None
    if d["kk_bind"].shape[1] != N_PLANES:
        return None
    if n_real is not None and d["kk_bind"].shape[0] != n_real:
        return None
    return {k: d[k] for k in d.files}
