"""LSST-Y10 shape-noise + smoothing statistics, exact Lee et al. 2022 recipe.

WHY. Every other statistic cache in this directory (`nu_grid.py`, `mf_cache.py`,
`field_cache.py`) works on the raw, noiseless kappa maps -- appropriate for a
closure/validation test against the truth simulation, but not for anything that
claims to look like a real survey measurement. This module adds realistic LSST-Y10
shape noise and a survey-scale Gaussian smoothing to the SAME kappa cubes, exactly
as the author specified for the BCM-vs-hydro peak-statistics paper (Lee, Lu, Haiman,
Liu & Osato 2022, MNRAS 519, 573; arXiv:2201.08320, referred to below as "L22"),
Sec 2.3-2.5, with the LSST-Y10 numbers. NO REPAINT happens here: this only reads the
released `kappa_maps.npz` cubes, adds noise, smooths, and computes statistics.

THE RECIPE, IN ORDER (L22 Sec 2.3-2.5):

1. Shape noise (L22 Eq. 7) -- ADDED FIRST, before any smoothing. Per pixel, i.i.d.
   Gaussian, mean 0, variance

       sigma^2 = sigma_e^2 / (2 * n_gal * A_pix)

   with sigma_e = 0.26 (per-component galaxy ellipticity dispersion), n_gal = 27
   arcmin^-2 (LSST-Y10 source density), A_pix = (0.29296875 arcmin)^2 the pixel
   area of these 5-deg / 1024-px maps (5*60/1024 arcmin/px). NOTE the factor of 2
   in the denominator -- this is L22's own convention and is NOT the same number as
   `bind.inference.stats._noise_sigma_pix` (`sigma_e / sqrt(n_gal * A_pix)`, no
   factor of 2), which implements a different paper's convention. Do not conflate
   the two; NOISE_SIGMA_PIX below is the L22 number.

2. Smoothing (L22 Eq. 6) -- applied to the ALREADY-NOISY map. L22's window is
       W(theta) ~ exp(-theta^2 / theta_G^2),   theta_G = 1 arcmin.
   *** theta_G IS THE 1/e RADIUS OF THAT GAUSSIAN, NOT ITS STANDARD DEVIATION. ***
   Writing the same window as exp(-theta^2 / (2 sigma^2)) (the usual Gaussian-filter
   convention, e.g. `scipy.ndimage.gaussian_filter`'s `sigma`) gives
       sigma = theta_G / sqrt(2) = 0.70710678... arcmin ~= 2.41359 pixels.
   `SMOOTH_SIGMA_ARCMIN` / `SMOOTH_SIGMA_PIX` below are this sigma, already
   converted -- pass `SMOOTH_SIGMA_PIX` straight to `gaussian_filter`. Getting this
   backwards (using theta_G=1' AS the filter sigma) over-smooths by a factor 1.414
   and is the single easiest way to silently break this whole cache.
   ALL statistics below are computed on the smoothed map -- L22's convention is that
   the analysis happens on the smoothed field, unlike `nu_grid.py`'s mixed
   smoothing scales (PDF unsmoothed, peaks/minima at 2', MF at 1').

3. nu normalisation -- adapted from L22 Sec 2.4/2.5 ("use kappa_TNG's kappa_rms for
   everything"): a single per-source-plane number, KAPPA_RMS[plane], measured ONCE
   from the FIDUCIAL suite's own noisy, smoothed maps (the mean, over fiducial
   realizations, of each map's std), then reused as the nu denominator for every
   other target (truth, DMO, all 256 Sobol runs). This is the direct analogue of
   `bind.inference.stats`'s `nu_norm="fixed"` convention (a shared sigma so that a
   parameter response is not partly absorbed by the map's own changing sigma_kappa),
   here with the shared sigma additionally carrying the shape-noise contribution.
   THE ACTUAL NUMBERS live in a small cache file (see `load_kappa_rms` /
   `compute_kappa_rms`) built once from the fiducial target and then required
   (refuse, don't guess -- `load_kappa_rms` raises if it is missing) by every other
   build. Every shard also carries its own copy under the `kappa_rms` key so it is
   traceable without re-reading the cache file.

4. Seeding -- REPRODUCIBLE and INDEPENDENT per target. `noise_rng(target,
   realization, plane)` seeds a `np.random.default_rng` from a stable (sha256, not
   Python's process-randomized `hash()`) hash of the `target` string plus the
   realization and plane indices. Different targets ("fid", "truth", "dmo",
   "run_0000", ...) therefore draw STATISTICALLY INDEPENDENT noise, matching a real
   survey (BIND and the hydro truth are not the same observation, so they do not
   see the same shape-noise realization) while remaining bit-reproducible across
   processes and reruns.

STATISTICS (per realization, per source plane, on the noisy smoothed map):
`peak_counts`, `minima_counts`, `pdf` -- histograms on the canonical `NU_EDGES` (23
edges, imported from `nu_grid.py`, NOT redefined here so the two caches can never
drift apart); `mf_v0`, `mf_v1`, `mf_v2` -- Minkowski-functional threshold
functionals evaluated AT the 22 bin centres `NU` (see `nu_grid.py`'s docstring for
the edges-vs-centres distinction, which is unchanged here); `cl_kappa` -- the
per-plane AUTO angular power spectrum of the noisy smoothed map, on
`bind.inference.stats.power_spectrum`'s natural ell grid (the same helper
`nu_grid.py`'s sibling caches would use).

WHAT IT DOES NOT DO. No repainting, no new maps: every target's `kappa_maps.npz` is
read as released. Nothing under `nu05_shards/` (the noiseless cache) or any released
`emulator_dataset*.npz` is touched. Shards land in `bind_sb35/nu05n_shards/`
("n" for "noisy"); the assembled Sobol dataset is a NEW file,
`emulator_dataset_nu05n.npz`.

BUILD:  see `build_noisy_cache.py --help` and `run_noisy_cache.sbatch`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

import nu_grid as ng

CEPH = Path("/mnt/home/mlee1/ceph")
SB35 = CEPH / "bind_sb35"
SCI = CEPH / "bind_science"
LC = CEPH / "bind_lightcone_tng"

# ── the one nu grid, shared with nu_grid.py by import, never redefined ────────
NU = ng.NU                 # 22 bin centres, -2.75 .. 7.75
NU_EDGES = ng.NU_EDGES     # 23 edges, width 0.5, -3 .. 8
DNU = ng.DNU

FOV_DEG = 5.0
NPIX = 1024
PIX_ARCMIN = FOV_DEG * 60.0 / NPIX     # 0.29296875 arcmin/px (5 deg * 60 / 1024)
N_PLANES = 5                            # expected source-plane count; asserted, not assumed

# ── 1. L22 Eq. 7 shape noise (LSST-Y10) ────────────────────────────────────────
SIGMA_E = 0.26
NGAL_ARCMIN2 = 27.0
A_PIX_ARCMIN2 = PIX_ARCMIN ** 2
NOISE_SIGMA_PIX = SIGMA_E / np.sqrt(2.0 * NGAL_ARCMIN2 * A_PIX_ARCMIN2)   # per-pixel kappa noise std

# ── 2. L22 Eq. 6 smoothing: theta_G is the 1/e RADIUS, sigma = theta_G/sqrt(2) ─
THETA_G_ARCMIN = 1.0
SMOOTH_SIGMA_ARCMIN = THETA_G_ARCMIN / np.sqrt(2.0)     # 0.70710678... arcmin
SMOOTH_SIGMA_PIX = SMOOTH_SIGMA_ARCMIN / PIX_ARCMIN     # 2.41359... px

KEYS = ("peak_counts", "minima_counts", "pdf", "mf_v0", "mf_v1", "mf_v2")
CL_KEY = "cl_kappa"

SHARD_DIR = SB35 / "nu05n_shards"
KAPPA_RMS_CACHE = SHARD_DIR / "_kappa_rms_fid.npz"
DATASET_IN = SB35 / "emulator_dataset_nu05.npz"          # noiseless nu05, for run_ids/params only
DATASET_OUT = SB35 / "emulator_dataset_nu05n.npz"

# target name -> kappa_maps.npz path. "fid" is bind_lightcone_tng (550 real, flat
# layout, NOT bind_science/runs/bind/run_0000 -- that is the older 50-real fiducial
# used by nu_grid.py/field_cache.py/mf_cache.py and is untouched by this module).
TARGETS = {
    "fid": LC / "kappa_maps.npz",
    "truth": SCI / "runs/truth/run_0000/kappa_maps.npz",
    "dmo": SCI / "runs/dmo/run_0000/kappa_maps.npz",
}


def sobol_path(run: str) -> Path:
    return SB35 / f"runs/{run}/kappa_maps.npz"


def kappa_path(target: str) -> Path:
    return TARGETS[target] if target in TARGETS else sobol_path(target)


def n_real_of(path: Path) -> int:
    """Realization count of a kappa_maps.npz WITHOUT decompressing the array.

    `kappa.npy`'s shape lives in its npy header, the first ~100 bytes of the
    (deflate-compressed) zip member; reading it costs milliseconds even for the
    fid/truth/dmo cubes, whose full `kappa` array takes ~110 s to decompress. Used
    by `run_noisy_cache.sbatch` to size realization chunks up front.
    """
    import zipfile
    import numpy.lib.format as fmt
    with zipfile.ZipFile(path) as z, z.open("kappa.npy") as f:
        ver = fmt.read_magic(f)
        shape, _, _ = fmt._read_array_header(f, ver)
    return int(shape[0])


# ── seeding: reproducible, independent per target ──────────────────────────────

def _stable_hash(s: str) -> int:
    """A `hash()` substitute that is stable across processes (PYTHONHASHSEED-proof)."""
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") % (2 ** 31 - 1)


def noise_rng(target: str, realization: int, plane: int) -> np.random.Generator:
    """Reproducible RNG for one (target, realization, plane) shape-noise draw.

    Seeded from a stable hash of `target` plus the two integer indices, so re-running
    a shard reproduces it bit-for-bit, and DIFFERENT targets draw INDEPENDENT noise
    (real surveys do not share a shape-noise realization between two different
    "observations" of the same sky patch -- BIND and truth must not either).
    """
    return np.random.default_rng((_stable_hash(target), int(realization), int(plane)))


def add_shape_noise(m: np.ndarray, target: str, realization: int, plane: int) -> np.ndarray:
    """L22 Eq. 7: i.i.d. Gaussian noise, mean 0, sigma=NOISE_SIGMA_PIX, added BEFORE smoothing."""
    rng = noise_rng(target, realization, plane)
    return m.astype(np.float64) + rng.normal(0.0, NOISE_SIGMA_PIX, m.shape)


def smooth(m: np.ndarray) -> np.ndarray:
    """L22 Eq. 6 Gaussian smoothing at sigma=SMOOTH_SIGMA_PIX (theta_G=1' 1/e radius / sqrt(2)).

    `mode="wrap"` matches `bind.inference.stats._gaussian_smooth`'s convention
    elsewhere in this pipeline.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(m, SMOOTH_SIGMA_PIX, mode="wrap")


# ── 3. kappa_rms: the one shared nu denominator, from the fiducial suite only ──

def compute_kappa_rms(n_real: int | None = None, target: str = "fid") -> tuple[np.ndarray, int]:
    """Per-plane kappa_rms of the noisy, smoothed FIDUCIAL suite (see module docstring #3).

    Mean, over fiducial realizations, of each noisy-smoothed map's own std. Cheap:
    only noise+smoothing, no extrema/MF/Cl, so even the full 550 realizations take
    minutes, not hours. `n_real` caps the realization count (smoke tests / speed);
    the default (None) uses every realization the fiducial cube holds.
    """
    a = np.load(kappa_path(target))
    K = a["kappa"]
    del a
    n_planes = K.shape[1]
    nr = K.shape[0] if n_real is None else min(n_real, K.shape[0])
    rms = np.empty(n_planes)
    for zi in range(n_planes):
        acc = 0.0
        for r in range(nr):
            sm = smooth(add_shape_noise(K[r, zi], target, r, zi))
            acc += float(sm.std())
        rms[zi] = acc / nr
    del K
    return rms, nr


def save_kappa_rms(rms: np.ndarray, n_real_used: int) -> Path:
    """Atomic write (tmp + os.replace), safe against concurrent disBatch tasks."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    tmp = KAPPA_RMS_CACHE.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(tmp, kappa_rms=rms, n_real_used=np.array(n_real_used))
    os.replace(tmp, KAPPA_RMS_CACHE)
    return KAPPA_RMS_CACHE


def load_kappa_rms() -> np.ndarray:
    """Refuse, don't guess: every OTHER build depends on this ONE per-plane number.

    Raises loudly if `compute_kappa_rms` + `save_kappa_rms` (i.e.
    `build_noisy_cache.py --kappa-rms`) has not run yet, rather than silently
    falling back to a per-map or per-run normalisation that would make every run's
    nu axis mean something different.
    """
    if not KAPPA_RMS_CACHE.exists():
        raise FileNotFoundError(
            f"{KAPPA_RMS_CACHE} missing -- run "
            "`python build_noisy_cache.py --kappa-rms` (reads the fiducial target) "
            "BEFORE building any other shard; every shard's nu normalisation depends "
            "on this one cached per-plane number, computed once from the fiducial "
            "suite's own noisy smoothed maps.")
    return np.load(KAPPA_RMS_CACHE)["kappa_rms"]


# ── per-(realization, plane) statistics on the noisy smoothed map ─────────────

def compute(
    target: str,
    real_start: int = 0,
    real_end: int | None = None,
    per_real: bool = True,
) -> dict[str, np.ndarray]:
    """All 6 nu statistics + cl_kappa for one target, one realization range.

    Realizations `[real_start, real_end)` of `target`'s kappa cube (`real_end=None`
    -> to the end of the cube; chunking exists purely so a 550-realization target
    can be split into <=30-min tasks, see `build_noisy_cache.py`). Every map gets
    noised (Eq. 7) then smoothed (Eq. 6) then normalised by the FIXED per-plane
    `load_kappa_rms()` value (module docstring #3): `nu = (smoothed -
    smoothed.mean()) / kappa_rms[plane]` -- the per-map mean subtraction matches
    every other nu convention in this pipeline (`nu_grid.py`, `peak_counts`,
    `nongaussian_stats`); only the denominator is fixed rather than per-map.

    `peak_counts`/`minima_counts`/`pdf` are histograms on `NU_EDGES`; `mf_v0/1/2`
    are threshold functionals evaluated AT `NU` (see `nu_grid.py`'s docstring).
    `cl_kappa` is the per-plane AUTO power spectrum of the same noisy smoothed map.

    `per_real=True` additionally keeps the `(n_real_chunk, plane, ...)` draws
    (fid/truth/dmo -- needed for the L22 Eq. 8 -style paired covariance); Sobol
    shards use `per_real=False` (mean + err only, to save space over 256 runs).
    """
    from bind.inference.stats import _local_extrema, minkowski_functionals, power_spectrum

    kappa_rms = load_kappa_rms()
    a = np.load(kappa_path(target))
    K = a["kappa"]
    fov = float(a["fov_deg"]) if "fov_deg" in a.files else FOV_DEG
    npix_file = int(a["npix"]) if "npix" in a.files else K.shape[-1]
    del a
    # NOISE_SIGMA_PIX / SMOOTH_SIGMA_PIX are derived once from the MODULE-level
    # FOV_DEG/NPIX (i.e. the 0.29296875 arcmin/px assumed throughout this module's
    # docstring); if a target ever stored a different geometry, using those module
    # constants against ITS pixel scale would be silently wrong. Refuse rather than
    # guess (checked against every target this session: all report fov_deg=5.0,
    # npix=1024, matching the module assumption).
    assert fov == FOV_DEG and npix_file == NPIX, (
        f"{target}: kappa_maps.npz reports fov_deg={fov}, npix={npix_file}, but "
        f"NOISE_SIGMA_PIX/SMOOTH_SIGMA_PIX assume FOV_DEG={FOV_DEG}, NPIX={NPIX} "
        "(0.29296875 arcmin/px) -- recheck the recipe constants before proceeding.")
    n_real_tot, n_planes = K.shape[:2]
    assert n_planes == N_PLANES, f"{target}: expected {N_PLANES} planes, found {n_planes}"
    assert len(kappa_rms) == n_planes, "kappa_rms cache does not match this cube's plane count"
    r_end = n_real_tot if real_end is None else min(real_end, n_real_tot)
    r_idx = list(range(real_start, r_end))
    nr = len(r_idx)
    if nr == 0:
        raise ValueError(f"{target}: empty realization range [{real_start}, {r_end})")

    out = {k: np.empty((n_planes, len(NU))) for k in KEYS}
    err = {k: np.empty((n_planes, len(NU))) for k in ("peak_counts", "minima_counts")}
    real = {k: np.empty((nr, n_planes, len(NU))) for k in KEYS} if per_real else {}
    cl_out = cl_err = cl_real = None
    ell = None

    for zi in range(n_planes):
        pk = np.empty((nr, len(NU))); mn = np.empty((nr, len(NU)))
        pdf = np.empty((nr, len(NU)))
        v0 = np.empty((nr, len(NU))); v1 = np.empty((nr, len(NU))); v2 = np.empty((nr, len(NU)))
        cl_chunk = None
        for ri, r in enumerate(r_idx):
            noisy = add_shape_noise(K[r, zi], target, r, zi)
            sm = smooth(noisy)
            nu = (sm - sm.mean()) / kappa_rms[zi]
            pkmask = _local_extrema(sm, True)
            mnmask = _local_extrema(sm, False)
            pk[ri] = np.histogram(nu[pkmask], bins=NU_EDGES)[0]
            mn[ri] = np.histogram(nu[mnmask], bins=NU_EDGES)[0]
            pdf[ri] = np.histogram(nu, bins=NU_EDGES, density=True)[0]
            v0[ri], v1[ri], v2[ri] = minkowski_functionals(nu, NU)
            ell_r, cl = power_spectrum(sm, fov_deg=fov)
            if cl_chunk is None:
                ell = ell_r
                cl_chunk = np.empty((nr, len(ell)))
            cl_chunk[ri] = cl
        out["peak_counts"][zi] = pk.mean(0)
        out["minima_counts"][zi] = mn.mean(0)
        err["peak_counts"][zi] = pk.std(0) / np.sqrt(nr)
        err["minima_counts"][zi] = mn.std(0) / np.sqrt(nr)
        out["pdf"][zi] = pdf.mean(0)
        out["mf_v0"][zi] = v0.mean(0); out["mf_v1"][zi] = v1.mean(0); out["mf_v2"][zi] = v2.mean(0)
        if cl_out is None:
            cl_out = np.empty((n_planes, len(ell)))
            cl_err = np.empty((n_planes, len(ell)))
            if per_real:
                cl_real = np.empty((nr, n_planes, len(ell)))
        cl_out[zi] = cl_chunk.mean(0)
        cl_err[zi] = cl_chunk.std(0) / np.sqrt(nr)
        if per_real:
            real["peak_counts"][:, zi] = pk
            real["minima_counts"][:, zi] = mn
            real["pdf"][:, zi] = pdf
            real["mf_v0"][:, zi] = v0; real["mf_v1"][:, zi] = v1; real["mf_v2"][:, zi] = v2
            cl_real[:, zi] = cl_chunk
        del pk, mn, pdf, v0, v1, v2, cl_chunk
    del K

    res = {**out, **{f"{k}_err": v for k, v in err.items()}, "nu": NU,
           "cl_kappa": cl_out, "cl_kappa_err": cl_err, "cl_ell": ell,
           "kappa_rms": kappa_rms, "real_start": real_start, "real_end": r_end,
           "n_real": nr, "target": target}
    if per_real:
        res.update({f"{k}_real": v for k, v in real.items()})
        res["cl_kappa_real"] = cl_real
    return res
