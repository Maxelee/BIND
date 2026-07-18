"""Flat mock-patch geometry: peaks, periodic padding, and the enmap bridge (WP-B4).

Produced by Project B WP-B4 (forward model; see bind-paper3-plans, private
repo). The twobound/bind/truth atlas stores flat-sky periodic 5°-FOV 1024²
patches; the frozen B2 measurement operators (`maps.stack.extract_thumbnails`,
`stack.cap.cap_filter_multi`) consume a pixell enmap + (ra, dec) — this module
is the bridge: periodic-pad a patch, wrap it into a CAR enmap centered on
(ra, dec) = (0, 0), and convert peak pixel indices to sky coordinates, so the
mock side runs the *identical imported* measurement code (MEASUREMENT_SPEC /
B4 acceptance: no mock-specific estimator).

Geometry caveat (documented, negligible): interpreting the flat patch's
uniform pixel grid as CAR pixels centered on the equator distorts physical
scales by at most 1 − cos(2.5° + pad) ≈ 0.12% at the patch corners — far
below the B4 tolerance (±5–10%).

Peak rule: PEAK_DEFINITION.md §2 verbatim — a pixel of the *smoothed* map is a
peak iff strictly greater than all 8 Moore neighbours, with periodic wrap
(`np.roll`), exactly as Paper-2's `bind.inference.stats._local_extrema`
(re-implemented here because `stats.py` lives on the lightcone branch, not in
this worktree; the unit test pins equivalence against a brute-force check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Atlas patch constants (every twobound/bind/truth npz: fov_deg=5.0, npix=1024).
ATLAS_FOV_DEG: float = 5.0
ATLAS_NPIX: int = 1024

# Padding for thumbnail extraction: THUMB_R_ARCMIN = 15' plus margin, in pixels
# (15.5' / 0.293' = 53; use 60 for slack).
DEFAULT_PAD_PIX: int = 60


@dataclass(frozen=True)
class PatchGeometry:
    """Flat periodic patch geometry (atlas defaults)."""

    fov_deg: float = ATLAS_FOV_DEG
    npix: int = ATLAS_NPIX

    def arcmin_per_pixel(self) -> float:
        return self.fov_deg * 60.0 / self.npix

    def pixel_area_arcmin2(self) -> float:
        return self.arcmin_per_pixel() ** 2


def find_peaks_flat(smoothed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(i, j) of strict 8-neighbour local maxima with periodic wrap.

    PEAK_DEFINITION §2: strictly greater than every Moore neighbour of the
    smoothed map. Operates on the *smoothed* map — smoothing is the caller's
    job (`smoothing.smooth_flat_sky`).
    """
    m = np.asarray(smoothed)
    out = np.ones(m.shape, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            out &= m > np.roll(np.roll(m, dx, 0), dy, 1)
    return np.nonzero(out)


def coarse_grid_peaks(smoothed: np.ndarray, block: int = 12
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coarse-grid peak FINDING (supersedes position-only quantization).

    Models the DATA side's peak-FINDING grid: DATA finds peaks directly on
    the Nside=1024 HEALPix map (mean pixel side 3.435'), so nearby maxima
    on the mock's much finer native 0.293' grid (1024^2, 5-deg FOV) that
    would be distinct HEALPix pixels' worth apart can be a single data peak.
    An audit found this MERGES nearby maxima relative to native-grid finding
    (~1.3-1.8x abundance effect) — too large to model with position-only
    quantization (`peak_sky_coords(quantize_arcmin=...)`), which snaps
    positions *after* finding and therefore never removes a peak.

    Pipeline
    --------
    1. Crop the native (N, N) smoothed map to (N', N') = (N // block) * block
       square by dropping the high-index remainder strip — e.g. at the atlas
       default N=1024, block=12: N//block = 85, N' = 1020, so the last 4 px
       of each side are dropped (1024 = 12*85 + 4). This is a DELIBERATE,
       documented approximation: it breaks the native patch's exact
       periodicity at that edge, but that is the correct behavior to model,
       since the real DES/ACT footprint is not periodic either — averaging
       across the wrapped seam would be a mock-only artifact the data chain
       does not share. The dropped strip is <= block/N ~ 0.4% of a side, and
       is simply excluded from peak finding (no peaks are ever found in it).
       [Deviation note: an earlier draft of this convention assumed a crop
       to (1008, 1008) -> 84 coarse cells (dropping 16 px); that number does
       not correspond to floor(1024/12) (=85, dropping only 4 px) for any
       consistent "crop to the nearest lower multiple of block" rule, so it
       is not implemented — this function uses the generic, self-consistent
       floor-division crop so it works for any (N, block), as exercised by
       the unit tests.]
    2. Block-AVERAGE into (N'/block, N'/block) cells. At block=12 the coarse
       pitch is 12 * (5deg*60/1024) = 3.516', a 2.4% approximation of the
       DATA side's actual mean HEALPix Nside=1024 pixel side (3.435',
       `HEALPIX_NSIDE1024_PIX_ARCMIN`) — the closest integer block factor of
       the native 1024-px grid to that target pitch (1024*0.29296875/3.435
       ~= 11.73, nearest integer 12).
    3. Find strict 8-neighbour local maxima on the coarse map, WITHOUT
       periodic wrap (unlike `find_peaks_flat`) — interior cells only (the
       outermost ring of coarse cells is excluded so every candidate has a
       full, unambiguous 8-neighbour set; this matches a finite, non-periodic
       footprint, where an edge pixel does not have a well-defined "beaten
       all 8 neighbours" test either).

    Parameters
    ----------
    smoothed : (N, N) smoothed κ map (already through `smooth_flat_sky`).
    block : coarse-grid block size in native pixels (default 12).

    Returns
    -------
    ci, cj : int arrays — coarse-grid (row, col) indices of found peaks, each
        satisfying ``1 <= ci/cj <= n_coarse - 2`` (interior only).
    coarse : (n_coarse, n_coarse) float64 block-averaged map, n_coarse =
        (N // block).
    """
    m = np.asarray(smoothed, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"smoothed must be square 2D, got {m.shape}")
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    n = m.shape[0]
    n_coarse = n // block
    if n_coarse < 3:
        raise ValueError(f"block={block} too large for map side {n} "
                         "(need >= 3 coarse cells for an interior)")
    n_crop = n_coarse * block
    cropped = m[:n_crop, :n_crop]                     # drop the high-index remainder
    coarse = cropped.reshape(n_coarse, block, n_coarse, block).mean(axis=(1, 3))

    interior = coarse[1:-1, 1:-1]
    is_max = np.ones(interior.shape, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neigh = coarse[1 + dx: 1 + dx + interior.shape[0],
                          1 + dy: 1 + dy + interior.shape[1]]
            is_max &= interior > neigh
    out = np.zeros(coarse.shape, dtype=bool)
    out[1:-1, 1:-1] = is_max
    ci, cj = np.nonzero(out)
    return ci, cj, coarse


def effective_patch_area_deg2(geom: PatchGeometry = PatchGeometry(),
                              block: int | None = None) -> float:
    """Effective patch area (deg^2) seen by peak finding.

    ``block=None``: the full atlas FOV^2 (e.g. 25.0 for the 5-deg default).
    ``block=<int>``: the area of the cropped square `coarse_grid_peaks` finds
    peaks on, ``(n_crop * fov_deg / npix)^2`` with ``n_crop = (npix // block)
    * block`` — e.g. ``(1008 * 5.0 / 1024)^2`` deg^2 for the atlas default at
    block=12.
    """
    if block is None:
        return geom.fov_deg ** 2
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    n_crop = (geom.npix // block) * block
    return (n_crop * geom.fov_deg / geom.npix) ** 2


def patch_to_enmap(patch: np.ndarray, geom: PatchGeometry = PatchGeometry(),
                   pad_pix: int = DEFAULT_PAD_PIX):
    """Periodically pad a (npix, npix) patch and wrap it into a CAR enmap.

    Returns the enmap, centered on (ra, dec) = (0, 0) at the patch's native
    pixel scale. Peaks found on the *unpadded* patch keep their (i, j); use
    `peak_sky_coords` to get their (ra, dec) on this enmap. The periodic pad
    means thumbnails within `pad_pix` pixels of the patch edge sample the
    periodic continuation — consistent with the periodic mock sky.
    """
    from pixell import enmap

    p = np.asarray(patch, dtype=np.float64)
    if p.shape != (geom.npix, geom.npix):
        raise ValueError(f"patch shape {p.shape} != ({geom.npix}, {geom.npix})")
    padded = np.pad(p, pad_pix, mode="wrap")
    res = np.deg2rad(geom.fov_deg / geom.npix)
    shape, wcs = enmap.geometry(pos=np.array([0.0, 0.0]), shape=padded.shape,
                                res=res, proj="car")
    m = enmap.zeros(shape, wcs)
    m[:] = padded
    return m


def peak_sky_coords(pi: np.ndarray, pj: np.ndarray, emap,
                    pad_pix: int = DEFAULT_PAD_PIX,
                    quantize_arcmin: float | None = None,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(ra_deg, dec_deg) on `emap` for unpadded-patch pixel indices (pi, pj).

    quantize_arcmin: if set, snap positions to a square grid of that pitch —
    the matched-convention model of the DATA side's peak-position
    quantization (B1 peaks live on the Nside=1024 HEALPix grid, mean pixel
    side ~3.435', and thumbnails are extracted at pixel centers; the mock's
    0.293' grid is effectively continuous). Measured effect (2026-07-18
    experiment, bind fiducial): suppresses <Y_CAP> by ~30%/15%/<10% at
    2'/4'/6-8' — a real chain asymmetry, found during the B5 radius
    diagnostic and promoted to a chain convention (see the B5 REPORT
    'ordering' note: the fix models a data-side property, it is not a
    fitted parameter).
    """
    pix = np.stack([np.asarray(pi) + pad_pix, np.asarray(pj) + pad_pix])
    dec, ra = np.rad2deg(emap.pix2sky(pix))
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    if quantize_arcmin is not None:
        q = quantize_arcmin / 60.0
        ra, dec = np.round(ra / q) * q, np.round(dec / q) * q
    return ra, dec


# mean HEALPix pixel side at the B1 catalog resolution (Nside=1024)
HEALPIX_NSIDE1024_PIX_ARCMIN: float = float(
    np.sqrt(4.0 * np.pi / (12 * 1024 ** 2)) * 180.0 / np.pi * 60.0)
