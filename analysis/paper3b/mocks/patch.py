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
                    pad_pix: int = DEFAULT_PAD_PIX) -> tuple[np.ndarray, np.ndarray]:
    """(ra_deg, dec_deg) on `emap` for unpadded-patch pixel indices (pi, pj)."""
    pix = np.stack([np.asarray(pi) + pad_pix, np.asarray(pj) + pad_pix])
    dec, ra = np.rad2deg(emap.pix2sky(pix))
    return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)
