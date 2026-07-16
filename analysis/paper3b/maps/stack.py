"""WP-B1 task 5 — cluster stacking on the native-CAR ACT y map (sanity anchor).

Stacks 2D thumbnails (``pixell.reproject.thumbnails``, native 0.5' resolution,
gnomonic-like local projection) of the y map at catalog positions, with an
optional per-object weight, and reduces to an azimuthally averaged radial
profile. Errors via bootstrap over objects. A random-position null (uniform
points where the apodized mask is fully interior) validates the zero level.

The y map stays at NATIVE resolution/projection (plan task 3: "keep y at
native for profile work"); nothing here touches HEALPix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARCMIN = np.pi / (180.0 * 60.0)
DEFAULT_R_MAX_ARCMIN = 15.0
DEFAULT_RES_ARCMIN = 0.5


def extract_thumbnails(imap, ra_deg: np.ndarray, dec_deg: np.ndarray,
                       r_max_arcmin: float = DEFAULT_R_MAX_ARCMIN,
                       res_arcmin: float = DEFAULT_RES_ARCMIN):
    """(N, ny, nx) thumbnails centered on (ra, dec) [deg], side ~2*r_max."""
    from pixell import reproject

    coords = np.deg2rad(np.column_stack([dec_deg, ra_deg]))  # pixell wants (dec, ra)
    return reproject.thumbnails(imap, coords, r=r_max_arcmin * ARCMIN,
                                res=res_arcmin * ARCMIN)


def stack_thumbnails(thumbs, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted mean stack over the object axis."""
    t = np.asarray(thumbs, dtype=np.float64)
    if weights is None:
        return t.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    return np.tensordot(w, t, axes=(0, 0)) / w.sum()


def radial_profile(stacked: np.ndarray, res_arcmin: float = DEFAULT_RES_ARCMIN,
                   r_bin_arcmin: float = 1.0,
                   r_max_arcmin: float = DEFAULT_R_MAX_ARCMIN
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(bin centers [arcmin], mean per annulus) about the thumbnail center."""
    ny, nx = stacked.shape
    y, x = np.indices((ny, nx))
    r = np.hypot(y - (ny - 1) / 2.0, x - (nx - 1) / 2.0) * res_arcmin
    edges = np.arange(0.0, r_max_arcmin + r_bin_arcmin, r_bin_arcmin)
    idx = np.digitize(r.ravel(), edges) - 1
    vals = stacked.ravel()
    nbin = len(edges) - 1
    prof = np.array([vals[idx == i].mean() if (idx == i).any() else np.nan
                     for i in range(nbin)])
    return 0.5 * (edges[:-1] + edges[1:]), prof


@dataclass
class StackResult:
    """Stack + profile + bootstrap errors for one catalog selection."""

    stacked: np.ndarray          # (ny, nx) mean thumbnail
    r_arcmin: np.ndarray         # (nbin,) profile bin centers
    profile: np.ndarray          # (nbin,) azimuthal mean
    profile_err: np.ndarray      # (nbin,) bootstrap std over objects
    n_obj: int

    @property
    def central_y(self) -> float:
        return float(self.profile[0])


def stack_catalog(imap, ra_deg: np.ndarray, dec_deg: np.ndarray,
                  weights: np.ndarray | None = None,
                  r_max_arcmin: float = DEFAULT_R_MAX_ARCMIN,
                  res_arcmin: float = DEFAULT_RES_ARCMIN,
                  r_bin_arcmin: float = 1.0,
                  n_boot: int = 200, seed: int = 0) -> StackResult:
    """Thumbnails -> stack -> profile, with bootstrap-over-objects errors."""
    thumbs = np.asarray(extract_thumbnails(imap, ra_deg, dec_deg,
                                           r_max_arcmin, res_arcmin), dtype=np.float64)
    stacked = stack_thumbnails(thumbs, weights)
    r, prof = radial_profile(stacked, res_arcmin, r_bin_arcmin, r_max_arcmin)

    rng = np.random.default_rng(seed)
    n = len(thumbs)
    boots = np.empty((n_boot, len(prof)))
    for b in range(n_boot):
        sel = rng.integers(0, n, n)
        w = None if weights is None else np.asarray(weights)[sel]
        _, boots[b] = radial_profile(stack_thumbnails(thumbs[sel], w),
                                     res_arcmin, r_bin_arcmin, r_max_arcmin)
    return StackResult(stacked=stacked, r_arcmin=r, profile=prof,
                       profile_err=boots.std(axis=0), n_obj=n)


def random_positions_in_mask(mask, n: int, interior_threshold: float = 0.99,
                             seed: int = 0, dec_range_deg: tuple[float, float] | None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
    """(ra_deg, dec_deg) uniform on the sphere where ``mask`` >= threshold.

    Rejection sampling: uniform in (RA, sin dec) over the requested dec band
    (default: the mask's full dec coverage), kept where the CAR mask samples
    fully interior. For the null stack.
    """
    from pixell import enmap

    rng = np.random.default_rng(seed)
    if dec_range_deg is None:
        box = np.rad2deg(mask.box())          # [[dec0, ra0], [dec1, ra1]]
        dec_lo, dec_hi = float(min(box[0][0], box[1][0])), float(max(box[0][0], box[1][0]))
    else:
        dec_lo, dec_hi = dec_range_deg
    ras, decs = [], []
    got = 0
    while got < n:
        m = max(4 * (n - got), 1024)
        ra = rng.uniform(0.0, 360.0, m)
        dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(dec_lo)),
                                               np.sin(np.radians(dec_hi)), m)))
        vals = mask.at(np.deg2rad(np.stack([dec, ra])), order=1)
        ok = vals >= interior_threshold
        ras.append(ra[ok]); decs.append(dec[ok])
        got += int(ok.sum())
    return np.concatenate(ras)[:n], np.concatenate(decs)[:n]
