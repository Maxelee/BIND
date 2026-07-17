"""WP-B2 compensated aperture photometry (CAP) on flat thumbnails.

`Y_CAP(θ_d) = Σ_disk y·dA − (A_disk/A_ann)·Σ_ann y·dA` with disk r ≤ θ_d and
annulus θ_d < r ≤ √2·θ_d, pixelized areas counted exactly so a UNIFORM map
filters to exactly 0 (same exact-compensation convention as the A-side
`observables.filters`, re-implemented here so the B chain stays importable
without the A package). Output units: y × arcmin².

Pure numpy on (..., ny, nx) thumbnail arrays — identical for data and B4
mocks (MEASUREMENT_SPEC: no data-only branches).
"""

from __future__ import annotations

import numpy as np

SQRT2 = np.sqrt(2.0)


def cap_masks(ny: int, nx: int, theta_d_arcmin: float, res_arcmin: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """(disk, annulus) bool masks about the thumbnail center."""
    y, x = np.indices((ny, nx))
    r = np.hypot(y - (ny - 1) / 2.0, x - (nx - 1) / 2.0) * res_arcmin
    disk = r <= theta_d_arcmin
    ann = (r > theta_d_arcmin) & (r <= SQRT2 * theta_d_arcmin)
    if not ann.any():
        raise ValueError(f"CAP annulus empty: theta_d={theta_d_arcmin}' at res={res_arcmin}'")
    return disk, ann


def cap_filter(thumbs: np.ndarray, theta_d_arcmin: float, res_arcmin: float) -> np.ndarray:
    """Per-thumbnail CAP Y [y·arcmin²]; thumbs is (..., ny, nx)."""
    t = np.asarray(thumbs, dtype=np.float64)
    disk, ann = cap_masks(t.shape[-2], t.shape[-1], theta_d_arcmin, res_arcmin)
    pix_area = res_arcmin ** 2
    s_disk = t[..., disk].sum(axis=-1)
    s_ann = t[..., ann].sum(axis=-1)
    # exact pixelized-area compensation: uniform map -> exactly 0
    return (s_disk - (disk.sum() / ann.sum()) * s_ann) * pix_area


def cap_filter_multi(thumbs: np.ndarray, radii_arcmin, res_arcmin: float) -> np.ndarray:
    """(..., n_radii) CAP Y at each radius in ``radii_arcmin``."""
    return np.stack([cap_filter(thumbs, r, res_arcmin) for r in radii_arcmin], axis=-1)
