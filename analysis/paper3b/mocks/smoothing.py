"""Gaussian smoothing — the frozen PEAK_DEFINITION kernel set (WP-B4).

Produced by Project B WP-B4 (forward model; see bind-paper3-plans, private
repo). Flat-sky path is pure numpy/scipy; the curved-sky path lazily imports
healpy so this module (and its tests) stay import-light.

Frozen spec (B1 PEAK_DEFINITION.md §1)
--------------------------------------
- Kernel: Gaussian.
- Fiducial scale: σ = 2.0 arcmin (``FIDUCIAL_SMOOTHING_ARCMIN``).
- Robustness set: σ ∈ {1, 2, 5, 8} arcmin (``PAPER2_SMOOTHING_ARCMIN``) — the
  Paper-2 ``_multi`` variant carried into B1's peak catalog and B4's mock grid.
- Flat-sky mock detail: ``scipy.ndimage.gaussian_filter(..., mode="wrap")`` on
  a periodic 5°-FOV patch (matches the Paper-2 atlas; ``mode`` is configurable
  so non-periodic mock cutouts can use ``"constant"``/``"reflect"``).
- Curved-sky/survey translation: ``healpy.sphtfunc.smoothing(map, sigma=...)``
  in harmonic space. PEAK_DEFINITION note: healpy's ``sigma`` keyword *is* the
  Gaussian σ (radians), matching the flat-sky σ directly — only an
  arcmin→radian unit conversion, no factor. The harmonized mask is applied
  before smoothing and the mask-exclusion zone after (that masking is B1's job,
  not this kernel's — see ``smooth_curved_sky`` docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIDUCIAL_SMOOTHING_ARCMIN: float = 2.0
PAPER2_SMOOTHING_ARCMIN: tuple[float, float, float, float] = (1.0, 2.0, 5.0, 8.0)

_FWHM_PER_SIGMA = float(np.sqrt(8.0 * np.log(2.0)))


@dataclass(frozen=True)
class SmoothingConfig:
    """Flat-sky Gaussian smoothing parameters.

    Attributes
    ----------
    sigma_arcmin : Gaussian σ in arcmin (not FWHM). ``<= 0`` is a no-op.
    arcmin_per_pixel : map pixel scale (arcmin/pixel); e.g.
        ``PatchGeometry.arcmin_per_pixel()`` or FOV_deg*60/npix.
    mode : ``scipy.ndimage.gaussian_filter`` boundary mode. Default ``"wrap"``
        per PEAK_DEFINITION (periodic 5° patch).
    """

    sigma_arcmin: float
    arcmin_per_pixel: float
    mode: str = "wrap"

    def sigma_pix(self) -> float:
        if self.arcmin_per_pixel <= 0:
            raise ValueError(f"arcmin_per_pixel must be > 0, got {self.arcmin_per_pixel}")
        return self.sigma_arcmin / self.arcmin_per_pixel

    def fwhm_arcmin(self) -> float:
        return self.sigma_arcmin * _FWHM_PER_SIGMA


def smooth_flat_sky(map2d, config: SmoothingConfig) -> np.ndarray:
    """Gaussian-smooth a 2D patch with ``scipy.ndimage.gaussian_filter``.

    ``sigma_arcmin <= 0`` returns a float64 copy unchanged (so scale sweeps can
    include a "no smoothing" entry without branching at the call site). The
    discrete Gaussian kernel is normalized to sum 1, so the smoothing conserves
    total flux (exactly for ``mode="wrap"``/``"reflect"``; up to 4σ-truncation
    for interior sources with ``mode="constant"``).
    """
    from scipy.ndimage import gaussian_filter

    m = np.asarray(map2d, dtype=float)
    s = config.sigma_pix()
    if s <= 0:
        return m.copy()
    return gaussian_filter(m, sigma=s, mode=config.mode)


def smooth_curved_sky(hp_map, sigma_arcmin: float, **healpy_kwargs) -> np.ndarray:
    """Curved-sky Gaussian smoothing via ``healpy.sphtfunc.smoothing`` (RING map).

    Thin wrapper implementing the PEAK_DEFINITION §1 curved-sky translation:
    healpy's ``sigma`` keyword is the Gaussian σ in *radians*, matching the
    flat-sky σ directly (only arcmin→radian). healpy is imported lazily so the
    flat-sky path and the unit tests do not require it.

    NOTE (B1 responsibility, not done here): the harmonized common mask must be
    applied *before* calling this, and the mask-proximity exclusion zone
    (PEAK_DEFINITION §4) *after*. This function is only the kernel.
    """
    import healpy as hp

    sigma_rad = np.deg2rad(sigma_arcmin / 60.0)
    return hp.sphtfunc.smoothing(np.asarray(hp_map, dtype=float), sigma=sigma_rad, **healpy_kwargs)
