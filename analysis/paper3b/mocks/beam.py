"""ACT beam applied to mock Compton-y maps (WP-B4).

Produced by Project B WP-B4 (forward model; see bind-paper3-plans, private
repo). Flat-sky path is pure numpy/scipy.

Published spec
--------------
The ACT DR6 component-separated Compton-y map — **Coulton, Madhavacheril,
Duivenvoorden & Hill 2024, "The Atacama Cosmology Telescope: High-resolution
component-separated maps ...", arXiv:2307.01258** — is an arcminute-resolution
y map. The released map is provided on ~0.5' pixels with information to
ℓ ≲ 17000; its effective Gaussian beam is ≈ 1.6 arcmin FWHM
(``ACT_DR6_YMAP_FWHM_ARCMIN``, a documented default). For map-level mocks we
apply a Gaussian beam of *configurable* FWHM to the mock y map to match the
data's angular resolution before the peak-located y stack.

Convention
----------
The beam is specified as an FWHM (the natural CMB/SZ beam unit); the Gaussian
σ is ``FWHM / sqrt(8 ln 2)``. Applied as a flat-sky convolution
(``scipy.ndimage.gaussian_filter``), consistent with ``smoothing.py``. Beam and
κ-smoothing are *separate* operations: the ACT beam matches the y map to the
instrument resolution, while the κ smoothing (``smoothing.py``) is the
peak-definition kernel applied to the convergence map.

TODO (flag in B/wp4 REPORT.md): ``ACT_DR6_YMAP_FWHM_ARCMIN = 1.6`` is a
documented approximate default. For the final mock grid, load the actual ACT
DR6 y-map beam transfer function / released beam file (arXiv:2307.01258 data
products) and apply it in harmonic space rather than a single-FWHM Gaussian —
``apply_beam`` takes the FWHM explicitly so no guessed value is baked in
silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Documented approximate default (Coulton et al. 2024, arXiv:2307.01258). Never
# used silently — apply_beam takes fwhm_arcmin as an explicit config input.
ACT_DR6_YMAP_FWHM_ARCMIN: float = 1.6

_FWHM_PER_SIGMA = float(np.sqrt(8.0 * np.log(2.0)))


@dataclass(frozen=True)
class BeamConfig:
    """Gaussian beam parameters for a flat-sky mock y map.

    Attributes
    ----------
    fwhm_arcmin : beam FWHM in arcmin. ``<= 0`` is a no-op (no beam).
    arcmin_per_pixel : map pixel scale (arcmin/pixel).
    mode : ``scipy.ndimage.gaussian_filter`` boundary mode. Default ``"wrap"``
        (periodic mock patch); use ``"constant"`` for a non-periodic cutout.
    """

    fwhm_arcmin: float
    arcmin_per_pixel: float
    mode: str = "wrap"

    def sigma_arcmin(self) -> float:
        return self.fwhm_arcmin / _FWHM_PER_SIGMA

    def sigma_pix(self) -> float:
        if self.arcmin_per_pixel <= 0:
            raise ValueError(f"arcmin_per_pixel must be > 0, got {self.arcmin_per_pixel}")
        return self.sigma_arcmin() / self.arcmin_per_pixel


def apply_beam(y_map, config: BeamConfig) -> np.ndarray:
    """Convolve a mock y map with a Gaussian beam of ``config.fwhm_arcmin``.

    ``fwhm_arcmin <= 0`` returns a float64 copy unchanged. The discrete kernel
    is normalized to sum 1, so total flux (∫ y) is conserved (exactly for
    ``mode="wrap"``/``"reflect"``; up to 4σ-truncation for interior sources
    with ``mode="constant"``), while a point source is spread into a Gaussian of
    the specified FWHM.
    """
    from scipy.ndimage import gaussian_filter

    m = np.asarray(y_map, dtype=float)
    if config.fwhm_arcmin <= 0:
        return m.copy()
    return gaussian_filter(m, sigma=config.sigma_pix(), mode=config.mode)
