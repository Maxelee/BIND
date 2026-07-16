"""Y-M operator: painted compton_y patches -> Y500 and the Y-M relation (WP-A2 task 3).

The BIND thermo channel `compton_y` is dimensionless y per pixel (line-of-
sight sum normalized by pixel area — see bind.inference.pipeline.
project_thermo_fullbox). The integrated SZ signal in a projected aperture is
then Y_cyl(<R) = sum(y_i) * A_pix, a *cylindrical* quantity with the same
sphere/cylinder caveat as f_gas — reuse `fgas.CylToSphCorrection` for the
truth-calibrated conversion when a spherical-Y_500 data point demands it.

Y is returned in proper Mpc^2 (the common cluster convention; multiply by
E(z)^{-2/3} etc. downstream as each paper's scaling demands) and optionally
in arcmin^2 for map-level comparisons.

Since BIND is generative, the relation fit deliberately returns the
*intrinsic scatter* alongside slope/normalization — Paper 2 validated
Y500-M against TNG at the halo level, and the scatter is a first-class
prediction for A4's emulator, not a nuisance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import aperture_sum
from .geometry import PatchGeometry


def y_aperture_mpc2(
    y_map: np.ndarray,
    geometry: PatchGeometry,
    r_hmpc: float,
    center: tuple[float, float] | None = None,
) -> float:
    """Cylindrical integrated Compton parameter Y(<R) in proper Mpc^2."""
    if center is None:
        shape = np.asarray(y_map).shape
        center = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
    y_sum = aperture_sum(y_map, center, r_hmpc / geometry.pixel_mpch)
    pixel_area_proper_mpc2 = (geometry.pixel_mpch / (1.0 + geometry.z) / geometry.cosmology.h) ** 2
    return y_sum * pixel_area_proper_mpc2


@dataclass(frozen=True)
class YMRelation:
    """log10 Y = alpha * (log10 M - pivot) + beta, with intrinsic scatter."""

    alpha: float
    beta: float
    pivot_log10m: float
    scatter_dex: float
    n_halos: int

    def predict_log10_y(self, log10_m):
        return self.alpha * (np.asarray(log10_m, dtype=float) - self.pivot_log10m) + self.beta


def fit_ym_relation(
    m500_msunh: np.ndarray,
    y500_mpc2: np.ndarray,
    pivot_log10m: float = 14.0,
) -> YMRelation:
    """Least-squares power law in log-log space + rms intrinsic scatter.

    Halos with non-positive Y (possible in low-mass painted patches after
    clipping) are dropped rather than clamped; their count is implicit in
    n_halos. The scatter is the raw rms residual in dex — measurement noise
    is zero here (these are model predictions), so it is genuinely the
    model's intrinsic Y-at-fixed-M scatter.
    """
    m = np.asarray(m500_msunh, dtype=float)
    y = np.asarray(y500_mpc2, dtype=float)
    good = (m > 0) & (y > 0)
    if good.sum() < 3:
        raise ValueError(f"need >= 3 halos with positive M and Y, got {int(good.sum())}")
    logm = np.log10(m[good]) - pivot_log10m
    logy = np.log10(y[good])
    alpha, beta = np.polyfit(logm, logy, 1)
    resid = logy - (alpha * logm + beta)
    return YMRelation(
        alpha=float(alpha),
        beta=float(beta),
        pivot_log10m=pivot_log10m,
        scatter_dex=float(np.std(resid)),
        n_halos=int(good.sum()),
    )
