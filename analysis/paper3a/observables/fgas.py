"""f_gas operator: painted gas maps -> f_gas(<R500c) (WP-A2 task 2).

The eROSITA vector (eRASS1 primary catalog, Bulbul et al. 2024) reports
FGAS500 = M_gas(<R500c, 3D sphere) / M500c. A projected map only measures
the *cylindrical* aperture mass, which includes line-of-sight material
outside the sphere, so the operator is split into two explicit pieces:

1. `fgas_cylindrical` — exact on the map: gas mass in the projected R500c
   disk over M500c (the mass-calibration mass, an *input*, matching how the
   observation divides by a scaling-relation/GGL mass rather than a summed
   pixel mass).
2. `CylToSphCorrection` — the multiplicative sphere/cylinder factor,
   *measured on TNG300-hydro truth* (Popeye half of WP-A2, task 5), never
   assumed. Until that calibration exists the correction carries
   ``factor=None`` and applying it raises — an uncalibrated f_gas cannot
   silently enter the A3 gate.

The projection depth of the map is part of the correction's identity: a
6.25 h^-1 Mpc patch, a 50 h^-1 Mpc composite, and a full TNG300 projection
pick up different amounts of unrelated line-of-sight gas, so a factor
calibrated at one depth must not be applied at another (enforced by
`depth_hmpc` matching).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import aperture_sum
from .geometry import PatchGeometry


@dataclass(frozen=True)
class CylToSphCorrection:
    """Sphere/cylinder aperture-mass ratio calibrated on truth.

    factor : multiplicative correction M_sphere/M_cylinder; None = not yet
        calibrated (the Popeye validation fills this in).
    depth_hmpc : projection depth (comoving, h^-1 Mpc) of the maps the factor
        was measured on.
    scatter : per-halo rms of the factor around its mean, carried into
        Sigma_model by WP-A5.
    provenance : where the calibration came from (script + truth inputs).
    """

    factor: float | None = None
    depth_hmpc: float | None = None
    scatter: float | None = None
    provenance: str = "UNCALIBRATED — run the WP-A2 Popeye truth validation"

    def apply(self, fgas_cyl: np.ndarray, map_depth_hmpc: float) -> np.ndarray:
        if self.factor is None:
            raise ValueError(
                "CylToSphCorrection is uncalibrated (factor=None) — the WP-A2 "
                "truth validation on Popeye must measure the sphere/cylinder "
                "ratio before any f_gas prediction is compared to data"
            )
        if self.depth_hmpc is not None and not np.isclose(map_depth_hmpc, self.depth_hmpc, rtol=0.05):
            raise ValueError(
                f"correction calibrated at projection depth {self.depth_hmpc} h^-1 Mpc "
                f"but the map was projected over {map_depth_hmpc} h^-1 Mpc — recalibrate "
                "rather than reuse across depths"
            )
        return np.asarray(fgas_cyl, dtype=float) * self.factor


def aperture_gas_mass_msunh(
    sigma_gas_msunh: np.ndarray,
    geometry: PatchGeometry,
    r500_hmpc: float,
    center: tuple[float, float] | None = None,
) -> float:
    """Cylindrical gas mass [Msun/h] inside the projected R500c disk."""
    if center is None:
        shape = np.asarray(sigma_gas_msunh).shape
        center = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
    return aperture_sum(sigma_gas_msunh, center, r500_hmpc / geometry.pixel_mpch)


def fgas_cylindrical(
    sigma_gas_msunh: np.ndarray,
    geometry: PatchGeometry,
    r500_hmpc: float,
    m500_msunh: float,
    center: tuple[float, float] | None = None,
) -> float:
    """f_gas^cyl = M_gas(cylinder, <R500c projected) / M500c.

    m500_msunh is the halo's M500c from the catalog (or, in a full forward
    model, from the same mass-observable relation the data uses) — the
    denominator is *not* re-summed from pixels, mirroring the observational
    definition where the mass comes from an external calibration.
    """
    if m500_msunh <= 0:
        raise ValueError(f"m500_msunh must be > 0, got {m500_msunh}")
    return aperture_gas_mass_msunh(sigma_gas_msunh, geometry, r500_hmpc, center) / m500_msunh
