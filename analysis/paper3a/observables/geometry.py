"""Angular <-> comoving geometry for placing BIND patches on the sky (WP-A2).

Every observational vector in the WP-A1 freeze is binned in *angular* units
(CAP radius in arcmin) or in a mass aperture defined at the halo's redshift
(R500c), while BIND patches live on a comoving h^-1 Mpc pixel grid. This
module owns that translation and nothing else.

Pure numpy flat-LambdaCDM: massless-neutrino, radiation-free comoving
distance by trapezoidal integration (validated against astropy in the tests
to <0.05%, well below any WP-A2 tolerance). Cosmology is an explicit input
everywhere: SB35 varies Omega_m and h, so the gate/emulator code must be
able to pass per-design-point cosmologies rather than inherit a hidden
fiducial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import C_KM_S, TNG_H, TNG_OMEGA_M

ARCMIN_IN_RAD = np.pi / (180.0 * 60.0)

# numpy renamed trapz -> trapezoid in 2.0; rusty (numpy 1.x) and Popeye
# (numpy 2.4) venvs sit on opposite sides of the rename.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass(frozen=True)
class FlatLCDM:
    """Minimal flat LambdaCDM background, distances in comoving h^-1 Mpc."""

    omega_m: float = TNG_OMEGA_M
    h: float = TNG_H

    def efunc(self, z):
        """H(z)/H0."""
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.omega_m * (1.0 + z) ** 3 + (1.0 - self.omega_m))

    def comoving_distance_hmpc(self, z: float, n_steps: int = 4096) -> float:
        """Line-of-sight comoving distance D_C(z) in h^-1 Mpc."""
        if z < 0:
            raise ValueError(f"z must be >= 0, got {z}")
        if z == 0:
            return 0.0
        zs = np.linspace(0.0, z, n_steps)
        integrand = 1.0 / self.efunc(zs)
        # c/H0 in h^-1 Mpc is h-independent: 2997.92 h^-1 Mpc.
        return float((C_KM_S / 100.0) * _trapezoid(integrand, zs))

    def arcmin_to_comoving_hmpc(self, theta_arcmin, z: float):
        """Transverse comoving size (h^-1 Mpc) subtended by theta at redshift z.

        Flat universe: transverse comoving distance == D_C, so
        L = D_C(z) * theta[rad]. Valid for the small angles used here.
        """
        theta = np.asarray(theta_arcmin, dtype=float)
        return self.comoving_distance_hmpc(z) * theta * ARCMIN_IN_RAD

    def comoving_hmpc_to_arcmin(self, length_hmpc, z: float):
        """Inverse of :meth:`arcmin_to_comoving_hmpc`."""
        length = np.asarray(length_hmpc, dtype=float)
        dc = self.comoving_distance_hmpc(z)
        if dc == 0.0:
            raise ValueError("z=0 has no angular scale (D_C=0)")
        return length / (dc * ARCMIN_IN_RAD)


@dataclass(frozen=True)
class PatchGeometry:
    """Pixel bookkeeping for one 2D map (painted patch, composite cutout, or truth).

    Attributes
    ----------
    pixel_mpch : comoving pixel side in h^-1 Mpc (e.g. 6.25/128 for the BIND
        core patch, 50/1024 for a full CAMELS box projection).
    z : redshift the map is placed at, for angular conversions and
        comoving->proper factors. z is the *observation* placement redshift
        (an operator choice), not necessarily the snapshot redshift — the two
        must agree for thermo channels (SHARED_CONTEXT caveat 6).
    cosmology : background used for the angular conversions.
    """

    pixel_mpch: float
    z: float
    cosmology: FlatLCDM = FlatLCDM()

    def arcmin_per_pixel(self) -> float:
        return float(self.cosmology.comoving_hmpc_to_arcmin(self.pixel_mpch, self.z))

    def arcmin_to_pixels(self, theta_arcmin):
        """Angular radius -> radius in pixels on this map."""
        return np.asarray(theta_arcmin, dtype=float) / self.arcmin_per_pixel()

    def pixel_area_arcmin2(self) -> float:
        return self.arcmin_per_pixel() ** 2

    def pixel_side_proper_m(self) -> float:
        """Proper pixel side in metres: comoving/(1+z), h^-1 Mpc -> Mpc -> m."""
        from .constants import MPC_IN_M

        return self.pixel_mpch / (1.0 + self.z) / self.cosmology.h * MPC_IN_M

    def pixel_area_proper_m2(self) -> float:
        return self.pixel_side_proper_m() ** 2
