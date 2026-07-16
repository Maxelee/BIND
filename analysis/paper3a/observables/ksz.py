"""kSZ operator: painted gas surface density -> CAP tau profile -> T_kSZ (WP-A2 task 1).

Chain, matching the measurement convention of the frozen kSZ vectors
(Schaan et al. 2021 CAP filtering as used by Qu et al. 2026 / Ried Guachalla
et al. 2025 / Hadzhiyska et al. 2024, 2026):

1. gas surface density Sigma_gas [Msun/h per pixel] (BIND `Gas` channel of a
   painted patch, a composite cutout, or a truth projection)
2. -> Thomson optical depth per pixel, tau = sigma_T * N_e / A_proper,
   assuming fully-ionized primordial-composition gas (x_e = 1.158; no
   per-pixel ionization channel exists in the painted output — the truth
   validation on Popeye quantifies the error from cold/star-forming gas)
3. -> optional Gaussian beam convolution (ACT f090/f150 as used by the stack)
4. -> CAP photometry at the data vector's disk radii -> tau_CAP(theta_d)
   [dimensionless x arcmin^2]
5. -> T_kSZ(theta_d) = T_CMB * (v_rms/c) * tau_CAP  [muK arcmin^2], the
   stacked-amplitude normalization used by the ACT x DESI pipeline papers.

Steps 1-4 are exact given the map; step 5's velocity normalization (v_rms
and the reconstruction transfer function) is a per-paper convention that A5
must confirm against each source before likelihood use — it enters as an
explicit config value here, never a hidden default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    C_KM_S,
    MSUN_KG,
    M_PROTON_KG,
    SIGMA_T_M2,
    T_CMB_UK,
    X_E_FULLY_IONIZED,
    X_H,
)
from .filters import cap_photometry, gaussian_beam_convolve
from .geometry import PatchGeometry


def tau_map_from_gas(
    sigma_gas_msunh: np.ndarray,
    geometry: PatchGeometry,
    x_e: float = X_E_FULLY_IONIZED,
) -> np.ndarray:
    """Thomson optical depth per pixel from a gas surface-mass map.

    tau = sigma_T * (electrons per pixel) / (proper pixel area)
    with N_e = x_e * X_H * M_gas / m_p. Masses arrive in Msun/h (BIND map
    convention); the h and the comoving->proper (1+z)^2 factor live in
    `geometry`, so the same painted patch gives the correct tau wherever it
    is placed along the line of sight.
    """
    m_gas_kg = np.asarray(sigma_gas_msunh, dtype=np.float64) / geometry.cosmology.h * MSUN_KG
    n_e = x_e * X_H * m_gas_kg / M_PROTON_KG
    return SIGMA_T_M2 * n_e / geometry.pixel_area_proper_m2()


@dataclass(frozen=True)
class KSZOperatorConfig:
    """One kSZ data vector's measurement convention (WP-A1 freeze counterpart).

    radii_arcmin : CAP disk radii — take these from the A1 loader's `bins`
        (e.g. ``load_ksz_qu2026_lrg_fiducial().bins``), never retype them.
    beam_fwhm_arcmin : effective Gaussian beam of the stacked CMB map.
        ACT DR6 f090 ~ 2.1', f150 ~ 1.3' — nominal values, flagged UNVERIFIED
        in the A1 freeze until the per-paper audit closes; 0 disables.
    z_eff : redshift the painted patch is placed at (per-bin effective
        redshift of the galaxy sample).
    v_rms_over_c : RMS reconstructed radial velocity over c, for the
        tau -> T_kSZ normalization (step 5 of the module chain). None means
        "return tau_CAP only" — forces A5 to make the per-paper choice
        explicitly.
    x_e : electron abundance assumed for the painted gas.
    """

    radii_arcmin: np.ndarray
    beam_fwhm_arcmin: float
    z_eff: float
    v_rms_over_c: float | None = None
    x_e: float = X_E_FULLY_IONIZED

    def __post_init__(self):
        object.__setattr__(self, "radii_arcmin", np.atleast_1d(np.asarray(self.radii_arcmin, dtype=float)))


def ksz_cap_profile(
    sigma_gas_msunh: np.ndarray,
    geometry: PatchGeometry,
    config: KSZOperatorConfig,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """CAP tau profile [arcmin^2] (or T_kSZ [muK arcmin^2]) for one map.

    center defaults to the map's geometric center; pass an offset center to
    realize miscentering (see `mock_sample`). If the largest annulus does not
    fit in the map this raises (see `filters.require_fits_in_patch`) — use a
    larger composite cutout, not a smaller aperture.

    Returns T_kSZ when ``config.v_rms_over_c`` is set, else tau_CAP.
    """
    tau = tau_map_from_gas(sigma_gas_msunh, geometry, x_e=config.x_e)
    fwhm_pix = config.beam_fwhm_arcmin / geometry.arcmin_per_pixel()
    tau = gaussian_beam_convolve(tau, fwhm_pix)
    if center is None:
        center = ((tau.shape[0] - 1) / 2.0, (tau.shape[1] - 1) / 2.0)
    theta_d_pix = geometry.arcmin_to_pixels(config.radii_arcmin)
    tau_cap = cap_photometry(tau, center, theta_d_pix, pixel_area=geometry.pixel_area_arcmin2())
    if config.v_rms_over_c is None:
        return tau_cap
    return T_CMB_UK * config.v_rms_over_c * tau_cap


def minimum_cutout_extent_hmpc(config: KSZOperatorConfig, geometry: PatchGeometry) -> float:
    """Comoving side length a cutout must have for the largest CAP annulus.

    The WP-A2 sizing finding, as a function: sqrt(2) x max radius plus beam
    support, doubled. Compare against 6.25 h^-1 Mpc to see why bare core
    patches cannot carry the large-radius kSZ bins at z >~ 0.4.
    """
    theta_max = float(np.max(config.radii_arcmin))
    support_arcmin = np.sqrt(2.0) * theta_max + 3.0 * (config.beam_fwhm_arcmin or 0.0)
    return 2.0 * float(geometry.cosmology.arcmin_to_comoving_hmpc(support_arcmin, config.z_eff))
