"""Analytic-fixture tests for the WP-A2 filters and geometry (no data files needed)."""

import numpy as np
import pytest

from analysis.paper3a.observables.filters import (
    aperture_sum,
    cap_photometry,
    cap_weight_map,
    disk_weight_map,
    gaussian_beam_convolve,
    require_fits_in_patch,
)
from analysis.paper3a.observables.geometry import FlatLCDM, PatchGeometry


CENTER_65 = (32.0, 32.0)


def test_disk_weight_area_matches_circle():
    w = disk_weight_map((65, 65), CENTER_65, radius_pix=10.0)
    assert w.sum() == pytest.approx(np.pi * 10.0**2, rel=1e-3)


def test_cap_weights_sum_to_zero():
    w = cap_weight_map((65, 65), CENTER_65, theta_d_pix=8.0)
    # disk and sqrt(2) annulus have equal area -> exact compensation
    assert abs(w.sum()) < 1e-2 * np.pi * 8.0**2


def test_cap_kills_uniform_background():
    m = np.full((65, 65), 3.7)
    out = cap_photometry(m, CENTER_65, theta_d_pix=np.array([4.0, 8.0, 12.0]))
    assert np.all(np.abs(out) < 1e-2 * 3.7)


def test_cap_passes_compact_source_flux():
    # all flux well inside theta_d -> CAP = total flux * pixel_area
    m = np.zeros((65, 65))
    m[31:34, 31:34] = 2.0  # 3x3 block, flux 18, extent ~2 px around center
    out = cap_photometry(m, CENTER_65, theta_d_pix=np.array([8.0, 12.0]), pixel_area=0.25)
    assert out == pytest.approx(np.full(2, 18.0 * 0.25), rel=1e-3)


def test_cap_on_gaussian_matches_analytic():
    # CAP on a Gaussian source has the closed form F(2 G(td) - G(sqrt2 td))
    # with G(r) = 1 - exp(-r^2 / (2 sigma^2)). Aperture photometry on a pixel
    # grid carries an O((px/sigma)^2) discretization bias from the hard
    # aperture edge crossing the source gradient, so the check runs at the
    # radius-to-pixel ratios the real operators see: on a BIND composite
    # cutout at z~0.55 the smallest frozen CAP radius (~1 arcmin) is already
    # ~9 px, and ACT itself measures on 0.5-arcmin pixels.
    sigma = 9.0
    yy, xx = np.mgrid[0:385, 0:385].astype(float)
    r2 = (yy - 192.0) ** 2 + (xx - 192.0) ** 2
    m = np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    td = np.array([6.0, 12.0, 24.0])
    out = cap_photometry(m, (192.0, 192.0), td)
    G = lambda r: 1.0 - np.exp(-(r**2) / (2 * sigma**2))
    expected = 2 * G(td) - G(np.sqrt(2.0) * td)
    assert out == pytest.approx(expected, abs=3e-3)


def test_aperture_sum_recovers_uniform_disk_mass():
    m = np.full((65, 65), 5.0)
    got = aperture_sum(m, CENTER_65, radius_pix=7.0)
    assert got == pytest.approx(5.0 * np.pi * 49.0, rel=1e-3)


def test_aperture_raises_when_annulus_leaves_patch():
    with pytest.raises(ValueError, match="larger cutout"):
        require_fits_in_patch((65, 65), CENTER_65, outer_radius_pix=40.0)
    m = np.zeros((65, 65))
    with pytest.raises(ValueError, match="larger cutout"):
        cap_photometry(m, CENTER_65, theta_d_pix=np.array([25.0]))  # sqrt2*25 > 32


def test_beam_conserves_flux_and_widens_source():
    m = np.zeros((129, 129))
    m[64, 64] = 1.0
    sm = gaussian_beam_convolve(m, fwhm_pix=6.0)
    assert sm.sum() == pytest.approx(1.0, rel=1e-6)  # zero-padded, flux conserved
    sigma = 6.0 / np.sqrt(8 * np.log(2))
    assert sm[64, 64] == pytest.approx(1.0 / (2 * np.pi * sigma**2), rel=1e-2)
    assert gaussian_beam_convolve(m, fwhm_pix=0.0) == pytest.approx(m)


def test_beam_then_cap_still_passes_point_source():
    # beam redistributes flux but CAP at theta_d >> fwhm still collects it all
    m = np.zeros((129, 129))
    m[64, 64] = 1.0
    sm = gaussian_beam_convolve(m, fwhm_pix=3.0)
    out = cap_photometry(sm, (64.0, 64.0), theta_d_pix=np.array([20.0]))
    assert out[0] == pytest.approx(1.0, rel=5e-3)


# --- geometry -------------------------------------------------------------


def test_comoving_distance_against_astropy():
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    cosmo = FlatLCDM(omega_m=0.3089, h=0.6774)
    ref = astropy_cosmo.FlatLambdaCDM(H0=67.74, Om0=0.3089)
    for z in (0.1, 0.55, 1.0, 2.0):
        ours = cosmo.comoving_distance_hmpc(z)
        theirs = ref.comoving_distance(z).value * 0.6774  # Mpc -> h^-1 Mpc
        assert ours == pytest.approx(theirs, rel=5e-4), f"z={z}"


def test_arcmin_roundtrip():
    cosmo = FlatLCDM()
    L = cosmo.arcmin_to_comoving_hmpc(3.0, z=0.55)
    assert cosmo.comoving_hmpc_to_arcmin(L, z=0.55) == pytest.approx(3.0, rel=1e-12)


def test_patch_geometry_core_patch_is_too_small_for_lrg_cap():
    # The WP-A2 sizing finding, locked in as a test: at z=0.55 the 6.25 Mpc/h
    # core patch subtends < 2 * sqrt(2) * 6 arcmin, so the largest kSZ CAP
    # radii cannot be measured on it.
    geom = PatchGeometry(pixel_mpch=6.25 / 128, z=0.55)
    patch_extent_arcmin = 128 * geom.arcmin_per_pixel()
    assert patch_extent_arcmin < 2 * np.sqrt(2.0) * 6.0


def test_proper_pixel_area_has_1plusz_squared():
    g0 = PatchGeometry(pixel_mpch=0.05, z=0.0)
    g1 = PatchGeometry(pixel_mpch=0.05, z=1.0)
    assert g0.pixel_area_proper_m2() / g1.pixel_area_proper_m2() == pytest.approx(4.0)
