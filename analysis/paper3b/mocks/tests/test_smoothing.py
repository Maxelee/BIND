"""Analytic-fixture tests for WP-B4 Gaussian smoothing (delta -> Gaussian)."""

import numpy as np
import pytest

from analysis.paper3b.mocks.smoothing import (
    FIDUCIAL_SMOOTHING_ARCMIN,
    PAPER2_SMOOTHING_ARCMIN,
    SmoothingConfig,
    smooth_curved_sky,
    smooth_flat_sky,
)


def _measured_sigma_pix(m):
    """Isotropic Gaussian sigma (pixels) from the map's second moment."""
    ny, nx = m.shape
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    tot = m.sum()
    xc = (m * x).sum() / tot
    yc = (m * y).sum() / tot
    vx = (m * (x - xc) ** 2).sum() / tot
    vy = (m * (y - yc) ** 2).sum() / tot
    return np.sqrt(0.5 * (vx + vy))


def test_delta_becomes_gaussian_of_correct_width():
    n = 201
    m = np.zeros((n, n))
    m[100, 100] = 1.0
    # 0.5 arcmin/pixel, sigma=4 arcmin -> sigma_pix = 8.
    cfg = SmoothingConfig(sigma_arcmin=4.0, arcmin_per_pixel=0.5, mode="constant")
    sm = smooth_flat_sky(m, cfg)
    s = cfg.sigma_pix()
    assert s == pytest.approx(8.0)
    # flux conserved (interior source, normalized kernel)
    assert sm.sum() == pytest.approx(1.0, rel=1e-6)
    # peak height of a unit-flux 2D Gaussian
    assert sm[100, 100] == pytest.approx(1.0 / (2 * np.pi * s**2), rel=1e-2)
    # width recovered from the second moment
    assert _measured_sigma_pix(sm) == pytest.approx(s, rel=2e-2)


def test_wrap_mode_conserves_flux_exactly():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(128, 128))
    cfg = SmoothingConfig(sigma_arcmin=2.0, arcmin_per_pixel=5.0 * 60 / 128, mode="wrap")
    sm = smooth_flat_sky(m, cfg)
    assert sm.sum() == pytest.approx(m.sum(), rel=1e-12)


def test_zero_sigma_is_noop():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(16, 16))
    cfg = SmoothingConfig(sigma_arcmin=0.0, arcmin_per_pixel=0.5)
    assert np.array_equal(smooth_flat_sky(m, cfg), m)


def test_fwhm_relation():
    cfg = SmoothingConfig(sigma_arcmin=2.0, arcmin_per_pixel=1.0)
    assert cfg.fwhm_arcmin() == pytest.approx(2.0 * np.sqrt(8 * np.log(2)))


def test_frozen_smoothing_set_matches_peak_definition():
    assert FIDUCIAL_SMOOTHING_ARCMIN == 2.0
    assert PAPER2_SMOOTHING_ARCMIN == (1.0, 2.0, 5.0, 8.0)
    assert FIDUCIAL_SMOOTHING_ARCMIN in PAPER2_SMOOTHING_ARCMIN


def test_bad_pixel_scale_raises():
    with pytest.raises(ValueError):
        SmoothingConfig(sigma_arcmin=2.0, arcmin_per_pixel=0.0).sigma_pix()


def test_curved_sky_smoothing_smoke():
    # Exercises the PEAK_DEFINITION §1 curved-sky path; healpy-guarded.
    # Band-limited (ell<=1) input so the SHT roundtrip is faithful at Nside 8.
    hp = pytest.importorskip("healpy")
    nside = 8
    npix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    m = 1.0 + np.cos(theta)  # monopole + dipole
    sm = smooth_curved_sky(m, sigma_arcmin=1800.0)  # 30 deg
    assert sm.shape == (npix,)
    assert np.all(np.isfinite(sm))
    # Gaussian beam has b_0 = 1 -> monopole (map mean) preserved;
    # b_1 < 1 -> dipole amplitude suppressed.
    assert sm.mean() == pytest.approx(m.mean(), rel=1e-3)
    assert np.ptp(sm) < np.ptp(m)
