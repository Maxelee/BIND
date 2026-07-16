"""Analytic-fixture tests for WP-B4 ACT beam on mock y maps (point -> Gaussian)."""

import numpy as np
import pytest

from analysis.paper3b.mocks.beam import (
    ACT_DR6_YMAP_FWHM_ARCMIN,
    BeamConfig,
    apply_beam,
)


def _measured_sigma_pix(m):
    ny, nx = m.shape
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    tot = m.sum()
    xc = (m * x).sum() / tot
    yc = (m * y).sum() / tot
    vx = (m * (x - xc) ** 2).sum() / tot
    vy = (m * (y - yc) ** 2).sum() / tot
    return np.sqrt(0.5 * (vx + vy))


def test_point_source_becomes_gaussian_of_right_fwhm():
    n = 161
    m = np.zeros((n, n))
    m[80, 80] = 1.0
    # 0.5 arcmin/pixel, FWHM 6 arcmin -> fwhm_pix = 12, sigma_pix = 12/2.3548.
    cfg = BeamConfig(fwhm_arcmin=6.0, arcmin_per_pixel=0.5, mode="constant")
    beamed = apply_beam(m, cfg)
    s = cfg.sigma_pix()
    assert s == pytest.approx(12.0 / np.sqrt(8 * np.log(2)))
    # total flux (integral of y) conserved
    assert beamed.sum() == pytest.approx(1.0, rel=1e-6)
    # peak height of a unit-flux 2D Gaussian
    assert beamed[80, 80] == pytest.approx(1.0 / (2 * np.pi * s**2), rel=1e-2)
    # recovered FWHM matches the requested one
    fwhm_pix_meas = _measured_sigma_pix(beamed) * np.sqrt(8 * np.log(2))
    assert fwhm_pix_meas == pytest.approx(12.0, rel=2e-2)


def test_wrap_mode_conserves_total_flux():
    rng = np.random.default_rng(3)
    m = np.abs(rng.normal(size=(96, 96)))  # y >= 0
    cfg = BeamConfig(fwhm_arcmin=1.6, arcmin_per_pixel=0.5, mode="wrap")
    assert apply_beam(m, cfg).sum() == pytest.approx(m.sum(), rel=1e-12)


def test_zero_fwhm_is_noop():
    rng = np.random.default_rng(4)
    m = rng.normal(size=(16, 16))
    cfg = BeamConfig(fwhm_arcmin=0.0, arcmin_per_pixel=0.5)
    assert np.array_equal(apply_beam(m, cfg), m)


def test_fwhm_sigma_conversion():
    cfg = BeamConfig(fwhm_arcmin=1.6, arcmin_per_pixel=0.5)
    assert cfg.sigma_arcmin() == pytest.approx(1.6 / np.sqrt(8 * np.log(2)))
    assert cfg.sigma_pix() == pytest.approx((1.6 / np.sqrt(8 * np.log(2))) / 0.5)


def test_act_dr6_default_documented():
    assert ACT_DR6_YMAP_FWHM_ARCMIN == 1.6
