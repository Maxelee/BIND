"""Analytic-fixture tests for the WP-B4/B5 transfer-matched filter."""

import numpy as np
import pytest

from analysis.paper3b.mocks.transfer import (
    ELL_MAX_NSIDE1024,
    TransferFunction,
    build_empirical_transfer,
    flat_sky_power,
)

FOV = 5.0
NPIX = 256
RNG = np.random.default_rng(20260717)


def _white(sigma=1.0, npix=NPIX):
    return RNG.normal(0.0, sigma, size=(npix, npix))


def test_white_noise_power_is_flat_at_sigma2_omegapix():
    sigma = 0.7
    omega_pix = (np.deg2rad(FOV) / NPIX) ** 2
    cls = []
    for _ in range(8):
        _, cl, n = flat_sky_power(_white(sigma), FOV)
        cls.append(cl)
    cl = np.nanmean(cls, axis=0)
    well_sampled = n > 200
    expected = sigma ** 2 * omega_pix
    assert np.nanmean(cl[well_sampled]) == pytest.approx(expected, rel=0.05)
    # flat: no bin far off the white level where sampling is good
    assert np.nanmax(np.abs(cl[well_sampled] / expected - 1.0)) < 0.35


def test_unit_transfer_is_identity():
    t = TransferFunction(np.array([10.0, 1e5]), np.array([1.0, 1.0]), ell_zero=None)
    p = _white()
    assert np.allclose(t.apply(p, FOV), p, atol=1e-12)


def test_flat_half_transfer_halves_fluctuations():
    t = TransferFunction(np.array([10.0, 1e5]), np.array([0.5, 0.5]), ell_zero=None)
    p = _white()
    assert np.allclose(t.apply(p, FOV), 0.5 * p, atol=1e-12)


def test_band_limit_zeroes_high_ell():
    ell_zero = 2000.0
    t = TransferFunction(np.array([10.0, 1e5]), np.array([1.0, 1.0]), ell_zero=ell_zero)
    filtered = t.apply(_white(), FOV)
    ell, cl, n = flat_sky_power(filtered, FOV)
    hi = (ell > 1.3 * ell_zero) & (n > 50)
    lo = (ell < 0.7 * ell_zero) & (n > 50)
    assert np.nanmean(cl[hi]) < 1e-6 * np.nanmean(cl[lo])


def test_empirical_transfer_recovers_known_filter():
    # target filter: smooth Gaussian suppression T = exp(-(ell/3000)^2)
    true_t = TransferFunction(np.geomspace(50, 3.5e4, 64),
                              np.exp(-(np.geomspace(50, 3.5e4, 64) / 3000.0) ** 2),
                              ell_zero=None)
    raw_cls, filt_cls = [], []
    for _ in range(24):
        p = _white()
        _, cl_raw, _ = flat_sky_power(p, FOV)
        ell, cl_f, _ = flat_sky_power(true_t.apply(p, FOV), FOV)
        raw_cls.append(cl_raw)
        filt_cls.append(cl_f)
    est = build_empirical_transfer(ell, np.nanmean(filt_cls, axis=0),
                                   ell, np.nanmean(raw_cls, axis=0),
                                   ell_min=200.0, ell_max=3000.0, smooth_dex=0.05)
    probe = np.geomspace(300, 2500, 12)
    assert np.allclose(est(probe), true_t(probe), rtol=0.08, atol=0.02)


def test_empirical_transfer_is_zero_above_band_limit():
    ell = np.geomspace(80, ELL_MAX_NSIDE1024, 32)
    est = build_empirical_transfer(ell, np.ones_like(ell), ell, np.ones_like(ell))
    assert est(np.array([ELL_MAX_NSIDE1024 * 1.01, 1e5])).max() == 0.0
    assert est(np.array([500.0])) == pytest.approx(1.0, rel=1e-6)


def test_transfer_validation_errors():
    with pytest.raises(ValueError):
        TransferFunction(np.array([1.0]), np.array([1.0]))          # too short
    with pytest.raises(ValueError):
        TransferFunction(np.array([2.0, 1.0]), np.array([1.0, 1.0]))  # not increasing
    with pytest.raises(ValueError):
        TransferFunction(np.array([1.0, 2.0]), np.array([1.0, -0.1]))  # negative
