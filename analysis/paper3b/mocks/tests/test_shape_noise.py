"""Analytic-fixture tests for WP-B4 DES shape noise (fixed-seed statistics)."""

import numpy as np
import pytest

from analysis.paper3b.mocks.shape_noise import (
    DESY3_SIGMA_E,
    ShapeNoiseConfig,
    add_shape_noise,
    noise_map,
)


def test_pixel_sigma_matches_formula():
    cfg = ShapeNoiseConfig(n_eff_arcmin2=5.0, pix_area_arcmin2=0.25, sigma_e=0.26)
    expected = 0.26 / np.sqrt(5.0 * 0.25)
    assert cfg.pixel_sigma() == pytest.approx(expected)


def test_noise_map_has_expected_variance():
    cfg = ShapeNoiseConfig(n_eff_arcmin2=5.59, pix_area_arcmin2=0.25)
    n = noise_map((900, 900), cfg, seed=1234)
    sigma = cfg.pixel_sigma()
    assert n.mean() == pytest.approx(0.0, abs=5e-3)
    assert n.std() == pytest.approx(sigma, rel=3e-2)
    assert n.var() == pytest.approx(sigma**2, rel=6e-2)


def test_add_shape_noise_is_signal_plus_noise():
    cfg = ShapeNoiseConfig(n_eff_arcmin2=5.0, pix_area_arcmin2=0.25)
    signal = np.full((64, 64), 0.03)
    noisy_a = add_shape_noise(signal, cfg, seed=7)
    noisy_b = signal + noise_map(signal.shape, cfg, seed=7)
    assert noisy_a == pytest.approx(noisy_b)  # same seed -> same realization
    # residual after removing signal has the right dispersion
    assert (noisy_a - signal).std() == pytest.approx(cfg.pixel_sigma(), rel=0.15)


def test_infinite_neff_means_no_noise():
    # zero 1/n_eff limit -> sigma_pix == 0 -> map returned unchanged.
    cfg = ShapeNoiseConfig(n_eff_arcmin2=np.inf, pix_area_arcmin2=0.25)
    assert cfg.pixel_sigma() == 0.0
    rng = np.random.default_rng(0)
    m = rng.normal(size=(32, 32))
    assert np.array_equal(add_shape_noise(m, cfg, seed=99), m)
    assert np.array_equal(noise_map((32, 32), cfg), np.zeros((32, 32)))


def test_seed_reproducibility_and_independence():
    cfg = ShapeNoiseConfig(n_eff_arcmin2=1.5, pix_area_arcmin2=0.25)
    assert np.array_equal(noise_map((16, 16), cfg, seed=5), noise_map((16, 16), cfg, seed=5))
    assert not np.array_equal(noise_map((16, 16), cfg, seed=5), noise_map((16, 16), cfg, seed=6))


def test_default_sigma_e_is_des_y3():
    cfg = ShapeNoiseConfig(n_eff_arcmin2=1.5, pix_area_arcmin2=0.25)
    assert cfg.sigma_e == DESY3_SIGMA_E == 0.261


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        ShapeNoiseConfig(n_eff_arcmin2=5.0, pix_area_arcmin2=0.0).pixel_sigma()
    with pytest.raises(ValueError):
        ShapeNoiseConfig(n_eff_arcmin2=-1.0, pix_area_arcmin2=0.25).pixel_sigma()
