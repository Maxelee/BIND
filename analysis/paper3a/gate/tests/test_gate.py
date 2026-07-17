"""Tests for the gate compute pieces that have analytic answers."""

import numpy as np
import pytest

from analysis.paper3a.gate.gate_operators import (
    blend_gas_background,
    cosmic_mean_gas_per_pixel,
    m500c_from_m200c,
    r500c_from_r200c,
)
from analysis.paper3a.observables import PatchGeometry, cap_photometry


def test_nfw_rescaling_textbook_values():
    assert r500c_from_r200c(np.array([1.0]), c200=5.0)[0] == pytest.approx(0.661, abs=0.01)
    assert m500c_from_m200c(np.array([1.0]), c200=5.0)[0] == pytest.approx(0.72, abs=0.02)


def test_cosmic_mean_gas_magnitude():
    # 51.25 Mpc/h slab, TNG300 gate pixel: known-value check (hand-computed)
    got = cosmic_mean_gas_per_pixel(51.25, 205.0 / 4198)
    assert got == pytest.approx(1.65e9, rel=0.02)


def test_blend_uniform_when_painted_equals_background():
    # painted == cosmic mean everywhere -> blended map is exactly uniform for
    # ANY alpha field -> CAP filters it to ~0 (the compensation property the
    # v2 fix exists to restore).
    rng = np.random.default_rng(5)
    sigma_bar = 1.65e9
    alpha = rng.random((129, 129))
    gas = blend_gas_background(np.full((129, 129), sigma_bar), alpha, sigma_bar)
    assert gas == pytest.approx(np.full((129, 129), sigma_bar), rel=1e-12)
    out = cap_photometry(gas, (64.0, 64.0), np.array([10.0, 20.0]))
    assert np.abs(out).max() < 1e-6 * sigma_bar


def test_blend_recovers_painted_inside_apertures():
    alpha = np.zeros((8, 8))
    alpha[2:6, 2:6] = 1.0
    painted = np.full((8, 8), 7.0)
    out = blend_gas_background(painted, alpha, 3.0)
    assert out[3, 3] == pytest.approx(7.0)
    assert out[0, 0] == pytest.approx(3.0)
