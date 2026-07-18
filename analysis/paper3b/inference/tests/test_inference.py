"""Analytic-fixture tests for the WP-B5 inference layer."""

import numpy as np
import pytest

from analysis.paper3b.inference.gridmodel import GP2D, GridEmulator, GridTable
from analysis.paper3b.inference.likelihood import B5Posterior, grid_loglike

RNG = np.random.default_rng(20260718)


def _synthetic_table(n=40, noise=0.01):
    """Smooth positive 4-bin surface over a 2D cloud."""
    c = RNG.uniform([-0.15, -0.01], [0.05, 0.04], size=(n, 2))
    y = np.stack([
        5e-6 * np.exp(3.0 * c[:, 0] + 8.0 * c[:, 1]),
        1.2e-5 * np.exp(2.0 * c[:, 0] - 4.0 * c[:, 1]),
        3.5e-5 * np.exp(4.0 * c[:, 0] + 2.0 * c[:, 1]),
        9.0e-5 * np.exp(1.5 * c[:, 0] + 6.0 * c[:, 1]),
    ], axis=1)
    y *= 1.0 + noise * RNG.standard_normal(y.shape)
    yerr = noise * y
    return GridTable(c, y, yerr, [f"pt{i}" for i in range(n)])


def test_gp_recovers_smooth_function():
    x = RNG.uniform(-1, 1, size=(60, 2))
    f = np.sin(2 * x[:, 0]) + 0.5 * x[:, 1]
    gp = GP2D(x, f + 0.01 * RNG.standard_normal(60), np.full(60, 0.01))
    xq = RNG.uniform(-0.8, 0.8, size=(20, 2))
    fq = np.sin(2 * xq[:, 0]) + 0.5 * xq[:, 1]
    m, v = gp.predict(xq)
    assert np.abs(m - fq).max() < 0.08
    assert (v > 0).all()


def test_gp_variance_grows_off_cloud():
    x = RNG.uniform(-0.5, 0.5, size=(50, 2))
    gp = GP2D(x, x[:, 0], np.full(50, 0.01))
    _, v_in = gp.predict(np.array([[0.0, 0.0]]))
    _, v_out = gp.predict(np.array([[3.0, 3.0]]))
    assert v_out[0] > 10 * v_in[0]


def test_emulator_roundtrip_and_positive():
    t = _synthetic_table()
    em = GridEmulator.fit(t)
    y, vy = em.predict(t.coords[:5])
    assert np.allclose(y, t.y[:5], rtol=0.08)
    assert (y > 0).all() and (vy > 0).all()


def test_posterior_peaks_at_injected_point():
    t = _synthetic_table(n=50, noise=0.005)
    em = GridEmulator.fit(t)
    truth = np.array([-0.05, 0.015])
    y_true, _ = em.predict(truth)
    cov = np.diag((0.03 * y_true[0]) ** 2)
    post = grid_loglike(y_true[0], cov, em,
                        window=((-0.15, 0.05), (-0.01, 0.04)), n=61)
    m, c = post.mean_and_cov()
    sig = np.sqrt(np.diag(c))
    assert np.all(np.abs(m - truth) < 3 * sig + 1e-3)
    assert post.contains(truth, 0.95)


def test_hpd_levels_monotonic():
    lnl = -0.5 * (np.linspace(-3, 3, 41)[:, None] ** 2
                  + np.linspace(-3, 3, 41)[None, :] ** 2)
    post = B5Posterior(np.linspace(-3, 3, 41), np.linspace(-3, 3, 41), lnl)
    assert post.hpd_level(0.68) > post.hpd_level(0.95)
    assert post.contains(np.zeros(2), 0.68)
    assert not post.contains(np.array([3.0, 3.0]), 0.68)


def test_real_grid_table_loads_with_anchor():
    t = GridTable.load()
    assert t.coords.shape[0] == 61 and t.y.shape == (61, 4)
    assert t.names[-1].startswith("bind/run_0000")
    assert np.allclose(t.coords[-1], 0.0)
    assert (t.y > 0).all()
