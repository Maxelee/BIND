"""WP-A5 inference tests (torch3 venv; needs the ceph artifacts)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.paper3a.emulator import params_meta as pm


@pytest.fixture(scope="module")
def block():
    from analysis.paper3a.inference.fgas import FgasBlock

    return FgasBlock()


def test_data_assembly(block):
    d = block.data
    assert 3 <= len(d.values) <= 5
    assert np.all(d.values > 0.01) and np.all(d.values < 0.3)
    assert np.all(d.stat_err > 0)
    assert np.allclose(d.weights.sum(axis=1), 1.0, atol=1e-8)
    # low-z eRASS1 groups/clusters: thousands of objects total
    assert d.n_per_bin.sum() > 500


def test_prediction_magnitude_and_response(block):
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    pred = block.predict(u_fid)[0]
    # spherical f_gas at group-to-cluster scales: sane range
    assert np.all(pred > 0.02) and np.all(pred < 0.25)
    # wind response survives the convolution chain with positive sign
    i = pm.ASTRO_NAMES.index("WindEnergyIn1e51erg")
    u_hi = u_fid.copy()
    u_hi[i] = 0.95
    u_lo = u_fid.copy()
    u_lo[i] = 0.05
    assert np.all(block.predict(u_hi)[0] > block.predict(u_lo)[0])


def test_loglike_shapes_and_finiteness(block):
    rng = np.random.default_rng(0)
    U = rng.uniform(0.1, 0.9, size=(8, 30))
    ll = block.loglike(U)
    assert ll.shape == (8,) and np.all(np.isfinite(ll))
    # chi2 consistent with loglike ordering: better chi2 -> higher loglike
    c = block.chi2(U)
    assert np.argmin(c) == np.argmax(ll + 0.5 * np.sum(
        np.log(2 * np.pi * block.sigma(block.predict(U))**2), axis=1))


def test_sigma_floor_exceeds_stat(block):
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    pred = block.predict(u_fid)
    sig = block.sigma(pred)[0]
    assert np.all(sig > block.data.stat_err)   # model terms strictly widen


def test_prior_bounds_rejected():
    from analysis.paper3a.inference.sampler import log_prob_factory

    class _Dummy:
        def loglike(self, U):
            return np.zeros(len(U))

    lp = log_prob_factory([_Dummy()])
    U = np.array([[0.5] * 30, [1.2] + [0.5] * 29, [-0.1] + [0.5] * 29])
    out = lp(U)
    assert np.isfinite(out[0]) and np.isinf(out[1]) and np.isinf(out[2])


def test_split_rhat_on_stationary_chain():
    from analysis.paper3a.inference.sampler import split_rhat

    rng = np.random.default_rng(1)
    chain = rng.normal(size=(400, 16, 3))
    r = split_rhat(chain)
    assert np.all(r < 1.05)
