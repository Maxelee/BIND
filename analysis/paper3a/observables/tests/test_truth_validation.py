"""Tests for the Sigma_model residual bookkeeping."""

import numpy as np
import pytest

from analysis.paper3a.observables.truth_validation import (
    save_sigma_model,
    stacked_residual_bootstrap,
)


def test_known_bias_and_covariance_recovered():
    rng = np.random.default_rng(11)
    n_halos, n_bins = 400, 5
    truth = rng.normal(10.0, 1.0, size=(n_halos, n_bins))
    bias_true = np.array([0.5, -0.2, 0.0, 0.1, 0.3])
    noise_sd = 0.8
    pred = truth + bias_true + rng.normal(0, noise_sd, size=(n_halos, n_bins))
    out = stacked_residual_bootstrap(pred, truth, n_boot=1500, seed=1)
    se = noise_sd / np.sqrt(n_halos)
    assert out["bias"] == pytest.approx(bias_true, abs=4 * se)
    assert np.sqrt(np.diag(out["sigma_model"])) == pytest.approx(np.full(n_bins, se), rel=0.15)
    assert out["n_halos"] == n_halos


def test_zero_residual_gives_zero_bias():
    x = np.random.default_rng(2).normal(size=(50, 3))
    out = stacked_residual_bootstrap(x, x, n_boot=200)
    assert out["bias"] == pytest.approx(np.zeros(3), abs=1e-14)
    assert np.abs(out["sigma_model"]).max() < 1e-28


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        stacked_residual_bootstrap(np.zeros((10, 3)), np.zeros((10, 4)))


def test_save_requires_provenance(tmp_path):
    x = np.random.default_rng(3).normal(size=(30, 2))
    out = stacked_residual_bootstrap(x + 0.1, x, n_boot=100)
    with pytest.raises(ValueError, match="provenance"):
        save_sigma_model(tmp_path / "sm", out, {"script": "s"})
    prov = {
        "truth_inputs": "TNG300 snap 99 (test)",
        "model_or_checkpoint": "fm_thermo (test)",
        "operator_config": {"radii_arcmin": [1, 2]},
        "script": "test",
        "date": "2026-07-16",
    }
    save_sigma_model(tmp_path / "sm", out, prov)
    loaded = np.load(tmp_path / "sm.npz")
    assert loaded["bias"] == pytest.approx(out["bias"])
    assert (tmp_path / "sm.provenance.json").exists()
