"""Tests for the B-side likelihood block (JOINT_AB_PLAN step 4)."""

import numpy as np
import pytest

from analysis.paper3a.inference.bblock import (
    OFFSET_SYS, REC_CHI2, REC_DATA, REC_SIGMA_TOT, SLOPE_SYS,
    TWOBOUND_REF_OFFSET, BBlock, _CapDilution)


@pytest.fixture(scope="module")
def blk():
    return BBlock()


def test_frozen_data_vector(blk):
    assert np.allclose(blk.data, REC_DATA, rtol=0, atol=5e-9)


def test_sigma_total_matches_recorded(blk):
    sig = np.sqrt(blk.var_stat + np.diag(blk.cov_sys))
    assert np.allclose(sig, REC_SIGMA_TOT, rtol=0.01)


def test_sys_cov_rank1_psd(blk):
    ev = np.linalg.eigvalsh(blk.cov_sys)
    assert ev.min() > -1e-12 * ev.max()               # PSD to float noise
    assert np.sum(ev > 1e-3 * ev.max()) == 1          # rank-1 CIB term


def test_gp_interpolates_training_nodes(blk):
    for b, gp in enumerate(blk.gps):
        mu, _ = gp.predict(gp.X)
        assert np.max(np.abs(mu - gp.y)) < 5 * np.max(gp.yerr) + 0.02


def test_chi2_anchors_reproduce_recorded(blk):
    chi2_fid, _, _ = blk.chi2(np.zeros(2))
    chi2_cor, _, _ = blk.chi2(np.array([-0.35, -0.03]))
    assert abs(chi2_fid / REC_CHI2["fiducial"] - 1) < 0.10
    assert abs(chi2_cor / REC_CHI2["corner"] - 1) < 0.20


def test_dilution_monotone_and_bounded(blk):
    sig = np.linspace(0, 3.6, 10)
    D = blk.dil.dilution_batch(sig)
    assert np.allclose(D[0], 1.0, atol=1e-9)
    assert np.all(D <= 1.0 + 1e-12)
    assert np.all(np.diff(D, axis=0) <= 1e-12)


def test_dilution_batch_matches_scalar(blk):
    for s in (0.5, 2.0, 3.5):
        assert np.allclose(blk.dil.dilution_batch(np.array([s]))[0],
                           blk.dil.dilution(s), rtol=1e-12)


def test_batch_matches_scalar_loglike(blk):
    rng = np.random.default_rng(1)
    c = rng.uniform(-0.2, 0.1, size=(5, 2))
    e = np.full((5, 2), 0.03)
    sig = rng.uniform(0, 3.0, 5)
    ll = blk.loglike_batch(c, e, sig)
    for i in range(5):
        cB = c[i] + TWOBOUND_REF_OFFSET
        cvar = e[i] ** 2 + (SLOPE_SYS * cB) ** 2 + OFFSET_SYS**2
        chi2, _, C = blk.chi2(cB, sig[i], coord_var=cvar)
        ll_s = -0.5 * (chi2 + np.linalg.slogdet(C)[1])
        assert abs(ll[i] - ll_s) < 1e-8


def test_cap_gaussian_closed_form():
    # single Gaussian: CAP(R) = (2 a s^2/R^2)(1 - exp(-R^2/2s^2))^2 must
    # match a brute-force 2D integral
    s, a, R = 2.0, 1.0, 4.0
    x = np.linspace(-20, 20, 2001)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    img = a * np.exp(-r2 / (2 * s**2))
    dA = (x[1] - x[0]) ** 2
    disc = img[r2 <= R**2].sum() * dA / (np.pi * R**2)
    ring = img[(r2 > R**2) & (r2 <= 2 * R**2)].sum() * dA / (np.pi * R**2)
    closed = float(_CapDilution._cap_matrix(np.array([R]),
                                            np.array([s]))[0, 0]) * a
    assert abs((disc - ring) - closed) < 2e-3 * closed + 1e-6
