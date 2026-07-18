"""WP-B3 battery-logic tests (synthetic; the stacking core is tested elsewhere)."""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import healpy as hp  # noqa: E402

from analysis.paper3b.stack.battery import (  # noqa: E402
    cib_band,
    distance_to_footprint_edge_deg,
    null_ensemble_stats,
    proximity_tercile_test,
    shift_catalog_ra,
)
from analysis.paper3b.stack.covariance import patch_ids  # noqa: E402

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)


def test_shift_catalog_ra_wraps_and_recuts():
    act = np.zeros(NPIX)
    # interior band: RA in [0, 90) at all dec
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    act[(np.degrees(phi) < 90)] = 1.0
    ra = np.array([10.0, 80.0, 350.0])
    dec = np.array([-30.0, 0.0, 20.0])
    ra2, dec2, keep, ret = shift_catalog_ra(ra, dec, 15.0, act)
    # 10->25 in, 80->95 out, 350->5 in (wrap)
    assert keep.tolist() == [True, False, True]
    np.testing.assert_allclose(ra2, [25.0, 5.0])
    assert ret == pytest.approx(2 / 3)


def test_distance_to_edge_orders_correctly():
    binary = np.zeros(NPIX, dtype=bool)
    disc = hp.query_disc(NSIDE, hp.ang2vec(np.pi / 2, 0.0), np.radians(20.0))
    binary[disc] = True
    d_center = distance_to_footprint_edge_deg(np.array([0.0]), np.array([0.0]), binary)
    d_near = distance_to_footprint_edge_deg(np.array([18.0]), np.array([0.0]), binary)
    assert d_center[0] > d_near[0] > 0
    assert d_center[0] == pytest.approx(20.0, abs=2.0)   # ~pixel-scale accuracy


def test_proximity_tercile_flags_planted_trend():
    rng = np.random.default_rng(0)
    n = 3000
    dist = rng.uniform(0, 3, n)
    y = np.zeros((n, 5))
    y[:, 2] = rng.normal(0, 0.1, n) + np.where(dist < 1.0, -1.0, 0.0)  # near-edge deficit
    nu = np.full(n, 2.5)                                               # all in bin 2
    out = proximity_tercile_test(y, nu, dist)
    assert out["bin2"]["sigma"] < -5           # strongly detected planted trend
    y2 = y.copy(); y2[:, 2] = rng.normal(0, 0.1, n)
    out2 = proximity_tercile_test(y2, nu, dist)
    assert abs(out2["bin2"]["sigma"]) < 3      # clean case consistent with zero


def test_cib_band_verdicts_and_escalation():
    sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    y_by = {"fid": np.zeros(5), "a": np.full(5, 0.3), "b": np.full(5, -0.1)}
    out = cib_band(y_by, sigma)                # band = 0.4 -> negligible
    assert out["verdict_per_bin"] == ["negligible"] * 5 and not out["escalate"]
    y_by["c"] = np.full(5, 1.2)                # band = 1.3 -> exceeds everywhere
    out2 = cib_band(y_by, sigma)
    assert out2["escalate"] and out2["verdict_per_bin"][2] == "exceeds_sigma"


def test_null_ensemble_stats_validates_jackknife():
    rng = np.random.default_rng(3)
    n, nbin = 64, 5
    means = rng.normal(0, 1.0, (n, nbin))
    jk = np.full((n, nbin), 1.0)               # jackknife sigma == true scatter
    st = null_ensemble_stats(means, jk)
    assert np.all(np.abs(np.array(st["jk_validation_ratio"]) - 1.0) < 0.3)
    assert np.all(np.abs(st["mean_over_2err"]) < 2.5)


def test_covariate_excision_test_stable_null_vs_planted_shift():
    from analysis.paper3b.stack.battery import covariate_excision_test

    rng = np.random.default_rng(11)
    n = 2000
    ra = rng.uniform(0, 10, n)
    dec = rng.uniform(-10, 0, n)
    patch8 = patch_ids(ra, dec, 8)
    covariate = rng.uniform(0, 1, n)             # e.g. E(B-V) or star density
    nu = np.full(n, 2.5)                          # all peaks in bin 2 (nu 2-3)
    y_null = np.zeros((n, 5))
    y_null[:, 2] = rng.normal(1.0, 0.1, n)         # J=2 is the fiducial radius
    y_mean_ref = np.array([0, 0, 1.0, 0, 0])
    y_err_ref = np.array([1, 1, 0.02, 1, 1])
    out_null = covariate_excision_test(y_null, nu, covariate, patch8,
                                       y_mean_ref, y_err_ref,
                                       keep_below_percentile=90.0)
    assert out_null["bin2"]["pass_0p5sig"]
    assert out_null["kept_fraction"] == pytest.approx(0.9, abs=0.02)

    y_shift = y_null.copy()
    y_shift[covariate > 0.9, 2] += 5.0            # contaminate the excised tail
    out_shift = covariate_excision_test(y_shift, nu, covariate, patch8,
                                        y_mean_ref, y_err_ref,
                                        keep_below_percentile=90.0)
    assert out_shift["bin2"]["pass_0p5sig"]        # excision removes the tail
    # without excision (percentile=100), the contamination shows up
    out_unexcised = covariate_excision_test(y_shift, nu, covariate, patch8,
                                            y_mean_ref, y_err_ref,
                                            keep_below_percentile=100.0)
    assert not out_unexcised["bin2"]["pass_0p5sig"]


def test_frozen_loader_verifies_and_loads():
    # Integration guard for the SIGNED freeze (2026-07-17): one call loads the
    # vector + total covariance, and any post-freeze change to the data files
    # raises on the hash check. Skipped where the ceph archive is absent.
    from pathlib import Path

    from analysis.paper3b.stack.frozen import B_ROOT, load_frozen

    if not (B_ROOT / "wp2_measurement/stack_wiener_sm2am_fid.npz").exists():
        pytest.skip("frozen archive not mounted")
    fm = load_frozen("wiener")
    assert fm.y.shape == (4,) and fm.cov_total.shape == (4, 4)
    assert np.all(np.diff(fm.y) > 0)                      # monotonic in nu
    assert np.all(np.linalg.eigvalsh(fm.cov_total) > 0)   # positive definite
    ev = np.linalg.eigvalsh
    assert ev(fm.cov_total).sum() > ev(fm.cov_stat).sum() # sys adds variance
    with pytest.raises(ValueError):
        load_frozen("ks")
