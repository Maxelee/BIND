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
