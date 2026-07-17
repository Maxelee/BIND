"""WP-B2 stacking-pipeline tests on synthetic inputs (no survey files).

Acceptance criterion "map-source-agnostic" is exercised directly: the full
run_peak_stack chain runs on a synthetic pixell enmap with injected sources.
"""

import numpy as np
import pytest

pytest.importorskip("pixell")
from pixell import enmap  # noqa: E402

from analysis.paper3b.stack.cap import cap_filter, cap_filter_multi, cap_masks  # noqa: E402
from analysis.paper3b.stack.covariance import (  # noqa: E402
    eigenvalue_drift,
    jackknife_mean_cov,
    patch_ids,
)
from analysis.paper3b.stack.stacker import run_peak_stack  # noqa: E402

RES = 0.5  # arcmin


# ----------------------------------------------------------------------- CAP

def test_cap_uniform_map_filters_to_exactly_zero():
    thumb = np.full((61, 61), 3.7)
    np.testing.assert_allclose(cap_filter(thumb, 4.0, RES), 0.0, atol=1e-12)


def test_cap_compact_source_returns_its_flux():
    # a source entirely inside the disk: CAP = flux (annulus empty of signal)
    thumb = np.zeros((61, 61))
    thumb[30, 30] = 5.0
    y = cap_filter(thumb, 4.0, RES)
    np.testing.assert_allclose(y, 5.0 * RES ** 2, rtol=1e-12)


def test_cap_multi_shapes_and_annulus_guard():
    thumbs = np.random.default_rng(0).normal(size=(7, 61, 61))
    out = cap_filter_multi(thumbs, (2.0, 4.0, 8.0), RES)
    assert out.shape == (7, 3)
    with pytest.raises(ValueError):
        cap_masks(9, 9, 0.2, RES)  # annulus smaller than a pixel


# ----------------------------------------------------------- jackknife cov

def test_jackknife_matches_analytic_iid_error():
    rng = np.random.default_rng(1)
    n = 20000
    v = rng.normal(0.0, 1.0, (n, 1))
    ids = rng.integers(0, 100, n)          # 100 equal patches
    mean, cov, k = jackknife_mean_cov(v, ids)
    assert k == 100
    np.testing.assert_allclose(mean[0], 0.0, atol=0.02)
    np.testing.assert_allclose(np.sqrt(cov[0, 0]), 1.0 / np.sqrt(n), rtol=0.25)


def test_jackknife_needs_three_patches():
    with pytest.raises(ValueError):
        jackknife_mean_cov(np.ones((10, 1)), np.zeros(10))


def test_eigenvalue_drift_zero_for_identical():
    c = np.array([[2.0, 0.3], [0.3, 1.0]])
    assert eigenvalue_drift(c, c) == 0.0
    assert eigenvalue_drift(c, 1.2 * c) == pytest.approx(0.2)


def test_patch_ids_groups_nearby_points():
    ids = patch_ids(np.array([10.0, 10.01, 120.0]), np.array([-30.0, -30.01, 40.0]), 8)
    assert ids[0] == ids[1] != ids[2]


# ------------------------------------------------- end-to-end on a mock map

def _synthetic_map_with_sources(n_src=60, amp=1e-5, seed=3):
    shape, wcs = enmap.geometry(pos=np.deg2rad([[-8, -8], [8, 8]]),
                                res=np.deg2rad(RES / 60.0), proj="car")
    m = enmap.zeros(shape, wcs)
    rng = np.random.default_rng(seed)
    ra = rng.uniform(-6, 6, n_src); dec = rng.uniform(-6, 6, n_src)
    pix = m.sky2pix(np.deg2rad(np.stack([dec, ra]))).astype(int)
    m[pix[0], pix[1]] = amp
    return m, ra, dec


def test_run_peak_stack_recovers_injected_sources_and_null():
    m, ra, dec = _synthetic_map_with_sources()
    nu = np.linspace(0.2, 5.0, len(ra))       # spread over all stack bins
    res = run_peak_stack(m, ra, dec, nu, label="synthetic")
    assert res.n_per_bin.sum() == len(ra)
    # every occupied bin recovers ~the injected per-source flux at theta_d=4'
    j = 2  # 4' is index 2 of the frozen radius set
    for b in range(len(res.n_per_bin)):
        if res.n_per_bin[b] >= 5:
            np.testing.assert_allclose(res.y_mean[b, j], 1e-5 * RES ** 2, rtol=0.05)
    # profiles centrally peaked
    occ = res.n_per_bin > 0
    assert (res.profiles[occ][:, 0] > res.profiles[occ][:, -1]).all()
    # null: random positions away from sources -> consistent with zero
    rng = np.random.default_rng(4)
    res0 = run_peak_stack(m, rng.uniform(-6, 6, 200), rng.uniform(-6, 6, 200),
                          np.full(200, 1.5), label="null")
    b = 1  # all null objects land in nu bin [1,2)
    assert abs(res0.y_mean[b, j]) < 5 * np.sqrt(res0.y_cov[b, j, j]) + 1e-12


def test_result_roundtrip(tmp_path):
    from analysis.paper3b.stack.results import load_result, save_result

    m, ra, dec = _synthetic_map_with_sources(n_src=30)
    res = run_peak_stack(m, ra, dec, np.full(30, 2.5), label="rt")
    save_result(res, tmp_path)
    back = load_result("rt", tmp_path)
    np.testing.assert_allclose(back.y_mean, res.y_mean)
    np.testing.assert_allclose(back.y_cov, res.y_cov)
    np.testing.assert_allclose(back.significance(), res.significance())
