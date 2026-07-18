"""Coarse-grid peak-FINDING convention tests (WP-B4/B5, supersedes quantize_arcmin).

Covers `patch.coarse_grid_peaks` / `patch.effective_patch_area_deg2` in
isolation (analytic bumps, block-average correctness) and the
`measure.measure_mock_patch(peak_grid_block=...)` integration: the
peak_grid_block=None path must stay bit-identical to the pre-existing
native-grid chain (regression), and peak_grid_block=12 must actually merge
nearby maxima (fewer peaks) on a noisy patch.
"""

import numpy as np
import pytest

pytest.importorskip("pixell")

from analysis.paper3b.mocks.beam import BeamConfig, apply_beam  # noqa: E402
from analysis.paper3b.mocks.measure import measure_mock_patch  # noqa: E402
from analysis.paper3b.mocks.nz import SourcePlaneWeighting  # noqa: E402
from analysis.paper3b.mocks.patch import (  # noqa: E402
    PatchGeometry,
    coarse_grid_peaks,
    effective_patch_area_deg2,
    find_peaks_flat,
    patch_to_enmap,
    peak_sky_coords,
)
from analysis.paper3b.mocks.shape_noise import ShapeNoiseConfig, noise_map  # noqa: E402
from analysis.paper3b.mocks.smoothing import SmoothingConfig, smooth_flat_sky  # noqa: E402
from analysis.paper3b.stack.cap import cap_filter_multi  # noqa: E402
from analysis.paper3b.stack.stacker import THUMB_R_ARCMIN, THUMB_RES_ARCMIN  # noqa: E402
from analysis.paper3b.maps.stack import extract_thumbnails  # noqa: E402


def _gaussian_bump(n: int, cy: int, cx: int, sigma_px: float, amp: float = 1.0) -> np.ndarray:
    yy, xx = np.indices((n, n))
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma_px ** 2))


# ── coarse_grid_peaks: analytic bumps ────────────────────────────────────────

def test_coarse_grid_peaks_single_bump_right_cell():
    n, block = 120, 12                                   # n_coarse = 10
    ci0, cj0 = 4, 6                                        # interior cell
    cy = ci0 * block + block // 2
    cx = cj0 * block + block // 2
    m = _gaussian_bump(n, cy, cx, sigma_px=1.5)
    ci, cj, coarse = coarse_grid_peaks(m, block=block)
    assert coarse.shape == (n // block, n // block)
    assert len(ci) == 1 and len(cj) == 1
    assert (ci[0], cj[0]) == (ci0, cj0)


def test_coarse_grid_peaks_two_cells_apart_gives_two_peaks():
    n, block = 120, 12
    m = (_gaussian_bump(n, 4 * block + 6, 3 * block + 6, sigma_px=1.5)
         + _gaussian_bump(n, 4 * block + 6, 6 * block + 6, sigma_px=1.5))
    ci, cj, _ = coarse_grid_peaks(m, block=block)
    found = set(zip(ci.tolist(), cj.tolist()))
    assert found == {(4, 3), (4, 6)}


def test_coarse_grid_peaks_same_cell_merges_to_one_peak():
    # Two bumps well-separated enough to be TWO distinct native 8-neighbour
    # maxima, but both landing inside the SAME coarse cell: the coarse grid
    # must merge them into exactly one peak — the audited "merges nearby
    # maxima" effect this module exists to model.
    n, block = 120, 12
    bi, bj = 4, 6
    y0, x0 = bi * block, bj * block
    p1 = (y0 + 4, x0 + 4)
    p2 = (y0 + 8, x0 + 8)
    m = (_gaussian_bump(n, *p1, sigma_px=0.6)
         + _gaussian_bump(n, *p2, sigma_px=0.6))

    # native (find_peaks_flat) sees two distinct local maxima
    pi, pj = find_peaks_flat(m)
    native_found = set(zip(pi.tolist(), pj.tolist()))
    assert p1 in native_found and p2 in native_found

    # coarse grid merges them into one
    ci, cj, _ = coarse_grid_peaks(m, block=block)
    assert len(ci) == 1
    assert (ci[0], cj[0]) == (bi, bj)


# ── block-average correctness ────────────────────────────────────────────────

def test_block_average_correctness_exact_multiple():
    arr = np.arange(81, dtype=float).reshape(9, 9)        # n_coarse = 3, no crop
    _, _, coarse = coarse_grid_peaks(arr, block=3)
    expected = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            expected[a, b] = arr[a * 3:(a + 1) * 3, b * 3:(b + 1) * 3].mean()
    np.testing.assert_allclose(coarse, expected)


def test_block_average_correctness_with_crop():
    # 11x11 at block=3 -> n_coarse=3, n_crop=9: the last 2 rows/cols are
    # dropped BEFORE averaging (documented non-periodic-edge behavior).
    arr = np.arange(121, dtype=float).reshape(11, 11)
    _, _, coarse = coarse_grid_peaks(arr, block=3)
    cropped = arr[:9, :9]
    expected = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            expected[a, b] = cropped[a * 3:(a + 1) * 3, b * 3:(b + 1) * 3].mean()
    np.testing.assert_allclose(coarse, expected)
    # the dropped strip must NOT leak into the averages: changing row/col 9-10
    # (outside the crop) leaves the coarse map unchanged
    arr2 = arr.copy()
    arr2[9:, :] = -1000.0
    arr2[:, 9:] = -1000.0
    _, _, coarse2 = coarse_grid_peaks(arr2, block=3)
    np.testing.assert_allclose(coarse2, coarse)


def test_effective_patch_area_deg2():
    # floor(1024/12) = 85 -> crop to 1020 (drop the last 4 px), NOT 1008/84
    # (see the coarse_grid_peaks docstring "Deviation note").
    geom = PatchGeometry(fov_deg=5.0, npix=1024)
    assert effective_patch_area_deg2(geom, None) == pytest.approx(25.0)
    expected = (1020 * 5.0 / 1024) ** 2
    assert effective_patch_area_deg2(geom, 12) == pytest.approx(expected)
    assert effective_patch_area_deg2(geom, 12) < 25.0


# ── measure_mock_patch integration ───────────────────────────────────────────

SMALL = PatchGeometry(fov_deg=2.5, npix=244)              # NOT a multiple of 12 ->
                                                           # exercises the crop path too
                                                           # (n_coarse=20, n_crop=240)
RES = SMALL.arcmin_per_pixel()
W = SourcePlaneWeighting(np.array([0.5, 1.0, 1.5, 2.0, 2.44]), np.full(5, 0.2))
NOISE_CFG = ShapeNoiseConfig(5.59, SMALL.pixel_area_arcmin2())
SMOOTH_CFG = SmoothingConfig(2.0, RES, "wrap")
BEAM_CFG = BeamConfig(1.6, RES)


def test_peak_grid_block_none_is_bit_identical_to_prior_chain():
    # Regression guard: peak_grid_block=None must reproduce EXACTLY the
    # pre-existing (pre-coarse-grid) native chain, re-implemented here from
    # the primitive building blocks (not by calling the refactored function),
    # so a bug in the refactor cannot hide behind its own code path.
    rng_map = np.random.default_rng(5)
    planes = rng_map.normal(0, 0.01, (5, SMALL.npix, SMALL.npix))
    ymap = rng_map.normal(1e-6, 1e-6, (SMALL.npix, SMALL.npix))

    res = measure_mock_patch(planes, ymap, W, NOISE_CFG, SMOOTH_CFG, BEAM_CFG,
                             rng=np.random.default_rng(42), geom=SMALL)

    # manual re-implementation of the ORIGINAL (pre-change) chain
    kappa_eff = W.effective_map(planes, plane_axis=0)
    kappa_n = kappa_eff + noise_map(kappa_eff.shape, NOISE_CFG, rng=np.random.default_rng(42))
    ksm = smooth_flat_sky(kappa_n, SMOOTH_CFG)
    sigma = float(ksm.std()) + 1e-30
    nu_map = (ksm - ksm.mean()) / sigma
    pi, pj = find_peaks_flat(ksm)
    pnu = nu_map[pi, pj]
    n_all = len(pnu)
    from analysis.paper3b.mocks.measure import NU_STACK_EDGES
    sel = (pnu >= NU_STACK_EDGES[0]) & (pnu < NU_STACK_EDGES[-1])
    pi, pj, pnu = pi[sel], pj[sel], pnu[sel]
    y_beamed = apply_beam(np.asarray(ymap, dtype=np.float64), BEAM_CFG)
    emap = patch_to_enmap(y_beamed, SMALL, 60)
    ra, dec = peak_sky_coords(pi, pj, emap, 60)
    thumbs = np.asarray(extract_thumbnails(emap, ra, dec, THUMB_R_ARCMIN, THUMB_RES_ARCMIN),
                        dtype=np.float64)
    per_peak_y = cap_filter_multi(thumbs, [2.0, 3.0, 4.0, 6.0, 8.0], THUMB_RES_ARCMIN)

    np.testing.assert_array_equal(res.per_peak_nu, pnu)
    np.testing.assert_array_equal(res.per_peak_y, per_peak_y)
    assert res.sigma_kappa_sm == pytest.approx(sigma)
    assert res.n_peaks_all == n_all
    assert res.patch_area_deg2 == pytest.approx(SMALL.fov_deg ** 2)


def test_peak_grid_block_12_finds_fewer_peaks_than_native():
    rng_map = np.random.default_rng(13)
    planes = rng_map.normal(0, 0.01, (5, SMALL.npix, SMALL.npix))
    ymap = rng_map.normal(1e-6, 1e-6, (SMALL.npix, SMALL.npix))
    kw = dict(weighting=W, noise_cfg=NOISE_CFG, smoothing_cfg=SMOOTH_CFG,
             beam_cfg=BEAM_CFG, geom=SMALL)

    native = measure_mock_patch(planes, ymap, rng=np.random.default_rng(99), **kw)
    coarse = measure_mock_patch(planes, ymap, rng=np.random.default_rng(99),
                                peak_grid_block=12, **kw)

    assert coarse.n_peaks_all < native.n_peaks_all
    assert coarse.patch_area_deg2 < native.patch_area_deg2
    n_crop = (SMALL.npix // 12) * 12
    expected_area = (n_crop * SMALL.fov_deg / SMALL.npix) ** 2
    assert coarse.patch_area_deg2 == pytest.approx(expected_area)
    assert native.patch_area_deg2 == pytest.approx(SMALL.fov_deg ** 2)
