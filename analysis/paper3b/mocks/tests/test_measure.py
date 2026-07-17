"""WP-B4 matched-measurement tests: shared-code-path guard + planted-source checks."""

import numpy as np
import pytest

pytest.importorskip("pixell")

from analysis.paper3b.mocks import measure  # noqa: E402
from analysis.paper3b.mocks.beam import BeamConfig  # noqa: E402
from analysis.paper3b.mocks.measure import (  # noqa: E402
    MockEnsembleAccumulator,
    measure_mock_patch,
)
from analysis.paper3b.mocks.nz import SourcePlaneWeighting  # noqa: E402
from analysis.paper3b.mocks.patch import (  # noqa: E402
    PatchGeometry,
    find_peaks_flat,
    patch_to_enmap,
    peak_sky_coords,
)
from analysis.paper3b.mocks.shape_noise import ShapeNoiseConfig  # noqa: E402
from analysis.paper3b.mocks.smoothing import SmoothingConfig  # noqa: E402

GEOM = PatchGeometry(fov_deg=5.0, npix=256)      # coarse test patch (1.17'/px)
RES = GEOM.arcmin_per_pixel()
NOISELESS = ShapeNoiseConfig(n_eff_arcmin2=np.inf, pix_area_arcmin2=GEOM.pixel_area_arcmin2())


def test_shared_code_path_import_identity():
    # B4 acceptance criterion: mock and data measurements share ONE code path.
    # The mock module must use the very same function objects and constants as
    # the frozen B2 chain — not copies.
    from analysis.paper3b.maps import stack as maps_stack
    from analysis.paper3b.stack import cap as stack_cap
    from analysis.paper3b.stack import stacker

    assert measure.extract_thumbnails is maps_stack.extract_thumbnails
    assert measure.cap_filter_multi is stack_cap.cap_filter_multi
    assert measure.CAP_RADII_ARCMIN is stacker.CAP_RADII_ARCMIN
    assert measure.NU_STACK_EDGES is stacker.NU_STACK_EDGES
    assert measure.THUMB_R_ARCMIN == stacker.THUMB_R_ARCMIN == 15.0
    assert measure.THUMB_RES_ARCMIN == stacker.THUMB_RES_ARCMIN == 0.5
    assert measure.FIDUCIAL_RADIUS_INDEX == stacker.FIDUCIAL_RADIUS_INDEX == 2


def test_find_peaks_flat_matches_bruteforce():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(40, 40))
    pi, pj = find_peaks_flat(m)
    found = set(zip(pi.tolist(), pj.tolist()))
    n = m.shape[0]
    for i in range(n):
        for j in range(n):
            nb = [m[(i + di) % n, (j + dj) % n]
                  for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
            assert ((i, j) in found) == bool(m[i, j] > max(nb))


def test_patch_roundtrip_center_and_edge():
    # A delta at the patch center and one near the edge must both come back at
    # the right sky position (periodic pad keeps edge thumbnails valid).
    p = np.zeros((GEOM.npix, GEOM.npix))
    p[GEOM.npix // 2, GEOM.npix // 2] = 1.0
    p[2, 3] = 1.0
    em = patch_to_enmap(p, GEOM, pad_pix=20)
    ra, dec = peak_sky_coords(np.array([GEOM.npix // 2, 2]), np.array([GEOM.npix // 2, 3]),
                              em, pad_pix=20)
    assert abs(dec[0]) < 1e-9 and abs(ra[0]) < 1e-9
    # edge pixel: |dec| just inside the 2.5 deg half-FOV
    assert 2.3 < abs(dec[1]) <= 2.5 and abs(ra[1]) <= 2.5


def test_planted_cluster_recovers_analytic_cap():
    # kappa: one strong Gaussian peak at center; y: Gaussian source at the same
    # spot. The chain must find the peak and the CAP Y must match the analytic
    # compensated-aperture value of a Gaussian (computed on the same pixel
    # grid via cap_filter directly).
    from analysis.paper3b.stack.cap import cap_filter

    n = GEOM.npix
    yy, xx = np.indices((n, n))
    r2 = ((yy - n / 2) ** 2 + (xx - n / 2) ** 2) * RES ** 2
    kappa = 0.05 * np.exp(-r2 / (2 * 3.0 ** 2))          # 3' kappa blob
    ymap = 1e-5 * np.exp(-r2 / (2 * 2.0 ** 2))           # 2' y source
    planes = np.repeat(kappa[None], 5, axis=0)
    w = SourcePlaneWeighting(np.array([0.5, 1.0, 1.5, 2.0, 2.44]),
                             np.array([0.2] * 5))
    res = measure_mock_patch(
        planes, ymap, w, NOISELESS,
        SmoothingConfig(2.0, RES, "wrap"),
        BeamConfig(0.0, RES),                             # no beam: exact analytic ref
        rng=np.random.default_rng(0), geom=GEOM,
        # a lone blob on a flat patch has nu ~ 60 — outside the frozen edges,
        # which is correct behavior; widen the window to test the chain itself
        nu_edges=np.array([0.0, 1e3]),
    )
    assert res.per_peak_nu.size >= 1
    top = int(np.argmax(res.per_peak_nu))
    # analytic reference: CAP of the same y map sampled on the 0.5' thumb grid
    ty, tx = np.indices((61, 61))
    tr2 = ((ty - 30) ** 2 + (tx - 30) ** 2) * 0.5 ** 2
    ref_thumb = 1e-5 * np.exp(-tr2 / (2 * 2.0 ** 2))
    ref = cap_filter(ref_thumb[None], 4.0, 0.5)[0]
    got = res.per_peak_y[top, 2]                          # 4' = FIDUCIAL_RADIUS_INDEX
    assert got == pytest.approx(ref, rel=0.05)


def test_noise_changes_selection_but_seeding_reproduces():
    small = PatchGeometry(fov_deg=2.5, npix=128)          # keep the peak count low
    res = small.arcmin_per_pixel()
    rng_map = np.random.default_rng(7)
    planes = rng_map.normal(0, 0.01, (5, small.npix, small.npix))
    ymap = rng_map.normal(0, 1e-6, (small.npix, small.npix))
    w = SourcePlaneWeighting(np.array([0.5, 1.0, 1.5, 2.0, 2.44]), np.full(5, 0.2))
    noisy = ShapeNoiseConfig(5.59, small.pixel_area_arcmin2())
    kw = dict(weighting=w, noise_cfg=noisy,
              smoothing_cfg=SmoothingConfig(2.0, res, "wrap"),
              beam_cfg=BeamConfig(1.6, res), geom=small)
    a = measure_mock_patch(planes, ymap, rng=np.random.default_rng((3, 0, 1)), **kw)
    b = measure_mock_patch(planes, ymap, rng=np.random.default_rng((3, 0, 1)), **kw)
    c = measure_mock_patch(planes, ymap, rng=np.random.default_rng((3, 0, 2)), **kw)
    np.testing.assert_array_equal(a.per_peak_nu, b.per_peak_nu)
    np.testing.assert_array_equal(a.per_peak_y, b.per_peak_y)
    assert a.per_peak_nu.size != c.per_peak_nu.size or not np.array_equal(a.per_peak_nu, c.per_peak_nu)


def test_accumulator_pools_and_errors():
    rng = np.random.default_rng(11)
    acc = MockEnsembleAccumulator()
    w = SourcePlaneWeighting(np.array([0.5, 1.0, 1.5, 2.0, 2.44]), np.full(5, 0.2))
    for s in range(4):
        planes = rng.normal(0, 0.01, (5, GEOM.npix, GEOM.npix))
        ymap = rng.normal(1e-6, 1e-6, (GEOM.npix, GEOM.npix))
        acc.add(measure_mock_patch(planes, ymap, w, NOISELESS,
                                   SmoothingConfig(2.0, RES, "wrap"),
                                   BeamConfig(1.6, RES),
                                   rng=np.random.default_rng(s), geom=GEOM))
    arrs = acc.arrays()
    assert arrs["counts"].shape == (4, 5) and arrs["y_sums"].shape == (4, 5, 5)
    s = acc.summary()
    assert s["y_mean"].shape == (5, 5)
    occ = s["n_per_bin"] > 0
    assert np.isfinite(s["y_mean"][occ]).all()
    # Gaussian random maps: low-nu bins are populated in every patch
    assert np.isfinite(s["y_mc_err"][0]).all()
