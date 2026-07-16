"""WP-B1 map-layer tests on synthetic maps (no survey files, no downloads)."""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import healpy as hp  # noqa: E402

from analysis.paper3b.maps.footprint import harmonize  # noqa: E402
from analysis.paper3b.maps.peaks import (  # noqa: E402
    NU_BIN_EDGES,
    build_peak_catalog,
    exclusion_keep,
    local_maxima,
    map_nu_stats,
    smooth_masked,
)
from analysis.paper3b.maps.stack import (  # noqa: E402
    radial_profile,
    stack_thumbnails,
)

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)


def _delta_map(pixels, amps=1.0):
    m = np.zeros(NPIX)
    m[np.asarray(pixels)] = amps
    return m


# ---------------------------------------------------------------- footprint

def test_harmonize_fractions_and_products():
    des = np.zeros(NPIX, dtype=bool)
    des[: NPIX // 2] = True                       # northern half
    act = np.zeros(NPIX)
    act[NPIX // 4:] = 1.0                         # southern 3/4 fully interior
    act[NPIX // 4: NPIX // 4 + NPIX // 8] = 0.5   # apodized band
    f = harmonize(des, act, act_threshold=0.99)
    assert f.binary.sum() == NPIX // 2 - NPIX // 4 - NPIX // 8
    np.testing.assert_allclose(f.sky_fractions["DES footprint (binary)"], 0.5)
    np.testing.assert_allclose(f.weight[NPIX // 4], 0.5)
    assert not f.binary[NPIX // 4]                # apodized pixel fails 0.99 cut
    assert len(f.summary_lines()) == 5


def test_harmonize_shape_mismatch_raises():
    with pytest.raises(ValueError):
        harmonize(np.ones(NPIX, dtype=bool), np.ones(NPIX // 4))


# -------------------------------------------------------------------- peaks

def test_injected_blob_recovered_as_peak():
    ipk = hp.ang2pix(NSIDE, np.pi / 3, 1.0)
    sm = smooth_masked(_delta_map([ipk]), np.ones(NPIX, dtype=bool), sigma_arcmin=60.0)
    peaks = local_maxima(sm, np.ones(NPIX, dtype=bool))
    assert ipk in peaks                            # blob center is a strict max
    assert sm[ipk] == sm[peaks].max()


def test_local_maxima_strictness_on_constant_map():
    # A constant map has NO strict local maxima.
    assert len(local_maxima(np.ones(NPIX), np.ones(NPIX, dtype=bool))) == 0


def test_nu_stats_and_binning():
    rng = np.random.default_rng(1)
    m = rng.normal(3.0, 2.0, NPIX)
    mean, sigma = map_nu_stats(m, np.ones(NPIX, dtype=bool))
    np.testing.assert_allclose([mean, sigma], [3.0, 2.0], atol=0.05)
    assert len(NU_BIN_EDGES) == 69 and NU_BIN_EDGES[0] == -5.0 and NU_BIN_EDGES[-1] == 12.0
    # stats over the valid footprint only
    half = np.zeros(NPIX, dtype=bool); half[: NPIX // 2] = True
    m2 = m.copy(); m2[NPIX // 2:] = 1e6
    mean2, _ = map_nu_stats(m2, half)
    np.testing.assert_allclose(mean2, mean, atol=0.1)


def test_exclusion_removes_edge_peak_keeps_interior():
    valid = np.ones(NPIX, dtype=bool)
    hole = hp.query_disc(NSIDE, hp.ang2vec(np.pi / 2, 0.0), np.radians(5.0))
    valid[hole] = False
    edge_pix = hp.ang2pix(NSIDE, np.pi / 2, np.radians(5.5))    # ~0.5 deg outside hole
    far_pix = hp.ang2pix(NSIDE, np.pi / 2, np.pi)               # opposite side
    # exclusion radius 2 * 60' = 2 deg: edge peak dies, far peak survives
    keep = exclusion_keep(np.array([edge_pix, far_pix]), valid, sigma_arcmin=60.0)
    assert list(keep) == [False, True]


def test_build_peak_catalog_end_to_end_schema():
    rng = np.random.default_rng(2)
    m = rng.normal(0.0, 1e-3, NPIX)
    ipk = hp.ang2pix(NSIDE, np.pi / 3, 2.0)
    m[ipk] += 1.0
    valid = np.ones(NPIX, dtype=bool)
    cat = build_peak_catalog(m, valid.astype(float), valid, sigma_arcmin=60.0, variant="test")
    assert cat.n_raw == len(cat.ipix) + cat.n_excluded
    assert ipk in cat.ipix
    j = int(np.flatnonzero(cat.ipix == ipk)[0])
    assert cat.nu[j] == cat.nu.max() and cat.nu[j] > 5
    d = cat.to_npz_dict()
    # the frozen nu bins [-5, 12] are a histogram RANGE, not a guarantee: the
    # injected nu >> 12 peak falls outside and the counts must reflect that
    in_range = (cat.nu >= NU_BIN_EDGES[0]) & (cat.nu <= NU_BIN_EDGES[-1])
    assert d["nu_counts"].sum() == in_range.sum() == len(cat.ipix) - 1
    theta, phi = hp.pix2ang(NSIDE, ipk)
    np.testing.assert_allclose(cat.ra_deg[j], np.degrees(phi))
    np.testing.assert_allclose(cat.dec_deg[j], 90 - np.degrees(theta))


# -------------------------------------------------------------------- stack

def test_stack_and_radial_profile_recover_injected_source():
    ny = nx = 31
    y, x = np.indices((ny, nx))
    r2 = (y - 15) ** 2 + (x - 15) ** 2
    thumb = np.exp(-r2 / (2 * 3.0 ** 2))          # Gaussian source, sigma=3px
    thumbs = np.repeat(thumb[None], 10, axis=0)
    st = stack_thumbnails(thumbs)
    np.testing.assert_allclose(st, thumb)
    r, prof = radial_profile(st, res_arcmin=1.0, r_bin_arcmin=2.0, r_max_arcmin=14.0)
    assert prof[0] > 0.8 and prof[-1] < 0.02      # centrally peaked, decays
    assert np.all(np.diff(prof[:5]) < 0)          # monotone decline in the core


def test_stack_weights():
    a = np.zeros((5, 5)); b = np.ones((5, 5))
    st = stack_thumbnails(np.stack([a, b]), weights=np.array([1.0, 3.0]))
    np.testing.assert_allclose(st, 0.75)
