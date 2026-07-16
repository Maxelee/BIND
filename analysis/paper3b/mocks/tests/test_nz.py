"""Analytic-fixture tests for WP-B4 source-plane n(z) weighting (no data files)."""

import numpy as np
import pytest

from analysis.paper3b.mocks.nz import (
    DESY3_BIN_MEAN_Z,
    SourcePlaneWeighting,
    smail_nz,
    source_plane_weights,
)

# twobound atlas source planes.
PLANES = np.array([0.5, 1.0, 1.5, 2.0, 2.44])


def _plane_maps(constants, ny=4, nx=3):
    """Stack of per-plane maps, each plane uniform at a distinct constant."""
    return np.stack([np.full((ny, nx), c) for c in constants], axis=0)


def _bump(zt, center, amp, width=0.04):
    """A narrow Gaussian n(z) bump. On a uniform grid with `center` on a grid
    node, two same-width bumps integrate to identical mass to machine precision,
    so equal/ratio-mass fixtures are exact (a boxcar's fractional edge cells are
    not)."""
    return amp * np.exp(-0.5 * ((zt - center) / width) ** 2)


def test_delta_nz_returns_that_plane_exactly():
    # n(z) as a narrow tophat fully inside plane index 2's (z=1.5) interval
    # (midpoints 1.25 and 1.75). Binning must put all weight on that plane.
    zt = np.linspace(1.4, 1.6, 201)
    nt = np.ones_like(zt)
    w = source_plane_weights(PLANES, zt, nt)
    assert w == pytest.approx([0, 0, 1, 0, 0], abs=1e-12)

    maps = _plane_maps([10.0, 20.0, 30.0, 40.0, 50.0])
    weighting = SourcePlaneWeighting.from_nz(PLANES, zt, nt)
    eff = weighting.effective_map(maps, plane_axis=0)
    assert np.array_equal(eff, np.full((4, 3), 30.0))


def test_from_delta_helper_matches_plane():
    maps = _plane_maps([10.0, 20.0, 30.0, 40.0, 50.0])
    w = SourcePlaneWeighting.from_delta(PLANES, 2.0)  # plane index 3
    assert np.array_equal(w.effective_map(maps, plane_axis=0), np.full((4, 3), 40.0))
    with pytest.raises(ValueError, match="not a source plane"):
        SourcePlaneWeighting.from_delta(PLANES, 1.234)


def test_two_plane_nz_gives_weighted_average():
    # Equal-mass bumps inside plane-1 (z=1.0) and plane-3 (z=2.0) intervals.
    zt = np.linspace(0.0, 3.0, 3001)  # z=1.0, 2.0 fall on grid nodes
    nt = _bump(zt, 1.0, 1.0) + _bump(zt, 2.0, 1.0)
    w = source_plane_weights(PLANES, zt, nt)
    # ~1e-8 tail of the z=2.0 bump leaks across the 2.22 midpoint into plane 4
    # (physically correct); comfortably tighter than a boxcar's ~1e-3 edge cells.
    assert w == pytest.approx([0.0, 0.5, 0.0, 0.5, 0.0], abs=1e-6)

    maps = _plane_maps([10.0, 20.0, 30.0, 40.0, 50.0])
    eff = SourcePlaneWeighting.from_nz(PLANES, zt, nt).effective_map(maps, plane_axis=0)
    assert eff == pytest.approx(np.full((4, 3), 0.5 * (20.0 + 40.0)), abs=1e-4)


def test_unequal_two_plane_weights_track_nz_mass():
    # 3:1 mass split between plane-0 (z=0.5) and plane-1 (z=1.0) -> 0.75/0.25.
    zt = np.linspace(0.0, 1.6, 1601)  # z=0.5, 1.0 fall on grid nodes
    nt = _bump(zt, 0.5, 3.0) + _bump(zt, 1.0, 1.0)
    w = source_plane_weights(PLANES, zt, nt)
    assert w == pytest.approx([0.75, 0.25, 0.0, 0.0, 0.0], abs=1e-6)


def test_weights_normalized_and_order_preserved():
    # Unsorted planes: weights must come back aligned to the input order.
    planes = np.array([2.0, 0.5, 1.5])
    z = np.linspace(0, 3, 3001)
    n = smail_nz(z, z0=0.7)
    w = source_plane_weights(planes, z, n)
    assert w.sum() == pytest.approx(1.0)
    # Same physical distribution, sorted input -> same weights, reordered.
    ws = source_plane_weights(np.array([0.5, 1.5, 2.0]), z, n)
    assert w == pytest.approx(ws[[2, 0, 1]])


def test_effective_map_on_atlas_layout_axis1():
    # twobound kappa layout (n_real, K, ny, nx): weight over plane axis=1.
    rng = np.random.default_rng(0)
    maps = rng.normal(size=(3, 5, 6, 7))
    w = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    weighting = SourcePlaneWeighting(PLANES, w)
    eff = weighting.effective_map(maps, plane_axis=1)
    assert eff.shape == (3, 6, 7)
    assert eff == pytest.approx(np.tensordot(w, maps, axes=([0], [1])))


def test_smail_positive_and_zero_below_zero():
    z = np.array([-0.5, 0.0, 0.5, 1.0, 2.0])
    n = smail_nz(z)
    assert n[0] == 0.0 and n[1] == 0.0
    assert np.all(n[2:] > 0)


def test_des_bin_mean_reference_is_documentation_only():
    # Guard the documented reference values (not used as a weighting input).
    assert DESY3_BIN_MEAN_Z == (0.34, 0.52, 0.74, 0.90)
