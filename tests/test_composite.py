"""Regression tests for the BIND paste composite (bind.inference.pipeline).

These guard the *default* compositing contract documented in CLAUDE.md and
docs/circular_aperture.md:

    r200_factor = 4.0   (circular R200-scaled aperture, not the legacy square taper)
    paste_mode  = "shared"  (overlapping halos share one realization)

The legacy ``paste_mode="average"`` blends *independent* generative
realizations wherever paste apertures overlap.  Averaging N independent draws
of the same region keeps their conditional mean but divides their stochastic
small-scale variance by ~N, which is what cost the composite ~10% of the
total-matter power at k = 40-70 h/Mpc.  ``test_shared_preserves_overlap_variance``
below is the unit-level statement of exactly that failure.

All geometry here is synthetic and chosen so that the box is 1 Mpc/h per pixel
(``BOX_SIZE == NPIX``), which makes every pixel index in the assertions exact
rather than a floating-point truncation away from it.
"""

import numpy as np
import pytest

from bind.inference import pipeline as P

NPIX = 64
BOX_SIZE = 64.0          # -> pixels_per_mpc == 1.0 exactly
PATCH_PIX = 32
TAPER_FRAC = 0.15
R200 = 2.5               # -> aperture radius = R200 * 4.0 = 10 px < PATCH_PIX // 2
R200_FACTOR = 4.0


def _halo(cx, cy, mass, r200=R200):
    return {
        "halo_center": np.array([float(cx), float(cy)], dtype=np.float64),
        "halo_mass": float(mass),
        "r200": float(r200),
    }


def _support(halo, r200_factor=R200_FACTOR):
    """Full-box weight canvas for one halo, mirroring paste_halos_2d's indexing."""
    ppm = NPIX / BOX_SIZE
    if r200_factor > 0:
        w = P.circular_taper_weight(
            PATCH_PIX, r_pix=halo["r200"] * ppm * r200_factor, taper_frac=TAPER_FRAC
        )
    else:
        w = P.square_taper_weight(PATCH_PIX, taper_frac=TAPER_FRAC)
    half = w.shape[0] // 2
    cx = int(halo["halo_center"][0] * ppm) % NPIX
    cy = int(halo["halo_center"][1] * ppm) % NPIX
    ix = (cx - half + np.arange(w.shape[0])) % NPIX
    iy = (cy - half + np.arange(w.shape[0])) % NPIX
    canvas = np.zeros((NPIX, NPIX), dtype=np.float32)
    canvas[np.ix_(ix, iy)] = w
    return canvas


def _footprint(halo):
    """Index tuple of the (PATCH_PIX, PATCH_PIX) square footprint of a halo."""
    ppm = NPIX / BOX_SIZE
    half = PATCH_PIX // 2
    cx = int(halo["halo_center"][0] * ppm) % NPIX
    cy = int(halo["halo_center"][1] * ppm) % NPIX
    ar = np.arange(PATCH_PIX)
    return np.ix_((cx - half + ar) % NPIX, (cy - half + ar) % NPIX)


def _scene(seed=0):
    """Two deliberately overlapping halos with *independent* random patches.

    Centres are 4 px apart while each paste aperture is 10 px in radius, so the
    apertures overlap heavily and the smaller halo satisfies
    share_overlap_content's "fits inside the host footprint" criterion.
    """
    rng = np.random.default_rng(seed)
    dmo = rng.uniform(1.0, 2.0, (NPIX, NPIX)).astype(np.float32)
    halos = [_halo(16, 32, 1e14), _halo(20, 32, 5e13)]
    # iid lognormal "realizations": same statistics, independent draws.
    patches = np.exp(rng.normal(0.0, 1.0, (2, 3, PATCH_PIX, PATCH_PIX))).astype(np.float32)
    cutouts = [{"condition": dmo[_footprint(h)].copy()} for h in halos]
    return dmo, halos, patches, cutouts


def _composite(dmo, halos, patches, cutouts, *, paste_mode, mass_match=False,
               r200_factor=R200_FACTOR):
    return P.build_bind_composite(
        dmo, halos, patches, cutouts, BOX_SIZE, NPIX, PATCH_PIX,
        mass_match, TAPER_FRAC, r200_factor=r200_factor, paste_mode=paste_mode,
    )


# --------------------------------------------------------------------------
# 1. shared vs average in overlapping apertures
# --------------------------------------------------------------------------

def test_shared_and_average_differ_only_where_apertures_touch():
    """shared != average inside the overlap, == in the host's exclusive region.

    Semantics of the "shared" fix: walking halos by descending mass, the most
    massive halo is the *host* and keeps its own realization; a smaller halo
    whose aperture fits inside the host footprint *adopts* the host's
    realization, rolled to its own frame.  So the two modes agree exactly where
    only the host paints, and disagree everywhere the adopted halo paints
    (both inside the overlap and in the adopted halo's own exclusive region --
    the latter is by design, not a bug).
    """
    dmo, halos, patches, cutouts = _scene()
    shared = _composite(dmo, halos, patches, cutouts, paste_mode="shared")
    average = _composite(dmo, halos, patches, cutouts, paste_mode="average")

    # The smaller halo adopted the more massive halo's realization.
    assert shared["host_idx"].tolist() == [0, 0]
    assert average["host_idx"] is None
    assert shared["paste_mode"] == "shared"

    w_host, w_adopted = _support(halos[0]), _support(halos[1])
    only_host = (w_host > 0) & (w_adopted == 0)
    only_adopted = (w_adopted > 0) & (w_host == 0)
    overlap = (w_host > 0) & (w_adopted > 0)
    assert only_host.sum() > 50 and only_adopted.sum() > 50 and overlap.sum() > 100

    c_shared, c_avg = shared["hydro_canvas"], average["hydro_canvas"]

    # Identical where only the host paints.
    np.testing.assert_allclose(c_shared[:, only_host], c_avg[:, only_host], rtol=1e-6)

    # Different wherever the adopted halo paints.
    for region, name in ((overlap, "overlap"), (only_adopted, "adopted-only")):
        diff = np.abs(c_shared[:, region] - c_avg[:, region])
        scale = np.abs(c_avg[:, region]).mean()
        assert diff.max() > 0.5 * scale, f"{name} region did not change"

    # The paste *weights* are content-independent, so alpha/coverage must match.
    np.testing.assert_allclose(shared["alpha"], average["alpha"], rtol=1e-6)
    assert shared["coverage_pct"] == pytest.approx(average["coverage_pct"])


def test_shared_preserves_overlap_variance():
    """The high-k regression test: averaging independent draws kills variance.

    In the deep overlap (both weights >= 0.9) the legacy "average" mode returns
    ~the mean of two iid fields, whose variance is ~half that of a single draw.
    "shared" pastes one realization there, so its variance is preserved.
    """
    dmo, halos, patches, cutouts = _scene()
    shared = _composite(dmo, halos, patches, cutouts, paste_mode="shared")
    average = _composite(dmo, halos, patches, cutouts, paste_mode="average")

    core = (_support(halos[0]) >= 0.9) & (_support(halos[1]) >= 0.9)
    assert core.sum() > 100

    std_shared = float(shared["hydro_canvas"][:, core].std())
    std_avg = float(average["hydro_canvas"][:, core].std())
    # Expectation for two iid fields is 1/sqrt(2) ~ 0.71; assert a clear loss.
    assert std_avg / std_shared < 0.85


def test_shared_matches_average_for_isolated_halos():
    """With no overlap the two modes must agree (up to the mass-match path)."""
    rng = np.random.default_rng(3)
    dmo = rng.uniform(1.0, 2.0, (NPIX, NPIX)).astype(np.float32)
    halos = [_halo(16, 16, 1e14), _halo(48, 48, 5e13)]   # 32 px apart, apertures 10 px
    patches = np.exp(rng.normal(0.0, 1.0, (2, 3, PATCH_PIX, PATCH_PIX))).astype(np.float32)
    cutouts = [{"condition": dmo[_footprint(h)].copy()} for h in halos]

    shared = _composite(dmo, halos, patches, cutouts, paste_mode="shared")
    average = _composite(dmo, halos, patches, cutouts, paste_mode="average")
    assert shared["host_idx"].tolist() == [0, 1]     # nobody adopted anybody
    np.testing.assert_allclose(shared["composite"], average["composite"], rtol=1e-5)


def test_unknown_paste_mode_rejected():
    dmo, halos, patches, cutouts = _scene()
    with pytest.raises(ValueError, match="paste_mode"):
        _composite(dmo, halos, patches, cutouts, paste_mode="bogus")


# --------------------------------------------------------------------------
# 2. mass conservation
# --------------------------------------------------------------------------

def test_shared_mass_match_is_aperture_local():
    """patch_mass_match in "shared" mode matches the *weighted* paste aperture.

    Contract: after the match, the weighted content mass inside a halo's paste
    aperture equals the weighted DMO mass in the same aperture.  (Whole-patch
    totals are meaningless in shared mode because adopted content is rolled.)
    """
    rng = np.random.default_rng(7)
    dmo = rng.uniform(1.0, 2.0, (NPIX, NPIX)).astype(np.float32)
    halos = [_halo(32, 32, 1e14)]
    patches = np.exp(rng.normal(0.0, 1.0, (1, 3, PATCH_PIX, PATCH_PIX))).astype(np.float32)
    cutouts = [{"condition": dmo[_footprint(halos[0])].copy()}]

    res = _composite(dmo, halos, patches, cutouts, paste_mode="shared", mass_match=True)

    w = P.circular_taper_weight(
        PATCH_PIX, r_pix=R200 * (NPIX / BOX_SIZE) * R200_FACTOR, taper_frac=TAPER_FRAC
    )
    fp = _footprint(halos[0])
    m_dmo = float((dmo[fp] * w).sum())
    m_composited = float((res["hydro_canvas"].sum(0)[fp] * w).sum())
    assert m_composited == pytest.approx(m_dmo, rel=1e-5)
    assert res["patch_scales"].shape == (1,)
    assert res["patch_scales"][0] > 0


def test_legacy_mass_match_is_whole_patch():
    """paste_mode="average" keeps the legacy whole-patch/whole-cutout ratio."""
    rng = np.random.default_rng(11)
    dmo = rng.uniform(1.0, 2.0, (NPIX, NPIX)).astype(np.float32)
    halos = [_halo(32, 32, 1e14)]
    patches = np.exp(rng.normal(0.0, 1.0, (1, 3, PATCH_PIX, PATCH_PIX))).astype(np.float32)
    cutouts = [{"condition": dmo[_footprint(halos[0])].copy()}]

    res = _composite(dmo, halos, patches, cutouts, paste_mode="average", mass_match=True)
    expected = float(cutouts[0]["condition"].sum()) / float(patches[0].sum())
    assert res["patch_scales"][0] == pytest.approx(expected, rel=1e-6)


def test_composite_conserves_total_box_mass():
    """scale_global renormalizes the composite to the DMO total-matter budget."""
    dmo, halos, patches, cutouts = _scene(seed=5)
    for mode in ("shared", "average"):
        res = _composite(dmo, halos, patches, cutouts, paste_mode=mode, mass_match=True)
        assert float(res["composite"].sum()) == pytest.approx(float(dmo.sum()), rel=1e-5)


# --------------------------------------------------------------------------
# 3. circular aperture vs legacy square taper
# --------------------------------------------------------------------------

def test_circular_aperture_footprint_differs_from_square_taper():
    """r200_factor=4.0 confines the paste to a disk; r200_factor=0 fills the square."""
    rng = np.random.default_rng(13)
    dmo = rng.uniform(1.0, 2.0, (NPIX, NPIX)).astype(np.float32)
    halos = [_halo(32, 32, 1e14)]
    patches = np.exp(rng.normal(0.0, 1.0, (1, 3, PATCH_PIX, PATCH_PIX))).astype(np.float32)
    cutouts = [{"condition": dmo[_footprint(halos[0])].copy()}]

    circular = _composite(dmo, halos, patches, cutouts, paste_mode="shared", r200_factor=4.0)
    square = _composite(dmo, halos, patches, cutouts, paste_mode="average", r200_factor=0.0)

    n_circ = int((circular["alpha"] > 0).sum())
    n_sq = int((square["alpha"] > 0).sum())
    assert n_circ < n_sq
    # A pixel just inside the square footprint's corner taper: covered by the
    # square taper, outside the 10 px disk.
    corner = (32 - PATCH_PIX // 2 + 4, 32 - PATCH_PIX // 2 + 4)
    assert square["alpha"][corner] > 0.5
    assert circular["alpha"][corner] == 0.0
    # r200_factor <= 0 always takes the legacy path, even asking for "shared".
    legacy_shared = _composite(dmo, halos, patches, cutouts, paste_mode="shared",
                               r200_factor=0.0)
    assert legacy_shared["host_idx"] is None


def test_circular_taper_weight_falls_back_to_square_without_r200():
    """A halo with no R200c gets the square taper (documented per-halo fallback)."""
    circ = P.circular_taper_weight(PATCH_PIX, r_pix=0.0, taper_frac=TAPER_FRAC)
    np.testing.assert_allclose(circ, P.square_taper_weight(PATCH_PIX, TAPER_FRAC))


# --------------------------------------------------------------------------
# 4. every entry point must default to the standard 4.0 / "shared"
# --------------------------------------------------------------------------

def _cli_defaults(module, monkeypatch):
    """Return the argparse defaults of a `parse_args()`-style CLI module.

    Intercepts ``ArgumentParser.parse_args`` so the parser is built (and its
    defaults readable) without needing the CLI's required arguments.
    """
    import argparse

    captured = {}

    def fake_parse_args(self, *args, **kwargs):
        captured["parser"] = self
        return argparse.Namespace()

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)
    module.parse_args()
    return captured["parser"]


@pytest.mark.parametrize(
    "module_name",
    [
        "bind.cli.paint",
        "bind.cli.paint_generate",
        "bind.cli.paint_recomposite",
        "bind.cli.camels_suite",
    ],
)
def test_cli_defaults_are_the_standard_composite(module_name, monkeypatch):
    import importlib

    parser = _cli_defaults(importlib.import_module(module_name), monkeypatch)
    assert parser.get_default("r200_factor") == 4.0
    assert parser.get_default("paste_mode") == "shared"


def test_python_api_defaults_are_the_standard_composite():
    import inspect

    from bind import paint
    from bind.inference.schemas import RunConfig

    sig = inspect.signature(paint)
    assert sig.parameters["r200_factor"].default == 4.0
    assert sig.parameters["paste_mode"].default == "shared"

    composite_sig = inspect.signature(P.build_bind_composite)
    assert composite_sig.parameters["r200_factor"].default == 4.0
    assert composite_sig.parameters["paste_mode"].default == "shared"

    cfg_fields = {f.name: f for f in RunConfig.__dataclass_fields__.values()}
    assert cfg_fields["r200_factor"].default == 4.0
    assert cfg_fields["paste_mode"].default == "shared"
