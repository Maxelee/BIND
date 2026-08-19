"""NormStats forward/inverse round trips and the back-compat load contract.

``NormStats`` is the contract between training and inference: ``AstroDataset``
applies the forward transform, ``bind.inference.pipeline._denormalize_to_physical``
applies the inverse.  These tests drive the *real* forward path (a synthetic
``.npz`` through ``AstroDataset``) rather than reimplementing it, so a change to
either side that is not mirrored on the other fails here.
"""

import numpy as np
import pytest

from bind import data as D
from bind.data import N_THERMO, THERMO_KEYS, AstroDataset, NormStats
from bind.inference.pipeline import _denormalize_to_physical

H = 16
RTOL = 1e-4

# Realistic-ish log10(1+x) statistics for [DM_hydro, Gas, Stars].
MASS_MEAN = np.array([1.0, 0.8, 0.4], dtype=np.float64)
MASS_STD = np.array([0.5, 0.6, 0.3], dtype=np.float64)

# Per-channel log10 statistics for (compton_y, temperature, entropy, pressure).
THERMO_MEAN = np.array([-6.0, 7.0, 3.0, -12.0], dtype=np.float64)
THERMO_STD = np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float64)
THERMO_FLOOR = np.array([1e-9, 1e3, 1.0, 1e-16], dtype=np.float64)


def _write_sample(tmp_path, target, thermo=None, name="sample.npz", extra=None):
    """Write one synthetic training-format sample and return its path."""
    rng = np.random.default_rng(0)
    payload = {
        "target": target.astype(np.float32),
        "condition": rng.uniform(0.0, 20.0, (H, H)).astype(np.float32),
        "large_scale": rng.uniform(0.0, 20.0, (3, H, H)).astype(np.float32),
        "params": D.PARAM_MIN_RAW.astype(np.float32)
        + 0.5 * (D.PARAM_MAX_RAW - D.PARAM_MIN_RAW).astype(np.float32),
    }
    if thermo is not None:
        payload.update({k: v.astype(np.float32) for k, v in zip(THERMO_KEYS, thermo)})
    if extra:
        payload.update(extra)
    path = tmp_path / name
    np.savez(path, **payload)
    return str(path)


def _forward(path, ns):
    """Run the real dataset forward transform and return the (1, C, H, W) target."""
    item = AstroDataset([path], ns)[0]
    return item["target"].numpy()[None], item


def _mass_target(seed=0, stars_zero_frac=0.0):
    rng = np.random.default_rng(seed)
    target = rng.uniform(0.0, 50.0, (3, H, H))
    if stars_zero_frac:
        holes = rng.random((H, H)) < stars_zero_frac
        target[2] = np.where(holes, 0.0, np.clip(target[2], 1e-3, None))
    return target.astype(np.float32)


# --------------------------------------------------------------------------
# mass channels: log10(1 + x) standardization
# --------------------------------------------------------------------------

def test_mass_channels_round_trip(tmp_path):
    target = _mass_target()
    ns = NormStats(target_mean=MASS_MEAN, target_std=MASS_STD)
    normed, item = _forward(_write_sample(tmp_path, target), ns)

    assert normed.shape == (1, 3, H, H)
    physical = _denormalize_to_physical(normed, ns)
    assert physical.shape == (1, 3, H, H)
    np.testing.assert_allclose(physical[0], target, rtol=RTOL)

    # Params are min/max scaled into the unit interval (log10 first where flagged).
    p = item["params"].numpy()
    assert p.shape == (35,)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_mass_channels_zero_round_trips_to_zero(tmp_path):
    """log10(1+x) is exact at x = 0, so empty pixels must come back as 0."""
    target = np.zeros((3, H, H), dtype=np.float32)
    ns = NormStats(target_mean=MASS_MEAN, target_std=MASS_STD)
    normed, _ = _forward(_write_sample(tmp_path, target), ns)
    physical = _denormalize_to_physical(normed, ns)
    np.testing.assert_allclose(physical[0], 0.0, atol=1e-5)


# --------------------------------------------------------------------------
# thermo channels: plain log10 with a per-channel floor (a different code path)
# --------------------------------------------------------------------------

def test_thermo_channels_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    decades = 10.0 ** rng.uniform(-0.8, 0.8, (N_THERMO, H, H))
    thermo = (10.0 ** THERMO_MEAN)[:, None, None] * decades   # spans the mean +/- ~1 dex
    target = _mass_target(seed=2)

    ns = NormStats(
        target_mean=MASS_MEAN, target_std=MASS_STD,
        predict_thermo=True, thermo_mean=THERMO_MEAN,
        thermo_std=THERMO_STD, thermo_floor=THERMO_FLOOR,
    )
    normed, _ = _forward(_write_sample(tmp_path, target, thermo=thermo), ns)
    assert normed.shape == (1, 3 + N_THERMO, H, H)

    physical = _denormalize_to_physical(normed, ns)
    assert physical.shape == (1, 3 + N_THERMO, H, H)
    np.testing.assert_allclose(physical[0, :3], target, rtol=RTOL)
    np.testing.assert_allclose(physical[0, 3:], thermo, rtol=RTOL)


def test_thermo_zero_pixels_round_trip_to_the_floor(tmp_path):
    """log10 is not zero-safe: an empty thermo pixel returns as thermo_floor.

    Documented behaviour (``thermo_forward`` clamps to the floor), not a bug --
    the floor is the 0.1st percentile of the positive pixels.
    """
    thermo = np.zeros((N_THERMO, H, H), dtype=np.float32)
    ns = NormStats(
        target_mean=MASS_MEAN, target_std=MASS_STD,
        predict_thermo=True, thermo_mean=THERMO_MEAN,
        thermo_std=THERMO_STD, thermo_floor=THERMO_FLOOR,
    )
    normed, _ = _forward(_write_sample(tmp_path, _mass_target(3), thermo=thermo), ns)
    physical = _denormalize_to_physical(normed, ns)
    for j in range(N_THERMO):
        np.testing.assert_allclose(physical[0, 3 + j], THERMO_FLOOR[j], rtol=RTOL)


# --------------------------------------------------------------------------
# two-head stars: occupancy mask + conditional log-density
# --------------------------------------------------------------------------

def test_stars_two_head_round_trip(tmp_path):
    target = _mass_target(seed=4, stars_zero_frac=0.4)
    occupied = target[2] > 0
    assert 0 < occupied.mean() < 1        # the split is only meaningful if both exist

    p = float(occupied.mean())
    ns = NormStats(
        target_mean=MASS_MEAN, target_std=MASS_STD,
        stars_two_head=True,
        stars_occ_mean=p, stars_occ_std=float(np.sqrt(p * (1 - p))),
        stars_cond_mean=float(np.log10(1 + target[2][occupied]).mean()),
        stars_cond_std=float(np.log10(1 + target[2][occupied]).std()),
    )
    normed, _ = _forward(_write_sample(tmp_path, target), ns)
    assert normed.shape == (1, 4, H, H)   # DM, Gas, occupancy, conditional density

    physical = _denormalize_to_physical(normed, ns)
    assert physical.shape == (1, 3, H, H)  # recombined back to [DM, Gas, Stars]
    np.testing.assert_allclose(physical[0, :2], target[:2], rtol=RTOL)
    # The hard 0.5 occupancy gate must be exact on both sides.
    np.testing.assert_array_equal(physical[0, 2] > 0, occupied)
    np.testing.assert_allclose(physical[0, 2][occupied], target[2][occupied], rtol=RTOL)
    np.testing.assert_array_equal(physical[0, 2][~occupied], 0.0)


def test_denormalize_rejects_wrong_channel_count():
    ns = NormStats(target_mean=MASS_MEAN, target_std=MASS_STD, stars_two_head=True)
    with pytest.raises(ValueError, match="channels"):
        _denormalize_to_physical(np.zeros((1, 3, H, H), dtype=np.float32), ns)


# --------------------------------------------------------------------------
# back-compat: an old norm_stats.npz must load with safe defaults
# --------------------------------------------------------------------------

def test_load_legacy_norm_stats_takes_safe_defaults(tmp_path):
    """A pre-two-head, pre-thermo norm_stats.npz must still load and behave
    exactly like the original mass-only pipeline (documented in CLAUDE.md)."""
    path = tmp_path / "norm_stats_legacy.npz"
    np.savez(
        path,
        target_mean=MASS_MEAN, target_std=MASS_STD,
        cond_mean=1.0, cond_std=0.5,
        ls_mean=np.zeros(3), ls_std=np.ones(3),
        param_min=D.PARAM_MIN_NORM, param_max=D.PARAM_MAX_NORM,
    )
    ns = NormStats.load(path)

    # Preserved.
    np.testing.assert_allclose(ns.target_mean, MASS_MEAN)
    assert ns.cond_mean == 1.0 and ns.cond_std == 0.5

    # Missing feature flags default off -> the legacy 3-channel pipeline.
    assert ns.stars_two_head is False
    assert ns.predict_thermo is False
    assert ns.condition_observables is False
    assert ns.mask_observables is False

    # Missing statistics default to identity transforms.
    assert ns.stars_occ_mean == 0.0 and ns.stars_occ_std == 1.0
    assert ns.stars_cond_mean == 0.0 and ns.stars_cond_std == 1.0
    np.testing.assert_allclose(ns.thermo_mean, np.zeros(N_THERMO))
    np.testing.assert_allclose(ns.thermo_std, np.ones(N_THERMO))
    np.testing.assert_allclose(ns.thermo_floor, np.ones(N_THERMO))

    # A missing param_log_flag falls back to the packaged SB35 flags.
    np.testing.assert_array_equal(ns.param_log_flag, D.PARAM_LOG_FLAG)

    # And it still round-trips mass channels.
    physical = _denormalize_to_physical(np.zeros((1, 3, H, H), dtype=np.float32), ns)
    np.testing.assert_allclose(physical[0, :, 0, 0], 10.0 ** MASS_MEAN - 1.0, rtol=RTOL)


def test_norm_stats_save_load_preserves_flag_types(tmp_path):
    """np.savez stores bools as 0-d arrays; load() must cast them back."""
    ns = NormStats(
        target_mean=MASS_MEAN, target_std=MASS_STD, cond_mean=0.7, cond_std=0.3,
        stars_two_head=True, stars_occ_mean=0.4, stars_occ_std=0.49,
        predict_thermo=True, thermo_mean=THERMO_MEAN,
        thermo_std=THERMO_STD, thermo_floor=THERMO_FLOOR,
    )
    path = tmp_path / "norm_stats.npz"
    ns.save(path)
    loaded = NormStats.load(path)

    assert loaded.stars_two_head is True
    assert loaded.predict_thermo is True
    assert isinstance(loaded.cond_mean, float) and loaded.cond_mean == pytest.approx(0.7)
    assert loaded.stars_occ_std == pytest.approx(0.49)
    np.testing.assert_allclose(loaded.thermo_floor, THERMO_FLOOR)
    np.testing.assert_allclose(loaded.target_std, MASS_STD)


# --------------------------------------------------------------------------
# misc contracts on the inverse transform
# --------------------------------------------------------------------------

def test_denormalize_clips_mass_to_non_negative():
    """10**x - 1 goes negative for x < 0; painted mass never may."""
    ns = NormStats(target_mean=MASS_MEAN, target_std=MASS_STD)
    very_negative = np.full((1, 3, 4, 4), -20.0, dtype=np.float32)
    physical = _denormalize_to_physical(very_negative, ns)
    assert (physical >= 0).all()


# --------------------------------------------------------------------------
# redshift conditioning: the scale factor a = 1/(1+z) the model is fed
# --------------------------------------------------------------------------

def test_scale_factor_is_read_from_the_snapshot_directory(tmp_path):
    ns = NormStats(target_mean=MASS_MEAN, target_std=MASS_STD)
    snap_dir = tmp_path / "sim_0" / "snap_60"
    snap_dir.mkdir(parents=True)
    path = _write_sample(snap_dir, _mass_target(9))

    item = AstroDataset([path], ns, condition_redshift=True)[0]
    expected = 1.0 / (1.0 + D.SNAPSHOT_REDSHIFTS[60])
    assert float(item["scale_factor"]) == pytest.approx(expected, rel=1e-6)

    # A non-redshift dataset must not emit the key at all (so a mass-only model
    # is never handed a scale factor it was not trained on).
    assert "scale_factor" not in AstroDataset([path], ns)[0]


def test_explicit_redshift_key_overrides_the_path(tmp_path):
    ns = NormStats(target_mean=MASS_MEAN, target_std=MASS_STD)
    snap_dir = tmp_path / "snap_60"
    snap_dir.mkdir()
    path = _write_sample(snap_dir, _mass_target(10),
                         extra={"redshift": np.float32(2.0)})
    item = AstroDataset([path], ns, condition_redshift=True)[0]
    assert float(item["scale_factor"]) == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_unknown_snapshot_raises_rather_than_guessing_a_redshift(tmp_path):
    ns = NormStats(target_mean=MASS_MEAN, target_std=MASS_STD)
    snap_dir = tmp_path / "snap_99"
    snap_dir.mkdir()
    path = _write_sample(snap_dir, _mass_target(11))
    with pytest.raises(KeyError):
        AstroDataset([path], ns, condition_redshift=True)[0]
