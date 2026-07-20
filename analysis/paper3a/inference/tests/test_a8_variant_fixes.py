"""Tests for the 2026-07-20 systematics-hunt fixes (torch3 venv; ceph artifacts).

Covers three things, all introduced together:

1. `BIND_PAPER3A_LOGM500_SHIFT` actually propagating into the kSZ
   prediction (the BUG FIX — FINDINGS_ksz.md defect 3), including the
   requirement that the DEFAULT path is an exact no-op.
2. `ksz_wp2bias` — the wp2 tau_CAP validation-bias VARIANT.
3. `c2s_massdep` — the mass-dependent CylToSph VARIANT.

The structural test worth reading is
`test_full_bin_span_shift_reproduces_the_other_gate_bin`: shifting by
exactly the two gate bins' separation must return the bin-1 stack to
machine precision. That is a statement about the interpolation being
correctly anchored, and — unlike "does +0.10 dex give 1.18?" — it is not
circular with the audit's own 0.722 slope measurement, because both
endpoints are pinned by construction rather than fitted.

KszBlock init is expensive (basis probing + MC dilution), so the shifted
blocks are module-scoped and the cheap variants are exercised by mutating
attributes on an already-built block rather than rebuilding.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.paper3a.emulator import params_meta as pm

U_FID = np.concatenate([np.atleast_2d(pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)),
                        [[0.5]]], axis=1)


def _build(shift):
    """(block, snapshot) with $BIND_PAPER3A_LOGM500_SHIFT set (or cleared).

    The second element is a SNAPSHOT of the module constants taken while the
    shift is still applied -- deliberately not the module object. `finally`
    below reloads the module back to its default state so the env var cannot
    leak between tests, which means every caller that held `kmod` was really
    holding one shared object already reset to the default. Two builds at
    +0.10 and -0.10 then compared equal, and a test asserting
    `kp.LOGM500_TARGET > km.LOGM500_TARGET` failed against a genuinely
    correct fix. The block objects were never affected -- they are
    constructed before the reset and carry the shifted state, which is why
    the substantive assertions (r_sat ordering, prediction ratios) passed
    throughout.
    """
    import analysis.paper3a.inference.ksz as kmod

    old = os.environ.get("BIND_PAPER3A_LOGM500_SHIFT")
    try:
        if shift is None:
            os.environ.pop("BIND_PAPER3A_LOGM500_SHIFT", None)
        else:
            os.environ["BIND_PAPER3A_LOGM500_SHIFT"] = f"{shift}"
        importlib.reload(kmod)
        snap = SimpleNamespace(
            LOGM500_SHIFT_DEX=kmod.LOGM500_SHIFT_DEX,
            LOGM500_TARGET=kmod.LOGM500_TARGET,
        )
        return kmod.KszBlock(), snap
    finally:
        if old is None:
            os.environ.pop("BIND_PAPER3A_LOGM500_SHIFT", None)
        else:
            os.environ["BIND_PAPER3A_LOGM500_SHIFT"] = old
        importlib.reload(kmod)          # leave the module in its default state


@pytest.fixture(scope="module")
def base():
    return _build(None)


# ---------------------------------------------------------------------------
# 1. the mass-shift fix
# ---------------------------------------------------------------------------

def test_default_is_an_exact_noop(base):
    """Env var unset -> shift 0 -> factor 1 EXACTLY, by not running at all."""
    blk, kmod = base
    assert kmod.LOGM500_SHIFT_DEX == 0.0
    assert blk.logm500_shift == 0.0
    # the response state is not even constructed, so `predict` executes the
    # identical statements it did before the fix
    assert blk._dlogm_gate_bins is None
    assert blk.wp2_bias_divisor is None and blk.wp2_bias_mode is None


def test_explicit_zero_shift_is_bitwise_identical_to_unset(base):
    blk0, _ = base
    blk_z, kmod_z = _build(0.0)
    assert kmod_z.LOGM500_SHIFT_DEX == 0.0
    assert np.array_equal(blk_z.predict(U_FID), blk0.predict(U_FID))


def test_gate_bin_span_is_sane(base):
    """The two emulated kSZ stacks must be well separated in mass, in the
    Msun frame the Siegel GGL target uses."""
    blk, _ = base
    for snap in (blk.snap_lo, blk.snap_hi):
        span = blk._gate_bin_logm500_span(snap)
        assert 0.4 < span < 0.8


def test_full_bin_span_shift_reproduces_the_other_gate_bin(base):
    """THE non-circular check. A log-linear interpolation between two
    stacks must return stack 1 exactly when shifted by their separation."""
    blk, _ = base
    span = {s: blk._gate_bin_logm500_span(s)
            for s in (blk.snap_lo, blk.snap_hi)}
    # the two snaps have slightly different spans, so test one snap at a time
    for snap in (blk.snap_lo, blk.snap_hi):
        pred = blk.emu.predict(U_FID[:, :30], snap, return_std=False)
        t_lo = np.atleast_2d(pred["ksz0_sr"]) @ blk._ops[snap].T
        t_hi = np.atleast_2d(pred["ksz1_sr"]) @ blk._ops[snap].T
        blk._dlogm_gate_bins = span
        blk.logm500_shift = span[snap]
        try:
            got = t_lo * blk._mass_response(snap, t_lo, t_hi)
        finally:
            blk._dlogm_gate_bins = None
            blk.logm500_shift = 0.0
        assert np.allclose(got, t_hi, rtol=1e-12)


def test_shift_raises_the_prediction_and_is_near_flat_in_radius():
    """Sign and radial shape — the two things the bug got wrong.

    The bug gave 0.990 at 1' rising monotonically to 0.9995 at 6' (the
    r_sat dilution term leaking). The fix must give a ratio > 1 for a
    positive shift, at every radius, with only a few-percent radial spread.

    NOTE ON CIRCULARITY: the MAGNITUDE (~1.18 at +0.10 dex) is NOT
    independent evidence — the audit's expected value was derived from the
    same two-bin ratio this code interpolates. Only the sign, the radial
    flatness and the log-symmetry below carry information.
    """
    blk0, _ = _build(None)
    p0 = blk0.predict(U_FID)[0]

    blk_p, kp = _build(+0.10)
    blk_m, km = _build(-0.10)
    rp = blk_p.predict(U_FID)[0] / p0
    rm = blk_m.predict(U_FID)[0] / p0

    assert np.all(rp > 1.0) and np.all(rm < 1.0)         # sign
    assert np.all(rp > 1.10) and np.all(rp < 1.30)       # order of magnitude
    # near-flat in radius: the bug's signature was a monotone ramp to 1
    assert rp.max() / rp.min() - 1.0 < 0.05
    assert not np.all(np.diff(rp) > 0)                   # not a monotone ramp
    # r_sat still moves with the target mass (the pre-existing, correct path)
    assert blk_p.r_sat > blk0.r_sat > blk_m.r_sat
    assert kp.LOGM500_TARGET > km.LOGM500_TARGET


# ---------------------------------------------------------------------------
# 2. the wp2 tau_CAP bias variant
# ---------------------------------------------------------------------------

def test_wp2_divisor_inner_only_by_default():
    from analysis.paper3a.inference.ksz import (
        WP2_TAU_CAP_BIAS_INNER_MAX_ARCMIN, wp2_tau_bias_divisor)

    r = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])
    inner = wp2_tau_bias_divisor(r)                      # default mode
    assert np.array_equal(inner, wp2_tau_bias_divisor(r, "inner"))
    outer = r > WP2_TAU_CAP_BIAS_INNER_MAX_ARCMIN
    assert np.all(inner[outer] == 1.0)                   # EXACT no-op outside
    assert np.all(inner[~outer] > 1.0)
    # the well-measured inner band is 7-13% at the measured radii
    assert 1.05 < inner[0] < 1.20
    allr = wp2_tau_bias_divisor(r, "all")
    assert np.array_equal(allr[~outer], inner[~outer])   # inner half unchanged
    assert np.all(allr[outer] > 1.4)                     # the factor ~2 regime
    with pytest.raises(ValueError):
        wp2_tau_bias_divisor(r, "outer")


def test_wp2_divisor_tracks_the_frozen_artifact():
    """The divisor must come from the wp2 npz, not from a transcription."""
    from analysis.paper3a.inference.ksz import (
        WP2_TAU_CAP_BIAS_RADII_ARCMIN, WP2_TAU_CAP_FRAC_BIAS,
        wp2_tau_bias_divisor, wp2_tau_cap_frac_bias)

    frac = wp2_tau_cap_frac_bias()                       # raises if it drifted
    assert frac.shape == (5,)
    assert np.allclose(frac, WP2_TAU_CAP_FRAC_BIAS, atol=5e-4)
    assert np.all(frac > 0)                              # model-high everywhere
    got = wp2_tau_bias_divisor(WP2_TAU_CAP_BIAS_RADII_ARCMIN, "all")
    assert np.allclose(got, 1.0 + frac, rtol=1e-14)


def test_wp2_bias_divides_the_prediction_exactly(base):
    from analysis.paper3a.inference.ksz import wp2_tau_bias_divisor

    blk, _ = base
    p0 = blk.predict(U_FID)
    div = wp2_tau_bias_divisor(blk.radii, "inner")
    blk.wp2_bias_divisor = div
    try:
        p1 = blk.predict(U_FID)
    finally:
        blk.wp2_bias_divisor = None
    assert np.allclose(p1, p0 / div[None, :], rtol=1e-14)
    assert np.array_equal(blk.predict(U_FID), p0)        # restored exactly


def test_wp2_bias_variant_wired_into_the_fit_driver():
    from analysis.paper3a.scripts.run_joint_ab_fit import VARIANTS

    assert "ksz_wp2bias" in VARIANTS and "ksz_wp2bias_all" in VARIANTS


# ---------------------------------------------------------------------------
# 3. the mass-dependent CylToSph variant
# ---------------------------------------------------------------------------

def test_c2s_default_unchanged_and_massdep_is_opt_in():
    from analysis.paper3a.emulator.cyltosph_theta import CylToSphTheta

    u = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    d = CylToSphTheta()
    assert d.mass_dependent is False
    f = d.factors_full_bins(u, "096")
    assert np.allclose(f, f[0])                          # still one scalar
    assert np.all(d._mass_shape("096", 5) == 1.0)        # exact ones


def test_c2s_massdep_monotonic_and_anchor_preserving():
    from analysis.paper3a.emulator.cyltosph_theta import (
        CylToSphTheta, c2s_massdep)
    from analysis.paper3a.gate.aggregate import cyltosph_for_snap

    u = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    m = CylToSphTheta(mass_dependent=True)
    f = m.factors_full_bins(u, "096")
    assert np.all(np.diff(f) > 0)                        # rises with mass
    # the shape is normalized, so the count-weighted mean returns the anchor
    base, _ = cyltosph_for_snap("096")
    shape = m._mass_shape("096", 5)
    assert np.isclose(np.average(f / shape, weights=np.ones(5)), base)
    # pooled-L50 values at the gate bin centres, from the module's own fit
    assert np.allclose(c2s_massdep([13.2, 13.6, 14.0, 14.4, 14.9]),
                       [0.4828, 0.6057, 0.7166, 0.8063, 0.8858], atol=5e-4)
    assert c2s_massdep(20.0) < 1.0 and c2s_massdep(20.0) > 0.99   # saturates


def test_c2s_massdep_moves_the_fgas_slope_not_the_anchor():
    """The reason this is candidate 1: it re-points the mass SLOPE."""
    from analysis.paper3a.emulator.cyltosph_theta import CylToSphTheta

    u = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    d = CylToSphTheta().factors_full_bins(u, "096")
    m = CylToSphTheta(mass_dependent=True).factors_full_bins(u, "096")
    ratio = m / d
    assert ratio[0] < 1.0 < ratio[-1]                    # group down, top up
    assert 1.5 < ratio[-1] / ratio[0] < 2.0              # ~1.8x slope change


def test_c2s_massdep_variant_wired_into_the_fit_driver():
    from analysis.paper3a.scripts.run_joint_ab_fit import VARIANTS

    assert "c2s_massdep" in VARIANTS
