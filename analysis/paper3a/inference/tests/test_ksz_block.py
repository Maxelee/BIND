"""KszBlock tests (torch3 venv; needs the ceph artifacts).

The block init is expensive (basis probing + MC dilution), so everything
shares one module-scoped instance, mirroring test_inference.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.paper3a.emulator import params_meta as pm


@pytest.fixture(scope="module")
def block():
    from analysis.paper3a.inference.ksz import KszBlock

    return KszBlock()


def _u31(u30, f_sat_unit=0.5):
    return np.concatenate([np.atleast_2d(u30),
                           np.full((np.atleast_2d(u30).shape[0], 1), f_sat_unit)],
                          axis=1)


def test_data_assembly(block):
    assert len(block.values) == 9
    assert np.all(np.diff(block.radii) > 0)
    assert np.all(block.values[:4] > 0)                  # detected inner radii
    assert "m3" in block.data.name                       # the mass-matched bin
    ev = np.linalg.eigvalsh(block.cov_data)
    assert np.all(ev > 0)                                # released cov is PD
    assert 0.70 < block.zeff < 0.78                      # fig03 count-weighted


def test_linear_operator_matches_direct_forward(block):
    """The basis-probed matrix must reproduce ksz_tksz_from_profile exactly
    (the chain is linear end to end) — this is the correctness anchor."""
    for snap in (block.snap_lo, block.snap_hi):
        sig = block._fid_sigma(snap)
        direct = block.fm.ksz_tksz_from_profile(
            sig, block.r_centers, block.emu.snap_z[snap], block.radii).tksz
        lin = block._ops[snap] @ sig
        assert np.max(np.abs(lin / direct - 1)) < 1e-10


def test_dilution_linear_in_fsat_and_shrinks(block):
    """T is exactly linear in f_sat, and satellites strictly dilute."""
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    t0 = block.predict(_u31(u_fid, 0.0))[0]              # f_sat = 0.10
    t1 = block.predict(_u31(u_fid, 1.0))[0]              # f_sat = 0.30
    tm = block.predict(_u31(u_fid, 0.5))[0]              # f_sat = 0.20
    assert np.allclose(tm, 0.5 * (t0 + t1), rtol=1e-12)
    assert np.all(t1 < t0)                               # more satellites -> less T
    for s in (block.snap_lo, block.snap_hi):
        assert np.all(block._ratio_sat[s] > 0)
        assert np.all(block._ratio_sat[s] < 1.0 + 1e-12)


def test_theta30_fixed_nuisance_mode(block):
    """30-column input uses the midpoint f_sat (diagnostics mode)."""
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    assert np.allclose(block.predict(u_fid), block.predict(_u31(u_fid, 0.5)),
                       rtol=1e-12)


def test_prediction_magnitude_and_wind_response(block):
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    pred = block.predict(_u31(u_fid))[0]
    assert np.all(pred > 0)
    assert np.all(np.diff(pred) > 0)                     # CAP T grows with aperture
    assert 0.1 < pred[0] < 100 and pred[-1] < 1000       # muK arcmin^2 scale
    i = pm.ASTRO_NAMES.index("WindEnergyIn1e51erg")
    u_hi, u_lo = u_fid.copy(), u_fid.copy()
    u_hi[i], u_lo[i] = 0.95, 0.05
    # stronger winds evacuate group gas -> larger kSZ-visible diffuse tau?
    # direction is emulator-measured, not asserted; just require a real,
    # finite, sign-stable response
    d = block.predict(_u31(u_hi))[0] - block.predict(_u31(u_lo))[0]
    assert np.all(np.isfinite(d)) and np.any(np.abs(d) > 1e-3)


def test_covariance_and_loglike(block):
    rng = np.random.default_rng(0)
    U = np.concatenate([rng.uniform(0.1, 0.9, (8, 30)),
                        rng.uniform(0, 1, (8, 1))], axis=1)
    pred = block.predict(U)
    c = block.cov(pred)
    assert c.shape == (8, 9, 9)
    assert np.all(np.linalg.eigvalsh(c) > 0)
    # model terms strictly widen the diagonal
    assert np.all(c[:, np.arange(9), np.arange(9)] >
                  np.diag(block.cov_data)[None, :])
    ll = block.loglike(U)
    chi2 = block.chi2(U)
    assert ll.shape == (8,) and np.all(np.isfinite(ll))
    assert np.all(chi2 > 0)


def test_systematic_fracs_sane(block):
    # emulation floor: few-percent (v4 kfold kSZ 1.7-2.5% band)
    assert np.all(block.emul_frac > 0) and np.all(block.emul_frac < 0.15)
    # totals dominated by the named terms, nothing exploding
    assert np.all(block.sys_frac >= 0.06)                # >= velocity norm
    assert np.all(block.sys_frac < 0.5)


def test_sampler_ndim31_bounds():
    from analysis.paper3a.inference.sampler import log_prob_factory

    class _Dummy:
        def loglike(self, U):
            return np.zeros(len(U))

    lp = log_prob_factory([_Dummy()])
    U = np.array([[0.5] * 31, [0.5] * 30 + [1.2], [0.5] * 30 + [-0.1]])
    out = lp(U)
    assert np.isfinite(out[0]) and np.isinf(out[1]) and np.isinf(out[2])
