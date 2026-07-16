"""Analytic-fixture tests for the kSZ / f_gas / Y-M operators and sample machinery."""

import numpy as np
import pytest

from analysis.paper3a.observables.constants import (
    MSUN_KG,
    M_PROTON_KG,
    SIGMA_T_M2,
    T_CMB_UK,
    X_E_FULLY_IONIZED,
    X_H,
)
from analysis.paper3a.observables.fgas import (
    CylToSphCorrection,
    fgas_cylindrical,
)
from analysis.paper3a.observables.geometry import PatchGeometry
from analysis.paper3a.observables.ksz import (
    KSZOperatorConfig,
    ksz_cap_profile,
    minimum_cutout_extent_hmpc,
    tau_map_from_gas,
)
from analysis.paper3a.observables.mock_sample import (
    MockSampleConfig,
    draw_center_offsets_hmpc,
    sample_weights,
)
from analysis.paper3a.observables.stacking import multi_sample_convergence, stack_profiles
from analysis.paper3a.observables.ym import fit_ym_relation, y_aperture_mpc2


GEOM_Z0 = PatchGeometry(pixel_mpch=50.0 / 1024, z=0.0)


def test_tau_map_uniform_slab_analytic():
    # 1 Msun/h per pixel everywhere -> tau = sigma_T * x_e * X_H * (M/h m_p) / A
    sigma = np.ones((32, 32))
    tau = tau_map_from_gas(sigma, GEOM_Z0)
    m_kg = 1.0 / GEOM_Z0.cosmology.h * MSUN_KG
    expected = SIGMA_T_M2 * X_E_FULLY_IONIZED * X_H * m_kg / M_PROTON_KG / GEOM_Z0.pixel_area_proper_m2()
    assert tau == pytest.approx(np.full((32, 32), expected), rel=1e-12)


def test_tau_redshift_placement_scales_as_1plusz_sq():
    sigma = np.ones((8, 8))
    tau0 = tau_map_from_gas(sigma, PatchGeometry(pixel_mpch=0.05, z=0.0))
    tau1 = tau_map_from_gas(sigma, PatchGeometry(pixel_mpch=0.05, z=1.0))
    # same comoving pixel, higher z -> smaller proper area -> higher tau
    assert tau1 / tau0 == pytest.approx(np.full((8, 8), 4.0), rel=1e-12)


def _compact_gas_map(n=257, mass_msunh=1e12):
    m = np.zeros((n, n))
    c = n // 2
    m[c - 1 : c + 2, c - 1 : c + 2] = mass_msunh / 9.0
    return m


def test_ksz_cap_profile_compact_source():
    # compact gas blob: tau_CAP(theta_d) = sigma_T N_e / A_prop * A_pix_arcmin2 * n_pix...
    # equivalently total tau * pixel_area_arcmin2 -> independent of theta_d.
    z = 0.55
    geom = PatchGeometry(pixel_mpch=50.0 / 1024, z=z)
    m = _compact_gas_map()
    cfg = KSZOperatorConfig(radii_arcmin=np.array([2.0, 4.0, 6.0]), beam_fwhm_arcmin=2.1, z_eff=z)
    prof = ksz_cap_profile(m, geom, cfg)
    n_e_total = X_E_FULLY_IONIZED * X_H * (1e12 / geom.cosmology.h * MSUN_KG) / M_PROTON_KG
    tau_times_area = SIGMA_T_M2 * n_e_total / geom.pixel_area_proper_m2() * geom.pixel_area_arcmin2()
    # beam spreads the source; by 4-6 arcmin the CAP disk has recollected it
    assert prof[-1] == pytest.approx(tau_times_area, rel=2e-2)
    assert prof[-2] == pytest.approx(tau_times_area, rel=5e-2)
    # profile grows toward the asymptote (beam-spread flux enters the annulus at small theta_d)
    assert prof[0] < prof[-1] * 1.02


def test_ksz_tksz_normalization():
    z = 0.55
    geom = PatchGeometry(pixel_mpch=50.0 / 1024, z=z)
    m = _compact_gas_map()
    base = KSZOperatorConfig(radii_arcmin=np.array([5.0]), beam_fwhm_arcmin=0.0, z_eff=z)
    with_v = KSZOperatorConfig(
        radii_arcmin=np.array([5.0]), beam_fwhm_arcmin=0.0, z_eff=z, v_rms_over_c=1.06e-3
    )
    tau_cap = ksz_cap_profile(m, geom, base)
    t_ksz = ksz_cap_profile(m, geom, with_v)
    assert t_ksz == pytest.approx(T_CMB_UK * 1.06e-3 * tau_cap, rel=1e-12)


def test_minimum_cutout_extent_exceeds_core_patch_at_lrg_z():
    # the sizing finding again, through the public helper
    geom = PatchGeometry(pixel_mpch=6.25 / 128, z=0.55)
    cfg = KSZOperatorConfig(radii_arcmin=np.array([6.0]), beam_fwhm_arcmin=2.1, z_eff=0.55)
    assert minimum_cutout_extent_hmpc(cfg, geom) > 6.25


def test_fgas_cylindrical_uniform_disk():
    # gas disk of known mass inside r500 -> fgas = M_disk / M500 exactly
    n = 129
    geom = PatchGeometry(pixel_mpch=0.05, z=0.0)
    r500 = 1.0  # h^-1 Mpc -> 20 px
    m500 = 1e14
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    r_pix = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2)
    sigma = np.where(r_pix <= 10.0, 1e10, 0.0)  # all gas well inside r500
    m_gas = sigma.sum()
    got = fgas_cylindrical(sigma, geom, r500_hmpc=r500, m500_msunh=m500)
    assert got == pytest.approx(m_gas / m500, rel=1e-6)


def test_cyl_to_sph_correction_gates():
    corr = CylToSphCorrection()
    with pytest.raises(ValueError, match="uncalibrated"):
        corr.apply(np.array([0.1]), map_depth_hmpc=50.0)
    cal = CylToSphCorrection(factor=0.8, depth_hmpc=50.0, scatter=0.05, provenance="test")
    assert cal.apply(np.array([0.1]), 50.0) == pytest.approx([0.08])
    with pytest.raises(ValueError, match="depth"):
        cal.apply(np.array([0.1]), map_depth_hmpc=205.0)


def test_y_aperture_uniform_disk():
    n = 129
    geom = PatchGeometry(pixel_mpch=0.05, z=0.0)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    r_pix = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2)
    y_map = np.where(r_pix <= 15.0, 1e-6, 0.0)
    Y = y_aperture_mpc2(y_map, geom, r_hmpc=1.5)  # 30 px aperture fully contains the source
    pix_mpc2 = (0.05 / geom.cosmology.h) ** 2
    expected = y_map.sum() * pix_mpc2  # pixelized source area, aperture weight exactly 1 inside
    assert Y == pytest.approx(expected, rel=1e-9)


def test_fit_ym_relation_recovers_powerlaw_and_scatter():
    rng = np.random.default_rng(42)
    logm = rng.uniform(13.0, 15.0, 400)
    scatter = 0.08
    logy = 1.66 * (logm - 14.0) - 4.5 + rng.normal(0, scatter, 400)
    rel = fit_ym_relation(10**logm, 10**logy)
    assert rel.alpha == pytest.approx(1.66, abs=0.02)
    assert rel.beta == pytest.approx(-4.5, abs=0.02)
    assert rel.scatter_dex == pytest.approx(scatter, abs=0.01)
    assert rel.n_halos == 400


def test_sample_weights_match_target_moments():
    rng = np.random.default_rng(0)
    # catalog: mass function-ish, heavily bottom-weighted
    logm = 13.0 + 2.0 * rng.power(3.0, 20000) * 0  # placeholder, use beta draw
    logm = 13.0 + rng.beta(1.0, 4.0, 20000) * 2.0
    cfg = MockSampleConfig(logm200c_mean=13.8, logm200c_sigma=0.25)
    w = sample_weights(logm, cfg)
    mu = np.sum(w * logm)
    sd = np.sqrt(np.sum(w * (logm - mu) ** 2))
    assert mu == pytest.approx(13.8, abs=0.03)
    assert sd == pytest.approx(0.25, abs=0.04)


def test_sample_weights_reject_uncovered_bin():
    logm = np.random.default_rng(1).uniform(13.0, 13.5, 1000)
    with pytest.raises(ValueError):
        sample_weights(logm, MockSampleConfig(logm200c_mean=15.5, logm200c_sigma=0.1))


def test_center_offsets_fractions_and_scale():
    rng = np.random.default_rng(7)
    cfg = MockSampleConfig(
        logm200c_mean=13.5, logm200c_sigma=0.3,
        f_mis=0.3, sigma_mis_hmpc=0.4, f_sat=0.2, r_sat_hmpc=0.8,
    )
    off = draw_center_offsets_hmpc(200_000, cfg, rng)
    r = np.hypot(off[:, 0], off[:, 1])
    centered = np.mean(r == 0)
    assert centered == pytest.approx((1 - 0.2) * (1 - 0.3), abs=0.01)
    # rms of the offset population mixes the two scales with their fractions
    expected_ms = 0.2 * 2 * 0.8**2 + (1 - 0.2) * 0.3 * 2 * 0.4**2
    assert np.mean(r**2) == pytest.approx(expected_ms, rel=0.03)


def test_stack_profiles_weighted():
    p = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert stack_profiles(p) == pytest.approx([2.0, 3.0])
    assert stack_profiles(p, np.array([3.0, 1.0])) == pytest.approx([1.5, 2.5])


def test_multi_sample_convergence_shrinks():
    rng = np.random.default_rng(3)
    truth = np.linspace(1.0, 2.0, 9)
    x = truth[None, None, :] + rng.normal(0, 0.3, size=(16, 50, 9))
    out = multi_sample_convergence(x)
    assert out["stack_full"] == pytest.approx(truth, abs=0.05)
    # deviation from the full stack must shrink as samples accumulate
    assert out["frac_dev_vs_k"][0] > out["frac_dev_vs_k"][7] > out["frac_dev_vs_k"][-2]
    assert out["frac_dev_vs_k"][-1] == pytest.approx(0.0, abs=1e-12)
    assert out["per_halo_scatter"] == pytest.approx(np.full(9, 0.3), abs=0.02)
