"""Tests for the WP-A4 emulator package (run in the torch3 venv)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from analysis.paper3a.emulator import params_meta as pm
from analysis.paper3a.emulator.gasemu import GasEmulator
from analysis.paper3a.emulator.velocity import LinearVelocity
from analysis.paper3a.observables.mock_sample import MockSampleConfig


# ------------------------------------------------------------ params_meta ----

def test_unit_cube_roundtrip():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.05, 0.95, size=(8, 30))
    back = pm.astro_physical_to_unit(pm.astro_unit_to_physical(u))
    assert np.allclose(back, u, atol=1e-10)


def test_fiducial_inside_box():
    # three SB35 fiducials legitimately sit ON a prior bound
    # (VariableWindSpecMomentum = 0 = min; UVBH0Deltaz/UVBHepDeltaz = 0 = max)
    u = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    assert (u >= 0).all() and (u <= 1).all()
    on_edge = {n for n, ui in zip(pm.ASTRO_NAMES, u) if ui in (0.0, 1.0)}
    assert on_edge == {"VariableWindSpecMomentum", "UVBH0Deltaz", "UVBHepDeltaz"}


def test_table_params_rejects_wrong_cosmology():
    p = np.concatenate([[0.25, 0.7], pm.ASTRO_FIDUCIAL[:4], [0.05, 0.7, 1.0],
                        pm.ASTRO_FIDUCIAL[4:]])
    with pytest.raises(ValueError, match="cosmology"):
        pm.table_params_to_unit(p)


# --------------------------------------------------------------- velocity ----

@pytest.fixture(scope="module")
def lv():
    return LinearVelocity()


def test_sigma8_normalization(lv):
    x = lv.k * 8.0
    w = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    s8 = np.sqrt(np.trapz(lv.k**2 * lv._pk * w**2, lv.k) / (2 * np.pi**2))
    assert abs(s8 - lv.sigma8) < 1e-3


def test_sigma_v_magnitude(lv):
    # linear-theory 1D rms velocity, TNG cosmology: ~300 km/s at z~0-0.5
    assert 250.0 < lv.sigma_v_1d_kms(0.0) < 350.0
    # and within ~10% of the A3 nominal v_rms/c = 1.06e-3 at LRG z
    assert abs(lv.sigma_v_over_c(0.5) / 1.06e-3 - 1.0) < 0.10


def test_rv_limits_and_decay(lv):
    r = lv.r_v_parallel(np.array([0.0, 5.0, 25.0, 50.0]))
    assert abs(r[0] - 1.0) < 1e-6
    assert np.all(np.diff(r) < 0)          # monotone decay over this range
    assert 0.0 < r[3] < 0.3


def test_slab_mean_between_bounds(lv):
    m = lv.slab_mean_rv(51.25)
    assert lv.r_v_parallel(51.25 / 2.0) < m < 1.0


# ------------------------------------------------------- forward-model kSZ ---

def test_transverse_weight_ordering(lv):
    from analysis.paper3a.emulator.forward import transverse_weight

    R = np.array([0.1, 1.0, 3.0, 8.0])
    w = transverse_weight(lv, R)
    assert np.all((w > 0) & (w < 1))
    assert np.all(np.diff(w) < 0)           # farther out -> longer LOS mix -> lower w
    # small-R excess is LOS-concentrated: well above the uniform slab mean
    assert w[0] > lv.slab_mean_rv(51.25) + 0.1


def test_xi_lin_positive_and_decaying(lv):
    s = np.array([1.0, 5.0, 20.0])
    xi = lv.xi_lin(s)
    assert np.all(xi > 0) and np.all(np.diff(xi) < 0)


# ------------------------------------------------------ per-theta CylToSph ---

def test_cyltosph_theta_model():
    from analysis.paper3a.emulator.cyltosph_theta import MODEL_NPZ, CylToSphTheta

    if not MODEL_NPZ.exists():
        pytest.skip("cyltosph_theta_model.npz not on this system")
    m = CylToSphTheta()
    i = pm.ASTRO_NAMES.index("WindEnergyIn1e51erg")
    assert m.slopes[i, 0] > 0.2          # wind sector dominates, positive sign
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    from analysis.paper3a.gate.aggregate import cyltosph_for_snap

    base, _ = cyltosph_for_snap("096")
    f_fid = m.factors(u_fid, "096")
    assert np.allclose(f_fid, base)      # trend is 1 at the fiducial
    u_weak = u_fid.copy()
    u_weak[i] = 0.0
    u_strong = u_fid.copy()
    u_strong[i] = 1.0
    f_w = m.factors(u_weak, "096")[0]
    f_s = m.factors(u_strong, "096")[0]
    assert f_w < f_fid[0] < f_s          # weak wind -> lower spherical conversion
    padded = m.factors_full_bins(u_fid, "096")
    assert padded.shape == (5,) and np.allclose(padded[3:], padded[2])


# --------------------------------------------------- kSZ sample model (A5) ---

def _synthetic_ksz_profile():
    """A smoothly declining Sigma(R) stack + its r-grid, matching the v3
    operator's ``sigma_r_centers_mpch`` convention (SIGMA_R_BIN_PIX=2 canvas
    pixels, CANVAS_PIXEL_MPCH = 205/4198)."""
    from analysis.paper3a.emulator.forward import CANVAS_PIXEL_MPCH

    n_bins = 90
    r = (np.arange(n_bins) + 0.5) * 2.0 * CANVAS_PIXEL_MPCH
    floor = 5e6
    sigma_r = floor + 5e8 / (1.0 + (r / 0.6) ** 2)
    return sigma_r, r


def test_ksz_sample_model_matches_central_when_undiluted():
    from analysis.paper3a.emulator.forward import ForwardModel

    sigma_r, r = _synthetic_ksz_profile()
    fm = ForwardModel(velocity=LinearVelocity())
    radii = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    z = 0.5
    central = fm.ksz_tksz_from_profile(sigma_r, r, z, radii)
    cfg = MockSampleConfig(logm200c_mean=13.3, logm200c_sigma=0.2, f_mis=0.0, f_sat=0.0)
    sampled = fm.ksz_tksz_sample_model(sigma_r, r, z, radii, cfg, n_mc=32)
    assert np.allclose(sampled.tksz, central.tksz)


def test_ksz_sample_model_satellite_dilution_shrinks_signal():
    from analysis.paper3a.emulator.forward import ForwardModel

    sigma_r, r = _synthetic_ksz_profile()
    fm = ForwardModel(velocity=LinearVelocity())
    radii = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    z = 0.5
    central = fm.ksz_tksz_from_profile(sigma_r, r, z, radii)
    cfg = MockSampleConfig(logm200c_mean=13.3, logm200c_sigma=0.2,
                           f_sat=0.25, r_sat_hmpc=0.5)
    sampled = fm.ksz_tksz_sample_model(sigma_r, r, z, radii, cfg, n_mc=1024, seed=1)
    # a centrally-peaked profile loses amplitude once a fraction of the
    # stack is recentered away from the peak -> dilution ratio in (0, 1]
    assert np.all(sampled.tksz <= central.tksz + 1e-12)
    assert np.all(sampled.tksz > 0)


def test_ksz_sample_model_dilution_cached():
    from analysis.paper3a.emulator.forward import ForwardModel

    sigma_r, r = _synthetic_ksz_profile()
    fm = ForwardModel(velocity=LinearVelocity())
    radii = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    cfg = MockSampleConfig(logm200c_mean=13.3, logm200c_sigma=0.2, f_sat=0.2, r_sat_hmpc=0.5)
    fm.ksz_tksz_sample_model(sigma_r, r, 0.5, radii, cfg, n_mc=64, seed=3)
    assert len(fm._dilution_cache) == 1
    fm.ksz_tksz_sample_model(sigma_r, r, 0.5, radii, cfg, n_mc=64, seed=3)
    assert len(fm._dilution_cache) == 1          # same key -> no recompute


# ------------------------------------------------ fit + predict integration --

def _synthetic_dataset(n_runs: int = 24, seed: int = 0) -> dict:
    """A one-snapshot dataset with a smooth 2-latent-dim response, shaped
    exactly like dataset.build's output (snap 096, all dims valid)."""
    rng = np.random.default_rng(seed)
    sizes = {"fgas_med": 5, "fgas_scat": 5, "ym": 3, "ksz0": 9, "ksz1": 9}
    D = sum(sizes.values())
    X = rng.uniform(0.1, 0.9, size=(n_runs, 30))
    latents = np.stack([X[:, 0] - 0.5, X[:, 3] ** 2], axis=1)     # (n, 2)
    W = rng.normal(size=(2, D)) * 0.15
    base = np.concatenate([
        np.full(5, np.log10(0.12)), np.full(5, np.log10(0.02)),
        [1.7, -5.2, 0.12], np.full(9, np.log10(0.05)), np.full(9, np.log10(0.1))])
    # small full-rank noise so no PCA component is float-eps degenerate
    Y_t = base[None, :] + latents @ W + rng.normal(0, 5e-3, size=(n_runs, D))
    Y = Y_t.copy()
    log_dims = np.r_[0:10, 13:31]
    Y[:, log_dims] = 10.0 ** Y_t[:, log_dims]

    return {
        "block_names": np.array(list(sizes)),
        "block_sizes": np.array(list(sizes.values())),
        "snaps": np.array(["096"]), "snap_z": np.array([0.034]),
        "radii_arcmin": np.linspace(1, 6, 9),
        "logm500_bin_edges": np.array([13.0, 13.4, 13.8, 14.2, 14.6, 15.2]),
        "ksz_bin0_range": np.array([13.2, 13.7]),
        "ksz_bin1_range": np.array([13.7, 15.5]),
        "param_names": np.array(pm.ASTRO_NAMES),
        "param_min": pm.ASTRO_MIN, "param_max": pm.ASTRO_MAX,
        "param_log": pm.ASTRO_LOG, "param_fiducial": pm.ASTRO_FIDUCIAL,
        "fixed_cosmology_names": np.array(list(pm.FIXED_COSMOLOGY)),
        "fixed_cosmology_values": np.array(list(pm.FIXED_COSMOLOGY.values())),
        "snap096_X_sb35": X, "snap096_Y_sb35": Y,
        "snap096_sem_sb35": np.full((n_runs, D), np.nan),
        "snap096_valid": np.ones(D, bool),
        "provenance": np.array(json.dumps({"synthetic": True})),
    }


def test_fit_predict_roundtrip():
    from analysis.paper3a.emulator.fit import build_arrays, crosscheck_vs_gpytorch

    f = _synthetic_dataset()
    rows = np.arange(24)
    arrays = build_arrays(f, rows, n_pca=4, iters=150, lr=0.1, seed=0,
                          device="cpu", verbose=False)
    worst = crosscheck_vs_gpytorch(arrays, f, rows, n_query=4)
    assert worst < 1e-6, f"numpy posterior != exact GP ({worst:.2e})"

    emu = GasEmulator(arrays)
    pred = emu.predict_vector(f["snap096_X_sb35"], "096", return_std=False)
    truth = f["snap096_Y_sb35"]
    frac = np.abs(pred - truth) / np.abs(truth)
    assert np.median(frac) < 0.05          # smooth response, near-interpolation

    # dict-input path hits the fiducial defaults and named overrides
    out = emu.predict({"WindEnergyIn1e51erg": 5.0}, "096")
    assert out["fgas_med"].shape == (5,)
    assert np.isfinite(out["fgas_med"]).all()
    assert (out["fgas_med"] > 0).all()


def test_partial_validity_masks():
    """Invalid dims (empty high-mass bins) come back NaN, valid dims finite."""
    from analysis.paper3a.emulator.fit import build_arrays

    f = _synthetic_dataset()
    f["snap096_valid"] = f["snap096_valid"].copy()
    f["snap096_valid"][[4, 9]] = False      # top mass bin med+scat invalid
    arrays = build_arrays(f, np.arange(24), n_pca=4, iters=40, lr=0.1, seed=0,
                          device="cpu", verbose=False)
    emu = GasEmulator(arrays)
    v = emu.predict_vector(f["snap096_X_sb35"][:3], "096", return_std=False)
    assert np.isnan(v[:, 4]).all() and np.isnan(v[:, 9]).all()
    keep = np.ones(31, bool)
    keep[[4, 9]] = False
    assert np.isfinite(v[:, keep]).all()
