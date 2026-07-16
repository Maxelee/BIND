"""Tests for the transformed-frame TNG300 truth projection (WP-A2 Popeye half).

Exercises the per-chunk accumulator on synthetic gas particles (no HDF5, no
real snapshot): mass conservation, the SFR>0 cut, mass-weighted temperature,
slab routing, and the x_e-residual electron channel. Requires the bind/Pylians
chain (Popeye venv); skipped on the pure-numpy rusty venv.
"""

import numpy as np
import pytest

pytest.importorskip("MAS_library")
pytest.importorskip("bind.inference.pipeline")

from analysis.paper3a.observables.constants import MPC_IN_M, MSUN_KG, M_PROTON_KG, X_H, X_E_FULLY_IONIZED
from analysis.paper3a.observables.frame_transforms import Stage1Manifest
from analysis.paper3a.observables.truth_projection import (
    TruthMapAccumulator,
    accumulate_gas_chunk,
    finalize_truth_maps,
)

BOX, NPIX, NSLABS, H = 100.0, 64, 2, 0.6774


def _manifest(proj_dir=2, disp=(0, 0, 0), flip=(False, False, False)):
    return Stage1Manifest(
        proj_dir=proj_dir, disp=np.asarray(disp, float), flip=np.asarray(flip, bool),
        box_size=BOX, npix=NPIX, n_slabs=NSLABS, slab_depth=BOX / NSLABS,
        snapshot_index=63, redshift=0.6, scale_factor=0.625,
        halo_mass_field="Group_M_Crit200", dmo_snapshot="", dmo_group_catalog="", raw={},
    )


def _acc():
    pixel_side_m = (BOX / NPIX) / H * MPC_IN_M
    return TruthMapAccumulator(
        n_slabs=NSLABS, npix=NPIX, box_size=BOX, slab_depth=BOX / NSLABS,
        pixel_area_m2=pixel_side_m ** 2,
    )


def _chunk(acc, pos, mass_code, xe, sfr, u=1e4, dens=1e-4, manifest=None):
    n = len(pos)
    accumulate_gas_chunk(
        acc, pos_mpch=np.asarray(pos, float),
        mass_code=np.full(n, mass_code, float) if np.isscalar(mass_code) else np.asarray(mass_code, float),
        density_code=np.full(n, dens, float),
        u_code=np.full(n, u, float),
        xe=np.full(n, xe, float) if np.isscalar(xe) else np.asarray(xe, float),
        sfr=np.full(n, sfr, float) if np.isscalar(sfr) else np.asarray(sfr, float),
        manifest=manifest or _manifest(), h=H,
    )


def test_gas_sigma_conserves_mass():
    acc = _acc()
    rng = np.random.default_rng(0)
    pos = rng.uniform(5, BOX - 5, size=(200, 3))  # keep away from wrap edges
    mass_code = 1e-4  # -> 1e6 Msun/h per particle
    _chunk(acc, pos, mass_code, xe=X_E_FULLY_IONIZED, sfr=-1.0)
    out = finalize_truth_maps(acc)
    total = out["gas_sigma"].sum()
    np.testing.assert_allclose(total, 200 * mass_code * 1e10, rtol=2e-4)


def test_sfr_cut_excludes_star_forming():
    acc = _acc()
    pos = np.array([[50.0, 50.0, 20.0], [50.0, 50.0, 20.0]])
    _chunk(acc, pos, mass_code=1e-4, xe=1.0, sfr=[5.0, -1.0])  # first is star-forming
    out = finalize_truth_maps(acc)
    # only the hot particle contributes
    np.testing.assert_allclose(out["gas_sigma"].sum(), 1e-4 * 1e10, rtol=2e-4)


def test_temperature_is_mass_weighted_mean():
    acc = _acc()
    pos = np.array([[30.0, 30.0, 20.0], [60.0, 60.0, 70.0]])
    _chunk(acc, pos, mass_code=1e-4, xe=1.0, sfr=-1.0, u=1234.0)  # constant u -> constant T
    out = finalize_truth_maps(acc)
    from bind.inference.pipeline import gas_temperature_K
    T_expect = gas_temperature_K(np.array([1234.0]), np.array([1.0]))[0]
    hot = out["gas_sigma"] > 0
    assert hot.sum() > 0
    np.testing.assert_allclose(out["temperature"][hot], T_expect, rtol=1e-4)


def test_xe_residual_true_equals_assumed_when_fully_ionized():
    # With xe == 1.158 (fully ionized) the x_e-weighted gas map equals x_e times
    # gas_sigma, so tau_true == tau_assumed (the residual vanishes).
    acc = _acc()
    rng = np.random.default_rng(2)
    pos = rng.uniform(5, BOX - 5, size=(300, 3))
    _chunk(acc, pos, mass_code=1e-4, xe=X_E_FULLY_IONIZED, sfr=-1.0)
    out = finalize_truth_maps(acc)
    # tau_true uses tau_map_from_gas(gas_sigma_xe, x_e=1.0);
    # tau_assumed uses tau_map_from_gas(gas_sigma, x_e=1.158). Their ratio is
    # sum(gas_sigma_xe) / (1.158 * sum(gas_sigma)).
    np.testing.assert_allclose(
        out["gas_sigma_xe"].sum(), X_E_FULLY_IONIZED * out["gas_sigma"].sum(), rtol=2e-4
    )


def test_xe_residual_true_below_assumed_when_partially_ionized():
    acc = _acc()
    rng = np.random.default_rng(3)
    pos = rng.uniform(5, BOX - 5, size=(300, 3))
    xe_low = 0.5  # < 1.158
    _chunk(acc, pos, mass_code=1e-4, xe=xe_low, sfr=-1.0)
    out = finalize_truth_maps(acc)
    # tau_true / tau_assumed = sum(gas_sigma_xe) / (1.158 * sum(gas_sigma)) = 0.5/1.158
    ratio = out["gas_sigma_xe"].sum() / (X_E_FULLY_IONIZED * out["gas_sigma"].sum())
    assert ratio < 1.0
    np.testing.assert_allclose(ratio, xe_low / X_E_FULLY_IONIZED, rtol=2e-4)


def test_slab_routing_by_los():
    # proj_dir=2 so LOS = z; two particles in different z-slabs land in
    # different slab maps.
    acc = _acc()
    pos = np.array([[40.0, 40.0, 10.0], [40.0, 40.0, 90.0]])  # slab 0 and slab 1 (slab_h=50)
    _chunk(acc, pos, mass_code=1e-4, xe=1.0, sfr=-1.0)
    out = finalize_truth_maps(acc)
    assert out["gas_sigma"][0].sum() > 0
    assert out["gas_sigma"][1].sum() > 0
    np.testing.assert_allclose(out["gas_sigma"][0].sum(), 1e-4 * 1e10, rtol=2e-4)
    np.testing.assert_allclose(out["gas_sigma"][1].sum(), 1e-4 * 1e10, rtol=2e-4)


def test_halo_catalog_carries_original_frame_centers(tmp_path):
    # Regression: the CylToSph particle pass queries RAW (original-frame)
    # snapshot coordinates, so the catalog must expose original-frame centers;
    # feeding it the transformed ones silently measured spheres at unrelated
    # points (factor ~1e-3, scatter >> mean — the 2026-07-16 production run).
    import h5py

    from analysis.paper3a.observables.frame_transforms import apply_frame_transform
    from analysis.paper3a.observables.truth_projection import load_hydro_halo_catalog

    gpos_kpch = np.array([[10.0, 20.0, 30.0], [70.0, 80.0, 90.0]]) * 1e3  # kpc/h
    with h5py.File(tmp_path / "fof_subhalo_tab_063.0.hdf5", "w") as f:
        grp = f.create_group("Group")
        grp["Group_M_Crit200"] = np.array([2e3, 3e3])   # 1e10 Msun/h units -> 2e13, 3e13
        grp["Group_M_Crit500"] = np.array([1.5e3, 2e3])
        grp["Group_R_Crit200"] = np.array([500.0, 600.0])
        grp["Group_R_Crit500"] = np.array([300.0, 400.0])
        grp["GroupPos"] = gpos_kpch

    manifest = _manifest(proj_dir=1, disp=(12.0, 34.0, 5.0), flip=(True, False, True))
    cat = load_hydro_halo_catalog(manifest, group_catalog=str(tmp_path), snapshot=63)

    np.testing.assert_allclose(cat.centers_orig, (gpos_kpch / 1e3) % BOX, rtol=1e-12)
    # transform(centers_orig) must reproduce the transformed-frame (xy, los)
    tpos = apply_frame_transform(cat.centers_orig, manifest.proj_dir, manifest.disp,
                                 manifest.flip, manifest.box_size)
    np.testing.assert_allclose(tpos[:, :2], cat.centers_xy, rtol=1e-12)
    np.testing.assert_allclose(tpos[:, 2], cat.los, rtol=1e-12)
    # and the transform is NOT the identity here — the two frames really differ
    assert not np.allclose(tpos, cat.centers_orig)
