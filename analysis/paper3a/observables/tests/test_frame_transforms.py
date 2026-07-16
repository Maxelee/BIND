"""Tests for the stage-1 frame transform (WP-A2 Popeye half).

Verifies the transform application against hand-computed cases and the exact
convention of ``bind.inference.lightcone_transforms`` (lightcone branch), plus
manifest parsing and the hydro-path derivation. Pure numpy — runs in the
minimal venv.
"""

import json

import numpy as np
import pytest

from analysis.paper3a.observables.frame_transforms import (
    PROJ_DIR_AXES,
    apply_frame_transform,
    assign_slabs,
    load_stage1_manifest,
)

BOX = 100.0


def test_projdir2_no_flip_no_disp_is_identity():
    pos = np.array([[5.0, 5.0, 5.0], [10.0, 20.0, 30.0]])
    out = apply_frame_transform(pos, proj_dir=2, disp=[0, 0, 0], flip=[False, False, False], box_size=BOX)
    np.testing.assert_allclose(out, pos, atol=1e-5)


def test_projdir2_displacement():
    pos = np.array([[5.0, 5.0, 5.0]])
    out = apply_frame_transform(pos, proj_dir=2, disp=[10, 20, 30], flip=[False, False, False], box_size=BOX)
    np.testing.assert_allclose(out, [[15.0, 25.0, 35.0]], atol=1e-5)


def test_projdir0_axis_permutation():
    # LOS along original x: col2 (LOS) must carry original axis 0.
    pos = np.array([[5.0, 6.0, 7.0]])
    out = apply_frame_transform(pos, proj_dir=0, disp=[10, 20, 30], flip=[False, False, False], box_size=BOX)
    # ix,iy,iz = (1,2,0): x<-pos[1]+d[1]=26, y<-pos[2]+d[2]=37, z<-pos[0]+d[0]=15
    np.testing.assert_allclose(out, [[26.0, 37.0, 15.0]], atol=1e-5)
    assert PROJ_DIR_AXES[0] == (1, 2, 0)


def test_projdir1_axis_permutation():
    pos = np.array([[5.0, 6.0, 7.0]])
    out = apply_frame_transform(pos, proj_dir=1, disp=[0, 0, 0], flip=[False, False, False], box_size=BOX)
    # ix,iy,iz = (2,0,1): x<-pos[2]=7, y<-pos[0]=5, z<-pos[1]=6
    np.testing.assert_allclose(out, [[7.0, 5.0, 6.0]], atol=1e-5)


def test_flip_then_wrap():
    # flip original axis 0 (=transverse x for proj_dir 2), no disp.
    pos = np.array([[5.0, 5.0, 5.0]])
    out = apply_frame_transform(pos, proj_dir=2, disp=[0, 0, 0], flip=[True, False, False], box_size=BOX)
    np.testing.assert_allclose(out, [[95.0, 5.0, 5.0]], atol=1e-5)  # -5 mod 100


def test_output_within_box():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, BOX, size=(1000, 3))
    out = apply_frame_transform(pos, proj_dir=1, disp=[203.6, 139.8, 146.2], flip=[True, False, True], box_size=BOX)
    assert out.min() >= 0.0 and out.max() < BOX


def test_transform_preserves_particle_count_and_is_deterministic():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, BOX, size=(500, 3))
    a = apply_frame_transform(pos, 0, [1, 2, 3], [True, True, False], BOX)
    b = apply_frame_transform(pos, 0, [1, 2, 3], [True, True, False], BOX)
    assert a.shape == (500, 3)
    np.testing.assert_array_equal(a, b)


def test_assign_slabs_boundaries():
    box, n = 100.0, 4  # slab_h = 25
    los = np.array([0.0, 24.9, 25.0, 49.9, 50.0, 99.9, 100.0])
    slabs = assign_slabs(los, box, n)
    np.testing.assert_array_equal(slabs, [0, 0, 1, 1, 2, 3, 3])  # 100.0 clipped into last slab


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        apply_frame_transform(np.zeros((3, 2)), 2, [0, 0, 0], [False] * 3, BOX)


def test_load_stage1_manifest(tmp_path):
    manifest = {
        "box_size": 205.0, "npix": 4198, "n_slabs": 4, "slab_depth": 50.0,
        "proj_dir": 2, "disp": [203.61, 139.77, 146.17], "flip": [False, True, False],
        "snapshot_index": 63, "redshift": 0.5985, "scale_factor": 0.6256,
        "halo_mass_field": "Group_M_Crit200",
        "snapshot": "/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output/snapdir_063",
        "group_catalog": "/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output/groups_063",
    }
    p = tmp_path / "stage1_manifest.json"
    p.write_text(json.dumps(manifest))
    # accepts either the dir or the file
    m = load_stage1_manifest(tmp_path)
    assert m.proj_dir == 2 and m.n_slabs == 4
    np.testing.assert_allclose(m.flip.astype(int), [0, 1, 0])
    assert np.isclose(m.slab_depth_hmpc, 205.0 / 4)
    # hydro twin derivation swaps _DM out of the sim dir only
    assert "L205n2500TNG_DM" not in m.hydro_snapdir()
    assert m.hydro_snapdir().endswith("L205n2500TNG/output/snapdir_063")
    assert m.hydro_group_catalog().endswith("L205n2500TNG/output/groups_063")
