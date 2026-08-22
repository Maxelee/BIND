"""bind.params: the user-facing parameter-vector helpers."""

import numpy as np
import pytest

from bind import params as PA


def test_fiducial_params_shape_and_ordering():
    p = PA.fiducial_params()
    assert p.shape == (PA.N_PARAMS,) == (35,)
    assert p.dtype == np.float64
    np.testing.assert_array_equal(p, PA.PARAM_FIDUCIAL)

    # Ordering: the vector is positional, so these anchors pin the convention
    # the trained checkpoints were conditioned on.
    assert PA.PARAM_NAMES[0] == "Omega0"
    assert PA.PARAM_NAMES[1] == "sigma8"
    assert PA.PARAM_NAMES[6] == "OmegaBaryon"
    assert p[0] == pytest.approx(0.3)
    assert p[1] == pytest.approx(0.8)

    # Every fiducial value lies inside its own prior box.
    assert ((PA.PARAM_MIN <= p) & (p <= PA.PARAM_MAX)).all()


def test_fiducial_params_returns_a_copy():
    p = PA.fiducial_params()
    p[0] = -999.0
    assert PA.fiducial_params()[0] != -999.0
    assert PA.PARAM_FIDUCIAL[0] != -999.0


def test_vary_param_moves_only_the_named_index():
    fid = PA.fiducial_params()
    idx = PA.PARAM_NAMES.index("WindEnergyIn1e51erg")

    by_name = PA.vary_param("WindEnergyIn1e51erg", 7.2)
    assert by_name[idx] == pytest.approx(7.2)
    changed = np.flatnonzero(by_name != fid)
    assert changed.tolist() == [idx]

    # The integer index form must agree exactly.
    np.testing.assert_array_equal(by_name, PA.vary_param(idx, 7.2))


def test_vary_param_fraction_respects_the_log_sampling_flag():
    log_idx = int(np.flatnonzero(PA.PARAM_LOG_FLAG == 1)[0])
    lin_idx = int(np.flatnonzero(PA.PARAM_LOG_FLAG == 0)[0])

    for idx in (log_idx, lin_idx):
        assert PA.vary_param(idx, fraction=0.0)[idx] == pytest.approx(PA.PARAM_MIN[idx])
        assert PA.vary_param(idx, fraction=1.0)[idx] == pytest.approx(PA.PARAM_MAX[idx])

    # Midpoint: geometric for log-flagged, arithmetic otherwise.
    mid_log = PA.vary_param(log_idx, fraction=0.5)[log_idx]
    assert mid_log == pytest.approx(np.sqrt(PA.PARAM_MIN[log_idx] * PA.PARAM_MAX[log_idx]))
    mid_lin = PA.vary_param(lin_idx, fraction=0.5)[lin_idx]
    assert mid_lin == pytest.approx(0.5 * (PA.PARAM_MIN[lin_idx] + PA.PARAM_MAX[lin_idx]))


def test_vary_param_input_validation():
    with pytest.raises(ValueError):
        PA.vary_param(0)                          # neither value nor fraction
    with pytest.raises(ValueError):
        PA.vary_param(0, 1.0, fraction=0.5)       # both
    with pytest.raises(KeyError):
        PA.vary_param("NotAParameter", 1.0)
    with pytest.raises(IndexError):
        PA.vary_param(PA.N_PARAMS, 1.0)


def test_vary_params_applies_several_overrides_on_a_base():
    base = PA.vary_param("Omega0", 0.25)
    out = PA.vary_params({"sigma8": 0.9, 6: 0.055}, base=base)
    assert out[0] == pytest.approx(0.25)          # base preserved
    assert out[1] == pytest.approx(0.9)
    assert out[6] == pytest.approx(0.055)
    assert np.flatnonzero(out != base).tolist() == [1, 6]
    np.testing.assert_array_equal(base, PA.vary_param("Omega0", 0.25))   # base untouched


def test_random_params_stays_inside_the_prior_box_and_is_seedable():
    single = PA.random_params(rng=0)
    assert single.shape == (PA.N_PARAMS,)
    batch = PA.random_params(8, rng=0)
    assert batch.shape == (8, PA.N_PARAMS)
    assert ((PA.PARAM_MIN <= batch) & (batch <= PA.PARAM_MAX)).all()
    np.testing.assert_allclose(PA.random_params(8, rng=0), batch)
    assert not np.allclose(PA.random_params(8, rng=1), batch)

    fixed = PA.random_params(4, rng=2, fix={"Omega0": 0.31, 1: 0.77})
    assert (fixed[:, 0] == 0.31).all() and (fixed[:, 1] == 0.77).all()


def test_param_dataframe_is_a_copy_of_the_metadata_table():
    df = PA.param_dataframe()
    assert len(df) == PA.N_PARAMS
    assert df["ParamName"].tolist() == PA.PARAM_NAMES
    df.loc[0, "ParamName"] = "mutated"
    assert PA.param_dataframe()["ParamName"][0] == "Omega0"
