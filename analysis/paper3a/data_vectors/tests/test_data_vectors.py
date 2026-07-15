"""WP-A1 unit tests: loader round-trips, covariance SPD checks, unit/h-convention presence.

These tests read the real downloaded products under data_root() (ceph, not
git) rather than synthetic fixtures, per the project's "never fabricate a
data value" rule — there is nothing to fabricate a fixture from that would
be more trustworthy than the archive itself. They are skipped wholesale if
the data root is not mounted (e.g. running off rusty).
"""

import numpy as np
import pytest

from analysis.paper3a.data_vectors.data_vectors import (
    DataVector,
    data_root,
    load_egas_erass1,
    load_kappa_y_pandey2025,
    load_kappa_y_pandey2025_full_covariance,
    load_kappa_y_pandey2025_nz,
    load_ksz_hadzhiyska2024,
    load_ksz_hadzhiyska2024_mass_bins,
    load_ksz_hadzhiyska2026_bgs_elg,
    load_ksz_qu2026_lrg_by_mass,
    load_ksz_qu2026_lrg_fiducial,
    load_ksz_ried_guachalla2025,
    load_ksz_ried_guachalla2025_correlation,
)

pytestmark = pytest.mark.skipif(
    not data_root().exists(),
    reason=f"WP1 data root {data_root()} not mounted on this host",
)


# ---------------------------------------------------------------------------
# Generic DataVector construction / validation behavior
# ---------------------------------------------------------------------------


def test_datavector_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        DataVector(
            name="bad",
            source="test",
            bins=np.arange(3),
            bin_type="theta_arcmin",
            bin_units="arcmin",
            values=np.arange(4),
            value_units="test",
        )


def test_datavector_rejects_bad_covariance_shape():
    with pytest.raises(ValueError):
        DataVector(
            name="bad",
            source="test",
            bins=np.arange(3),
            bin_type="theta_arcmin",
            bin_units="arcmin",
            values=np.arange(3),
            value_units="test",
            covariance=np.eye(4),
        )


def test_datavector_derives_errors_from_covariance():
    cov = np.diag([1.0, 4.0, 9.0])
    dv = DataVector(
        name="ok",
        source="test",
        bins=np.arange(3),
        bin_type="theta_arcmin",
        bin_units="arcmin",
        values=np.zeros(3),
        value_units="test",
        covariance=cov,
    )
    np.testing.assert_allclose(dv.errors, [1.0, 2.0, 3.0])


def test_datavector_check_covariance_flags_asymmetric():
    dv = DataVector(
        name="bad_cov",
        source="test",
        bins=np.arange(2),
        bin_type="theta_arcmin",
        bin_units="arcmin",
        values=np.zeros(2),
        value_units="test",
    )
    dv.covariance = np.array([[1.0, 0.5], [0.9, 1.0]])  # not symmetric
    with pytest.raises(AssertionError):
        dv.check_covariance()


def test_datavector_check_covariance_flags_non_psd():
    dv = DataVector(
        name="bad_cov",
        source="test",
        bins=np.arange(2),
        bin_type="theta_arcmin",
        bin_units="arcmin",
        values=np.zeros(2),
        value_units="test",
    )
    dv.covariance = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric but indefinite
    with pytest.raises(AssertionError):
        dv.check_covariance()


def test_mass_floor_mask_requires_mass_axis():
    dv = DataVector(
        name="theta_binned",
        source="test",
        bins=np.arange(3),
        bin_type="theta_arcmin",
        bin_units="arcmin",
        values=np.zeros(3),
        value_units="test",
    )
    with pytest.raises(NotImplementedError):
        dv.mass_floor_mask()


# ---------------------------------------------------------------------------
# Real-data round-trips
# ---------------------------------------------------------------------------


def _check_common(dv: DataVector):
    assert isinstance(dv, DataVector)
    assert dv.bins.shape == dv.values.shape
    assert dv.bin_type and dv.bin_units and dv.value_units
    assert dv.h_convention  # asserted (may be "unspecified" — still an explicit statement, not silent)
    dv.check_covariance()


def test_hadzhiyska2024_roundtrip():
    for pzbin in (1, 2, 3, 4):
        dv = load_ksz_hadzhiyska2024(pzbin=pzbin)
        _check_common(dv)
        assert dv.redshift_range is not None
        assert len(dv.values) == 9


def test_hadzhiyska2024_mass_bins_roundtrip():
    out = load_ksz_hadzhiyska2024_mass_bins()
    assert len(out) >= 1
    for dv in out.values():
        _check_common(dv)


def test_ried_guachalla2025_roundtrip():
    for binned_by in ("mass", "redshift"):
        out = load_ksz_ried_guachalla2025(binned_by=binned_by)
        assert len(out) == 4
        for dv in out.values():
            _check_common(dv)
            assert len(dv.values) == 9


def test_ried_guachalla2025_correlation_shape():
    cor = load_ksz_ried_guachalla2025_correlation()
    assert cor.shape == (9, 9)
    np.testing.assert_allclose(np.diag(cor), 1.0, atol=1e-6)


def test_qu2026_lrg_fiducial_roundtrip():
    dv = load_ksz_qu2026_lrg_fiducial()
    _check_common(dv)
    assert len(dv.values) > 0


def test_qu2026_lrg_by_mass_roundtrip():
    out = load_ksz_qu2026_lrg_by_mass()
    assert len(out) == 4
    floor_flags = []
    for dv in out.values():
        _check_common(dv)
        floor_flags.append("clears" in dv.notes)
    # per WP1's freeze-doc read: only the top 2 of 4 stellar-mass bins clear M>=1e13
    assert sum(floor_flags) == 2


def test_hadzhiyska2026_bgs_elg_roundtrip():
    dv_c = load_ksz_hadzhiyska2026_bgs_elg(sample="BGS_BRIGHT-20.2", log10_mstar=11.25, space="config")
    _check_common(dv_c)
    dv_h = load_ksz_hadzhiyska2026_bgs_elg(sample="BGS_BRIGHT-20.2", log10_mstar=11.25, space="harmonic")
    _check_common(dv_h)


def test_egas_erass1_primary_roundtrip():
    cat = load_egas_erass1(catalog="primary")
    assert cat.n_objects == 12247
    assert cat.log10_m500_msun is not None
    assert cat.f_gas500 is not None
    assert cat.log10_m500_msun.shape == (cat.n_objects,)
    frac = cat.sub_floor_fraction(13.0)
    assert 0.0 <= frac <= 1.0


def test_egas_erass1_cosmology_has_no_mass_column():
    cat = load_egas_erass1(catalog="cosmology")
    assert cat.n_objects == 5259
    assert cat.log10_m500_msun is None
    with pytest.raises(ValueError):
        cat.mass_floor_mask()


def test_kappay_pandey2025_roundtrip():
    for component in ("xip", "xim", "compton_shear"):
        dv = load_kappa_y_pandey2025(component=component)
        _check_common(dv)
        assert len(dv.values) > 0


def test_kappay_pandey2025_compton_shear_length():
    dv = load_kappa_y_pandey2025(component="compton_shear")
    assert len(dv.values) == 80  # per the on-disk COVMAT block sizing (200+200+80=480)


def test_kappay_pandey2025_full_covariance_shape_and_psd():
    cov = load_kappa_y_pandey2025_full_covariance()
    assert cov.shape == (480, 480)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-5)
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() > -1e-6 * eigvals.max()


def test_kappay_pandey2025_nz():
    for which in ("lens", "source"):
        nz = load_kappa_y_pandey2025_nz(which=which)
        assert "z_mid" in nz or "Z_MID" in nz
