"""WP-B4 amplitude-corrected weighting tests (synthetic shell fixtures)."""

import numpy as np
import pytest

from analysis.paper3b.mocks.nz import (
    ShellAmplitudeModel,
    SourcePlaneWeighting,
    amplitude_corrected_weights,
    lensing_efficiency,
    source_plane_weights,
)

PLANES = np.array([0.5, 1.0, 1.5, 2.0, 2.44])


def _single_shell_model(z_shell=0.3, power=4.0):
    return ShellAmplitudeModel(shells=np.array([z_shell]), q=np.array([power]),
                               omega_m=0.3089, max_fit_frac_err=0.0)


def test_fit_recovers_planted_shells():
    # Build a synthetic plane covariance from known shells; the fit must
    # reproduce the amplitude function A(z) it implies.
    true = ShellAmplitudeModel(shells=np.array([0.2, 0.6]), q=np.array([3.0, 1.5]),
                               omega_m=0.3089, max_fit_frac_err=0.0)
    W = np.array([lensing_efficiency(zj, PLANES) for zj in true.shells]).T
    C = (W * true.q) @ W.T
    # fit grid CONTAINS the planted shells -> exact recovery is achievable
    fit = ShellAmplitudeModel.fit_plane_cov(C, PLANES,
                                            shells=np.array([0.1, 0.2, 0.4, 0.6, 1.0]))
    assert fit.max_fit_frac_err < 1e-8
    zt = np.linspace(0.05, 2.4, 40)
    # atol covers z below the lowest planted shell, where true A = 0 and NNLS
    # leaves ~1e-14 spurious shell powers (~1e-7 in amplitude)
    np.testing.assert_allclose(fit.amplitude(zt), true.amplitude(zt),
                               rtol=1e-4, atol=1e-6)


def test_amplitude_weights_all_mass_at_a_plane_gives_unit_weight():
    # n(z) concentrated exactly at a plane: A(z)/A(z_k) = 1 there, so the
    # corrected weight equals the plain binned weight (1 on that plane).
    model = _single_shell_model()
    zt = np.linspace(1.4, 1.6, 101)
    nt = np.exp(-0.5 * ((zt - 1.5) / 0.02) ** 2)
    w = amplitude_corrected_weights(PLANES, zt, nt, model)
    assert w[2] == pytest.approx(1.0, abs=1e-3)
    assert np.delete(w, 2).max() < 1e-12


def test_amplitude_weights_downweight_sub_plane_mass():
    # Mass below the lowest plane must get weight < its plain binned weight,
    # by the ratio A(z)/A(0.5) < 1.
    model = _single_shell_model()
    zt = np.linspace(0.31, 0.45, 101)          # entirely below 0.5, above shell
    nt = np.ones_like(zt)
    w_plain = source_plane_weights(PLANES, zt, nt)
    w_amp = amplitude_corrected_weights(PLANES, zt, nt, model)
    assert w_plain[0] == pytest.approx(1.0)
    assert 0 < w_amp[0] < 1.0
    ratio = np.trapezoid(model.amplitude(zt), zt) / ((zt[-1] - zt[0]) * model.amplitude([0.5])[0])
    assert w_amp[0] == pytest.approx(ratio, rel=1e-6)


def test_weighting_accepts_subunity_sum_rejects_over_unity():
    w = SourcePlaneWeighting(PLANES, np.array([0.5, 0.2, 0.05, 0.01, 0.01]))
    assert w.weights.sum() < 1
    with pytest.raises(ValueError):
        SourcePlaneWeighting(PLANES, np.array([0.9, 0.2, 0.05, 0.01, 0.01]))
    with pytest.raises(ValueError):
        SourcePlaneWeighting(PLANES, np.array([0.5, -0.1, 0.05, 0.01, 0.01]))


def test_efficiency_basics():
    w = lensing_efficiency(0.3, np.array([0.1, 0.3, 1.0, 2.0]))
    assert w[0] == 0.0 and w[1] == 0.0        # sources at/inside the lens
    assert 0 < w[2] < w[3] < 1                # efficiency grows with z_s
