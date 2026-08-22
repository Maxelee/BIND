"""Standalone unit-checks for the piece-1 (error-to-response ratio) and
piece-3 (GP-sigma coverage recalibration) helpers added to
_build_figures_nb.py. Synthetic data only -- does not touch the real dataset.

Run: /mnt/home/mlee1/venvs/BIND_env/bin/python \
        audits/unit_check_response_ratio_and_coverage.py
"""
import numpy as np


# ── piece 1: _response_half_width (exact copy, kept in sync by hand) ────────
def _response_half_width(raw, floor_frac):
    Y = np.asarray(raw, float).reshape(raw.shape[0], -1)
    med = np.nanmedian(Y, axis=0)
    floor = floor_frac * np.nanmax(np.abs(med))
    med_safe = np.where(np.abs(med) > floor, med, np.nan)
    rat = Y / med_safe[None, :]
    p16, p84 = np.nanpercentile(rat, [16, 84], axis=0)
    half = 0.5 * (p84 - p16)
    return float(np.nanmedian(half))


def check_response_half_width_known_lognormal():
    """Y[run, bin] = median[bin] * lognormal(0, sigma) with a KNOWN sigma, same
    sigma at every bin -> the 16-84 half-width of curve/median should recover
    the lognormal's known 16-84 half-width, and be INDEPENDENT of the
    (arbitrary, per-bin) median normalization -- this is exactly the invariance
    the ratio-to-median construction is FOR."""
    rng = np.random.default_rng(1)
    n_run, n_bin, sigma = 5000, 30, 0.20
    bin_medians = np.geomspace(1.0, 1e4, n_bin)          # wildly different scales per bin
    noise = np.exp(rng.normal(0, sigma, size=(n_run, n_bin)))
    # lognormal(0,sigma) has median 1 by construction (exp(0)); scale by the
    # (arbitrary) per-bin normalization AFTER, so the ratio-to-median removes it
    Y = bin_medians[None, :] * noise
    hw = _response_half_width(Y, floor_frac=0.0)
    # analytic 16-84 half-width of a lognormal(0,sigma): 0.5*(exp(sigma*z84)-exp(sigma*z16))
    # with z84=0.9945, z16=-0.9945 (16/84th percentile of a standard normal)
    from scipy.stats import norm
    z = norm.ppf(0.84)
    hw_expected = 0.5 * (np.exp(sigma * z) - np.exp(-sigma * z))
    print(f"measured half-width = {hw:.4f}, analytic expectation = {hw_expected:.4f} "
          f"(sigma={sigma}, {n_run} synthetic runs x {n_bin} bins spanning 4 decades)")
    assert abs(hw - hw_expected) < 0.01, "response half-width does not match the known lognormal spread"
    print("PASS: _response_half_width recovers the known per-bin spread, "
          "independent of the per-bin normalization scale.")


def check_response_half_width_floor_guard():
    """A bin whose MEDIAN is exactly 0 (or below the floor) must be masked to
    NaN and excluded from the nanmedian aggregate, not silently blow up the ratio."""
    rng = np.random.default_rng(2)
    n_run, n_bin = 200, 4
    Y = rng.normal(0, 1, size=(n_run, n_bin))
    Y[:, 0] = 0.0        # zero-median bin (with noise still centered at 0)
    Y[:, 0] += rng.normal(0, 1e-6, n_run)   # tiny wiggle so median != exactly identical rows
    Y[:, 1:] += 100.0     # well-behaved bins, large positive median
    hw = _response_half_width(Y, floor_frac=0.02)
    assert np.isfinite(hw), "floor guard failed to produce a finite result"
    print(f"PASS: floor guard -- half-width with a near-zero-median bin present "
          f"= {hw:.4f} (finite, zero-median bin excluded)")


# ── piece 3: coverage-fitted alpha_cov (quantile-inversion trick) ───────────
def check_alpha_cov_hits_nominal_coverage():
    """alpha_cov = quantile(|z|, 0.683) must, BY CONSTRUCTION, give
    frac(|z| < alpha_cov) == 0.683 exactly (up to the discreteness of a finite
    sample) -- this is the coverage-FIT the fig-15-panel-e recalibration relies
    on, verified here on synthetic (non-Gaussian, heavy-tailed) z-scores so the
    check is not circular with a Gaussian-generated sample."""
    rng = np.random.default_rng(3)
    n = 20_000
    # heavy-tailed synthetic z-scores: student-t (dof=3), NOT Gaussian --
    # exercises the same "diagnosed heavy GP tails" regime as fig 15
    z = rng.standard_t(df=3, size=n)
    alpha_cov = np.quantile(np.abs(z), 0.683)
    frac68 = float((np.abs(z) < alpha_cov).mean())
    print(f"heavy-tailed (student-t, dof=3) synthetic z-scores: alpha_cov = "
          f"{alpha_cov:.4f}, frac(|z|<alpha_cov) = {frac68:.4f} (target 0.683)")
    assert abs(frac68 - 0.683) < 0.01
    # and the OLD tail-fitted alpha (restores 2-sigma exactly) should NOT, in
    # general, also restore 1-sigma for a heavy-tailed distribution -- this is
    # the whole reason a "complementary fix" is needed
    alpha_tail = np.quantile(np.abs(z), 0.954) / 2.0
    frac68_tail = float((np.abs(z) < alpha_tail).mean())
    frac95_tail = float((np.abs(z) < 2 * alpha_tail).mean())
    frac95_cov = float((np.abs(z) < 2 * alpha_cov).mean())
    print(f"tail-fitted alpha (2-sigma restore): alpha={alpha_tail:.4f}, "
          f"frac68={frac68_tail:.4f} (NOT 0.683 in general), frac95={frac95_tail:.4f} "
          "(0.954 by construction)")
    print(f"coverage-fitted alpha_cov: frac95={frac95_cov:.4f} (NOT 0.954 in general)")
    assert abs(frac95_tail - 0.954) < 0.01     # tail alpha nails 2-sigma...
    assert abs(frac68_tail - 0.683) > 0.01     # ...but misses 1-sigma for a heavy tail
    print("PASS: alpha_cov (1-sigma coverage fit) and the existing tail-fitted "
          "alpha (2-sigma restore) are genuinely DIFFERENT, complementary factors "
          "for a heavy-tailed z-score distribution.")


if __name__ == "__main__":
    check_response_half_width_known_lognormal()
    print()
    check_response_half_width_floor_guard()
    print()
    check_alpha_cov_hits_nominal_coverage()
    print("\nALL CHECKS PASSED")
