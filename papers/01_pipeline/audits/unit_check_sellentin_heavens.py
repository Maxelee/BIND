"""Standalone unit-check for sellentin_heavens_loglike (piece 2, shared setup
cell of _build_figures_nb.py) and the Gaussian/Hartlap vs Sellentin-Heavens
p-value conversion used in the fig04 chi2 stamps.

Run: /mnt/home/mlee1/venvs/BIND_env/bin/python audits/unit_check_sellentin_heavens.py

This is a throwaway synthetic-data check, NOT part of the builder/notebook.
"""
import numpy as np
from scipy.stats import chi2 as chi2dist, f as fdist


def sellentin_heavens_loglike(chi2, n_real, n_data):
    """Exact copy of the helper added to the shared setup cell -- kept in sync
    by hand (no import path back into the builder's generated notebook)."""
    n_real = np.asarray(n_real, float)
    return -0.5 * n_real * np.log1p(np.asarray(chi2, float) / (n_real - 1))


def gaussian_loglike(chi2):
    return -0.5 * np.asarray(chi2, float)


def check_gaussian_limit():
    """As n_real -> infinity, SH loglike -> Gaussian loglike (up to the same
    'additive constant' convention), for a range of chi2 values."""
    chi2_vals = np.array([0.1, 1.0, 5.0, 25.0, 100.0])
    print("chi2      n_real     SH loglike     Gaussian loglike     |diff|")
    for n_real in [10, 50, 200, 1_000, 10_000, 1_000_000]:
        sh = sellentin_heavens_loglike(chi2_vals, n_real, n_data=5)
        gg = gaussian_loglike(chi2_vals)
        diff = np.abs(sh - gg)
        print(f"n_real={n_real:>9d}: max|SH-Gaussian| over chi2={list(chi2_vals)} "
              f"-> {diff.max():.6f}")
    # quantitative pass/fail: the leading-order expansion is
    # SH - Gaussian ~ chi2^2/(4*n_real), so agreement to 1e-3 at chi2=100
    # needs n_real >~ 100^2/(4e-3) = 2.5e6; use n_real=1e8 for headroom.
    sh_big = sellentin_heavens_loglike(chi2_vals, 100_000_000, 5)
    gg_big = gaussian_loglike(chi2_vals)
    ok = np.allclose(sh_big, gg_big, atol=1e-3)
    print(f"PASS (n_real=1e8 matches Gaussian to 1e-3)? {ok}")
    assert ok, "sellentin_heavens_loglike does not converge to the Gaussian limit"


def check_monotone_and_less_confident():
    """SH loglike must (a) be monotonically DEcreasing in chi2 (still a valid
    likelihood ranking) and (b) be <= the Gaussian loglike at finite n_real
    (heavier tails -> less confident / less negative penalty at large chi2,
    i.e. SH - Gaussian should be an INCREASING function of chi2, positive at
    large chi2 -- SH assigns MORE probability to large-chi2 outliers)."""
    chi2_grid = np.linspace(0.01, 50, 200)
    for n_real in (20, 50, 100):
        sh = sellentin_heavens_loglike(chi2_grid, n_real, n_data=5)
        assert np.all(np.diff(sh) < 0), f"SH loglike not monotone decreasing at n_real={n_real}"
        gg = gaussian_loglike(chi2_grid)
        gap = sh - gg
        # heavy tails: SH should become RELATIVELY more permissive (gap
        # increasing) as chi2 grows, at finite n_real
        assert gap[-1] > gap[0], f"SH does not show heavier tails at n_real={n_real}"
    print("PASS: SH loglike monotone decreasing in chi2, and relatively "
          "heavier-tailed than Gaussian at finite n_real.")


def check_hotelling_p_value_uniform_under_null():
    """The Gaussian/Hartlap-vs-SH p-value machinery added to the fig04 chi2
    stamps (Hotelling T^2 -> F(p, n-p) equivalence): under the NULL (data
    truly drawn from a multivariate normal with the estimated covariance),
    the resulting p-value should be ~Uniform(0,1). Monte Carlo check."""
    rng = np.random.default_rng(0)
    n_real, p = 50, 25
    n_trials = 20_000
    pvals = []
    for _ in range(n_trials):
        # draw a "true" covariance-generating sample as in chi2_full: n_real
        # realizations of a p-dim Gaussian with identity covariance
        X = rng.standard_normal((n_real, p))
        Cm = np.cov(X, rowvar=False) / n_real
        dv = X.mean(0)                      # true mean is 0
        chi2_raw = float(dv @ np.linalg.solve(Cm, dv))
        F_stat = ((n_real - p) / (p * (n_real - 1))) * chi2_raw
        pvals.append(fdist.sf(F_stat, p, n_real - p))
    pvals = np.array(pvals)
    # KS-style sanity: mean should be ~0.5, and the fraction below 0.05
    # should be ~5% (both within a few Monte Carlo sigma at n_trials=20000)
    frac_below_05 = float((pvals < 0.05).mean())
    print(f"Hotelling-F p-value under the null: mean={pvals.mean():.3f} "
          f"(expect ~0.5), frac(p<0.05)={frac_below_05:.4f} (expect ~0.05)")
    assert abs(pvals.mean() - 0.5) < 0.01
    assert abs(frac_below_05 - 0.05) < 0.01
    print("PASS: Hotelling-F p-value is calibrated (uniform) under the null.")


if __name__ == "__main__":
    check_gaussian_limit()
    print()
    check_monotone_and_less_confident()
    print()
    check_hotelling_p_value_uniform_under_null()
    print("\nALL CHECKS PASSED")
