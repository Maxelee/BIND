"""Lightweight emulator + Gaussian posterior for the kSZ τ(M) observable.

Used by validation_e (SBI coverage) and validation_f (v_los robustness).
We keep this dependency-free (no torch / sbi / GPy) by assuming a *linear*
emulator x = A θ_std + b plus a Gaussian likelihood with diagonal residual
covariance.  With a Gaussian prior on standardised θ, the posterior is
analytic — which is enough for the §4.E coverage and §4.F robustness checks
in docs/paper2_ksz_plan.md.

Conventions
-----------
- θ ∈ R^{P}     : raw 35-dim CAMELS parameter vector for a sim.
- x ∈ R^{K}    : per-mass-bin stacked τ values for a sim (output of the
                  same stacking used in validation_d).
- θ_std        : (θ − θ_mean) / θ_std_dev, where the moments come from the
                  training set.

The fit uses per-bin ridge regression on θ_std → x_k.  Σ_resid is the
diagonal of the training residuals.  Total noise covariance for inference
is Σ_obs = Σ_resid + Σ_meas where Σ_meas comes from the data-side error
budget (defaults: 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Emulator
# --------------------------------------------------------------------------- #


@dataclass
class StackedEmulator:
    """Per-mass-bin ridge regression θ_std → x_k.

    Attributes
    ----------
    theta_mean, theta_std : (P,) standardisation moments.
    A : (K, P)   regression coefficients in std-θ space.
    b : (K,)     intercept (≈ training mean of x_k).
    sigma_resid : (K,)   per-bin residual standard deviation.
    ridge : float        ridge regularisation used at fit time.
    """

    theta_mean: np.ndarray
    theta_std: np.ndarray
    A: np.ndarray
    b: np.ndarray
    sigma_resid: np.ndarray
    ridge: float

    # --------------------------------------------------------------- fit --
    @classmethod
    def fit(
        cls,
        theta: np.ndarray,           # (N, P)
        x: np.ndarray,               # (N, K)
        ridge: float = 1e-2,
        eps: float = 1e-12,
    ) -> "StackedEmulator":
        theta = np.asarray(theta, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        n, p = theta.shape
        k = x.shape[1]
        mu = theta.mean(axis=0)
        sd = theta.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        ts = (theta - mu) / sd

        # ridge solve per bin: A x_k = (ts^T ts + λI)^{-1} ts^T (x_k − x̄_k)
        b = x.mean(axis=0)
        x_c = x - b
        gram = ts.T @ ts + ridge * np.eye(p)
        rhs = ts.T @ x_c                              # (P, K)
        coef = np.linalg.solve(gram, rhs)             # (P, K)
        A = coef.T                                    # (K, P)

        pred = ts @ coef + b                           # (N, K)
        sigma_resid = np.sqrt(((x - pred) ** 2).mean(axis=0))
        return cls(
            theta_mean=mu,
            theta_std=sd,
            A=A,
            b=b,
            sigma_resid=sigma_resid,
            ridge=float(ridge),
        )

    # --------------------------------------------------------- prediction --
    def predict(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        ts = (theta - self.theta_mean) / self.theta_std
        return ts @ self.A.T + self.b                  # (N, K)


# --------------------------------------------------------------------------- #
# Posterior
# --------------------------------------------------------------------------- #


@dataclass
class GaussianPosterior:
    """Analytic Gaussian posterior on raw θ given x_obs.

    Built from a `StackedEmulator` plus a diagonal observation covariance.
    Prior on θ_std is N(0, prior_std² I).  Posterior on θ_std is also Gaussian;
    raw-θ mean/std follow by un-standardising.

    Attributes
    ----------
    mean : (P,)     posterior mean in raw θ units.
    std  : (P,)     1-σ marginal on each raw θ_i.
    """

    mean: np.ndarray
    std: np.ndarray
    cov_std: np.ndarray   # posterior covariance in std-θ space, (P, P)

    @classmethod
    def from_observation(
        cls,
        emu: StackedEmulator,
        x_obs: np.ndarray,            # (K,)
        sigma_meas: np.ndarray | None = None,  # (K,) extra meas. σ to add in quad
        prior_std: float = 3.0,
    ) -> "GaussianPosterior":
        x_obs = np.asarray(x_obs, dtype=np.float64).ravel()
        sigma = emu.sigma_resid.copy()
        if sigma_meas is not None:
            sigma = np.sqrt(sigma ** 2 + np.asarray(sigma_meas, dtype=np.float64) ** 2)
        sigma = np.where(sigma <= 0, 1e-12, sigma)
        prec_obs = 1.0 / (sigma ** 2)                        # (K,)

        p = emu.A.shape[1]
        # Precision in std-θ space: A^T diag(prec_obs) A + I/prior_std²
        AtP = emu.A.T * prec_obs                              # (P, K)
        prec_std = AtP @ emu.A + np.eye(p) / (prior_std ** 2)
        cov_std = np.linalg.inv(prec_std)
        mean_std = cov_std @ (AtP @ (x_obs - emu.b))          # (P,)

        # Un-standardise: θ = θ_mean + θ_std_dev * θ_std
        mean_raw = emu.theta_mean + emu.theta_std * mean_std
        std_raw = emu.theta_std * np.sqrt(np.diag(cov_std))
        return cls(mean=mean_raw, std=std_raw, cov_std=cov_std)


# --------------------------------------------------------------------------- #
# Per-sim stacked τ(M) (shared by E and F)
# --------------------------------------------------------------------------- #


def stack_per_sim(
    tau_per_halo: np.ndarray,
    masses: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean τ in each mass bin for a single sim.

    Returns (x, n) where x has NaN in bins with no halos and n the per-bin count.
    """
    nb = len(edges) - 1
    idx = np.digitize(masses, edges) - 1
    x = np.full(nb, np.nan, dtype=np.float64)
    n = np.zeros(nb, dtype=np.int64)
    for k in range(nb):
        sel = idx == k
        if sel.any():
            x[k] = tau_per_halo[sel].mean()
            n[k] = int(sel.sum())
    return x, n


# --------------------------------------------------------------------------- #
# Coverage utility (shared by E and F)
# --------------------------------------------------------------------------- #


def central_credible_contains(mu_post: np.ndarray, sd_post: np.ndarray,
                              theta_true: np.ndarray, level: float = 0.6827) -> np.ndarray:
    """Boolean (P,) — does each marginal `level` CI contain `theta_true`?"""
    from scipy.special import erfinv
    z = float(np.sqrt(2.0) * erfinv(level))
    lo = mu_post - z * sd_post
    hi = mu_post + z * sd_post
    return (theta_true >= lo) & (theta_true <= hi)
