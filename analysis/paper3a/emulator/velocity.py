"""Linear-theory velocity statistics for the kSZ forward model (numpy-only).

Two quantities feed WP-A4/A5 (DECISION_MEMO axis 3, "fix b"):

- ``sigma_v_1d_kms(z)``: the linear-theory 1D rms peculiar velocity — the
  sigma_true(z) in the Qu et al. normalization T_kSZ = T_CMB (sigma/c) tau_CAP
  (their Eq. 30 convention; the estimator's velocity-correlation coefficient
  r ~ 0.65 is applied on the data side, per the A1 freeze audit).
- ``r_v_parallel(dchi)``: the longitudinal velocity correlation coefficient
  between two points separated by ``dchi`` along the line of sight,
  Psi_par(r)/Psi_par(0) with (Gorski 1988)

      Psi_par(r) = (f a H D)^2 / (2 pi^2) * Int dk P0(k) [j0(kr) - 2 j1(kr)/(kr)]

  The prefactor cancels in the ratio, so r_v is redshift-independent in
  linear theory. ``slab_mean_rv(L)`` averages it over a uniform line-of-sight
  offset within a slab of comoving depth L — the weight the co-moving-column
  approximation of the painted composites must be corrected by.

Power spectrum: Eisenstein & Hu (1998) zero-baryon (no-wiggle) transfer,
normalized to sigma8. BAO wiggles are irrelevant here — the velocity
integrands weight P(k) (not k^2 P), so they are dominated by k ~ 0.01-0.1
h/Mpc where the smooth form tracks Boltzmann codes to a few percent; that
accuracy enters the kSZ *correction factor*, not the observable itself.
Growth rate f = Omega_m(z)^0.55; nonlinear velocity contributions (~10% on
sigma_v at z<1) are carried in the WP-A4 error budget, not modeled.
"""

from __future__ import annotations

import numpy as np

from .params_meta import FIXED_COSMOLOGY


def _j0(x):
    return np.where(x < 1e-6, 1.0 - x**2 / 6.0, np.sin(x) / np.maximum(x, 1e-300))


def _j1_over_x(x):
    small = x < 1e-4
    xs = np.maximum(x, 1e-4)          # both where-branches evaluate; keep finite
    return np.where(small, 1.0 / 3.0 - x**2 / 30.0,
                    (np.sin(xs) / xs**2 - np.cos(xs) / xs) / xs)


class LinearVelocity:
    """Linear-theory velocity statistics at the fixed TNG fiducial cosmology."""

    C_KMS = 299792.458

    def __init__(self, omega_m: float | None = None, omega_b: float | None = None,
                 h: float | None = None, n_s: float | None = None,
                 sigma8: float | None = None, t_cmb: float = 2.7255,
                 k_grid: np.ndarray | None = None):
        self.om = FIXED_COSMOLOGY["Omega0"] if omega_m is None else omega_m
        self.ob = FIXED_COSMOLOGY["OmegaBaryon"] if omega_b is None else omega_b
        self.h = FIXED_COSMOLOGY["HubbleParam"] if h is None else h
        self.ns = FIXED_COSMOLOGY["n_s"] if n_s is None else n_s
        self.sigma8 = FIXED_COSMOLOGY["sigma8"] if sigma8 is None else sigma8
        self.t_cmb = t_cmb
        self.k = k_grid if k_grid is not None else np.geomspace(1e-4, 50.0, 4096)  # h/Mpc
        self._pk = self._pk_z0(self.k)

    # ------------------------------------------------- P(k), EH98 no-wiggle --

    def _transfer_nowiggle(self, k_hmpc: np.ndarray) -> np.ndarray:
        omh2 = self.om * self.h**2
        obh2 = self.ob * self.h**2
        theta = self.t_cmb / 2.7
        s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2**0.75)  # Mpc
        a_gam = (1.0 - 0.328 * np.log(431.0 * omh2) * self.ob / self.om
                 + 0.38 * np.log(22.3 * omh2) * (self.ob / self.om) ** 2)
        ks = k_hmpc * self.h * s                       # dimensionless
        gamma_eff = self.om * self.h * (a_gam + (1.0 - a_gam) / (1.0 + (0.43 * ks) ** 4))
        q = k_hmpc * theta**2 / gamma_eff
        L0 = np.log(2.0 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        return L0 / (L0 + C0 * q**2)

    def _pk_z0(self, k: np.ndarray) -> np.ndarray:
        """sigma8-normalized z=0 linear P(k) in (Mpc/h)^3, k in h/Mpc."""
        pk = k**self.ns * self._transfer_nowiggle(k) ** 2
        x = k * 8.0
        w = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
        s8sq = np.trapz(k**2 * pk * w**2, k) / (2.0 * np.pi**2)
        return pk * self.sigma8**2 / s8sq

    # ------------------------------------------------------------- growth ----

    def _efunc(self, z):
        return np.sqrt(self.om * (1.0 + z) ** 3 + (1.0 - self.om))

    def growth(self, z: float) -> float:
        """D(z)/D(0), flat LCDM integral form."""

        def _d(a):
            aa = np.linspace(1e-4, a, 2048)
            ea = np.sqrt(self.om / aa**3 + (1.0 - self.om))
            integ = np.trapz(1.0 / (aa * ea) ** 3, aa)
            return ea[-1] * integ

        return _d(1.0 / (1.0 + z)) / _d(1.0)

    def growth_rate(self, z: float) -> float:
        """f = dlnD/dlna ~ Omega_m(z)^0.55."""
        om_z = self.om * (1.0 + z) ** 3 / self._efunc(z) ** 2
        return om_z**0.55

    # ------------------------------------------------- velocity statistics ---

    def sigma_v_1d_kms(self, z: float) -> float:
        """Linear-theory 1D rms peculiar velocity at z (km/s):
        sigma^2 = (f a H D)^2 / (6 pi^2) Int dk P0(k)."""
        a = 1.0 / (1.0 + z)
        faHD = (self.growth_rate(z) * a * 100.0 * self._efunc(z) * self.growth(z))
        i0 = np.trapz(self._pk, self.k)                       # (Mpc/h)^2
        return faHD * np.sqrt(i0 / (6.0 * np.pi**2))

    def sigma_v_over_c(self, z: float) -> float:
        return self.sigma_v_1d_kms(z) / self.C_KMS

    def r_v_parallel(self, dchi_mpch: np.ndarray) -> np.ndarray:
        """Longitudinal velocity correlation coefficient at comoving LOS
        separation ``dchi`` (Mpc/h); redshift-independent in linear theory."""
        r = np.atleast_1d(np.abs(np.asarray(dchi_mpch, float)))
        kr = self.k[None, :] * r[:, None]
        kernel = _j0(kr) - 2.0 * _j1_over_x(kr)
        psi = np.trapz(self._pk[None, :] * kernel, self.k, axis=1)
        psi0 = np.trapz(self._pk / 3.0, self.k)
        out = psi / psi0
        return out if np.ndim(dchi_mpch) else float(out[0])

    def slab_mean_rv(self, slab_depth_mpch: float) -> float:
        """Mean r_v over a uniform LOS offset within a slab of depth L:
        (2/L) Int_0^{L/2} r_v(x) dx — the decorrelation weight of the
        painted co-moving column / in-slab two-halo term."""
        x = np.linspace(0.0, slab_depth_mpch / 2.0, 512)
        return float(np.trapz(self.r_v_parallel(x), x) / (slab_depth_mpch / 2.0))

    def xi_lin(self, s_mpch: np.ndarray) -> np.ndarray:
        """z=0 linear matter correlation function xi(s), s in Mpc/h.
        Gaussian small-scale damping (k_damp = 20 h/Mpc) regularizes the
        oscillatory tail; below s ~ 0.3 Mpc/h treat values as indicative."""
        s = np.atleast_1d(np.asarray(s_mpch, float))
        damp = np.exp(-((self.k / 20.0) ** 2))
        ks = self.k[None, :] * np.maximum(s, 1e-3)[:, None]
        xi = np.trapz(self.k[None, :] ** 2 * self._pk[None, :] * damp[None, :]
                      * _j0(ks), self.k, axis=1) / (2.0 * np.pi**2)
        return xi if np.ndim(s_mpch) else float(xi[0])
