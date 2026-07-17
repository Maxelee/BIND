"""Forward model: emulated painted observables -> data-space predictions.

This is the layer the WP-A5 likelihood calls. The emulator (:mod:`gasemu`)
predicts *painted-frame* observables; everything comparing to data applies:

f_gas (eRASS1 frame)
    - CylToSph: the wp2 truth-calibrated cylindrical -> spherical factor
      (run-2451362 summaries via ``aggregate.cyltosph_for_snap``); its
      feedback dependence is the gate call's mandatory component #2 —
      carried as an explicit fractional systematic until the CAMELS-1P
      measurement lands.
    - The eRASS1 mass-PDF convolution (mandatory component #1) belongs to
      the A5 likelihood: this layer predicts f_gas at *true* mass.

kSZ (Qu et al. frame) — the velocity-decorrelation forward model (fix b)
    - Normalization: T_kSZ = T_CMB * (sigma_true(z)/c) * tau with the
      linear-theory sigma_true from :mod:`velocity` (replaces the A3
      NOMINAL_VRMS_OVER_C; Qu's estimator correlation r ~ 0.65 is applied
      on the data side per the A1 freeze audit — do not divide it in here).
    - Decorrelation: the painted composite treats every gas parcel in the
      51.25 Mpc/h slab as co-moving with the stacked central; the real
      pairwise-velocity coherence falls off along the LOS as r_v(dchi)
      (:meth:`LinearVelocity.r_v_parallel`). What survives stacking is the
      halo-correlated excess profile Sigma(R) - Sigma_floor, whose LOS
      extent at transverse radius R is distributed like xi(sqrt(R^2+x^2)),
      so each transverse radius carries the weight

          w(R) = Int dx xi(sqrt(R^2+x^2)) r_v(x) / Int dx xi(sqrt(R^2+x^2))

      (x over the half-slab; xi = linear matter correlation — conservative
      by a few % at 1-halo radii where gas is strictly co-moving with the
      halo bulk). The v2 dry run showed the two-halo term is *negative* in
      CAP space at large aperture (neighbors fill the compensation
      annulus), so scalar own/total weighting is wrong — the weight must be
      applied to the stacked Sigma(R) profile *before* CAP projection.
      That profile is the v3 operator column ``ksz_binN_sigma_r``
      (+ ``_own`` diagnostic split); :meth:`ksz_tksz_from_profile`
      implements weight -> synthetic map -> beam -> CAP with the same
      operator code path as the tables.
    - Not modeled (documented in the A4 error budget): correlated gas
      beyond the +-25.6 Mpc/h slab (adds signal; the painted maps cannot
      contain it) and nonlinear/1-halo velocity corrections.

Pre-v3 fallback: :meth:`ksz_tksz_bracket` brackets the emulated v2 CAP
profile between all-decorrelated and all-co-moving scalings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.paper3a.gate.aggregate import cyltosph_for_snap
from analysis.paper3a.observables import (
    FlatLCDM,
    KSZOperatorConfig,
    PatchGeometry,
    ksz_cap_profile,
)
from analysis.paper3a.observables.constants import T_CMB_UK

from .gasemu import GasEmulator
from .velocity import LinearVelocity

SLAB_DEPTH_HMPC = 51.25
CANVAS_PIXEL_MPCH = 205.0 / 4198
CUTOUT_PIX = 361
HILC_BEAM_FWHM_ARCMIN = 1.6


def transverse_weight(lv: LinearVelocity, r_mpch: np.ndarray,
                      slab_depth: float = SLAB_DEPTH_HMPC) -> np.ndarray:
    """w(R): xi-weighted mean LOS velocity correlation of the excess gas at
    transverse radius R (see module docstring)."""
    r = np.atleast_1d(np.asarray(r_mpch, float))
    x = np.linspace(0.0, slab_depth / 2.0, 257)
    s = np.sqrt(r[:, None] ** 2 + x[None, :] ** 2)
    xi = lv.xi_lin(s.ravel()).reshape(s.shape)
    xi = np.maximum(xi, 0.0)                      # in-slab s < 30 Mpc/h: xi > 0 anyway
    rv = lv.r_v_parallel(x)
    num = np.trapz(xi * rv[None, :], x, axis=1)
    den = np.trapz(xi, x, axis=1)
    return num / np.maximum(den, 1e-300)


@dataclass
class KszForwardResult:
    radii_arcmin: np.ndarray
    tksz: np.ndarray              # muK arcmin^2, decorrelated
    tksz_raw: np.ndarray          # muK arcmin^2, co-moving (no decorrelation)
    sigma_v_over_c: float
    weight_at_r: np.ndarray | None = None


class ForwardModel:
    """Data-space predictions from a fitted :class:`GasEmulator`."""

    def __init__(self, emulator: GasEmulator | None = None,
                 velocity: LinearVelocity | None = None):
        self.emu = emulator
        self.lv = velocity or LinearVelocity()
        self._w_nbr = self.lv.slab_mean_rv(SLAB_DEPTH_HMPC)

    # ------------------------------------------------------------- f_gas ----

    def fgas_sph(self, params, snap: str, cyltosph: float | None = None,
                 cyltosph_sys_frac: float = 0.0) -> dict:
        """Spherical-equivalent f_gas medians (and scatters) per gate mass
        bin at true mass. ``cyltosph_sys_frac`` widens the result by the
        (currently unmeasured) feedback dependence of the factor."""
        if cyltosph is None:
            cyltosph, src = cyltosph_for_snap(snap)
        else:
            src = "caller"
        pred = self.emu.predict(params, snap)
        med = pred["fgas_med"] * cyltosph
        return {
            "fgas_sph_med": med,
            "fgas_sph_scat": pred["fgas_scat"] * cyltosph,
            "fgas_sph_med_std": pred["fgas_med_std"] * cyltosph,
            "cyltosph": cyltosph, "cyltosph_source": src,
            "cyltosph_sys": np.abs(med) * cyltosph_sys_frac,
            "note": "true-mass frame; A5 convolves with eRASS1 per-cluster mass PDFs",
        }

    # --------------------------------------------------------------- kSZ ----

    def ksz_tksz_from_profile(self, sigma_r: np.ndarray, r_centers_mpch: np.ndarray,
                              z: float, radii_arcmin: np.ndarray,
                              n_floor_bins: int = 10,
                              cosmology: FlatLCDM | None = None) -> KszForwardResult:
        """Decorrelated T_kSZ CAP profile from a stacked Sigma(R) profile
        (the v3 ``ksz_binN_sigma_r`` column, comoving Msun/h per pixel).

        Weights the excess over the outer-plateau floor by w(R), rebuilds a
        synthetic radial map on the operator's cutout grid, and runs the
        *same* beam + CAP code path as the gate tables.
        """
        sigma_r = np.asarray(sigma_r, float)
        floor = float(np.mean(sigma_r[-n_floor_bins:]))
        w = transverse_weight(self.lv, r_centers_mpch)
        sigma_w = floor + w * (sigma_r - floor)

        geom = PatchGeometry(pixel_mpch=CANVAS_PIXEL_MPCH, z=z,
                             cosmology=cosmology or FlatLCDM())
        cfg = KSZOperatorConfig(radii_arcmin=np.asarray(radii_arcmin, float),
                                beam_fwhm_arcmin=HILC_BEAM_FWHM_ARCMIN,
                                z_eff=z, v_rms_over_c=None)
        n = CUTOUT_PIX
        c = (n - 1) / 2.0
        yy, xx = np.mgrid[0:n, 0:n]
        rr_mpch = np.hypot(yy - c, xx - c) * CANVAS_PIXEL_MPCH

        def _cap(profile):
            m = np.interp(rr_mpch, r_centers_mpch, profile,
                          left=profile[0], right=floor)
            return ksz_cap_profile(m, geom, cfg)

        svc = self.lv.sigma_v_over_c(z)
        norm = T_CMB_UK * svc
        return KszForwardResult(
            radii_arcmin=np.asarray(radii_arcmin, float),
            tksz=norm * _cap(sigma_w),
            tksz_raw=norm * _cap(sigma_r),
            sigma_v_over_c=svc,
            weight_at_r=w,
        )

    def ksz_tksz_bracket(self, params, snap: str, mass_bin: int) -> dict:
        """Pre-v3 fallback from the emulated v2 CAP profile: T_kSZ bracketed
        between full decorrelation of everything (slab-mean r_v) and the
        co-moving limit. The profile-space model supersedes this once the
        v3 sigma_r columns land."""
        z = self.emu.snap_z[snap]
        pred = self.emu.predict(params, snap, return_std=False)
        tau = pred[f"ksz{mass_bin}"]
        norm = T_CMB_UK * self.lv.sigma_v_over_c(z)
        return {
            "radii_arcmin": self.emu.radii_arcmin,
            "tksz_lo": norm * self._w_nbr * tau,
            "tksz_hi": norm * tau,
            "sigma_v_over_c": self.lv.sigma_v_over_c(z),
            "w_slab": self._w_nbr,
        }
