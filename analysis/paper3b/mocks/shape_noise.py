"""Gaussian shape noise at DES Y3 source density (WP-B4).

Produced by Project B WP-B4 (forward model; see bind-paper3-plans, private
repo). Pure numpy.

Published spec
--------------
Per-component shape dispersion and effective source density from the DES Y3
weak-lensing shape catalogue: **Gatti, Sheldon et al. 2021, "DES Y3 Results:
Weak Lensing Shape Catalogue", arXiv:2011.03408** — ``σ_e = 0.261`` per
ellipticity component and total ``n_eff = 5.59 gal/arcmin²`` (their C13/
Heymans-convention value), split across the four tomographic bins at
``n_eff ≈ 1.3-1.5 gal/arcmin²`` each. These are recorded as documented
defaults (``DESY3_SIGMA_E``, ``DESY3_NEFF_TOTAL_ARCMIN2``,
``DESY3_NEFF_PERBIN_ARCMIN2``); every noise call still takes n_eff, pixel
area, and σ_e as *explicit* config inputs — no survey value is baked into a
map silently.

Convention (read before trusting a number)
------------------------------------------
Shape noise physically enters the *shear* (γ) field as the per-galaxy
ellipticity scatter σ_e averaged over the ``N = n_eff · A_pix`` galaxies in a
pixel. For a *convergence* (κ) map — the object the B1 peak pipeline runs on —
the standard mock approach (Paper-2 / DES peak-statistics mocks) is to add
white Gaussian noise directly to the pixelized κ with per-pixel dispersion

    σ_pix = σ_e / sqrt(n_eff · A_pix),

then smooth (the smoothing kernel beats the white noise down by the beam
area). This module implements exactly that map-level convention, following the
formula in the WP-B4 plan.

⚠ Factor-of-two convention: some references write ``σ_pix = σ_e /
sqrt(2 · n_eff · A_pix)`` (the ``sqrt(2)`` folds the two ellipticity
components into the E-mode/κ). We follow the plan's per-component form
(no ``sqrt(2)``). Before the B4 mock grid is compared to B1 data, this must be
reconciled with whatever ``bind.inference.stats`` uses for its
``shape_noise_ngal`` path (the twobound atlas' ``peak_cross_ngal10`` variant),
so both sides carry the *same* noise normalization — flag in REPORT.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# DES Y3 documented defaults (Gatti et al. 2021, arXiv:2011.03408).
DESY3_SIGMA_E: float = 0.261
DESY3_NEFF_TOTAL_ARCMIN2: float = 5.59
DESY3_NEFF_PERBIN_ARCMIN2: tuple[float, float, float, float] = (1.48, 1.48, 1.44, 1.19)


@dataclass(frozen=True)
class ShapeNoiseConfig:
    """Per-pixel κ shape-noise parameters (all explicit — no hidden survey value).

    Attributes
    ----------
    n_eff_arcmin2 : effective source number density (gal/arcmin²). ``np.inf``
        means an infinitely sampled (noiseless) map — the "zero 1/n_eff" limit.
    pix_area_arcmin2 : pixel area (arcmin²); e.g. ``PatchGeometry.pixel_area_arcmin2()``.
    sigma_e : per-component shape dispersion (default DES Y3, arXiv:2011.03408).
    """

    n_eff_arcmin2: float
    pix_area_arcmin2: float
    sigma_e: float = DESY3_SIGMA_E

    def pixel_sigma(self) -> float:
        """Per-pixel κ noise dispersion σ_pix = σ_e / sqrt(n_eff · A_pix).

        ``n_eff = inf`` (the zero-1/n_eff limit) returns 0.0 -> no noise.
        """
        if self.pix_area_arcmin2 <= 0:
            raise ValueError(f"pix_area_arcmin2 must be > 0, got {self.pix_area_arcmin2}")
        if self.sigma_e < 0:
            raise ValueError(f"sigma_e must be >= 0, got {self.sigma_e}")
        if not np.isfinite(self.n_eff_arcmin2):
            return 0.0  # infinite density -> noiseless
        if self.n_eff_arcmin2 < 0:
            raise ValueError(f"n_eff_arcmin2 must be >= 0 or inf, got {self.n_eff_arcmin2}")
        if self.n_eff_arcmin2 == 0:
            return np.inf  # zero galaxies -> undefined; caller should not use n_eff=0
        return self.sigma_e / np.sqrt(self.n_eff_arcmin2 * self.pix_area_arcmin2)


def _as_rng(rng=None, seed=None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def noise_map(shape, config: ShapeNoiseConfig, rng=None, seed=None) -> np.ndarray:
    """White Gaussian κ shape-noise realization of the given ``shape``.

    Draws N(0, σ_pix²) per pixel (σ_pix from ``config.pixel_sigma()``); a
    noiseless config (σ_pix == 0) returns exact zeros. Pass ``seed`` for a
    reproducible draw or a pre-seeded ``rng``.
    """
    sigma = config.pixel_sigma()
    if sigma == 0.0:
        return np.zeros(shape, dtype=float)
    if not np.isfinite(sigma):
        raise ValueError("pixel_sigma is not finite (n_eff=0?) — cannot make a noise map")
    return _as_rng(rng, seed).normal(loc=0.0, scale=sigma, size=shape)


def add_shape_noise(kappa_map, config: ShapeNoiseConfig, rng=None, seed=None) -> np.ndarray:
    """Return ``kappa_map`` plus a fresh shape-noise realization (float64 copy).

    A noiseless config (``n_eff=inf``) returns an unmodified copy of the input.
    """
    kappa_map = np.asarray(kappa_map, dtype=float)
    return kappa_map + noise_map(kappa_map.shape, config, rng=rng, seed=seed)
