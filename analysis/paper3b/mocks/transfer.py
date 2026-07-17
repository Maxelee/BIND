"""WP-B4/B5 transfer-matched filter — closing the reconstruction gap (B5 design gate).

Produced by Project B WP-B4/B5 (see bind-paper3-plans, private repo,
`projectB/wp5-inference-decomposition/DESIGN_DECISION_FILTER.md`). The frozen
data kappa is a Wiener (resp. GLIMPSE) *reconstruction* on an Nside=1024 grid;
the mock kappa_eff is unfiltered truth + white shape noise. The effective
filters differ, so equal-nu peaks need not select comparable objects. This
module supplies the adopted fix: an isotropic, empirically measured transfer
function

    T(ell) = sqrt( C_ell^{data map} / C_ell^{mock kappa_eff + noise} )

applied to the noisy mock kappa in Fourier space BEFORE the PEAK_DEFINITION
smoothing. Construction properties (the reasons it was chosen over
abundance-quantile matching — see the decision doc):

- zero tunable parameters: numerator = the released map's measured
  pseudo-spectrum over the common footprint (public 2-pt information; the y
  map and the frozen stacks are never touched — blinding-compatible);
  denominator = the atlas bind-fiducial mock through the frozen n(z) weights
  and shape-noise config;
- it absorbs, in one measured operator, the Wiener/GLIMPSE suppression AND
  the Nside=1024 pixel window of the data map;
- only the *shape* of T matters downstream: nu is map-normalized
  (PEAK_DEFINITION section 3), so any overall amplitude error (e.g. the
  f_sky approximation in the pseudo-C_ell) cancels in the measurement;
- peak abundances remain a genuine model *prediction* (abundance matching
  would have equalized them by construction, killing the B5 abundance
  channel).

Band limits: T is measured on ell in [ELL_MIN, ELL_MAX=3*1024-1]. Above
ELL_MAX the data map carries no information (band-limited grid), so T = 0 —
the filtered mock becomes band-limited like the data. Below ELL_MIN (the 5-deg
patch fundamental is ell ~ 72) T holds its ELL_MIN value; the DC mode is
irrelevant (nu subtracts the patch mean).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "ELL_MIN", "ELL_MAX_NSIDE1024",
    "TransferFunction", "flat_sky_power", "build_empirical_transfer",
    "load_transfer",
]

ELL_MIN: float = 80.0                 # ~ the 5-deg patch fundamental (2*pi/L ~ 72)
ELL_MAX_NSIDE1024: float = 3 * 1024 - 1.0


def _ell_grid(npix: int, fov_deg: float) -> np.ndarray:
    """|ell| for every 2D FFT mode of an npix^2 patch of side fov_deg."""
    L = np.deg2rad(fov_deg)
    f = np.fft.fftfreq(npix, d=L / npix)          # cycles / radian
    fx, fy = np.meshgrid(f, f, indexing="ij")
    return 2.0 * np.pi * np.hypot(fx, fy)


def flat_sky_power(patch: np.ndarray, fov_deg: float,
                   n_bins: int = 96,
                   ell_range: tuple[float, float] | None = None,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthally averaged flat-sky power spectrum of one periodic patch.

    Normalization is the standard flat-sky convention C(ell) =
    (L^2 / npix^4) <|FFT|^2>, which matches curved-sky C_ell in the small-angle
    limit (checked in the unit tests against white noise: C_ell = sigma_pix^2 *
    Omega_pix).

    Returns (ell_centers, cl, n_modes) over log-spaced bins; empty bins get
    cl = nan.
    """
    p = np.asarray(patch, dtype=np.float64)
    npix = p.shape[0]
    if p.shape != (npix, npix):
        raise ValueError(f"patch must be square, got {p.shape}")
    L = np.deg2rad(fov_deg)
    ell = _ell_grid(npix, fov_deg)
    power = (L ** 2 / npix ** 4) * np.abs(np.fft.fft2(p)) ** 2

    if ell_range is None:
        ell_range = (2.0 * np.pi / L * 0.9, float(ell.max()))
    edges = np.geomspace(ell_range[0], ell_range[1], n_bins + 1)
    idx = np.digitize(ell.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < n_bins)
    cl = np.full(n_bins, np.nan)
    n_modes = np.bincount(idx[valid], minlength=n_bins)
    sums = np.bincount(idx[valid], weights=power.ravel()[valid], minlength=n_bins)
    occ = n_modes > 0
    cl[occ] = sums[occ] / n_modes[occ]
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, cl, n_modes


@dataclass(frozen=True)
class TransferFunction:
    """Isotropic ell-space transfer, applied multiplicatively in Fourier space.

    ell/t tabulate T; outside the table T holds its end values, EXCEPT above
    `ell_zero` (if set) where T = 0 exactly (band limit of the data grid).
    """

    ell: np.ndarray
    t: np.ndarray
    ell_zero: float | None = ELL_MAX_NSIDE1024

    def __post_init__(self):
        e, t = np.asarray(self.ell, float), np.asarray(self.t, float)
        if e.ndim != 1 or e.shape != t.shape or len(e) < 2:
            raise ValueError("ell and t must be equal-length 1D, len >= 2")
        if not (np.diff(e) > 0).all():
            raise ValueError("ell must be strictly increasing")
        if not np.isfinite(t).all() or (t < 0).any():
            raise ValueError("t must be finite and >= 0")

    def __call__(self, ell) -> np.ndarray:
        out = np.interp(np.asarray(ell, float), self.ell, self.t)
        if self.ell_zero is not None:
            out = np.where(np.asarray(ell, float) > self.ell_zero, 0.0, out)
        return out

    def apply(self, patch: np.ndarray, fov_deg: float) -> np.ndarray:
        """Filter a periodic flat patch: IFFT( FFT(patch) * T(|ell|) )."""
        p = np.asarray(patch, dtype=np.float64)
        ell = _ell_grid(p.shape[0], fov_deg)
        return np.fft.ifft2(np.fft.fft2(p) * self(ell)).real


def build_empirical_transfer(ell_data: np.ndarray, cl_data: np.ndarray,
                             ell_mock: np.ndarray, cl_mock: np.ndarray,
                             ell_min: float = ELL_MIN,
                             ell_max: float = ELL_MAX_NSIDE1024,
                             n_ell: int = 64,
                             smooth_dex: float = 0.15,
                             t_cap: float = 2.0) -> TransferFunction:
    """T(ell) = sqrt(C_data / C_mock), log-ell smoothed, on [ell_min, ell_max].

    Both inputs are interpolated in log-log onto a common log-spaced grid;
    the ratio is smoothed with a Gaussian of width `smooth_dex` in log10(ell)
    (suppresses bin noise near the band edges), clipped to [0, t_cap]. The
    returned TransferFunction is 0 above ell_max (data band limit).
    """
    grid = np.geomspace(ell_min, ell_max, n_ell)

    def _loginterp(e, c):
        e, c = np.asarray(e, float), np.asarray(c, float)
        ok = np.isfinite(c) & (c > 0) & (e > 0)
        if ok.sum() < 4:
            raise ValueError("need >= 4 positive spectrum bins")
        return np.exp(np.interp(np.log(grid), np.log(e[ok]), np.log(c[ok])))

    ratio = _loginterp(ell_data, cl_data) / _loginterp(ell_mock, cl_mock)
    t = np.sqrt(np.clip(ratio, 0.0, t_cap ** 2))

    if smooth_dex > 0:
        lg = np.log10(grid)
        w = np.exp(-0.5 * ((lg[:, None] - lg[None, :]) / smooth_dex) ** 2)
        t = (w @ t) / w.sum(axis=1)
    return TransferFunction(grid, t, ell_zero=ell_max)


def load_transfer(npz_path: str | Path, variant: str) -> TransferFunction:
    """Load a persisted transfer (build_b5_transfer.py output) by variant name."""
    d = np.load(Path(npz_path))
    key_e, key_t = f"ell_{variant}", f"t_{variant}"
    if key_e not in d.files:
        raise KeyError(f"variant '{variant}' not in {npz_path} "
                       f"(has: {sorted(k[4:] for k in d.files if k.startswith('ell_'))})")
    return TransferFunction(d[key_e], d[key_t], ell_zero=float(d["ell_zero"]))
