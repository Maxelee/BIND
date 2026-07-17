"""DES Y3 source redshift distribution n(z) source-plane weighting (WP-B4).

Produced by Project B WP-B4 (forward model; see bind-paper3-plans, private
repo). Pure numpy — no bind/torch/healpy import — so the survey-realism layer
runs identically wherever the raytraced maps live (Popeye) and inside any
mock-measurement loop without dragging in the painting stack.

Published spec
--------------
DES Y3 weak-lensing source redshift distribution, 4 tomographic bins:
**Myles, Alarcon et al. 2021, "DES Y3 Results: Redshift Calibration of the
Weak Lensing Source Galaxies", arXiv:2012.08566** (SOMPZ + clustering + shear-
ratio calibration). The four tomographic bins have approximate mean redshifts
``z ~ 0.34, 0.52, 0.74, 0.90`` (bins 1-4) — recorded here as reference only in
``DESY3_BIN_MEAN_Z``; the *effective* κ requires the full n(z) shape, not the
mean, so the real tabulated n(z) must be loaded (see TODO).

Convention
----------
The raytraced atlas stores κ (and y) on a *discrete* set of source planes
``z_s`` (twobound atlas: ``{0.5, 1.0, 1.5, 2.0, 2.44}``). Convergence is linear
in the lensing efficiency kernel, so the map seen by a source population with
normalized distribution ``n(z_s)`` is the n(z)-weighted combination of the
single-source-plane maps::

    κ_eff = ∫ n(z_s) κ(z_s) dz_s   ->   κ_eff = Σ_k w_k κ_k ,   Σ_k w_k = 1.

With κ available only at the discrete planes we *bin* n(z) onto the planes:
plane ``k`` receives the n(z) mass in the redshift interval bounded by the
midpoints to its two neighbours (outer edges open to ±∞), so the weights are
``w_k = CDF(edge_{k+1}) - CDF(edge_k)`` normalized to sum to 1. A source
distribution concentrated entirely within one plane's interval returns that
plane's map exactly; a distribution split across planes returns the correctly
weighted average.

Resolution of the two session-2 TODOs (2026-07-17, B/wp4 session)
-----------------------------------------------------------------
1. Real n(z): ``combined_desy3_source_nz`` loads the fiducial SOMPZ source
   n(z) (``nz_source`` HDU of the Y3 2pt data vector,
   ``2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits``) and forms
   the *combined-map* distribution as the n_eff-weighted sum of the four bins
   (the B1 frozen measurement stacks on the ``*_full`` Jeffrey maps, i.e. the
   all-bins source sample). The Smail form below remains a test fixture only.
2. Sub-0.5 support: 41% of the combined n(z) lies below the atlas' lowest
   source plane (z_s = 0.5). *Plain* midpoint binning maps that mass onto the
   z_s = 0.5 plane at full amplitude, overestimating the κ_eff amplitude by
   ~20% — quantified and REJECTED as the B4 default. Instead,
   ``amplitude_corrected_weights`` scales each source redshift's contribution
   by ``A(z)/A(z_k)`` where ``A(z)`` is the κ rms amplitude vs source
   redshift, modelled as a non-negative lens-shell decomposition
   ``A²(z_s) = Σ_j q_j W²(z_j, z_s)`` (W = flat-ΛCDM lensing efficiency,
   TNG/atlas Ω_m) with ``q_j`` fitted by NNLS to the atlas' *own* empirical
   5×5 cross-plane κ covariance (``ShellAmplitudeModel.fit_plane_cov``; ≤1%
   fit residual on all 15 entries with the default 12-shell grid). The
   resulting weights sum to ≈0.80, not 1 — sub-0.5 sources genuinely carry
   less κ. Shell-grid sensitivity ≈2% on the summed amplitude — recorded as a
   B4 model systematic (REPORT.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# DES Y3 tomographic-bin mean redshifts (approximate), Myles et al. 2021,
# arXiv:2012.08566. Reference values only — the effective κ needs the full
# n(z), not the mean. Never used as a weighting input silently.
DESY3_BIN_MEAN_Z: tuple[float, float, float, float] = (0.34, 0.52, 0.74, 0.90)


def smail_nz(z, alpha: float = 2.0, beta: float = 1.5, z0: float = 0.5) -> np.ndarray:
    """Analytic Smail-form n(z) ∝ z**alpha * exp(-(z/z0)**beta) (fallback only).

    A stand-in for the real DES Y3 n(z) file (arXiv:2012.08566). The returned
    curve is *unnormalized*; ``source_plane_weights`` normalizes internally.
    Callers must pass the real tabulated n(z) once it is available — this
    fallback is documented, never silently substituted for the survey value.
    """
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    pos = z > 0
    out[pos] = z[pos] ** alpha * np.exp(-((z[pos] / z0) ** beta))
    return out


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of y over x, with a leading 0.0 (pure numpy)."""
    dx = np.diff(x)
    seg = 0.5 * (y[1:] + y[:-1]) * dx
    return np.concatenate([[0.0], np.cumsum(seg)])


def source_plane_weights(source_z, nz_z, nz) -> np.ndarray:
    """Bin a tabulated n(z) onto discrete source planes -> normalized weights.

    Parameters
    ----------
    source_z : (K,) array
        Discrete source-plane redshifts (need not be pre-sorted).
    nz_z, nz : (M,) arrays
        Tabulated redshift distribution: nz sampled at nz_z. Need not be
        normalized; may be unsorted (sorted internally).

    Returns
    -------
    (K,) weights, aligned to the *input* order of ``source_z``, summing to 1.

    The plane intervals are the midpoints between adjacent (sorted) planes,
    with the outermost edges opened to ±∞ so all n(z) mass is captured (mass
    below the lowest / above the highest plane is assigned to it — see module
    TODO 2 for the DES bin-1/2 caveat).
    """
    source_z = np.asarray(source_z, dtype=float)
    nz_z = np.asarray(nz_z, dtype=float)
    nz = np.asarray(nz, dtype=float)
    if source_z.ndim != 1 or source_z.size == 0:
        raise ValueError("source_z must be a non-empty 1D array")
    if nz_z.shape != nz.shape or nz_z.ndim != 1:
        raise ValueError("nz_z and nz must be 1D arrays of equal length")

    # Sort the tabulated n(z) and the planes; remember how to unsort the planes.
    tab_order = np.argsort(nz_z)
    nz_z, nz = nz_z[tab_order], nz[tab_order]
    if np.any(nz < 0):
        raise ValueError("n(z) must be non-negative")

    order = np.argsort(source_z)
    zs = source_z[order]
    k = zs.size

    edges = np.empty(k + 1)
    edges[1:-1] = 0.5 * (zs[:-1] + zs[1:])
    edges[0] = -np.inf
    edges[-1] = np.inf

    cdf = _cumtrapz(nz, nz_z)
    total = cdf[-1]
    if total <= 0:
        raise ValueError("n(z) integrates to zero over the tabulated range")

    def cdf_at(e: float) -> float:
        e = np.clip(e, nz_z[0], nz_z[-1])
        return float(np.interp(e, nz_z, cdf))

    w_sorted = np.array([cdf_at(edges[i + 1]) - cdf_at(edges[i]) for i in range(k)])
    w_sorted /= w_sorted.sum() if w_sorted.sum() > 0 else 1.0

    # Restore input plane order.
    w = np.empty(k)
    w[order] = w_sorted
    return w


@dataclass(frozen=True)
class SourcePlaneWeighting:
    """A fixed set of source-plane weights and the machine to apply them.

    Attributes
    ----------
    source_z : (K,) source-plane redshifts, in the same order as the plane axis
        of the maps this weighting will be applied to.
    weights : (K,) non-negative weights summing to 1.
    """

    source_z: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        sz = np.asarray(self.source_z, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        if sz.shape != w.shape or sz.ndim != 1:
            raise ValueError("source_z and weights must be 1D arrays of equal length")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        # Plain binned weights sum to 1; amplitude-corrected weights sum to
        # <= 1 (sub-lowest-plane sources carry less kappa). A sum above 1
        # would mean amplifying the maps and is always a bug.
        if not w.sum() <= 1.0 + 1e-9:
            raise ValueError(f"weights must sum to <= 1, got {w.sum():.6g}")
        if w.sum() <= 0:
            raise ValueError("weights sum to zero")
        object.__setattr__(self, "source_z", sz)
        object.__setattr__(self, "weights", w)

    @classmethod
    def from_nz(cls, source_z, nz_z, nz) -> "SourcePlaneWeighting":
        """Build weights by binning a tabulated n(z) onto ``source_z``."""
        source_z = np.asarray(source_z, dtype=float)
        return cls(source_z, source_plane_weights(source_z, nz_z, nz))

    @classmethod
    def from_delta(cls, source_z, z_source: float) -> "SourcePlaneWeighting":
        """Delta-function n(z): all weight on the plane nearest ``z_source``.

        Raises if ``z_source`` is not (within a tiny tolerance) one of the
        source planes — a delta between planes has no exact discrete
        representation and silently snapping it would hide a selection error.
        """
        source_z = np.asarray(source_z, dtype=float)
        j = int(np.argmin(np.abs(source_z - z_source)))
        if not np.isclose(source_z[j], z_source, atol=1e-9):
            raise ValueError(
                f"z_source={z_source} is not a source plane {source_z.tolist()}; "
                "a delta between planes cannot be represented exactly"
            )
        w = np.zeros_like(source_z)
        w[j] = 1.0
        return cls(source_z, w)

    @classmethod
    def from_nz_amplitude_corrected(cls, source_z, nz_z, nz,
                                    model: "ShellAmplitudeModel"
                                    ) -> "SourcePlaneWeighting":
        """Amplitude-corrected binning (the B4 default — module docstring §2).

        Each tabulated source redshift contributes ``n(z)·A(z)/A(z_k)`` to its
        plane's weight, so ``Σ w_k κ_k`` reproduces the n(z)-averaged κ
        *amplitude* ``∫ n(z) A(z) dz / ∫ n(z) dz`` exactly under the fitted
        amplitude model. Weights sum to <= 1 by construction.
        """
        source_z = np.asarray(source_z, dtype=float)
        w = amplitude_corrected_weights(source_z, nz_z, nz, model)
        return cls(source_z, w)

    def effective_map(self, plane_maps, plane_axis: int = 0) -> np.ndarray:
        """Weighted sum of per-source-plane maps along ``plane_axis``.

        ``plane_maps`` has some axis of length K holding the source planes
        (for the twobound atlas ``kappa`` array of shape (n_real, K, Ny, Nx)
        pass ``plane_axis=1``). Returns the array with that axis removed.
        """
        m = np.asarray(plane_maps, dtype=float)
        if m.shape[plane_axis] != self.weights.size:
            raise ValueError(
                f"plane_maps axis {plane_axis} has length {m.shape[plane_axis]}, "
                f"expected {self.weights.size} to match the weights"
            )
        moved = np.moveaxis(m, plane_axis, 0)
        return np.tensordot(self.weights, moved, axes=([0], [0]))


# ── DES Y3 combined n(z) + the amplitude-corrected plane weighting ───────────

# TNG300 / atlas cosmology (params.npy dim 0 of every twobound run).
ATLAS_OMEGA_M: float = 0.3089


def combined_desy3_source_nz(fits_path, neff_perbin=None) -> tuple[np.ndarray, np.ndarray]:
    """(z, n) of the combined DES Y3 source sample from the 2pt data vector.

    Reads the fiducial SOMPZ ``nz_source`` HDU (4 tomographic bins), normalizes
    each bin to unit integral, and combines with per-bin ``n_eff`` weights
    (default: the Gatti et al. 2021 values recorded in ``shape_noise.py``) —
    the source population of the Jeffrey ``*_full`` mass maps that the B1
    frozen measurement stacks on. Returns an *unnormalized* (z, n) pair
    suitable for ``source_plane_weights`` / ``amplitude_corrected_weights``.
    """
    from astropy.io import fits as _fits

    if neff_perbin is None:
        from .shape_noise import DESY3_NEFF_PERBIN_ARCMIN2
        neff_perbin = DESY3_NEFF_PERBIN_ARCMIN2
    neff = np.asarray(neff_perbin, dtype=float)
    with _fits.open(fits_path) as f:
        d = f["nz_source"].data
        z = np.asarray(d["Z_MID"], dtype=float)
        combined = np.zeros_like(z)
        for i, ne in enumerate(neff, start=1):
            b = np.asarray(d[f"BIN{i}"], dtype=float)
            combined += ne * b / np.trapezoid(b, z)
    return z, combined / neff.sum()


def comoving_distance_flat(z, omega_m: float = ATLAS_OMEGA_M) -> np.ndarray:
    """Flat-ΛCDM comoving distance in c/H0 units (only ratios are ever used)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    zi = np.linspace(0.0, max(3.0, float(z.max()) + 0.1), 4001)
    integrand = 1.0 / np.sqrt(omega_m * (1.0 + zi) ** 3 + (1.0 - omega_m))
    chi = _cumtrapz(integrand, zi)
    return np.interp(z, zi, chi)


def lensing_efficiency(z_lens: float, z_source, omega_m: float = ATLAS_OMEGA_M) -> np.ndarray:
    """W(z_l, z_s) = (χ_s − χ_l)/χ_s for χ_s > χ_l, else 0 (flat ΛCDM)."""
    zs = np.atleast_1d(np.asarray(z_source, dtype=float))
    chi_s = comoving_distance_flat(zs, omega_m)
    chi_l = comoving_distance_flat(z_lens, omega_m)[0]
    return np.where(chi_s > chi_l, (chi_s - chi_l) / np.maximum(chi_s, 1e-30), 0.0)


@dataclass(frozen=True)
class ShellAmplitudeModel:
    """κ rms amplitude vs source redshift, A²(z_s) = Σ_j q_j W²(z_j, z_s).

    Fitted (NNLS, non-negative q) to the atlas' empirical cross-plane κ
    covariance — 15 constraints from the 5×5 plane covariance, which pins the
    low-z lens-shell content far better than the 5 auto-amplitudes alone
    (module docstring §2). ``max_fit_frac_err`` records the worst relative
    residual over the fitted covariance entries; reject fits above ~2%.
    """

    shells: np.ndarray            # (J,) lens-shell redshifts
    q: np.ndarray                 # (J,) non-negative shell powers
    omega_m: float
    max_fit_frac_err: float

    DEFAULT_SHELLS = tuple(np.linspace(0.05, 1.6, 12))

    @classmethod
    def fit_plane_cov(cls, plane_cov, source_z, shells=None,
                      omega_m: float = ATLAS_OMEGA_M) -> "ShellAmplitudeModel":
        """Fit q >= 0 to the empirical (K, K) cross-plane κ covariance."""
        from scipy.optimize import nnls

        C = np.asarray(plane_cov, dtype=float)
        zs = np.asarray(source_z, dtype=float)
        sh = np.asarray(shells if shells is not None else cls.DEFAULT_SHELLS, dtype=float)
        iu = np.triu_indices(len(zs))
        Wm = np.array([lensing_efficiency(zj, zs, omega_m) for zj in sh]).T  # (K, J)
        M = Wm[iu[0]] * Wm[iu[1]]                                            # (npair, J)
        q, _ = nnls(M, C[iu])
        pred = M @ q
        err = float(np.abs(pred / C[iu] - 1.0).max())
        return cls(shells=sh, q=q, omega_m=omega_m, max_fit_frac_err=err)

    @classmethod
    def fit_plane_maps(cls, kappa_maps, source_z, shells=None,
                       omega_m: float = ATLAS_OMEGA_M) -> "ShellAmplitudeModel":
        """Fit from raw (n_real, K, ny, nx) plane maps (computes the cov)."""
        k = np.asarray(kappa_maps, dtype=np.float64)
        n_real, K = k.shape[:2]
        X = k.reshape(n_real, K, -1)
        X = X - X.mean(axis=2, keepdims=True)
        C = np.einsum("rip,rjp->ij", X, X) / (n_real * X.shape[2])
        return cls.fit_plane_cov(C, source_z, shells, omega_m)

    def amplitude(self, z) -> np.ndarray:
        """A(z): rms κ amplitude for sources at z (arbitrary common units)."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        a2 = np.zeros_like(z)
        for qj, zj in zip(self.q, self.shells):
            a2 += qj * lensing_efficiency(zj, z, self.omega_m) ** 2
        return np.sqrt(a2)


def amplitude_corrected_weights(source_z, nz_z, nz,
                                model: ShellAmplitudeModel) -> np.ndarray:
    """Plane weights w_k = ∫_bin_k n(z) A(z)/A(z_k) dz / ∫ n(z) dz.

    The B4 default (module docstring §2): reproduces the n(z)-averaged κ
    amplitude under the fitted shell model instead of assigning sub-lowest-
    plane sources the full lowest-plane amplitude. Bin intervals are the same
    midpoint edges as ``source_plane_weights``; weights come back aligned to
    the input ``source_z`` order and sum to <= 1.
    """
    source_z = np.asarray(source_z, dtype=float)
    nz_z = np.asarray(nz_z, dtype=float)
    nz = np.asarray(nz, dtype=float)
    if np.any(nz < 0):
        raise ValueError("n(z) must be non-negative")
    tab_order = np.argsort(nz_z)
    nz_z, nz = nz_z[tab_order], nz[tab_order]
    order = np.argsort(source_z)
    zs_sorted = source_z[order]
    edges = np.concatenate([[-np.inf], 0.5 * (zs_sorted[:-1] + zs_sorted[1:]), [np.inf]])
    total = np.trapezoid(nz, nz_z)
    if total <= 0:
        raise ValueError("n(z) integrates to zero over the tabulated range")
    A_planes = model.amplitude(zs_sorted)
    A_tab = model.amplitude(nz_z)
    w_sorted = np.empty(len(zs_sorted))
    for k in range(len(zs_sorted)):
        m = (nz_z > edges[k]) & (nz_z <= edges[k + 1])
        if m.sum() < 2:
            w_sorted[k] = 0.0
            continue
        w_sorted[k] = np.trapezoid(nz[m] * A_tab[m], nz_z[m]) / (A_planes[k] * total)
    w = np.empty_like(w_sorted)
    w[order] = w_sorted
    return w
