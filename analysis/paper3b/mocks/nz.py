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

TODOs (flag in B/wp4 REPORT.md)
-------------------------------
1. Load the real DES Y3 n(z) tabulation (the SOMPZ/HYPERRANK n(z) realizations
   released with arXiv:2012.08566) rather than the analytic Smail fallback.
   ``source_plane_weights`` already accepts a tabulated ``(z, n)`` array, so
   this is a data-loading step, not a code change.
2. The atlas' lowest source plane is ``z_s = 0.5``, but DES bins 1-2 have n(z)
   support well below 0.5. Binning assigns that low-z mass to the z_s=0.5 plane
   (open lower edge), which is an *approximation*: those bins really need lower
   source planes (a new generation round) or an explicitly accepted
   extrapolation. Do not treat DES bin-1/2 effective κ as validated until this
   is resolved.
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
        if not np.isclose(w.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got {w.sum():.6g}")
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
