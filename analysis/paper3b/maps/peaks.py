"""WP-B1 task 4 — curved-sky peak finder implementing PEAK_DEFINITION.md verbatim.

The frozen chain (B4 must reproduce it exactly):

1. **Smoothing**: apply the harmonized common mask (zero outside), then
   Gaussian-smooth in harmonic space — ``healpy.smoothing(map, sigma=σ_rad)``
   (healpy's ``sigma`` IS the Gaussian σ, matching the flat-sky σ directly).
   Fiducial σ = 2 arcmin; robustness set {1, 2, 5, 8}.
2. **Peaks**: a pixel of the smoothed map is a peak iff strictly greater than
   ALL of its HEALPix neighbours (``get_all_neighbours``, RING; up to 8).
3. **ν**: ν = (κ_sm − mean)/σ with mean/σ of the smoothed map over the VALID
   (common binary) footprint (``nu_norm="map"``). Bins linspace(-5, 12, 69).
4. **Mask-proximity exclusion** (new for B1): drop peaks within r_ex = 2σ_smooth
   of the nearest pixel OUTSIDE the common binary footprint (query_disc,
   inclusive). Counts removed are recorded per catalog.

Pure healpy/numpy; testable on synthetic low-Nside maps (no survey files).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIDUCIAL_SMOOTHING_ARCMIN = 2.0
SMOOTHING_SET_ARCMIN = (1.0, 2.0, 5.0, 8.0)
NU_BIN_EDGES = np.linspace(-5.0, 12.0, 69)
EXCLUSION_SIGMA_FACTOR = 2.0
ARCMIN = np.pi / (180.0 * 60.0)


def smooth_masked(m: np.ndarray, mask: np.ndarray, sigma_arcmin: float) -> np.ndarray:
    """Zero-fill outside ``mask`` then harmonic-space Gaussian smooth (σ in arcmin).

    PEAK_DEFINITION §1: mask BEFORE smoothing; no edge renormalization (the
    edge bias is handled by the exclusion cut, §4). use_pixel_weights improves
    the SHT quadrature at Nside=1024 (falls back silently at toy Nside in tests).
    """
    import healpy as hp

    zeroed = np.where(np.asarray(mask, dtype=bool), np.asarray(m, dtype=np.float64), 0.0)
    try:
        return hp.smoothing(zeroed, sigma=float(sigma_arcmin) * ARCMIN, use_pixel_weights=True)
    except (ValueError, OSError):  # pixel weights unavailable for this Nside
        return hp.smoothing(zeroed, sigma=float(sigma_arcmin) * ARCMIN)


def local_maxima(smoothed: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pixel indices of strict local maxima of ``smoothed`` within ``valid``.

    Strictly greater than every existing neighbour (missing neighbours — the
    handful of -1 entries get_all_neighbours returns — are ignored, matching
    "all of its neighbours"). Neighbour values are compared regardless of the
    neighbour's own validity, exactly like the periodic flat-sky code compares
    against all 8 surrounding pixels.
    """
    import healpy as hp

    smoothed = np.asarray(smoothed, dtype=np.float64)
    nside = hp.npix2nside(smoothed.size)
    cand = np.flatnonzero(np.asarray(valid, dtype=bool))
    neigh = hp.get_all_neighbours(nside, cand)          # (8, n_cand), -1 = missing
    nv = np.where(neigh >= 0, smoothed[np.clip(neigh, 0, None)], -np.inf)
    return cand[(smoothed[cand][None, :] > nv).all(axis=0)]


def map_nu_stats(smoothed: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """(mean, std) of the smoothed map over the valid footprint (nu_norm="map")."""
    vals = np.asarray(smoothed, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    if vals.size == 0:
        raise ValueError("empty valid footprint")
    return float(vals.mean()), float(vals.std())


def exclusion_keep(peak_pix: np.ndarray, binary_mask: np.ndarray,
                   sigma_arcmin: float,
                   factor: float = EXCLUSION_SIGMA_FACTOR) -> np.ndarray:
    """Bool keep-mask: peak farther than ``factor * σ`` from any masked pixel.

    Exact per-peak test (PEAK_DEFINITION §4): query_disc(inclusive) of radius
    r_ex around the peak must contain no pixel outside the binary footprint.
    Cost is per-PEAK (thousands), not per-pixel — cheap.
    """
    import healpy as hp

    binary_mask = np.asarray(binary_mask, dtype=bool)
    nside = hp.npix2nside(binary_mask.size)
    r_ex = factor * float(sigma_arcmin) * ARCMIN
    keep = np.ones(len(peak_pix), dtype=bool)
    for i, p in enumerate(np.asarray(peak_pix)):
        disc = hp.query_disc(nside, hp.pix2vec(nside, int(p)), r_ex, inclusive=True)
        keep[i] = binary_mask[disc].all()
    return keep


@dataclass
class PeakCatalog:
    """One variant x one smoothing scale, schema shared across variants (§5)."""

    variant: str
    sigma_arcmin: float
    ipix: np.ndarray        # HEALPix RING indices (post-exclusion)
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    kappa_sm: np.ndarray    # smoothed-map value at the peak
    nu: np.ndarray
    map_mean: float         # nu_norm="map" stats over the valid footprint
    map_sigma: float
    n_raw: int              # peaks before the exclusion cut
    n_excluded: int
    nside: int

    def nu_histogram(self, edges: np.ndarray = NU_BIN_EDGES) -> np.ndarray:
        return np.histogram(self.nu, bins=edges)[0]

    def to_npz_dict(self) -> dict:
        return {
            "variant": self.variant, "sigma_arcmin": self.sigma_arcmin,
            "ipix": self.ipix, "ra_deg": self.ra_deg, "dec_deg": self.dec_deg,
            "kappa_sm": self.kappa_sm, "nu": self.nu,
            "map_mean": self.map_mean, "map_sigma": self.map_sigma,
            "n_raw": self.n_raw, "n_excluded": self.n_excluded,
            "nside": self.nside, "nu_bin_edges": NU_BIN_EDGES,
            "nu_counts": self.nu_histogram(),
        }


def build_peak_catalog(m: np.ndarray, weight_mask: np.ndarray, binary_mask: np.ndarray,
                       sigma_arcmin: float, variant: str) -> PeakCatalog:
    """Run the full frozen chain on one map + harmonized masks -> catalog.

    Smoothing uses the (apodized) weight mask's support zero-fill via the
    binary of weight>0 — i.e. the map is masked to the DES x ACT support before
    the SHT; peak finding and ν stats use the conservative binary footprint;
    the exclusion cut then trims the edge zone (counts recorded).
    """
    import healpy as hp

    sm = smooth_masked(m, np.asarray(weight_mask) > 0, sigma_arcmin)
    mean, sigma = map_nu_stats(sm, binary_mask)
    pk = local_maxima(sm, binary_mask)
    keep = exclusion_keep(pk, binary_mask, sigma_arcmin)
    kept = pk[keep]
    nside = hp.npix2nside(np.asarray(m).size)
    theta, phi = hp.pix2ang(nside, kept)
    return PeakCatalog(
        variant=variant, sigma_arcmin=float(sigma_arcmin), ipix=kept,
        ra_deg=np.degrees(phi), dec_deg=90.0 - np.degrees(theta),
        kappa_sm=np.asarray(sm)[kept], nu=(np.asarray(sm)[kept] - mean) / sigma,
        map_mean=mean, map_sigma=sigma,
        n_raw=int(len(pk)), n_excluded=int((~keep).sum()), nside=nside,
    )
