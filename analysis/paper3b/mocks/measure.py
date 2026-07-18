"""WP-B4 matched mock measurement: atlas patch -> the frozen B2 observable.

Produced by Project B WP-B4 (forward model; see bind-paper3-plans, private
repo). One patch = one (realization, noise seed) of one atlas run. The chain,
in frozen order:

1. κ_eff = Σ_k w_k κ_k  — amplitude-corrected DES n(z) source-plane weights
   (`nz.SourcePlaneWeighting`, B4 default `from_nz_amplitude_corrected`).
2. + shape noise BEFORE smoothing (Paper-2 order, `stats.peak_counts`
   convention; per-patch seeded, σ_pix = σ_e/√(n_eff·A_pix), no √2 — verified
   identical to `bind.inference.stats._noise_sigma_pix`).
3. Gaussian smoothing, PEAK_DEFINITION §1 (σ = 2', mode="wrap").
4. ν = (κ_sm − mean)/std of the patch itself — `nu_norm="map"`, the SAME
   normalization the B1 data chain uses (PEAK_DEFINITION §3). This closes
   §3's re-expression flag: B4 does NOT use the atlas' "fixed" heritage.
5. Peaks: strict 8-neighbour maxima (PEAK_DEFINITION §2, periodic).
6. Mock y beamed to ACT resolution (`beam.apply_beam`).
7. THE IMPORTED B2 OPERATORS on a CAR enmap of the beamed y patch:
   `maps.stack.extract_thumbnails` (15', 0.5') then `stack.cap.
   cap_filter_multi` at the frozen radius set — the very same function
   objects the frozen data measurement called (import-verified in tests).

Errors: a single 5° patch spans <3 nside-8 jackknife patches, so the spatial
jackknife CANNOT apply; the model grid's uncertainty is ensemble Monte-Carlo
over (realization, seed) patches instead. That is an error-estimation choice —
every *measurement* operator and constant is shared with the frozen chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The frozen B2 measurement operators and constants — imported, never copied.
from ..maps.stack import extract_thumbnails
from ..stack.cap import cap_filter_multi
from ..stack.stacker import (
    CAP_RADII_ARCMIN,
    FIDUCIAL_RADIUS_INDEX,
    NU_STACK_EDGES,
    THUMB_R_ARCMIN,
    THUMB_RES_ARCMIN,
)
from .beam import BeamConfig, apply_beam
from .nz import SourcePlaneWeighting
from .patch import DEFAULT_PAD_PIX, PatchGeometry, find_peaks_flat, patch_to_enmap, peak_sky_coords
from .shape_noise import ShapeNoiseConfig, noise_map
from .smoothing import SmoothingConfig, smooth_flat_sky
from .transfer import TransferFunction

__all__ = [
    "CAP_RADII_ARCMIN", "FIDUCIAL_RADIUS_INDEX", "NU_STACK_EDGES",
    "MockPatchResult", "MockEnsembleAccumulator", "measure_mock_patch",
]


@dataclass
class MockPatchResult:
    """One patch's per-peak table + diagnostics."""

    per_peak_nu: np.ndarray          # (npk,)
    per_peak_y: np.ndarray           # (npk, nrad) CAP Y [y arcmin^2]
    sigma_kappa_sm: float            # the "map" nu normalization actually used
    n_peaks_all: int                 # peaks before the nu-range cut

    def bin_table(self, nu_edges: np.ndarray = NU_STACK_EDGES
                  ) -> tuple[np.ndarray, np.ndarray]:
        """(counts (nbin,), y_sums (nbin, nrad)) over the stacking-nu bins."""
        nbin = len(nu_edges) - 1
        nrad = self.per_peak_y.shape[1] if self.per_peak_y.size else len(CAP_RADII_ARCMIN)
        counts = np.zeros(nbin, dtype=np.int64)
        y_sums = np.zeros((nbin, nrad))
        if self.per_peak_nu.size:
            b = np.digitize(self.per_peak_nu, nu_edges) - 1
            for bb in range(nbin):
                m = b == bb
                counts[bb] = int(m.sum())
                if counts[bb]:
                    y_sums[bb] = self.per_peak_y[m].sum(axis=0)
        return counts, y_sums


def measure_mock_patch(kappa_planes: np.ndarray, y_total: np.ndarray,
                       weighting: SourcePlaneWeighting,
                       noise_cfg: ShapeNoiseConfig,
                       smoothing_cfg: SmoothingConfig,
                       beam_cfg: BeamConfig,
                       rng: np.random.Generator,
                       geom: PatchGeometry = PatchGeometry(),
                       nu_edges: np.ndarray = NU_STACK_EDGES,
                       cap_radii=CAP_RADII_ARCMIN,
                       pad_pix: int = DEFAULT_PAD_PIX,
                       transfer: TransferFunction | None = None,
                       quantize_arcmin: float | None = None) -> MockPatchResult:
    """Run the full frozen chain (module docstring) on one atlas patch.

    kappa_planes: (K, npix, npix) single realization, all source planes.
    y_total: (npix, npix) cumulative Compton-y to the deepest plane (the
    matched convention for the full-column ACT y map).
    transfer: optional reconstruction transfer T(ell) (B5 design decision) —
    applied to the NOISY kappa before smoothing, matching where the data's
    Wiener/GLIMPSE reconstruction acts (on noisy shear, before B1's
    smoothing). None = the unfiltered "intrinsic" convention.
    """
    kappa_eff = weighting.effective_map(kappa_planes, plane_axis=0)
    kappa_n = kappa_eff + noise_map(kappa_eff.shape, noise_cfg, rng=rng)
    if transfer is not None:
        kappa_n = transfer.apply(kappa_n, geom.fov_deg)
    ksm = smooth_flat_sky(kappa_n, smoothing_cfg)
    sigma = float(ksm.std()) + 1e-30                    # nu_norm="map"
    nu_map = (ksm - ksm.mean()) / sigma

    pi, pj = find_peaks_flat(ksm)
    pnu = nu_map[pi, pj]
    n_all = len(pnu)
    sel = (pnu >= nu_edges[0]) & (pnu < nu_edges[-1])
    pi, pj, pnu = pi[sel], pj[sel], pnu[sel]

    y_beamed = apply_beam(np.asarray(y_total, dtype=np.float64), beam_cfg)
    if len(pnu) == 0:
        return MockPatchResult(pnu, np.zeros((0, len(cap_radii))), sigma, n_all)
    emap = patch_to_enmap(y_beamed, geom, pad_pix)
    ra, dec = peak_sky_coords(pi, pj, emap, pad_pix,
                              quantize_arcmin=quantize_arcmin)
    thumbs = np.asarray(extract_thumbnails(emap, ra, dec, THUMB_R_ARCMIN,
                                           THUMB_RES_ARCMIN), dtype=np.float64)
    per_peak_y = cap_filter_multi(thumbs, cap_radii, THUMB_RES_ARCMIN)
    return MockPatchResult(pnu, per_peak_y, sigma, n_all)


@dataclass
class MockEnsembleAccumulator:
    """Per-θ aggregation over (realization, seed) patches.

    Keeps the per-patch (counts, y_sums) tables so the assembly step can form
    both the pooled peak-weighted mean (the data-estimator match) and the
    between-patch Monte-Carlo error, plus per-bin abundances per patch area.

    With ``keep_peaks`` (default True since the B5 filter decision) the raw
    per-peak (nu, CAP-Y) tables are also kept, concatenated across patches —
    ~40 MB/unit — so ANY later re-binning / re-thresholding (quantile
    diagnostics, finer nu bins) is possible without re-running the grid.
    """

    nu_edges: np.ndarray = field(default_factory=lambda: np.asarray(NU_STACK_EDGES, dtype=float))
    nrad: int = len(CAP_RADII_ARCMIN)
    keep_peaks: bool = True
    counts: list = field(default_factory=list)     # each (nbin,)
    y_sums: list = field(default_factory=list)     # each (nbin, nrad)
    sigma_kappa: list = field(default_factory=list)
    n_peaks_all: list = field(default_factory=list)
    peak_nu: list = field(default_factory=list)    # each (npk,)
    peak_y: list = field(default_factory=list)     # each (npk, nrad)
    peak_patch: list = field(default_factory=list)  # each (npk,) patch index

    def add(self, res: MockPatchResult) -> None:
        c, s = res.bin_table(self.nu_edges)
        self.counts.append(c)
        self.y_sums.append(s)
        self.sigma_kappa.append(res.sigma_kappa_sm)
        self.n_peaks_all.append(res.n_peaks_all)
        if self.keep_peaks:
            ipatch = len(self.counts) - 1
            self.peak_nu.append(np.asarray(res.per_peak_nu, dtype=np.float64))
            self.peak_y.append(np.asarray(res.per_peak_y, dtype=np.float64))
            self.peak_patch.append(np.full(len(res.per_peak_nu), ipatch, dtype=np.int64))

    def arrays(self) -> dict:
        """Stacked per-patch arrays for the per-θ npz bundle."""
        out = {
            "nu_edges": self.nu_edges,
            "cap_radii_arcmin": np.asarray(CAP_RADII_ARCMIN, dtype=float),
            "counts": np.asarray(self.counts, dtype=np.int64),          # (npatch, nbin)
            "y_sums": np.asarray(self.y_sums, dtype=np.float64),        # (npatch, nbin, nrad)
            "sigma_kappa": np.asarray(self.sigma_kappa, dtype=np.float64),
            "n_peaks_all": np.asarray(self.n_peaks_all, dtype=np.int64),
        }
        if self.keep_peaks:
            nrad = self.nrad if not self.peak_y else self.peak_y[0].shape[1]
            out["peak_nu"] = (np.concatenate(self.peak_nu) if self.peak_nu
                              else np.zeros(0))
            out["peak_y"] = (np.concatenate(self.peak_y) if self.peak_y
                             else np.zeros((0, nrad)))
            out["peak_patch"] = (np.concatenate(self.peak_patch) if self.peak_patch
                                 else np.zeros(0, dtype=np.int64))
        return out

    def summary(self) -> dict:
        """Pooled mean + between-patch MC error of the mean, per (bin, radius).

        Pooled mean = Σ y_sums / Σ counts (peak-weighted — matches the data
        estimator, a mean over all peaks in the footprint). MC error =
        weighted between-patch scatter of patch means / √n_eff_patches.
        """
        counts = np.asarray(self.counts, dtype=np.float64)              # (P, nbin)
        y_sums = np.asarray(self.y_sums, dtype=np.float64)              # (P, nbin, nrad)
        tot = counts.sum(axis=0)                                        # (nbin,)
        pooled = y_sums.sum(axis=0) / np.where(tot > 0, tot, np.nan)[:, None]
        # patch means where occupied
        with np.errstate(invalid="ignore", divide="ignore"):
            pmeans = y_sums / counts[:, :, None]
        mc_err = np.full_like(pooled, np.nan)
        for b in range(counts.shape[1]):
            occ = counts[:, b] > 0
            if occ.sum() < 2:
                continue
            w = counts[occ, b]
            mu = pooled[b]
            var = np.average((pmeans[occ, b] - mu) ** 2, weights=w, axis=0)
            neff = w.sum() ** 2 / (w ** 2).sum()
            mc_err[b] = np.sqrt(var / neff)
        return {"y_mean": pooled, "y_mc_err": mc_err, "n_per_bin": tot,
                "n_patches": counts.shape[0]}
