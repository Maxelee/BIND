"""The eRASS1 f_gas(M500) likelihood block with mass-PDF convolution.

Data side (fixed at construction)
---------------------------------
- eRASS1 primary catalog (frozen, wp1), BEST_Z < ``z_max`` (the model is
  the snap-096 z=0.034 emulator; f_gas evolution is weak over z < 0.2 —
  documented analysis choice).
- Clusters binned by their *observed* mass (M500 point estimate, converted
  to Msun/h) into the gate bins; per-bin data value = median FGAS500,
  statistical error = 1.4826 MAD / sqrt(n) (the gate-overlay convention).
- Per bin, the stacked TRUE-mass distribution w_b(M) = mean of the members'
  normalized per-cluster M500 PDFs (``M500_PDF`` columns — the catalog's
  own X-ray-likelihood posteriors). This is where the low-mass
  point-estimate bias the freeze documented enters the model instead of
  being ignored: the model is evaluated at true mass and *convolved into
  the observed-bin frame*.

Model side (per theta)
----------------------
emulated cylindrical f_gas medians (5 gate bins, snap 096)
  -> per-theta CylToSph factors (CAMELS-1P trend x wp2 anchor)
  -> painted-bias correction (/1.074, the snap-096 wp2 measurement)
  -> piecewise-linear f_sph(log M) over bin centers, linearly extrapolated
     in log M below the model floor and held flat above the top bin
  -> pred_b = Int f_sph(M) w_b(M) dM   (mean over the bin's true-mass
     population; mean ~ median at these kernel widths — documented).

Covariance (diagonal, per bin)
------------------------------
data MAD^2  (+)  emulation floor (v4 k-fold rms of fgas_med, x CylToSph)
 (+)  paint-bias residual (2% of prediction; the 7.4-10.8% range's spread)
 (+)  CylToSph trend-transfer systematic (3% of prediction).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.paper3a.emulator.cyltosph_theta import CylToSphTheta
from analysis.paper3a.emulator.gasemu import GasEmulator
from analysis.paper3a.observables.constants import TNG_H

ERASS1_FITS = Path("/mnt/ceph/users/mlee1/paper3/A/wp1_data/egas_erass1_bulbul/"
                   "erass1cl_primary_v3.2.fits")
SNAP = "096"
PAINT_BIAS = 1.074            # wp2 snap-096: painted f_gas high by 7.4%
PAINT_BIAS_RESID = 0.02       # residual band after correcting (7.4-10.8% spread)
C2S_TRANSFER_SYS = 0.03       # L50->painted trend-transfer systematic
Z_MAX = 0.2
MIN_PER_BIN = 10


@dataclass
class FgasData:
    values: np.ndarray            # (B,) binned medians
    stat_err: np.ndarray          # (B,)
    n_per_bin: np.ndarray         # (B,)
    logm_grid: np.ndarray         # (G,) log10 Msun/h true-mass grid
    weights: np.ndarray           # (B, G) stacked true-mass PDFs (rows sum 1)
    bin_centers: np.ndarray       # (B,) log10 Msun/h observed-bin centers


def build_data(emu: GasEmulator, fits_path: Path = ERASS1_FITS,
               z_max: float = Z_MAX) -> FgasData:
    from astropy.io import fits as afits

    with afits.open(fits_path) as f:
        d = f[1].data
        m500_msunh = np.asarray(d["M500"], float) * 1e13 * TNG_H
        fgas = np.asarray(d["FGAS500"], float)
        z = np.asarray(d["BEST_Z"], float)
        pdf_grid_msun = np.asarray(d["M500_PDF_array"][0], float)
        pdfs = np.asarray(d["M500_PDF"], float)          # (N, G)

    ok = np.isfinite(m500_msunh) & (m500_msunh > 0) & np.isfinite(fgas) & (fgas > 0) \
        & np.isfinite(z) & (z < z_max)
    logm_obs = np.log10(m500_msunh[ok])
    fgas = fgas[ok]
    pdfs = pdfs[ok]

    logm_grid = np.log10(pdf_grid_msun * TNG_H)          # shared grid -> Msun/h
    edges = emu.logm500_bin_edges
    vals, errs, ns, wts, cens = [], [], [], [], []
    for b in range(len(edges) - 1):
        sel = (logm_obs >= edges[b]) & (logm_obs < edges[b + 1])
        if sel.sum() < MIN_PER_BIN:
            continue
        fg = fgas[sel]
        med = float(np.median(fg))
        vals.append(med)
        errs.append(1.4826 * float(np.median(np.abs(fg - med))) / np.sqrt(sel.sum()))
        ns.append(int(sel.sum()))
        w = pdfs[sel].mean(axis=0)
        w = np.maximum(w, 0.0)
        w /= w.sum()
        wts.append(w)
        cens.append(0.5 * (edges[b] + edges[b + 1]))
    return FgasData(values=np.array(vals), stat_err=np.array(errs),
                    n_per_bin=np.array(ns), logm_grid=logm_grid,
                    weights=np.array(wts), bin_centers=np.array(cens))


class FgasBlock:
    """Callable likelihood block: theta (unit cube, batched) -> per-bin
    predictions, diagonal sigma, and chi^2 against the frozen data."""

    def __init__(self, emu: GasEmulator | None = None,
                 c2s: CylToSphTheta | None = None,
                 kfold_path: Path | None = None,
                 fits_path: Path = ERASS1_FITS):
        from analysis.paper3a.emulator.sigma_theory import WP4

        self.emu = emu or GasEmulator.load()
        self.c2s = c2s or CylToSphTheta()
        self.data = build_data(self.emu, fits_path)
        edges = self.emu.logm500_bin_edges
        self.model_centers = 0.5 * (edges[1:] + edges[:-1])   # (5,)

        kf = np.load(kfold_path or (WP4 / "gasemu_kfold_v4.npz"), allow_pickle=False)
        zi = [str(s) for s in kf["snaps"]].index(SNAP)
        t = kf["truth"][:, zi, :5].astype(float)
        p = kf["pred"][:, zi, :5].astype(float)
        self.emul_frac = np.sqrt(np.nanmean(((p - t) / t) ** 2, axis=0))   # (5,)

        # data-bin rows of the convolution operator are fixed; only f values
        # change with theta, so precompute nothing else
        self._B = len(self.data.values)

    # ------------------------------------------------------------ forward ---

    def _fsph_bins(self, u: np.ndarray) -> np.ndarray:
        """(N, 5) spherical-equivalent, bias-corrected f_gas at the model
        bin centers (true mass)."""
        pred = self.emu.predict(u, SNAP, return_std=False)
        c2s = np.atleast_2d(self.c2s.factors_full_bins(u, SNAP))
        return np.atleast_2d(pred["fgas_med"]) * c2s / PAINT_BIAS

    def predict(self, u: np.ndarray) -> np.ndarray:
        """(N, B) predictions in the observed-bin frame (PDF-convolved).

        Only the first 30 columns (the SB35 unit cube) are used — wider
        chains carrying nuisance dimensions (e.g. the kSZ f_sat in column
        30) pass through unchanged."""
        u = np.atleast_2d(np.asarray(u, float))[:, :30]
        f5 = self._fsph_bins(u)                                # (N, 5)
        x = self.model_centers
        g = self.data.logm_grid
        # piecewise-linear in log M; linear extrapolation below, flat above
        out = np.empty((len(u), self._B))
        for i in range(len(u)):
            f = f5[i]
            fg = np.interp(g, x, f)
            lo = g < x[0]
            slope = (f[1] - f[0]) / (x[1] - x[0])
            fg[lo] = np.maximum(f[0] + slope * (g[lo] - x[0]), 0.0)
            out[i] = self.data.weights @ fg
        return out

    def sigma(self, pred: np.ndarray) -> np.ndarray:
        """(N, B) diagonal sigma: data stat + emulation floor + systematics."""
        emul = np.interp(self.data.bin_centers, self.model_centers, self.emul_frac)
        frac_model = np.sqrt(emul**2 + PAINT_BIAS_RESID**2 + C2S_TRANSFER_SYS**2)
        return np.sqrt(self.data.stat_err[None, :] ** 2
                       + (frac_model[None, :] * pred) ** 2)

    # --------------------------------------------------------- likelihood ---

    def loglike(self, u: np.ndarray) -> np.ndarray:
        """(N,) Gaussian log-likelihood (diagonal; documented choice)."""
        pred = self.predict(u)
        sig = self.sigma(pred)
        r = (pred - self.data.values[None, :]) / sig
        return -0.5 * np.sum(r**2 + np.log(2 * np.pi * sig**2), axis=1)

    def chi2(self, u: np.ndarray) -> np.ndarray:
        pred = self.predict(u)
        sig = self.sigma(pred)
        return np.sum(((pred - self.data.values[None, :]) / sig) ** 2, axis=1)
