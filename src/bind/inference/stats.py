"""Stage 3 — summary statistics on the lightcone kappa / y maps (1a + 1b).

Flat-sky estimators that turn the maps from :mod:`bind.inference.lightcone_maps`
into the emulator inputs:

* :func:`cl_kappa`        — WL auto-power ``C_ell`` (tomographic matrix), + the
  BIND/DMO suppression ``S(ell)`` when DMO maps are supplied.
* :func:`cl_kappa_y`      — WL x tSZ cross-power, plus the tSZ auto ``C_ell^yy``.
* :func:`peak_counts`     — convergence peak AND minimum counts vs S/N ``nu``
  (a pixel above / below all 8 neighbours).
* :func:`nongaussian_stats` — convergence PDF, variance/skewness/kurtosis per
  smoothing scale, and Minkowski functionals ``(V0, V1, V2)`` vs threshold.
* :func:`peak_cross_stats` — tSZ stacked at WL peaks: ``R(nu)=<y>/<kappa>``
  (a specific-thermal-energy proxy: ``R ∝ Y/M ∝ f_gas·T_mw`` — NOT a pure gas
  fraction), peak counts, the stacked radial y profile, and
  ``Cov(kappa_peak, y_peak)`` per S/N bin and source redshift.

``peak_counts`` / ``peak_cross_stats`` optionally add galaxy shape noise
(``shape_noise_ngal`` gal/arcmin^2, ``sigma_e`` per shear component) to kappa
*before* smoothing, and can define ``nu`` against the smoothed-noise rms
(``nu_norm="noise"``, the survey S/N convention) instead of the map std.
* :func:`halo_scaling`    — per-halo ``Y_500c, f_gas, f_star, T_mw`` from the
  composite per-halo patches.

Angular power spectra are computed with **Pylians** ``Pk_plane`` / ``XPk_plane``
using ``BoxSize = fov`` in radians, so the returned wavenumber ``k`` is the
multipole ``ell`` and the binning matches the rest of the group's pipelines.
"""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np

#: Threads handed to Pylians per spectrum.  Under MPI every rank should use 1 —
#: N ranks x 2 threads oversubscribes the node and costs more than it buys.
#: Override with ``BIND_PK_THREADS``.
PK_THREADS: int = int(os.environ.get("BIND_PK_THREADS", "2"))


# ── power spectra (Pylians) ───────────────────────────────────────────────────

def power_spectrum(
    m1: np.ndarray,
    m2: np.ndarray | None = None,
    *,
    fov_deg: float = 5.0,
    subtract_mean: bool = True,
    threads: int | None = None,
    auto1: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat-sky auto (``m2=None``) or cross angular power via Pylians.

    ``BoxSize = deg2rad(fov_deg)`` so the returned ``k`` is the multipole ``ell``.
    Maps are already gridded, so ``MAS='None'`` (no deconvolution).  Returns
    ``(ell, C_ell)``.

    A cross-spectrum needs ``m1``'s AUTO spectrum purely to fix Pylians'
    ``XPk_plane`` normalisation.  Callers that already have it (a tomographic
    matrix recomputes the same 5 autos 20 times over) can pass it as ``auto1``
    to skip that redundant ``Pk_plane`` call — the result is identical.
    """
    import Pk_library as PKL

    if threads is None:
        threads = PK_THREADS
    fov = np.deg2rad(fov_deg)
    a = (m1 - m1.mean() if subtract_mean else m1).astype(np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        if m2 is None:
            p = PKL.Pk_plane(a, fov, MAS="None", threads=threads, verbose=False)
            return np.asarray(p.k), np.asarray(p.Pk)
        b = (m2 - m2.mean() if subtract_mean else m2).astype(np.float32)
        x = PKL.XPk_plane(a, b, fov, MAS1="None", MAS2="None", threads=threads)
        if auto1 is None:
            pa = PKL.Pk_plane(a, fov, MAS="None", threads=threads, verbose=False)
            auto1 = np.asarray(pa.Pk)
    # Pylians XPk_plane normalises its power differently from Pk_plane (a constant
    # ~1.2e7 convention factor for these grids); rescale the cross to the Pk_plane
    # convention so cross and auto spectra are consistent (else C_ky is ~1e7 too low).
    norm = np.asarray(auto1) / np.asarray(x.Pk)[:, 0]
    return np.asarray(x.k), np.asarray(x.XPk) * norm


def cl_kappa(
    kappa_maps: np.ndarray,
    *,
    fov_deg: float = 5.0,
    kappa_dmo: np.ndarray | None = None,
) -> dict:
    """Tomographic WL auto/cross power averaged over realizations.

    ``kappa_maps`` is ``(n_real, n_src, npix, npix)``.  Returns ``ell``,
    ``cl`` ``(n_src, n_src, n_ell)`` (mean over realizations), ``cl_err`` (std/
    sqrt(n_real)), and — when ``kappa_dmo`` is given — the suppression
    ``S = cl_bind / cl_dmo`` (auto bins only).
    """
    kappa_maps = np.asarray(kappa_maps)
    n_real, n_src = kappa_maps.shape[:2]
    ell0 = None
    cube = []   # (n_real, n_src, n_src, n_ell)
    for r in range(n_real):
        # the matrix is symmetric and every cross reuses its row's auto spectrum
        # for the Pylians normalisation: 15 spectra, not 45.
        autos = []
        for i in range(n_src):
            ell0, cl = power_spectrum(kappa_maps[r, i], fov_deg=fov_deg)
            autos.append(cl)
        mat = [[None] * n_src for _ in range(n_src)]
        for i in range(n_src):
            mat[i][i] = autos[i]
            for j in range(i + 1, n_src):
                _, clx = power_spectrum(kappa_maps[r, i], kappa_maps[r, j],
                                        fov_deg=fov_deg, auto1=autos[i])
                mat[i][j] = mat[j][i] = clx
        cube.append(mat)
    cube = np.asarray(cube)
    out = {"ell": ell0, "cl": cube.mean(0),
           "cl_err": cube.std(0) / np.sqrt(max(n_real, 1))}
    if kappa_dmo is not None:
        kd = np.asarray(kappa_dmo)
        dmo = []
        for r in range(kd.shape[0]):
            dmo.append([power_spectrum(kd[r, i], fov_deg=fov_deg)[1]
                        for i in range(n_src)])
        dmo = np.asarray(dmo).mean(0)                       # (n_src, n_ell)
        auto = np.array([out["cl"][i, i] for i in range(n_src)])
        out["suppression"] = auto / np.where(dmo > 0, dmo, np.nan)
        out["cl_dmo"] = dmo
    return out


def cl_kappa_y(
    kappa_maps: np.ndarray,
    y_maps: np.ndarray,
    *,
    fov_deg: float = 5.0,
) -> dict:
    """WL x tSZ cross-power (per source bin) + tSZ auto ``C_ell^yy``.

    ``kappa_maps`` ``(n_real, n_src, npix, npix)``, ``y_maps`` ``(n_real, npix,
    npix)``.  Returns ``ell``, ``cl_ky`` ``(n_src, n_ell)``, ``cl_yy`` ``(n_ell,)``
    (means over realizations) and their errors.
    """
    kappa_maps = np.asarray(kappa_maps)
    y_maps = np.asarray(y_maps)
    n_real, n_src = kappa_maps.shape[:2]
    ky, yy, ell0 = [], [], None
    for r in range(n_real):
        ky.append([power_spectrum(kappa_maps[r, i], y_maps[r], fov_deg=fov_deg)[1]
                   for i in range(n_src)])
        ell0, cyy = power_spectrum(y_maps[r], fov_deg=fov_deg)
        yy.append(cyy)
    ky = np.asarray(ky)
    yy = np.asarray(yy)
    return {
        "ell": ell0,
        "cl_ky": ky.mean(0), "cl_ky_err": ky.std(0) / np.sqrt(max(n_real, 1)),
        "cl_yy": yy.mean(0), "cl_yy_err": yy.std(0) / np.sqrt(max(n_real, 1)),
    }


def cl_kappa_tau(
    kappa_maps: np.ndarray,
    tau_maps: np.ndarray,
    *,
    y_maps: np.ndarray | None = None,
    fov_deg: float = 5.0,
) -> dict:
    """WL x kSZ/FRB cross-power (per source bin) + tau auto ``C_ell^{tau tau}``.

    The density-weighted electron-column probe that, with ``kappa`` (total mass)
    and ``y`` (pressure = gas x T), completes the field-level ``(kappa, tau, y)``
    decomposition into ``(mass, f_gas, T)``.

    ``kappa_maps`` ``(n_real, n_src, npix, npix)``, ``tau_maps`` ``(n_real, npix,
    npix)`` (the total electron column to the last source plane).  ``y_maps``
    ``(n_real, npix, npix)`` optional (matching total Compton-y) adds the
    ``C_ell^{y tau}`` leg.  Returns ``ell``, ``cl_kt`` ``(n_src, n_ell)``,
    ``cl_tt`` ``(n_ell,)``, optionally ``cl_yt`` ``(n_ell,)`` (means over
    realizations) and their errors.
    """
    kappa_maps = np.asarray(kappa_maps)
    tau_maps = np.asarray(tau_maps)
    n_real, n_src = kappa_maps.shape[:2]
    kt, tt, yt, ell0 = [], [], [], None
    for r in range(n_real):
        kt.append([power_spectrum(kappa_maps[r, i], tau_maps[r], fov_deg=fov_deg)[1]
                   for i in range(n_src)])
        ell0, ctt = power_spectrum(tau_maps[r], fov_deg=fov_deg)
        tt.append(ctt)
        if y_maps is not None:
            yt.append(power_spectrum(np.asarray(y_maps)[r], tau_maps[r],
                                     fov_deg=fov_deg)[1])
    kt = np.asarray(kt)
    tt = np.asarray(tt)
    out = {
        "ell": ell0,
        "cl_kt": kt.mean(0), "cl_kt_err": kt.std(0) / np.sqrt(max(n_real, 1)),
        "cl_tt": tt.mean(0), "cl_tt_err": tt.std(0) / np.sqrt(max(n_real, 1)),
    }
    if y_maps is not None:
        yt = np.asarray(yt)
        out["cl_yt"] = yt.mean(0)
        out["cl_yt_err"] = yt.std(0) / np.sqrt(max(n_real, 1))
    return out


# ── peaks / PDF / moments / Betti ─────────────────────────────────────────────

#: Canonical S/N axis shared by PDF, peaks, minima and V0/V1/V2: the 22 values
#: -2.5, -2.0, ... 8.0 (step 0.5).  Every one of the six statistics reports
#: length-22 arrays on EXACTLY these nu values — histograms (PDF/peaks/minima)
#: are binned on edges *centred* on them (see :func:`nu_edges`), Minkowski
#: functionals are evaluated *at* them.
NU_CANON: np.ndarray = np.linspace(-2.5, 8.0, 22)


def nu_edges(centers: np.ndarray) -> np.ndarray:
    """Histogram edges centred on uniformly spaced ``centers`` (n -> n+1)."""
    c = np.asarray(centers, dtype=float)
    h = 0.5 * (c[1] - c[0])
    return np.concatenate([c - h, [c[-1] + h]])


#: Histogram edges whose bin centres are exactly :data:`NU_CANON` (-2.75 ... 8.25).
NU_EDGES_CANON: np.ndarray = nu_edges(NU_CANON)

#: Default smoothing scales of :func:`nongaussian_stats` (its FIRST entry is the
#: scale the Minkowski functionals are computed at — a fixed-sigma caller must
#: normalise V0/V1/V2 by sigma at THIS scale, not at the peak scale).
NG_SCALES_DEFAULT: tuple[float, ...] = (1.0, 2.0, 5.0, 8.0)

_GK_CACHE: dict = {}


def _gaussian_smooth(m: np.ndarray, smoothing_arcmin: float, fov_deg: float) -> np.ndarray:
    """Periodic Gaussian smoothing, done in Fourier space.

    Equivalent to ``scipy.ndimage.gaussian_filter(..., mode="wrap")`` — for a
    periodic domain the FFT convolution is the *exact* same operator, not an
    approximation (verified: correlation > 0.999999 at every production scale)
    — but 4–5x faster on the 1024^2 maps, and smoothing dominates the stats
    budget (nongaussian_stats alone is ~50% of it).  The transfer function per
    (shape, sigma) is cached, so repeated calls at the same scale are just two
    FFTs and a multiply.
    """
    if smoothing_arcmin <= 0:
        return m
    n = m.shape[0]
    pix_arcmin = fov_deg * 60.0 / n
    sig = smoothing_arcmin / pix_arcmin
    key = (n, m.shape[1], round(float(sig), 10))
    W = _GK_CACHE.get(key)
    if W is None:
        ky = np.fft.fftfreq(m.shape[0])[:, None]
        kx = np.fft.rfftfreq(m.shape[1])[None, :]
        W = np.exp(-2.0 * (np.pi * sig) ** 2 * (ky ** 2 + kx ** 2))
        if len(_GK_CACHE) > 32:          # bounded: a run uses a handful of scales
            _GK_CACHE.clear()
        _GK_CACHE[key] = W
    return np.fft.irfft2(np.fft.rfft2(m) * W, s=m.shape)


def _local_extrema(m: np.ndarray, maxima: bool) -> np.ndarray:
    """Mask of pixels strictly above (maxima) / below (minima) all 8 neighbours."""
    out = np.ones(m.shape, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = np.roll(np.roll(m, dx, 0), dy, 1)
            out &= (m > nb) if maxima else (m < nb)
    return out


def _noise_sigma_pix(shape_noise_ngal, sigma_e: float, pix_arcmin: float):
    """Per-pixel kappa shape-noise rms, ``sigma_e / sqrt(n_gal * A_pix)``.

    ``sigma_e`` is the per-component ellipticity dispersion (KS inversion gives
    kappa the same white-noise level as a single shear component).  Returns
    ``None`` when shape noise is off.
    """
    if shape_noise_ngal is None or shape_noise_ngal <= 0:
        return None
    return sigma_e / np.sqrt(shape_noise_ngal * pix_arcmin ** 2)


def _nu_sigma(sm: np.ndarray, sig_pix, smoothing_arcmin: float,
              pix_arcmin: float, nu_norm: str, sigma0=None) -> float:
    """Normalisation for ``nu``: map std, smoothed-noise rms, or a FIXED sigma.

    ``nu_norm="noise"`` (requires shape noise on, smoothing > 0) uses the white
    noise rms after the Gaussian smoothing, ``sigma_pix / (2 sqrt(pi) sigma_s)``
    with ``sigma_s`` the kernel sigma in pixels — the survey S/N convention.
    ``nu_norm="fixed"`` uses the supplied ``sigma0`` (e.g. the FIDUCIAL map rms)
    so that parameter variations are measured against one common nu scale —
    per-map normalisation would absorb the few-% sigma_kappa response itself.
    """
    if nu_norm == "fixed":
        if sigma0 is None:
            raise ValueError("nu_norm='fixed' requires nu_sigma0")
        return float(sigma0)
    if nu_norm == "noise":
        if sig_pix is None or smoothing_arcmin <= 0:
            raise ValueError("nu_norm='noise' requires shape_noise_ngal and smoothing > 0")
        return sig_pix / (2.0 * np.sqrt(np.pi) * (smoothing_arcmin / pix_arcmin))
    return float(sm.std()) + 1e-30


def _maybe_add_noise(m: np.ndarray, sig_pix, noise_seed: int, r: int, i: int):
    if sig_pix is None:
        return m
    rng = np.random.default_rng((noise_seed, r, i))
    return m + rng.normal(0.0, sig_pix, m.shape)


def peak_counts(
    kappa_maps: np.ndarray,
    *,
    fov_deg: float = 5.0,
    smoothing_arcmin=2.0,
    nu_bins: np.ndarray | None = None,
    shape_noise_ngal: float | None = None,
    sigma_e: float = 0.26,
    noise_seed: int = 0,
    nu_norm: str = "map",
    nu_sigma0=None,
    return_realizations: bool = False,
) -> dict:
    """Convergence peak AND minimum counts vs S/N ``nu``.

    Peaks (minima) are pixels strictly greater (less) than all 8 neighbours of
    the smoothed map.  ``kappa_maps`` ``(n_real, n_src, npix, npix)``.

    ``smoothing_arcmin`` may be a scalar (legacy output shapes ``(n_src, n_nu)``)
    or a sequence of scales (outputs gain a leading scale axis,
    ``(n_scale, n_src, n_nu)``).  ``shape_noise_ngal`` (gal/arcmin^2) adds white
    kappa noise *before* smoothing — one galaxy realization per (real, src),
    shared across scales, seeded by ``noise_seed``.  ``nu_norm``: ``"map"``
    normalises by the smoothed map's own std (default, matches noiseless runs);
    ``"noise"`` by the analytic smoothed-noise rms (survey S/N convention).
    """
    kappa_maps = np.asarray(kappa_maps)
    n_real, n_src = kappa_maps.shape[:2]
    npix = kappa_maps.shape[-1]
    pix_arcmin = fov_deg * 60.0 / npix
    scalar_scale = np.isscalar(smoothing_arcmin)
    scales = np.atleast_1d(np.asarray(smoothing_arcmin, dtype=float))
    sig_pix = _noise_sigma_pix(shape_noise_ngal, sigma_e, pix_arcmin)
    if nu_bins is None:
        nu_bins = np.linspace(-5.0, 12.0, 69)
    centres = 0.5 * (nu_bins[1:] + nu_bins[:-1])
    s0 = (np.broadcast_to(np.atleast_2d(np.asarray(nu_sigma0, dtype=float)),
                          (len(scales), n_src))
          if nu_sigma0 is not None else None)
    peaks = np.zeros((len(scales), n_real, n_src, len(centres)))
    mins = np.zeros((len(scales), n_real, n_src, len(centres)))
    for r in range(n_real):
        for i in range(n_src):
            m = _maybe_add_noise(kappa_maps[r, i], sig_pix, noise_seed, r, i)
            for s, sc in enumerate(scales):
                sm = _gaussian_smooth(m, sc, fov_deg)
                sig = _nu_sigma(sm, sig_pix, sc, pix_arcmin, nu_norm,
                                s0[s, i] if s0 is not None else None)
                nu = (sm - sm.mean()) / sig
                peaks[s, r, i] = np.histogram(nu[_local_extrema(sm, True)], bins=nu_bins)[0]
                mins[s, r, i] = np.histogram(nu[_local_extrema(sm, False)], bins=nu_bins)[0]
    rt = np.sqrt(max(n_real, 1))
    out = {"nu": centres,
           "peak_counts": peaks.mean(1), "peak_counts_err": peaks.std(1) / rt,
           "minima_counts": mins.mean(1), "minima_counts_err": mins.std(1) / rt,
           "smoothing_arcmin": scales[0] if scalar_scale else scales,
           "shape_noise_ngal": shape_noise_ngal or 0.0,
           "sigma_e": sigma_e, "nu_norm": nu_norm}
    if scalar_scale:
        for k in ("peak_counts", "peak_counts_err", "minima_counts", "minima_counts_err"):
            out[k] = out[k][0]
    if return_realizations:                          # (n_real, n_src, n_nu) [scalar scale]
        out["peak_counts_real"] = peaks[0] if scalar_scale else peaks
        out["minima_counts_real"] = mins[0] if scalar_scale else mins
    return out


def _radial_index(stamp_r: int, n_r: int):
    """Radial-bin index for a (2r+1)^2 stamp; -1 outside the disc of radius r."""
    off = np.arange(-stamp_r, stamp_r + 1)
    rr = np.hypot(*np.meshgrid(off, off, indexing="ij"))
    edges = np.linspace(0.0, stamp_r, n_r + 1)
    idx = np.digitize(rr.ravel(), edges) - 1
    idx[idx >= n_r] = -1            # corners beyond the disc
    return off, idx, edges


def peak_cross_stats(
    kappa_maps: np.ndarray,
    y_maps: np.ndarray,
    *,
    fov_deg: float = 5.0,
    smoothing_arcmin: float = 2.0,
    nu_bins: np.ndarray | None = None,
    profile_r_arcmin: float = 10.0,
    n_r: int = 12,
    want_profile: bool = True,
    shape_noise_ngal: float | None = None,
    sigma_e: float = 0.26,
    noise_seed: int = 0,
    nu_norm: str = "map",
    nu_sigma0=None,
) -> dict:
    """tSZ stacked at WL peaks: ``R(nu) = <y>/<kappa>`` + the full data vector.

    ``kappa_maps`` and ``y_maps`` are ``(n_real, n_src, npix, npix)`` and paired
    per source bin (``y_maps[:, i]`` = cumulative Compton-y to the same source
    plane as ``kappa_maps[:, i]``).  Peaks are local maxima of the smoothed
    convergence; at each peak we read kappa (S/N) and the y value, and stack a
    radial y profile.  ``R = <y>/<kappa>`` at fixed nu is a *specific thermal
    energy* (``R ∝ Y/M ∝ f_gas T_mw``), NOT a pure gas fraction — pair with a
    density-weighted probe (tau / kSZ / DM) to separate ``f_gas`` from ``T``.

    Shape noise (``shape_noise_ngal``, ``sigma_e``, ``noise_seed``, ``nu_norm``)
    is applied to kappa only, exactly as in :func:`peak_counts` (same seeds →
    same noisy skies); y is read at the noisy-kappa peak positions, which is the
    survey-realistic selection.

    Returns per source bin (arrays shaped ``(n_src, n_nu[, n_r])``):
    ``nu``, ``n_peaks`` (mean per realization), ``kappa_peak``, ``y_peak``,
    ``R``, ``cov_ky`` / ``corr_ky`` (halo-to-halo scatter at fixed nu),
    ``profile`` (stacked ``<y(theta)>_peaks``) with ``r_arcmin``.
    """
    kappa_maps = np.asarray(kappa_maps)
    y_maps = np.asarray(y_maps)
    n_real, n_src, npix, _ = kappa_maps.shape
    if nu_bins is None:
        nu_bins = np.concatenate([np.arange(1.0, 6.0, 0.5),
                                  [6.0, 7.0, 8.5, 10.5, 13.0]])
    nu_c = 0.5 * (nu_bins[1:] + nu_bins[:-1])
    n_nu = len(nu_c)
    pix_arcmin = fov_deg * 60.0 / npix
    sig_pix = _noise_sigma_pix(shape_noise_ngal, sigma_e, pix_arcmin)
    sr = max(1, int(round(profile_r_arcmin / pix_arcmin)))
    off, ridx, redges = _radial_index(sr, n_r)
    r_arcmin = 0.5 * (redges[1:] + redges[:-1]) * pix_arcmin

    kap_s = np.zeros((n_src, n_nu))
    kap_sq = np.zeros((n_src, n_nu))
    y_s = np.zeros((n_src, n_nu))
    y_sq = np.zeros((n_src, n_nu))
    ky = np.zeros((n_src, n_nu))
    cnt = np.zeros((n_src, n_nu))
    prof_s = np.zeros((n_src, n_nu, n_r))
    prof_n = np.zeros((n_src, n_nu, n_r))
    npk_real = np.zeros((n_real, n_src))
    valid_r = ridx >= 0
    ridx_v = ridx[valid_r]

    s0 = (np.broadcast_to(np.atleast_1d(np.asarray(nu_sigma0, dtype=float)), (n_src,))
          if nu_sigma0 is not None else None)
    for r in range(n_real):
        for i in range(n_src):
            kin = _maybe_add_noise(kappa_maps[r, i], sig_pix, noise_seed, r, i)
            ksm = _gaussian_smooth(kin, smoothing_arcmin, fov_deg)
            ymap = y_maps[r, i]
            sig = _nu_sigma(ksm, sig_pix, smoothing_arcmin, pix_arcmin, nu_norm,
                            s0[i] if s0 is not None else None)
            nu = (ksm - ksm.mean()) / sig
            pi, pj = np.where(_local_extrema(ksm, True))
            pnu = nu[pi, pj]
            sel = (pnu >= nu_bins[0]) & (pnu < nu_bins[-1])
            pi, pj, pnu = pi[sel], pj[sel], pnu[sel]
            npk_real[r, i] = len(pi)
            if len(pi) == 0:
                continue
            b = np.digitize(pnu, nu_bins) - 1
            kval = ksm[pi, pj]
            yval = ymap[pi, pj]
            for bb in range(n_nu):
                m = b == bb
                if not m.any():
                    continue
                kap_s[i, bb] += kval[m].sum()
                kap_sq[i, bb] += (kval[m] ** 2).sum()
                y_s[i, bb] += yval[m].sum()
                y_sq[i, bb] += (yval[m] ** 2).sum()
                ky[i, bb] += (kval[m] * yval[m]).sum()
                cnt[i, bb] += m.sum()
            if want_profile:
                ix = (pi[:, None] + off[None, :]) % npix       # (npk, S)
                iy = (pj[:, None] + off[None, :]) % npix
                stamps = ymap[ix[:, :, None], iy[:, None, :]]  # (npk, S, S)
                vals = stamps.reshape(len(pi), -1)[:, valid_r]
                flat = (b[:, None] * n_r + ridx_v[None, :]).ravel()
                np.add.at(prof_s[i].reshape(-1), flat, vals.ravel())
                np.add.at(prof_n[i].reshape(-1), flat, 1.0)

    c = np.where(cnt > 0, cnt, 1.0)
    kap = kap_s / c
    ym = y_s / c
    cov = ky / c - kap * ym
    vk = kap_sq / c - kap ** 2
    vy = y_sq / c - ym ** 2
    corr = cov / np.sqrt(np.where(vk > 0, vk, np.nan) * np.where(vy > 0, vy, np.nan))
    return {
        "nu": nu_c,
        "n_peaks": npk_real.mean(0),
        "n_peaks_total": cnt,
        "kappa_peak": kap,
        "y_peak": ym,
        "R": ym / np.where(kap > 0, kap, np.nan),
        "cov_ky": cov, "corr_ky": corr,
        "profile": prof_s / np.where(prof_n > 0, prof_n, 1.0),
        "r_arcmin": r_arcmin,
        "smoothing_arcmin": smoothing_arcmin,
        "shape_noise_ngal": shape_noise_ngal or 0.0,
        "sigma_e": sigma_e, "nu_norm": nu_norm,
    }


def minkowski_functionals(
    field: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D Minkowski functionals ``V0, V1, V2`` of ``field`` vs ``thresholds``.

    Differential (Schmalzing & Buchert 1998) estimator over the field gradients:

    * ``V0(nu) = <Theta(f - nu)>`` — area fraction above threshold,
    * ``V1(nu) = (1/4) <delta(f - nu) |grad f|>`` — boundary length,
    * ``V2(nu) = (1/2pi) <delta(f - nu) kappa_g>`` — Euler characteristic, with
      ``kappa_g = (2 f_x f_y f_xy - f_x^2 f_yy - f_y^2 f_xx)/(f_x^2 + f_y^2)``.

    ``delta(f - nu)`` is estimated by binning the field at ``thresholds``.
    Gradients are in pixel units (an overall scale, constant across runs).
    """
    f = np.asarray(field, dtype=np.float64)
    fx, fy = np.gradient(f)
    fxx, fxy = np.gradient(fx)
    _, fyy = np.gradient(fy)
    grad = np.hypot(fx, fy)
    denom = fx ** 2 + fy ** 2
    kappa_g = (2 * fx * fy * fxy - fx ** 2 * fyy - fy ** 2 * fxx) / np.where(denom > 0, denom, 1.0)

    thr = np.asarray(thresholds, dtype=np.float64)
    dnu = float(thr[1] - thr[0])
    edges = np.concatenate([thr - dnu / 2, [thr[-1] + dnu / 2]])
    N = f.size

    V0 = np.array([(f > nu).mean() for nu in thr])
    V1 = np.histogram(f, bins=edges, weights=grad)[0] / (4.0 * N * dnu)
    V2 = np.histogram(f, bins=edges, weights=kappa_g)[0] / (2.0 * np.pi * N * dnu)
    return V0, V1, V2


def nongaussian_stats(
    kappa_maps: np.ndarray,
    *,
    fov_deg: float = 5.0,
    pdf_bins: int = 41,
    smoothing_scales_arcmin=(1.0, 2.0, 5.0, 8.0),
    mf_thresholds: np.ndarray | None = None,
    return_realizations: bool = False,
    nu_centers: np.ndarray | None = None,
    nu_sigma0=None,
    nu_sigma0_unsmoothed=None,
) -> dict:
    """Convergence PDF, per-scale moments, and Minkowski functionals (V0,V1,V2).

    ``kappa_maps`` ``(n_real, n_src, npix, npix)``.  Returns ``pdf`` /
    ``pdf_bins`` ``(n_src, ...)``; ``variance/skewness/kurtosis``
    ``(n_src, n_scales)``; ``V0/V1/V2`` ``(n_src, n_thr)`` vs ``mf_nu`` of the
    map smoothed at the first scale and normalised to S/N units.

    ``nu_centers`` (e.g. :data:`NU_CANON`) switches the PDF and the Minkowski
    functionals onto a common S/N axis: the PDF is histogrammed in
    ``nu = kappa / sigma`` on bins *centred* on those values and the MFs are
    evaluated *at* them, so ``pdf``, ``V0``, ``V1``, ``V2`` (and the peak /
    minimum counts computed with ``nu_bins=nu_edges(nu_centers)``) all share one
    length-``len(nu_centers)`` axis.  ``nu_sigma0`` ``(n_scales, n_src)`` and ``nu_sigma0_unsmoothed``
    ``(n_src,)`` supply a FIXED sigma — normally the fiducial run's — so every
    run shares one nu scale; per-map sigma would absorb the few-% sigma_kappa
    response itself.  Both default to each map's own std (per-map convention).
    """
    kappa_maps = np.asarray(kappa_maps)
    n_real, n_src = kappa_maps.shape[:2]
    scales = np.asarray(smoothing_scales_arcmin, dtype=float)
    nu_units = nu_centers is not None
    if mf_thresholds is None:
        mf_thresholds = (np.asarray(nu_centers, float) if nu_units
                         else np.linspace(-3.0, 4.0, 29))
    s0 = (np.broadcast_to(np.atleast_2d(np.asarray(nu_sigma0, float)),
                          (len(scales), n_src)) if nu_sigma0 is not None else None)
    s0u = (np.broadcast_to(np.asarray(nu_sigma0_unsmoothed, float).reshape(-1), (n_src,))
           if nu_sigma0_unsmoothed is not None else None)

    if nu_units:
        edges = nu_edges(nu_centers)              # bins centred on nu_centers
    else:
        sig = kappa_maps.std()                     # symmetric range from global std
        edges = np.linspace(-6 * sig, 6 * sig, pdf_bins + 1)
    pcent = 0.5 * (edges[1:] + edges[:-1])

    pdf = np.zeros((n_src, len(pcent)))
    var = np.zeros((n_src, len(scales)))
    skew = np.zeros((n_src, len(scales)))
    kurt = np.zeros((n_src, len(scales)))
    V0 = np.zeros((n_src, len(mf_thresholds)))
    V1 = np.zeros((n_src, len(mf_thresholds)))
    V2 = np.zeros((n_src, len(mf_thresholds)))
    # per-realization cubes (only kept/returned when return_realizations)
    skew_r = np.zeros((n_real, n_src, len(scales)))
    V0_r = np.zeros((n_real, n_src, len(mf_thresholds)))
    V1_r = np.zeros((n_real, n_src, len(mf_thresholds)))
    V2_r = np.zeros((n_real, n_src, len(mf_thresholds)))

    for i in range(n_src):
        for r in range(n_real):
            m = kappa_maps[r, i]
            d0 = m - m.mean()
            if nu_units:                       # PDF in nu = kappa / sigma units
                d0 = d0 / (float(s0u[i]) if s0u is not None else (d0.std() + 1e-30))
            pdf[i] += np.histogram(d0, bins=edges, density=True)[0]
            for k, sc in enumerate(scales):
                sm = _gaussian_smooth(m, sc, fov_deg)
                d = sm - sm.mean()
                s2 = d.var()
                var[i, k] += s2
                sk = (d ** 3).mean() / (s2 ** 1.5 + 1e-30)
                skew[i, k] += sk
                skew_r[r, i, k] = sk
                kurt[i, k] += (d ** 4).mean() / (s2 ** 2 + 1e-30) - 3.0
            sm0 = _gaussian_smooth(m, scales[0], fov_deg)
            sig_mf = float(s0[0, i]) if s0 is not None else (sm0.std() + 1e-30)
            nu = (sm0 - sm0.mean()) / sig_mf
            v0, v1, v2 = minkowski_functionals(nu, mf_thresholds)
            V0[i] += v0
            V1[i] += v1
            V2[i] += v2
            V0_r[r, i] = v0
            V1_r[r, i] = v1
            V2_r[r, i] = v2
    for arr in (pdf, var, skew, kurt, V0, V1, V2):
        arr /= n_real
    out = {
        "pdf": pdf, "pdf_bins": pcent,
        "smoothing_scales_arcmin": scales,
        "variance": var, "skewness": skew, "kurtosis": kurt,
        "mf_nu": mf_thresholds, "V0": V0, "V1": V1, "V2": V2,
    }
    if return_realizations:
        out.update(skewness_real=skew_r, V0_real=V0_r, V1_real=V1_r, V2_real=V2_r)
    return out


# ── wavelet scattering transform (WST) ────────────────────────────────────────

def _scattering2d_cls():
    """Return kymatio's torch ``Scattering2D``, shimming ``scipy.special.sph_harm``.

    kymatio 0.3's package import eagerly pulls in the 3-D solid-harmonic frontend,
    which imports ``scipy.special.sph_harm`` — removed in scipy>=1.13.  2-D
    scattering never uses it, so we install a harmless stub when it is absent.
    """
    import scipy.special as _sp
    if not hasattr(_sp, "sph_harm"):
        try:
            from scipy.special import sph_harm_y as _shy
            _sp.sph_harm = lambda m, n, theta, phi: _shy(n, m, phi, theta)
        except Exception:
            _sp.sph_harm = lambda *a, **k: 0.0
    from kymatio.torch import Scattering2D
    return Scattering2D


def wst(
    kappa_maps: np.ndarray,
    *,
    J: int = 4,
    L: int = 4,
    n_real: int | None = None,
    device: str | None = None,
    return_realizations: bool = False,
) -> dict:
    """2-D wavelet scattering coefficients of the convergence maps (kymatio).

    ``kappa_maps`` ``(n_real, n_src, npix, npix)``.  For each map we compute the
    order 0/1/2 ``Scattering2D`` coefficients and spatially average each path,
    giving a fixed-length scattering vector per source plane.  ``J`` octaves and
    ``L`` orientations set the coefficient count
    ``C = 1 + J*L + L^2 * J*(J-1)/2`` (e.g. ``J=4, L=4`` -> 113).

    The transform runs on GPU when available (set ``device``).  WST is the
    heaviest field statistic, so by default only the first ``n_real`` (``None`` =
    all) realizations are used — ~10 is plenty for the realization mean.  Returns
    ``wst`` ``(n_src, C)`` (mean over realizations), ``wst_err``, the path
    metadata, and — with ``return_realizations`` — the ``(n_real, n_src, C)`` cube
    used for covariance / the generative flow target.
    """
    import torch

    Scattering2D = _scattering2d_cls()
    kappa_maps = np.asarray(kappa_maps)
    nR_all, nS, npix, _ = kappa_maps.shape
    nR = nR_all if n_real is None else min(nR_all, int(n_real))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scattering = Scattering2D(J=J, L=L, shape=(npix, npix)).to(device)

    coeffs = np.empty((nR, nS, 0), dtype=np.float64)
    rows = []
    with torch.no_grad():
        for r in range(nR):
            x = torch.from_numpy(kappa_maps[r].astype(np.float32)).to(device)  # (nS,npix,npix)
            x = x - x.mean(dim=(-2, -1), keepdim=True)
            s = scattering(x).mean(dim=(-2, -1))          # (nS, C)
            rows.append(s.detach().cpu().numpy())
    coeffs = np.asarray(rows)                              # (nR, nS, C)
    out = {
        "wst": coeffs.mean(0), "wst_err": coeffs.std(0) / np.sqrt(max(nR, 1)),
        "J": J, "L": L, "n_coeff": coeffs.shape[-1], "n_real_used": nR,
    }
    if return_realizations:
        out["wst_real"] = coeffs
    return out


# ── dispersion measure (DM / kSZ optical depth) field statistics ──────────────

def dm_stats(
    tau_maps: np.ndarray,
    *,
    n_bins: int = 41,
    smoothing_arcmin: float = 0.0,
    fov_deg: float = 5.0,
    dm_edges: np.ndarray | None = None,
    return_realizations: bool = False,
) -> dict:
    """Dispersion-measure PDF + fluctuation moments from the tau (electron-column) maps.

    ``tau_maps`` ``(n_real, n_src, npix, npix)`` is the cumulative kSZ optical
    depth to each source plane; ``DM[pc/cm^3] = tau / TAU_PER_DM`` (the constant
    from :mod:`bind.inference.lightcone_maps`).  Per source plane we return the DM
    PDF, the mean DM (a nuisance amplitude — the Macquart term), and the
    *fluctuation* statistics the feedback model actually controls: ``sigma_dm``,
    the ``F = sigma_dm / <DM>`` parameter (CAMELS feedback-sensitive), skewness
    and kurtosis.  ``smoothing_arcmin > 0`` smooths each map first.

    Returns ``dm_bins`` ``(n_bins,)``, ``dm_pdf`` ``(n_src, n_bins)``,
    ``dm_mean/sigma_dm/F/skewness/kurtosis`` ``(n_src,)`` (means over
    realizations), and — with ``return_realizations`` — per-realization cubes.

    The histogram range defaults to the 0.1/99.9 percentiles of ``tau_maps``, so
    a caller that splits the realizations across processes MUST pass a common
    ``dm_edges`` (``n_bins + 1`` edges): axes derived per call would differ from
    each other and averaging their PDFs would be meaningless.
    """
    from bind.inference.lightcone_maps import TAU_PER_DM

    tau = np.asarray(tau_maps)
    nR, nS = tau.shape[:2]
    dm = tau / TAU_PER_DM
    if dm_edges is None:
        lo, hi = np.percentile(dm, [0.1, 99.9])
        edges = np.linspace(float(lo), float(hi), n_bins + 1)
    else:
        edges = np.asarray(dm_edges, dtype=float)
        n_bins = len(edges) - 1
    cent = 0.5 * (edges[1:] + edges[:-1])

    pdf = np.zeros((nS, n_bins))
    mean = np.zeros((nR, nS))
    sig = np.zeros((nR, nS))
    sk = np.zeros((nR, nS))
    ku = np.zeros((nR, nS))
    pdf_r = np.zeros((nR, nS, n_bins))
    for r in range(nR):
        for i in range(nS):
            m = dm[r, i]
            if smoothing_arcmin > 0:
                m = _gaussian_smooth(m, smoothing_arcmin, fov_deg)
            h = np.histogram(m, bins=edges, density=True)[0]
            pdf[i] += h
            pdf_r[r, i] = h
            mu = m.mean()
            d = m - mu
            s2 = d.var()
            mean[r, i] = mu
            sig[r, i] = np.sqrt(s2)
            sk[r, i] = (d ** 3).mean() / (s2 ** 1.5 + 1e-30)
            ku[r, i] = (d ** 4).mean() / (s2 ** 2 + 1e-30) - 3.0
    pdf /= max(nR, 1)
    out = {
        "dm_bins": cent, "dm_pdf": pdf,
        "dm_mean": mean.mean(0), "sigma_dm": sig.mean(0),
        "F": (sig / np.where(mean > 0, mean, np.nan)).mean(0),
        "skewness": sk.mean(0), "kurtosis": ku.mean(0),
        "TAU_PER_DM": float(TAU_PER_DM), "smoothing_arcmin": smoothing_arcmin,
    }
    if return_realizations:
        out["dm_pdf_real"] = pdf_r
        out["sigma_dm_real"] = sig
    return out


# ── per-halo scaling relations ────────────────────────────────────────────────

def halo_scaling(
    composite_slab_paths,
    *,
    r500_over_r200: float = 0.659,
    pixel_size_mpch: float = 6.25 / 128.0,
) -> dict:
    """Per-halo ``Y_500c, f_gas, f_star, T_mw`` from composite per-halo patches.

    Reads ``generated_patches`` (DM, gas, stars mass) and ``thermo_patches``
    (compton_y, T, entropy, P_e) from each ``composite_slab*.npz`` (stage 2 with
    ``save_patches``).  Apertures are circular at ``R_500c`` and ``R_200c`` about
    the patch centre.  ``Y`` is ``sum(y) * pixel_area`` within R_500c [(Mpc/h)^2].

    Returns 1-D arrays concatenated over all halos in all slabs.
    """
    import numpy as np
    out = {k: [] for k in ("halo_mass", "r200", "Y_500c", "f_gas_500c", "f_star_500c",
                           "f_gas_200c", "f_star_200c", "T_mw_500c")}
    pix_area = pixel_size_mpch ** 2
    for p in composite_slab_paths:
        d = np.load(p)
        if "generated_patches" not in d.files or int(d["n_halos"]) == 0:
            continue
        gen = d["generated_patches"]                       # (n, 3, P, P)
        thermo = d["thermo_patches"] if "thermo_patches" in d.files else None
        masses, r200 = d["halo_masses"], d["halo_r200"]
        n, _, P, _ = gen.shape
        cen = P // 2
        yy, xx = np.mgrid[0:P, 0:P]
        rr = np.hypot(xx - cen, yy - cen) * pixel_size_mpch   # Mpc/h from centre
        for h in range(n):
            dm, gas, star = gen[h, 0], gen[h, 1], gen[h, 2]
            tot = dm + gas + star
            for tag, rad in (("500c", r500_over_r200 * float(r200[h])),
                             ("200c", float(r200[h]))):
                ap = rr <= rad
                mt = float(tot[ap].sum()) + 1e-30
                out[f"f_gas_{tag}"].append(float(gas[ap].sum()) / mt)
                out[f"f_star_{tag}"].append(float(star[ap].sum()) / mt)
            ap5 = rr <= r500_over_r200 * float(r200[h])
            if thermo is not None:
                yv = thermo[h, 0]
                out["Y_500c"].append(float(yv[ap5].sum()) * pix_area)
                gw = gas[ap5]
                out["T_mw_500c"].append(
                    float((thermo[h, 1][ap5] * gw).sum()) / (float(gw.sum()) + 1e-30))
            else:
                out["Y_500c"].append(np.nan)
                out["T_mw_500c"].append(np.nan)
            out["halo_mass"].append(float(masses[h]))
            out["r200"].append(float(r200[h]))
    return {k: np.asarray(v) for k, v in out.items()}


def scaling_relations(
    halo_mass: np.ndarray,
    *,
    Y: np.ndarray | None = None,
    f_gas: np.ndarray | None = None,
    f_star: np.ndarray | None = None,
    T: np.ndarray | None = None,
    mass_bins: np.ndarray | None = None,
    min_per_bin: int = 5,
) -> dict:
    """Bin per-halo quantities into fixed-length Y-M / f_gas-M / f_star-M / T-M vectors.

    Turns the variable-length per-halo arrays (from :func:`halo_scaling` or a
    slice of the SB35 ``integrated.parquet`` cache) into fixed-length emulator
    targets: the median and (log-)scatter of each supplied quantity in fixed
    ``log10(M)`` bins.  ``halo_mass`` is in Msun/h; ``mass_bins`` are the
    ``log10(M)`` bin *edges* (default 10 bins over 12.5-15.0).  Bins with fewer
    than ``min_per_bin`` halos are NaN (the compressor treats NaN as "no data").

    Returns ``log_mass_bins`` (bin centres) + ``<q>_median`` / ``<q>_scatter``
    ``(n_bin,)`` for every provided quantity, and ``n_halos`` per bin.
    """
    halo_mass = np.asarray(halo_mass, dtype=float)
    if mass_bins is None:
        mass_bins = np.linspace(12.5, 15.0, 11)
    lm = np.log10(np.where(halo_mass > 0, halo_mass, np.nan))
    idx = np.digitize(lm, mass_bins) - 1
    nb = len(mass_bins) - 1
    out = {
        "log_mass_bins": 0.5 * (mass_bins[1:] + mass_bins[:-1]),
        "n_halos": np.array([int(np.sum(idx == b)) for b in range(nb)]),
    }
    for name, arr in (("Y", Y), ("f_gas", f_gas), ("f_star", f_star), ("T", T)):
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        med = np.full(nb, np.nan)
        scat = np.full(nb, np.nan)
        for b in range(nb):
            sel = (idx == b) & np.isfinite(arr)
            if int(sel.sum()) >= min_per_bin:
                v = arr[sel]
                med[b] = float(np.median(v))
                pos = v[v > 0]
                scat[b] = float(np.std(np.log10(pos))) if pos.size == v.size else float(np.std(v))
        out[f"{name}_median"] = med
        out[f"{name}_scatter"] = scat
    return out
