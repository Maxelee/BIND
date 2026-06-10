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
  (a projected gas-fraction proxy), peak counts, the stacked radial y profile,
  and ``Cov(kappa_peak, y_peak)`` per S/N bin and source redshift.
* :func:`halo_scaling`    — per-halo ``Y_500c, f_gas, f_star, T_mw`` from the
  composite per-halo patches.

Angular power spectra are computed with **Pylians** ``Pk_plane`` / ``XPk_plane``
using ``BoxSize = fov`` in radians, so the returned wavenumber ``k`` is the
multipole ``ell`` and the binning matches the rest of the group's pipelines.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np


# ── power spectra (Pylians) ───────────────────────────────────────────────────

def power_spectrum(
    m1: np.ndarray,
    m2: np.ndarray | None = None,
    *,
    fov_deg: float = 5.0,
    subtract_mean: bool = True,
    threads: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat-sky auto (``m2=None``) or cross angular power via Pylians.

    ``BoxSize = deg2rad(fov_deg)`` so the returned ``k`` is the multipole ``ell``.
    Maps are already gridded, so ``MAS='None'`` (no deconvolution).  Returns
    ``(ell, C_ell)``.
    """
    import Pk_library as PKL

    fov = np.deg2rad(fov_deg)
    a = (m1 - m1.mean() if subtract_mean else m1).astype(np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        if m2 is None:
            p = PKL.Pk_plane(a, fov, MAS="None", threads=threads, verbose=False)
            return np.asarray(p.k), np.asarray(p.Pk)
        b = (m2 - m2.mean() if subtract_mean else m2).astype(np.float32)
        x = PKL.XPk_plane(a, b, fov, MAS1="None", MAS2="None", threads=threads)
    return np.asarray(x.k), np.asarray(x.XPk)


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
        mat = []
        for i in range(n_src):
            row = []
            for j in range(n_src):
                m2 = None if i == j else kappa_maps[r, j]
                ell, cl = power_spectrum(kappa_maps[r, i], m2, fov_deg=fov_deg)
                ell0 = ell
                row.append(cl)
            mat.append(row)
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
    ky = np.asarray(ky); yy = np.asarray(yy)
    return {
        "ell": ell0,
        "cl_ky": ky.mean(0), "cl_ky_err": ky.std(0) / np.sqrt(max(n_real, 1)),
        "cl_yy": yy.mean(0), "cl_yy_err": yy.std(0) / np.sqrt(max(n_real, 1)),
    }


# ── peaks / PDF / moments / Betti ─────────────────────────────────────────────

def _gaussian_smooth(m: np.ndarray, smoothing_arcmin: float, fov_deg: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    if smoothing_arcmin <= 0:
        return m
    pix_arcmin = fov_deg * 60.0 / m.shape[0]
    return gaussian_filter(m, smoothing_arcmin / pix_arcmin, mode="wrap")


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


def peak_counts(
    kappa_maps: np.ndarray,
    *,
    fov_deg: float = 5.0,
    smoothing_arcmin: float = 2.0,
    nu_bins: np.ndarray | None = None,
) -> dict:
    """Convergence peak AND minimum counts vs S/N ``nu = (kappa - mean) / std``.

    Peaks (minima) are pixels strictly greater (less) than all 8 neighbours of
    the smoothed map.  ``kappa_maps`` ``(n_real, n_src, npix, npix)``.  Returns
    ``nu`` (bin centres), ``peak_counts`` / ``minima_counts`` ``(n_src, n_nu)``
    (mean over realizations) and their errors.
    """
    kappa_maps = np.asarray(kappa_maps)
    n_real, n_src = kappa_maps.shape[:2]
    if nu_bins is None:
        nu_bins = np.linspace(-5.0, 6.0, 45)
    centres = 0.5 * (nu_bins[1:] + nu_bins[:-1])
    peaks = np.zeros((n_real, n_src, len(centres)))
    mins = np.zeros((n_real, n_src, len(centres)))
    for r in range(n_real):
        for i in range(n_src):
            sm = _gaussian_smooth(kappa_maps[r, i], smoothing_arcmin, fov_deg)
            nu = (sm - sm.mean()) / (sm.std() + 1e-30)
            peaks[r, i] = np.histogram(nu[_local_extrema(sm, True)], bins=nu_bins)[0]
            mins[r, i] = np.histogram(nu[_local_extrema(sm, False)], bins=nu_bins)[0]
    rt = np.sqrt(max(n_real, 1))
    return {"nu": centres,
            "peak_counts": peaks.mean(0), "peak_counts_err": peaks.std(0) / rt,
            "minima_counts": mins.mean(0), "minima_counts_err": mins.std(0) / rt,
            "smoothing_arcmin": smoothing_arcmin}


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
) -> dict:
    """tSZ stacked at WL peaks: ``R(nu) = <y>/<kappa>`` + the full data vector.

    ``kappa_maps`` and ``y_maps`` are ``(n_real, n_src, npix, npix)`` and paired
    per source bin (``y_maps[:, i]`` = cumulative Compton-y to the same source
    plane as ``kappa_maps[:, i]``).  Peaks are local maxima of the smoothed
    convergence; at each peak we read kappa (S/N) and the y value, and stack a
    radial y profile.  ``R = <y>/<kappa>`` at fixed nu is a projected gas-fraction
    proxy (``R ∝ Y/M ∝ f_gas T_mw``).

    Returns per source bin (arrays shaped ``(n_src, n_nu[, n_r])``):
    ``nu``, ``n_peaks`` (mean per realization), ``kappa_peak``, ``y_peak``,
    ``R``, ``cov_ky`` / ``corr_ky`` (halo-to-halo scatter at fixed nu),
    ``profile`` (stacked ``<y(theta)>_peaks``) with ``r_arcmin``.
    """
    kappa_maps = np.asarray(kappa_maps)
    y_maps = np.asarray(y_maps)
    n_real, n_src, npix, _ = kappa_maps.shape
    if nu_bins is None:
        nu_bins = np.linspace(1.0, 6.0, 11)
    nu_c = 0.5 * (nu_bins[1:] + nu_bins[:-1])
    n_nu = len(nu_c)
    pix_arcmin = fov_deg * 60.0 / npix
    sr = max(1, int(round(profile_r_arcmin / pix_arcmin)))
    off, ridx, redges = _radial_index(sr, n_r)
    r_arcmin = 0.5 * (redges[1:] + redges[:-1]) * pix_arcmin

    kap_s = np.zeros((n_src, n_nu)); kap_sq = np.zeros((n_src, n_nu))
    y_s = np.zeros((n_src, n_nu)); y_sq = np.zeros((n_src, n_nu))
    ky = np.zeros((n_src, n_nu)); cnt = np.zeros((n_src, n_nu))
    prof_s = np.zeros((n_src, n_nu, n_r)); prof_n = np.zeros((n_src, n_nu, n_r))
    npk_real = np.zeros((n_real, n_src))
    valid_r = ridx >= 0
    ridx_v = ridx[valid_r]

    for r in range(n_real):
        for i in range(n_src):
            ksm = _gaussian_smooth(kappa_maps[r, i], smoothing_arcmin, fov_deg)
            ymap = y_maps[r, i]
            nu = (ksm - ksm.mean()) / (ksm.std() + 1e-30)
            pi, pj = np.where(_local_extrema(ksm, True))
            pnu = nu[pi, pj]
            sel = (pnu >= nu_bins[0]) & (pnu < nu_bins[-1])
            pi, pj, pnu = pi[sel], pj[sel], pnu[sel]
            npk_real[r, i] = len(pi)
            if len(pi) == 0:
                continue
            b = np.digitize(pnu, nu_bins) - 1
            kval = ksm[pi, pj]; yval = ymap[pi, pj]
            for bb in range(n_nu):
                m = b == bb
                if not m.any():
                    continue
                kap_s[i, bb] += kval[m].sum(); kap_sq[i, bb] += (kval[m] ** 2).sum()
                y_s[i, bb] += yval[m].sum(); y_sq[i, bb] += (yval[m] ** 2).sum()
                ky[i, bb] += (kval[m] * yval[m]).sum(); cnt[i, bb] += m.sum()
            if want_profile:
                ix = (pi[:, None] + off[None, :]) % npix       # (npk, S)
                iy = (pj[:, None] + off[None, :]) % npix
                stamps = ymap[ix[:, :, None], iy[:, None, :]]  # (npk, S, S)
                vals = stamps.reshape(len(pi), -1)[:, valid_r]
                flat = (b[:, None] * n_r + ridx_v[None, :]).ravel()
                np.add.at(prof_s[i].reshape(-1), flat, vals.ravel())
                np.add.at(prof_n[i].reshape(-1), flat, 1.0)

    c = np.where(cnt > 0, cnt, 1.0)
    kap = kap_s / c; ym = y_s / c
    cov = ky / c - kap * ym
    vk = kap_sq / c - kap ** 2; vy = y_sq / c - ym ** 2
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
) -> dict:
    """Convergence PDF, per-scale moments, and Minkowski functionals (V0,V1,V2).

    ``kappa_maps`` ``(n_real, n_src, npix, npix)``.  Returns ``pdf`` /
    ``pdf_bins`` ``(n_src, ...)``; ``variance/skewness/kurtosis``
    ``(n_src, n_scales)``; ``V0/V1/V2`` ``(n_src, n_thr)`` vs ``mf_nu`` of the
    map smoothed at the first scale and normalised to S/N units.
    """
    kappa_maps = np.asarray(kappa_maps)
    n_real, n_src = kappa_maps.shape[:2]
    scales = np.asarray(smoothing_scales_arcmin, dtype=float)
    if mf_thresholds is None:
        mf_thresholds = np.linspace(-3.0, 4.0, 29)

    # symmetric PDF range from the global std
    sig = kappa_maps.std()
    edges = np.linspace(-6 * sig, 6 * sig, pdf_bins + 1)
    pcent = 0.5 * (edges[1:] + edges[:-1])

    pdf = np.zeros((n_src, len(pcent)))
    var = np.zeros((n_src, len(scales)))
    skew = np.zeros((n_src, len(scales)))
    kurt = np.zeros((n_src, len(scales)))
    V0 = np.zeros((n_src, len(mf_thresholds)))
    V1 = np.zeros((n_src, len(mf_thresholds)))
    V2 = np.zeros((n_src, len(mf_thresholds)))

    for i in range(n_src):
        for r in range(n_real):
            m = kappa_maps[r, i]
            pdf[i] += np.histogram(m - m.mean(), bins=edges, density=True)[0]
            for k, sc in enumerate(scales):
                sm = _gaussian_smooth(m, sc, fov_deg)
                d = sm - sm.mean()
                s2 = d.var()
                var[i, k] += s2
                skew[i, k] += (d ** 3).mean() / (s2 ** 1.5 + 1e-30)
                kurt[i, k] += (d ** 4).mean() / (s2 ** 2 + 1e-30) - 3.0
            sm0 = _gaussian_smooth(m, scales[0], fov_deg)
            nu = (sm0 - sm0.mean()) / (sm0.std() + 1e-30)
            v0, v1, v2 = minkowski_functionals(nu, mf_thresholds)
            V0[i] += v0; V1[i] += v1; V2[i] += v2
    for arr in (pdf, var, skew, kurt, V0, V1, V2):
        arr /= n_real
    return {
        "pdf": pdf, "pdf_bins": pcent,
        "smoothing_scales_arcmin": scales,
        "variance": var, "skewness": skew, "kurtosis": kurt,
        "mf_nu": mf_thresholds, "V0": V0, "V1": V1, "V2": V2,
    }


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
                out["Y_500c"].append(np.nan); out["T_mw_500c"].append(np.nan)
            out["halo_mass"].append(float(masses[h]))
            out["r200"].append(float(r200[h]))
    return {k: np.asarray(v) for k, v in out.items()}
