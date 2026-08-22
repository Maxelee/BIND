"""Weak-lensing convergence summary statistics (numpy-only).

These are the *exact* estimators the BIND WL statistics emulator
(:mod:`bind.wlemu`) was trained on, so statistics measured here on your own
maps are directly comparable to emulator predictions. All estimators are
flat-sky on a square periodic patch of side ``fov_deg``.

Blocks (names match the emulator's output blocks):

- ``Cl``      : flat-sky angular power spectrum of the raw map (18 log bins)
- ``pdf``     : one-point PDF of the smoothed map in S/N units
- ``peak``    : peak counts (local maxima) of the smoothed S/N map
- ``min``     : minima counts of the smoothed S/N map
- ``V0/V1/V2``: Minkowski functionals of the smoothed S/N map
- ``scat``    : wavelet-scattering coefficients (box-averaged raw map)
- ``moments`` : raw mean/std and smoothed mean/std (the amplitude information
  the S/N-normalized statistics discard)

Conventions (fixed by the training data — keep them to stay comparable):
smoothing is a **periodic Fourier-space** Gaussian at ``smoothing_arcmin``
(sigma, not FWHM); the smoothed map is standardized per map by its own
mean/std ("S/N units"); peaks/minima are strict 8-neighbor extrema with the
map border excluded; the scattering transform runs on the map box-averaged
to ``scat_res``.
"""

from __future__ import annotations

import numpy as np

DEG2RAD = np.pi / 180.0

BLOCKS = ("Cl", "pdf", "peak", "min", "V0", "V1", "V2", "scat", "moments")


def default_bins() -> dict:
    """Binning used by the training statistics cache (do not change if you
    want to compare against the emulator)."""
    return {
        "ell_bins": 18,
        "pdf_edges": np.linspace(-4, 6, 61),   # S/N units of the smoothed map
        "peak_edges": np.linspace(-2, 8, 41),  # S/N units (also used for minima)
        "mink_thr": np.linspace(-3, 4, 36),    # S/N units
    }


# ------------------------------------------------------------------ helpers --

def _as_batch(maps: np.ndarray) -> np.ndarray:
    maps = np.asarray(maps)
    if maps.ndim == 2:
        maps = maps[None]
    if maps.ndim != 3 or maps.shape[-1] != maps.shape[-2]:
        raise ValueError(f"expected (N, n, n) or (n, n) square maps, got {maps.shape}")
    return maps


def box_downsample(maps: np.ndarray, out_res: int) -> np.ndarray:
    """Box-average ``(N, n, n) -> (N, out_res, out_res)`` (physically correct
    coarsening for convergence, which averages linearly)."""
    maps = _as_batch(maps)
    n = maps.shape[-1]
    if n % out_res:
        raise ValueError(f"cannot box-average {n} -> {out_res}")
    f = n // out_res
    return maps.reshape(-1, out_res, f, out_res, f).mean(axis=(2, 4))


def _histogram_batch(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """``np.histogram`` semantics for a batch: vals (N, M) -> counts (N, nbins)."""
    nbins = len(edges) - 1
    idx = np.searchsorted(edges, vals, side="right") - 1  # [e_i, e_{i+1})
    idx = np.where(vals == edges[-1], nbins - 1, idx)     # last bin closed
    valid = (idx >= 0) & (idx < nbins)
    idx = np.where(valid, idx, 0)
    N = vals.shape[0]
    flat = (idx + nbins * np.arange(N)[:, None]).ravel()
    counts = np.bincount(flat, weights=valid.ravel().astype(np.float64),
                         minlength=N * nbins)
    return counts.reshape(N, nbins)


def _weighted_histogram_batch(vals: np.ndarray, weights: np.ndarray,
                              edges: np.ndarray) -> np.ndarray:
    nbins = len(edges) - 1
    idx = np.searchsorted(edges, vals, side="right") - 1
    valid = (idx >= 0) & (idx < nbins)
    idx = np.where(valid, idx, 0)
    N = vals.shape[0]
    flat = (idx + nbins * np.arange(N)[:, None]).ravel()
    w = np.where(valid, weights, 0.0).astype(np.float64)
    sums = np.bincount(flat, weights=w.ravel(), minlength=N * nbins)
    return sums.reshape(N, nbins)


# ----------------------------------------------------------- power spectrum --

def ell_bin_centers(n: int, fov_deg: float = 5.0, n_bins: int = 18) -> np.ndarray:
    L = fov_deg * DEG2RAD
    pix = L / n
    edges = np.logspace(np.log10(2 * np.pi / L), np.log10(np.pi / pix), n_bins + 1)
    return np.sqrt(edges[:-1] * edges[1:])


def power_spectrum(maps: np.ndarray, fov_deg: float = 5.0, n_bins: int = 18):
    """Flat-sky angular power spectrum C(l) of ``(N, n, n)`` maps.

    Returns ``(ell_centers, Cl)`` with ``Cl`` shaped ``(N, n_bins)``.
    C(l) is dimensionless for convergence.
    """
    maps = _as_batch(maps)
    n = maps.shape[-1]
    L = fov_deg * DEG2RAD
    pix = L / n
    ft = np.fft.fft2(maps) * pix**2
    power = (ft.real**2 + ft.imag**2) / L**2

    freq = np.fft.fftfreq(n, d=pix) * 2 * np.pi
    lx, ly = np.meshgrid(freq, freq, indexing="ij")
    ell = np.sqrt(lx**2 + ly**2).ravel()
    edges = np.logspace(np.log10(2 * np.pi / L), np.log10(np.pi / pix), n_bins + 1)
    idx = np.digitize(ell, edges) - 1
    idx[ell == edges[-1]] = n_bins - 1
    valid = (idx >= 0) & (idx < n_bins)
    counts = np.bincount(idx[valid], minlength=n_bins).astype(np.float64)

    p = power.reshape(len(maps), -1)[:, valid].astype(np.float64)
    sums = np.zeros((len(maps), n_bins))
    np.add.at(sums.T, idx[valid], p.T)
    return np.sqrt(edges[:-1] * edges[1:]), sums / counts


# ---------------------------------------------------------------- smoothing --

def fourier_smooth(maps: np.ndarray, theta_arcmin: float, fov_deg: float = 5.0) -> np.ndarray:
    """Periodic Gaussian smoothing at ``theta_arcmin`` (the kernel *sigma*),
    applied in Fourier space with an untruncated kernel — the convention the
    emulator's training statistics were computed with."""
    maps = _as_batch(maps)
    if theta_arcmin <= 0:
        return maps
    n = maps.shape[-1]
    sigma_pix = theta_arcmin / (fov_deg * 60.0 / n)
    k = np.fft.fftfreq(n) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kernel = np.exp(-0.5 * sigma_pix**2 * (kx**2 + ky**2))
    return np.fft.ifft2(np.fft.fft2(maps) * kernel).real


# ------------------------------------------------------------- peaks/minima --

def _neighbor_extreme(field: np.ndarray, kind: str) -> np.ndarray:
    """Max (or min) over the 8 neighbors, replicate ("nearest") padding."""
    p = np.pad(field, ((0, 0), (1, 1), (1, 1)), mode="edge")
    n = field.shape[-1]
    shifts = [p[:, dy:dy + n, dx:dx + n]
              for dy in (0, 1, 2) for dx in (0, 1, 2) if not (dy == 1 and dx == 1)]
    stack = np.stack(shifts)
    return stack.max(0) if kind == "max" else stack.min(0)


def extrema_histogram(field: np.ndarray, edges: np.ndarray, kind: str) -> np.ndarray:
    """Histogram of strict local-maximum ("max") or local-minimum ("min")
    values, border pixels excluded. field (N, n, n) -> counts (N, nbins)."""
    field = _as_batch(field)
    neigh = _neighbor_extreme(field, kind)
    mask = field > neigh if kind == "max" else field < neigh
    mask[:, 0, :] = mask[:, -1, :] = False
    mask[:, :, 0] = mask[:, :, -1] = False
    sentinel = edges[0] - 1.0
    vals = np.where(mask, field, sentinel)
    return _histogram_batch(vals.reshape(len(field), -1), edges)


# ----------------------------------------------------------------- Minkowski --

def minkowski(field: np.ndarray, thresholds: np.ndarray):
    """Minkowski functionals V0 (area), V1 (boundary), V2 (Euler) vs threshold
    for uniformly spaced ``thresholds`` (Schmalzing-Buchert delta-function
    estimator). field (N, n, n), typically the smoothed S/N map."""
    field = _as_batch(field)
    N, n, _ = field.shape
    npix = n * n
    u = field.astype(np.float64)
    gy, gx = np.gradient(u, axis=(-2, -1))
    gyy, _ = np.gradient(gy, axis=(-2, -1))
    gxy, gxx = np.gradient(gx, axis=(-2, -1))
    grad = np.sqrt(gx**2 + gy**2)
    curv = (2 * gx * gy * gxy - gx**2 * gyy - gy**2 * gxx) / (gx**2 + gy**2 + 1e-12)

    thr = np.asarray(thresholds, np.float64)
    dnu = np.gradient(thr)
    if not np.allclose(dnu, dnu[0]):
        raise ValueError("minkowski assumes uniformly spaced thresholds")
    half = 0.5 * dnu[0]

    u_sorted = np.sort(u.reshape(N, -1), axis=1)
    below = np.stack([np.searchsorted(u_sorted[i], thr) for i in range(N)])
    V0 = 1.0 - below / npix

    edges = np.concatenate([thr - half, thr[-1:] + half])
    uflat = u.reshape(N, -1)
    norm = npix * dnu[0]
    V1 = _weighted_histogram_batch(uflat, (0.25 * grad).reshape(N, -1), edges) / norm
    V2 = _weighted_histogram_batch(uflat, (curv / (2 * np.pi)).reshape(N, -1), edges) / norm
    return V0, V1, V2


# ---------------------------------------------------------------- scattering --

def _scattering_bank(n: int, J: int, L: int):
    """Oriented multi-scale Gabor-like band-pass bank in Fourier space, plus a
    Gaussian low-pass. Octave bands from ~Nyquist down by ``J`` scales, ``L``
    orientations."""
    k = np.fft.fftfreq(n) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    psis, metas = [], []
    for j in range(J):
        k0 = np.pi / 2.0 ** (j + 1)
        sig = k0 / 2.0
        for ell in range(L):
            theta = np.pi * ell / L
            kr = np.cos(theta) * KX + np.sin(theta) * KY
            kp = -np.sin(theta) * KX + np.cos(theta) * KY
            psis.append(np.exp(-((kr - k0) ** 2 + kp**2) / (2 * sig**2)))
            metas.append((j, ell))
    phi = np.exp(-(KX**2 + KY**2) / (2 * (np.pi / 2.0 ** (J + 1)) ** 2))
    return np.stack(psis), phi, metas


_SCAT_CACHE: dict = {}


def scattering(maps: np.ndarray, J: int = 4, L: int = 4) -> np.ndarray:
    """Translation-invariant wavelet-scattering coefficients of ``(N, n, n)``
    maps: S0 (low-pass mean), S1[j,l], and S2[(j1,l1),(j2,l2)] for j2 > j1.
    Returns ``(N, n_coeff)`` (n_coeff = 113 for J=L=4)."""
    maps = _as_batch(maps)
    n = maps.shape[-1]
    key = (n, J, L)
    if key not in _SCAT_CACHE:
        _SCAT_CACHE[key] = _scattering_bank(n, J, L)
    psis, phi, metas = _SCAT_CACHE[key]
    pairs = [(i1, i2) for i1, (j1, _) in enumerate(metas)
             for i2, (j2, _) in enumerate(metas) if j2 > j1]

    x_hat = np.fft.fft2(maps)
    out = [np.abs(np.fft.ifft2(x_hat * phi)).mean(axis=(-2, -1))]
    m1_hats = []
    for psi in psis:
        m1 = np.abs(np.fft.ifft2(x_hat * psi))
        out.append(m1.mean(axis=(-2, -1)))
        m1_hats.append(np.fft.fft2(m1))
    for i1, i2 in pairs:
        out.append(np.abs(np.fft.ifft2(m1_hats[i1] * psis[i2])).mean(axis=(-2, -1)))
    return np.stack(out, axis=1)


# --------------------------------------------------------------- full bundle --

def measure_stats(maps: np.ndarray, fov_deg: float = 5.0,
                  smoothing_arcmin: float = 2.0, scat_res: int = 256,
                  scat_J: int = 4, scat_L: int = 4) -> dict:
    """All statistic blocks for raw convergence maps ``(N, n, n)``.

    Returns a dict with one ``(N, D)`` array per block in
    :data:`BLOCKS` plus the bin coordinates ``ell``, ``pdf_x``, ``peak_x``,
    ``mink_thr``. With the default arguments this reproduces the emulator's
    training statistics, so the output is directly comparable to
    :meth:`bind.wlemu.WLEmulator.predict`.
    """
    maps = _as_batch(maps).astype(np.float64)
    b = default_bins()
    N = len(maps)
    out: dict = {}

    ell, Cl = power_spectrum(maps, fov_deg, b["ell_bins"])
    out["ell"], out["Cl"] = ell, Cl

    sm = fourier_smooth(maps, smoothing_arcmin, fov_deg)
    mu_r, sd_r = maps.mean(axis=(-2, -1)), maps.std(axis=(-2, -1), ddof=1)
    mu_s, sd_s = sm.mean(axis=(-2, -1)), sm.std(axis=(-2, -1), ddof=1)
    out["moments"] = np.stack([mu_r, sd_r, mu_s, sd_s], axis=1)

    nu = (sm - mu_s[:, None, None]) / (sd_s[:, None, None] + 1e-12)

    pdf_counts = _histogram_batch(nu.reshape(N, -1), b["pdf_edges"])
    width = b["pdf_edges"][1] - b["pdf_edges"][0]
    out["pdf"] = pdf_counts / (pdf_counts.sum(axis=1, keepdims=True) * width)
    out["pdf_x"] = 0.5 * (b["pdf_edges"][:-1] + b["pdf_edges"][1:])

    out["peak"] = extrema_histogram(nu, b["peak_edges"], "max")
    out["min"] = extrema_histogram(nu, b["peak_edges"], "min")
    out["peak_x"] = 0.5 * (b["peak_edges"][:-1] + b["peak_edges"][1:])

    V0, V1, V2 = minkowski(nu, b["mink_thr"])
    out["V0"], out["V1"], out["V2"] = V0, V1, V2
    out["mink_thr"] = b["mink_thr"]

    out["scat"] = scattering(box_downsample(maps, scat_res), scat_J, scat_L)
    return out


def stats_vector(stats: dict) -> np.ndarray:
    """Concatenate a :func:`measure_stats` dict into the emulator's ``(N, D)``
    statistics vector (block order = :data:`BLOCKS`)."""
    return np.concatenate([np.atleast_2d(stats[k]) for k in BLOCKS], axis=1)
