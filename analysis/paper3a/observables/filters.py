"""Aperture filters and beam convolution on flat 2D patches (WP-A2).

Implements the exact filters the WP-A1 data vectors were measured with:

- **CAP** (compensated aperture photometry, Schaan et al. 2021 convention,
  used by all four frozen kSZ vectors): disk of radius theta_d minus the
  equal-area annulus theta_d < theta < sqrt(2) theta_d. A signal confined
  well inside theta_d passes through with unit weight; any uniform
  background (CMB, mean map level) cancels identically.
- **Top-hat disk** aperture (cylindrical masses for f_gas, Y500).
- **Gaussian beam** convolution (zero-padded FFT, not periodic — patches are
  not periodic objects once cut from a composite).

Pixelization convention: weight maps are anti-aliased by supersampling each
pixel `supersample x supersample` (default 8 -> boundary-pixel area errors
< ~1%, and exactly compensated between disk and annulus in the CAP case).
All functions take radii in *pixels*; use `PatchGeometry` to get there from
arcmin or h^-1 Mpc.
"""

from __future__ import annotations

import numpy as np


def _radius_grid(shape: tuple[int, int], center: tuple[float, float], supersample: int = 1) -> np.ndarray:
    """Distance (in coarse-pixel units) of every (super)pixel center from `center`.

    center is in pixel coordinates of the coarse grid, (row, col), where
    integer values sit at pixel centers.
    """
    ny, nx = shape
    s = supersample
    # Supersampled pixel centers: pixel i spans [i-0.5, i+0.5); subcell k of s
    # sits at i - 0.5 + (k + 0.5)/s.
    ys = (np.arange(ny * s) + 0.5) / s - 0.5 - center[0]
    xs = (np.arange(nx * s) + 0.5) / s - 0.5 - center[1]
    return np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)


def _downsample_mean(fine: np.ndarray, supersample: int) -> np.ndarray:
    s = supersample
    ny, nx = fine.shape[0] // s, fine.shape[1] // s
    return fine.reshape(ny, s, nx, s).mean(axis=(1, 3))


def disk_weight_map(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius_pix: float,
    supersample: int = 8,
) -> np.ndarray:
    """Anti-aliased top-hat disk: per-pixel covered-area fraction in [0, 1]."""
    if radius_pix <= 0:
        raise ValueError(f"radius_pix must be > 0, got {radius_pix}")
    r = _radius_grid(shape, center, supersample)
    return _downsample_mean((r <= radius_pix).astype(np.float64), supersample)


def cap_weight_map(
    shape: tuple[int, int],
    center: tuple[float, float],
    theta_d_pix: float,
    supersample: int = 8,
) -> np.ndarray:
    """CAP filter weights: +1 inside theta_d, -1 in the sqrt(2) annulus, anti-aliased.

    The disk and annulus have equal area analytically; on a pixel grid they
    differ by O(1) px^2, which would leak a uniform background through the
    filter. The annulus weight is therefore renormalized to the pixelized
    disk/annulus area ratio so the weights sum to exactly zero and constant
    offsets cancel identically (the compensation property the filter exists
    for), at the cost of a <~1% reweighting of the annulus.
    """
    if theta_d_pix <= 0:
        raise ValueError(f"theta_d_pix must be > 0, got {theta_d_pix}")
    r = _radius_grid(shape, center, supersample)
    disk = r <= theta_d_pix
    annulus = (r > theta_d_pix) & (r <= np.sqrt(2.0) * theta_d_pix)
    n_ann = annulus.sum()
    if n_ann == 0:
        raise ValueError(f"theta_d_pix={theta_d_pix} annulus contains no (sub)pixels")
    fine = np.zeros(r.shape, dtype=np.float64)
    fine[disk] = 1.0
    fine[annulus] = -disk.sum() / n_ann
    return _downsample_mean(fine, supersample)


def require_fits_in_patch(shape: tuple[int, int], center: tuple[float, float], outer_radius_pix: float) -> None:
    """Raise unless a circle of outer_radius_pix around center fits inside the map.

    This is the guard for the WP-A2 patch-size finding: at LRG redshifts the
    CAP outer annulus (sqrt(2) x theta_d) of the largest data-vector radii
    subtends more than the 6.25 h^-1 Mpc BIND core patch — those radii must
    be measured on composite-map cutouts, and silently truncating the annulus
    would bias the profile low. Fail loudly instead.
    """
    ny, nx = shape
    cy, cx = center
    margin = min(cy, cx, ny - 1 - cy, nx - 1 - cx)
    if outer_radius_pix > margin:
        raise ValueError(
            f"aperture outer radius {outer_radius_pix:.1f} px exceeds the "
            f"{margin:.1f} px margin from center {center} to the edge of a "
            f"{shape} map — measure this radius on a larger cutout "
            "(composite map), do not truncate the aperture"
        )


def cap_photometry(
    map2d: np.ndarray,
    center: tuple[float, float],
    theta_d_pix: np.ndarray,
    pixel_area: float = 1.0,
    supersample: int = 8,
) -> np.ndarray:
    """CAP-filtered aperture photometry at each disk radius.

    Returns sum(map * cap_weights) * pixel_area for each theta_d, i.e. the
    filtered signal in (map units) x (pixel_area units). Pass
    ``pixel_area=geometry.pixel_area_arcmin2()`` to match the muK arcmin^2 /
    tau arcmin^2 convention of the frozen kSZ vectors.
    """
    map2d = np.asarray(map2d, dtype=np.float64)
    theta_d_pix = np.atleast_1d(np.asarray(theta_d_pix, dtype=float))
    out = np.empty(theta_d_pix.shape, dtype=np.float64)
    for i, td in enumerate(theta_d_pix):
        require_fits_in_patch(map2d.shape, center, np.sqrt(2.0) * td)
        w = cap_weight_map(map2d.shape, center, td, supersample)
        out[i] = float((map2d * w).sum()) * pixel_area
    return out


def aperture_sum(
    map2d: np.ndarray,
    center: tuple[float, float],
    radius_pix: float,
    supersample: int = 8,
) -> float:
    """Anti-aliased sum of map values inside a disk (cylindrical aperture).

    Returns sum(map * disk_fraction) — for a surface-mass-density map in
    Msun/h per pixel this is directly the cylindrical aperture mass.
    """
    require_fits_in_patch(np.asarray(map2d).shape, center, radius_pix)
    w = disk_weight_map(np.asarray(map2d).shape, center, radius_pix, supersample)
    return float((np.asarray(map2d, dtype=np.float64) * w).sum())


def gaussian_beam_convolve(map2d: np.ndarray, fwhm_pix: float) -> np.ndarray:
    """Convolve with a Gaussian beam via zero-padded FFT (non-periodic).

    fwhm_pix <= 0 is a no-op (returns a float64 copy), so operator configs
    can express "no beam" without branching at every call site.
    """
    map2d = np.asarray(map2d, dtype=np.float64)
    if fwhm_pix <= 0:
        return map2d.copy()
    sigma = fwhm_pix / np.sqrt(8.0 * np.log(2.0))
    pad = int(np.ceil(5.0 * sigma))
    padded = np.pad(map2d, pad, mode="constant")
    ny, nx = padded.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    beam_ft = np.exp(-2.0 * np.pi**2 * sigma**2 * (ky**2 + kx**2))
    smoothed = np.fft.ifft2(np.fft.fft2(padded) * beam_ft).real
    return smoothed[pad : pad + map2d.shape[0], pad : pad + map2d.shape[1]]
