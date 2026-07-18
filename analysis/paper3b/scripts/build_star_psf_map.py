"""WP-B3 §7 prep: HEALPix star-density + PSF-residual hotspot maps.

Reads the public DES Y3 PIFF reserved-star diagnostics catalog
(`psf_y3a1-v29.fits`, 56.7M rows: ra, dec, obs_{e1,e2,T}, piff_{e1,e2,T}, ...)
and bins two covariates to nside=1024 (matching the common footprint):

  star_count   — reserved-star number density per pixel (the literal
                 "star hotspot" map).
  psf_dresid   — mean sqrt((obs_e1-piff_e1)^2 + (obs_e2-piff_e2)^2) per pixel
                 (the PSF-modeling-residual ellipticity amplitude; the
                 "PSF-error hotspot" map — large where PIFF fits poorly,
                 e.g. near bright stars/saturation/CCD edges).

Output: wp3_nulls/star_psf_map_nside1024.npz (star_count, psf_dresid,
psf_dT_frac, n_stars_per_pix, nside, n_stars_total).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PSF_CAT = Path("/mnt/home/mlee1/ceph/paper3/B/downloads/des_y3_psf_y3a1v29/psf_y3a1-v29.fits")
OUT = Path("/mnt/home/mlee1/ceph/paper3/B/wp3_nulls")
NSIDE = 1024


def main() -> None:
    import healpy as hp
    from astropy.io import fits

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[star-psf] reading {PSF_CAT} ...", flush=True)
    with fits.open(PSF_CAT, memmap=True) as hdul:
        d = hdul[1].data
        ra = np.asarray(d["ra"], dtype=np.float64)
        dec = np.asarray(d["dec"], dtype=np.float64)
        de1 = np.asarray(d["obs_e1"], dtype=np.float64) - np.asarray(d["piff_e1"], dtype=np.float64)
        de2 = np.asarray(d["obs_e2"], dtype=np.float64) - np.asarray(d["piff_e2"], dtype=np.float64)
        obs_T = np.asarray(d["obs_T"], dtype=np.float64)
        piff_T = np.asarray(d["piff_T"], dtype=np.float64)
    n = ra.size
    print(f"[star-psf] {n} reserved stars", flush=True)

    good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(de1) & np.isfinite(de2) \
        & np.isfinite(obs_T) & (obs_T > 0) & np.isfinite(piff_T)
    ra, dec, de1, de2 = ra[good], dec[good], de1[good], de2[good]
    obs_T, piff_T = obs_T[good], piff_T[good]
    dresid = np.hypot(de1, de2)
    dT_frac = (obs_T - piff_T) / obs_T
    print(f"[star-psf] {good.sum()} finite rows kept", flush=True)

    pix = hp.ang2pix(NSIDE, np.radians(90.0 - dec), np.radians(ra % 360.0))
    npix = hp.nside2npix(NSIDE)
    star_count = np.bincount(pix, minlength=npix).astype(np.float64)
    dresid_sum = np.bincount(pix, weights=dresid, minlength=npix)
    dT_sum = np.bincount(pix, weights=np.abs(dT_frac), minlength=npix)

    with np.errstate(invalid="ignore", divide="ignore"):
        psf_dresid = np.where(star_count > 0, dresid_sum / np.maximum(star_count, 1), np.nan)
        psf_dT_frac = np.where(star_count > 0, dT_sum / np.maximum(star_count, 1), np.nan)

    np.savez_compressed(
        OUT / "star_psf_map_nside1024.npz",
        nside=NSIDE, star_count=star_count.astype(np.float32),
        psf_dresid=psf_dresid.astype(np.float32),
        psf_dT_frac=psf_dT_frac.astype(np.float32),
        n_stars_total=int(good.sum()),
        source="despublic/y3a2_files/psf/psf_y3a1-v29.fits",
    )
    occ = int((star_count > 0).sum())
    print(f"[star-psf] wrote {OUT}/star_psf_map_nside1024.npz "
          f"({occ} occupied pixels of {npix})", flush=True)


if __name__ == "__main__":
    main()
