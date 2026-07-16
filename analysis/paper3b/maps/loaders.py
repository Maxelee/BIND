"""WP-B1 data-side loaders: DES Y3 mass maps, ACT DR6 y-map products, DR5 clusters.

Popeye-resident paths (SHARED_CONTEXT §1b: B1–B4 run where the survey maps
live); every loader takes the path as an argument with the canonical download
location (B/wp1 `provenance.json`) as default, so tests can feed synthetic
files. healpy/pixell/astropy imports are kept inside functions where practical
— the peak logic itself (peaks.py) is testable on synthetic arrays without any
survey file.

Facts locked by the session-2/3 audits (B/wp1 REPORT):
* DES maps (Jeffrey et al. 2105.13539): HEALPix Nside=1024 RING, one map per
  variant. `wiener_full` is nonzero over the FULL sky (Wiener reconstruction
  leakage outside the footprint at ~8x lower rms — audited 2026-07-16), so the
  footprint must always come from the release mask; `glimpse_full` marks
  off-footprint pixels with a ~-1.6e30 sentinel. `glimpse_mask` (0/1 int) is
  the release footprint mask: 11.49% of sky = 4742 deg^2.
* ACT DR6+Planck y map (Coulton et al. 2307.01258): CAR (pixell), shape
  (10320, 43200), 0.5' pixels; `wide_mask_GAL070_apod_1.50_deg_wExtended` is
  the apodized analysis mask in [0, 1] (~49.5% sky).
* ACT DR5 SZ catalog (Hilton et al. 2009.11043) v1.1: 4195 clusters, SNR >= 4,
  z in [0.035, 1.91]; `y_c`/`fixed_y_c` are central Comptonization in 1e-4
  units (catalog convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DOWNLOADS = Path("/mnt/home/mlee1/ceph/paper3/B/downloads")
DES_DIR = DOWNLOADS / "des_y3_massmaps_jeffrey2105.13539"
ACT_YMAP_DIR = DOWNLOADS / "act_dr6_planck_ymap"
DR5_DIR = DOWNLOADS / "act_dr5_szcluster_hilton2009.11043"

DES_NSIDE = 1024
DES_VARIANTS = ("wiener", "glimpse")
GLIMPSE_SENTINEL_THRESHOLD = -1e20  # off-footprint marker in glimpse_full
DR5_YC_UNIT = 1e-4                  # y_c / fixed_y_c catalog columns are in 1e-4


def load_des_mask(des_dir: Path = DES_DIR) -> np.ndarray:
    """DES Y3 release footprint mask (bool, Nside=1024 RING).

    The release ships one mask (`glimpse_mask.fits`); it is the footprint for
    BOTH variants (the Wiener file has no embedded mask — see module notes).
    """
    import healpy as hp

    return hp.read_map(str(Path(des_dir) / "glimpse_mask.fits")).astype(bool)


def load_des_map(variant: str, tomo: int | None = None, des_dir: Path = DES_DIR,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """One DES Y3 mass-map variant (Nside=1024 RING), off-footprint pixels -> 0.

    ``variant`` in {"wiener", "glimpse"}; ``tomo`` None for the full map or
    1–4 for a tomographic bin. GLIMPSE sentinel values and everything outside
    the release mask are zeroed so downstream masked-smoothing (PEAK_DEFINITION
    §1) sees clean zeros, never ~-1e30.
    """
    import healpy as hp

    if variant not in DES_VARIANTS:
        raise ValueError(f"variant {variant!r} not in {DES_VARIANTS}")
    stem = f"{variant}_full" if tomo is None else f"{variant}_tomo{int(tomo)}"
    m = hp.read_map(str(Path(des_dir) / f"{stem}.fits")).astype(np.float64)
    if mask is None:
        mask = load_des_mask(des_dir)
    m[~mask] = 0.0
    m[m < GLIMPSE_SENTINEL_THRESHOLD] = 0.0  # belt+braces vs sentinel inside mask
    return m


def load_act_ymap(ymap_dir: Path = ACT_YMAP_DIR):
    """ACT DR6+Planck Compton-y map as a pixell enmap (CAR, 0.5' pixels)."""
    from pixell import enmap

    return enmap.read_map(str(Path(ymap_dir) / "ilc_actplanck_ymap.fits"))


def load_act_mask(ymap_dir: Path = ACT_YMAP_DIR):
    """ACT DR6 apodized analysis mask (CAR, values in [0, 1])."""
    from pixell import enmap

    return enmap.read_map(
        str(Path(ymap_dir) / "wide_mask_GAL070_apod_1.50_deg_wExtended.fits"))


@dataclass
class DR5Catalog:
    """ACT DR5 SZ clusters: positions [deg], SNR, z, central y (dimensionless)."""

    ra_deg: np.ndarray
    dec_deg: np.ndarray
    snr: np.ndarray
    redshift: np.ndarray
    y_c: np.ndarray        # central Comptonization, DIMENSIONLESS (unit applied)
    fixed_y_c: np.ndarray  # same, at the 2.4' fixed filter scale

    def __len__(self) -> int:
        return len(self.ra_deg)

    def select(self, keep: np.ndarray) -> "DR5Catalog":
        return DR5Catalog(*(getattr(self, f)[keep] for f in
                            ("ra_deg", "dec_deg", "snr", "redshift", "y_c", "fixed_y_c")))


def load_dr5_catalog(dr5_dir: Path = DR5_DIR, snr_min: float | None = None) -> DR5Catalog:
    """DR5 cluster catalog v1.1; y_c columns converted from 1e-4 units."""
    from astropy.io import fits

    with fits.open(str(Path(dr5_dir) / "DR5_cluster-catalog_v1.1.fits")) as h:
        d = h[1].data
        cat = DR5Catalog(
            ra_deg=np.asarray(d["RADeg"], dtype=np.float64),
            dec_deg=np.asarray(d["decDeg"], dtype=np.float64),
            snr=np.asarray(d["SNR"], dtype=np.float64),
            redshift=np.asarray(d["redshift"], dtype=np.float64),
            y_c=np.asarray(d["y_c"], dtype=np.float64) * DR5_YC_UNIT,
            fixed_y_c=np.asarray(d["fixed_y_c"], dtype=np.float64) * DR5_YC_UNIT,
        )
    if snr_min is not None:
        cat = cat.select(cat.snr >= snr_min)
    return cat
