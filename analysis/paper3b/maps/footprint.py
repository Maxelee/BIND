"""WP-B1 task 3 — footprint harmonization: DES (HEALPix) x ACT y mask (CAR).

Common analysis grid = HEALPix Nside=1024 RING (the DES maps' native grid; the
y map itself stays at native CAR for profile/stack work per the plan — only its
MASK is brought over here). Products:

* ``act_mask_to_healpix``  — the ACT apodized mask sampled onto HEALPix by
  bilinear interpolation (``pixell.reproject.map2healpix(method="spline",
  order=1)``); harmonic resampling would ring on a mask.
* ``harmonize``            — the three mask products B1/B2 consume:
    - ``weight``  (float in [0,1]): DES footprint x ACT apodization — the
      stacking weight;
    - ``binary``  (bool): DES & (ACT >= act_threshold) — the conservative
      common footprint (default threshold 0.99 keeps only the fully
      un-apodized ACT interior);
    - sky fractions of every ingredient/combination, for the REPORT.

Sky fraction convention: fraction of the FULL sphere (f_sky), and deg^2 via
41252.96 deg^2. All fractions are mean(mask) for binary, mean(weight) for
apodized (i.e. effective f_sky).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FULL_SKY_DEG2 = 41252.96
DEFAULT_ACT_THRESHOLD = 0.99


def act_mask_to_healpix(act_mask, nside: int = 1024) -> np.ndarray:
    """Sample a CAR (pixell) mask onto HEALPix RING by bilinear interpolation.

    Values are clipped to [0, 1] (bilinear can over/undershoot at apodization
    edges by float error). Pixels outside the CAR map's declination coverage
    sample to 0 (pixell extends with 0 for order=1 sampling here).
    """
    from pixell import reproject

    hp_mask = reproject.map2healpix(act_mask, nside=nside, method="spline", order=1)
    return np.clip(np.asarray(hp_mask, dtype=np.float64), 0.0, 1.0)


@dataclass
class HarmonizedFootprint:
    """Common-footprint products + the sky-fraction bookkeeping table."""

    nside: int
    weight: np.ndarray          # (npix,) float [0,1]: DES bool x ACT apod
    binary: np.ndarray          # (npix,) bool: DES & (ACT >= act_threshold)
    act_threshold: float
    sky_fractions: dict = field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        return [f"{k}: f_sky={v:.4f} ({v * FULL_SKY_DEG2:.0f} deg^2)"
                for k, v in self.sky_fractions.items()]


def harmonize(des_mask: np.ndarray, act_mask_hp: np.ndarray,
              act_threshold: float = DEFAULT_ACT_THRESHOLD) -> HarmonizedFootprint:
    """Build the DES x ACT common-footprint products on the HEALPix grid."""
    import healpy as hp

    des = np.asarray(des_mask, dtype=bool)
    act = np.asarray(act_mask_hp, dtype=np.float64)
    if des.shape != act.shape:
        raise ValueError(f"mask shapes differ: DES {des.shape} vs ACT {act.shape}")
    nside = hp.npix2nside(des.size)

    weight = des * act
    binary = des & (act >= act_threshold)
    fr = {
        "DES footprint (binary)": des.mean(),
        "ACT mask (effective, apodized)": act.mean(),
        f"ACT mask >= {act_threshold} (binary)": (act >= act_threshold).mean(),
        "common weight (DES x ACT apod, effective)": weight.mean(),
        f"common binary (DES & ACT >= {act_threshold})": binary.mean(),
    }
    return HarmonizedFootprint(nside=nside, weight=weight, binary=binary,
                               act_threshold=act_threshold, sky_fractions=fr)
