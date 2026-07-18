"""WP-B5 interpretation-ladder item 7: DES shear multiplicative-bias (m) propagation.

Analytic closure + numerical demonstration that a scale-independent
multiplicative shear bias CANCELS EXACTLY through the entire frozen
matched-nu chain, on both sides:

* **Data side**: the Wiener reconstruction is linear in shear, so a residual
  bias rescales the kappa map by (1+m). The frozen chain
  (PEAK_DEFINITION.md) is mask -> harmonic smoothing (linear) -> local
  maxima (invariant under positive scaling) -> nu = (kappa_sm - mean)/std
  over the footprint (``nu_norm="map"``, an affine self-normalization that
  removes any positive scale factor exactly). Peak SET, positions, and nu
  values are all invariant; the y side never sees m. The frozen data vector
  is unchanged.
* **Model side**: m enters the mocks ONLY through the empirical transfer
  T(ell) = sqrt(C_data/C_mock) -> (1+m) T. ``measure_mock_patch`` applies T
  to the NOISY kappa (signal+noise together), then smooths and
  self-normalizes nu on the patch itself — the same positive scale factor
  cancels identically; mock peak sets, per-peak nu and per-peak y are
  unchanged. The abundance channel (matched-nu counts/deg^2) is invariant
  on both sides for the same reason.

NOT covered (and out of scope for this rung): scale-DEPENDENT m(ell) —
shear calibration bias is scale-independent at leading order, and
PSF-driven scale-dependent residuals are the object of the B3 star/PSF
battery (section 7, PASS) — and additive biases (also B3 territory).

The demo uses |m| = 0.03, deliberately several times larger than the DES Y3
per-bin m-prior widths, so the exact-zero result is not a small-parameter
accident. Job-free, CPU-light (one extra SHT at Nside=1024).

Outputs -> /mnt/home/mlee1/ceph/paper3/B/wp5_inference/mbias/mbias_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(_repo))

from analysis.paper3b.maps.loaders import load_des_map  # noqa: E402
from analysis.paper3b.maps.peaks import build_peak_catalog  # noqa: E402
from analysis.paper3b.mocks.measure import measure_mock_patch  # noqa: E402
from analysis.paper3b.mocks.nz import SourcePlaneWeighting  # noqa: E402
from analysis.paper3b.mocks.patch import PatchGeometry  # noqa: E402
from analysis.paper3b.mocks.shape_noise import (  # noqa: E402
    DESY3_NEFF_TOTAL_ARCMIN2,
    DESY3_SIGMA_E,
    ShapeNoiseConfig,
)
from analysis.paper3b.mocks.smoothing import (  # noqa: E402
    FIDUCIAL_SMOOTHING_ARCMIN,
    SmoothingConfig,
)
from analysis.paper3b.mocks.beam import ACT_DR6_YMAP_FWHM_ARCMIN, BeamConfig  # noqa: E402
from analysis.paper3b.mocks.transfer import TransferFunction, load_transfer  # noqa: E402

B_ROOT = Path("/mnt/home/mlee1/ceph/paper3/B")
OUT = B_ROOT / "wp5_inference" / "mbias"
FOOTPRINT = B_ROOT / "wp1_maps" / "common_footprint_nside1024.npz"
TRANSFER_NPZ = B_ROOT / "wp4_mocks" / "transfer_desy3.npz"
M_BIAS = 0.03


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"m_bias_demo": M_BIAS}

    # ---- data side: catalog on m and on (1+m)*m ----------------------------
    fp = np.load(FOOTPRINT)
    wmask, bmask = fp["weight"], fp["binary"].astype(bool)
    kappa = load_des_map("wiener")
    cat0 = build_peak_catalog(kappa, wmask, bmask,
                              FIDUCIAL_SMOOTHING_ARCMIN, "wiener")
    cat1 = build_peak_catalog((1.0 + M_BIAS) * kappa, wmask, bmask,
                              FIDUCIAL_SMOOTHING_ARCMIN, "wiener_m")
    same_set = (cat0.ipix.size == cat1.ipix.size
                and np.array_equal(np.sort(cat0.ipix), np.sort(cat1.ipix)))
    dnu = (np.max(np.abs(cat1.nu[np.argsort(cat1.ipix)]
                         - cat0.nu[np.argsort(cat0.ipix)]))
           if same_set and cat0.ipix.size else np.nan)
    summary["data_side"] = {
        "n_peaks": int(cat0.ipix.size),
        "identical_peak_set": bool(same_set),
        "max_abs_delta_nu": float(dnu),
    }
    print(f"[data] n_peaks={cat0.ipix.size}  identical set: {same_set}  "
          f"max|dnu| = {dnu:.3e}")

    # ---- model side: one mock patch with T vs (1+m)*T, same rng ------------
    geom = PatchGeometry(fov_deg=5.0, npix=512)
    rng_field = np.random.default_rng(20260718)
    planes = rng_field.normal(0.0, 0.02, size=(5, geom.npix, geom.npix))
    planes = np.cumsum(planes, axis=0)          # crudely nested source planes
    ymap = np.abs(rng_field.normal(1e-6, 1e-6, size=(geom.npix, geom.npix)))
    weighting = SourcePlaneWeighting(np.array([0.5, 1.0, 1.5, 2.0, 2.44]),
                                     np.full(5, 0.2))
    noise_cfg = ShapeNoiseConfig(DESY3_NEFF_TOTAL_ARCMIN2,
                                 geom.pixel_area_arcmin2(), DESY3_SIGMA_E)
    smooth_cfg = SmoothingConfig(FIDUCIAL_SMOOTHING_ARCMIN,
                                 geom.arcmin_per_pixel(), "wrap")
    beam_cfg = BeamConfig(ACT_DR6_YMAP_FWHM_ARCMIN, geom.arcmin_per_pixel(),
                          "wrap")
    t0 = load_transfer(TRANSFER_NPZ, "wiener")
    t1 = TransferFunction(t0.ell, (1.0 + M_BIAS) * t0.t, t0.ell_zero)
    res = []
    for t in (t0, t1):
        rng = np.random.default_rng((20260718, 0, 0, 0))
        res.append(measure_mock_patch(planes, ymap, weighting, noise_cfg,
                                      smooth_cfg, beam_cfg, rng, geom=geom,
                                      transfer=t))
    same_n = res[0].per_peak_nu.size == res[1].per_peak_nu.size
    dnu_m = (np.max(np.abs(res[0].per_peak_nu - res[1].per_peak_nu))
             if same_n and res[0].per_peak_nu.size else np.nan)
    dy_m = (np.max(np.abs(res[0].per_peak_y - res[1].per_peak_y))
            if same_n and res[0].per_peak_y.size else np.nan)
    summary["model_side"] = {
        "n_peaks": int(res[0].per_peak_nu.size),
        "same_n_peaks": bool(same_n),
        "max_abs_delta_nu": float(dnu_m),
        "max_abs_delta_per_peak_y": float(dy_m),
    }
    print(f"[mock] n_peaks={res[0].per_peak_nu.size}  same count: {same_n}  "
          f"max|dnu| = {dnu_m:.3e}  max|dY| = {dy_m:.3e}")

    verdict = (same_set and same_n and dnu < 1e-6 and dnu_m < 1e-9
               and dy_m < 1e-12)
    summary["verdict"] = ("NULL-EXACT: scale-independent m cancels through "
                          "the matched-nu chain on both sides"
                          if verdict else "UNEXPECTED - inspect")
    print(f"verdict: {summary['verdict']}")
    with open(OUT / "mbias_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {OUT/'mbias_summary.json'}")


if __name__ == "__main__":
    main()
