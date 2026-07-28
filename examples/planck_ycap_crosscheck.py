#!/usr/bin/env python
"""Round 2 track T2 (docs/paper_improvement_plan.md, B1): cross-check the
paper's ACT DR6+Planck ILC y-CAP measurement (examples/act_ycap_measure.py)
against a genuinely INDEPENDENT y-map — Planck 2015 MILCA all-sky Compton-y
(NSIDE=2048, ~10' FWHM effective beam) — stacked at the SAME 160,150 DESI
LRGs (z=0.4-0.6) used in the paper's real-data measurement
(``LC/T2f_lrg_z0406_cib1.7.npz``), at large (>=4') apertures where Planck's
coarse beam still resolves the compensated-aperture signal.

This is deliberately a LARGE-APERTURE consistency check, not a replacement
for the paper's small-aperture fit (theta<~2') — Planck's 10' beam cannot
resolve those scales. Footprint-correlated systematics common to both maps
(large-scale Galactic residuals, CIB leakage morphology at few-arcmin scale,
correlated LSS) cancel in the ACT-minus-Planck difference; genuinely
independent systematics (different frequency channels, different component
separation algorithms: ACT DR6 NILC-with-CIB-deprojection vs Planck MILCA)
do not.

Pipeline
--------
1. Extract ``milca_ymaps.fits`` (+ associated confidence mask) from the
   Planck 2015 SZ product tarball (downloaded once into
   ``DL/planck_ysz/``).
2. Planck side: per-object CAP via healpy ``query_disc`` — disk mean minus
   equal-area ring [theta_d, sqrt2*theta_d] mean, times the analytic disk
   area pi*theta_d^2 (arcmin^2) — the continuum limit of
   ``act_ycap_measure.cap_thumbs``' discretized disk_sum - scaled ring_sum
   formula (see module docstring there). Galactic/point-source masking via
   the Planck confidence mask, gated the same way as the ACT engine
   (mean mask over the full r<sqrt2*theta_d aperture > 0.99).
3. ACT side: beam-match by smoothing the deproj_cib_1.7 map from 1.6' to
   Planck's 10' (Gaussian FWHM sqrt(10^2-1.6^2)=9.87' in quadrature) via
   ``scipy.ndimage.gaussian_filter`` on postage-stamp thumbnails (NOT a
   whole-map FFT — the 43200x10320 map is ~1.8GB and a global harmonic
   transform risks the memory budget; local smoothing on padded thumbnails
   is numerically equivalent at these scales and far cheaper), then the
   SAME CAP filter (``act_ycap_measure.cap_thumbs``, reused unmodified) at
   the same 5 apertures.
4. Both sides: weighted stack + spatial jackknife (10x10 RA/Dec cells) via
   ``act_ycap_measure.stack_with_covariances`` (imported, not reimplemented)
   — identical recipe to the paper's own measurement.
5. Delta = CAP_ACT_smoothed - CAP_Planck; chi2 = Delta^T diag(sig_ACT^2 +
   sig_Planck^2)^-1 Delta over the 5 apertures (errors independent: distinct
   noise realizations/maps sharing only the sky signal and object
   positions — the difference test is standard for this kind of two-map
   consistency check); PTE via scipy.stats.chi2.sf.

This script is a NEW standalone engine (parallel to ``act_ycap_measure.py``,
which it imports as a module and does not modify) — not a notebook or
builder file.

Usage
-----
    python examples/planck_ycap_crosscheck.py                # run everything
    python examples/planck_ycap_crosscheck.py --step extract # just extraction
    python examples/planck_ycap_crosscheck.py --step measure # planck+act cap
    python examples/planck_ycap_crosscheck.py --step compare # chi2/pte/save
"""
from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import act_ycap_measure as aym  # noqa: E402  (reuse, do not edit)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DL = aym.DL
KS = aym.KS
LC = KS / "lightcone"
PLANCK_DIR = DL / "planck_ysz"
TGZ_URL = ("https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/"
           "maps/component-maps/foregrounds/COM_CompMap_YSZ_R2.00.fits.tgz")
TGZ = PLANCK_DIR / "COM_CompMap_YSZ_R2.00.fits.tgz"
CATALOG = LC / "T2f_lrg_z0406_cib1.7.npz"
OUT = LC / "planck_crosscheck.npz"

MILCA_PATH = PLANCK_DIR / "milca_ymaps.fits"
MASK_PATH_PLANCK = PLANCK_DIR / "planck_ysz_mask.fits"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THETA_ARCMIN = np.array([4.0, 5.0, 6.0, 8.0, 10.0])
SQ2 = np.sqrt(2.0)
PLANCK_FWHM_ARCMIN = 10.0
ACT_FWHM_ARCMIN = 1.6
SMOOTH_FWHM_ARCMIN = float(np.sqrt(PLANCK_FWHM_ARCMIN ** 2 - ACT_FWHM_ARCMIN ** 2))  # 9.871'
SIGMA_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # FWHM -> sigma

MASK_APERTURE_MIN = 0.99


# ---------------------------------------------------------------------------
# Step 1: download + extract
# ---------------------------------------------------------------------------

def ensure_download():
    PLANCK_DIR.mkdir(parents=True, exist_ok=True)
    if TGZ.exists():
        print(f"[planck] tgz already present: {TGZ} ({TGZ.stat().st_size/1e9:.2f} GB)")
        return
    print(f"[planck] downloading {TGZ_URL} -> {TGZ}")
    t0 = time.time()
    subprocess.run(["curl", "-sS", "--retry", "3", "--retry-delay", "5",
                     "-o", str(TGZ), TGZ_URL], check=True)
    print(f"[planck] download done in {time.time()-t0:.0f}s, "
          f"{TGZ.stat().st_size/1e9:.2f} GB")


def list_tgz_members():
    out = subprocess.run(["tar", "tzf", str(TGZ)], capture_output=True,
                          text=True, check=True)
    return out.stdout.splitlines()


def extract_maps():
    if MILCA_PATH.exists():
        print(f"[planck] already extracted: {MILCA_PATH}")
        return
    members = list_tgz_members()
    milca = [m for m in members if m.split("/")[-1] == "milca_ymaps.fits"]
    masks = [m for m in members if "mask" in m.lower() and m.endswith(".fits")]
    print(f"[planck] tgz has {len(members)} members; milca={milca}; "
          f"mask candidates={masks}")
    if not milca:
        raise SystemExit("[planck] milca_ymaps.fits not found in tarball listing "
                          "-- inspect members manually before proceeding.")
    need = milca + masks
    subprocess.run(["tar", "xzf", str(TGZ), *need, "-C", str(PLANCK_DIR)],
                    check=True)
    # tar preserves internal directory structure; move to flat expected paths.
    extracted_milca = PLANCK_DIR / milca[0]
    if extracted_milca != MILCA_PATH:
        extracted_milca.rename(MILCA_PATH)
    if masks:
        extracted_mask = PLANCK_DIR / masks[0]
        if extracted_mask.exists() and extracted_mask != MASK_PATH_PLANCK:
            extracted_mask.rename(MASK_PATH_PLANCK)
    print(f"[planck] extracted milca -> {MILCA_PATH}"
          f"{', mask -> ' + str(MASK_PATH_PLANCK) if masks else ' (no mask found)'}")


# ---------------------------------------------------------------------------
# Step 2: Planck CAP (healpy)
# ---------------------------------------------------------------------------

BAD_DATA_THRESH = -1.0e10  # Planck BAD_DATA sentinel is -1.6375e30; anything this negative is invalid


def planck_cap_per_object(ymap, mask, lon_deg, lat_deg, theta_arcmin,
                          mask_aperture_min=MASK_APERTURE_MIN, verbose=True):
    """Per-object CAP via healpy query_disc: area_disk*(mean_disk-mean_ring),
    ring=[theta_d, sqrt2*theta_d] (equal-area compensated aperture -- same
    geometry as act_ycap_measure.cap_thumbs). lon_deg/lat_deg MUST already be
    in the map's native pixelization frame (Planck MILCA is COORDSYS=GALACTIC
    -- convert from equatorial RA/Dec via astropy before calling). Returns
    (cap (n,n_theta) in y*arcmin^2, mask_ok (n,n_theta) bool)."""
    import healpy as hp
    nside = hp.get_nside(ymap)
    n = len(lon_deg)
    n_theta = len(theta_arcmin)
    cap = np.full((n, n_theta), np.nan, dtype=np.float64)
    mfrac = np.ones((n, n_theta), dtype=np.float64)  # default: unmasked if no mask given
    vecs = hp.ang2vec(np.asarray(lon_deg, dtype=np.float64),
                       np.asarray(lat_deg, dtype=np.float64), lonlat=True)
    theta_max = theta_arcmin.max()
    rad_max = np.deg2rad(SQ2 * theta_max / 60.0)

    t0 = time.time()
    for i in range(n):
        v = vecs[i]
        pix = hp.query_disc(nside, v, rad_max, inclusive=False)
        if len(pix) == 0:
            continue
        vp = np.array(hp.pix2vec(nside, pix))
        cosang = np.clip(v @ vp, -1.0, 1.0)
        ang_arcmin = np.degrees(np.arccos(cosang)) * 60.0
        vals = ymap[pix]
        good = vals > BAD_DATA_THRESH
        mvals = mask[pix] if mask is not None else None
        for j, td in enumerate(theta_arcmin):
            tr = SQ2 * td
            in_disk = (ang_arcmin <= td) & good
            in_ring = (ang_arcmin > td) & (ang_arcmin <= tr) & good
            nd, nr = int(in_disk.sum()), int(in_ring.sum())
            if nd < 3 or nr < 5:
                continue
            mean_disk = vals[in_disk].mean()
            mean_ring = vals[in_ring].mean()
            cap[i, j] = np.pi * td ** 2 * (mean_disk - mean_ring)
            if mvals is not None:
                in_full = ang_arcmin <= tr
                mfrac[i, j] = mvals[in_full].mean()
        if verbose and (i + 1) % 20000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [planck cap] {i+1}/{n} ({rate:.0f} obj/s)", flush=True)
    mask_ok = (mfrac > mask_aperture_min) & np.isfinite(cap)
    return cap, mask_ok


# ---------------------------------------------------------------------------
# Step 3: ACT beam-matched CAP (reuses act_ycap_measure thumbnail/cap code)
# ---------------------------------------------------------------------------

def _act_y_pass(ymap, ra_deg, dec_deg, theta_arcmin, r_thumb, res_arcmin,
                sigma_px, chunk, verbose):
    """Pass 1: y-map thumbnails -> Gaussian smooth -> cap_thumbs. Only the
    y-map (+ its prefiltered copy) needs to be resident -- keeping this
    separate from the mask pass roughly halves peak RSS (~5.3GB -> ~3.6GB
    baseline in this session's ~10GB cgroup), which is what lets chunk be
    large enough to finish in reasonable wall time on 1 CPU."""
    from scipy.ndimage import spline_filter, gaussian_filter

    n = len(ra_deg)
    n_theta = len(theta_arcmin)
    thetas_px_row = theta_arcmin / res_arcmin
    cap = np.full((n, n_theta), np.nan, dtype=np.float32)

    t0 = time.time()
    ymap_f = spline_filter(np.asarray(ymap), order=3, mode="grid-wrap",
                            output=np.float32)
    print(f"  [act smooth] spline prefilter done ({time.time()-t0:.0f}s), "
          f"r_thumb={r_thumb:.2f}'", flush=True)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        ty = aym.batch_thumbnails(ymap, ra_deg[lo:hi], dec_deg[lo:hi],
                                  r_arcmin=r_thumb, res_arcmin=res_arcmin,
                                  order=3, arr=ymap_f, prefilter=False)
        ty_sm = gaussian_filter(ty, sigma=(0.0, sigma_px, sigma_px),
                                mode="nearest")
        thetas_px = np.broadcast_to(thetas_px_row, (hi - lo, n_theta))
        cap[lo:hi] = aym.cap_thumbs(ty_sm, thetas_px) * (res_arcmin ** 2)
        del ty, ty_sm
        gc.collect()
        if verbose:
            rate = hi / (time.time() - t0)
            print(f"  [act smooth y-pass] [{hi}/{n}] {rate:.0f} obj/s "
                  f"eta={((n-hi)/max(rate,1e-9))/60:.1f}min", flush=True)
    del ymap_f
    gc.collect()
    return cap


def _act_mask_pass(mask, ra_deg, dec_deg, theta_arcmin, r_thumb, res_arcmin,
                   chunk, verbose):
    """Pass 2: raw (unsmoothed) mask thumbnails -> mask_fraction/center gate.
    Mask gating uses the NATIVE mask (not beam-matched) -- same convention
    as act_ycap_measure.measure()."""
    n = len(ra_deg)
    n_theta = len(theta_arcmin)
    thetas_px_row = theta_arcmin / res_arcmin
    mfrac = np.zeros((n, n_theta), dtype=np.float32)
    mcen = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        tm = np.clip(aym.batch_thumbnails(mask, ra_deg[lo:hi], dec_deg[lo:hi],
                                          r_arcmin=r_thumb, res_arcmin=res_arcmin),
                     0.0, 1.0)
        ny_, nx_ = tm.shape[-2:]
        mcen[lo:hi] = tm[:, ny_ // 2, nx_ // 2]
        thetas_px = np.broadcast_to(thetas_px_row, (hi - lo, n_theta))
        mfrac[lo:hi] = aym.mask_fraction(tm, thetas_px)
        del tm
        gc.collect()
        if verbose:
            rate = hi / (time.time() - t0)
            print(f"  [act smooth mask-pass] [{hi}/{n}] {rate:.0f} obj/s "
                  f"eta={((n-hi)/max(rate,1e-9))/60:.1f}min", flush=True)
    return mfrac, mcen


def act_smoothed_cap_per_object(ymap, mask, ra_deg, dec_deg, theta_arcmin,
                                smooth_fwhm_arcmin=SMOOTH_FWHM_ARCMIN,
                                res_arcmin=1.0, chunk=6000,
                                mask_aperture_min=MASK_APERTURE_MIN,
                                margin_sigma=5.0, verbose=True):
    """Postage-stamp thumbnails (act_ycap_measure.batch_thumbnails) ->
    real-space Gaussian smoothing (scipy.ndimage.gaussian_filter, NOT a
    whole-map FFT -- memory-safe) to Planck's 10' beam -> the exact
    act_ycap_measure.cap_thumbs discretized CAP filter, at the same 5
    apertures used on the Planck side. ymap and mask are processed in two
    separate passes (see _act_y_pass/_act_mask_pass) to bound peak memory.

    res_arcmin=1.0' (coarser than the map's native 0.5'/px): safe because
    the post-smoothing effective resolution is ~9.87' FWHM (sigma~4.2'), so
    1'/px sampling is still >4x oversampled relative to the kernel and the
    thumbnail pixel count (hence wall time, single-CPU session) drops ~4x
    vs native sampling with no measurable loss of accuracy at these >=4'
    apertures -- this is the CPU-budget analogue of the note in
    act_ycap_measure.measure() about res_arcmin controlling thumbnail cost."""
    sigma_arcmin = smooth_fwhm_arcmin / SIGMA_FACTOR
    sigma_px = sigma_arcmin / res_arcmin
    r_thumb = SQ2 * theta_arcmin.max() + margin_sigma * sigma_arcmin

    cap = _act_y_pass(ymap, ra_deg, dec_deg, theta_arcmin, r_thumb, res_arcmin,
                      sigma_px, chunk, verbose)
    mfrac, mcen = _act_mask_pass(mask, ra_deg, dec_deg, theta_arcmin, r_thumb,
                                 res_arcmin, chunk, verbose)

    in_foot = mcen > aym.MASK_CENTER_MIN
    mask_ok = (mfrac > mask_aperture_min) & in_foot[:, None] & np.isfinite(cap)
    return cap, mask_ok


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_catalog():
    d = np.load(CATALOG, allow_pickle=True)
    ra = d["per_obj_ra"].astype(np.float64)
    dec = d["per_obj_dec"].astype(np.float64)
    w = d["per_obj_w"].astype(np.float64) if "per_obj_w" in d.files else np.ones(len(ra))
    return ra, dec, w, int(d["n_gal"])


def read_planck_healpix(path, field=0):
    import healpy as hp
    m = hp.read_map(str(path), field=field, dtype=np.float64)
    return m


def radec_to_galactic(ra_deg, dec_deg):
    """milca_ymaps.fits/masks.fits are COORDSYS=GALACTIC (verified from the
    FITS header); the catalog is equatorial (ICRS) RA/Dec -- must rotate
    before indexing the healpix map, or query_disc looks at the wrong sky
    patch entirely."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg,
                 frame="icrs").galactic
    return c.l.deg, c.b.deg


def run_measure_planck():
    import healpy as hp
    ra, dec, w, n_gal = load_catalog()
    print(f"[planck] catalog n={n_gal}")
    lon, lat = radec_to_galactic(ra, dec)
    print(f"[planck] converted ICRS->Galactic for query positions "
          f"(l range {lon.min():.2f}-{lon.max():.2f}, "
          f"b range {lat.min():.2f}-{lat.max():.2f})")

    # HDU1 "Y-MAP": columns FULL/FIRST/LAST (field 0 = FULL, the full-mission
    # combined map); COORDSYS=GALACTIC, ORDERING=RING, NSIDE=2048,
    # BAD_DATA=-1.6375e30 (handled via BAD_DATA_THRESH in the CAP function).
    ymap = read_planck_healpix(MILCA_PATH, field=0)
    mask = None
    if MASK_PATH_PLANCK.exists():
        try:
            # masks.fits columns: M1..M4 nominally "GALACTIC MASK 40/50/60/70%",
            # M5 "POINT SOURCE MASK". This masks.fits ships as part of the
            # R2.00 tarball; the PLA changelog documents a KNOWN BUG in the
            # R2.00 masks file (fixed in R2.01, not separately downloadable
            # as a small standalone product): a region around the GALACTIC
            # POLE was erroneously masked in addition to the Galactic plane.
            # We verified this empirically at our own catalog positions
            # (DESI SGC LRGs, |b| in [20.6, 79.9] deg): M1 retention rises
            # monotonically with |b| (0.85 -> 1.00, exactly the expected
            # behavior of a real Galactic-latitude mask) while M2/M3/M4
            # retention is erratic and non-monotonic vs |b| (e.g. M2 drops
            # from 0.39 to 0.00 then M3/M4 partially recover) -- the
            # signature of the documented pole-region bug. We therefore use
            # ONLY M1 (Galactic, well-behaved for this sample) and M5 (point
            # source, flat ~0.80-0.88 vs |b| as expected for a source mask)
            # and do NOT use M2/M3/M4.
            m_gal = read_planck_healpix(MASK_PATH_PLANCK, field=0)   # M1
            m_ps = read_planck_healpix(MASK_PATH_PLANCK, field=4)    # M5
            mask = (m_gal > 0.5) & (m_ps > 0.5)
            mask = mask.astype(np.float64)
            if hp.get_nside(mask) != hp.get_nside(ymap):
                mask = hp.ud_grade(mask, hp.get_nside(ymap))
            print(f"[planck] mask (M1 Galactic & M5 point-source) good fraction: "
                  f"{mask.mean():.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"[planck] WARNING: could not read mask ({e}); proceeding unmasked")
            mask = None
    else:
        print("[planck] WARNING: no mask file found; proceeding without Galactic/point-source masking")

    cap, ok = planck_cap_per_object(ymap, mask, lon, lat, THETA_ARCMIN)
    del ymap, mask
    gc.collect()

    res = aym.stack_with_covariances(cap.astype(np.float64), ok, w, ra, dec,
                                     n_boot=aym.N_BOOT, seed=aym.SEED)
    out = LC / "planck_crosscheck_planck_side.npz"
    np.savez(out, mean=res["mean"], err_boot=res["err_boot"], err_jk=res["err_jk"],
             cov_boot=res["cov_boot"], cov_jk=res["cov_jk"], n_used=res["n_used"],
             n_jk_cells=res["n_jk_cells"], theta_arcmin=THETA_ARCMIN,
             per_obj_cap=cap.astype(np.float32), per_obj_ok=ok, n_gal=n_gal)
    print(f"[planck] saved {out}: mean={res['mean']}, err_jk={res['err_jk']}")
    del cap, ok
    gc.collect()


def run_measure_act(chunk=6000, res_arcmin=1.0):
    ra, dec, w, n_gal = load_catalog()
    print(f"[act-smoothed] catalog n={n_gal}")

    from pixell import enmap

    sigma_arcmin = SMOOTH_FWHM_ARCMIN / SIGMA_FACTOR
    sigma_px = sigma_arcmin / res_arcmin
    r_thumb = SQ2 * THETA_ARCMIN.max() + 5.0 * sigma_arcmin

    # Two-pass, one map resident at a time -- caller-level memory bound (see
    # act_smoothed_cap_per_object docstring for why this matters at 1 CPU /
    # ~10GB cgroup).
    ymap = enmap.read_map(str(aym.YMAP_DIR / aym.MAP_VARIANTS["cib1.7"]))
    print("[act-smoothed] y-map loaded")
    cap = _act_y_pass(ymap, ra, dec, THETA_ARCMIN, r_thumb, res_arcmin,
                      sigma_px, chunk, verbose=True)
    del ymap
    gc.collect()

    mask = enmap.read_map(str(aym.MASK_PATH))
    print("[act-smoothed] mask loaded")
    mfrac, mcen = _act_mask_pass(mask, ra, dec, THETA_ARCMIN, r_thumb,
                                 res_arcmin, chunk, verbose=True)
    del mask
    gc.collect()

    in_foot = mcen > aym.MASK_CENTER_MIN
    ok = (mfrac > MASK_APERTURE_MIN) & in_foot[:, None] & np.isfinite(cap)

    res = aym.stack_with_covariances(cap.astype(np.float64), ok, w, ra, dec,
                                     n_boot=aym.N_BOOT, seed=aym.SEED)
    out = LC / "planck_crosscheck_act_side.npz"
    np.savez(out, mean=res["mean"], err_boot=res["err_boot"], err_jk=res["err_jk"],
             cov_boot=res["cov_boot"], cov_jk=res["cov_jk"], n_used=res["n_used"],
             n_jk_cells=res["n_jk_cells"], theta_arcmin=THETA_ARCMIN,
             per_obj_cap=cap.astype(np.float32), per_obj_ok=ok, n_gal=n_gal,
             smooth_fwhm_arcmin=SMOOTH_FWHM_ARCMIN)
    print(f"[act-smoothed] saved {out}: mean={res['mean']}, err_jk={res['err_jk']}")
    del cap, ok
    gc.collect()


def run_compare():
    from scipy.stats import chi2 as chi2_dist

    dp = np.load(LC / "planck_crosscheck_planck_side.npz")
    da = np.load(LC / "planck_crosscheck_act_side.npz")

    cap_planck = dp["mean"]
    err_planck = dp["err_jk"]
    cap_act = da["mean"]
    err_act = da["err_jk"]

    delta = cap_act - cap_planck
    var = err_act ** 2 + err_planck ** 2
    chi2_val = float(np.sum(delta ** 2 / var))
    dof = len(THETA_ARCMIN)
    pte = float(chi2_dist.sf(chi2_val, dof))

    n_gal = int(dp["n_gal"])
    n_gal_used_planck = int(dp["n_used"][0])
    frac_used_planck = n_gal_used_planck / n_gal
    notes = (
        f"Round-2 T2 (docs/paper_improvement_plan.md B1): independent y-map "
        f"cross-check at the same {n_gal} DESI SGC LRGs (z=0.4-0.6) used in "
        f"the paper's real-data measurement (LC/T2f_lrg_z0406_cib1.7.npz). "
        f"Maps: (1) Planck 2015 MILCA all-sky Compton-y "
        f"(COM_CompMap_YSZ_R2.00/milca_ymaps.fits, NSIDE=2048, ~10' FWHM "
        f"effective beam), full-sky so the ACT DR6 footprint is applied "
        f"implicitly by using the identical object list; masking uses the "
        f"bundled masks.fits M1 (Galactic) AND M5 (point source) columns, "
        f"gated per-object as mean(mask) over r<sqrt2*theta_d > 0.99 (same "
        f"threshold convention as act_ycap_measure.py). NOTE: this "
        f"masks.fits ships inside the R2.00 tarball, which the PLA "
        f"changelog documents as having a bug (fixed in R2.01, not "
        f"separately downloadable as a small product): a region around the "
        f"Galactic POLE was erroneously masked in addition to the Galactic "
        f"plane. We verified empirically at our own catalog positions "
        f"(|b| in [20.6,79.9] deg) that M1 behaves as a real Galactic mask "
        f"should (retention rises monotonically with |b|, 0.85->1.00) while "
        f"M2/M3/M4 are non-monotonic/erratic vs |b| (the bug signature) -- "
        f"so M2/M3/M4 are deliberately NOT used; M5 (point source) is flat "
        f"vs |b| as expected (~0.80-0.88) and is used as-is. Combined M1&M5 "
        f"retains {n_gal_used_planck} of {n_gal} objects at theta=4' after "
        f"gating (fraction {frac_used_planck:.3f}). RA/Dec (equatorial) "
        f"catalog positions were rotated to Galactic l/b via astropy "
        f"before indexing the map (COORDSYS=GALACTIC). (2) ACT DR6+Planck "
        f"NILC deproj_cib_1.7_10.7 map (the paper's baseline "
        f"systematics-hardened variant), smoothed with an additional "
        f"Gaussian FWHM={SMOOTH_FWHM_ARCMIN:.3f}' (= sqrt(10.0^2-1.6^2), "
        f"quadrature beam-match to Planck) applied on padded thumbnails "
        f"(r_thumb sized to sqrt2*theta_max + 5*sigma_smooth margin) via "
        f"scipy.ndimage.gaussian_filter at 1.0'/px working resolution "
        f"(coarser than the map's native 0.5'/px -- safe since the "
        f"post-smoothing effective resolution is ~9.87' FWHM, still >4x "
        f"oversampled at 1'/px; chosen for wall-time/memory on a "
        f"single-CPU, ~10GB-cgroup session, not a whole-map FFT), then the "
        f"identical act_ycap_measure.cap_thumbs discretized CAP filter as "
        f"the paper. "
        f"CAP filter definition: compensated aperture, disk[0,theta_d] "
        f"minus equal-area ring[theta_d,sqrt2*theta_d], in y*arcmin^2 "
        f"units; Planck side uses the continuum area*(mean_disk-mean_ring) "
        f"form (area=pi*theta_d^2), ACT side reuses the paper's discretized "
        f"pixel-count form -- both are the same compensated-aperture "
        f"estimator, differing only in discretization convention (negligible "
        f"at these >=4' apertures and >=0.5'/px sampling). Errors: 10x10 "
        f"RA/Dec spatial jackknife (same recipe/cell count as "
        f"act_ycap_measure.stack_with_covariances), reported as err_jk; "
        f"bootstrap (err_boot) also stored in the per-side npz files for "
        f"reference but not used in chi2. This is a LARGE-aperture "
        f"(theta>=4') consistency check only -- Planck's 10' beam cannot "
        f"resolve the paper's small-aperture (theta<~2') fit range, so a "
        f"pass here does not validate that regime. Footprint-correlated "
        f"systematics common to both maps (shared large-scale Galactic "
        f"residuals, shared LSS sample variance from the same galaxies) "
        f"cancel in the ACT-minus-Planck difference; genuinely independent "
        f"systematics (different frequency channels/component-separation "
        f"algorithm: ACT DR6 NILC-CIB-deproj vs Planck MILCA) do not."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, theta_arcmin=THETA_ARCMIN, cap_planck=cap_planck,
             err_planck=err_planck, cap_act_smoothed=cap_act,
             err_act_smoothed=err_act, delta=delta, chi2=chi2_val, dof=dof,
             pte=pte, n_gal=n_gal, smooth_fwhm_arcmin=SMOOTH_FWHM_ARCMIN,
             planck_fwhm_arcmin=PLANCK_FWHM_ARCMIN,
             act_fwhm_arcmin=ACT_FWHM_ARCMIN, notes=notes)
    print(f"[compare] saved {OUT}")
    print(f"theta (arcmin): {THETA_ARCMIN}")
    print(f"cap_planck:       {cap_planck}")
    print(f"err_planck (jk):  {err_planck}")
    print(f"cap_act_smoothed: {cap_act}")
    print(f"err_act (jk):     {err_act}")
    print(f"ratio (act/planck): {cap_act/cap_planck}")
    print(f"delta: {delta}")
    print(f"chi2={chi2_val:.3f}, dof={dof}, PTE={pte:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default="all",
                    choices=["all", "download", "extract", "measure_planck",
                             "measure_act", "compare"])
    args = ap.parse_args()

    if args.step in ("all", "download"):
        ensure_download()
    if args.step in ("all", "extract"):
        extract_maps()
    if args.step in ("all", "measure_planck"):
        run_measure_planck()
    if args.step in ("all", "measure_act"):
        run_measure_act()
    if args.step in ("all", "compare"):
        run_compare()


if __name__ == "__main__":
    main()
