#!/usr/bin/env python
"""T1/T2 (docs/tsz_des_data_plan.md §2): the ONE real-data y-CAP measurement
engine — stacked compensated-aperture Compton-y at catalog positions on the
ACT DR6+Planck y-map.

Pipeline (per plan T1): catalog in → pixell ``reproject.thumbnails`` (gnomonic,
0.5'/px — removes the CAR RA-pixel compression at |dec|>0, box half-size
R_THUMB ≥ √2·θ_max) → the repo CAP filter (disk sum − equal-area-scaled ring
sum, EXACTLY ``lightcone_cap_stack.cap_batch``'s cumsum/searchsorted formula
and validity gates, at 0.5'/px instead of BIND's 0.29297'/px) → mask gating
(thumbnail the apodized mask identically; require center>0.5 AND mean mask
over the full r<√2·θ_d aperture > 0.99; a mask-weighted CAP variant is kept
alongside) → weight-stacked mean → bootstrap AND spatial-jackknife (~30
RA/Dec quantile cells) covariances → npz.

Units: CAP values are stored in y·arcmin² (pixel sums × RES_ARCMIN², the
Riemann-sum convention of ``lightcone_m2_ycap_liu.py`` item 2), directly
comparable to the Liu+2025 flux convention and to BIND shards after their own
pixel-area conversion.

Aperture grids (plan T2):
  rap — Liu's fixed-arcmin grid [1.0, 1.625, …, 6.0]' (9 pts), identical for
        every object; the drop-in replacement for the buggy Liu csv.
  xb  — per-galaxy θ_d = xb·θ200(z_gal), xb = linspace(0.3, 3.0, 18)
        (``lightcone_cap_stack.XB``), θ200 from the logM200=13.18 (Msun/h,
        Sailer+24 LRG anchor) r200c at the galaxy's own redshift
        (``_build_ksz_paper2_nb.py`` §0 cosmology helpers, replicated here).

Catalog modes:
  dr5      — ACT DR5 SZ clusters (SNR>5), uniform weights: pipeline validation.
  lrg      — DESI DR1 SGC spec LRGs in a z window, WEIGHT-weighted.
  random   — positions from the LRG randoms file (same z window), the
             random-position null.
  rotated  — the lrg sample with per-galaxy RA offsets (Dec-preserving),
             redrawn from a discrete offset set until in-footprint: the
             rotated null.

Usage
-----
    python examples/act_ycap_measure.py --mode dr5 --grid rap --out T1_dr5.npz
    python examples/act_ycap_measure.py --mode lrg --zmin 0.4 --zmax 0.6 \
        --grid both --map_variant baseline --out act_ycap_lrg_z0406.npz
    python examples/act_ycap_measure.py --help
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

DL = Path("/mnt/home/mlee1/ceph/paper3/B/downloads")
KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
YMAP_DIR = DL / "act_dr6_planck_ymap"
MASK_PATH = YMAP_DIR / "wide_mask_GAL070_apod_1.50_deg_wExtended.fits"
DR5_PATH = DL / "act_dr5_szcluster_hilton2009.11043/DR5_cluster-catalog_v1.1.fits"
LRG_PATH = DL / "desi_dr1_lrg_spec/LRG_SGC_clustering.dat.fits"
RAN_PATH = DL / "desi_dr1_lrg_spec/LRG_SGC_0_clustering.ran.fits"
PZBINS_PATH = DL / "liu2025_tsz_desi/dr9_lrg_pzbins.fits"

MAP_VARIANTS = {
    "baseline": "ilc_actplanck_ymap.fits",
    "cib1.0": "ilc_actplanck_ymap_deproj_cib_1.0_10.7.fits",
    "cib1.2": "ilc_actplanck_ymap_deproj_cib_1.2_10.7.fits",
    "cib1.4": "ilc_actplanck_ymap_deproj_cib_1.4_10.7.fits",
    "cib1.6": "ilc_actplanck_ymap_deproj_cib_1.6_10.7.fits",
    "cib1.7": "ilc_actplanck_ymap_deproj_cib_1.7_10.7.fits",
    "cib1.8": "ilc_actplanck_ymap_deproj_cib_1.8_10.7.fits",
    "cib2.0": "ilc_actplanck_ymap_deproj_cib_2.0_10.7.fits",
    "cib1.7_24": "ilc_actplanck_ymap_deproj_cib_1.7_24.0.fits",
    "cibdBeta": "ilc_actplanck_ymap_deproj_cib_cibdBeta_1.7_10.7.fits",
    "cibdBetadT": "ilc_actplanck_ymap_deproj_cib_cibdBeta_cibdT_1.7_10.7.fits",
}

RES_ARCMIN = 0.5                      # working pixel scale (the map's own)
PIX_AREA_ARCMIN2 = RES_ARCMIN ** 2    # pixel-sum CAP -> y·arcmin²
R_THUMB_ARCMIN = 9.0                  # half-size; √2·θ_max = 8.49' for θ_max=6'
SQ2 = np.sqrt(2.0)

RAP_ARCMIN = np.arange(1.0, 6.0 + 1e-9, 0.625)      # Liu grid, 9 pts
XB = np.linspace(0.3, 3.0, 18)                       # lightcone_cap_stack.XB
LOGM200_ANCHOR = 13.18                               # Msun/h, Sailer+24

MASK_CENTER_MIN = 0.5
MASK_APERTURE_MIN = 0.99
N_BOOT = 400
# R0 (docs/p4c_referee_hardening_plan.md): 100 spatial jackknife cells so the
# Hartlap correction stays mild for the 27-column covariance ((100-27-2)/99 =
# 0.72 full-grid, 0.93 for the 6-column fit block) — was 6x5=30, which is
# singular-regime for the full grid and cost x1.32 in chi2 bias on 6 bins.
N_JK_RA, N_JK_DEC = 10, 10
CHUNK = 20000
SEED = 20260726

# --- cosmology (Planck18-ish), replicated from _build_ksz_paper2_nb.py §0 ---
Om, OL, h = 0.3089, 0.6911, 0.6774
C_KMS, H0, ARCMIN = 299792.458, 100 * 0.6774, 180 * 60 / np.pi


def Ez(z):
    return np.sqrt(Om * (1 + z) ** 3 + OL)


def DA(z, n=30000):
    """Vectorized via one shared cumulative integral (the per-object
    linspace of the notebook version OOMs on 160k-galaxy arrays); matches
    _build_ksz_paper2_nb.py::DA to <1e-5 relative."""
    from scipy.integrate import cumulative_trapezoid
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    zg = np.linspace(0.0, max(1.5, float(z.max())), n)
    integ = cumulative_trapezoid(1.0 / Ez(zg), zg, initial=0.0)
    return (C_KMS / H0) * np.interp(z, zg, integ) / (1 + z)


def r200phys(logM200, z):
    M = 10 ** logM200 / h
    rho = 2.775e11 * h ** 2 * Ez(z) ** 2
    return (3 * M / (4 * np.pi * 200 * rho)) ** (1 / 3)


def theta200_arcmin(z, logM200=LOGM200_ANCHOR):
    return r200phys(logM200, z) / DA(z) * ARCMIN


# ---------------------------------------------------------------------------
# CAP filter on a stack of thumbnails (shared-grid version of cap_batch)
# ---------------------------------------------------------------------------

def _thumb_radial_order(ny: int, nx: int):
    """Radial sort order for the shared thumbnail grid, center pixel at
    (ny//2, nx//2) — the thumbnail analogue of cap_batch's _make_window."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - ny // 2, xx - nx // 2).ravel()
    order = np.argsort(r, kind="stable")
    return order, r[order]


def cap_thumbs(thumbs: np.ndarray, thetas_px: np.ndarray):
    """CAP(θ) per thumbnail. thumbs (N, ny, nx); thetas_px (N, n_theta) in
    thumbnail pixels. Returns (N, n_theta) pixel-sum CAP values with the
    exact cap_batch validity gates (n_disk>=3, n_ring>=5), NaN otherwise."""
    N, ny, nx = thumbs.shape
    order, rs_sorted = _thumb_radial_order(ny, nx)
    vs = thumbs.reshape(N, -1)[:, order].astype(np.float64)
    cs = np.cumsum(vs, axis=1)
    Wtot = cs.shape[1]

    i_d = np.searchsorted(rs_sorted, thetas_px, side="left")
    i_r = np.searchsorted(rs_sorted, SQ2 * thetas_px, side="left")
    row = np.arange(N)[:, None]
    idx_d = np.clip(i_d - 1, 0, Wtot - 1)
    idx_r = np.clip(i_r - 1, 0, Wtot - 1)
    disk_sum = np.where(i_d > 0, cs[row, idx_d], 0.0)
    above = np.where(i_r > 0, cs[row, idx_r], 0.0)
    ring_sum = above - disk_sum
    n_ring = i_r - i_d
    valid = (i_d >= 3) & (n_ring >= 5) & (i_r <= Wtot)
    wgt = np.where(n_ring > 0, i_d / np.maximum(n_ring, 1), 0.0)
    cap = disk_sum - ring_sum * wgt
    return np.where(valid, cap, np.nan)


def cap_thumbs_maskweighted(thumbs, mthumbs, thetas_px):
    """Mask-weighted CAP variant: pixel sums of y·m, ring rescaled by the
    masked area ratio Σm_disk/Σm_ring instead of the count ratio."""
    N, ny, nx = thumbs.shape
    order, rs_sorted = _thumb_radial_order(ny, nx)
    vs = (thumbs * mthumbs).reshape(N, -1)[:, order].astype(np.float64)
    ms = mthumbs.reshape(N, -1)[:, order].astype(np.float64)
    cs, cm = np.cumsum(vs, axis=1), np.cumsum(ms, axis=1)
    Wtot = cs.shape[1]
    i_d = np.searchsorted(rs_sorted, thetas_px, side="left")
    i_r = np.searchsorted(rs_sorted, SQ2 * thetas_px, side="left")
    row = np.arange(N)[:, None]
    idx_d = np.clip(i_d - 1, 0, Wtot - 1)
    idx_r = np.clip(i_r - 1, 0, Wtot - 1)
    dsum = np.where(i_d > 0, cs[row, idx_d], 0.0)
    rsum = np.where(i_r > 0, cs[row, idx_r], 0.0) - dsum
    dm = np.where(i_d > 0, cm[row, idx_d], 0.0)
    rm = np.where(i_r > 0, cm[row, idx_r], 0.0) - dm
    valid = (i_d >= 3) & ((i_r - i_d) >= 5) & (i_r <= Wtot) & (rm > 0)
    return np.where(valid, dsum - rsum * dm / np.maximum(rm, 1e-30), np.nan)


def mask_fraction(mthumbs, thetas_px):
    """Mean mask over the full compensated aperture r < √2·θ_d, per (obj, θ)."""
    N, ny, nx = mthumbs.shape
    order, rs_sorted = _thumb_radial_order(ny, nx)
    ms = mthumbs.reshape(N, -1)[:, order].astype(np.float64)
    cm = np.cumsum(ms, axis=1)
    i_r = np.searchsorted(rs_sorted, SQ2 * thetas_px, side="left")
    row = np.arange(N)[:, None]
    tot = np.where(i_r > 0, cm[row, np.clip(i_r - 1, 0, cm.shape[1] - 1)], 0.0)
    return tot / np.maximum(i_r, 1)


# ---------------------------------------------------------------------------
# Batched gnomonic thumbnails
# ---------------------------------------------------------------------------
# pixell's reproject.thumbnails computes each postage stamp's rotated
# coordinate grid in a per-object Python loop (~100 obj/s single-core, and
# this session is cgroup-pinned to 1 CPU) — unusable for 160k LRGs + 10x
# randoms. This is the identical tangent-plane construction (local east/north
# basis rotation, exact spherical trig; gnomonic-vs-arc distortion at
# rho<=9' is O(rho^2)~4e-6, irrelevant) vectorized over objects, with ONE
# scipy map_coordinates call per chunk. Validated against
# reproject.thumbnails in the T1 verdict (stacked-CAP agreement).

def batch_thumbnails(imap, ra_deg, dec_deg, r_arcmin=R_THUMB_ARCMIN,
                     res_arcmin=RES_ARCMIN, order=1, arr=None, prefilter=True):
    """arr/prefilter: pass a spline_filter(...)-prefiltered array with
    prefilter=False to get pixell-identical cubic interpolation without
    re-filtering the 450-Mpx map on every chunk."""
    from scipy.ndimage import map_coordinates
    n = len(ra_deg)
    npix = 2 * int(round(r_arcmin / res_arcmin)) + 1
    off = (np.arange(npix) - npix // 2) * np.deg2rad(res_arcmin / 60.0)
    dyg, dxg = np.meshgrid(off, off, indexing="ij")   # rows = north offset
    x, y = dxg.ravel(), dyg.ravel()
    rho = np.hypot(x, y)
    inv = 1.0 / np.maximum(rho, 1e-30)
    ex, ey = x * inv, y * inv
    cosr, sinr = np.cos(rho)[None, :], np.sin(rho)[None, :]
    ext, eyt = ex[None, :] * sinr, ey[None, :] * sinr

    dec0 = np.deg2rad(dec_deg)[:, None]
    ra0 = np.deg2rad(ra_deg)[:, None]
    sd, cd = np.sin(dec0), np.cos(dec0)
    sr, cr = np.sin(ra0), np.cos(ra0)
    # center n, local east e=(-sin ra, cos ra, 0), north u=(-sd cr, -sd sr, cd)
    px = cd * cr * cosr - sr * ext - sd * cr * eyt
    py = cd * sr * cosr + cr * ext - sd * sr * eyt
    pz = sd * cosr + cd * eyt
    dec = np.arcsin(np.clip(pz, -1.0, 1.0))
    ra = np.arctan2(py, px)
    pix = imap.sky2pix(np.array([dec.ravel(), ra.ravel()]))
    # grid-wrap wraps RA correctly (the map is full-360 in x); the dec axis
    # never reaches its edges because the apodized mask keeps centers >~1.5
    # deg inside, far beyond the 9' stamp half-size.
    src = np.asarray(imap) if arr is None else arr
    vals = map_coordinates(src, pix, order=order, mode="grid-wrap",
                           prefilter=prefilter)
    return vals.reshape(n, npix, npix)


# ---------------------------------------------------------------------------
# Measurement core
# ---------------------------------------------------------------------------

def build_theta_grid(grid: str, z: np.ndarray | None, n_obj: int):
    """Returns (theta_arcmin (n_obj, n_theta), theta_kind, theta_value)."""
    cols, kinds, values = [], [], []
    if grid in ("rap", "both"):
        cols.append(np.broadcast_to(RAP_ARCMIN, (n_obj, len(RAP_ARCMIN))))
        kinds += ["fixed_arcmin"] * len(RAP_ARCMIN)
        values += list(RAP_ARCMIN)
    if grid in ("xb", "both"):
        if z is None:
            raise SystemExit("xb grid needs per-object redshifts")
        th200 = theta200_arcmin(z)[:, None]
        cols.append(XB[None, :] * th200)
        kinds += ["xb"] * len(XB)
        values += list(XB)
    theta = np.concatenate(cols, axis=1)
    return theta, np.array(kinds), np.array(values)


def measure(ymap, mask, ra_deg, dec_deg, weights, theta_arcmin,
            chunk=CHUNK, verbose=True, res_arcmin=RES_ARCMIN):
    """Thumbnails + CAP + mask gates for every object. Returns dict of
    per-object arrays: cap (y·arcmin²), cap_mw, mask_ok, in_foot.

    res_arcmin: thumbnail sampling scale. 0.5 = the map's native pixel;
    0.29296875 = BIND's lux map pixel, so the CAP aperture discretization
    matches the BIND shards EXACTLY (T3 finding: at 0.5'/px, theta_d<~1.6'
    disks hold <=10 px and the discrete CAP is a different estimator than
    BIND's — the dominant small-aperture artifact in the first M2R)."""
    n = len(ra_deg)
    n_theta = theta_arcmin.shape[1]
    pix_area = res_arcmin ** 2
    # thumbnail memory scales as chunk/res^2 — shrink the chunk at finer
    # sampling so peak RSS stays at the 0.5'-default level (~10GB cgroup)
    chunk = max(1000, int(chunk * (res_arcmin / RES_ARCMIN) ** 2))
    cap = np.full((n, n_theta), np.nan, dtype=np.float32)
    cap_mw = np.full((n, n_theta), np.nan, dtype=np.float32)
    mfrac = np.zeros((n, n_theta), dtype=np.float32)
    mcen = np.zeros(n, dtype=np.float32)
    thetas_px = theta_arcmin / res_arcmin

    t0 = time.time()
    from scipy.ndimage import spline_filter
    ymap_f = spline_filter(np.asarray(ymap), order=3, mode="grid-wrap",
                           output=np.float32)
    print(f"  spline prefilter done ({time.time()-t0:.0f}s)", flush=True)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        ty = batch_thumbnails(ymap, ra_deg[lo:hi], dec_deg[lo:hi],
                              res_arcmin=res_arcmin,
                              order=3, arr=ymap_f, prefilter=False)
        tm = np.clip(batch_thumbnails(mask, ra_deg[lo:hi], dec_deg[lo:hi],
                                      res_arcmin=res_arcmin), 0.0, 1.0)
        ny_, nx_ = ty.shape[-2:]
        mcen[lo:hi] = tm[:, ny_ // 2, nx_ // 2]
        cap[lo:hi] = cap_thumbs(ty, thetas_px[lo:hi]) * pix_area
        cap_mw[lo:hi] = cap_thumbs_maskweighted(ty, tm, thetas_px[lo:hi]) * pix_area
        mfrac[lo:hi] = mask_fraction(tm, thetas_px[lo:hi])
        if verbose:
            rate = hi / (time.time() - t0)
            print(f"  [{hi}/{n}] {rate:.0f} obj/s", flush=True)

    in_foot = mcen > MASK_CENTER_MIN
    mask_ok = (mfrac > MASK_APERTURE_MIN) & in_foot[:, None] & np.isfinite(cap)
    return dict(cap=cap, cap_mw=cap_mw, mask_ok=mask_ok, in_foot=in_foot,
                mask_frac=mfrac)


def _weighted_stack(cap, ok, w):
    """Per-θ weighted mean over objects with ok gate (NaN-safe)."""
    wcol = np.where(ok, w[:, None], 0.0)
    num = np.nansum(np.where(ok, cap, 0.0) * wcol, axis=0)
    den = wcol.sum(axis=0)
    return num / np.maximum(den, 1e-30), den


def stack_with_covariances(cap, ok, w, ra_deg, dec_deg, n_boot=N_BOOT,
                           seed=SEED):
    """Weighted stacked mean + bootstrap and spatial-jackknife covariances."""
    mean, wsum = _weighted_stack(cap, ok, w)
    n_used = ok.sum(axis=0)

    rng = np.random.default_rng(seed)
    n = len(w)
    boot = np.empty((n_boot, cap.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b], _ = _weighted_stack(cap[idx], ok[idx], w[idx])
    cov_boot = np.cov(boot.T)

    # spatial jackknife: RA quantile slices × Dec quantiles within each slice
    cell = np.zeros(n, dtype=np.int64)
    ra_edges = np.quantile(ra_deg, np.linspace(0, 1, N_JK_RA + 1))
    ra_bin = np.clip(np.searchsorted(ra_edges, ra_deg, side="right") - 1, 0, N_JK_RA - 1)
    for i in range(N_JK_RA):
        s = ra_bin == i
        if s.sum() == 0:
            continue
        de = np.quantile(dec_deg[s], np.linspace(0, 1, N_JK_DEC + 1))
        db = np.clip(np.searchsorted(de, dec_deg[s], side="right") - 1, 0, N_JK_DEC - 1)
        cell[s] = i * N_JK_DEC + db
    cells = np.unique(cell)
    jk = []
    for c in cells:
        keep = cell != c
        m, _ = _weighted_stack(cap[keep], ok[keep], w[keep])
        jk.append(m)
    jk = np.array(jk)
    njk = len(jk)
    dev = jk - jk.mean(axis=0)
    cov_jk = (njk - 1) / njk * dev.T @ dev

    return dict(mean=mean, n_used=n_used, wsum=wsum,
                cov_boot=cov_boot, err_boot=np.sqrt(np.diag(cov_boot)),
                cov_jk=cov_jk, err_jk=np.sqrt(np.diag(cov_jk)),
                n_jk_cells=njk, boot_samples=boot.astype(np.float32))


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------

def load_dr5(snr_min=5.0):
    from astropy.io import fits
    c = fits.getdata(DR5_PATH)
    m = c["SNR"] > snr_min
    return dict(ra=np.asarray(c["RADeg"][m], dtype=np.float64),
                dec=np.asarray(c["decDeg"][m], dtype=np.float64),
                w=np.ones(int(m.sum())), z=np.asarray(c["redshift"][m], dtype=np.float64),
                fixed_y_c=np.asarray(c["fixed_y_c"][m], dtype=np.float64),
                name=f"dr5_snr{snr_min:g}")


def load_lrg(zmin, zmax, ebv_max=None):
    from astropy.io import fits
    c = fits.getdata(LRG_PATH)
    m = (c["Z"] >= zmin) & (c["Z"] <= zmax)
    out = dict(ra=np.asarray(c["RA"][m], dtype=np.float64),
               dec=np.asarray(c["DEC"][m], dtype=np.float64),
               w=np.asarray(c["WEIGHT"][m], dtype=np.float64),
               z=np.asarray(c["Z"][m], dtype=np.float64),
               targetid=np.asarray(c["TARGETID"][m]),
               name=f"lrg_sgc_z{zmin:g}-{zmax:g}")
    if ebv_max is not None:
        import fitsio
        t = fitsio.read(PZBINS_PATH, columns=["TARGETID", "EBV"])
        order = np.argsort(t["TARGETID"])
        pos = np.searchsorted(t["TARGETID"], out["targetid"], sorter=order)
        pos = np.clip(pos, 0, len(order) - 1)
        hit = t["TARGETID"][order[pos]] == out["targetid"]
        ebv = np.full(len(out["ra"]), np.nan)
        ebv[hit] = t["EBV"][order[pos[hit]]]
        keep = np.isfinite(ebv) & (ebv < ebv_max)
        for k in ("ra", "dec", "w", "z", "targetid"):
            out[k] = out[k][keep]
        out["name"] += f"_ebv{ebv_max:g}"
        out["ebv_matched_frac"] = float(hit.mean())
    return out


def load_liu(pz_bin, n_sub=400000, seed=SEED):
    """R5c (docs/p4c_referee_hardening_plan.md): Liu+2025's OWN DR9
    photometric LRG sample (Main, lrg_mask==0 quality cut), per pz_bin —
    measured with OUR pipeline for the direct apples-to-apples against the
    (bit-identical-bins) release csv. Uniform weights, subsampled for
    tractability (errors stay well below the amplitude question)."""
    import fitsio
    t = fitsio.read(PZBINS_PATH, columns=["RA", "DEC", "Z_PHOT_MEDIAN",
                                          "pz_bin", "lrg_mask"])
    m = (t["pz_bin"] == pz_bin) & (t["lrg_mask"] == 0)
    idx = np.nonzero(m)[0]
    n_parent = len(idx)
    rng = np.random.default_rng(seed)
    if n_sub and n_sub < len(idx):
        idx = rng.choice(idx, n_sub, replace=False)
    return dict(ra=np.asarray(t["RA"][idx], dtype=np.float64),
                dec=np.asarray(t["DEC"][idx], dtype=np.float64),
                w=np.ones(len(idx)),
                z=np.asarray(t["Z_PHOT_MEDIAN"][idx], dtype=np.float64),
                name=f"liu_pz{pz_bin}", n_parent=n_parent)


def load_random(zmin, zmax, n_target, seed=SEED):
    from astropy.io import fits
    c = fits.getdata(RAN_PATH)
    m = (c["Z"] >= zmin) & (c["Z"] <= zmax)
    idx = np.nonzero(m)[0]
    rng = np.random.default_rng(seed)
    if n_target < len(idx):
        idx = rng.choice(idx, n_target, replace=False)
    return dict(ra=np.asarray(c["RA"][idx], dtype=np.float64),
                dec=np.asarray(c["DEC"][idx], dtype=np.float64),
                w=np.ones(len(idx)), z=np.asarray(c["Z"][idx], dtype=np.float64),
                name=f"random_z{zmin:g}-{zmax:g}")


def load_rotated(base, mask, offsets=(30.0, 60.0, 90.0, 120.0, 150.0), seed=SEED):
    """Dec-preserving RA rotation null: each galaxy tries the offsets in a
    per-galaxy random order and keeps the first in-footprint position."""
    from pixell import utils  # noqa: F401  (pixell already imported by caller)
    rng = np.random.default_rng(seed + 1)
    n = len(base["ra"])
    ra_new = np.full(n, np.nan)
    perm = rng.permuted(np.tile(np.arange(len(offsets)), (n, 1)), axis=1)
    pending = np.arange(n)
    for k in range(len(offsets)):
        if len(pending) == 0:
            break
        off = np.array(offsets)[perm[pending, k]]
        cand = (base["ra"][pending] + off) % 360.0
        coords = np.deg2rad(np.column_stack([base["dec"][pending], cand]))
        pix = mask.sky2pix(coords.T)
        iy = np.clip(np.round(pix[0]).astype(int), 0, mask.shape[-2] - 1)
        ix = np.clip(np.round(pix[1]).astype(int), 0, mask.shape[-1] - 1)
        good = np.asarray(mask)[iy, ix] > 0.99
        ra_new[pending[good]] = cand[good]
        pending = pending[~good]
    keep = np.isfinite(ra_new)
    return dict(ra=ra_new[keep], dec=base["dec"][keep], w=base["w"][keep],
                z=base["z"][keep], name=base["name"] + "_rotated",
                rotated_kept_frac=float(keep.mean()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=["dr5", "lrg", "random", "rotated", "liu", "rotated_liu"])
    ap.add_argument("--pz_bin", type=int, default=1)
    ap.add_argument("--n_sub", type=int, default=400000)
    ap.add_argument("--map_variant", default="baseline", choices=sorted(MAP_VARIANTS))
    ap.add_argument("--grid", default="rap", choices=["rap", "xb", "both"])
    ap.add_argument("--zmin", type=float, default=0.4)
    ap.add_argument("--zmax", type=float, default=0.6)
    ap.add_argument("--snr_min", type=float, default=5.0)
    ap.add_argument("--ebv_max", type=float, default=None)
    ap.add_argument("--n_random", type=int, default=None,
                    help="random-null count (default 10x the z-window LRG count)")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--res_arcmin", type=float, default=RES_ARCMIN,
                    help="thumbnail sampling scale; 0.29296875 matches the "
                         "BIND lux pixel exactly (T3 discretization fix)")
    ap.add_argument("--out", required=True, help="output npz path (relative -> KS/lightcone/)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = KS / "lightcone" / out_path
    if out_path.exists() and not args.force:
        print(f"[ycap] exists, skipping: {out_path}")
        return

    from pixell import enmap
    t0 = time.time()
    ymap = enmap.read_map(str(YMAP_DIR / MAP_VARIANTS[args.map_variant]))
    mask = enmap.read_map(str(MASK_PATH))
    print(f"[ycap] maps loaded ({time.time()-t0:.0f}s): {args.map_variant}", flush=True)

    if args.mode == "dr5":
        cat = load_dr5(args.snr_min)
    elif args.mode == "lrg":
        cat = load_lrg(args.zmin, args.zmax, args.ebv_max)
    elif args.mode == "random":
        n_lrg = len(load_lrg(args.zmin, args.zmax)["ra"])
        n_tgt = args.n_random or 10 * n_lrg
        cat = load_random(args.zmin, args.zmax, n_tgt)
    elif args.mode == "rotated":
        cat = load_rotated(load_lrg(args.zmin, args.zmax), mask)
    elif args.mode == "liu":
        cat = load_liu(args.pz_bin, args.n_sub)
    elif args.mode == "rotated_liu":
        cat = load_rotated(load_liu(args.pz_bin, args.n_sub), mask)

    n = len(cat["ra"])
    print(f"[ycap] catalog {cat['name']}: n={n}", flush=True)
    theta_arcmin, theta_kind, theta_value = build_theta_grid(args.grid, cat.get("z"), n)

    per = measure(ymap, mask, cat["ra"], cat["dec"], cat["w"], theta_arcmin,
                  res_arcmin=args.res_arcmin)
    res = stack_with_covariances(per["cap"], per["mask_ok"], cat["w"],
                                 cat["ra"], cat["dec"], n_boot=args.n_boot)
    res_mw = stack_with_covariances(per["cap_mw"], per["mask_ok"], cat["w"],
                                    cat["ra"], cat["dec"], n_boot=args.n_boot)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # R0: per-object caps + gates + positions/weights are ALWAYS stored now
    # (float32, ~20MB per run) so covariances can be re-estimated offline
    # (finer jackknife, Hartlap-aware fits, HOD re-weighting) without
    # re-measuring thumbnails.
    extras = dict(per_obj_cap=per["cap"], per_obj_ok=per["mask_ok"],
                  per_obj_ra=cat["ra"].astype(np.float32),
                  per_obj_dec=cat["dec"].astype(np.float32),
                  per_obj_w=cat["w"].astype(np.float32))
    if "z" in cat:
        extras["per_obj_z"] = np.asarray(cat["z"], dtype=np.float32)
    if args.mode == "dr5":
        extras["fixed_y_c"] = cat["fixed_y_c"]
    np.savez(
        out_path,
        mean=res["mean"], err_boot=res["err_boot"], err_jk=res["err_jk"],
        cov_boot=res["cov_boot"], cov_jk=res["cov_jk"], n_used=res["n_used"],
        n_jk_cells=res["n_jk_cells"], boot_samples=res["boot_samples"],
        mean_maskweighted=res_mw["mean"], err_boot_maskweighted=res_mw["err_boot"],
        err_jk_maskweighted=res_mw["err_jk"],
        theta_kind=theta_kind, theta_value=theta_value,
        theta_arcmin_mean=theta_arcmin.mean(axis=0),
        n_gal=n, in_foot_frac=float(per["in_foot"].mean()),
        mode=args.mode, map_variant=args.map_variant, grid=args.grid,
        zmin=args.zmin, zmax=args.zmax, sample_name=cat["name"],
        logM200_anchor=LOGM200_ANCHOR, res_arcmin=args.res_arcmin,
        wall_time_sec=time.time() - t0,
        **extras,
    )
    print(f"[ycap] {cat['name']}/{args.map_variant}: n={n}, "
          f"in_foot={per['in_foot'].mean():.2%}, "
          f"total={time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
