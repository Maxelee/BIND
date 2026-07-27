#!/usr/bin/env python
"""D2 (docs/tsz_des_data_plan.md §3): DES Y3 data-side extraction.

(a) 2pt leg — pull ξ±, the xip/xim COVMAT blocks, nz_source and the 1000
    nz_source realisations out of the official 2pt FITS into a plain npz.
    The official data vector IS the measurement; nothing is re-measured.

(b) Map leg — tile the Jeffrey+21 mass-map footprint interior with
    non-overlapping 5°×5° gnomonic patches at 1024² px (0.29296875'/px —
    matching the BIND lightcone map geometry exactly), sampled from the
    NSIDE=1024 HEALPix maps with bilinear ``hp.get_interp_val``. A patch is
    accepted when the glimpse_mask, probed on a buffered 64² subgrid
    (±(2.5°+buffer)), is everywhere >0.99 (strict interior, edge-buffered).
    Per-patch stacks are written per map (glimpse/wiener × full/tomo1..4 +
    nullB_full) to KS/lightcone/des_patches/.

Environment note: the plan's §1.5 two-stage (module-python → BIND_env) design
is OBSOLETE — BIND_env's numpy was downgraded to 2.1.3 and healpy 1.18.0 works
(verified in D2); everything runs single-stage in BIND_env.

Reconstruction-filter caveat (must appear on every map-leg figure): the
Jeffrey+21 maps are GLIMPSE/Wiener-filtered reconstructions, not raw KS maps;
map-leg comparisons are consistency-check-grade at smoothing ≥5'.

Usage
-----
    python examples/des_y3_extract.py --twopt          # (a)
    python examples/des_y3_extract.py --patches        # (b)
    python examples/des_y3_extract.py --twopt --patches
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

DL = Path("/mnt/home/mlee1/ceph/paper3/B/downloads")
KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
TWOPT_FITS = DL / "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
MASSMAP_DIR = DL / "des_y3_massmaps_jeffrey2105.13539"
OUT_DIR = KS / "lightcone"
PATCH_DIR = OUT_DIR / "des_patches"

# BIND lightcone map geometry (bind.inference.lux_geometry.DTHETA_RAD)
NPIX = 1024
PIX_ARCMIN = 0.29296875
FOV_DEG = NPIX * PIX_ARCMIN / 60.0            # 5.0 deg
EDGE_BUFFER_DEG = 0.25
DEC_MIN, DEC_MAX = -62.0, 6.0
GRID_STEP_DEG = 5.3                            # > FOV -> non-overlapping

MAPS = (["glimpse_full"] + [f"glimpse_tomo{i}" for i in range(1, 5)]
        + ["wiener_full"] + [f"wiener_tomo{i}" for i in range(1, 5)]
        + ["nullB_full"])

N_NZ_REAL = 1000


# ---------------------------------------------------------------------------
# (a) 2pt extraction
# ---------------------------------------------------------------------------

def extract_twopt(out_path=OUT_DIR / "des_y3_2pt.npz"):
    from astropy.io import fits
    f = fits.open(TWOPT_FITS)
    cov = np.asarray(f["COVMAT"].data, dtype=np.float64)
    hdr = f["COVMAT"].header
    # block layout per plan §1.3: xip[0:200] xim[200:400] gammat[400:880] wtheta[880:1000]
    blocks = {"xip": (0, 200), "xim": (200, 400)}
    out = {"cov_full": cov}
    for name, (lo, hi) in blocks.items():
        d = f[name].data
        out[f"{name}_bin1"] = np.asarray(d["BIN1"], dtype=np.int64)
        out[f"{name}_bin2"] = np.asarray(d["BIN2"], dtype=np.int64)
        out[f"{name}_ang_arcmin"] = np.asarray(d["ANG"], dtype=np.float64)
        out[f"{name}_value"] = np.asarray(d["VALUE"], dtype=np.float64)
        out[f"cov_{name}"] = cov[lo:hi, lo:hi]
    nz = f["nz_source"].data
    out["nz_z_mid"] = np.asarray(nz["Z_MID"], dtype=np.float64)
    out["nz_source"] = np.stack([np.asarray(nz[f"BIN{i}"], dtype=np.float64)
                                 for i in range(1, 5)])
    reals = np.empty((N_NZ_REAL, 4, len(out["nz_z_mid"])), dtype=np.float32)
    for r in range(N_NZ_REAL):
        d = f[f"nz_source_realisation_{r}"].data
        for i in range(1, 5):
            reals[r, i - 1] = d[f"BIN{i}"]
    out["nz_source_realisations"] = reals
    out["covmat_header_note"] = str(hdr.get("COMMENT", ""))
    np.savez(out_path, **out)
    print(f"[2pt] xip/xim (200 each) + cov blocks + nz_source(+{N_NZ_REAL} reals) "
          f"-> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# (b) patch tiling
# ---------------------------------------------------------------------------

def _tangent_lonlat(ra0_deg, dec0_deg, npix, pix_arcmin):
    """(lon, lat) [deg] of an npix² gnomonic tangent grid centred on
    (ra0, dec0) — the act_ycap_measure.batch_thumbnails rotation, reused."""
    off = (np.arange(npix) - (npix - 1) / 2.0) * np.deg2rad(pix_arcmin / 60.0)
    dyg, dxg = np.meshgrid(off, off, indexing="ij")
    x, y = dxg.ravel(), dyg.ravel()
    rho = np.hypot(x, y)
    inv = 1.0 / np.maximum(rho, 1e-30)
    ex, ey = x * inv, y * inv
    cosr, sinr = np.cos(rho), np.sin(rho)
    ext, eyt = ex * sinr, ey * sinr
    dec0 = np.deg2rad(dec0_deg)
    ra0 = np.deg2rad(ra0_deg)
    sd, cd = np.sin(dec0), np.cos(dec0)
    sr, cr = np.sin(ra0), np.cos(ra0)
    px = cd * cr * cosr - sr * ext - sd * cr * eyt
    py = cd * sr * cosr + cr * ext - sd * sr * eyt
    pz = sd * cosr + cd * eyt
    lat = np.rad2deg(np.arcsin(np.clip(pz, -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(py, px)) % 360.0
    return lon, lat


COVERAGE_MIN = 0.96      # tolerated point-source-hole area (mask is Swiss-cheesed:
RING_COVERAGE_MIN = 0.92  # strictly hole-free 5° boxes do NOT exist — verified D2)


def _band_accept(mask_hp, dec0, ra_offset, probe_grid):
    import healpy as hp
    probe_pix, probe_res_arcmin, ring = probe_grid
    out = []
    ra_step = GRID_STEP_DEG / max(np.cos(np.deg2rad(dec0)), 1e-3)
    for ra0 in np.arange(ra_offset, 360.0 + ra_offset, ra_step) % 360.0:
        lon, lat = _tangent_lonlat(ra0, dec0, probe_pix, probe_res_arcmin)
        v = hp.get_interp_val(mask_hp, lon, lat, lonlat=True)
        cov = float((v > 0.99).mean())
        ring_cov = float((v[ring] > 0.99).mean())
        if cov >= COVERAGE_MIN and ring_cov >= RING_COVERAGE_MIN:
            out.append((ra0, dec0, cov))
    return out


def find_patch_centers(mask_hp, verbose=True):
    """Grid-tile the footprint. The glimpse_mask has arcmin-scale star/point-
    source holes everywhere (a strict interior test accepts 0 patches), so
    acceptance is: buffered-box coverage >= COVERAGE_MIN AND outer-ring
    coverage >= RING_COVERAGE_MIN (a survey-boundary intrusion knocks out a
    contiguous ring chunk; isolated holes don't). Because a rigid grid wastes
    the irregular footprint (9 patches), each dec band scans 12 RA-phase
    offsets and 4 global dec phases are tried, keeping the best packing —
    still non-overlapping by construction (fixed step > FOV). Per-patch
    coverage is recorded for downstream honesty."""
    half = FOV_DEG / 2.0 + EDGE_BUFFER_DEG
    probe_pix = 64
    probe_res_arcmin = 2 * half * 60.0 / probe_pix
    off = np.arange(probe_pix) - (probe_pix - 1) / 2.0
    rr = np.hypot(*np.meshgrid(off, off)) * (2 * half / probe_pix)
    probe_grid = (probe_pix, probe_res_arcmin, (rr > 2.3).ravel())

    best = []
    for dec_phase in (0.0, 1.325, 2.65, 3.975):
        got = []
        for dec0 in np.arange(DEC_MIN + dec_phase, DEC_MAX, GRID_STEP_DEG):
            ra_step = GRID_STEP_DEG / max(np.cos(np.deg2rad(dec0)), 1e-3)
            band_best = []
            for ra_phase in np.linspace(0.0, ra_step, 12, endpoint=False):
                cand = _band_accept(mask_hp, dec0, ra_phase, probe_grid)
                if len(cand) > len(band_best):
                    band_best = cand
            got.extend(band_best)
        if len(got) > len(best):
            best = got
    centers = np.array([(ra, dec) for ra, dec, _ in best])
    coverages = np.array([c for _, _, c in best])
    if verbose:
        print(f"[patches] accepted {len(centers)} non-overlapping "
              f"{FOV_DEG:g}° patches (coverage>={COVERAGE_MIN}, "
              f"ring>={RING_COVERAGE_MIN}, buffer {EDGE_BUFFER_DEG}°)")
    return centers, coverages


def extract_patches(centers=None):
    import healpy as hp
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    mask = hp.read_map(str(MASSMAP_DIR / "glimpse_mask.fits"))
    if centers is None:
        centers, coverages = find_patch_centers(mask)
    if len(centers) == 0:
        raise SystemExit("no patches accepted — acceptance rule broken?")
    np.savez(PATCH_DIR / "patch_centers.npz", centers=centers,
             coverages=coverages, fov_deg=FOV_DEG, npix=NPIX,
             pix_arcmin=PIX_ARCMIN, edge_buffer_deg=EDGE_BUFFER_DEG)
    for name in MAPS:
        out_path = PATCH_DIR / f"{name}_patches.npz"
        if out_path.exists():
            print(f"[patches] exists, skipping {name}")
            continue
        t0 = time.time()
        m = hp.read_map(str(MASSMAP_DIR / f"{name}.fits"))
        unseen = m < -1e20
        if name.startswith("glimpse"):
            # GLIMPSE leaves star/point-source holes UNSEEN (verified: 100%
            # of interior holes); fill from the release's own Wiener
            # reconstruction of the SAME tomographic bin so hole pixels are
            # data-consistent rather than zero. The GLIMPSE-Wiener spread
            # band (D2 caveat) then also brackets this inpainting choice.
            wname = name.replace("glimpse", "wiener")
            wmap = hp.read_map(str(MASSMAP_DIR / f"{wname}.fits"))
            m = np.where(unseen, wmap, m)
        else:
            m = np.where(unseen, 0.0, m)
        stack = np.empty((len(centers), NPIX, NPIX), dtype=np.float32)
        holefrac = np.empty(len(centers), dtype=np.float32)
        for k, (ra0, dec0) in enumerate(centers):
            lon, lat = _tangent_lonlat(ra0, dec0, NPIX, PIX_ARCMIN)
            stack[k] = hp.get_interp_val(m, lon, lat, lonlat=True).reshape(NPIX, NPIX)
            holefrac[k] = 1.0 - hp.get_interp_val(mask, lon, lat, lonlat=True).mean()
        np.savez(out_path, patches=stack, centers=centers, hole_frac=holefrac,
                 fov_deg=FOV_DEG, pix_arcmin=PIX_ARCMIN)
        print(f"[patches] {name}: {len(centers)} patches "
              f"(hole_frac med {np.median(holefrac):.3f}) in {time.time()-t0:.0f}s "
              f"-> {out_path}", flush=True)
    return centers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twopt", action="store_true")
    ap.add_argument("--patches", action="store_true")
    args = ap.parse_args()
    if args.twopt:
        extract_twopt()
    if args.patches:
        extract_patches()
    if not (args.twopt or args.patches):
        raise SystemExit("nothing to do: pass --twopt and/or --patches")


if __name__ == "__main__":
    main()
