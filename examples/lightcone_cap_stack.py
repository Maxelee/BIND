#!/usr/bin/env python
"""
E3 (P4b plan, phase P4): CAP-filter stacking of DESI mock galaxies on the
ray-traced SB35 lightcone maps (tau / y / kappa).

For one (run, map, sample) triple: load the run's ``{map}_maps.npz`` cube
once, select the sample's galaxies (BGS via a stellar-mass cut on the
``mstar_matrix_snap085.npz`` column for that run, or the fiducial's own
central-M* computed on the fly from ``FID/snap_085/composite_slab*.npz``
with the ``_p3_mstar_sweep.py`` convention; ELG via the fixed
``elg_sel`` mass-proxy flag), then for every realization r=1..50 recover
every periodic-copy map pixel of every selected (``in_crop``) halo via
``bind.inference.lux_geometry`` (E1) and measure the CAP filter
(disk r<theta_d minus equal-area ring theta_d<r<sqrt2*theta_d, exactly
``examples/_reduce_fgas_lightcone.py::cap_on_map``'s formula) at two theta
grids:

  (i)  the paper's reference ``xb`` grid (``examples/_reduce_fgas_cap.py``:
       ``XB = linspace(0.3, 3.0, 18)``, dimensionless theta_d/r200 multiples)
  (ii) a coarser ``theta(r200)``-scaled grid ``[0.25, 0.5, 0.75, 1.0, 1.4]``

Both grids are per-halo: theta_d(halo, k) = grid_value(k) * theta200(halo),
theta200 = r200_comoving / chi[plane_p] (radians -> arcmin -> map px via
the fixed 0.29297'/px map pixel scale, ``bind.inference.lux_geometry.
DTHETA_RAD``). Copies of one halo (periodic tiling at high z) are averaged
first; the per-halo CAP values are then stacked (mean over galaxies) per
realization, giving a (50, n_theta) array per shard.

Performance note: the disk/ring pixel masks depend only on the *relative*
offset from a halo's (nearest-pixel) center, not on the center itself, so
for a single global cutout half-size ``w`` (set by the largest theta_d in
the selected sample) the radial sort order is identical for every halo.
This lets the whole per-realization CAP measurement run as a handful of
vectorized numpy ops (batched fancy-indexing gather + shared
``argsort``/``cumsum`` + vectorized ``searchsorted``) instead of a
per-halo Python loop; only halos whose full cutout would fall outside the
1024x1024 map (a minority, near the FOV edge) fall back to a slower
per-halo direct calculation (mirrors ``cap_on_map``'s own boundary clip).

Output shard (idempotent -- skipped if it already exists, use --force to
overwrite):
    KS/lightcone/shards/{sample}_{map}_run{run}_beam{beam}_src{src_idx}.npz
    keys: stack_real (50, n_theta), mean (n_theta,), std (n_theta,),
    n_gal (total selected before per-realization FOV landing), n_gal_real
    (50,) (galaxies with >=1 valid copy landing that realization),
    theta_kind (n_theta,) <U8 ('xb'|'r200mult'), theta_value (n_theta,),
    sample, run, map, src_idx, beam_fwhm_arcmin, wall_time_sec.

``--map massplane`` (P6b addition, docs/ksz_lightcone_map_plan.md P6-stage-B):
a 5th, fiducial-ONLY map type for the f~gas=CAP_tau(gas)/CAP_mat denominator.
Reads ``FID/haloplane_trace/massplane_maps.npz`` (key ``'tau'`` -- the
total-matter column was painted into the tau slot before the fiducial
co-trace, see P2/plan §1.3) instead of ``{map}_maps.npz``, regardless of
``--run`` (always the fiducial geometry; pass ``--run fid`` by convention).
That file has only 47 realizations (``n_real=47``, the co-trace's completed
subset -- P2 verdict), so ``--n_realizations`` is silently capped to
``min(requested, 47)``; the 47 realizations use the identical per-realization
(rot, disp) transforms as realizations 1..47 of the regular maps (same seed
1992, same config.dat, kappa-identity bit-exact -- P2 certified), so mixing a
47-realization matter stack with a 50-realization gas stack introduces no
geometry inconsistency, only 3 fewer realizations of averaging. Because the
matter column is the SAME fixed fiducial painting for every Sobol node (mass
is feedback-insensitive at these apertures -- P2/plan), only ONE massplane
shard per sample is ever needed; it is not part of the 256-node sweep.

``--sample massbin85`` / ``massbin46`` (M5 addition, docs/ksz_lightcone_map_plan.md
phase M5): the map-level recreation of the per-halo headline figure
(``_build_ksz_paper_nb.py`` SS6b ``f6b_lowmass``), f~gas vs logM200 in 7 fixed
mass bins instead of one M*-selected sample. ``massbin85`` reads
``KS/lightcone/catalogs/desi_mock_massbin_snap085.npz`` (snap 85, z=0.18, BGS
shell, M200>=1e12, 28919 halos); ``massbin46`` reads the *existing*
``desi_mock_snap046.npz`` (snap 46, z=1.16, ELG shell -- already P3-complete)
joined by row position with the companion ``desi_mock_snap046_massbin.npz``
for its ``mass_bin`` column (both files share row order/``fof_index`` by
construction -- see that npz's own ``note`` field; joined and asserted here,
not merged into either file). Selection is a **pure FoF M200 cut** -- unlike
bgs110/bgs1125 it never depends on ``--run``/the Sobol node's painted stellar
mass, and unlike elg/lrg it is not one pooled sample but 7 mass bins (edges
``desi_mock_massbin_snap085.npz['mass_bin_edges']`` = ``[12.0, 12.33, 12.66,
13.0, 13.4, 13.8, 14.2, 14.8]``, matching ``examples/_reduce_fgas_lightcone.
py::MB``). Three differences from the pooled-sample path above:

  (a) **Per-bin stacking.** Shard arrays gain a leading bin axis:
      ``stack_real`` is ``(n_bins, n_real, n_theta)`` and ``n_gal`` is
      ``(n_bins,)`` (see :func:`compute_stack_massbin`), by looping the
      existing per-halo :func:`compute_stack` once per bin -- the CAP math
      itself (:func:`cap_batch`) is untouched.
  (b) **Aperture grid.** The standard per-halo theta200-multiples grid
      (:func:`theta_pix_grid`, unchanged) PLUS one extra **fixed** aperture
      column, ``theta_d = FIXED_APERTURE_ARCMIN = 1.0'`` for *every* halo
      regardless of its own r200 (marked ``'fixed_arcmin'`` in
      ``theta_kind``) -- this is the ELG data's own innermost measured
      aperture (``m5_data_points.npz['elg_theta_data_arcmin']`` = 1.0' for
      all 3 cuts), needed because ELG hosts never reach their own r200 in
      the data. See :func:`theta_pix_grid_massbin`.
  (c) **in_patch substack.** For the 3 bins entirely below the 1e13 painting
      floor (upper edge <= 13.0 -- BIND painted only M200>=1e13 halos as
      *centrals*; below that a halo's field is physical only via patch
      *reuse*, memory ``lowmass-reuse-capture``), a parallel in_patch-only
      substack (``stack_real_inpatch`` etc., NaN for the >=1e13 bins) is
      also recorded, so the M5 figure can show the reuse-regime honestly
      alongside the full (reuse+capture) stack.

The mass-binned **CAP_mat denominator** (the massplane field, ``--map
massplane --sample massbin85/46``) reuses this exact same per-bin selection
and grid -- so the numerator (this file, ``--map tau``) and denominator
(this file, ``--map massplane``) are stacked on IDENTICAL halo samples per
bin *by construction*: both read the same ``mass_bin`` column from the same
catalog file, with no node-dependent re-selection anywhere in the chain.
This is automatic here (unlike bgs110/bgs1125, which needed the P6b
per-node CAP_mat fix in ``lightcone_capmat_merge.py`` precisely because
that sample's selection DOES depend on the node) because a pure FoF M200
cut never varies with the Sobol feedback parameters -- see
``lightcone_capmat_merge.py::main_massbin`` for the merge step.

Usage
-----
    python examples/lightcone_cap_stack.py --run fid --map tau --sample bgs110
    python examples/lightcone_cap_stack.py --run 0 --map tau --sample bgs110 \
        --beam_fwhm_arcmin 1.6
    python examples/lightcone_cap_stack.py --run fid --map massplane --sample bgs110
    python examples/lightcone_cap_stack.py --run fid --map tau --sample massbin85
    python examples/lightcone_cap_stack.py --run fid --map massplane --sample massbin46
    python examples/lightcone_cap_stack.py --help
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")

from bind.inference.lux_geometry import (
    DTHETA_RAD,
    NATIVE_PIXEL_SIZE_MPCH,
    N_REALIZATIONS,
    RT_GRID,
    load_geometry,
    plane_to_map,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LIGHTCONE = KS / "lightcone"
GEOM_PATH = LIGHTCONE / "geometry.npz"
CAT_DIR = LIGHTCONE / "catalogs"
SHARD_DIR_DEFAULT = LIGHTCONE / "shards"

FID_ROOT = CEPH / "bind_lightcone_tng"
RUNS_ROOT = CEPH / "bind_sb35/runs"
MASSPLANE_PATH = FID_ROOT / "haloplane_trace" / "massplane_maps.npz"
MASSPLANE_KEY = "tau"  # total-matter column was painted into the tau slot (P2)

# theta grids (see module docstring). XB matches examples/_reduce_fgas_cap.py.
XB = np.linspace(0.3, 3.0, 18)
R200_MULT = np.array([0.25, 0.5, 0.75, 1.0, 1.4])
N_THETA = len(XB) + len(R200_MULT)

DTHETA_ARCMIN = np.degrees(DTHETA_RAD) * 60.0  # map pixel scale, 0.29297'/px

MSTAR_AP_KPCH = 50.0  # central-M* aperture, matches _p3_mstar_sweep.py
MSTAR_CUTS = {"bgs110": 11.0, "bgs1125": 11.25}

SQ2 = np.sqrt(2.0)
MAX_W_PX = 400  # safety cap on the shared cutout half-size


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def _fiducial_bgs_logmstar(catalog: dict) -> np.ndarray:
    """Central M* (log10 Msun/h, 50 kpc/h aperture) for every BGS catalog row,
    computed directly from FID/snap_085/composite_slab*.npz, replicating
    examples/_p3_mstar_sweep.py::central_mstar exactly (Stars channel,
    rr_kpch < 50, log10(max(sum, 1)))."""
    slab = catalog["slab"]
    idx_in_slab = catalog["mstar_idx_in_slab"]
    out = np.full(len(slab), np.nan, dtype=np.float32)
    for s in range(4):
        f = FID_ROOT / "snap_085" / f"composite_slab{s:02d}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        if int(d["n_halos"]) == 0:
            continue
        stars = d["generated_patches"][:, 2]  # (n, 128, 128), Msun/h per pixel
        P = stars.shape[-1]
        cc = P // 2
        yy, xx = np.mgrid[0:P, 0:P]
        rr_kpch = np.hypot(xx - cc, yy - cc) * NATIVE_PIXEL_SIZE_MPCH * 1e3
        ap = rr_kpch < MSTAR_AP_KPCH
        central = (stars * ap[None, :, :]).sum(axis=(1, 2))
        log_c = np.log10(np.maximum(central, 1.0)).astype(np.float32)

        sel = slab == s
        rows = idx_in_slab[sel]
        valid = (rows >= 0) & (rows < len(log_c))
        out_sel = out[sel]
        out_sel[valid] = log_c[rows[valid]]
        out[sel] = out_sel
    return out


def select_sample(sample: str, run: str):
    """Returns dict(pixel_i, pixel_j, plane_p, r200, logM200, extra) for the
    selected + in_crop halos of `sample` under Sobol node `run` ('fid' or
    an int 0..255)."""
    if sample in MSTAR_CUTS:
        cat = np.load(CAT_DIR / "desi_mock_snap085.npz", allow_pickle=True)
        cut = MSTAR_CUTS[sample]
        if run == "fid":
            log_mstar = _fiducial_bgs_logmstar(cat)
        else:
            mm = np.load(CAT_DIR / "mstar_matrix_snap085.npz", allow_pickle=True)
            run_ids = mm["run_ids"]
            col = np.nonzero(run_ids == int(run))[0]
            if len(col) == 0:
                raise SystemExit(f"run {run} not found in mstar_matrix_snap085.npz run_ids")
            log_mstar = mm["log_mstar_central"][:, col[0]]
        mask = (log_mstar > cut) & cat["in_crop"] & np.isfinite(log_mstar)
        extra = {
            "slab": cat["slab"][mask].astype(np.int64),
            "mstar_idx_in_slab": cat["mstar_idx_in_slab"][mask].astype(np.int64),
        }
    elif sample == "elg":
        cat = np.load(CAT_DIR / "desi_mock_snap046.npz", allow_pickle=True)
        mask = cat["elg_sel"] & cat["in_crop"]
        extra = {"in_patch": cat["in_patch"][mask]}
    elif sample == "lrg":
        # P3-LRG extension (2026-07-24): mass-proxy selection at snap 67
        # (z=0.50), built by examples/lightcone_desi_catalog.py --lrg_snap 67.
        # window-tuned to match Liu+2025's DESI-LRG lensing host mass
        # (logM200~13.18); all halos >=1e13 -- no in_patch flag needed (unlike
        # ELG, these are all BIND-painted, no patch-reuse regime).
        cat = np.load(CAT_DIR / "desi_mock_snap067.npz", allow_pickle=True)
        mask = cat["lrg_sel"] & cat["in_crop"]
        extra = {}
    else:
        raise SystemExit(f"unknown --sample {sample!r} (expected bgs110|bgs1125|elg|lrg)")

    return {
        "pixel_i": cat["pixel_i"][mask].astype(np.float64),
        "pixel_j": cat["pixel_j"][mask].astype(np.float64),
        "plane_p": cat["plane_p"][mask].astype(np.int64),
        "r200": cat["r200"][mask].astype(np.float64),
        "logM200": cat["logM200"][mask].astype(np.float64),
        "n_total": int(len(cat["plane_p"])),
        "n_sel": int(mask.sum()),
        **extra,
    }


def theta_pix_grid(r200: np.ndarray, plane_p: np.ndarray, geom) -> np.ndarray:
    """(n_sel, N_THETA) map-pixel theta_d grid: columns 0:18 = XB*theta200,
    18:23 = R200_MULT*theta200; theta200 = r200/chi[plane_p] (radians)."""
    chi_p = geom.chi[plane_p]
    theta200_arcmin = np.degrees(r200 / chi_p) * 60.0  # (n_sel,)
    grid_mult = np.concatenate([XB, R200_MULT])  # (N_THETA,)
    theta_arcmin = theta200_arcmin[:, None] * grid_mult[None, :]
    return theta_arcmin / DTHETA_ARCMIN  # -> map pixels


# ---------------------------------------------------------------------------
# Mass-binned sample selection (M5 addition -- see module docstring)
# ---------------------------------------------------------------------------

MASSBIN_SAMPLES = {"massbin85", "massbin46"}
MASSBIN_SNAP_IDX = {"massbin85": 2, "massbin46": 12}  # geometry.npz snap_idx (§1.2 Step A)
IN_PATCH_BIN_MAX_LOGM = 13.0  # bins with upper edge <= this are the "sub-1e13" reuse regime

# The ELG data's own innermost measured aperture (m5_data_points.npz:
# elg_theta_data_arcmin == 1.0' for all 3 M*-cuts -- ELG hosts never reach
# their own r200 in the data). Used as a fixed (non-r200-scaled) extra
# theta column so one shard covers both the BGS r200-matched panel and the
# ELG fixed-aperture panel.
FIXED_APERTURE_ARCMIN = 1.0

THETA_KIND_MASSBIN = np.concatenate(
    [np.array(["xb"] * len(XB) + ["r200mult"] * len(R200_MULT)), ["fixed_arcmin"]])
THETA_VALUE_MASSBIN = np.concatenate([XB, R200_MULT, [FIXED_APERTURE_ARCMIN]])
N_THETA_MASSBIN = N_THETA + 1


def select_sample_massbin(sample: str, run: str):
    """Mass-binned catalog selection (M5 plan). Returns every in_crop,
    in-range (mass_bin >= 0) halo of the sample's snapshot shell together
    with its mass_bin index -- selection is a pure FoF M200 cut, so unlike
    :func:`select_sample`'s bgs110/bgs1125 branch it does NOT depend on
    `run` (accepted only for CLI symmetry)."""
    if sample == "massbin85":
        cat = np.load(CAT_DIR / "desi_mock_massbin_snap085.npz", allow_pickle=True)
        mass_bin = cat["mass_bin"]
        mass_bin_edges = cat["mass_bin_edges"]
        in_patch = cat["in_patch"]
    elif sample == "massbin46":
        # desi_mock_snap046.npz is the existing (P3-complete) ELG-shell
        # catalog; it has no mass_bin column, so join by row position with
        # the M5-prep companion npz (row order + fof_index identical by
        # construction -- see that file's own `note` field). Asserted, not
        # assumed, in case either file is ever rebuilt independently.
        cat = np.load(CAT_DIR / "desi_mock_snap046.npz", allow_pickle=True)
        mb = np.load(CAT_DIR / "desi_mock_snap046_massbin.npz", allow_pickle=True)
        if not np.array_equal(cat["fof_index"], mb["fof_index"]):
            raise SystemExit(
                "desi_mock_snap046.npz / desi_mock_snap046_massbin.npz fof_index "
                "mismatch -- row-position join is no longer valid, re-join by fof_index")
        mass_bin = mb["mass_bin"]
        mass_bin_edges = mb["mass_bin_edges"]
        in_patch = cat["in_patch"]
    else:
        raise SystemExit(f"unknown massbin sample {sample!r} (expected massbin85|massbin46)")

    mask = cat["in_crop"] & (mass_bin >= 0)
    return {
        "pixel_i": cat["pixel_i"][mask].astype(np.float64),
        "pixel_j": cat["pixel_j"][mask].astype(np.float64),
        "plane_p": cat["plane_p"][mask].astype(np.int64),
        "r200": cat["r200"][mask].astype(np.float64),
        "logM200": cat["logM200"][mask].astype(np.float64),
        "mass_bin": mass_bin[mask].astype(np.int64),
        "in_patch": in_patch[mask].astype(bool),
        "mass_bin_edges": np.asarray(mass_bin_edges, dtype=np.float64),
        "n_total": int(len(cat["plane_p"])),
        "n_sel": int(mask.sum()),
    }


def theta_pix_grid_massbin(r200: np.ndarray, plane_p: np.ndarray, geom) -> np.ndarray:
    """theta grid for the massbin samples: :func:`theta_pix_grid`'s standard
    per-halo theta200-multiples grid PLUS one extra fixed-arcmin column
    (theta_d = FIXED_APERTURE_ARCMIN for every halo, independent of r200).
    Returns (n_sel, N_THETA_MASSBIN)."""
    base = theta_pix_grid(r200, plane_p, geom)  # (n_sel, N_THETA)
    fixed_px = np.full((len(r200), 1), FIXED_APERTURE_ARCMIN / DTHETA_ARCMIN)
    return np.concatenate([base, fixed_px], axis=1)


# ---------------------------------------------------------------------------
# CAP filter on the map (vectorized fast path + per-halo fallback)
# ---------------------------------------------------------------------------

def _make_window(w: int):
    rel = np.arange(-w, w + 1, dtype=np.int64)
    di, dj = np.meshgrid(rel, rel, indexing="ij")
    r = np.hypot(di, dj).ravel()
    order = np.argsort(r, kind="stable")
    return rel.astype(np.int32), order, r[order]


def _cap_direct(map2d: np.ndarray, ic: int, jc: int, thetas: np.ndarray) -> np.ndarray:
    """Per-halo fallback: literal cap_on_map formula (boundary-clipped),
    used only for cutouts that would spill off the 1024x1024 map."""
    P0, P1 = map2d.shape
    w = int(np.ceil(SQ2 * float(np.max(thetas)))) + 2
    i0, i1 = max(0, ic - w), min(P0, ic + w + 1)
    j0, j1 = max(0, jc - w), min(P1, jc + w + 1)
    out = np.full(len(thetas), np.nan)
    if i1 - i0 < 4 or j1 - j0 < 4:
        return out
    sub = map2d[i0:i1, j0:j1]
    ii, jj = np.mgrid[i0:i1, j0:j1]
    r = np.hypot(ii - ic, jj - jc)
    for m, th in enumerate(thetas):
        disk = r < th
        ring = (r >= th) & (r < SQ2 * th)
        nd, nr = int(disk.sum()), int(ring.sum())
        if nd < 3 or nr < 5:
            continue
        w_area = nd / nr
        out[m] = float(sub[disk].sum() - sub[ring].sum() * w_area)
    return out


def cap_batch(map2d: np.ndarray, ic: np.ndarray, jc: np.ndarray,
              thetas: np.ndarray) -> np.ndarray:
    """CAP(theta) for every (copy) row. ic, jc: (N,) nearest-pixel integer
    centers. thetas: (N, N_THETA) per-row theta_d grid (map px).
    Returns (N, N_THETA), NaN where the disk/ring gate fails."""
    N = len(ic)
    out = np.full((N, thetas.shape[1]), np.nan)
    if N == 0:
        return out
    P0, P1 = map2d.shape
    w = int(min(MAX_W_PX, np.ceil(SQ2 * float(thetas.max())) + 2))
    i0 = ic - w
    i1 = ic + w + 1
    j0 = jc - w
    j1 = jc + w + 1
    full_ok = (i0 >= 0) & (i1 <= P0) & (j0 >= 0) & (j1 <= P1) & (thetas.max(axis=1) <= w - 2)

    idx_fast = np.nonzero(full_ok)[0]
    if len(idx_fast):
        rel, order, rs_sorted = _make_window(w)
        i_idx = ic[idx_fast, None, None] + rel[None, :, None]
        j_idx = jc[idx_fast, None, None] + rel[None, None, :]
        sub = map2d[i_idx, j_idx]                       # (Nf, 2w+1, 2w+1)
        Nf = len(idx_fast)
        vs = sub.reshape(Nf, -1)[:, order].astype(np.float64)
        cs = np.cumsum(vs, axis=1)                       # (Nf, Wtot)
        Wtot = cs.shape[1]

        th = thetas[idx_fast]                             # (Nf, N_THETA)
        i_d = np.searchsorted(rs_sorted, th, side="left")
        i_r = np.searchsorted(rs_sorted, SQ2 * th, side="left")

        row = np.arange(Nf)[:, None]
        idx_d = np.clip(i_d - 1, 0, Wtot - 1)
        idx_r = np.clip(i_r - 1, 0, Wtot - 1)
        disk_sum = np.where(i_d > 0, cs[row, idx_d], 0.0)
        above = np.where(i_r > 0, cs[row, idx_r], 0.0)
        ring_sum = above - disk_sum
        n_ring = i_r - i_d
        valid = (i_d >= 3) & (n_ring >= 5) & (i_r <= Wtot)
        wgt = np.where(n_ring > 0, i_d / np.maximum(n_ring, 1), 0.0)
        cap = disk_sum - ring_sum * wgt
        out[idx_fast] = np.where(valid, cap, np.nan)

    idx_slow = np.nonzero(~full_ok)[0]
    for k in idx_slow:
        out[k] = _cap_direct(map2d, int(ic[k]), int(jc[k]), thetas[k])

    return out


# ---------------------------------------------------------------------------
# Per-realization stack, full run
# ---------------------------------------------------------------------------

def compute_stack(map_cube: np.ndarray, geom, pixel_i, pixel_j, plane_p,
                   theta_grid: np.ndarray, n_realizations: int = N_REALIZATIONS,
                   return_per_halo: bool = False, verbose: bool = False):
    """map_cube: (n_real, 1024, 1024) single-source-plane slice.
    Returns (stack_real (n_real, N_THETA), n_gal_real (n_real,)[, per_halo_list])."""
    n_sel = len(pixel_i)
    n_theta = theta_grid.shape[1]
    stack_real = np.full((n_realizations, n_theta), np.nan)
    n_gal_real = np.zeros(n_realizations, dtype=np.int64)
    per_halo_list = [] if return_per_halo else None

    for r in range(1, n_realizations + 1):
        copies = plane_to_map(pixel_i, pixel_j, plane_p, r, geom)
        halo_idx_list, i_map_list, j_map_list = [], [], []
        for h, rows in enumerate(copies):
            if rows.shape[0] == 0:
                continue
            halo_idx_list.append(np.full(rows.shape[0], h, dtype=np.int64))
            i_map_list.append(rows[:, 0])
            j_map_list.append(rows[:, 1])
        if not halo_idx_list:
            if return_per_halo:
                per_halo_list.append(np.full((n_sel, n_theta), np.nan))
            continue
        halo_idx = np.concatenate(halo_idx_list)
        i_map = np.concatenate(i_map_list)
        j_map = np.concatenate(j_map_list)
        ic = np.clip(np.round(i_map), 0, RT_GRID - 1).astype(np.int64)
        jc = np.clip(np.round(j_map), 0, RT_GRID - 1).astype(np.int64)

        map2d = map_cube[r - 1]
        cap_copy = cap_batch(map2d, ic, jc, theta_grid[halo_idx])

        valid = ~np.isnan(cap_copy)
        sums = np.zeros((n_sel, n_theta))
        cnts = np.zeros((n_sel, n_theta))
        np.add.at(sums, halo_idx, np.where(valid, cap_copy, 0.0))
        np.add.at(cnts, halo_idx, valid.astype(np.float64))
        per_halo = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)

        with np.errstate(invalid="ignore"):
            stack_real[r - 1] = np.nanmean(per_halo, axis=0)
        n_gal_real[r - 1] = int(np.sum(~np.all(np.isnan(per_halo), axis=1)))
        if return_per_halo:
            per_halo_list.append(per_halo)
        if verbose:
            print(f"    realization {r}/{n_realizations}: n_landing={n_gal_real[r-1]}", flush=True)

    if return_per_halo:
        return stack_real, n_gal_real, per_halo_list
    return stack_real, n_gal_real


def compute_stack_massbin(map_cube: np.ndarray, geom, sel: dict, theta_grid: np.ndarray,
                           n_realizations: int, verbose: bool = False) -> dict:
    """Per-mass-bin CAP stack (M5 plan). Loops the 7 mass bins, calling the
    existing per-halo :func:`compute_stack` on each bin's halo subset --
    reuses the same vectorized :func:`cap_batch`/``plane_to_map`` machinery
    unchanged, just partitioned by bin instead of pooled into one sample.
    For bins entirely below the 1e13 painting floor (upper edge <=
    IN_PATCH_BIN_MAX_LOGM) also computes an in_patch-only substack (parallel
    array, NaN elsewhere) -- reuse-regime honesty, see module docstring."""
    edges = sel["mass_bin_edges"]
    n_bins = len(edges) - 1
    n_theta = theta_grid.shape[1]
    mb = sel["mass_bin"]

    stack_real = np.full((n_bins, n_realizations, n_theta), np.nan)
    n_gal_real = np.zeros((n_bins, n_realizations), dtype=np.int64)
    n_gal = np.zeros(n_bins, dtype=np.int64)
    logM200_mean = np.full(n_bins, np.nan)

    stack_real_ip = np.full((n_bins, n_realizations, n_theta), np.nan)
    n_gal_real_ip = np.zeros((n_bins, n_realizations), dtype=np.int64)
    n_gal_ip = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask_b = mb == b
        nb = int(mask_b.sum())
        n_gal[b] = nb
        if nb == 0:
            continue
        logM200_mean[b] = float(np.mean(sel["logM200"][mask_b]))
        sr, ngr = compute_stack(
            map_cube, geom, sel["pixel_i"][mask_b], sel["pixel_j"][mask_b],
            sel["plane_p"][mask_b], theta_grid[mask_b],
            n_realizations=n_realizations, verbose=False)
        stack_real[b] = sr
        n_gal_real[b] = ngr

        if edges[b + 1] <= IN_PATCH_BIN_MAX_LOGM:
            mask_ip = mask_b & sel["in_patch"]
            nip = int(mask_ip.sum())
            n_gal_ip[b] = nip
            if nip > 0:
                sr_ip, ngr_ip = compute_stack(
                    map_cube, geom, sel["pixel_i"][mask_ip], sel["pixel_j"][mask_ip],
                    sel["plane_p"][mask_ip], theta_grid[mask_ip],
                    n_realizations=n_realizations, verbose=False)
                stack_real_ip[b] = sr_ip
                n_gal_real_ip[b] = ngr_ip
        if verbose:
            print(f"    bin {b} [{edges[b]:.2f},{edges[b+1]:.2f}): n_gal={nb} "
                  f"n_gal_inpatch={int(n_gal_ip[b])}", flush=True)

    return dict(stack_real=stack_real, n_gal_real=n_gal_real, n_gal=n_gal,
                logM200_mean=logM200_mean, stack_real_inpatch=stack_real_ip,
                n_gal_real_inpatch=n_gal_real_ip, n_gal_inpatch=n_gal_ip)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def map_path_for_run(run: str, map_type: str) -> Path:
    if map_type == "massplane":
        return MASSPLANE_PATH  # fiducial-only, shared across all runs/nodes
    if run == "fid":
        return FID_ROOT / f"{map_type}_maps.npz"
    return RUNS_ROOT / f"run_{int(run):04d}" / f"{map_type}_maps.npz"


def shard_path(out_dir: Path, sample: str, map_type: str, run: str,
               beam_fwhm_arcmin, src_idx: int) -> Path:
    run_tag = "fid" if run == "fid" else f"{int(run):04d}"
    beam_tag = "none" if beam_fwhm_arcmin is None else f"{beam_fwhm_arcmin:g}am"
    return out_dir / f"{sample}_{map_type}_run{run_tag}_beam{beam_tag}_src{src_idx}.npz"


def apply_beam(cube: np.ndarray, fwhm_arcmin: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    sigma_px = fwhm_arcmin / (2.0 * np.sqrt(2.0 * np.log(2.0))) / DTHETA_ARCMIN
    return gaussian_filter(cube, sigma=(0.0, sigma_px, sigma_px), mode="nearest")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="'fid' or a Sobol node 0..255")
    ap.add_argument("--map", required=True, choices=["tau", "y", "kappa", "massplane"], dest="map_type")
    ap.add_argument("--sample", required=True,
                     choices=["bgs110", "bgs1125", "elg", "lrg", "massbin85", "massbin46"])
    ap.add_argument("--beam_fwhm_arcmin", type=float, default=None)
    ap.add_argument("--src_idx", type=int, default=4)
    ap.add_argument("--out_dir", type=Path, default=SHARD_DIR_DEFAULT)
    ap.add_argument("--n_realizations", type=int, default=N_REALIZATIONS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def run_one_massbin(run: str, map_type: str, sample: str, *, beam_fwhm_arcmin=None,
                     src_idx: int = 4, out_dir: Path = SHARD_DIR_DEFAULT,
                     n_realizations: int = N_REALIZATIONS, force: bool = False,
                     verbose: bool = False) -> Path:
    """Mass-binned stacking mode (M5 plan) -- see module docstring for the
    three differences from :func:`run_one`'s pooled-sample path. Because
    :func:`select_sample_massbin`'s selection never depends on `run`, this
    is called with ``--run fid --map tau`` for the numerator and with
    ``--map massplane`` (any `run`, by :func:`map_path_for_run`'s
    fiducial-only convention) for the shared CAP_mat denominator -- the two
    are therefore always stacked on IDENTICAL per-bin halo samples (same
    catalog, same mass_bin mask), unlike bgs110/bgs1125 which needed the
    P6b per-node CAP_mat fix precisely because that selection DOES depend
    on the node."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = shard_path(out_dir, sample, map_type, run, beam_fwhm_arcmin, src_idx)
    if out_path.exists() and not force:
        if verbose:
            print(f"[cap_stack] shard exists, skipping: {out_path}")
        return out_path

    t0 = time.time()
    geom = load_geometry(GEOM_PATH)
    sel = select_sample_massbin(sample, run)
    if sel["n_sel"] == 0:
        raise SystemExit(f"no halos selected for sample={sample} run={run}")
    tgrid = theta_pix_grid_massbin(sel["r200"], sel["plane_p"], geom)

    mpath = map_path_for_run(run, map_type)
    if verbose:
        print(f"[cap_stack] loading {mpath} (src_idx={src_idx}) ...", flush=True)
    d = np.load(mpath)
    in_file_key = MASSPLANE_KEY if map_type == "massplane" else map_type
    cube = np.array(d[in_file_key][:, src_idx])  # (n_real, 1024, 1024)
    del d
    if cube.shape[0] < n_realizations:
        if verbose:
            print(f"[cap_stack] {map_type}: only {cube.shape[0]} realizations on disk "
                  f"(requested {n_realizations}) -- capping.", flush=True)
        n_realizations = cube.shape[0]
    t_load = time.time()

    if beam_fwhm_arcmin is not None:
        cube = apply_beam(cube, beam_fwhm_arcmin)
    t_beam = time.time()

    res = compute_stack_massbin(cube, geom, sel, tgrid, n_realizations, verbose=verbose)
    t_stack = time.time()

    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # bins >=1e13 have no in_patch substack by construction (see module
        # docstring (c)) -- their stack_real_inpatch rows are all-NaN, which
        # legitimately triggers numpy's "Mean/Degrees of freedom" warnings on
        # nanmean/nanstd; harmless, silenced here rather than at every caller.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(res["stack_real"], axis=1)            # (n_bins, n_theta)
        std = np.nanstd(res["stack_real"], axis=1)
        mean_ip = np.nanmean(res["stack_real_inpatch"], axis=1)
        std_ip = np.nanstd(res["stack_real_inpatch"], axis=1)

    np.savez(
        out_path,
        stack_real=res["stack_real"], mean=mean, std=std,
        n_gal=res["n_gal"], n_gal_total_catalog=sel["n_total"], n_gal_real=res["n_gal_real"],
        stack_real_inpatch=res["stack_real_inpatch"], mean_inpatch=mean_ip, std_inpatch=std_ip,
        n_gal_inpatch=res["n_gal_inpatch"], n_gal_real_inpatch=res["n_gal_real_inpatch"],
        logM200_mean=res["logM200_mean"], mass_bin_edges=sel["mass_bin_edges"],
        theta_kind=THETA_KIND_MASSBIN, theta_value=THETA_VALUE_MASSBIN,
        sample=sample, run=run, map=map_type, src_idx=src_idx,
        beam_fwhm_arcmin=(-1.0 if beam_fwhm_arcmin is None else beam_fwhm_arcmin),
        wall_time_sec=time.time() - t0,
        wall_time_load_sec=t_load - t0,
        wall_time_beam_sec=t_beam - t_load,
        wall_time_stack_sec=t_stack - t_beam,
    )
    print(f"[cap_stack] {sample}/{map_type}/run={run}"
          f"{'' if beam_fwhm_arcmin is None else f'/beam={beam_fwhm_arcmin}am'}: "
          f"n_gal_per_bin={res['n_gal'].tolist()} "
          f"load={t_load-t0:.1f}s beam={t_beam-t_load:.1f}s stack={t_stack-t_beam:.1f}s "
          f"total={time.time()-t0:.1f}s -> {out_path}", flush=True)
    return out_path


def run_one(run: str, map_type: str, sample: str, *, beam_fwhm_arcmin=None,
            src_idx: int = 4, out_dir: Path = SHARD_DIR_DEFAULT,
            n_realizations: int = N_REALIZATIONS, force: bool = False,
            verbose: bool = False) -> Path:
    if sample in MASSBIN_SAMPLES:
        return run_one_massbin(run, map_type, sample, beam_fwhm_arcmin=beam_fwhm_arcmin,
                                src_idx=src_idx, out_dir=out_dir, n_realizations=n_realizations,
                                force=force, verbose=verbose)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = shard_path(out_dir, sample, map_type, run, beam_fwhm_arcmin, src_idx)
    if out_path.exists() and not force:
        if verbose:
            print(f"[cap_stack] shard exists, skipping: {out_path}")
        return out_path

    t0 = time.time()
    geom = load_geometry(GEOM_PATH)
    sel = select_sample(sample, run)
    if sel["n_sel"] == 0:
        raise SystemExit(f"no halos selected for sample={sample} run={run}")
    tgrid = theta_pix_grid(sel["r200"], sel["plane_p"], geom)

    mpath = map_path_for_run(run, map_type)
    if verbose:
        print(f"[cap_stack] loading {mpath} (src_idx={src_idx}) ...", flush=True)
    d = np.load(mpath)
    in_file_key = MASSPLANE_KEY if map_type == "massplane" else map_type
    cube = np.array(d[in_file_key][:, src_idx])  # (n_real, 1024, 1024), copy -> free the 5-plane array
    del d
    if cube.shape[0] < n_realizations:
        if verbose:
            print(f"[cap_stack] {map_type}: only {cube.shape[0]} realizations on disk "
                  f"(requested {n_realizations}) -- capping.", flush=True)
        n_realizations = cube.shape[0]
    t_load = time.time()

    if beam_fwhm_arcmin is not None:
        cube = apply_beam(cube, beam_fwhm_arcmin)
    t_beam = time.time()

    stack_real, n_gal_real = compute_stack(
        cube, geom, sel["pixel_i"], sel["pixel_j"], sel["plane_p"], tgrid,
        n_realizations=n_realizations, verbose=verbose)
    t_stack = time.time()

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack_real, axis=0)
        std = np.nanstd(stack_real, axis=0)

    theta_kind = np.array(["xb"] * len(XB) + ["r200mult"] * len(R200_MULT))
    theta_value = np.concatenate([XB, R200_MULT])

    np.savez(
        out_path,
        stack_real=stack_real, mean=mean, std=std,
        n_gal=sel["n_sel"], n_gal_total_catalog=sel["n_total"], n_gal_real=n_gal_real,
        theta_kind=theta_kind, theta_value=theta_value,
        sample=sample, run=run, map=map_type, src_idx=src_idx,
        beam_fwhm_arcmin=(-1.0 if beam_fwhm_arcmin is None else beam_fwhm_arcmin),
        wall_time_sec=time.time() - t0,
        wall_time_load_sec=t_load - t0,
        wall_time_beam_sec=t_beam - t_load,
        wall_time_stack_sec=t_stack - t_beam,
    )
    print(f"[cap_stack] {sample}/{map_type}/run={run}"
          f"{'' if beam_fwhm_arcmin is None else f'/beam={beam_fwhm_arcmin}am'}: "
          f"n_gal={sel['n_sel']} mean_n_landing={n_gal_real.mean():.0f} "
          f"load={t_load-t0:.1f}s beam={t_beam-t_load:.1f}s stack={t_stack-t_beam:.1f}s "
          f"total={time.time()-t0:.1f}s -> {out_path}", flush=True)
    return out_path


def main():
    args = parse_args()
    run_one(args.run, args.map_type, args.sample,
            beam_fwhm_arcmin=args.beam_fwhm_arcmin, src_idx=args.src_idx,
            out_dir=args.out_dir, n_realizations=args.n_realizations,
            force=args.force, verbose=args.verbose)


if __name__ == "__main__":
    main()
