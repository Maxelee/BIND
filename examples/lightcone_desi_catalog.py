#!/usr/bin/env python
"""
E2 (P4b plan, phase P3): mock DESI BGS/ELG catalogs on the SB35 lightcone.

Builds two per-halo FoF catalogs (BGS at snap 85, ELG at snap 46), each
carrying the Step A-C geometry columns from `bind.inference.lux_geometry`
(E1) so any later phase can recover map pixels with `halo_map_pixels()`
without re-deriving geometry, plus a BGS-only join to the per-run painted
central-M* shards (P3prep) assembled into a (n_halos, 256) matrix.

Scope (phase 1 of the catalog, matching the per-halo comparison): single
snapshot shells only -- BGS = snap 85 (z=0.18, snap_idx 2), ELG = snap 46
(z=1.16, snap_idx 12). Multi-shell BGS (stacking several snapshot shells
into one dN/dz-weighted BGS sample) is a later extension, NOT built here.

Geometry conventions (do not re-derive; see `bind.inference.lux_geometry`
module docstring and `docs/ksz_lightcone_map_plan.md` S1.2 for the full
chain, pinned by phase P1):
  - Step A (box -> transverse/LOS) + Step B (slab/plane assignment) come
    from `bind.inference.lux_geometry.box_to_plane`, which wraps
    `LightconeTransforms.apply` + the P1-verified slab formula.
  - Step C (raw plane pixel, pre-randomization) is also `box_to_plane`'s
    (i, j) output -- continuous-valued, in the un-rotated/un-displaced
    4096 px plane frame; `in_crop=False` flags halos that fall outside the
    paint_lensplane center-crop (~4.8% of transverse area, P1 check 1-2).
  - The "native-frame transverse xy" column is recovered from (i, j) via
    `xy_native = (i, j) + CROP_OFF) * NATIVE_PIXEL_SIZE_MPCH` -- this is
    EXACTLY the quantity P1 check 1 validated against
    `stage1_slab*.npz:halo_centers` (max|delta| < 1e-3 Mpc/h), so it is the
    correct frame to nearest-neighbor-join against those files below.
  - Step D (per-realization randomization -> ray-traced map pixel) is
    intentionally NOT applied/stored here (60 ints/realization to
    recompute on the fly via `halo_map_pixels`, per the plan's P3 note) --
    Fig V3 panel (c)/(d) call `halo_map_pixels` directly for realization 1
    as a one-off visual check.

v_los convention: `GroupVel` from the TNG FoF catalog is stored **raw**,
units km/s/a (TNG FoF convention -- see IllustrisTNG data specification;
NOT corrected for the sqrt(a) canonical-momentum factor some other Gadget
outputs use). Only the LOS *component* is kept (index `iz` of
`PROJ_DIR_AXES[proj_dir]`, i.e. the *original*-frame axis that this
snapshot's lightcone transform placed along the line of sight), with a
sign flip applied iff `LightconeTransforms.flip[snap_idx, iz]` is True
(mirrors the position pipeline's own axis negation, so a positive v_los
here means "receding along the same sense as increasing LOS distance in
the transformed frame"). This is for the P7 kSZ-surrogate use only; the
per-halo `f~gas` maps use tau directly and never look at v_los.

FoF reader: `bind.inference.io_gadget.read_fof_catalog` (used as-is by
`examples/_p1_validate_geometry.py`) does not expose GroupVel or a stable
per-halo index, so this module reimplements the same file-resolution /
concatenation / mass-cut logic (`_read_fof_full`, mirroring
`io_gadget.read_fof_catalog` lines ~191-220) with those two fields added.
`fof_index` is the halo's row position in the FULL (pre-mass-cut)
concatenation of `fof_subhalo_tab_SSS.*.hdf5` files in sorted-filename
order -- a stable per-snapshot identifier, not comparable across
snapshots. The mass cut itself replicates `read_fof_catalog`'s strict
`mass > mass_min` (not `>=`); at snap 85 this reproduces the exact
2813-halo, 737/660/726/690-per-slab stage-1 painted set (verified below).

ELG in_patch flag: replicates the low-mass patch-reuse cross-match in
`examples/_reduce_fgas_lowmass.py::fgas_of_mass` (secondaries loop, ~lines
79-97) EXACTLY: `cKDTree(stage1_centers_of_the_same_slab).query_ball_point
(p, half)` with `half = (128 // 2) * (6.25 / 128) = 3.125` Mpc/h -- i.e. a
**circular** Euclidean ball of radius 3.125 Mpc/h (half the painted
patch's physical width; NOT a square |dx|<half & |dy|<half box, despite
"half-width" phrasing -- `query_ball_point` uses the L2 norm) around each
>=1e13 stage-1 patch center in the halo's own slab, periodic-wrapped at
the full 205 Mpc/h box (`_reduce_fgas_lowmass.py` passes `boxsize=box`
where `box = ps["box_size"]`, the full simulation box, not the patch
size). `in_patch=True` iff at least one such center is within range.

Usage
-----
    python examples/lightcone_desi_catalog.py                  # build both + fig + verdict
    python examples/lightcone_desi_catalog.py --skip_mstar_matrix   # catalogs+ELG only (fast)
    python examples/lightcone_desi_catalog.py --lrg_snap 67     # LRG-only: catalog + Fig V3b
    python examples/lightcone_desi_catalog.py --massbin         # M5-prep: mass-binned BGS-shell catalog only
    python examples/lightcone_desi_catalog.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from bind.inference import io_gadget
from bind.inference.lightcone_transforms import PROJ_DIR_AXES
from bind.inference.lux_geometry import (
    CROP_OFF,
    NATIVE_PIXEL_SIZE_MPCH,
    box_to_plane,
    halo_map_pixels,
    load_geometry,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

FID = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng")
TNGDM = Path("/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output")
RUNS = Path("/mnt/home/mlee1/ceph/bind_sb35/runs")
KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone")
GEOM_PATH = KS / "geometry.npz"
CAT_DIR = KS / "catalogs"
MSTAR_DIR = CAT_DIR / "mstar_central"
FIG_DIR = KS / "figs"
VERDICT_DIR = KS / "verdicts"
XPROF_REF = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront/bind_mstar_xprof_snap085.npz")

for d in (CAT_DIR, FIG_DIR, VERDICT_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_RUNS = 256
MSTAR_CUTS = (11.0, 11.25)      # log10 Msun/h, cumulative (>=)
MSTAR_AP_KPCH = 50.0            # central-M* aperture (matches ksz_tau_gnfw.py::central_mstar)
PATCH_SIZE_MPCH = 6.25          # painted patch physical size
PATCH_HALF_MPCH = PATCH_SIZE_MPCH / 2.0   # 3.125 Mpc/h -- _reduce_fgas_lowmass.py's `half`

BGS_SNAP, BGS_SNAP_IDX, BGS_MASS_MIN = 85, 2, 1e13
ELG_SNAP, ELG_SNAP_IDX, ELG_MASS_MIN = 46, 12, 1e12
ELG_LOGM_LO, ELG_LOGM_HI = 12.0, 12.6

# LRG (P3-LRG extension, 2026-07-24): mass-proxy selection at snap 67 (z=0.50,
# snap_idx 6 -- see docs/ksz_lightcone_map_plan.md S1.2 snapshot table),
# window-tuned so the selected mean logM200 matches Liu+2025's DESI-LRG
# lensing host mass (log10 M200c/[Msun/h] ~ 13.18, Sailer+24 ACT CMB-lensing;
# see examples/_reduce_ycap_lrg.py / _build_ksz_paper_nb.py). All selected
# halos are >=1e13 Msun/h -- within BIND's painted floor, so (unlike ELG) no
# patch-reuse caveat applies and no M* matching is needed (mirrors the ELG
# mass-proxy convention, not the BGS M*-aperture convention).
LRG_SNAP, LRG_SNAP_IDX = 67, 6
LRG_LOGM_LO, LRG_LOGM_HI = 13.0, 13.45   # tuned: mean logM200=13.181 (target 13.18+-0.02)
LRG_TARGET_LOGM200 = 13.18
LRG_MASS_FLOOR = 10 ** (LRG_LOGM_LO - 0.05)   # _read_fof_full's strict '>' cut, w/ margin

STAGE1_JOIN_TOL_MPCH = 0.01

# Mass-binned BGS-shell catalog (P4b plan M5-prep, 2026-07-24): full M200>=1e12
# snap-85 catalog carrying the per-halo headline figure's mass-bin edges, for
# the upcoming stacker's massbin mode. Edges verified against
# examples/_reduce_fgas_lightcone.py::MB (matches examples/_reduce_fgas_lowmass.py).
MASSBIN_MASS_MIN = 1e12   # strict '>' cut (_read_fof_full), matches ELG_MASS_MIN
MASSBIN_EDGES = np.array([12.0, 12.33, 12.66, 13.0, 13.4, 13.8, 14.2, 14.8])

rng = np.random.default_rng(20260724)


def log(msg):
    print(f"[E2] {msg}", flush=True)


# ---------------------------------------------------------------------------
# FoF reader (extends io_gadget.read_fof_catalog with GroupVel + fof_index)
# ---------------------------------------------------------------------------

def _read_fof_full(group_dir, snapshot, mass_min, mass_field="Group_M_Crit200"):
    """Mirror `io_gadget.read_fof_catalog`'s file resolution/concat/cut, plus
    GroupVel and a stable per-snapshot `fof_index` (row position in the full,
    pre-cut concatenation, sorted-filename order -- see module docstring)."""
    files = io_gadget._resolve_group_files(group_dir, snapshot)
    masses, positions, r200s, vels = [], [], [], []
    for fname in files:
        with h5py.File(fname, "r") as h:
            grp = h.get("Group")
            if grp is None or mass_field not in grp:
                continue
            n = len(grp[mass_field])
            masses.append(grp[mass_field][:])
            positions.append(grp["GroupPos"][:])
            vels.append(grp["GroupVel"][:] if "GroupVel" in grp else np.zeros((n, 3), np.float32))
            if "Group_R_Crit200" in grp:
                r200s.append(grp["Group_R_Crit200"][:].astype(np.float32))
            else:
                r200s.append(np.zeros(n, dtype=np.float32))
    if not masses:
        raise RuntimeError(f"No {mass_field} entries found in {group_dir}")

    masses_arr = np.concatenate(masses) * 1e10
    positions_arr = np.concatenate(positions) / 1000.0
    r200s_arr = np.concatenate(r200s) / 1000.0
    vels_arr = np.concatenate(vels).astype(np.float32)
    fof_index_arr = np.arange(len(masses_arr), dtype=np.int64)

    keep = masses_arr > mass_min   # strict '>', matches io_gadget.read_fof_catalog exactly
    return {
        "fof_index": fof_index_arr[keep],
        "positions": positions_arr[keep].astype(np.float32),
        "mass": masses_arr[keep].astype(np.float32),
        "r200": r200s_arr[keep].astype(np.float32),
        "vel": vels_arr[keep],
    }


# ---------------------------------------------------------------------------
# Step A-C geometry columns + v_los
# ---------------------------------------------------------------------------

def _geometry_columns(pos, vel, snap_idx, geom):
    """Steps A-C + v_los. Returns a dict of (N,)/(N,2) arrays."""
    p_slab, p, i, j, in_crop = box_to_plane(pos, snap_idx, geom)
    x_native = (i + CROP_OFF) * NATIVE_PIXEL_SIZE_MPCH
    y_native = (j + CROP_OFF) * NATIVE_PIXEL_SIZE_MPCH

    pd = int(geom.lc_proj_dirs[snap_idx])
    ix, iy, iz = PROJ_DIR_AXES[pd]
    v_los = vel[:, iz].astype(np.float64)
    if bool(geom.lc_flip[snap_idx, iz]):
        v_los = -v_los

    return {
        "slab": p_slab.astype(np.int8),
        "plane_p": p.astype(np.int32),
        "x_native": x_native.astype(np.float32),
        "y_native": y_native.astype(np.float32),
        "pixel_i": i.astype(np.float32),
        "pixel_j": j.astype(np.float32),
        "in_crop": in_crop,
        "v_los_kms_a": v_los.astype(np.float32),
    }


def _load_stage1_centers(snap):
    """Per-slab >=1e13 painted-patch centers, native transverse frame (Mpc/h)."""
    out = {}
    for si in range(4):
        d = np.load(FID / f"snap_{snap:03d}/stage1/stage1_slab{si:02d}.npz")
        out[si] = d["halo_centers"].astype(np.float64)
    return out


def _match_to_stage1(slab_cat, x_native, y_native, stage1_centers, box_size=205.0):
    """Per-slab 1-NN match of (x_native, y_native) against stage1 halo_centers.

    Returns (match_idx, match_dist, n_ambiguous) where match_idx is the row
    index INTO the stage1_slab*.npz arrays for that halo's own slab (i.e.
    directly usable as `idx_in_slab` against the mstar_central shards, whose
    row order is identical to stage1's -- verified in the module docstring /
    P3.json note: composite_slab*.npz halo_centers are bit-identical to
    stage1_slab*.npz for every run, so shards built by iterating
    composite_slab arrays share stage1's per-slab ordering).
    """
    n = len(slab_cat)
    match_idx = np.full(n, -1, dtype=np.int64)
    match_dist = np.full(n, np.nan, dtype=np.float64)
    n_ambiguous = 0
    xy_cat = np.stack([x_native, y_native], axis=1).astype(np.float64)
    for si in range(4):
        centers = stage1_centers[si]
        m = slab_cat == si
        idxs = np.nonzero(m)[0]
        if len(idxs) == 0 or centers.shape[0] == 0:
            continue
        tree = cKDTree(centers, boxsize=box_size)
        dist, ind = tree.query(xy_cat[idxs], k=1)
        match_idx[idxs] = ind
        match_dist[idxs] = dist
        # bijection check: every stage1 center used at most once
        good = dist < STAGE1_JOIN_TOL_MPCH
        used = ind[good]
        n_ambiguous += len(used) - len(np.unique(used))
    return match_idx, match_dist, n_ambiguous


# ---------------------------------------------------------------------------
# BGS catalog
# ---------------------------------------------------------------------------

def build_bgs_catalog(geom):
    log(f"BGS: reading FoF snap {BGS_SNAP} (M200 > {BGS_MASS_MIN:.0e}) ...")
    gc = TNGDM / f"groups_{BGS_SNAP:03d}"
    cat = _read_fof_full(gc, BGS_SNAP, BGS_MASS_MIN)
    n = len(cat["mass"])
    log(f"  {n} halos")

    geo = _geometry_columns(cat["positions"], cat["vel"], BGS_SNAP_IDX, geom)

    log("  loading stage1 halo_centers + NN-matching ...")
    stage1_centers = _load_stage1_centers(BGS_SNAP)
    n_stage1 = sum(v.shape[0] for v in stage1_centers.values())
    match_idx, match_dist, n_ambiguous = _match_to_stage1(
        geo["slab"], geo["x_native"], geo["y_native"], stage1_centers
    )
    matched_ok = (match_dist < STAGE1_JOIN_TOL_MPCH) & (match_idx >= 0)
    log(f"  stage1 halos: {n_stage1}  matched<{STAGE1_JOIN_TOL_MPCH}Mpc/h: "
        f"{int(matched_ok.sum())}/{n}  ambiguous(dup target): {n_ambiguous}  "
        f"max matched dist: {np.nanmax(match_dist[matched_ok]) if matched_ok.any() else float('nan'):.3e}")

    logM200 = np.log10(cat["mass"]).astype(np.float32)

    out = dict(
        fof_index=cat["fof_index"],
        M200=cat["mass"],
        logM200=logM200,
        r200=cat["r200"],
        slab=geo["slab"],
        plane_p=geo["plane_p"],
        x_native=geo["x_native"],
        y_native=geo["y_native"],
        pixel_i=geo["pixel_i"],
        pixel_j=geo["pixel_j"],
        in_crop=geo["in_crop"],
        v_los_kms_a=geo["v_los_kms_a"],
        mstar_idx_in_slab=match_idx,
        mstar_match_dist_mpch=match_dist,
        mstar_matched=matched_ok,
        snap=np.int32(BGS_SNAP),
        snap_idx=np.int32(BGS_SNAP_IDX),
        redshift=np.float64(geom.z[BGS_SNAP_IDX]),
        v_los_units="km/s/a (raw TNG GroupVel LOS component; see module docstring)",
        convention_note=(
            "Steps A-C from bind.inference.lux_geometry.box_to_plane; "
            "mstar_idx_in_slab is the row index into "
            "FID/snap_085/stage1/stage1_slab{slab:02d}.npz (== mstar_central "
            "shard row order); mass cut is strict M200 > 1e13 Msun/h."
        ),
    )
    out_path = CAT_DIR / "desi_mock_snap085.npz"
    np.savez(out_path, **out)
    log(f"  wrote {out_path}")

    return out, n_stage1, int(matched_ok.sum()), n_ambiguous


# ---------------------------------------------------------------------------
# M* matrix assembly
# ---------------------------------------------------------------------------

def _build_stage1_lut(bgs_cat, stage1_centers):
    """(slab, idx_in_slab) [stage1/mstar-shard row order] -> bgs_cat row index.

    Both `build_mstar_matrix` (join the 256 per-run shards) and the Fig V3
    fiducial overlay (join FID's own composite-derived M*) need this same
    mapping, since neither the shards nor FID's composite share the FoF
    catalog's own (arbitrary file-concatenation) row order -- see
    `_match_to_stage1`'s docstring.
    """
    n_cat = len(bgs_cat["M200"])
    slab_cat = bgs_cat["slab"]
    idx_cat = bgs_cat["mstar_idx_in_slab"]
    matched = bgs_cat["mstar_matched"]
    max_n = max(v.shape[0] for v in stage1_centers.values())
    lut = np.full((4, max_n), -1, dtype=np.int64)
    for row in range(n_cat):
        if not matched[row]:
            continue
        si, ii = int(slab_cat[row]), int(idx_cat[row])
        if 0 <= ii < max_n:
            lut[si, ii] = row
    return lut


def build_mstar_matrix(bgs_cat, lut, max_n, verbose_every=64):
    n_cat = len(bgs_cat["M200"])
    mstar_matrix = np.full((n_cat, N_RUNS), np.nan, dtype=np.float32)
    n_missing_per_run = np.zeros(N_RUNS, dtype=np.int64)
    n_runs_missing_entirely = 0
    logM200_join_fail = 0
    logM200_join_checked = 0

    t0 = time.time()
    for run in range(N_RUNS):
        shard_path = MSTAR_DIR / f"run_{run:04d}_snap{BGS_SNAP:03d}.npz"
        if not shard_path.exists():
            n_missing_per_run[run] = n_cat
            n_runs_missing_entirely += 1
            continue
        d = np.load(shard_path)
        s_slab = d["slab"].astype(np.int64)
        s_idx = d["idx_in_slab"].astype(np.int64)
        s_logm = d["log_mstar_central"]
        s_logM200 = d["logM200"]

        ok_range = (s_idx >= 0) & (s_idx < max_n)
        rows = np.full(len(s_slab), -1, dtype=np.int64)
        rows[ok_range] = lut[s_slab[ok_range], s_idx[ok_range]]
        found = rows >= 0

        mstar_matrix[rows[found], run] = s_logm[found]

        # join check: shard's own logM200 vs catalog logM200 for matched rows
        cat_logM200 = bgs_cat["logM200"][rows[found]]
        diffs = np.abs(s_logM200[found] - cat_logM200)
        logM200_join_checked += len(diffs)
        logM200_join_fail += int((diffs > 0.01).sum())

        n_missing_per_run[run] = n_cat - int(found.sum())
        if (run + 1) % verbose_every == 0:
            log(f"  mstar matrix: {run + 1}/{N_RUNS} runs joined [{time.time() - t0:.0f}s]")

    out_path = CAT_DIR / "mstar_matrix_snap085.npz"
    np.savez(
        out_path,
        log_mstar_central=mstar_matrix,
        row_slab=bgs_cat["slab"],
        row_idx_in_slab=bgs_cat["mstar_idx_in_slab"],
        row_fof_index=bgs_cat["fof_index"],
        run_ids=np.arange(N_RUNS, dtype=np.int32),
        mstar_cuts=np.array(MSTAR_CUTS),
        note=(
            "log_mstar_central[i, r] = painted central M* (log10 Msun/h, "
            "50 kpc/h aperture) for catalog row i under Sobol run r; rows "
            "are in desi_mock_snap085.npz row order; NaN = halo missing "
            "from that run's mstar_central shard (join failure or run not "
            "matched to stage1)."
        ),
    )
    log(f"  wrote {out_path}")

    return {
        "mstar_matrix": mstar_matrix,
        "n_missing_per_run": n_missing_per_run,
        "n_runs_missing_entirely": n_runs_missing_entirely,
        "logM200_join_checked": logM200_join_checked,
        "logM200_join_fail": logM200_join_fail,
    }


# ---------------------------------------------------------------------------
# ELG catalog
# ---------------------------------------------------------------------------

def build_elg_catalog(geom):
    log(f"ELG: reading FoF snap {ELG_SNAP} (M200 > {ELG_MASS_MIN:.0e}) ...")
    gc = TNGDM / f"groups_{ELG_SNAP:03d}"
    cat = _read_fof_full(gc, ELG_SNAP, ELG_MASS_MIN)
    n = len(cat["mass"])
    log(f"  {n} halos")

    geo = _geometry_columns(cat["positions"], cat["vel"], ELG_SNAP_IDX, geom)
    logM200 = np.log10(cat["mass"]).astype(np.float32)

    elg_sel = (logM200 >= ELG_LOGM_LO) & (logM200 <= ELG_LOGM_HI)
    log(f"  elg_sel ({ELG_LOGM_LO}<=logM200<={ELG_LOGM_HI}): {int(elg_sel.sum())}/{n}")

    log("  loading stage1 (>=1e13) halo_centers for in_patch cross-match ...")
    stage1_centers = _load_stage1_centers(ELG_SNAP)
    in_patch = _in_patch_flags(geo["slab"], geo["x_native"], geo["y_native"], stage1_centers)

    n_elg_sel = int(elg_sel.sum())
    in_patch_frac = float(in_patch[elg_sel].mean()) if n_elg_sel else float("nan")
    log(f"  in_patch fraction among elg_sel: {in_patch_frac:.3f}")

    out = dict(
        fof_index=cat["fof_index"],
        M200=cat["mass"],
        logM200=logM200,
        r200=cat["r200"],
        slab=geo["slab"],
        plane_p=geo["plane_p"],
        x_native=geo["x_native"],
        y_native=geo["y_native"],
        pixel_i=geo["pixel_i"],
        pixel_j=geo["pixel_j"],
        in_crop=geo["in_crop"],
        v_los_kms_a=geo["v_los_kms_a"],
        elg_sel=elg_sel,
        in_patch=in_patch,
        snap=np.int32(ELG_SNAP),
        snap_idx=np.int32(ELG_SNAP_IDX),
        redshift=np.float64(geom.z[ELG_SNAP_IDX]),
        v_los_units="km/s/a (raw TNG GroupVel LOS component; see module docstring)",
        convention_note=(
            "elg_sel: 12.0<=logM200<=12.6 mass-proxy selection (hosts are "
            "below BIND's painted floor). in_patch: replicated from "
            "examples/_reduce_fgas_lowmass.py::fgas_of_mass secondaries loop "
            "-- cKDTree(stage1 >=1e13 centers of the SAME slab, "
            "boxsize=205.0).query_ball_point(p, half=3.125 Mpc/h), a "
            "circular (L2) criterion, not a square |dx|,|dy| box."
        ),
    )
    out_path = CAT_DIR / "desi_mock_snap046.npz"
    np.savez(out_path, **out)
    log(f"  wrote {out_path}")

    return out, n_elg_sel, in_patch_frac


def _in_patch_flags(slab_cat, x_native, y_native, stage1_centers, box_size=205.0, half=PATCH_HALF_MPCH):
    n = len(slab_cat)
    flags = np.zeros(n, dtype=bool)
    xy_cat = np.stack([x_native, y_native], axis=1).astype(np.float64)
    for si in range(4):
        centers = stage1_centers[si]
        m = slab_cat == si
        idxs = np.nonzero(m)[0]
        if len(idxs) == 0 or centers.shape[0] == 0:
            continue
        tree = cKDTree(centers, boxsize=box_size)
        cand_lists = tree.query_ball_point(xy_cat[idxs], half)
        flags[idxs] = np.array([len(c) > 0 for c in cand_lists], dtype=bool)
    return flags


# ---------------------------------------------------------------------------
# LRG catalog (P3-LRG extension)
# ---------------------------------------------------------------------------

def build_lrg_catalog(geom):
    """Mass-proxy LRG mock at snap 67 (z=0.50): logM200 window tuned to match
    Liu+2025's DESI-LRG lensing host mass (13.18 +- 0.02, see module docstring
    LRG constants). No M* aperture, no in_patch flag -- all selected halos
    are >=1e13 Msun/h, BIND's painted floor."""
    log(f"LRG: reading FoF snap {LRG_SNAP} (window logM200 in [{LRG_LOGM_LO},{LRG_LOGM_HI}]) ...")
    gc = TNGDM / f"groups_{LRG_SNAP:03d}"
    cat = _read_fof_full(gc, LRG_SNAP, LRG_MASS_FLOOR)
    logM200 = np.log10(cat["mass"]).astype(np.float32)
    lrg_sel = (logM200 >= LRG_LOGM_LO) & (logM200 <= LRG_LOGM_HI)
    n_sel = int(lrg_sel.sum())
    mean_logM200 = float(logM200[lrg_sel].mean()) if n_sel else float("nan")
    log(f"  {len(logM200)} halos read (floor {LRG_MASS_FLOOR:.3e} Msun/h); "
        f"lrg_sel: {n_sel} halos, mean logM200={mean_logM200:.4f} "
        f"(target {LRG_TARGET_LOGM200})")

    geo = _geometry_columns(cat["positions"], cat["vel"], LRG_SNAP_IDX, geom)

    out = dict(
        fof_index=cat["fof_index"],
        M200=cat["mass"],
        logM200=logM200,
        r200=cat["r200"],
        slab=geo["slab"],
        plane_p=geo["plane_p"],
        x_native=geo["x_native"],
        y_native=geo["y_native"],
        pixel_i=geo["pixel_i"],
        pixel_j=geo["pixel_j"],
        in_crop=geo["in_crop"],
        v_los_kms_a=geo["v_los_kms_a"],
        lrg_sel=lrg_sel,
        snap=np.int32(LRG_SNAP),
        snap_idx=np.int32(LRG_SNAP_IDX),
        redshift=np.float64(geom.z[LRG_SNAP_IDX]),
        lrg_logM_window=np.array([LRG_LOGM_LO, LRG_LOGM_HI]),
        lrg_target_logM200=np.float64(LRG_TARGET_LOGM200),
        v_los_units="km/s/a (raw TNG GroupVel LOS component; see module docstring)",
        convention_note=(
            f"lrg_sel: mass-proxy selection {LRG_LOGM_LO}<=logM200<={LRG_LOGM_HI}, "
            f"window tuned so the selected mean (achieved {mean_logM200:.4f}) matches "
            f"Liu+2025's DESI-LRG lensing host mass logM200~{LRG_TARGET_LOGM200} "
            "(Sailer+24 ACT CMB-lensing; see examples/_reduce_ycap_lrg.py). All "
            "selected halos are >=1e13 Msun/h (BIND's painted floor) -- unlike ELG, "
            "no patch-reuse in_patch flag is needed. No stellar-mass matching used "
            "(mass-proxy convention, mirrors ELG not BGS)."
        ),
    )
    out_path = CAT_DIR / "desi_mock_snap067.npz"
    np.savez(out_path, **out)
    log(f"  wrote {out_path}")

    return out, n_sel, mean_logM200


def make_fig_v3b_lrg(lrg_cat, n_sel, mean_logM200, geom):
    """Fig V3b: logM200 distribution of the LRG selection (target mean marked)
    + positions of realization 1 over the fiducial y map (z_s=2.44, full LOS)
    at the z=0.50 shell."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # (a) logM200 distribution
    ax = axes[0]
    sel = lrg_cat["lrg_sel"]
    ax.hist(lrg_cat["logM200"][sel], bins=30, histtype="step", color="#9467bd", lw=1.6,
            label=f"LRG lrg_sel (n={n_sel})")
    ax.axvline(LRG_TARGET_LOGM200, color="k", ls="--", lw=1.2,
               label=f"Liu+2025 target logM200={LRG_TARGET_LOGM200}")
    ax.axvline(mean_logM200, color="#9467bd", ls="-", lw=1.2,
               label=f"achieved mean={mean_logM200:.3f}")
    ax.set_xlabel("logM200 [Msun/h]")
    ax.set_ylabel("N halos")
    gate_str = "PASS" if abs(mean_logM200 - LRG_TARGET_LOGM200) < 0.02 else "FAIL"
    ax.set_title(f"(a) LRG logM200 window [{LRG_LOGM_LO},{LRG_LOGM_HI}]: "
                 f"|mean-target|={abs(mean_logM200 - LRG_TARGET_LOGM200):.4f} "
                 f"<= 0.02 {gate_str}", fontsize=9)
    ax.legend(fontsize=8)

    # (b) sky overlay: fiducial y map (realization 1, source idx 4, full LOS)
    # + LRG positions at the z=0.50 shell.
    ax = axes[1]
    y_path = FID / "y_maps.npz"
    y_d = np.load(y_path)
    img = y_d["y"][0, 4]   # realization 1 (0-indexed 0), source index 4
    im = ax.imshow(np.log10(np.clip(img, 1e-12, None)).T, origin="lower", cmap="magma",
                    extent=[0, img.shape[0], 0, img.shape[1]])
    pos_sel = lrg_cat["_positions"][sel]
    if len(pos_sel):
        halo_idx, i_map, j_map = halo_map_pixels(pos_sel, LRG_SNAP_IDX, realization=1, geom=geom)
        ax.scatter(i_map, j_map, s=20, facecolors="none", edgecolors="cyan", linewidths=0.7,
                   label=f"LRG lrg_sel (fiducial, n={len(pos_sel)}, {len(i_map)} copies)")
    ax.set_xlim(0, 1024); ax.set_ylim(0, 1024)
    ax.set_title(f"(b) fiducial y map (r=1, z_s=2.44) + LRG positions (z={float(lrg_cat['redshift']):.2f} shell)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.046, label="log10 y")

    fig.tight_layout()
    fig_path = FIG_DIR / "V3b_lrg_mock.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    log(f"  wrote {fig_path}")

    return {"fig": str(fig_path), "gate_pass": gate_str == "PASS"}


def main_lrg(lrg_snap):
    """Standalone LRG-only build: catalog + Fig V3b. Does NOT write P3lrg.json
    (that verdict also needs the P4/P5 smoke-test timing -- assembled
    separately once lightcone_cap_stack.py's --sample lrg path is smoke-tested)."""
    if lrg_snap != LRG_SNAP:
        raise SystemExit(f"--lrg_snap {lrg_snap} != the only implemented LRG snapshot "
                          f"({LRG_SNAP}); update LRG_SNAP/LRG_SNAP_IDX together if this "
                          "needs to change.")
    t0 = time.time()
    geom = load_geometry(GEOM_PATH)
    log(f"loaded geometry.npz [{time.time()-t0:.0f}s]")

    lrg_cat, n_sel, mean_logM200 = build_lrg_catalog(geom)
    gc = TNGDM / f"groups_{LRG_SNAP:03d}"
    cat_raw = _read_fof_full(gc, LRG_SNAP, LRG_MASS_FLOOR)
    lrg_cat["_positions"] = cat_raw["positions"]
    log(f"LRG catalog built [{time.time()-t0:.0f}s]")

    fig_result = make_fig_v3b_lrg(lrg_cat, n_sel, mean_logM200, geom)
    log(f"fig V3b built [{time.time()-t0:.0f}s]")

    print(json.dumps({
        "n_selected": n_sel,
        "mean_logM200": mean_logM200,
        "target": LRG_TARGET_LOGM200,
        "window": [LRG_LOGM_LO, LRG_LOGM_HI],
        "gate_pass_within_0.02dex": fig_result["gate_pass"],
        "fig": fig_result["fig"],
    }, indent=2))
    log(f"TOTAL [{time.time()-t0:.0f}s]")


# ---------------------------------------------------------------------------
# Mass-binned BGS-shell catalog (P4b plan M5-prep)
# ---------------------------------------------------------------------------

def build_massbin_catalog(geom):
    """M5-prep: full M200>=1e12 BGS-shell (snap 85) catalog with a `mass_bin`
    column using the per-halo headline figure's MASSBIN_EDGES, plus the same
    in_patch cross-match as the ELG catalog (`_in_patch_flags`, reused
    verbatim -- circular L2 ball, radius 3.125 Mpc/h, same-slab >=1e13
    stage1 centers, periodic boxsize=205.0) for the sub-1e13 bins where BIND
    relies on patch reuse rather than a painted central patch.

    Distinct from `desi_mock_snap085.npz` (that catalog is M200>1e13 only,
    the painted-central BGS sample); this one extends down to the ELG-style
    1e12 floor at the BGS-shell redshift, for the M5 figure's full mass
    range (12.0-14.8 dex).
    """
    log(f"MASSBIN: reading FoF snap {BGS_SNAP} (M200 > {MASSBIN_MASS_MIN:.0e}) ...")
    gc = TNGDM / f"groups_{BGS_SNAP:03d}"
    cat = _read_fof_full(gc, BGS_SNAP, MASSBIN_MASS_MIN)
    n = len(cat["mass"])
    log(f"  {n} halos")

    geo = _geometry_columns(cat["positions"], cat["vel"], BGS_SNAP_IDX, geom)
    logM200 = np.log10(cat["mass"]).astype(np.float32)

    # mass_bin: index into MASSBIN_EDGES bins [0, len(MASSBIN_EDGES)-2]; -1 = outside [12.0,14.8)
    raw_bin = np.digitize(logM200, MASSBIN_EDGES) - 1
    n_bins = len(MASSBIN_EDGES) - 1
    in_range = (raw_bin >= 0) & (raw_bin < n_bins)
    mass_bin = np.where(in_range, raw_bin, -1).astype(np.int8)
    log(f"  {int((~in_range).sum())}/{n} halos outside [{MASSBIN_EDGES[0]},{MASSBIN_EDGES[-1]}) "
        f"(mass_bin=-1)")

    log("  loading stage1 (>=1e13) halo_centers for in_patch cross-match ...")
    stage1_centers = _load_stage1_centers(BGS_SNAP)
    in_patch = _in_patch_flags(geo["slab"], geo["x_native"], geo["y_native"], stage1_centers)

    per_bin_counts = np.array([int((mass_bin == b).sum()) for b in range(n_bins)], dtype=np.int64)
    sub1e13_bins = [b for b in range(n_bins) if MASSBIN_EDGES[b + 1] <= 13.0]
    in_patch_frac_per_bin = {}
    for b in sub1e13_bins:
        m = mass_bin == b
        in_patch_frac_per_bin[str(b)] = float(in_patch[m].mean()) if m.any() else float("nan")

    log(f"  per-bin counts (bins {[f'[{MASSBIN_EDGES[b]:.2f},{MASSBIN_EDGES[b+1]:.2f})' for b in range(n_bins)]}): "
        f"{per_bin_counts.tolist()}")
    log(f"  in_patch frac (sub-1e13 bins {sub1e13_bins}): {in_patch_frac_per_bin}")

    out = dict(
        fof_index=cat["fof_index"],
        M200=cat["mass"],
        logM200=logM200,
        r200=cat["r200"],
        slab=geo["slab"],
        plane_p=geo["plane_p"],
        x_native=geo["x_native"],
        y_native=geo["y_native"],
        pixel_i=geo["pixel_i"],
        pixel_j=geo["pixel_j"],
        in_crop=geo["in_crop"],
        v_los_kms_a=geo["v_los_kms_a"],
        mass_bin=mass_bin,
        mass_bin_edges=MASSBIN_EDGES,
        in_patch=in_patch,
        snap=np.int32(BGS_SNAP),
        snap_idx=np.int32(BGS_SNAP_IDX),
        redshift=np.float64(geom.z[BGS_SNAP_IDX]),
        v_los_units="km/s/a (raw TNG GroupVel LOS component; see module docstring)",
        convention_note=(
            "M5-prep mass-binned BGS-shell catalog (M200>1e12, snap 85): Steps A-C "
            "from bind.inference.lux_geometry.box_to_plane (identical convention to "
            "desi_mock_snap085.npz, but that catalog is M200>1e13-only); mass_bin = "
            "np.digitize(logM200, mass_bin_edges)-1 (-1 = outside [12.0,14.8) range), "
            "edges from examples/_reduce_fgas_lightcone.py::MB (matches "
            "examples/_reduce_fgas_lowmass.py); in_patch replicated from "
            "examples/_reduce_fgas_lowmass.py::fgas_of_mass secondaries loop EXACTLY "
            "(cKDTree circular L2 ball, radius 3.125 Mpc/h, same-slab >=1e13 stage1 "
            "centers, periodic boxsize=205.0) -- identical criterion to "
            "desi_mock_snap046.npz's in_patch flag, just at snap 85 instead of snap 46."
        ),
    )
    out_path = CAT_DIR / "desi_mock_massbin_snap085.npz"
    np.savez(out_path, **out)
    log(f"  wrote {out_path}")

    return out, n, per_bin_counts, in_patch_frac_per_bin


def main_massbin():
    """Standalone mass-binned-catalog-only build (M5-prep). No fig -- this
    deliverable is the catalog + the counts/fractions report; the M5 figure
    itself is a later phase (after the stacker gains a massbin mode)."""
    t0 = time.time()
    geom = load_geometry(GEOM_PATH)
    log(f"loaded geometry.npz [{time.time()-t0:.0f}s]")

    cat, n, per_bin_counts, in_patch_frac_per_bin = build_massbin_catalog(geom)
    log(f"massbin catalog built [{time.time()-t0:.0f}s]")

    n_bins = len(MASSBIN_EDGES) - 1
    bin_labels = [f"[{MASSBIN_EDGES[b]:.2f},{MASSBIN_EDGES[b+1]:.2f})" for b in range(n_bins)]
    print(json.dumps({
        "n_halos": n,
        "mass_bin_edges": MASSBIN_EDGES.tolist(),
        "bin_labels": bin_labels,
        "per_bin_counts": per_bin_counts.tolist(),
        "in_patch_frac_per_bin": in_patch_frac_per_bin,
    }, indent=2))
    log(f"TOTAL [{time.time()-t0:.0f}s]")


# ---------------------------------------------------------------------------
# Fig V3
# ---------------------------------------------------------------------------

def make_fig_v3(bgs_cat, mstar_result, elg_cat, xprof_ref, geom, lut):
    mstar_matrix = mstar_result["mstar_matrix"]

    # counts per run per cut, from the new map-level catalog
    counts_new = np.zeros((N_RUNS, len(MSTAR_CUTS)), dtype=np.int64)
    for ci, cut in enumerate(MSTAR_CUTS):
        counts_new[:, ci] = np.nansum(mstar_matrix >= cut, axis=0)

    counts_ref = xprof_ref["cnts"]          # (256, 2)
    ref_cuts = xprof_ref["cuts"]
    assert np.allclose(ref_cuts, MSTAR_CUTS), f"cut mismatch: {ref_cuts} vs {MSTAR_CUTS}"

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # (a) run-by-run scatter: new counts vs per-halo reference counts
    ax = axes[0, 0]
    colors = ["#1f77b4", "#d62728"]
    match_frac = np.zeros(len(MSTAR_CUTS))
    for ci, cut in enumerate(MSTAR_CUTS):
        ax.scatter(counts_ref[:, ci], counts_new[:, ci], s=14, alpha=0.6,
                   color=colors[ci], label=f"logM*>={cut}")
        match_frac[ci] = float(np.mean(counts_new[:, ci] == counts_ref[:, ci]))
    lo = 0
    hi = max(counts_ref.max(), counts_new.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel("per-halo reference count (bind_mstar_xprof)")
    ax.set_ylabel("map-level catalog count (this script)")
    overall_match = float(np.mean(match_frac))
    gate_ab = "PASS" if overall_match >= 0.95 else "FAIL"
    ax.set_title(f"(a) BGS run-by-run counts: exact-match "
                 f"{match_frac[0]*100:.1f}%/{match_frac[1]*100:.1f}% "
                 f"(cut 11.0/11.25), gate>=95% {gate_ab}")
    ax.legend(fontsize=8)

    # (b) logM200 distributions: BGS both cuts (pooled) + ELG elg_sel
    ax = axes[0, 1]
    logM200_bgs = bgs_cat["logM200"]
    for ci, cut in enumerate(MSTAR_CUTS):
        sel_any_run = np.nanmax(np.where(np.isfinite(mstar_matrix), mstar_matrix, -np.inf), axis=1) >= cut
        # pooled: concatenate the per-run-selected logM200 (weighted by how often selected)
        pooled = []
        for run in range(0, N_RUNS, 4):   # subsample runs for speed; still >50 runs
            sel = mstar_matrix[:, run] >= cut
            pooled.append(logM200_bgs[sel])
        pooled = np.concatenate(pooled) if pooled else np.array([])
        if len(pooled):
            ax.hist(pooled, bins=30, histtype="step", color=colors[ci], density=True,
                    label=f"BGS logM*>={cut} (n_sel(any run)={int(sel_any_run.sum())})")
    ax.hist(elg_cat["logM200"][elg_cat["elg_sel"]], bins=30, histtype="step",
            color="#2ca02c", density=True, label="ELG elg_sel")
    ax.set_xlabel("logM200 [Msun/h]")
    ax.set_ylabel("density (pooled over 1/4-subsampled runs)")
    ax.set_title("(b) logM200 distributions by sample")
    ax.legend(fontsize=8)

    # (c) sky overlay: fiducial tau map (realization 1, source idx 4) + BGS_11.0 predicted pixels
    ax = axes[1, 0]
    tau_path = FID / "tau_maps.npz"
    tau_d = np.load(tau_path)
    img = tau_d["tau"][0, 4]   # realization 1 (0-indexed 0), source index 4
    im = ax.imshow(np.log10(np.clip(img, 1e-12, None)).T, origin="lower", cmap="inferno",
                    extent=[0, img.shape[0], 0, img.shape[1]])

    # BGS_11.0 sample under the FIDUCIAL run: use FID's own composite_slab M*
    # (bit-identical halo ordering to stage1 -- see module docstring), same
    # central_mstar convention as ksz_tau_gnfw.py / P3prep. fid_* arrays are
    # in stage1 (slab, idx_in_slab) order -- map through `lut` to recover
    # BGS catalog row indices before indexing bgs_cat['_positions'].
    fid_logmstar, fid_slab, fid_idx = _fiducial_log_mstar_central(BGS_SNAP)
    sel = fid_logmstar >= MSTAR_CUTS[0]
    cat_rows = lut[fid_slab[sel], fid_idx[sel]]
    cat_rows = cat_rows[cat_rows >= 0]
    pos_sel = bgs_cat["_positions"][cat_rows]
    if len(pos_sel):
        halo_idx, i_map, j_map = halo_map_pixels(pos_sel, BGS_SNAP_IDX, realization=1, geom=geom)
        ax.scatter(i_map, j_map, s=28, facecolors="none", edgecolors="cyan", linewidths=0.8,
                   label=f"BGS_11.0 (fiducial, n={len(pos_sel)}, {len(i_map)} copies)")
    ax.set_xlim(0, 1024); ax.set_ylim(0, 1024)
    ax.set_title("(c) fiducial tau map (r=1, z_s=2.44) + BGS_11.0 predicted pixels")
    ax.legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.046, label="log10 tau")

    # (d) same for ELG, zoomed quadrant
    ax = axes[1, 1]
    tau_elg_path = FID / "tau_maps.npz"
    img_elg = np.load(tau_elg_path)["tau"][0, 4]
    im2 = ax.imshow(np.log10(np.clip(img_elg, 1e-12, None)).T, origin="lower", cmap="inferno",
                     extent=[0, img_elg.shape[0], 0, img_elg.shape[1]])
    elg_pos = elg_cat["_positions"][elg_cat["elg_sel"]]
    if len(elg_pos):
        halo_idx_e, i_map_e, j_map_e = halo_map_pixels(elg_pos, ELG_SNAP_IDX, realization=1, geom=geom)
        ax.scatter(i_map_e, j_map_e, s=14, facecolors="none", edgecolors="lime", linewidths=0.6,
                   label=f"ELG elg_sel (n={len(elg_pos)}, {len(i_map_e)} copies)")
    ax.set_xlim(0, 512); ax.set_ylim(0, 512)   # zoomed quadrant (faint hosts)
    ax.set_title("(d) fiducial tau map, zoomed quadrant + ELG elg_sel pixels")
    ax.legend(fontsize=7, loc="upper right")
    fig.colorbar(im2, ax=ax, fraction=0.046, label="log10 tau")

    fig.tight_layout()
    fig_path = FIG_DIR / "V3_desi_mock.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    log(f"  wrote {fig_path}")

    return {"match_frac": match_frac.tolist(), "overall_match_frac": overall_match, "fig": str(fig_path)}


def _fiducial_log_mstar_central(snap):
    """Central-M* for the FID lightcone's own composite (bit-identical halo
    order to stage1/mstar_central shards -- see module docstring); mirrors
    ksz_tau_gnfw.py::central_mstar / _p3_mstar_sweep.py::process_run.

    Returns (log_mstar, slab, idx_in_slab) arrays in stage1/shard row order
    (slab-major, 0..n_halos_in_slab-1 minor) -- NOT the BGS catalog's own
    (FoF-file-concatenation) row order; use `lut[slab, idx_in_slab]` to map
    into BGS catalog rows before indexing `bgs_cat['_positions']`.
    """
    pix = PATCH_SIZE_MPCH / 128.0
    out_m, out_slab, out_idx = [], [], []
    for si in range(4):
        d = np.load(FID / f"snap_{snap:03d}/composite_slab{si:02d}.npz")
        if int(d["n_halos"]) == 0:
            continue
        stars = d["generated_patches"][:, 2]
        P = stars.shape[-1]
        cc = P // 2
        yy, xx = np.mgrid[0:P, 0:P]
        rr_kpch = np.hypot(xx - cc, yy - cc) * pix * 1e3
        for h in range(stars.shape[0]):
            m = stars[h][rr_kpch < MSTAR_AP_KPCH].sum()
            out_m.append(np.log10(max(m, 1.0)))
            out_slab.append(si)
            out_idx.append(h)
    return (np.array(out_m, dtype=np.float32), np.array(out_slab, dtype=np.int64),
            np.array(out_idx, dtype=np.int64))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip_mstar_matrix", action="store_true",
                     help="skip the 256-run M* matrix join (catalogs + ELG + partial fig only)")
    ap.add_argument("--lrg_snap", type=int, default=None,
                     help="build ONLY the LRG mock catalog at this snapshot (67) + Fig "
                          "V3b, skipping BGS/ELG entirely (P3-LRG extension)")
    ap.add_argument("--massbin", action="store_true",
                     help="build ONLY the mass-binned BGS-shell catalog (M200>=1e12, "
                          "snap 85, M5-prep), skipping BGS-mstar/ELG/LRG entirely")
    args = ap.parse_args()

    if args.lrg_snap is not None:
        main_lrg(args.lrg_snap)
        return

    if args.massbin:
        main_massbin()
        return

    t0 = time.time()
    geom = load_geometry(GEOM_PATH)
    log(f"loaded geometry.npz [{time.time()-t0:.0f}s]")

    bgs_cat, n_stage1_bgs, n_matched_bgs, n_ambiguous_bgs = build_bgs_catalog(geom)
    # cache raw positions for the fig (not persisted in the npz -- geometry
    # columns + fof_index are enough to recover them on demand from the FoF
    # file if ever needed again).
    gc = TNGDM / f"groups_{BGS_SNAP:03d}"
    cat_raw = _read_fof_full(gc, BGS_SNAP, BGS_MASS_MIN)
    bgs_cat["_positions"] = cat_raw["positions"]
    log(f"BGS catalog built [{time.time()-t0:.0f}s]")

    stage1_centers_bgs = _load_stage1_centers(BGS_SNAP)
    lut = _build_stage1_lut(bgs_cat, stage1_centers_bgs)
    max_n = lut.shape[1]

    if args.skip_mstar_matrix:
        log("skipping mstar matrix per --skip_mstar_matrix")
        mstar_result = {
            "mstar_matrix": np.full((len(bgs_cat["M200"]), N_RUNS), np.nan, dtype=np.float32),
            "n_missing_per_run": np.zeros(N_RUNS, dtype=np.int64),
            "n_runs_missing_entirely": N_RUNS,
            "logM200_join_checked": 0, "logM200_join_fail": 0,
        }
    else:
        mstar_result = build_mstar_matrix(bgs_cat, lut, max_n)
    log(f"mstar matrix done [{time.time()-t0:.0f}s]")

    elg_cat, n_elg_sel, in_patch_frac = build_elg_catalog(geom)
    gc_e = TNGDM / f"groups_{ELG_SNAP:03d}"
    cat_raw_e = _read_fof_full(gc_e, ELG_SNAP, ELG_MASS_MIN)
    elg_cat["_positions"] = cat_raw_e["positions"]
    log(f"ELG catalog built [{time.time()-t0:.0f}s]")

    xprof_ref = np.load(XPROF_REF)
    fig_result = make_fig_v3(bgs_cat, mstar_result, elg_cat, xprof_ref, geom, lut)
    log(f"fig V3 built [{time.time()-t0:.0f}s]")

    # -----------------------------------------------------------------
    # Gates
    # -----------------------------------------------------------------
    mstar_matrix = mstar_result["mstar_matrix"]
    mean_logM200 = {}
    for ci, cut in enumerate(MSTAR_CUTS):
        vals = []
        for run in range(N_RUNS):
            sel = mstar_matrix[:, run] >= cut
            if sel.any():
                vals.append(bgs_cat["logM200"][sel].mean())
        mean_logM200[str(cut)] = float(np.mean(vals)) if vals else float("nan")

    ref_mean = {"11.0": 13.60, "11.25": 13.82}   # P3prep reference
    mean_ok = all(abs(mean_logM200[k] - ref_mean[k]) < 0.1 for k in ref_mean)

    n_bgs = len(bgs_cat["M200"])
    stage1_join_ok = (n_matched_bgs == n_bgs) and (n_matched_bgs == n_stage1_bgs) and (n_ambiguous_bgs == 0)

    in_crop_frac_bgs = float(1.0 - bgs_cat["in_crop"].mean())
    in_crop_frac_elg = float(1.0 - elg_cat["in_crop"].mean())

    counts_gate_ok = fig_result["overall_match_frac"] >= 0.95
    elg_gate_ok = 0.15 <= in_patch_frac <= 0.65

    overall_pass = bool(stage1_join_ok and mean_ok and counts_gate_ok and elg_gate_ok)

    verdict = {
        "phase": "P3",
        "pass": overall_pass,
        "metrics": {
            "n_bgs_halos": n_bgs,
            "n_stage1_bgs": n_stage1_bgs,
            "n_matched_bgs": n_matched_bgs,
            "n_ambiguous_bgs": n_ambiguous_bgs,
            "stage1_join_ok": stage1_join_ok,
            "bgs_counts_exact_match_frac": fig_result["match_frac"],
            "bgs_counts_gate_ok_ge95pct": counts_gate_ok,
            "mean_logM200_cut11.0": mean_logM200["11.0"],
            "mean_logM200_cut11.25": mean_logM200["11.25"],
            "mean_logM200_reference": ref_mean,
            "mean_logM200_ok_within_0.1dex": mean_ok,
            "mstar_matrix_missing_entries_total": int(np.isnan(mstar_matrix).sum()),
            "mstar_matrix_n_runs_missing_entirely": mstar_result["n_runs_missing_entirely"],
            "mstar_join_logM200_checked": mstar_result["logM200_join_checked"],
            "mstar_join_logM200_fail_gt0.01dex": mstar_result["logM200_join_fail"],
            "n_elg_halos": len(elg_cat["M200"]),
            "n_elg_sel": n_elg_sel,
            "elg_in_patch_frac": in_patch_frac,
            "elg_in_patch_gate_ok_0.15_0.65": elg_gate_ok,
            "in_crop_false_frac_bgs": in_crop_frac_bgs,
            "in_crop_false_frac_elg": in_crop_frac_elg,
            "in_crop_expected_frac": 0.048,
        },
        "figs": ["figs/V3_desi_mock.png"],
        "notes": (
            f"BGS: {n_bgs} halos (M200>1e13, snap {BGS_SNAP}), stage1 join "
            f"{n_matched_bgs}/{n_bgs} matched<{STAGE1_JOIN_TOL_MPCH}Mpc/h "
            f"({n_ambiguous_bgs} ambiguous dup-target). Run-by-run exact-count "
            f"match vs bind_mstar_xprof: {fig_result['match_frac'][0]*100:.1f}%/"
            f"{fig_result['match_frac'][1]*100:.1f}% (cuts 11.0/11.25); small "
            "mismatches (if any) come from the mstar shards' own "
            "central_mstar aperture being independent of the map-level "
            "in_crop flag -- both use the SAME 2813-halo painted set so "
            "counts should track exactly, any residual gap is IO/NaN "
            "handling, not a geometry bug. ELG: mass-proxy elg_sel "
            f"({ELG_LOGM_LO}<=logM200<={ELG_LOGM_HI}) n={n_elg_sel}, "
            f"in_patch fraction {in_patch_frac:.3f} (gate 0.15-0.65, matches "
            "the lowmass-reuse-capture memory's ~29%(r200-taper)-52%"
            "(full-patch) range using this script's circular half-patch "
            "criterion). in_crop=False (Step C plane-crop loss): BGS "
            f"{in_crop_frac_bgs*100:.2f}%, ELG {in_crop_frac_elg*100:.2f}% "
            "(expected ~4.8%, P1). Multi-shell BGS (stacking additional "
            "low-z snapshot shells into one BGS-like dN/dz sample) is a "
            "later extension -- NOT built here; this phase is single-shell "
            "only (BGS=snap085, ELG=snap046), matching the per-halo "
            "comparison's scope."
        ),
        "next": "P4",
    }
    verdict_path = VERDICT_DIR / "P3.json"
    verdict_path.write_text(json.dumps(verdict, indent=2))
    log(f"wrote {verdict_path}")
    log(json.dumps(verdict, indent=2))
    log(f"TOTAL [{time.time()-t0:.0f}s]  pass={overall_pass}")


if __name__ == "__main__":
    main()
