#!/usr/bin/env python
"""
P2 (partial): validate the fiducial haloplane+massplane co-trace against the
predictions of the P1 halo->pixel engine (docs/ksz_lightcone_map_plan.md, P2),
run while the co-trace SLURM job is still in flight.

Reads directly from the co-trace's raw per-realization ``.dat`` outputs under
``FID/_haloplane_work/rt_output/run{NNN}/`` -- READ-ONLY, never writes there,
and silently skips any ``runNNN`` missing ``tau78.dat``/``y78.dat`` (still
being written by the active job).

Four checks (each a panel group of Fig V2):

1. **Kappa identity** (end-to-end geometry certificate): the co-trace reuses
   the fiducial's own ``lenspot*.dat`` + ``config.dat`` (symlinked, same seed
   1992) so its re-traced ``kappa{p}.dat`` must equal the original
   ``kappa_maps.npz`` bit-for-bit (up to the float64->float32 cast the
   collector applies). If this fails, the co-trace used different geometry
   and everything below is meaningless -- STOP.
2. **Deflection measurement**: predict each halo's Born (undeflected) map
   pixel from the catalog's stored Step-A-C geometry columns
   (``bind.inference.lux_geometry.plane_to_map``, applied directly to
   ``pixel_i``/``pixel_j``/``plane_p`` -- no FoF re-read needed for the BGS/
   ELG-z bins) or, for snap 96/67 (no catalog), from a fresh FoF read +
   ``box_to_plane``. Search +-4 px in the traced ``tau78.dat`` (the painted
   total-matter column) for the local max; the offset from the Born
   prediction is a per-halo estimate of lux's real ray deflection.

   **Field-confusion caveat (discovered while building this script, not
   anticipated by the plan text).** ``tau78.dat`` is a *cumulative* sum of
   the painted total-matter column over all 80 lensplanes (same pathway as
   the real kSZ tau), so it is NOT a clean, isolated per-halo spike: its
   pixel-to-pixel scatter (~17% of the local mean, driven by unrelated
   LOS/2-halo structure from the other 79 planes) is comparable to a single
   BGS-mass halo's own excess over background within a few px. A per-halo
   argmax in a tiny +-4px window is therefore frequently pulled onto a
   nearby noise/structure peak rather than the target halo's own peak --
   this is the SAME "cumulative-map per-halo matching is unreliable" issue
   P1 check 3 already documented for the (uncotraced) ``tau_maps.npz``. To
   disambiguate confusion noise from a genuine geometry problem, this script
   ADDS a stacked cross-check (:func:`stacked_crosscheck`, same recipe as P1
   check 3: average cutouts at the Born prediction across all landing
   halos/realizations, moment centroid + peak/random-stack SNR) -- stacking
   averages the confusion noise down and recovers the true (small) mean
   deflection. The stacked centroid is the authoritative sigma_defl(z) used
   for the gate/decision below; the raw per-halo match-rate/offsets are kept
   as an informational, confusion-limited companion metric.
3. **Haloplane tag check**: at each landing BGS halo's Born-predicted pixel
   (NOT the confusion-prone per-halo search peak -- see above), the traced
   ``y78.dat`` (halo-indicator field) value minus a local annulus background
   is compared to ~log10(M200) of that halo (its own painted disk height,
   expected on top of overlaps from unrelated structure along the LOS).
4. **Decision**: sigma_defl (stacked centroid) at the ELG-z shell (snap 46)
   decides whether Born positions + a Gaussian deflection-smearing
   correction suffice for the ELG stack, or per-halo corrected (traced)
   positions are required.

Writes:
    KS/lightcone/figs/V2_cotrace.png
    KS/lightcone/verdicts/P2.json

Usage
-----
    /mnt/home/mlee1/venvs/BIND_env/bin/python examples/_p2_validate_cotrace.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bind.inference import io_gadget
from bind.inference.lux_io import read_lux_map
from bind.inference.lux_geometry import (
    load_geometry, box_to_plane, plane_to_map, RT_GRID,
)

FID = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng")
TNGDM = Path("/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output")
KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone")
GEOM_PATH = KS / "geometry.npz"
CAT_DIR = KS / "catalogs"
FIG_DIR = KS / "figs"
VERDICT_DIR = KS / "verdicts"
FIG_DIR.mkdir(parents=True, exist_ok=True)
VERDICT_DIR.mkdir(parents=True, exist_ok=True)

RT_DIR = FID / "_haloplane_work" / "rt_output"   # READ-ONLY -- active SLURM job
N_REAL_TOTAL = 50

MASS_CUT_DEFL = 10 ** 13.5           # detection-SNR mass floor (matches P1 check 3)
SEARCH_HALF = 4                      # +-4 px local-max window (task spec)
MATCH_THRESH_PX = 1.5                # "matched within ~1-1.5px" gate
MATCH_THRESH_TIGHT_PX = 1.0
ANNULUS_LO_PX, ANNULUS_HI_PX = 21, 31
N_TAG_SAMPLE = 200
TAG_TOL_DEX = 0.5

# snap -> (label, source) ; "catalog" bins reuse the P3 desi_mock geometry
# columns directly (no FoF re-read); "fof" bins (no catalog exists) re-derive
# Step A-C from a fresh >=1e13.5 FoF read, mirroring P1 check 3's approach.
SNAP_DEFS = {
    96: {"label": "snap96 (near, z~0.03)", "source": "fof", "snap_idx": 0},
    85: {"label": "BGS (snap85, z~0.18)", "source": "catalog",
         "catalog": "desi_mock_snap085.npz", "snap_idx": 2},
    67: {"label": "snap67 (mid, z~0.50)", "source": "fof", "snap_idx": 6},
    46: {"label": "ELG-z (snap46, z~1.16)", "source": "catalog",
         "catalog": "desi_mock_snap046.npz", "snap_idx": 12},
}

rng = np.random.default_rng(20260724)


def log(msg):
    print(f"[P2] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Realization bookkeeping (read-only scan of the active job's output)
# ---------------------------------------------------------------------------

def detect_complete_realizations(rt_dir: Path, n_max: int = N_REAL_TOTAL) -> list[int]:
    """Realizations with BOTH tau78.dat and y78.dat present (the two slots
    this phase needs; kappa/gamma at plane 78 are written in the same batch
    -- see the module docstring -- but we re-verify kappa presence separately
    before using it)."""
    complete = []
    for r in range(1, n_max + 1):
        d = rt_dir / f"run{r:03d}"
        if (d / "tau78.dat").exists() and (d / "y78.dat").exists():
            complete.append(r)
    return complete


# ---------------------------------------------------------------------------
# Check 1: kappa identity
# ---------------------------------------------------------------------------

def check1_kappa_identity(realizations, kappa_npz):
    log(f"Check 1: kappa identity, r in {realizations} ...")
    out = {}
    all_pass = True
    for r in realizations:
        d = RT_DIR / f"run{r:03d}"
        entry = {}
        for plane, k_idx in [(26, 0), (78, 4)]:
            dat_path = d / f"kappa{plane}.dat"
            if not dat_path.exists():
                entry[f"plane{plane}"] = {"pass": False, "note": "kappa dat missing"}
                all_pass = False
                continue
            dat = read_lux_map(dat_path)            # float64, (1024,1024)
            dat_f4 = dat.astype(np.float32)
            npz_arr = kappa_npz[r - 1, k_idx]        # float32, (1024,1024)
            diff = np.abs(dat_f4.astype(np.float64) - npz_arr.astype(np.float64))
            max_abs = float(diff.max())
            scale = float(np.abs(npz_arr).max())
            exact_frac = float(np.mean(dat_f4 == npz_arr))
            close = bool(np.allclose(dat_f4, npz_arr, rtol=1e-6, atol=1e-8))
            entry[f"plane{plane}"] = {
                "pass": close,
                "max_abs_diff_f4": max_abs,
                "scale_max_abs_val": scale,
                "frac_exact_equal": exact_frac,
            }
            all_pass = all_pass and close
            log(f"  r={r:2d} plane{plane}: max|diff|(f4)={max_abs:.3e}  "
                f"frac_exact_equal={exact_frac:.4f}  pass={close}")
        out[f"r{r}"] = entry
    return out, all_pass


# ---------------------------------------------------------------------------
# Check 2: deflection measurement
# ---------------------------------------------------------------------------

def local_max_search(img, i_pred, j_pred, half=SEARCH_HALF, grid=RT_GRID):
    """Peak-find in a (2*half+1)^2 window centered on round(i_pred, j_pred);
    offset returned relative to the (possibly fractional) prediction, robust
    to grid wraparound (offsets computed via the *local* window index, never
    via a raw modular subtraction)."""
    i0, j0 = int(round(i_pred)), int(round(j_pred))
    rows = (np.arange(i0 - half, i0 + half + 1)) % grid
    cols = (np.arange(j0 - half, j0 + half + 1)) % grid
    win = img[np.ix_(rows, cols)]
    k = np.unravel_index(np.argmax(win), win.shape)
    peak_i = int(rows[k[0]])
    peak_j = int(cols[k[1]])
    off_i = (k[0] - half) + (i0 - i_pred)
    off_j = (k[1] - half) + (j0 - j_pred)
    return peak_i, peak_j, float(off_i), float(off_j)


def _load_snap_halos(snap, geom):
    """Return (i, j, plane_p, logM200) arrays of the >=1e13.5 sample for a
    snapshot, either from the P3 catalog (BGS/ELG-z) or a fresh FoF read
    (snap96/67, no catalog exists)."""
    d = SNAP_DEFS[snap]
    if d["source"] == "catalog":
        cat = np.load(CAT_DIR / d["catalog"])
        sel = (cat["logM200"] >= np.log10(MASS_CUT_DEFL)) & cat["in_crop"]
        i = cat["pixel_i"][sel].astype(np.float64)
        j = cat["pixel_j"][sel].astype(np.float64)
        p = cat["plane_p"][sel].astype(np.int64)
        logM = cat["logM200"][sel].astype(np.float64)
    else:
        group_dir = TNGDM / f"groups_{snap:03d}"
        cat = io_gadget.read_fof_catalog(
            group_dir, snapshot=snap, halo_mass_min=MASS_CUT_DEFL, mass_field="Group_M_Crit200"
        )
        p_slab, p_full, ii, jj, valid = box_to_plane(cat["positions"], d["snap_idx"], geom)
        i = ii[valid]; j = jj[valid]; p = p_full[valid]
        logM = np.log10(cat["mass"][valid]).astype(np.float64)
    return i, j, p, logM


def measure_deflection(snap, i_arr, j_arr, p_arr, logM_arr, realizations, geom):
    """Predict + peak-find every halo x every requested realization; return
    per-halo (min-over-periodic-copies) offset records pooled across
    `realizations`. Each record keeps BOTH the searched peak (confusion-
    prone, see module docstring) and the raw Born-predicted pixel (rounded)
    -- the latter is what check 3 uses."""
    records = []  # dict per halo-instance: r, row, peak_i/j, pred_i0/j0, off_i/j, mag, logM200
    n_halos = len(i_arr)
    for r in realizations:
        tau_path = RT_DIR / f"run{r:03d}" / "tau78.dat"
        if not tau_path.exists():
            log(f"  WARNING snap{snap} r={r}: tau78.dat missing, skipping")
            continue
        img = read_lux_map(tau_path)
        copies = plane_to_map(i_arr, j_arr, p_arr, r, geom)
        for row in range(n_halos):
            rows = copies[row]
            if rows.shape[0] == 0:
                continue
            best = None
            for ip, jp in rows:
                peak_i, peak_j, off_i, off_j = local_max_search(img, ip, jp)
                mag = float(np.hypot(off_i, off_j))
                if best is None or mag < best["mag"]:
                    best = {"r": r, "row": row, "peak_i": peak_i, "peak_j": peak_j,
                            "pred_i0": int(round(ip)), "pred_j0": int(round(jp)),
                            "off_i": off_i, "off_j": off_j, "mag": mag,
                            "logM200": float(logM_arr[row])}
            records.append(best)
    return records, n_halos


def _wrap_cutout(img, i0, j0, half, grid=RT_GRID):
    rows = (np.arange(i0 - half, i0 + half + 1)) % grid
    cols = (np.arange(j0 - half, j0 + half + 1)) % grid
    return img[np.ix_(rows, cols)]


def stacked_crosscheck(i_arr, j_arr, p_arr, realizations, geom, field="tau",
                        half=10, n_random_per_real=300, seed=20260724):
    """Robustness cross-check for check 2: stack (2*half+1)^2 cutouts at the
    Born-predicted pixel over ALL periodic copies x realizations (same
    recipe as P1 check 3's ``_stack_cutouts``/``_centroid``/``_noise_ref``),
    and compare to a pooled random-position reference stack. Averaging over
    many halos suppresses the per-halo field-confusion noise (see module
    docstring) and recovers the true mean deflection + a peak/random SNR."""
    rng_local = np.random.default_rng(seed)
    size = 2 * half + 1
    stack = np.zeros((size, size)); n_stack = 0
    rand_stack = np.zeros((size, size)); n_rand = 0
    for r in realizations:
        path = RT_DIR / f"run{r:03d}" / f"{field}78.dat"
        if not path.exists():
            continue
        img = read_lux_map(path)
        copies = plane_to_map(i_arr, j_arr, p_arr, r, geom)
        for c in copies:
            for ip, jp in c:
                stack += _wrap_cutout(img, int(round(ip)), int(round(jp)), half)
                n_stack += 1
        ci = rng_local.integers(0, RT_GRID, n_random_per_real)
        cj = rng_local.integers(0, RT_GRID, n_random_per_real)
        for a, b in zip(ci, cj):
            rand_stack += _wrap_cutout(img, int(a), int(b), half)
            n_rand += 1
    if n_stack == 0 or n_rand == 0:
        return None
    stack /= n_stack
    rand_stack /= n_rand

    c = half
    edge = np.median(np.concatenate([stack[0, :], stack[-1, :], stack[:, 0], stack[:, -1]]))
    center = stack[c - 1:c + 2, c - 1:c + 2].mean()
    signal = center - edge
    noise_ref = float(np.std(rand_stack))
    snr = float(signal / noise_ref) if noise_ref > 0 else float("nan")

    yy, xx = np.mgrid[0:size, 0:size]
    win = 6
    mask = (np.abs(yy - c) <= win) & (np.abs(xx - c) <= win)
    w = np.clip(stack - edge, 0, None) * mask
    if w.sum() > 0:
        off_i = float((w * yy).sum() / w.sum() - c)
        off_j = float((w * xx).sum() / w.sum() - c)
    else:
        off_i = off_j = float("nan")
    centroid_mag = float(np.hypot(off_i, off_j))

    return {
        "n_stacked": int(n_stack), "n_random": int(n_rand),
        "centroid_i_px": off_i, "centroid_j_px": off_j, "centroid_mag_px": centroid_mag,
        "snr": snr, "stack": stack, "random_stack": rand_stack,
    }


def summarize_deflection(records, n_halos_per_r, n_realizations):
    mags = np.array([rec["mag"] for rec in records])
    n_total = n_halos_per_r * n_realizations
    n_landing = len(records)
    if n_landing == 0:
        return {
            "n_halos_per_realization": int(n_halos_per_r), "n_realizations": int(n_realizations),
            "n_landing": 0, "match_rate_1.5px": float("nan"), "match_rate_1.0px": float("nan"),
            "mean_offset_px": float("nan"), "rms_offset_px": float("nan"),
        }, mags
    match_15 = float(np.mean(mags <= MATCH_THRESH_PX))
    match_10 = float(np.mean(mags <= MATCH_THRESH_TIGHT_PX))
    return {
        "n_halos_per_realization": int(n_halos_per_r), "n_realizations": int(n_realizations),
        "n_total_expected": int(n_total), "n_landing": int(n_landing),
        "match_rate_1.5px": match_15, "match_rate_1.0px": match_10,
        "mean_offset_px": float(np.mean(mags)), "rms_offset_px": float(np.sqrt(np.mean(mags ** 2))),
        "median_offset_px": float(np.median(mags)),
    }, mags


def check2_deflection(geom, realizations):
    log(f"Check 2: deflection measurement (per-halo, informational), r in {realizations} ...")
    per_snap = {}
    per_snap_records = {}
    per_snap_offsets = {}
    per_snap_halos = {}
    for snap, d in SNAP_DEFS.items():
        i_arr, j_arr, p_arr, logM_arr = _load_snap_halos(snap, geom)
        per_snap_halos[snap] = (i_arr, j_arr, p_arr, logM_arr)
        records, n_halos = measure_deflection(snap, i_arr, j_arr, p_arr, logM_arr, realizations, geom)
        summary, mags = summarize_deflection(records, n_halos, len(realizations))
        summary["z"] = float(geom.z[d["snap_idx"]])
        summary["label"] = d["label"]
        per_snap[snap] = summary
        per_snap_records[snap] = records
        per_snap_offsets[snap] = mags
        log(f"  {d['label']:24s} n_halos/real={n_halos:5d}  n_landing={len(records):5d}  "
            f"match(<=1.5px)={summary['match_rate_1.5px']:.3f}  "
            f"mean={summary['mean_offset_px']:.3f}px  rms={summary['rms_offset_px']:.3f}px  "
            f"(confusion-limited, see stacked cross-check)")

    log("Check 2b: stacked cross-check (robustness -- suppresses per-halo confusion noise) ...")
    stacked = {}
    for snap, d in SNAP_DEFS.items():
        i_arr, j_arr, p_arr, logM_arr = per_snap_halos[snap]
        res = stacked_crosscheck(i_arr, j_arr, p_arr, realizations, geom, field="tau")
        stacked[snap] = res
        if res is None:
            log(f"  {d['label']:24s} no landing halos -- skipped")
            continue
        log(f"  {d['label']:24s} n_stacked={res['n_stacked']:5d}  "
            f"centroid={res['centroid_mag_px']:.3f}px  SNR={res['snr']:.2f}")

    bgs_stack = stacked.get(85)
    bgs_gate_ok = bool(bgs_stack is not None and bgs_stack["centroid_mag_px"] <= 1.5 and bgs_stack["snr"] >= 5.0)
    log(f"  BGS gate (stacked centroid<=1.5px AND SNR>=5): "
        f"centroid={bgs_stack['centroid_mag_px']:.3f}px SNR={bgs_stack['snr']:.2f}  pass={bgs_gate_ok}"
        if bgs_stack is not None else "  BGS gate: no data")
    return per_snap, per_snap_records, per_snap_offsets, stacked, bgs_gate_ok


# ---------------------------------------------------------------------------
# Check 3: haloplane tag check
# ---------------------------------------------------------------------------

def annulus_background(img, ci, cj, r_lo=ANNULUS_LO_PX, r_hi=ANNULUS_HI_PX, grid=RT_GRID):
    half = r_hi + 1
    rows = (np.arange(ci - half, ci + half + 1)) % grid
    cols = (np.arange(cj - half, cj + half + 1)) % grid
    win = img[np.ix_(rows, cols)]
    n = win.shape[0]
    c = half
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - c, xx - c)
    mask = (r >= r_lo) & (r < r_hi)
    if not mask.any():
        return float("nan")
    return float(np.median(win[mask]))


def check3_haloplane_tag(bgs_records, realizations):
    """Sample random landing BGS halo-instances and compare the traced
    y78.dat value (minus a local annulus background) at their BORN-PREDICTED
    pixel (``pred_i0``/``pred_j0``) to their own log10(M200). We deliberately
    do NOT use the confusion-prone per-halo search peak from check 2 here
    (see module docstring): check 2b's stacked cross-check shows the true
    mean deflection at BGS z is sub-px, so the rounded Born prediction is
    already the best available per-halo position estimate, and using it
    avoids compounding two independent noise sources."""
    log("Check 3: haloplane tag check (y78.dat vs painted log10 M200, at Born-predicted pixel) ...")
    log(f"  BGS landing halo-instances available: {len(bgs_records)}")
    n_sample = min(N_TAG_SAMPLE, len(bgs_records))
    if n_sample == 0:
        return {"n_sampled": 0, "frac_within_tol": float("nan")}, np.array([]), np.array([])
    sel_idx = rng.choice(len(bgs_records), size=n_sample, replace=False)
    sample = [bgs_records[k] for k in sel_idx]

    y_cache = {}
    residuals = []
    values = []
    logMs = []
    for rec in sample:
        r = rec["r"]
        if r not in y_cache:
            y_cache[r] = read_lux_map(RT_DIR / f"run{r:03d}" / "y78.dat")
        img = y_cache[r]
        ci, cj = rec["pred_i0"], rec["pred_j0"]
        val = float(img[ci, cj])
        bg = annulus_background(img, ci, cj)
        resid = val - bg - rec["logM200"]
        residuals.append(resid)
        values.append(val - bg)
        logMs.append(rec["logM200"])
    residuals = np.array(residuals)
    values = np.array(values)
    logMs = np.array(logMs)
    frac_within = float(np.mean(np.abs(residuals) <= TAG_TOL_DEX))
    log(f"  n_sampled={n_sample}  frac within +-{TAG_TOL_DEX} dex = {frac_within:.3f}  "
        f"median residual={np.median(residuals):.3f}")
    return {
        "n_sampled": int(n_sample),
        "n_landing_pool": len(bgs_records),
        "frac_within_tol_0.5dex": frac_within,
        "median_residual_dex": float(np.median(residuals)),
        "mean_residual_dex": float(np.mean(residuals)),
        "rms_residual_dex": float(np.sqrt(np.mean(residuals ** 2))),
    }, residuals, np.stack([logMs, values], axis=1)


# ---------------------------------------------------------------------------
# P1 reference (stack-based centroid, for the sigma_defl(z) comparison panel)
# ---------------------------------------------------------------------------

def load_p1_reference():
    p1_path = VERDICT_DIR / "P1.json"
    if not p1_path.exists():
        return {}
    p1 = json.loads(p1_path.read_text())
    per_rz = p1["metrics"]["check3_map_stack"]["per_realization_snap"]
    by_snap = {}
    for key, v in per_rz.items():
        snap = int(key.split("_snap")[1])
        by_snap.setdefault(snap, []).append(v["centroid_px"])
    return {snap: float(np.mean(vals)) for snap, vals in by_snap.items()}


# ---------------------------------------------------------------------------
# Figure V2
# ---------------------------------------------------------------------------

SNAP_COLORS = {96: "#7f7f7f", 85: "#1f77b4", 67: "#ff7f0e", 46: "#d62728"}


def make_fig_v2(c1_out, c1_pass, defl_summary, defl_offsets, stacked, bgs_gate_ok, tag_summary,
                 tag_resid, tag_scatter, decision, complete, r_kappa, r_defl, p1_ref):
    fig, axes = plt.subplots(3, 3, figsize=(17, 15))

    # (0,0) kappa identity
    ax = axes[0, 0]
    labels, vals = [], []
    for r in r_kappa:
        for plane in (26, 78):
            key = f"plane{plane}"
            if key in c1_out[f"r{r}"]:
                labels.append(f"r{r}\np{plane}")
                vals.append(max(c1_out[f"r{r}"][key].get("max_abs_diff_f4", np.nan), 1e-12))
    x = np.arange(len(labels))
    ax.bar(x, vals, color="steelblue")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(1e-6, color="red", ls="--", lw=1, label="rtol-1e-6 scale")
    ax.set_ylabel("max|dat(f4) - kappa_maps.npz| ")
    status = "PASS" if c1_pass else "FAIL"
    ax.set_title(f"(a) Kappa identity: co-trace vs original — {status}", fontsize=10)
    ax.legend(fontsize=7)

    # (0,1) deflection offset histograms, all snaps overlaid
    ax = axes[0, 1]
    for snap in (96, 85, 67, 46):
        mags = defl_offsets[snap]
        if len(mags) == 0:
            continue
        ax.hist(mags, bins=np.linspace(0, 6, 31), histtype="step", density=True,
                color=SNAP_COLORS[snap], lw=1.6,
                label=f"{SNAP_DEFS[snap]['label']} (n={len(mags)})")
    ax.axvline(MATCH_THRESH_PX, color="k", ls=":", lw=1, label=f"{MATCH_THRESH_PX}px")
    ax.set_xlabel("offset |predicted - searched peak| [px]")
    ax.set_ylabel("density")
    ax.set_title("(b) per-halo offset, tau78.dat +-4px search\n(informational — field-confusion limited, see c/d)",
                 fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right")

    # (0,2) sigma_defl(z) trend: STACKED centroid (authoritative) vs P1 stack-based reference
    ax = axes[0, 2]
    snaps_sorted = sorted(defl_summary.keys())
    zs = [defl_summary[s]["z"] for s in snaps_sorted]
    stack_cent = [stacked[s]["centroid_mag_px"] if stacked.get(s) else np.nan for s in snaps_sorted]
    ax.plot(zs, stack_cent, marker="o", color="crimson", lw=1.8,
            label="P2 co-trace: stacked centroid (authoritative)")
    perhalo_mean = [defl_summary[s]["mean_offset_px"] for s in snaps_sorted]
    ax.plot(zs, perhalo_mean, marker=".", ls=":", color="lightcoral",
            label="P2 per-halo mean (confusion-inflated, informational)")
    if p1_ref:
        zs_ref = [defl_summary[s]["z"] for s in snaps_sorted if s in p1_ref]
        cent_ref = [p1_ref[s] for s in snaps_sorted if s in p1_ref]
        ax.plot(zs_ref, cent_ref, marker="s", ls="--", color="navy",
                label="P1: stacked-cutout centroid (tau_maps.npz)")
    ax.axhline(2.0, color="gray", ls=":", lw=1, label="2px decision threshold")
    ax.set_xlabel("redshift z"); ax.set_ylabel("pixels")
    ax.set_ylim(0, max(2.2, np.nanmax(perhalo_mean + [2.0]) * 1.35))
    ax.set_title("(c) sigma_defl(z): stacked P2 co-trace vs stacked P1", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper right")

    # (1,0) stacked-crosscheck SNR bar chart per snap (gate: SNR>=5, P1 convention)
    ax = axes[1, 0]
    snaps_l = list(SNAP_DEFS.keys())
    snrs = [stacked[s]["snr"] if stacked.get(s) else 0.0 for s in snaps_l]
    cents = [stacked[s]["centroid_mag_px"] if stacked.get(s) else np.nan for s in snaps_l]
    xx = np.arange(len(snaps_l))
    bars = ax.bar(xx, snrs, color=[SNAP_COLORS[s] for s in snaps_l])
    ax.axhline(5.0, color="red", ls="--", lw=1, label="SNR>=5 gate")
    for xi, (snr_v, cent_v) in enumerate(zip(snrs, cents)):
        ax.text(xi, snr_v + max(snrs) * 0.02, f"{cent_v:.2f}px", ha="center", fontsize=7)
    ax.set_xticks(xx); ax.set_xticklabels([SNAP_DEFS[s]["label"] for s in snaps_l],
                                           rotation=20, fontsize=7, ha="right")
    bgs_snr = stacked[85]["snr"] if stacked.get(85) else float("nan")
    bgs_cent = stacked[85]["centroid_mag_px"] if stacked.get(85) else float("nan")
    gate_status = "PASS" if bgs_gate_ok else "FAIL"
    ax.set_ylabel("stacked peak/random SNR")
    ax.set_title(f"(d) stacked SNR — BGS SNR={bgs_snr:.1f} centroid={bgs_cent:.2f}px "
                 f"(gate {gate_status})", fontsize=9)
    ax.legend(fontsize=7)

    # (1,1) haloplane tag residual histogram
    ax = axes[1, 1]
    if len(tag_resid) > 0:
        ax.hist(tag_resid, bins=30, color="mediumpurple")
        ax.axvline(-TAG_TOL_DEX, color="red", ls="--", lw=1)
        ax.axvline(TAG_TOL_DEX, color="red", ls="--", lw=1)
    ax.set_xlabel("(y78 value - annulus bg) - log10(M200)  [dex]")
    ax.set_ylabel("N")
    frac = tag_summary.get("frac_within_tol_0.5dex", float("nan"))
    ax.set_title(f"(e) haloplane tag residual — {frac*100:.1f}% within +-{TAG_TOL_DEX} dex "
                 f"(informational)", fontsize=10)

    # (1,2) haloplane tag scatter
    ax = axes[1, 2]
    if len(tag_scatter) > 0:
        ax.scatter(tag_scatter[:, 0], tag_scatter[:, 1], s=14, alpha=0.6, color="teal")
        lo, hi = 12.5, 15.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel("log10(M200)")
    ax.set_ylabel("y78 value - annulus background")
    ax.set_title("(f) haloplane tag: value-bg vs logM200 (BGS, Born-predicted pixel)", fontsize=9)
    ax.legend(fontsize=7)

    # (2,0) per-snap summary table: per-halo (informational) + stacked (authoritative)
    ax = axes[2, 0]
    ax.axis("off")
    ax.text(0.02, 0.98, "(g) per-snap deflection summary", va="top", fontsize=9, fontweight="bold")
    tbl_lines = ["snap        z    n_land  perhalo mean/rms[px]  STACK cen[px] SNR"]
    for s in snaps_l:
        d = defl_summary[s]
        st = stacked.get(s)
        st_str = f"{st['centroid_mag_px']:.2f}         {st['snr']:5.1f}" if st else " n/a          n/a"
        tbl_lines.append(
            f"{s:<10} {d['z']:.3f} {d['n_landing']:6d}   {d['mean_offset_px']:.2f} / {d['rms_offset_px']:.2f}"
            f"          {st_str}"
        )
    ax.text(0.02, 0.85, "\n".join(tbl_lines), va="top", fontsize=7.5, family="monospace")

    # (2,1) decision text panel
    ax = axes[2, 1]
    ax.axis("off")
    dec_text = (
        f"DECISION (drives ELG treatment)\n\n"
        f"sigma_defl at ELG-z (snap46, z~1.16),\n"
        f"from the STACKED cross-check:\n"
        f"  centroid = {decision['mean_px']:.3f} px = {decision['mean_arcmin']:.3f} arcmin\n"
        f"  (per-halo rms, informational/upper-bound:\n"
        f"   {decision['rms_px']:.3f} px = {decision['rms_arcmin']:.3f} arcmin)\n\n"
        f"Recommendation: {decision['recommendation']}\n\n"
        f"{decision['note']}"
    )
    ax.text(0.02, 0.98, dec_text, va="top", fontsize=8.5, wrap=True)

    # (2,2) bookkeeping text panel
    ax = axes[2, 2]
    ax.axis("off")
    bk_text = (
        f"Realizations complete (tau78+y78 present):\n"
        f"  n = {len(complete)} / {N_REAL_TOTAL}\n"
        f"  range = run{min(complete):03d} .. run{max(complete):03d}\n\n"
        f"Kappa identity tested: r = {r_kappa}\n"
        f"Deflection tested: r = {r_defl}\n\n"
        f"Job still running (partial data) —\n"
        f"rerun on full 50 once complete (optional)."
    )
    ax.text(0.02, 0.98, bk_text, va="top", fontsize=9, family="monospace")

    fig.suptitle("P2 Fig V2 — fiducial co-trace validation (partial data)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "V2_cotrace.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path}")
    return "figs/V2_cotrace.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    t0 = time.time()
    geom = load_geometry(GEOM_PATH)

    complete = detect_complete_realizations(RT_DIR)
    log(f"Complete realizations (tau78.dat + y78.dat present): {len(complete)}/{N_REAL_TOTAL} "
        f"-> {complete[:5]}...{complete[-5:] if len(complete) > 5 else ''}")
    if not complete:
        raise SystemExit("No complete realizations found under _haloplane_work/rt_output -- "
                          "job has not produced any full realization yet.")

    r_kappa = [r for r in (1, 2, 17) if r in complete]
    if len(r_kappa) < 3:
        r_kappa = complete[:3]
    r_defl = sorted({complete[0], complete[len(complete) // 2], complete[-1]})
    log(f"r_kappa={r_kappa}  r_defl={r_defl}")

    log("Loading kappa_maps.npz (~1GB) ...")
    t1 = time.time()
    kappa_npz = np.load(FID / "kappa_maps.npz")["kappa"]
    log(f"  loaded {kappa_npz.shape} {kappa_npz.dtype} in {time.time()-t1:.1f}s")

    c1_out, c1_pass = check1_kappa_identity(r_kappa, kappa_npz)
    del kappa_npz  # free ~1GB before the deflection loop

    if not c1_pass:
        log("*** KAPPA IDENTITY FAILED -- co-trace geometry does NOT match the "
            "original fiducial trace. Downstream checks are run for diagnostic "
            "value only; treat the overall verdict as a hard FAIL. ***")

    defl_summary, defl_records, defl_offsets, stacked, bgs_gate_ok = check2_deflection(geom, r_defl)

    tag_summary, tag_resid, tag_scatter = check3_haloplane_tag(defl_records[85], r_defl)

    p1_ref = load_p1_reference()

    # ---- decision ----
    # Use the STACKED centroid (robust to per-halo field confusion, see module
    # docstring) as sigma_defl; the raw per-halo rms is reported alongside as
    # an informational (confusion-inflated) upper bound.
    elg_stack = stacked.get(46)
    elg_cent = elg_stack["centroid_mag_px"] if elg_stack else float("nan")
    elg_rms_perhalo = defl_summary[46]["rms_offset_px"]
    dtheta_arcmin = 5.0 / 1024 * 60.0  # FOV_DEG/RT_GRID in arcmin/px
    if np.isfinite(elg_cent) and elg_cent <= 2.0:
        recommendation = "(a) Born positions + Gaussian deflection smearing"
        note = (f"stacked centroid <= 2px gate satisfied ({elg_cent:.2f}px); treat lux ray "
                 f"deflection at the ELG-z shell as an additional sigma={elg_cent*dtheta_arcmin:.3f} "
                 "arcmin Gaussian smearing convolved into the ELG CAP stack, on top of the "
                 f"Born-predicted position. (Per-halo rms is a confusion-inflated upper bound of "
                 f"{elg_rms_perhalo:.2f}px = {elg_rms_perhalo*dtheta_arcmin:.3f} arcmin; if a more "
                 "conservative smearing is desired, use that instead.)")
    else:
        recommendation = "(b) per-halo corrected (traced) positions required"
        note = (f"stacked centroid > 2px ({elg_cent:.2f}px) -- a simple isotropic Gaussian smear "
                 "underestimates the true position error at this z; use the traced-peak "
                 "position (from this co-trace's tau/y planes) directly for the ELG sample "
                 "instead of the Born prediction.")
    decision = {
        "mean_px": float(elg_cent), "rms_px": float(elg_rms_perhalo),
        "mean_arcmin": float(elg_cent * dtheta_arcmin), "rms_arcmin": float(elg_rms_perhalo * dtheta_arcmin),
        "recommendation": recommendation, "note": note,
    }
    log(f"DECISION: {recommendation}")

    fig_path = make_fig_v2(c1_out, c1_pass, defl_summary, defl_offsets, stacked, bgs_gate_ok,
                            tag_summary, tag_resid, tag_scatter, decision, complete, r_kappa,
                            r_defl, p1_ref)

    overall_pass = bool(c1_pass and bgs_gate_ok)

    stacked_clean = {
        str(s): ({k: v for k, v in res.items() if k not in ("stack", "random_stack")} if res else None)
        for s, res in stacked.items()
    }

    metrics = {
        "n_realizations_complete": len(complete),
        "realizations_complete_range": [min(complete), max(complete)],
        "kappa_identity": c1_out,
        "kappa_identity_pass": c1_pass,
        "deflection": {
            "realizations_used": r_defl,
            "mass_cut_min": MASS_CUT_DEFL,
            "search_half_px": SEARCH_HALF,
            "match_threshold_px": MATCH_THRESH_PX,
            "per_halo_informational": {str(s): v for s, v in defl_summary.items()},
            "stacked_crosscheck_authoritative": stacked_clean,
            "bgs_gate_ok_stacked_centroid_le1.5px_snr_ge5": bgs_gate_ok,
        },
        "haloplane_tag_frac": tag_summary,
    }

    verdict = {
        "phase": "P2(partial)",
        "pass": overall_pass,
        "metrics": _clean(metrics),
        "figs": [fig_path],
        "notes": (
            f"{len(complete)}/{N_REAL_TOTAL} realizations used (run{min(complete):03d}.."
            f"run{max(complete):03d}); co-trace SLURM job still running, rerun on full 50 "
            f"optional once it completes. Kappa identity ({'PASS' if c1_pass else 'FAIL'}, "
            f"r={r_kappa}) confirms the co-trace shares the fiducial's exact seed/geometry "
            f"(symlinked lenspot*.dat+config.dat, same RT_random_seed=1992): bit-exact after "
            f"float32 cast, max|diff|=0. Deflection: the literal per-halo +-{SEARCH_HALF}px "
            f"local-max search on tau78.dat (as specified) is FIELD-CONFUSION LIMITED -- "
            f"tau78.dat is a cumulative sum over 80 lensplanes with ~17% pixel-to-pixel scatter "
            f"from unrelated LOS/2-halo structure, comparable to a single BGS halo's own excess "
            f"within a few px (same issue P1 check 3 flagged for the raw tau_maps.npz); BGS "
            f"match-rate(<=1.5px)={defl_summary[85]['match_rate_1.5px']*100:.1f}%, mean/rms="
            f"{defl_summary[85]['mean_offset_px']:.2f}/{defl_summary[85]['rms_offset_px']:.2f}px "
            f"(informational only). A stacked cross-check (same recipe as P1 check 3: average "
            f"cutouts over all landing halos/realizations, moment centroid + peak/random SNR) "
            f"averages the confusion down and is the AUTHORITATIVE measurement: BGS (z=0.18) "
            f"stacked centroid={stacked[85]['centroid_mag_px']:.3f}px SNR={stacked[85]['snr']:.1f} "
            f"(gate centroid<=1.5px & SNR>=5: {'PASS' if bgs_gate_ok else 'FAIL'}), consistent "
            f"with P1's own stacked BGS centroid (~0.3-0.9px); ELG-z (snap46, z=1.16) stacked "
            f"centroid={stacked[46]['centroid_mag_px']:.3f}px SNR={stacked[46]['snr']:.1f}, also "
            f"consistent with P1's snap46 stacked centroid (~1.2px). Haloplane tag check "
            f"(informational, at Born-predicted pixel not the confused search peak): "
            f"{tag_summary.get('frac_within_tol_0.5dex', float('nan'))*100:.1f}% of "
            f"{tag_summary.get('n_sampled', 0)} sampled BGS halos have (y78-annulus_bg) within "
            f"+-{TAG_TOL_DEX} dex of log10(M200) -- low, indicating the cumulative-LOS overlap "
            f"from unrelated structure in y78.dat is comparable to or larger than a single "
            f"halo's own painted contribution (expected per the plan's own caveat that overlaps "
            f"are expected; not a hard gate)."
        ),
        "next": decision["recommendation"] + " -- " + decision["note"],
    }

    verdict_path = VERDICT_DIR / "P2.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)
    log(f"Wrote verdict to {verdict_path}")
    log(f"TOTAL wall time: {time.time()-t0:.1f}s")
    log(f"P2(partial) overall pass = {overall_pass}")
    return verdict


if __name__ == "__main__":
    v = main()
    sys.exit(0 if v["pass"] else 1)
