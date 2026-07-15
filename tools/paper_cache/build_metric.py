#!/usr/bin/env python3
"""Parallel, resumable cache builders for the BIND2 paper figures.

Each metric computes a per-sim *partial* (saved under CACHE_DIR/partials/<metric>/
as <suite>__<sim>.npz); ``--reduce`` collates the partials into the compact
artifact(s) the load-only figure notebooks read. Mirrors the array-ready +
resumable + ``--reduce`` pattern of tools/partial_supp_sobol.py.

Examples
--------
    # local, all suites, 8-way pool
    python build_metric.py --metric mass --pool 8
    python build_metric.py --metric mass --reduce

    # SLURM array slice
    python build_metric.py --metric pk --suite CV --chunk $SLURM_ARRAY_TASK_ID --n-chunks 4

    # explicit sims (smoke)
    python build_metric.py --metric mass --sim_ids CV/sim_0,CV/sim_1
"""
from __future__ import annotations

import argparse
import json
import sys
import pickle
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_config as C  # noqa: E402
from bind.metrics import radial_profile, power_spectrum_pylians_2d  # noqa: E402

R_OVER_R200 = np.linspace(0.05, 2.0, 16)      # r/R200 grid for the profile-Spearman
PROF_NBINS = 32                                # linear radial bins (mass profiles)
THERMO_PROF_NBINS = 24
THERMO_INT_KEYS = ["Y", "Tx", "K", "P", "Mgas"]  # aperture-integrated thermo/gas
_THERMO_PIX_HALOS = 20                         # halos/sim sampled for the pixel PDF
_THERMO_PIX_PER_HALO = 800


def _spearman_vec(P, y):
    """Spearman rho of each of the N_PARAMS columns of P against per-sim vector y."""
    from scipy.stats import spearmanr
    rho = np.full(C.N_PARAMS, np.nan)
    y = np.asarray(y, float)
    fy = np.isfinite(y)
    for j in range(C.N_PARAMS):
        x = P[:, j]
        m = fy & np.isfinite(x)
        if m.sum() >= 5 and np.ptp(x[m]) > 0:
            rho[j] = spearmanr(x[m], y[m]).statistic
    return rho


def _window_mask(masses, window):
    lo, hi = window
    m = np.asarray(masses, np.float64)
    return (m >= lo) & (m < hi)


# ════════════════════════════════════════════════════════════════════════════
# Shared per-sim helpers
# ════════════════════════════════════════════════════════════════════════════
def _load_mass_patches(rec):
    """Return (truth_mass (N,3,128,128), gen (N,7,128,128), catalog, centers_pix)."""
    fm = C.load_full_maps(rec)
    cat = C.load_catalog(rec)
    gen = C.load_generated(rec)
    centers_pix = C.centers_to_pixels(cat["centers"])
    truth = C.extract_truth_mass_patches(fm, centers_pix)
    return fm, truth, gen, cat, centers_pix


def _aperture_masks(cat):
    """List of boolean R200c aperture masks (one per halo) on the 128² patch grid."""
    r_pix = C.r200_pix_patch(cat)
    return [C.RR_PIX_PATCH < r for r in r_pix], r_pix


def _run_config(rec) -> dict:
    """Composite settings from the sim's summary.json (fallback to eval defaults)."""
    cfg = {"patch_mass_match": True, "taper_frac": 0.15, "r200_factor": 4.0}
    try:
        rc = json.loads(Path(rec["summary"]).read_text())["run_config"]
        for k in cfg:
            if k in rc:
                cfg[k] = rc[k]
    except Exception:
        pass
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# mass — integrated + R200-aperture per-channel masses
# ════════════════════════════════════════════════════════════════════════════
def compute_mass(rec):
    fm, truth, gen7, cat, _ = _load_mass_patches(rec)
    gen = gen7[:, : C.N_MASS_CH]
    masses = np.asarray(cat["masses"], np.float64)
    n = len(masses)
    tm_full = truth.sum((2, 3)).astype(np.float64)
    gm_full = gen.sum((2, 3)).astype(np.float64)
    masks, _ = _aperture_masks(cat)
    tm_rv = np.zeros((n, C.N_MASS_CH))
    gm_rv = np.zeros((n, C.N_MASS_CH))
    for i in range(n):
        m = masks[i]
        for c in range(C.N_MASS_CH):
            tm_rv[i, c] = np.maximum(truth[i, c], 0.0)[m].sum()
            gm_rv[i, c] = np.maximum(gen[i, c], 0.0)[m].sum()
    return dict(masses=masses, tm_full=tm_full, gm_full=gm_full, tm_rv=tm_rv, gm_rv=gm_rv,
                params=C.sim_params(cat))


def reduce_mass(items):
    rows = []
    for rec, d in items:
        masses = d["masses"]
        logm = np.log10(masses)
        mb = C.mass_bin_index(logm)
        for i in range(len(masses)):
            row = dict(suite=rec["suite"], sim_id=rec["sim_id"], halo_mass=float(masses[i]),
                       log_m200c=float(logm[i]), mass_bin=int(mb[i]))
            for c, ch in enumerate(C.MASS_CHANNELS):
                row[f"truth_{ch}"] = d["tm_full"][i, c]
                row[f"gen_{ch}"] = d["gm_full"][i, c]
                row[f"truth_{ch}_rvir"] = d["tm_rv"][i, c]
                row[f"gen_{ch}_rvir"] = d["gm_rv"][i, c]
            rows.append(row)
    df = pd.DataFrame(rows)
    out = C.CACHE_DIR / "mass_table.pkl"
    df.to_pickle(out)
    print(f"  wrote {out}  ({len(df)} halos, {df['sim_id'].nunique()} sims)")

    # per-sim summary + parameter-response Spearman per mass window (Fig 3a + appendix)
    ch_keys = C.MASS_CHANNELS + ["Total"]
    cols = list(zip(C.MASS_CHANNELS, range(C.N_MASS_CH))) + [("Total", None)]
    Plist, srows = [], []
    vals = {w: {ch: {"true": [], "gen": []} for ch in ch_keys} for w in C.PARAM_WINDOWS}
    for rec, d in items:
        if "params" not in d:
            continue
        Plist.append(d["params"])
        masses = np.asarray(d["masses"], np.float64)
        row = {"suite": rec["suite"], "sim_id": rec["sim_id"]}
        row.update({f"p{j+1}": float(d["params"][j]) for j in range(C.N_PARAMS)})
        for w, win in C.PARAM_WINDOWS.items():
            sel = _window_mask(masses, win)
            tmf, gmf = d["tm_full"][sel], d["gm_full"][sel]
            for ch, c in cols:
                t = tmf[:, c] if c is not None else tmf.sum(1)
                g = gmf[:, c] if c is not None else gmf.sum(1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    tv = np.nanmean(np.log10(np.where(t > 0, t, np.nan))) if sel.any() else np.nan
                    gv = np.nanmean(np.log10(np.where(g > 0, g, np.nan))) if sel.any() else np.nan
                vals[w][ch]["true"].append(tv)
                vals[w][ch]["gen"].append(gv)
                row[f"true_logmean_{ch}_{w}"] = tv
                row[f"gen_logmean_{ch}_{w}"] = gv
        srows.append(row)
    sdf = pd.DataFrame(srows)
    P = np.array(Plist)
    rho_mass = {}
    for w in C.PARAM_WINDOWS:
        rho_mass[w] = {"True": np.array([_spearman_vec(P, vals[w][ch]["true"]) for ch in ch_keys]),
                       "BIND": np.array([_spearman_vec(P, vals[w][ch]["gen"]) for ch in ch_keys])}
    pickle.dump(dict(sim_table=sdf, channels=ch_keys, rho_mass=rho_mass),
                open(C.CACHE_DIR / "mass_param.pkl", "wb"))
    print(f"  wrote mass_param.pkl  ({len(sdf)} sims, windows {list(C.PARAM_WINDOWS)})")


# ════════════════════════════════════════════════════════════════════════════
# profiles — per-halo mass-channel radial profiles, reduced to bands by mass bin
# ════════════════════════════════════════════════════════════════════════════
def compute_profiles(rec):
    fm, truth, gen7, cat, _ = _load_mass_patches(rec)
    gen = gen7[:, : C.N_MASS_CH]
    masses = np.asarray(cat["masses"], np.float64)
    n = len(masses)
    r = None
    tprof = np.zeros((n, C.N_MASS_CH, PROF_NBINS))
    gprof = np.zeros((n, C.N_MASS_CH, PROF_NBINS))
    for h in range(n):
        for c in range(C.N_MASS_CH):
            rr, tp = radial_profile(truth[h, c], n_bins=PROF_NBINS, logspace=False)
            _, gp = radial_profile(gen[h, c], n_bins=PROF_NBINS, logspace=False)
            tprof[h, c] = tp
            gprof[h, c] = gp
            if r is None:
                r = rr
    logm = np.log10(masses)
    return dict(r=r * C.MPC_PER_PIX_PATCH, truth=tprof, gen=gprof,
                mass_bin=C.mass_bin_index(logm))


def _profile_bands(t_list, g_list):
    t = np.concatenate(t_list)  # (M, nch, nb)
    g = np.concatenate(g_list)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdiff = np.where(t > 0, (g - t) / t, np.nan)
    res = {"n": int(len(t))}
    for name, arr in [("truth", t), ("gen", g), ("pctdiff", pdiff)]:
        res[f"{name}_med"] = np.nanmedian(arr, 0)
        res[f"{name}_p16"] = np.nanpercentile(arr, 16, 0)
        res[f"{name}_p84"] = np.nanpercentile(arr, 84, 0)
    return res


def _reduce_profile_like(items, r_key, t_key, g_key, mb_key, out_name):
    r = None
    by_suite_bin = defaultdict(lambda: {"t": [], "g": []})
    by_bin = defaultdict(lambda: {"t": [], "g": []})
    for rec, d in items:
        if r is None:
            r = d[r_key]
        mb = d[mb_key]
        for b in range(C.N_MASS_BINS):
            m = mb == b
            if m.any():
                by_suite_bin[(rec["suite"], b)]["t"].append(d[t_key][m])
                by_suite_bin[(rec["suite"], b)]["g"].append(d[g_key][m])
                by_bin[b]["t"].append(d[t_key][m])
                by_bin[b]["g"].append(d[g_key][m])
    out = {"r": r, "by_suite_bin": {}, "by_bin": {}}
    for k, v in by_suite_bin.items():
        out["by_suite_bin"][k] = _profile_bands(v["t"], v["g"])
    for b, v in by_bin.items():
        out["by_bin"][b] = _profile_bands(v["t"], v["g"])
    path = C.CACHE_DIR / out_name
    pickle.dump(out, open(path, "wb"))
    tot = sum(v["n"] for v in out["by_bin"].values())
    print(f"  wrote {path}  ({tot} halos over {len(out['by_bin'])} mass bins)")


def reduce_profiles(items):
    _reduce_profile_like(items, "r", "truth", "gen", "mass_bin", "profiles.pkl")


# ════════════════════════════════════════════════════════════════════════════
# profiles_r200 — per-sim mean r/R200 profile + 35-param Spearman (fixes Fig 3b)
# ════════════════════════════════════════════════════════════════════════════
def compute_profiles_r200(rec):
    fm, truth, gen7, cat, _ = _load_mass_patches(rec)
    gen = gen7[:, : C.N_MASS_CH]
    r_pix = C.r200_pix_patch(cat)
    masses = np.asarray(cat["masses"], np.float64)
    n = len(masses)
    t_all = np.zeros((n, C.N_MASS_CH, len(R_OVER_R200)))
    g_all = np.zeros((n, C.N_MASS_CH, len(R_OVER_R200)))
    for h in range(n):
        rfine, _ = radial_profile(truth[h, 0], n_bins=64, logspace=False)
        rq = np.clip(R_OVER_R200 * r_pix[h], rfine[0], rfine[-1])
        for c in range(C.N_MASS_CH):
            _, tf = radial_profile(truth[h, c], n_bins=64, logspace=False)
            _, gf = radial_profile(gen[h, c], n_bins=64, logspace=False)
            t_all[h, c] = np.interp(rq, rfine, tf)
            g_all[h, c] = np.interp(rq, rfine, gf)
    out = dict(params=C.sim_params(cat))
    empty = np.full((C.N_MASS_CH, len(R_OVER_R200)), np.nan)
    for w, win in C.PARAM_WINDOWS.items():  # per-sim mean profile per mass window
        sel = _window_mask(masses, win)
        out[f"truth_mean_{w}"] = np.nanmean(t_all[sel], 0) if sel.any() else empty
        out[f"gen_mean_{w}"] = np.nanmean(g_all[sel], 0) if sel.any() else empty
        out[f"n_{w}"] = np.array(int(sel.sum()))
    return out


def _spearman_grid(P, arr):
    """arr (S, nch, nk) → per-channel (nk, N_PARAMS) rho & p."""
    from scipy.stats import spearmanr
    nch, nk = arr.shape[1], arr.shape[2]
    rho = [np.full((nk, C.N_PARAMS), np.nan) for _ in range(nch)]
    pval = [np.full((nk, C.N_PARAMS), np.nan) for _ in range(nch)]
    for c in range(nch):
        for k in range(nk):
            y = arr[:, c, k]
            for j in range(C.N_PARAMS):
                x = P[:, j]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() >= 5 and np.ptp(x[m]) > 0:
                    r, p = spearmanr(x[m], y[m])
                    rho[c][k, j] = r
                    pval[c][k, j] = p
    return rho, pval


def reduce_profiles_r200(items):
    P = np.array([d["params"] for _, d in items])
    out = dict(r_over_r200=R_OVER_R200, params=P,
               suites=[r["suite"] for r, _ in items], sims=[r["sim_id"] for r, _ in items],
               rho_prof={}, mean_prof={})
    for w in C.PARAM_WINDOWS:
        T = np.array([d[f"truth_mean_{w}"] for _, d in items])
        G = np.array([d[f"gen_mean_{w}"] for _, d in items])
        rho_t, _ = _spearman_grid(P, T)
        rho_g, _ = _spearman_grid(P, G)
        out["rho_prof"][w] = {"True": rho_t, "BIND": rho_g}
        out["mean_prof"][w] = {"truth": T, "gen": G}
    path = C.CACHE_DIR / "profiles_r200.pkl"
    pickle.dump(out, open(path, "wb"))
    print(f"  wrote {path}  ({len(P)} sims, windows {list(C.PARAM_WINDOWS)})")


# ════════════════════════════════════════════════════════════════════════════
# pk — full-box total-matter P(k): DMO / truth / BIND / hydro-replace
# ════════════════════════════════════════════════════════════════════════════
def _hydro_replace_map(rec, fm):
    """Truth-hydro pasted composite (model-error control). Uses hydro_replace.npy
    if present, else rebuilds it via build_bind_composite with truth patches."""
    hr = Path(rec["composite"]).parent / "hydro_replace.npy"
    if hr.exists():
        return np.load(hr)
    if not rec["cutouts"].exists():
        return None
    from bind.inference.artifacts import load_halo_catalog, load_halo_cutouts
    from bind.inference.pipeline import build_bind_composite
    halos, _, _, _ = load_halo_catalog(rec["catalog"])
    cutouts = load_halo_cutouts(rec["cutouts"])
    centers_pix = C.centers_to_pixels(np.array([h["halo_center"] for h in halos]))
    truth_patches = C.extract_truth_mass_patches(fm, centers_pix)
    cfg = _run_config(rec)
    b = build_bind_composite(fm["dmo_fullbox"], halos, truth_patches, cutouts,
                             C.BOX_SIZE, C.N_PIX_FULL, C.PATCH_PIX,
                             patch_mass_match=cfg["patch_mass_match"],
                             taper_frac=cfg["taper_frac"], r200_factor=cfg["r200_factor"])
    return b["composite"]


def compute_pk(rec, threads=2):
    fm = C.load_full_maps(rec)
    comp = C.load_composite(rec)
    k, dmo, _ = power_spectrum_pylians_2d(fm["dmo_fullbox"], box_size=C.BOX_SIZE, MAS="None", threads=threads)
    _, truth, _ = power_spectrum_pylians_2d(fm["truth_maps"].sum(0), box_size=C.BOX_SIZE, MAS="None", threads=threads)
    _, bind, _ = power_spectrum_pylians_2d(comp["composite"].sum(0), box_size=C.BOX_SIZE, MAS="None", threads=threads)
    hr_map = _hydro_replace_map(rec, fm)
    if hr_map is not None:
        _, hydro_replace, _ = power_spectrum_pylians_2d(np.asarray(hr_map).sum(0), box_size=C.BOX_SIZE, MAS="None", threads=threads)
    else:
        hydro_replace = np.full_like(dmo, np.nan)
    return dict(k=k, dmo=dmo, truth=truth, bind=bind, hydro_replace=hydro_replace)


def reduce_pk(items):
    out = {"k": None}
    per_suite = defaultdict(lambda: defaultdict(list))
    for rec, d in items:
        if out["k"] is None:
            out["k"] = d["k"]
        for key in ("dmo", "truth", "bind", "hydro_replace"):
            per_suite[rec["suite"]][key].append(d[key])
    for suite, dd in per_suite.items():
        for key, lst in dd.items():
            out[f"{suite}_{key}"] = np.stack(lst)
    out["has_hydro_replace"] = bool(np.isfinite(
        np.concatenate([out[f"{s}_hydro_replace"] for s in per_suite]).ravel()).any())
    path = C.CACHE_DIR / "pk.npz"
    np.savez_compressed(path, **{k: v for k, v in out.items() if v is not None})
    print(f"  wrote {path}  (suites {list(per_suite)}, hydro_replace={out['has_hydro_replace']})")


# ════════════════════════════════════════════════════════════════════════════
# shapes — mass-weighted 2D quadrupole moments (R200 aperture) + param response
# ════════════════════════════════════════════════════════════════════════════
def shape_moments_batch(maps, threshold=0.0, aperture_radii=None, min_pixels=5):
    """Mass-weighted 2D quadrupole q, e1, e2 per map (NaN if invalid)."""
    N, H, W = maps.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dist2 = (xx - (W - 1) / 2.0) ** 2 + (yy - (H - 1) / 2.0) ** 2
    per_halo = (aperture_radii is not None) and not np.isscalar(aperture_radii)
    q_out = np.full(N, np.nan)
    e1_out = np.full(N, np.nan)
    e2_out = np.full(N, np.nan)
    for i in range(N):
        if aperture_radii is None:
            aperture = np.ones((H, W))
        else:
            r = float(aperture_radii[i]) if per_halo else float(aperture_radii)
            aperture = (dist2 <= r ** 2).astype(np.float64)
        w = np.maximum(maps[i].astype(np.float64) - threshold, 0.0) * aperture
        total = w.sum()
        if total < 1e-30 or int((w > 0).sum()) < min_pixels:
            continue
        x_c = (xx * w).sum() / total
        y_c = (yy * w).sum() / total
        dx, dy = xx - x_c, yy - y_c
        Qxx = (dx ** 2 * w).sum() / total
        Qyy = (dy ** 2 * w).sum() / total
        Qxy = (dx * dy * w).sum() / total
        evals, evecs = np.linalg.eigh([[Qxx, Qxy], [Qxy, Qyy]])
        lam_min, lam_max = evals[0], evals[1]
        if lam_max < 1e-30 or lam_min < 0:
            continue
        q = np.sqrt(lam_min / lam_max)
        pa = np.arctan2(evecs[1, 1], evecs[0, 1])
        eps = (1 - q) / (1 + q)
        q_out[i] = q
        e1_out[i] = eps * np.cos(2 * pa)
        e2_out[i] = eps * np.sin(2 * pa)
    return q_out, e1_out, e2_out


_SHAPE_CH = [(0, "dm", 0.0), (1, "gas", 0.0), (2, "star", C.STAR_THRESH)]


def compute_shapes(rec):
    fm, truth, gen7, cat, centers_pix = _load_mass_patches(rec)
    gen = gen7[:, : C.N_MASS_CH]
    dmo_patches = np.stack([C.extract_patch(fm["dmo_fullbox"], cx, cy) for cx, cy in centers_pix])
    masses = np.asarray(cat["masses"], np.float64)
    r200 = C.r200_pix_patch(cat)
    out = dict(masses=masses, params=np.tile(C.sim_params(cat), (len(masses), 1)))
    for src, maps in [("truth", truth), ("gen", gen)]:
        for ci, key, thr in _SHAPE_CH:
            q, e1, e2 = shape_moments_batch(maps[:, ci], threshold=thr, aperture_radii=r200)
            out[f"{src}_{key}_q"] = q
            out[f"{src}_{key}_e1"] = e1
            out[f"{src}_{key}_e2"] = e2
    q, e1, e2 = shape_moments_batch(dmo_patches, threshold=0.0, aperture_radii=r200)
    out["dmo_q"], out["dmo_e1"], out["dmo_e2"] = q, e1, e2
    return out


def reduce_shapes(items):
    keys = [k for k in items[0][1] if k != "params"]
    cat = {k: [] for k in keys}
    cat["suite"] = []
    cat["params"] = []
    metrics = [f"{s}_{key}_{m}" for s in ("truth", "gen") for _, key, _ in _SHAPE_CH for m in ("q", "eps")]
    win_vals = {w: {mn: [] for mn in metrics} for w in C.PARAM_WINDOWS}
    Plist, srows = [], []
    for rec, d in items:
        n = len(d["masses"])
        masses = np.asarray(d["masses"], np.float64)
        for k in keys:
            cat[k].append(d[k])
        cat["suite"].append(np.full(n, rec["suite"]))
        cat["params"].append(d["params"])
        Plist.append(d["params"][0])
        row = {"suite": rec["suite"]}
        row.update({f"p{j+1}": d["params"][0, j] for j in range(C.N_PARAMS)})
        for w, win in C.PARAM_WINDOWS.items():
            sel = _window_mask(masses, win)
            for src in ("truth", "gen"):
                for _, key, _ in _SHAPE_CH:
                    q = d[f"{src}_{key}_q"]
                    eps = np.hypot(d[f"{src}_{key}_e1"], d[f"{src}_{key}_e2"])
                    mq = np.nanmedian(q[sel]) if sel.any() else np.nan
                    me = np.nanmedian(eps[sel]) if sel.any() else np.nan
                    win_vals[w][f"{src}_{key}_q"].append(mq)
                    win_vals[w][f"{src}_{key}_eps"].append(me)
                    if w == "trained":
                        row[f"{src}_{key}_q"], row[f"{src}_{key}_eps"] = mq, me
        srows.append(row)
    arrs = {k: (np.concatenate(v) if k != "params" else np.concatenate(v, axis=0)) for k, v in cat.items()}
    sim_df = pd.DataFrame(srows)
    Pv = np.array(Plist)
    rho_shape = {w: np.array([_spearman_vec(Pv, win_vals[w][mn]) for mn in metrics]) for w in C.PARAM_WINDOWS}
    out = dict(arrays=arrs, sim_table=sim_df, response_metrics=metrics, rho_shape=rho_shape)
    pickle.dump(out, open(C.CACHE_DIR / "shapes.pkl", "wb"))
    print(f"  wrote shapes.pkl  ({len(arrs['masses'])} halos; windows {list(C.PARAM_WINDOWS)})")


# ════════════════════════════════════════════════════════════════════════════
# thermo — Y/T/K/P scaling, thermo profiles, pixel PDF, param response
# ════════════════════════════════════════════════════════════════════════════
def _aperture_thermo(thermo4, gas2d, mask):
    """(Y, Tx, K, P, Mgas) within an aperture. y extensive; T/K/P gas-weighted."""
    y, T, S, Pe = thermo4
    gw = np.maximum(gas2d[mask], 0.0)
    gsum = gw.sum()
    Y = np.maximum(y, 0.0)[mask].sum()
    Mgas = gsum
    if gsum <= 0:
        return np.array([Y, np.nan, np.nan, np.nan, Mgas])
    Tx = (T[mask] * gw).sum() / gsum
    K = (S[mask] * gw).sum() / gsum
    P = (Pe[mask] * gw).sum() / gsum
    return np.array([Y, Tx, K, P, Mgas])


def compute_thermo(rec, rng_seed=0):
    tth = C.load_truth_thermo(rec)
    if tth is None:
        return None  # no thermo truth for this sim
    fm, truth, gen7, cat, _ = _load_mass_patches(rec)
    gen_mass, gen_th = gen7[:, : C.N_MASS_CH], gen7[:, C.N_MASS_CH:]
    truth_gas = truth[:, 1]
    gen_gas = gen_mass[:, 1]
    masses = np.asarray(cat["masses"], np.float64)
    logm = np.log10(masses)
    n = len(masses)
    masks, _ = _aperture_masks(cat)

    t_int = np.full((n, len(THERMO_INT_KEYS)), np.nan)
    g_int = np.full((n, len(THERMO_INT_KEYS)), np.nan)
    tprof = np.zeros((n, C.N_THERMO_CH, THERMO_PROF_NBINS))
    gprof = np.zeros((n, C.N_THERMO_CH, THERMO_PROF_NBINS))
    r = None
    for i in range(n):
        t_int[i] = _aperture_thermo(tth[i], truth_gas[i], masks[i])
        g_int[i] = _aperture_thermo(gen_th[i], gen_gas[i], masks[i])
        for c in range(C.N_THERMO_CH):
            rr, tp = radial_profile(tth[i, c], n_bins=THERMO_PROF_NBINS, logspace=False)
            _, gp = radial_profile(gen_th[i, c], n_bins=THERMO_PROF_NBINS, logspace=False)
            tprof[i, c] = tp
            gprof[i, c] = gp
            if r is None:
                r = rr

    # pixel PDF sample (jointly-positive pixels, capped)
    rng = np.random.default_rng(rng_seed)
    hsel = rng.choice(n, size=min(_THERMO_PIX_HALOS, n), replace=False)
    tpix = [[] for _ in range(C.N_THERMO_CH)]
    gpix = [[] for _ in range(C.N_THERMO_CH)]
    for i in hsel:
        for c in range(C.N_THERMO_CH):
            tv = tth[i, c].ravel()
            gv = gen_th[i, c].ravel()
            pos = (tv > 0) & (gv > 0)
            idx = np.where(pos)[0]
            if len(idx) > _THERMO_PIX_PER_HALO:
                idx = rng.choice(idx, _THERMO_PIX_PER_HALO, replace=False)
            tpix[c].append(np.log10(tv[idx]))
            gpix[c].append(np.log10(gv[idx]))
    out = dict(logm=logm, mass_bin=C.mass_bin_index(logm),
               t_int=t_int, g_int=g_int,
               r=r * C.MPC_PER_PIX_PATCH, truth=tprof, gen=gprof,
               params=C.sim_params(cat))
    # per-channel pixel samples as separate float keys (no ragged object arrays)
    for c in range(C.N_THERMO_CH):
        out[f"tpix{c}"] = (np.concatenate(tpix[c]) if tpix[c] else np.zeros(0)).astype(np.float32)
        out[f"gpix{c}"] = (np.concatenate(gpix[c]) if gpix[c] else np.zeros(0)).astype(np.float32)
    return out


def reduce_thermo(items):
    # 1) scaling table (per halo)
    rows = []
    for rec, d in items:
        for i in range(len(d["logm"])):
            row = dict(suite=rec["suite"], sim_id=rec["sim_id"],
                       logM=float(d["logm"][i]), mass_bin=int(d["mass_bin"][i]))
            for k, key in enumerate(THERMO_INT_KEYS):
                row[f"truth_{key}"] = d["t_int"][i, k]
                row[f"gen_{key}"] = d["g_int"][i, k]
            rows.append(row)
    scaling = pd.DataFrame(rows)
    scaling.to_pickle(C.CACHE_DIR / "thermo_scaling.pkl")

    # 2) thermo profiles → bands by mass bin
    _reduce_profile_like(items, "r", "truth", "gen", "mass_bin", "thermo_profiles.pkl")

    # 3) pixel PDF (pooled, capped per channel)
    cap = 300_000
    rng = np.random.default_rng(0)
    tp = [[] for _ in range(C.N_THERMO_CH)]
    gp = [[] for _ in range(C.N_THERMO_CH)]
    for _, d in items:
        for c in range(C.N_THERMO_CH):
            tp[c].append(np.asarray(d[f"tpix{c}"], dtype=np.float64))
            gp[c].append(np.asarray(d[f"gpix{c}"], dtype=np.float64))
    pix = {}
    for c, name in enumerate(C.THERMO_CHANNELS):
        t = np.concatenate(tp[c]).astype(np.float64) if tp[c] else np.zeros(0)
        g = np.concatenate(gp[c]).astype(np.float64) if gp[c] else np.zeros(0)
        if len(t) > cap:
            sel = rng.choice(len(t), cap, replace=False)
            t, g = t[sel], g[sel]
        pix[f"{name}_truth"] = t
        pix[f"{name}_gen"] = g
    np.savez_compressed(C.CACHE_DIR / "thermo_pixels.npz", **pix)

    # 4) parameter response per mass window: per-sim mean log10 aperture-quantity vs params
    nk = len(THERMO_INT_KEYS)
    Plist, srows = [], []
    win_vals = {w: {"true": [[] for _ in range(nk)], "bind": [[] for _ in range(nk)]}
                for w in C.PARAM_WINDOWS}
    for rec, d in items:
        Plist.append(d["params"])
        logm = np.asarray(d["logm"], float)
        row = {"suite": rec["suite"]}
        row.update({f"p{j+1}": d["params"][j] for j in range(C.N_PARAMS)})
        for w, (lo, hi) in C.PARAM_WINDOWS.items():
            sel = (logm >= np.log10(lo)) & (logm < np.log10(hi))
            for ki, key in enumerate(THERMO_INT_KEYS):
                with np.errstate(divide="ignore", invalid="ignore"):
                    tv = np.nanmean(np.log10(np.where(d["t_int"][sel, ki] > 0, d["t_int"][sel, ki], np.nan))) if sel.any() else np.nan
                    gv = np.nanmean(np.log10(np.where(d["g_int"][sel, ki] > 0, d["g_int"][sel, ki], np.nan))) if sel.any() else np.nan
                win_vals[w]["true"][ki].append(tv)
                win_vals[w]["bind"][ki].append(gv)
                if w == "trained":
                    row[f"true_{key}"], row[f"bind_{key}"] = tv, gv
        srows.append(row)
    sdf = pd.DataFrame(srows)
    P = np.array(Plist)
    rho_thermo = {w: {"True": np.array([_spearman_vec(P, win_vals[w]["true"][ki]) for ki in range(nk)]),
                      "BIND": np.array([_spearman_vec(P, win_vals[w]["bind"][ki]) for ki in range(nk)])}
                  for w in C.PARAM_WINDOWS}
    pickle.dump(dict(sim_table=sdf, keys=THERMO_INT_KEYS, rho_thermo=rho_thermo),
                open(C.CACHE_DIR / "thermo_param.pkl", "wb"))
    print(f"  wrote thermo_scaling/profiles/pixels/param  ({len(scaling)} halos, {len(sdf)} sims)")


# ════════════════════════════════════════════════════════════════════════════
# field1p — 1P hi/lo most-massive-halo patches (mass+thermo) for butterfly figs
# ════════════════════════════════════════════════════════════════════════════
# Astrophysical 1P params the butterfly figure features (A_SN1, A_AGN1, A_ASN2,
# IMFslope, SNII_MinMass, BlackHoleFeedbackFactor). Loading generated_halos.npz is
# ~175 MB/sim, so default to this subset; pass params="all" for every 1P pair.
HERO_1P_PARAMS = [3, 4, 5, 12, 13, 25]


def build_field1p(params=None):
    if params is None:
        param_list = HERO_1P_PARAMS
    elif params == "all":
        param_list = list(range(1, C.N_PARAMS + 1))
    else:
        param_list = [int(p) for p in params.split(",")]
    recs = {r["sim_id"]: r for r in C.discover_sims(("1P",))}
    saved = {}
    params_done = []
    for j in param_list:
        hi = recs.get(f"1P_p{j}_2")
        lo = recs.get(f"1P_p{j}_n2")
        if hi is None or lo is None:
            continue
        ok = True
        pair = {}
        for tag, rec in [("hi", hi), ("lo", lo)]:
            fm = C.load_full_maps(rec)
            cat = C.load_catalog(rec)
            gen = C.load_generated(rec)
            i = int(np.argmax(cat["masses"]))
            centers_pix = C.centers_to_pixels(cat["centers"])
            tmass = C.extract_truth_mass_patches(fm, centers_pix[i:i + 1])[0]  # (3,H,W)
            tth = C.load_truth_thermo(rec)
            g = gen[i]  # (7,H,W)
            t = np.concatenate([tmass, tth[i]], 0) if tth is not None else tmass
            pair[f"t_{tag}"] = t.astype(np.float32)
            pair[f"g_{tag}"] = g.astype(np.float32)
        for k, v in pair.items():
            saved[f"p{j}_{k}"] = v
        params_done.append(j)
    saved["params"] = np.array(params_done)
    path = C.CACHE_DIR / "field1p.npz"
    np.savez_compressed(path, **saved)
    print(f"  wrote {path}  (1P params {params_done})")


# ════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════
METRICS = {
    "mass": (compute_mass, reduce_mass, True),
    "profiles": (compute_profiles, reduce_profiles, True),
    "profiles_r200": (compute_profiles_r200, reduce_profiles_r200, True),
    "pk": (compute_pk, reduce_pk, True),
    "shapes": (compute_shapes, reduce_shapes, True),
    "thermo": (compute_thermo, reduce_thermo, True),
    "field1p": (None, None, False),  # special single-process
}


def _partial_path(metric, rec):
    return C.PARTIAL_DIR / metric / f"{rec['suite']}__{rec['sim_id']}.npz"


def _save_partial(path, d):
    path.parent.mkdir(parents=True, exist_ok=True)
    if d is None:
        np.savez(path, _skip=np.array(True))
    else:
        np.savez(path, _skip=np.array(False), **{k: v for k, v in d.items()})


def _load_partial(path):
    z = np.load(path, allow_pickle=True)
    if bool(z["_skip"]):
        return None
    return {k: z[k] for k in z.files if k != "_skip"}


def _select_records(args):
    if args.sim_ids:
        return [C.resolve_record(s.strip()) for s in args.sim_ids.split(",") if s.strip()]
    suites = C.SUITES if args.suite in (None, "all") else (args.suite,)
    recs = C.discover_sims(suites)
    if args.n_chunks > 1:
        size = (len(recs) + args.n_chunks - 1) // args.n_chunks
        recs = recs[args.chunk * size:(args.chunk + 1) * size]
    return recs


def _build_one(args_tuple):
    metric, key, force = args_tuple
    rec = C.resolve_record(key)
    path = _partial_path(metric, rec)
    if path.exists() and not force:
        return f"skip(exists) {key}"
    try:
        compute = METRICS[metric][0]
        d = compute(rec)
        _save_partial(path, d)
        return f"ok {key}" + ("" if d is not None else " [skip-marker]")
    except Exception as exc:
        return f"FAIL {key}: {exc}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, choices=list(METRICS))
    ap.add_argument("--suite", default="all", choices=["all", *C.SUITES])
    ap.add_argument("--sim_ids", default=None, help="Comma list e.g. CV/sim_0,CV/sim_1")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--pool", type=int, default=1, help="local multiprocessing workers")
    ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--params", default=None,
                    help="field1p only: comma list of 1P param indices, or 'all' (default: hero subset)")
    args = ap.parse_args()

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metric = args.metric
    _, reduce_fn, per_sim = METRICS[metric]

    if metric == "field1p":
        build_field1p(args.params)
        return

    if args.reduce:
        pdir = C.PARTIAL_DIR / metric
        items = []
        for p in sorted(pdir.glob("*.npz")):
            suite, sim = p.stem.split("__", 1)
            d = _load_partial(p)
            if d is not None:
                items.append((C.sim_record(C.SUITE_ROOT / suite / sim, suite), d))
        print(f"[reduce {metric}] {len(items)} usable partials")
        if items:
            reduce_fn(items)
        return

    recs = _select_records(args)
    print(f"[build {metric}] {len(recs)} sims  pool={args.pool}")
    tasks = [(metric, r["key"], args.force) for r in recs]
    if args.pool > 1:
        import multiprocessing as mp
        with mp.Pool(args.pool) as pool:
            for msg in pool.imap_unordered(_build_one, tasks):
                if msg.startswith("FAIL"):
                    print(msg)
                else:
                    print(" ", msg)
    else:
        for t in tasks:
            msg = _build_one(t)
            print(" ", msg if not msg.startswith("FAIL") else msg)


if __name__ == "__main__":
    main()
