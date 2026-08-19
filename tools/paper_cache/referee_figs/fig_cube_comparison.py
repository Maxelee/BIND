#!/usr/bin/env python3
"""Projection-depth characterization appendix (truth-vs-truth LOS contamination).

Characterizes how much foreground/background material the fiducial full-depth
projection (50 Mpc/h line of sight, the fm_two_head training/eval convention)
incorporates into halo-aperture quantities, measured against halo-local
6.25 Mpc/h cube projections of the SAME halos as the reference.  The two
evaluation trees (/mnt/home/mlee1/ceph/fm_testsuite{,_cube}) share the same
FoF catalogs, so halos are matched one-to-one (asserted on mass + center) and
both truths are cached per halo:

  X       = M_truth_full / M_truth_cube - 1  within R200c   (LOS excess)
  f-bias  = f^proj / f^cube - 1  for f_b, f_gas, f_star     (fraction bias)
  Sigma ratio vs radius (32 linear annuli)                  (localization)

The model-vs-model accuracy comparison (fm_two_head vs fm_cube_two_head) is
retained as a printout only (`--model-table`) and feeds a single supporting
sentence in the appendix.

Outputs
-------
- cache:  <CACHE>/cube_comparison_table.pkl    per-halo aperture masses (both truths + both models)
          <CACHE>/cube_comparison_profiles.npz per-halo 32-bin radial profiles
- figure: examples/paper_figures/fig_cube_comparison.{pdf,png}
          + a copy of the pdf into "BIND__methods_paper (1)/imgs/"

Run:
    source /mnt/home/mlee1/venvs/torch3/bin/activate
    python /mnt/home/mlee1/vdm_bind2/tools/paper_cache/referee_figs/fig_cube_comparison.py
    # force recompute of the per-halo cache (~4 min, CPU):
    python .../fig_cube_comparison.py --recompute
    # also print the model-accuracy table + unmatched Test population check:
    python .../fig_cube_comparison.py --model-table --popcheck
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "paper_cache"))
import paper_config as C  # geometry + styling helpers (env defaults unused here)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "notebook"])
except Exception:
    pass

# ── Locations ───────────────────────────────────────────────────────────────
CUBE_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite_cube")
FULL_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite")
MASS_SUB = "snap_090/mass_threshold_1p000e13"
CUBE_MODEL = "fm_cube_two_head"
FULL_MODEL = "fm_two_head"

CACHE = Path("/mnt/home/mlee1/ceph/paper_cache/cube_comparison")
FIG_DIR = Path("/mnt/home/mlee1/vdm_bind2/examples/paper_figures")
IMGS_DIR = Path("/mnt/home/mlee1/vdm_bind2/BIND__methods_paper (1)/imgs")

SUITES = C.SUITES
SUITE_DISPLAY = C.SUITE_DISPLAY
CHANNELS = C.MASS_CHANNELS         # DM_hydro, Gas, Stars
MODELS = ("full", "cube")

PROF_NBINS = 32                    # bind.metrics.radial_profile(n_bins=32, logspace=False)

# Mass bins for the LOS-excess trend (log10 M200c); catalogs are >= 1e13
X_EDGES = np.array([13.0, 13.25, 13.5, 14.0, 15.0])
X_CENTERS = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])

# Channel styling (suite colors are not used here: series are channels)
CH_COLORS = {"DM_hydro": "0.15", "Gas": "tab:purple", "Stars": "tab:orange",
             "Total": "0.55"}
CH_DISP = {"DM_hydro": "DM", "Gas": "Gas", "Stars": "Stars", "Total": "Total"}
FRAC_COLORS = {"f_b": "#00796b", "f_gas": "tab:purple", "f_star": "tab:orange"}
FRAC_DISP = {"f_b": r"$f_{\rm b}$", "f_gas": r"$f_{\rm gas}$",
             "f_star": r"$f_\star$"}


# ── radial-profile machinery (exact vectorized bind.metrics.radial_profile) ─
def _annulus_matrix(n_pix=C.PATCH_PIX, n_bins=PROF_NBINS):
    y, x = np.mgrid[:n_pix, :n_pix] - np.array([n_pix / 2, n_pix / 2])[:, None, None]
    r = np.hypot(x, y)
    bins = np.linspace(0, n_pix / 2, n_bins + 1)
    masks = np.stack([(r >= bins[i]) & (r < bins[i + 1]) for i in range(n_bins)])
    counts = masks.sum((1, 2)).astype(np.float64)
    M = masks.reshape(n_bins, -1).astype(np.float64)
    return M, counts, 0.5 * (bins[:-1] + bins[1:])


ANN_M, ANN_COUNTS, ANN_R_PIX = _annulus_matrix()
PROF_R_MPC = ANN_R_PIX * C.MPC_PER_PIX_PATCH


def profiles(patches):
    """(N, 3, 128, 128) -> (N, 3, PROF_NBINS) azimuthally averaged profiles."""
    n, nc = patches.shape[:2]
    flat = patches.reshape(n * nc, -1).astype(np.float64)
    return ((flat @ ANN_M.T) / ANN_COUNTS).reshape(n, nc, PROF_NBINS)


# ── discovery + matching ────────────────────────────────────────────────────
def _has_eval(md: Path, model: str, cube: bool) -> bool:
    ok = (md / "halo_catalog.npz").exists() and (md / model / "generated_halos.npz").exists()
    if cube:
        ok = ok and (md / "truth_halos_cube.npz").exists()
    else:
        ok = ok and (md.parent / "full_maps.npz").exists()
    return ok


def matched_sims(suite):
    cube_sims = {p.name for p in (CUBE_ROOT / suite).iterdir()
                 if p.is_dir() and _has_eval(p / MASS_SUB, CUBE_MODEL, cube=True)}
    full_sims = {p.name for p in (FULL_ROOT / suite).iterdir()
                 if p.is_dir() and _has_eval(p / MASS_SUB, FULL_MODEL, cube=False)}
    return sorted(cube_sims & full_sims), len(cube_sims), len(full_sims)


def r200_pix(cat_c, cat_f, masses):
    """R200c in patch pixels; prefer stored FoF radii (cube `r200s` in Mpc/h,
    full-depth `radii` in kpc/h), analytic fallback (CV/sim_17 full lacks both)."""
    if "r200s" in cat_c:
        r_mpc = np.asarray(cat_c["r200s"], np.float64)
        if "radii" in cat_f:
            assert np.allclose(r_mpc, np.asarray(cat_f["radii"], np.float64) / 1e3,
                               rtol=1e-3), "cube r200s != full radii/1e3"
    elif "radii" in cat_f:
        r_mpc = np.asarray(cat_f["radii"], np.float64) / 1e3
    else:
        r_mpc = C.r200c_mpc(masses)
    return r_mpc / C.MPC_PER_PIX_PATCH


# ── per-sim measurement (compute_mass conventions from build_metric.py) ─────
def measure(truth, gen, r_pix):
    n = len(truth)
    t_full = truth.sum((2, 3)).astype(np.float64)
    g_full = gen.sum((2, 3)).astype(np.float64)
    t_rv = np.zeros((n, len(CHANNELS)))
    g_rv = np.zeros((n, len(CHANNELS)))
    for i in range(n):
        m = C.RR_PIX_PATCH < r_pix[i]
        for c in range(len(CHANNELS)):
            t_rv[i, c] = np.maximum(truth[i, c], 0.0)[m].sum()
            g_rv[i, c] = np.maximum(gen[i, c], 0.0)[m].sum()
    return dict(t_rv=t_rv, g_rv=g_rv, t_full=t_full, g_full=g_full,
                t_prof=profiles(truth), g_prof=profiles(gen))


def process_sim(suite, sim):
    md_c = CUBE_ROOT / suite / sim / MASS_SUB
    md_f = FULL_ROOT / suite / sim / MASS_SUB
    cat_c = np.load(md_c / "halo_catalog.npz")
    cat_f = np.load(md_f / "halo_catalog.npz")
    m_c = np.asarray(cat_c["masses"], np.float64)
    m_f = np.asarray(cat_f["masses"], np.float64)
    n = len(m_c)
    if n == 0:
        return None
    assert len(m_f) == n, f"{suite}/{sim}: halo count {len(m_f)} != {n}"
    assert np.allclose(m_c, m_f, rtol=1e-5), f"{suite}/{sim}: halo masses differ"
    assert np.allclose(cat_c["centers"], cat_f["centers"], atol=1e-3), \
        f"{suite}/{sim}: halo centers differ"
    r_pix = r200_pix(cat_c, cat_f, m_c)

    truth_c = np.load(md_c / "truth_halos_cube.npz")["truth_halos"]
    gen_c = np.load(md_c / CUBE_MODEL / "generated_halos.npz")["generated"][:, :3]
    assert len(truth_c) == n and len(gen_c) == n

    fmaps = np.load(md_f.parent / "full_maps.npz")
    centers_pix = C.centers_to_pixels(cat_f["centers"])
    truth_f = C.extract_truth_mass_patches({"truth_maps": fmaps["truth_maps"]}, centers_pix)
    gen_f = np.load(md_f / FULL_MODEL / "generated_halos.npz")["generated"][:, :3]
    assert len(gen_f) == n

    out = {"masses": m_c, "r_pix": r_pix,
           "cube": measure(truth_c, gen_c, r_pix),
           "full": measure(truth_f, gen_f, r_pix)}
    for f in (cat_c, cat_f, fmaps):
        f.close()
    return out


def build_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, prof_store = [], {m: {"t": [], "g": []} for m in MODELS}
    prof_meta = {"suite": [], "mass": []}
    for suite in SUITES:
        sims, n_cube, n_full = matched_sims(suite)
        print(f"[{suite}] cube={n_cube} full={n_full} matched={len(sims)}")
        for k, sim in enumerate(sims):
            d = process_sim(suite, sim)
            if d is None:
                print(f"  skip {suite}/{sim}: 0 halos")
                continue
            n = len(d["masses"])
            for i in range(n):
                row = dict(suite=suite, sim_id=sim, halo_mass=float(d["masses"][i]),
                           log_m200c=float(np.log10(d["masses"][i])),
                           r200_pix=float(d["r_pix"][i]))
                for m in MODELS:
                    for c, ch in enumerate(CHANNELS):
                        row[f"{m}_truth_{ch}_rvir"] = d[m]["t_rv"][i, c]
                        row[f"{m}_gen_{ch}_rvir"] = d[m]["g_rv"][i, c]
                        row[f"{m}_truth_{ch}"] = d[m]["t_full"][i, c]
                        row[f"{m}_gen_{ch}"] = d[m]["g_full"][i, c]
                rows.append(row)
            for m in MODELS:
                prof_store[m]["t"].append(d[m]["t_prof"])
                prof_store[m]["g"].append(d[m]["g_prof"])
            prof_meta["suite"] += [suite] * n
            prof_meta["mass"] += list(d["masses"])
            if (k + 1) % 20 == 0:
                print(f"  ... {k + 1}/{len(sims)} sims")
    df = pd.DataFrame(rows)
    df.to_pickle(CACHE / "cube_comparison_table.pkl")
    np.savez_compressed(
        CACHE / "cube_comparison_profiles.npz",
        r_mpc=PROF_R_MPC,
        suite=np.array(prof_meta["suite"]),
        mass=np.array(prof_meta["mass"]),
        **{f"{m}_{k}": np.concatenate(v[k]).astype(np.float32)
           for m in MODELS for k, v in [("t", prof_store[m]), ("g", prof_store[m])]},
    )
    print(f"wrote cache ({len(df)} matched halos) -> {CACHE}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# Truth-vs-truth characterization
# ════════════════════════════════════════════════════════════════════════════
def _pstats(x):
    """(median, p16, p84, n) over finite entries."""
    x = x[np.isfinite(x)]
    return np.median(x), np.percentile(x, 16), np.percentile(x, 84), len(x)


def truth_aperture_masses(df):
    """{(depth, ch): (N,) aperture mass} for depth in full/cube, ch incl. Total."""
    T = {}
    for m in MODELS:
        for ch in CHANNELS:
            T[(m, ch)] = df[f"{m}_truth_{ch}_rvir"].to_numpy(float)
        T[(m, "Total")] = sum(T[(m, ch)] for ch in CHANNELS)
    return T


def los_excess(df):
    """{ch: (N,) X = M_full/M_cube - 1 within R200c}."""
    T = truth_aperture_masses(df)
    X = {}
    for ch in CHANNELS + ["Total"]:
        with np.errstate(divide="ignore", invalid="ignore"):
            X[ch] = T[("full", ch)] / T[("cube", ch)] - 1.0
    return X


def fraction_biases(df):
    """{f: (N,) f^proj/f^cube - 1} for f_b, f_gas, f_star."""
    T = truth_aperture_masses(df)
    out = {}
    for name, num in [("f_b", ("Gas", "Stars")), ("f_gas", ("Gas",)),
                      ("f_star", ("Stars",))]:
        with np.errstate(divide="ignore", invalid="ignore"):
            f_full = sum(T[("full", c)] for c in num) / T[("full", "Total")]
            f_cube = sum(T[("cube", c)] for c in num) / T[("cube", "Total")]
            out[name] = f_full / f_cube - 1.0
    return out


def truth_profile_ratio(npz):
    """(ratio (N,3,nb), ratio_total (N,nb)): Sigma_full/Sigma_cube - 1."""
    tf = npz["full_t"].astype(np.float64)
    tc = npz["cube_t"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tc > 0, tf / tc - 1.0, np.nan)
        ratio_tot = np.where(tc.sum(1) > 0, tf.sum(1) / tc.sum(1) - 1.0, np.nan)
    return ratio, ratio_tot


def print_characterization(df, npz):
    logm = df["log_m200c"].to_numpy()
    X = los_excess(df)
    print(f"\nMatched halos: {len(df)}  per suite: "
          + ", ".join(f"{SUITE_DISPLAY[s]}={np.sum(df['suite'] == s)}" for s in SUITES))

    print("\n=== LOS excess X = M_truth^50/M_truth^6.25 - 1 within R200c (%) ===")
    for ch in CHANNELS + ["Total"]:
        med, p16, p84, n = _pstats(X[ch])
        xf = X[ch][np.isfinite(X[ch])]
        line = (f"{CH_DISP[ch]:6s} pooled {100*med:+6.1f} [{100*p16:+6.1f},{100*p84:+6.1f}]"
                f"  X>0.5: {100*np.mean(xf > 0.5):.2f}%  X>1: {100*np.mean(xf > 1.0):.2f}%")
        for s in SUITES:
            m2, q1, q3, _ = _pstats(X[ch][(df["suite"] == s).to_numpy()])
            line += f"  {SUITE_DISPLAY[s]} {100*m2:+.1f}"
        print(line)
    print("  by mass bin (median):")
    for ch in CHANNELS + ["Total"]:
        cells = []
        for i in range(len(X_EDGES) - 1):
            sel = (logm >= X_EDGES[i]) & (logm < X_EDGES[i + 1])
            med, p16, p84, n = _pstats(X[ch][sel])
            cells.append(f"[{X_EDGES[i]:.2f},{X_EDGES[i+1]:.2f}): {100*med:+5.1f} (n={n})")
        print(f"    {CH_DISP[ch]:6s} " + " | ".join(cells))

    print("\n=== Fraction biases f^proj/f^cube - 1 (%) ===")
    F = fraction_biases(df)
    for f in ("f_b", "f_gas", "f_star"):
        med, p16, p84, n = _pstats(F[f])
        cells = []
        for i in range(len(X_EDGES) - 1):
            sel = (logm >= X_EDGES[i]) & (logm < X_EDGES[i + 1])
            m2, q1, q3, _ = _pstats(F[f][sel])
            cells.append(f"{100*m2:+5.1f}")
        print(f"{f:7s} pooled {100*med:+6.2f} [{100*p16:+6.2f},{100*p84:+6.2f}]"
              f"   by mass: " + " | ".join(cells))

    ratio, ratio_tot = truth_profile_ratio(npz)
    r = npz["r_mpc"]
    print("\n=== Radial localization: median Sigma_full/Sigma_cube - 1 (%) ===")
    picks = [np.argmin(np.abs(r - x)) for x in (0.15, 0.25, 0.5, 1.0, 2.0, 3.0)]
    for ci, ch in enumerate(CHANNELS):
        med = np.nanmedian(ratio[:, ci], 0)
        print(f"{CH_DISP[ch]:6s} " + "  ".join(f"r={r[j]:.2f}: {100*med[j]:+7.1f}" for j in picks))
    med = np.nanmedian(ratio_tot, 0)
    print("Total  " + "  ".join(f"r={r[j]:.2f}: {100*med[j]:+7.1f}" for j in picks))


# ── model-accuracy printout (feeds one supporting sentence) ─────────────────
def resid_stats(df, model, ch, suite=None, aperture="_rvir"):
    sub = df if suite is None else df[df["suite"] == suite]
    if ch == "Total":
        t = sum(sub[f"{model}_truth_{c}{aperture}"] for c in CHANNELS).to_numpy()
        g = sum(sub[f"{model}_gen_{c}{aperture}"] for c in CHANNELS).to_numpy()
    else:
        t = sub[f"{model}_truth_{ch}{aperture}"].to_numpy()
        g = sub[f"{model}_gen_{ch}{aperture}"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = g / t - 1.0
    r = r[np.isfinite(r)]
    return np.median(r), np.percentile(r, 16), np.percentile(r, 84), len(r)


def print_model_table(df):
    print("\n=== model accuracy, each vs its own truth: median [16,84] % ===")
    for ap, ap_name in [("_rvir", "r<=R200c"), ("", "full patch")]:
        print(f"--- {ap_name}")
        for ch in CHANNELS + ["Total"]:
            for m in MODELS:
                cells = []
                for s in SUITES:
                    med, p16, p84, _ = resid_stats(df, m, ch, s, ap)
                    cells.append(f"{SUITE_DISPLAY[s]} {100*med:+6.1f} [{100*p16:+6.1f},{100*p84:+6.1f}]")
                print(f"  {m:5s}{ch:9s} " + "  ".join(cells))


def population_check_test():
    """Unmatched Test-population medians (each eval over ALL its Test sims)."""
    res = {m: {ch: [] for ch in CHANNELS} for m in MODELS}
    for m, root, model, cube in [("cube", CUBE_ROOT, CUBE_MODEL, True),
                                 ("full", FULL_ROOT, FULL_MODEL, False)]:
        for p in sorted((root / "Test").iterdir()):
            md = p / MASS_SUB
            if not (p.is_dir() and _has_eval(md, model, cube)):
                continue
            cat = np.load(md / "halo_catalog.npz")
            masses = np.asarray(cat["masses"], np.float64)
            if masses.size == 0:
                continue
            if cube:
                truth = np.load(md / "truth_halos_cube.npz")["truth_halos"]
            else:
                fmaps = np.load(md.parent / "full_maps.npz")
                truth = C.extract_truth_mass_patches(
                    {"truth_maps": fmaps["truth_maps"]}, C.centers_to_pixels(cat["centers"]))
            gen = np.load(md / model / "generated_halos.npz")["generated"][:, :3]
            if "r200s" in cat:
                r_pix = np.asarray(cat["r200s"], np.float64) / C.MPC_PER_PIX_PATCH
            elif "radii" in cat:
                r_pix = np.asarray(cat["radii"], np.float64) / 1e3 / C.MPC_PER_PIX_PATCH
            else:
                r_pix = C.r200c_mpc(masses) / C.MPC_PER_PIX_PATCH
            for i in range(len(masses)):
                msk = C.RR_PIX_PATCH < r_pix[i]
                for c, ch in enumerate(CHANNELS):
                    t = np.maximum(truth[i, c], 0.0)[msk].sum()
                    g = np.maximum(gen[i, c], 0.0)[msk].sum()
                    if t > 0:
                        res[m][ch].append(g / t - 1.0)
            cat.close()
    print("\nUnmatched Test-population check (R200c aperture, all sims of each eval):")
    for m in MODELS:
        meds = "  ".join(f"{ch} {100*np.median(res[m][ch]):+.1f}% (n={len(res[m][ch])})"
                         for ch in CHANNELS)
        print(f"  {m:5s}: {meds}")


# ════════════════════════════════════════════════════════════════════════════
# Figure: 3 panels — X vs logM; fraction biases vs logM; radial localization
# ════════════════════════════════════════════════════════════════════════════
def make_figure(df, npz):
    logm = df["log_m200c"].to_numpy()
    X = los_excess(df)
    F = fraction_biases(df)
    ratio, _ = truth_profile_ratio(npz)
    r = npz["r_mpc"]
    med_r200_mpc = float(np.median(df["r200_pix"])) * C.MPC_PER_PIX_PATCH

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    def binned(ax, series, colors, disp, offsets):
        for k, (key, y) in enumerate(series.items()):
            meds, lo, hi = [], [], []
            for i in range(len(X_EDGES) - 1):
                sel = (logm >= X_EDGES[i]) & (logm < X_EDGES[i + 1])
                m2, q1, q3, _ = _pstats(y[sel])
                meds.append(100 * m2), lo.append(100 * (m2 - q1)), hi.append(100 * (q3 - m2))
            xx = X_CENTERS + offsets[k]
            ax.errorbar(xx, meds, yerr=[lo, hi],
                        fmt="D--" if key == "Total" else "o-",
                        ms=4.5 if key == "Total" else 5.5, lw=1.6,
                        capsize=2.5, elinewidth=1.2, color=colors[key],
                        label=disp[key])

    # (a) LOS excess vs halo mass
    ax = axes[0]
    offs = np.linspace(-0.036, 0.036, 4)
    binned(ax, {ch: X[ch] for ch in ["DM_hydro", "Gas", "Stars", "Total"]},
           CH_COLORS, CH_DISP, offs)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"$\log_{10} M_{200c}\ [M_\odot/h]$")
    ax.set_ylabel(r"$M^{50}_{R_{200c}}/M^{6.25}_{R_{200c}} - 1$ [%]")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # (b) fraction biases vs halo mass
    ax = axes[1]
    offs = np.linspace(-0.03, 0.03, 3)
    binned(ax, F, FRAC_COLORS, FRAC_DISP, offs)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"$\log_{10} M_{200c}\ [M_\odot/h]$")
    ax.set_ylabel(r"$f^{\,50}_{R_{200c}}/f^{\,6.25}_{R_{200c}} - 1$ [%]")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # (c) radial localization (skip innermost annulus: sub-pixel registration)
    ax = axes[2]
    sl = slice(1, None)
    for ci, ch in enumerate(CHANNELS):
        med = np.nanmedian(ratio[:, ci], 0)
        lw, alpha = (1.4, 0.85) if ch == "Stars" else (2.0, 1.0)
        ax.plot(r[sl], 100.0 * med[sl], "-", lw=lw, alpha=alpha,
                color=CH_COLORS[ch], label=CH_DISP[ch])
    p16 = np.nanpercentile(ratio[:, 1], 16, 0)
    p84 = np.nanpercentile(ratio[:, 1], 84, 0)
    ax.fill_between(r[sl], 100.0 * p16[sl], 100.0 * p84[sl], color="tab:purple",
                    alpha=0.18, lw=0)
    ax.axhline(10.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(med_r200_mpc, color="0.4", lw=1.0, ls=":")
    ax.text(med_r200_mpc * 1.1, 130.0, r"median $R_{200c}$", rotation=90,
            fontsize=8, color="0.35", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r$ [Mpc$/h$]")
    ax.set_ylabel(r"$\Sigma^{50}/\Sigma^{6.25} - 1$ [%]")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig_cube_comparison.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    if IMGS_DIR.exists():
        shutil.copy(FIG_DIR / "fig_cube_comparison.pdf", IMGS_DIR / "fig_cube_comparison.pdf")
        print(f"copied pdf -> {IMGS_DIR}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--model-table", action="store_true",
                    help="also print the model-accuracy comparison")
    ap.add_argument("--popcheck", action="store_true",
                    help="also print the unmatched Test population check")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    tbl_path = CACHE / "cube_comparison_table.pkl"
    if args.recompute or not tbl_path.exists():
        df = build_cache()
    else:
        df = pd.read_pickle(tbl_path)
        print(f"loaded cache ({len(df)} halos) from {CACHE}")
    npz = np.load(CACHE / "cube_comparison_profiles.npz")

    print_characterization(df, npz)
    if args.model_table:
        print_model_table(df)
    if args.popcheck:
        population_check_test()
    if not args.no_fig:
        make_figure(df, npz)


if __name__ == "__main__":
    main()
