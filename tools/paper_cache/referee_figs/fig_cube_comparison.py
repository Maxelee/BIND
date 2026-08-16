#!/usr/bin/env python3
"""Projection-depth comparison appendix figure (fm_two_head vs fm_cube_two_head).

Compares the fiducial full-depth model (fm_two_head: DMO patches projected over
the full 50 Mpc/h line of sight + 3 large-scale context channels) against the
cube variant (fm_cube_two_head: 6.25 Mpc/h halo-centered cube projections, no
large-scale context, trained on CubeAstroDataset).  Each model is measured
against ITS OWN truth: the cube model against 6.25 Mpc/h-deep truth projections
(truth_halos_cube.npz), the full-depth model against 128x128 patches extracted
from the 50 Mpc/h-deep full-box truth maps (full_maps.npz), following the
compute_mass conventions in tools/paper_cache/build_metric.py.

Halo correspondence: the two evals share the same FoF catalogs, so halos are
matched per sim by index and the match is asserted on (mass, center).  The
Test (SB35) suites of the two evals sampled mostly different sims: only the
intersection is used for the matched comparison (asserted per sim); an
unmatched population cross-check over all Test sims of each eval is also
computed and printed.

Outputs
-------
- cache:  <CACHE>/cube_comparison_table.pkl   per-halo integrated masses
          <CACHE>/cube_comparison_profiles.npz per-halo 32-bin radial profiles
- figure: examples/paper_figures/fig_cube_comparison.{pdf,png}
          + a copy of the pdf into "BIND__methods_paper (1)/imgs/"

Run:
    source /mnt/home/mlee1/venvs/torch3/bin/activate
    python /mnt/home/mlee1/vdm_bind2/tools/paper_cache/referee_figs/fig_cube_comparison.py
    # force recompute of the per-halo cache:
    python .../fig_cube_comparison.py --recompute
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/home/mlee1/vdm_bind2/tools/paper_cache")
import paper_config as C  # geometry + suite styling (env defaults unused here)

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

SUITES = C.SUITES                  # ("CV", "1P", "Test")
SUITE_COLORS = C.SUITE_COLORS      # CV green, 1P blue, Test red
SUITE_DISPLAY = C.SUITE_DISPLAY    # Test -> SB35
CHANNELS = C.MASS_CHANNELS         # DM_hydro, Gas, Stars
MODELS = ("full", "cube")
MODEL_DISPLAY = {"full": "full depth (fiducial)", "cube": "cube (6.25 Mpc/h)"}

PROF_NBINS = 32                    # bind.metrics.radial_profile(n_bins=32, logspace=False)


# ── radial-profile machinery (exact vectorized bind.metrics.radial_profile) ─
def _annulus_matrix(n_pix=C.PATCH_PIX, n_bins=PROF_NBINS):
    """(M, counts, r_centers): M @ field.ravel() / counts == per-annulus mean,
    identical to bind.metrics.radial_profile(field, n_bins, logspace=False)."""
    y, x = np.mgrid[:n_pix, :n_pix] - np.array([n_pix / 2, n_pix / 2])[:, None, None]
    r = np.hypot(x, y)
    bins = np.linspace(0, n_pix / 2, n_bins + 1)
    masks = np.stack([(r >= bins[i]) & (r < bins[i + 1]) for i in range(n_bins)])
    counts = masks.sum((1, 2)).astype(np.float64)
    M = masks.reshape(n_bins, -1).astype(np.float64)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    return M, counts, r_centers


ANN_M, ANN_COUNTS, ANN_R_PIX = _annulus_matrix()
PROF_R_MPC = ANN_R_PIX * C.MPC_PER_PIX_PATCH


def profiles(patches):
    """(N, 3, 128, 128) -> (N, 3, PROF_NBINS) azimuthally averaged profiles."""
    n, nc = patches.shape[:2]
    flat = patches.reshape(n * nc, -1).astype(np.float64)
    prof = (flat @ ANN_M.T) / ANN_COUNTS
    return prof.reshape(n, nc, PROF_NBINS)


# ── discovery + matching ────────────────────────────────────────────────────
def _has_eval(md: Path, model: str, cube: bool) -> bool:
    ok = (md / "halo_catalog.npz").exists() and (md / model / "generated_halos.npz").exists()
    if cube:
        ok = ok and (md / "truth_halos_cube.npz").exists()
    else:
        ok = ok and (md.parent / "full_maps.npz").exists()
    return ok


def matched_sims(suite):
    """Sim names present (catalog + generated + truth) in BOTH evals."""
    cube_sims = {p.name for p in (CUBE_ROOT / suite).iterdir()
                 if p.is_dir() and _has_eval(p / MASS_SUB, CUBE_MODEL, cube=True)}
    full_sims = {p.name for p in (FULL_ROOT / suite).iterdir()
                 if p.is_dir() and _has_eval(p / MASS_SUB, FULL_MODEL, cube=False)}
    return sorted(cube_sims & full_sims), len(cube_sims), len(full_sims)


def r200_pix(cat_c, cat_f, masses):
    """R200c in patch pixels; prefer stored FoF radii, analytic fallback
    (paper_config.r200_pix_patch convention; full-depth catalogs store `radii`
    in kpc/h, cube catalogs `r200s` in Mpc/h; CV/sim_17 full lacks both)."""
    r_mpc = None
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
    """Integrated masses within R200c (clipped >= 0, as compute_mass) and over
    the full patch (unclipped), plus radial profiles.  truth/gen: (N,3,128,128)."""
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
    # per-halo correspondence: same FoF catalog, same order — assert it
    assert len(m_f) == n, f"{suite}/{sim}: halo count {len(m_f)} != {n}"
    assert np.allclose(m_c, m_f, rtol=1e-5), f"{suite}/{sim}: halo masses differ"
    assert np.allclose(cat_c["centers"], cat_f["centers"], atol=1e-3), \
        f"{suite}/{sim}: halo centers differ"
    r_pix = r200_pix(cat_c, cat_f, m_c)

    # cube model vs cube truth (6.25 Mpc/h deep)
    truth_c = np.load(md_c / "truth_halos_cube.npz")["truth_halos"]
    gen_c = np.load(md_c / CUBE_MODEL / "generated_halos.npz")["generated"][:, :3]
    assert len(truth_c) == n and len(gen_c) == n

    # full-depth model vs full-depth truth (50 Mpc/h deep)
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


def population_check_test():
    """Unmatched Test-population medians (each eval over ALL its Test sims):
    robustness check for the small matched-Test intersection."""
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
    print("\nUnmatched Test-population cross-check (R200c aperture, all sims of each eval):")
    for m in MODELS:
        meds = "  ".join(f"{ch} {100 * np.median(res[m][ch]):+.1f}% (n={len(res[m][ch])})"
                         for ch in CHANNELS)
        print(f"  {m:5s}: {meds}")


# ── residual statistics ─────────────────────────────────────────────────────
def resid_stats(df, model, ch, suite=None, aperture="_rvir"):
    """(median, p16, p84, n) of per-halo gen/truth - 1."""
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


def print_table(df):
    print(f"\nMatched halos: {len(df)}  per suite: "
          + ", ".join(f"{SUITE_DISPLAY[s]}={np.sum(df['suite'] == s)}" for s in SUITES))
    for ap, ap_name in [("_rvir", "r<=R200c"), ("", "full patch")]:
        print(f"\n=== {ap_name} — median [16,84] of gen/truth - 1 (%) ===")
        hdr = f"{'channel':9s} " + "".join(f"{SUITE_DISPLAY[s]:^32s}" for s in SUITES)
        print(f"{'':6s}{hdr}")
        for ch in CHANNELS + ["Total"]:
            for m in MODELS:
                cells = []
                for s in SUITES:
                    med, p16, p84, _ = resid_stats(df, m, ch, s, ap)
                    cells.append(f"{100 * med:+6.1f} [{100 * p16:+6.1f},{100 * p84:+6.1f}]")
                print(f"{m:6s}{ch:9s} " + " ".join(f"{c:^32s}" for c in cells))


def profile_summary(npz):
    """Median profile residual (g-t)/t per suite x channel x model, plus the
    radius range where the median |residual| < 10% (printed for the text)."""
    suite_arr = npz["suite"]
    out = {}
    for m in MODELS:
        t, g = npz[f"{m}_t"].astype(np.float64), npz[f"{m}_g"].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(t > 0, g / t - 1.0, np.nan)
        for s in SUITES:
            sel = suite_arr == s
            out[(m, s)] = (np.nanmedian(r[sel], 0),
                           np.nanpercentile(r[sel], 16, 0),
                           np.nanpercentile(r[sel], 84, 0))
        out[(m, "all")] = (np.nanmedian(r, 0),
                          np.nanpercentile(r, 16, 0),
                          np.nanpercentile(r, 84, 0))
    return out


# ── figure ──────────────────────────────────────────────────────────────────
def make_figure(df, npz):
    prof = profile_summary(npz)
    ch_disp = {"DM_hydro": "DM", "Gas": "Gas", "Stars": "Stars", "Total": "Total"}
    groups = CHANNELS + ["Total"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    # ── left: R200c-aperture median residuals, per component x suite x model ─
    ax = axes[0]
    n_s = len(SUITES)
    width = 0.34                       # half-offset between the two models
    for gi, ch in enumerate(groups):
        for si, s in enumerate(SUITES):
            x0 = gi * (n_s + 1) + si
            for m, dx, filled in [("full", -width / 2, True), ("cube", width / 2, False)]:
                med, p16, p84, _ = resid_stats(df, m, ch, s, "_rvir")
                kw = dict(color=SUITE_COLORS[s], zorder=5)
                ax.errorbar(x0 + dx, med, yerr=[[med - p16], [p84 - med]],
                            fmt="o" if filled else "s",
                            mfc=SUITE_COLORS[s] if filled else "white",
                            mec=SUITE_COLORS[s], ms=6.5, mew=1.4,
                            ecolor=SUITE_COLORS[s], elinewidth=1.4, capsize=2.5,
                            alpha=1.0 if filled else 0.9, **{k: v for k, v in kw.items() if k != "color"})
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    centers = [gi * (n_s + 1) + (n_s - 1) / 2 for gi in range(len(groups))]
    for gi in range(len(groups) - 1):
        ax.axvline(gi * (n_s + 1) + n_s, color="0.85", lw=0.8, zorder=0)
    ax.set_xticks(centers)
    ax.set_xticklabels([ch_disp[ch] for ch in groups])
    ax.set_ylabel(r"$M_{\rm gen}/M_{\rm truth} - 1$")
    ax.set_title(r"Integrated mass, $r \leq R_{200c}$")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # ── right: median radial-profile residual, per channel x model (all suites)
    # Stars omitted: stellar maps are sparse, so outer-annulus medians are
    # noise-dominated (quoted in the text for the core only).
    ax = axes[1]
    prof_channels = ["DM_hydro", "Gas"]
    ch_colors = {"DM_hydro": "0.15", "Gas": "tab:purple"}
    r = npz["r_mpc"]
    for ch in prof_channels:
        ch_i = CHANNELS.index(ch)
        for m, ls, lw in [("full", "-", 2.0), ("cube", "--", 2.0)]:
            med = prof[(m, "all")][0][ch_i]
            ax.plot(r, 100 * med, ls=ls, lw=lw, color=ch_colors[ch],
                    label=ch_disp[ch] if m == "full" else None)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"$r$ [Mpc$/h$]")
    ax.set_ylabel(r"median $\rho_{\rm gen}/\rho_{\rm truth} - 1$ [%]")
    ax.set_title("Radial-profile residual (all suites)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # legends
    from matplotlib.lines import Line2D
    suite_handles = [Line2D([], [], marker="o", ls="", color=SUITE_COLORS[s],
                            label=SUITE_DISPLAY[s]) for s in SUITES]
    model_handles = [
        Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="full depth"),
        Line2D([], [], marker="s", ls="", mfc="white", mec="k", label="cube"),
    ]
    axes[0].legend(handles=suite_handles + model_handles, ncol=2, fontsize=9,
                   loc="lower left", framealpha=0.9)
    model_handles2 = [Line2D([], [], ls="-", color="k", label="full depth"),
                      Line2D([], [], ls="--", color="k", label="cube")]
    ch_handles = [Line2D([], [], ls="-", color=ch_colors[ch], label=ch_disp[ch])
                  for ch in prof_channels]
    axes[1].legend(handles=ch_handles + model_handles2, ncol=2, fontsize=9,
                   loc="upper right", framealpha=0.9)

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
    ap.add_argument("--no-popcheck", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    tbl_path = CACHE / "cube_comparison_table.pkl"
    if args.recompute or not tbl_path.exists():
        df = build_cache()
    else:
        df = pd.read_pickle(tbl_path)
        print(f"loaded cache ({len(df)} halos) from {CACHE}")
    npz = np.load(CACHE / "cube_comparison_profiles.npz")

    print_table(df)

    # print profile summary numbers for the appendix text
    prof = profile_summary(npz)
    r = npz["r_mpc"]
    print("\nProfile median residual (all suites), selected radii:")
    for ch_i, ch in enumerate(CHANNELS):
        for m in MODELS:
            med = prof[(m, "all")][0][ch_i]
            picks = [np.argmin(np.abs(r - x)) for x in (0.2, 0.5, 1.0, 2.0, 3.0)]
            cells = "  ".join(f"r={r[j]:.2f}: {100 * med[j]:+.1f}%" for j in picks)
            print(f"  {m:5s} {ch:9s} {cells}")

    if not args.no_popcheck:
        population_check_test()
    if not args.no_fig:
        make_figure(df, npz)


if __name__ == "__main__":
    main()
