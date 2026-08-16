#!/usr/bin/env python3
"""F7-fig2-baryon: rebuild two paper figures from the paper cache selected
via the PAPER_* environment variables (M200c >= 1e13).

  1. fig2_mass_comparison.{pdf,png}  — per-halo integrated-mass comparison
     (layout copied from examples/paper_figures2_lowmass.ipynb cell 11):
     2 rows (scatter+binned medians, marginal histograms) x 2 cols
     (r<=R200c aperture, full 6.25 Mpc/h patch), channels overplotted.
     Error bars = POPULATION scatter: np.std of log10(M_BIND) within each
     truth-mass bin (NOT the standard error on the median; both are printed).
  2. obs_baryon_fraction.{pdf,png}   — f_b(<R200c)/f_b0 vs M200c, one panel
     per suite (layout copied from scaling_relations.ipynb cell 8), with
     f_b0 = Omega_b/Omega_m per sim (params p7/p1, i.e. zero-based 6/0).
     Truth-halo cloud in the background; binned medians (16-84 bars) for
     truth (red circles) and BIND (blue squares). 1P/SB35 clouds colored by
     f_b0; CV cloud a single color (suite green — fixed cosmology).

CPU-only, load-only: reads mass_table.pkl + mass_param.pkl, no model code.
"""
import os
import sys
import pickle
from pathlib import Path

# Model selection comes from the environment (must be set before importing
# paper_config): PAPER_SUITE_ROOT, PAPER_MODEL_SUBDIR, PAPER_MASS_DIR,
# PAPER_MODEL_TAG. No hardcoded model defaults — fail fast if unset.
_REQUIRED_ENV = ("PAPER_SUITE_ROOT", "PAPER_MODEL_SUBDIR",
                 "PAPER_MASS_DIR", "PAPER_MODEL_TAG")
_missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
if _missing:
    sys.exit(f"fig2_baryon: missing required env vars: {', '.join(_missing)}")

sys.path.insert(0, "/mnt/home/mlee1/vdm_bind2/tools/paper_cache")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic

import paper_config as C

# ── style: identical to the paper notebooks (examples/_build_paper_nbs.py SETUP)
try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "notebook"])
except Exception:
    pass

SUITE_COLORS = C.SUITE_COLORS
SUITE_DISPLAY = C.SUITE_DISPLAY
SUITES = C.SUITES

FIG_DIR = Path("/mnt/home/mlee1/vdm_bind2/examples/paper_figures")
FIG_DIR.mkdir(exist_ok=True)


def save_fig(fig, name, ext=("pdf", "png")):
    for e in ext:
        fig.savefig(FIG_DIR / f"{name}.{e}", dpi=300, bbox_inches="tight")
    print("  saved", name)


# ── load spine cache ────────────────────────────────────────────────────────
CACHE = C.CACHE_DIR
print("Cache :", CACHE)
halo_tbl = pickle.load(open(CACHE / "mass_table.pkl", "rb"))
sim_tbl = pickle.load(open(CACHE / "mass_param.pkl", "rb"))["sim_table"]

# Standard cut: the spine mass_table is already >=1e13 — verify, don't re-cut.
assert halo_tbl["log_m200c"].min() >= 13.0 - 1e-9, "mass_table not cut at 1e13!"
assert halo_tbl["halo_mass"].min() >= 1e13 * (1 - 1e-6)
print(f"halos  : {len(halo_tbl)}  "
      f"{halo_tbl.groupby('suite').size().to_dict()}  "
      f"log M200c in [{halo_tbl.log_m200c.min():.3f}, {halo_tbl.log_m200c.max():.3f}]")

# Per-sim params: sim-level merge on (suite, sim_id) — one row per sim in
# mass_param, so no per-halo ordering is involved. Assert exact coverage.
par = sim_tbl[["suite", "sim_id", "p1", "p7"]].rename(
    columns={"p1": "omega_m", "p7": "omega_b"}
)
n0 = len(halo_tbl)
halo_tbl = halo_tbl.merge(par, on=["suite", "sim_id"], how="left", validate="m:1")
assert len(halo_tbl) == n0
assert halo_tbl["omega_m"].notna().all() and halo_tbl["omega_b"].notna().all()
# CV fiducial sanity (Omega_m=0.3, Omega_b=0.049)
cv = halo_tbl[halo_tbl.suite == "CV"]
assert np.allclose(cv.omega_m, 0.3, atol=1e-6) and np.allclose(cv.omega_b, 0.049)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 · fig2_mass_comparison  (layout copied from the notebook cell)
# ════════════════════════════════════════════════════════════════════════════
panel_fields = [
    [
        ("truth_DM_hydro_rvir", "gen_DM_hydro_rvir"),
        ("truth_Gas_rvir", "gen_Gas_rvir"),
        ("truth_Stars_rvir", "gen_Stars_rvir"),
    ],
    [
        ("truth_DM_hydro", "gen_DM_hydro"),
        ("truth_Gas", "gen_Gas"),
        ("truth_Stars", "gen_Stars"),
    ],
]
titles = [r"$r \leq R_{200c}$", "Full patch"]
colors = ["C0", "C1", "C2"]
labels = ["DM hydro", "Gas", "Stars"]

fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.3 * 2))
for ax in axes.flat[1:]:
    ax.sharex(axes[0, 0])
axes[0, 1].sharey(axes[0, 0])
axes[1, 1].sharey(axes[1, 0])

bar_stats = []  # (panel, channel, bin_center, N, median, sigma_pop, sem_med)

for col, (fields, title) in enumerate(zip(panel_fields, titles)):
    valid_mask = np.ones(len(halo_tbl), dtype=bool)
    for t, g in fields:
        valid_mask &= (halo_tbl[t] > 0) & (halo_tbl[g] > 0)
    tbl = halo_tbl[valid_mask]

    # ── row 0: scatter + binned errorbars ──────────────────────────────
    ax = axes[0, col]
    bins = np.linspace(
        min(np.log10(tbl[t]).min() for t, _ in fields),
        max(np.log10(tbl[t]).max() for t, _ in fields),
        20,
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for (truth_col, gen_col), color, label in zip(fields, colors, labels):
        x = np.log10(tbl[truth_col])
        y = np.log10(tbl[gen_col])

        ax.scatter(x, y, s=5, alpha=0.05, color=color, rasterized=True)

        bin_idx = np.digitize(x, bins) - 1
        bin_mean, bin_std, bin_cx = [], [], []
        for i in range(len(bins) - 1):
            m = bin_idx == i
            if m.sum() < 5:
                continue
            med = np.median(y[m])
            sig = np.std(y[m])
            bin_mean.append(med)
            bin_std.append(sig)  # BAR = population scatter of log10 M_BIND
            bin_cx.append(bin_centers[i])
            sem_med = 1.2533 * sig / np.sqrt(m.sum())  # SE on the median
            bar_stats.append((title, label, bin_centers[i], int(m.sum()),
                              med, sig, sem_med))

        ax.errorbar(bin_cx, bin_mean, yerr=bin_std,
                    fmt="o", ms=6, color=color, label=label,
                    capsize=3, lw=1.5, zorder=5, mec="k", ecolor="k")

    lo, hi = bins[0] - 0.25, bins[-1] + 0.25
    ax.plot(np.linspace(lo, hi, 100), np.linspace(lo, hi, 100), "k--")
    ax.set_title(title)
    if col == 0:
        ax.set_ylabel(r"$\log_{10}(M_{\rm BIND} / {\rm M}_\odot)$")
        ax.legend()
    else:
        plt.setp(ax.get_yticklabels(), visible=False)

    # ── row 1: distribution histograms ────────────────────────────────
    ax = axes[1, col]
    for (truth_col, gen_col), color, label in zip(fields, colors, labels):
        x_true = np.log10(tbl[truth_col])
        x_gen = np.log10(tbl[gen_col])
        hbins = np.linspace(x_true.min(), x_true.max(), 40)

        ax.hist(x_true, bins=hbins, color=color, alpha=0.4,
                density=True, histtype="stepfilled", label=label)
        ax.hist(x_gen, bins=hbins, color=color, alpha=0.9,
                density=True, histtype="step", linestyle="--", lw=1.5)

    ax.set_xlabel(r"$\log_{10}(M / {\rm M}_\odot)$")
    if col == 0:
        ax.set_ylabel("PDF")
        legend_handles = (
            [Patch(facecolor=c, alpha=0.6, label=l) for c, l in zip(colors, labels)]
            + [
                Patch(facecolor="gray", alpha=0.4, label="Truth"),
                Line2D([0], [0], color="gray", lw=1.5, linestyle="--", label="BIND"),
            ]
        )
        ax.legend(handles=legend_handles, ncols=2)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)

for ax in axes[0]:
    plt.setp(ax.get_xticklabels(), visible=False)

plt.tight_layout()
save_fig(fig, "fig2_mass_comparison")
plt.close(fig)

# ── fig2 bar-definition numbers ─────────────────────────────────────────────
bs = pd.DataFrame(bar_stats, columns=["panel", "channel", "bin_center", "N",
                                      "median", "sigma_pop", "sem_med"])
print("\n=== fig2 bars: sigma_pop = std(log10 M_BIND | truth bin)  vs  "
      "sem_med = 1.2533*sigma_pop/sqrt(N) ===")
for panel in bs.panel.unique():
    for ch in labels:
        sub = bs[(bs.panel == panel) & (bs.channel == ch)]
        print(f"{panel:14s} {ch:9s}  bins={len(sub):2d}  "
              f"sigma_pop med={sub.sigma_pop.median():.3f} "
              f"range=[{sub.sigma_pop.min():.3f},{sub.sigma_pop.max():.3f}] dex   "
              f"sem_med med={sub.sem_med.median():.4f} "
              f"range=[{sub.sem_med.min():.4f},{sub.sem_med.max():.4f}] dex")
# a concrete sample bin (Gas, R200c aperture, most-populated bin)
samp = bs[(bs.panel == titles[0]) & (bs.channel == "Gas")].nlargest(1, "N").iloc[0]
print(f"sample bin: Gas, r<=R200c, truth bin center {samp.bin_center:.2f} dex, "
      f"N={samp.N}: sigma_pop={samp.sigma_pop:.3f} dex, sem_med={samp.sem_med:.4f} dex")
bs.to_csv(FIG_DIR / "fig2_mass_comparison_barstats.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 · obs_baryon_fraction  (layout copied from scaling_relations cell 8)
# ════════════════════════════════════════════════════════════════════════════
def _binned_stats(x, y, bins, min_count=15):
    """Median + 16th/84th percentile errorbars in x-bins; drop sparse bins."""
    if len(x) == 0:
        return (np.array([]),) * 4
    bm = binned_statistic(x, y, statistic="median", bins=bins)
    blo = binned_statistic(x, y, statistic=lambda v: np.percentile(v, 16), bins=bins)
    bhi = binned_statistic(x, y, statistic=lambda v: np.percentile(v, 84), bins=bins)
    bn = binned_statistic(x, y, statistic="count", bins=bins)
    xc = 0.5 * (bm.bin_edges[:-1] + bm.bin_edges[1:])
    ok = np.isfinite(bm.statistic) & (bn.statistic >= min_count)
    med = bm.statistic[ok]
    return xc[ok], med, med - blo.statistic[ok], bhi.statistic[ok] - med


# f_b(<R200c) from the _rvir mass columns; f_b0 = Omega_b/Omega_m per sim
d = halo_tbl
fb0 = (d["omega_b"] / d["omega_m"]).to_numpy()
t_b = (d["truth_Gas_rvir"] + d["truth_Stars_rvir"]).to_numpy()
t_tot = t_b + d["truth_DM_hydro_rvir"].to_numpy()
g_b = (d["gen_Gas_rvir"] + d["gen_Stars_rvir"]).to_numpy()
g_tot = g_b + d["gen_DM_hydro_rvir"].to_numpy()
obs_df = pd.DataFrame({
    "suite": d["suite"].to_numpy(),
    "logM": d["log_m200c"].to_numpy(),
    "f_b_cos": fb0,
    "truth_f_b_norm": t_b / t_tot / fb0,
    "gen_f_b_norm": g_b / g_tot / fb0,
})

# Global colour normalisation for the f_b0-coded suites (1P + SB35 only;
# CV has fixed cosmology and is drawn in a single colour)
fb_cos_all = obs_df.loc[obs_df.suite != "CV", "f_b_cos"].to_numpy()
fb_cos_all = fb_cos_all[np.isfinite(fb_cos_all)]
vmin, vmax = np.nanpercentile(fb_cos_all, [2, 98])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"

TRUTH_MARKER = "o"  # truth medians: circles
GEN_MARKER = "s"    # gen medians:   squares

fig, axes = plt.subplots(3, 1, figsize=(5.5, 12), sharex=True,
                         constrained_layout=True)
fb_rows = []  # (suite, bin_center, N, med_t, med_g, off_pct, scat_t, scat_g)

for row, suite in enumerate(SUITES):
    ax = axes[row]
    sub = obs_df[obs_df.suite == suite]
    logm = sub["logM"].to_numpy()
    t = sub["truth_f_b_norm"].to_numpy()
    g = sub["gen_f_b_norm"].to_numpy()
    f_b_cos = sub["f_b_cos"].to_numpy()

    m = np.isfinite(logm) & np.isfinite(t) & np.isfinite(g) & np.isfinite(f_b_cos)

    # Halo cloud (truth): 1P/SB35 coloured by Omega_b/Omega_m, CV single colour
    if suite == "CV":
        ax.scatter(logm[m], t[m], color=SUITE_COLORS["CV"], s=2, alpha=0.10,
                   label="Truth halos" if row == 0 else None)
    else:
        ax.scatter(logm[m], t[m], c=f_b_cos[m], cmap=cmap, norm=norm,
                   s=2, alpha=0.10,
                   label="Truth halos" if row == 0 else None)

    if m.sum() >= 30:
        x_lo = np.nanpercentile(logm[m], 1)
        bins = np.linspace(x_lo, 15.0, 13)

        xt, tm, el_t, eh_t = _binned_stats(logm[m], t[m], bins)
        xg, gm, el_g, eh_g = _binned_stats(logm[m], g[m], bins)

        if len(xg):
            ax.errorbar(xg, gm, yerr=[el_g, eh_g],
                        fmt=GEN_MARKER, mfc="blue", mew=1.1, ms=8,
                        color=SUITE_COLORS[suite], linestyle="None",
                        capsize=3, elinewidth=0.8,
                        label="BIND" if row == 0 else None)
        if len(xt):
            ax.errorbar(xt, tm, yerr=[el_t, eh_t],
                        fmt=TRUTH_MARKER, color="red", ms=8, linestyle="None",
                        capsize=3, elinewidth=0.8,
                        label="Truth" if row == 0 else None)

        # stats (identical halo set + bins for truth and BIND)
        assert np.array_equal(xt, xg)
        for k in range(len(xt)):
            off = 100.0 * (gm[k] - tm[k]) / tm[k]
            scat_t = 0.5 * (el_t[k] + eh_t[k])
            scat_g = 0.5 * (el_g[k] + eh_g[k])
            nbin = int(((logm[m] >= bins[0]) & (logm[m] < 15.0)
                        & (np.digitize(logm[m], bins) - 1
                           == np.searchsorted(bins, xt[k]) - 1)).sum())
            fb_rows.append((suite, xt[k], nbin, tm[k], gm[k], off,
                            scat_t, scat_g))

    ax.axhline(1.0, color="0.35", lw=0.8, ls=":", alpha=0.8)
    ax.set_ylabel(SUITE_DISPLAY[suite] + "\n" + r"$f_b\,(\Omega_m/\Omega_b)$")
    ax.set_ylim(0.4, 1.1)

axes[-1].set_xlabel(r"$\log_{10} M_{200c}\;[{\rm M}_\odot/h]$")
axes[0].legend(loc="best")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.1, pad=0.02, aspect=150)
cbar.set_label(r"$\Omega_b/\Omega_m$")

save_fig(fig, "obs_baryon_fraction")
plt.close(fig)

# ── f_b agreement numbers ───────────────────────────────────────────────────
fb = pd.DataFrame(fb_rows, columns=["suite", "bin_center", "N", "med_truth",
                                    "med_bind", "offset_pct",
                                    "scat_truth", "scat_bind"])
fb["scat_ratio"] = fb["scat_bind"] / fb["scat_truth"]
print("\n=== obs_baryon_fraction: per-bin f_b(<R200c)/f_b0 medians ===")
print(fb.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
for suite in SUITES:
    s = fb[fb.suite == suite]
    imax = s["offset_pct"].abs().idxmax()
    print(f"{SUITE_DISPLAY[suite]:5s}: max |median offset| = "
          f"{s.loc[imax, 'offset_pct']:+.2f}% at logM={s.loc[imax, 'bin_center']:.2f} "
          f"(mean |off| {s.offset_pct.abs().mean():.2f}%)   "
          f"scatter ratio BIND/truth: median {s.scat_ratio.median():.3f}, "
          f"range [{s.scat_ratio.min():.3f}, {s.scat_ratio.max():.3f}]")
fb.to_csv(FIG_DIR / "obs_baryon_fraction_binstats.csv", index=False)

print("\ndone.")
