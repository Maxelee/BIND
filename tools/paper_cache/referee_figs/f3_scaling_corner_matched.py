#!/usr/bin/env python3
"""F3-scaling-corner-matched: three publication figures from the fiducial-model
cache (env-selected via PAPER_MODEL_TAG etc.), CV suite, M200c >= 1e13.

  1. scaling_fits            2x(Truth/BIND) x 4 relations, residual-colored
  2. residual_corner_plot    COMMON-X corner: all 4 residuals about M200c
  3. fig_matched_residuals   2x3 per-halo Delta_true vs Delta_BIND (aperture-clean)

Plotting style + the aperture-contamination recipe are copied from
examples/_build_paper_nbs.py (figT6 cells); the matched-residual panel layout
follows the referee prototype referee_response/matched/mkfig.py.

CPU-only; reads only cached pickles + halo catalogs (no model, no GPU).
"""
import os
import sys

# Fiducial-model env (must be set before importing paper_config; export the
# PAPER_* vars to point at any other model tag — these are fallback defaults).
os.environ.setdefault("PAPER_SUITE_ROOT", "/mnt/home/mlee1/ceph/fm_testsuite")
os.environ.setdefault("PAPER_MODEL_SUBDIR", "fm_two_head")
os.environ.setdefault("PAPER_MASS_DIR", "mass_threshold_1p000e13")
os.environ.setdefault("PAPER_MODEL_TAG", "fm_two_head")

sys.path.insert(0, "/mnt/home/mlee1/vdm_bind2/tools/paper_cache")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

import paper_config as C

# Style: identical to the notebook-builder SETUP
try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "notebook"])
except Exception:
    pass

FIG_DIR = Path("/mnt/home/mlee1/vdm_bind2/examples/paper_figures")
FIG_DIR.mkdir(exist_ok=True)
# DMO(<R200c) aperture sums (model-independent; recipe: referee_response/matched/
# p3_dmo.py — sum the DMO fullbox patch inside the catalog R200c aperture).
# The join below asserts consistency with the current suite's catalogs.
DMO_SUMS = os.environ.get(
    "PAPER_DMO_SUMS",
    "/mnt/home/mlee1/ceph/paper_cache/referee_response/matched/dmo_sums.pkl")
rng = np.random.default_rng(11)


def save_fig(fig, name, ext=("pdf", "png")):
    for e in ext:
        fig.savefig(FIG_DIR / f"{name}.{e}", dpi=300, bbox_inches="tight")
    print("  saved", name)


# ════════════════════════════════════════════════════════════════════════════
# Data: spine mass_table (CV, already >=1e13) + DMO aperture sums
# ════════════════════════════════════════════════════════════════════════════
mt = pd.read_pickle(C.CACHE_DIR / "mass_table.pkl")
tab = mt[mt.suite == "CV"].reset_index(drop=True)
assert (tab.log_m200c >= 13.0).all(), "spine mass_table should already be cut"
assert len(tab) == 1154, f"expected 1154 CV halos, got {len(tab)}"
tab = tab.assign(row=tab.groupby(["suite", "sim_id"]).cumcount())

# DMO(<R200c) aperture sums: join on (suite, sim_id, per-sim cumcount) after
# applying the same >=1e13 cut, then assert on the shared mass column.
ds = pd.read_pickle(DMO_SUMS)
ds = ds[(ds.suite == "CV") & (ds.M200 >= 1e13)].reset_index(drop=True)
ds = ds.assign(row=ds.groupby(["suite", "sim_id"]).cumcount())
tab = tab.merge(ds[["suite", "sim_id", "row", "M200", "dmo_r200"]],
                on=["suite", "sim_id", "row"])
assert len(tab) == 1154, "dmo_sums join changed the row count"
assert np.allclose(tab.halo_mass, tab.M200), "dmo_sums join misaligned"

# ── Aperture-contamination mask (recipe copied from figT6, _build_paper_nbs) ─
# The external-mass census uses the 1e12-threshold catalogs of the SAME CV sims
# (fm_lowmass eval; catalog properties are DMO-only, model-independent), so
# projected neighbors in the 1e12-1e13 decade are counted too — this matches
# the referee-response run (clean N=1089), whereas the 1e13-threshold spine
# catalogs would miss those contaminants (clean N=1118).
LOWMASS_CAT = ("/mnt/home/mlee1/ceph/fm_lowmass/CV/{sim}/snap_090/"
               "mass_threshold_1p000e12/halo_catalog.npz")
contam = np.full(len(tab), np.nan)          # external catalog mass / own M200
for sim_id, sub in tab.groupby("sim_id"):
    cat = np.load(LOWMASS_CAT.format(sim=sim_id))
    cen2 = np.asarray(cat["centers"]); m_all = np.asarray(cat["masses"], float)
    r200 = np.asarray(cat["r200s"], float)
    sel = np.flatnonzero(m_all >= 1e13)     # spine rows = >=1e13 subset, in order
    rows = sel[sub["row"].to_numpy()]
    assert np.allclose(np.log10(m_all[rows]), sub["log_m200c"]), sim_id
    d = np.abs(cen2[rows][:, None, :] - cen2[None, :, :])
    d = np.minimum(d, C.BOX_SIZE - d)                       # periodic
    dproj = np.hypot(d[..., 0], d[..., 1])                  # (n_sub, n_all)
    for k, (i, ri) in enumerate(zip(sub.index, rows)):
        ins = dproj[k] < r200[ri]; ins[ri] = False
        contam[i] = m_all[ins].sum() / m_all[ri]
CONTAM_MAX = 0.2
clean = contam < CONTAM_MAX
N_ALL, N_CLEAN = len(tab), int(clean.sum())
print(f"CV >=1e13: N={N_ALL} halos, {tab.sim_id.nunique()} sims; "
      f"aperture-clean (M_ext < {CONTAM_MAX} M200): N={N_CLEAN} "
      f"({N_ALL - N_CLEAN} dropped)")

# ── Log-mass columns ────────────────────────────────────────────────────────
q = {}
for src in ("truth", "gen"):
    q[f"{src}_star"] = tab[f"{src}_Stars_rvir"].to_numpy(float)
    q[f"{src}_gas"] = tab[f"{src}_Gas_rvir"].to_numpy(float)
    q[f"{src}_dm"] = tab[f"{src}_DM_hydro_rvir"].to_numpy(float)
    q[f"{src}_bar"] = q[f"{src}_gas"] + q[f"{src}_star"]
q["M200"] = tab.halo_mass.to_numpy(float)
q["DMO"] = tab.dmo_r200.to_numpy(float)
SIM = tab.sim_id.to_numpy()
SRC_COL = {"Truth": "truth", "BIND": "gen"}


def fit_relation(x, y, fit_on=None):
    """OLS log10(y) = alpha*log10(x) + beta with the x>1, y>1 validity mask,
    fitted on `fit_on & valid`; residuals returned for all valid halos.
    sigma = residual std over the fit sample."""
    valid = np.isfinite(x) & np.isfinite(y) & (x > 1) & (y > 1)
    fmask = valid if fit_on is None else (valid & fit_on)
    alpha, beta = np.polyfit(np.log10(x[fmask]), np.log10(y[fmask]), 1)
    res = np.full(len(x), np.nan)
    res[valid] = np.log10(y[valid]) - (alpha * np.log10(x[valid]) + beta)
    return dict(alpha=alpha, beta=beta, sigma=np.nanstd(res[fmask]),
                valid=valid, fmask=fmask, resid=res, x=x, y=y)


# relation key -> (y-quantity, x-quantity, xlabel, ylabel, title, pivot)
PAPER_RELS = {
    "Mgas-Mstar": ("gas", "star", r"$M_\star$", r"$M_{\rm gas}$",
                   r"$M_{\rm gas}-M_\star$", 11.5),
    "Mdm-Mstar": ("dm", "star", r"$M_\star$", r"$M_{\rm dm}$",
                  r"$M_{\rm dm}-M_\star$", 11.5),
    "SHMR": ("star", "M200", r"$M_{200c}$", r"$M_\star$", "SHMR", 13.5),
    "Mbar-M200": ("bar", "M200", r"$M_{200c}$", r"$M_{\rm baryon}$",
                  r"$(M_{\rm gas}+M_\star)-M_{200}$", 13.5),
}
COMMONX_RELS = {
    "Mstar|M200": ("star", "M200", 13.5), "Mgas|M200": ("gas", "M200", 13.5),
    "Mdm|M200": ("dm", "M200", 13.5), "Mbar|M200": ("bar", "M200", 13.5),
}
DMO_RELS = {"Mgas|DMO": ("gas", "DMO", 13.5), "Mstar|DMO": ("star", "DMO", 13.5)}


def xvec(src, xn):
    return q[xn] if xn in ("M200", "DMO") else q[f"{src}_{xn}"]


def fit_set(rels, fit_on):
    out = {"Truth": {}, "BIND": {}}
    for key, spec in rels.items():
        yn, xn = spec[0], spec[1]
        for lab, src in SRC_COL.items():
            out[lab][key] = fit_relation(xvec(src, xn), q[f"{src}_{yn}"], fit_on)
    return out


SAMPLES = {"full": None, "clean": clean}
FITS = {samp: {**fit_set(PAPER_RELS, m), } for samp, m in SAMPLES.items()}
CFITS = {samp: fit_set({**COMMONX_RELS, **DMO_RELS}, m) for samp, m in SAMPLES.items()}

# ── Stats: fit tables ───────────────────────────────────────────────────────
print("\n=== OLS fits: log10(y) = alpha*log10(x) + beta  (sigma = scatter, dex) ===")
for samp in ("full", "clean"):
    n = N_ALL if samp == "full" else N_CLEAN
    print(f"\n-- sample: {samp} (fit on N<={n}) --")
    print(f"{'relation':>12s} | {'aT':>6s} {'bT':>7s} {'y(piv)T':>8s} {'sigT':>6s} |"
          f" {'aB':>6s} {'bB':>7s} {'y(piv)B':>8s} {'sigB':>6s}")
    for rels, fits in ((PAPER_RELS, FITS[samp]), ({**COMMONX_RELS, **DMO_RELS}, CFITS[samp])):
        for key, spec in rels.items():
            piv = spec[-1]
            ft, fb = fits["Truth"][key], fits["BIND"][key]
            print(f"{key:>12s} | {ft['alpha']:+6.3f} {ft['beta']:+7.3f} "
                  f"{ft['alpha']*piv+ft['beta']:8.3f} {ft['sigma']:6.3f} |"
                  f" {fb['alpha']:+6.3f} {fb['beta']:+7.3f} "
                  f"{fb['alpha']*piv+fb['beta']:8.3f} {fb['sigma']:6.3f}"
                  f"   (pivot logx={piv})")

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — scaling_fits: 2 (Truth/BIND) x 4 relations, residual-colored
# ════════════════════════════════════════════════════════════════════════════
keys = list(PAPER_RELS)
VLIM = 0.25
fig, axes = plt.subplots(2, len(keys), figsize=(3.3 * len(keys), 6.4),
                         constrained_layout=True)
for row, src in enumerate(["Truth", "BIND"]):
    for col, key in enumerate(keys):
        ax = axes[row, col]; fd = FITS["full"][src][key]
        m = fd["valid"]
        lx, ly = np.log10(fd["x"][m]), np.log10(fd["y"][m])
        sc = ax.scatter(lx, ly, c=fd["resid"][m], cmap="RdBu_r",
                        vmin=-VLIM, vmax=VLIM, s=14, rasterized=True, zorder=3)
        xl_ = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(xl_, fd["alpha"] * xl_ + fd["beta"], "k--", lw=1.2)
        ax.fill_between(xl_, fd["alpha"] * xl_ + fd["beta"] - fd["sigma"],
                        fd["alpha"] * xl_ + fd["beta"] + fd["sigma"],
                        color="k", alpha=0.10)
        ax.text(0.04, 0.96,
                f"{src}\n" + rf"$\alpha={fd['alpha']:.2f},\ \sigma={fd['sigma']:.2f}$ dex",
                transform=ax.transAxes, va="top", fontsize=8, color="0.35")
        if row == 0:
            ax.set_title(PAPER_RELS[key][4])
        if row == 1:
            ax.set_xlabel(r"$\log_{10}($" + PAPER_RELS[key][2] + r"$)$")
        ax.set_ylabel(r"$\log_{10}($" + PAPER_RELS[key][3] + r"$)$")
for col in range(len(keys)):        # share limits within a column (Truth vs BIND)
    xlo = min(axes[r, col].get_xlim()[0] for r in (0, 1))
    xhi = max(axes[r, col].get_xlim()[1] for r in (0, 1))
    ylo = min(axes[r, col].get_ylim()[0] for r in (0, 1))
    yhi = max(axes[r, col].get_ylim()[1] for r in (0, 1))
    for r in (0, 1):
        axes[r, col].set_xlim(xlo, xhi); axes[r, col].set_ylim(ylo, yhi)
cb = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01)
cb.set_label("residual [dex]")
save_fig(fig, "scaling_fits")
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — residual_corner_plot: COMMON-X corner (all residuals about M200c)
# ════════════════════════════════════════════════════════════════════════════
ckeys = ["Mstar|M200", "Mgas|M200", "Mdm|M200", "Mbar|M200"]
CORNER_LABELS = {
    "Mstar|M200": r"$\Delta\log M_\star\,|\,M_{200c}$",
    "Mgas|M200": r"$\Delta\log M_{\rm gas}\,|\,M_{200c}$",
    "Mdm|M200": r"$\Delta\log M_{\rm dm}\,|\,M_{200c}$",
    "Mbar|M200": r"$\Delta\log M_{\rm bar}\,|\,M_{200c}$",
}
# Corner sample: aperture-clean, residuals about the clean fits (the figT6b
# recipe — contaminated halos are dropped before fitting/plotting).
cfits = CFITS["clean"]
both = clean.copy()
for k in ckeys:
    both &= cfits["Truth"][k]["valid"] & cfits["BIND"][k]["valid"]
print(f"\ncorner sample (aperture-clean, all four residuals valid): N={both.sum()}")
R = {s: np.stack([cfits[s][k]["resid"][both] for k in ckeys], 1) for s in ("Truth", "BIND")}
sig = {s: np.array([cfits[s][k]["sigma"] for k in ckeys]) for s in ("Truth", "BIND")}

valid_all = np.ones(len(tab), bool)
for k in ckeys:
    valid_all &= (CFITS["full"]["Truth"][k]["valid"] & CFITS["full"]["BIND"][k]["valid"])
print("\n=== COMMON-X residual correlation matrices (Pearson) ===")
for samp, msk in (("full", valid_all), ("clean", both)):
    cf = CFITS[samp]
    print(f"-- {samp} (N={msk.sum()}), format truth|BIND --")
    print("%12s" % "", "".join("%19s" % k for k in ckeys))
    for ka in ckeys:
        line = "%12s" % ka
        for kb in ckeys:
            rt = stats.pearsonr(cf["Truth"][ka]["resid"][msk], cf["Truth"][kb]["resid"][msk])[0]
            rb = stats.pearsonr(cf["BIND"][ka]["resid"][msk], cf["BIND"][kb]["resid"][msk])[0]
            line += "%9.3f|%+8.3f" % (rt, rb)
        print(line)

nrel = len(ckeys)
rng_lim = []
for j in range(nrel):
    v = np.r_[R["Truth"][:, j], R["BIND"][:, j]]
    lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
    pad = 0.15 * (hi - lo); rng_lim.append((lo - pad, hi + pad))


def kde_contour(ax, x, y, color, ls):
    """1/2/3-sigma-enclosure KDE contours of the (x, y) residual sub-population."""
    if len(x) < 10 or np.ptp(x) < 1e-10 or np.ptp(y) < 1e-10:
        return
    kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
    xg, yg = np.mgrid[x.min() - 0.2 * np.ptp(x):x.max() + 0.2 * np.ptp(x):80j,
                      y.min() - 0.2 * np.ptp(y):y.max() + 0.2 * np.ptp(y):80j]
    z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    zs = np.sort(z.ravel())[::-1]; cf = np.cumsum(zs) / zs.sum()
    lv = sorted({float(zs[min(np.searchsorted(cf, f), len(zs) - 1)])
                 for f in (0.6827, 0.9545, 0.9973)})
    ax.contour(xg, yg, z, levels=lv, colors=[color], linewidths=1.2,
               linestyles=ls, alpha=0.85)


fig, axes = plt.subplots(nrel, nrel, figsize=(2.4 * nrel, 2.4 * nrel),
                         constrained_layout=True)
for i in range(nrel):
    for j in range(nrel):
        ax = axes[i, j]
        if j > i:
            ax.set_visible(False); continue
        tails = {s: (R[s][:, j] > sig[s][j], R[s][:, j] < -sig[s][j]) for s in R}
        if i == j:
            bins = np.linspace(*rng_lim[i], 35)
            ax.hist(R["Truth"][tails["Truth"][0], i], bins=bins, color="tomato",
                    density=True, alpha=0.7)
            ax.hist(R["Truth"][tails["Truth"][1], i], bins=bins, color="steelblue",
                    density=True, alpha=0.7)
            ax.hist(R["BIND"][tails["BIND"][0], i], bins=bins, color="tomato",
                    density=True, histtype="step", ls=":", lw=1.8)
            ax.hist(R["BIND"][tails["BIND"][1], i], bins=bins, color="steelblue",
                    density=True, histtype="step", ls=":", lw=1.8)
            ax.axvline(sig["Truth"][i], c="tomato", lw=0.8, ls=":", alpha=0.6)
            ax.axvline(-sig["Truth"][i], c="steelblue", lw=0.8, ls=":", alpha=0.6)
            ax.axvline(0, c="k", lw=0.7, ls="--", alpha=0.4)
            ax.set_title(CORNER_LABELS[ckeys[i]], fontsize=10, pad=2)
        else:
            kde_contour(ax, R["Truth"][tails["Truth"][0], j], R["Truth"][tails["Truth"][0], i], "tomato", "-")
            kde_contour(ax, R["Truth"][tails["Truth"][1], j], R["Truth"][tails["Truth"][1], i], "steelblue", "-")
            kde_contour(ax, R["BIND"][tails["BIND"][0], j], R["BIND"][tails["BIND"][0], i], "tomato", ":")
            kde_contour(ax, R["BIND"][tails["BIND"][1], j], R["BIND"][tails["BIND"][1], i], "steelblue", ":")
            ax.axhline(0, c="k", lw=0.4, ls="--", alpha=0.3)
            ax.axvline(0, c="k", lw=0.4, ls="--", alpha=0.3)
            ax.set_ylim(rng_lim[i])
        ax.set_xlim(rng_lim[j])
        if j == 0 and i > 0:
            ax.set_ylabel(CORNER_LABELS[ckeys[i]] + "\n[dex]")
        elif j > 0:
            ax.set_yticklabels([])
        if i == nrel - 1:
            ax.set_xlabel(CORNER_LABELS[ckeys[j]] + "\n[dex]")
        else:
            ax.set_xticklabels([])
fig.legend(handles=[Line2D([0], [0], color=c, ls=ls, lw=1.6, label=lb)
                    for c, ls, lb in
                    [("tomato", "-", r"Truth $>+1\sigma$"),
                     ("steelblue", "-", r"Truth $<-1\sigma$"),
                     ("tomato", ":", r"BIND $>+1\sigma$"),
                     ("steelblue", ":", r"BIND $<-1\sigma$")]],
           loc="upper right", frameon=True, fontsize=11)
save_fig(fig, "residual_corner_plot")
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — fig_matched_residuals: per-halo Delta_true vs Delta_BIND (clean)
# ════════════════════════════════════════════════════════════════════════════
def block_boot_ci(x, y, sim, nb=2000):
    """95% CI on Pearson r from a sim-block bootstrap (sims are the
    independent unit; halos within a sim share a box)."""
    groups = [np.where(sim == s)[0] for s in np.unique(sim)]
    out = np.empty(nb)
    for b in range(nb):
        idx = np.concatenate([groups[p] for p in rng.integers(0, len(groups), len(groups))])
        out[b] = stats.pearsonr(x[idx], y[idx])[0]
    return np.percentile(out, [2.5, 97.5])


MATCH_PANELS = [
    ("Mgas-Mstar", r"$M_{\rm gas}-M_\star$", "PAPER"),
    ("Mdm-Mstar", r"$M_{\rm dm}-M_\star$", "PAPER"),
    ("SHMR", r"SHMR  $M_\star-M_{200c}$", "PAPER"),
    ("Mbar-M200", r"$M_{\rm bar}-M_{200c}$", "PAPER"),
    ("Mgas|DMO", r"$M_{\rm gas}$ at fixed $M_{\rm DMO}(<R_{200c})$", "COMMON"),
    ("Mstar|DMO", r"$M_\star$ at fixed $M_{\rm DMO}(<R_{200c})$", "COMMON"),
]
print(f"\n=== Matched per-halo residuals, aperture-clean CV (N={N_CLEAN}) ===")
print(f"{'panel':>12s} {'r':>7s} {'95% CI (sim-block)':>20s} {'spearman':>9s} {'slope':>7s} {'N':>5s}")
fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), constrained_layout=True)
match_stats = {}
for ax, (key, title, which) in zip(axes.ravel(), MATCH_PANELS):
    fits = FITS["clean"] if which == "PAPER" else CFITS["clean"]
    ft, fb = fits["Truth"][key], fits["BIND"][key]
    m = ft["valid"] & fb["valid"] & clean
    a, b = ft["resid"][m], fb["resid"][m]
    r = stats.pearsonr(a, b)[0]
    ci = block_boot_ci(a, b, SIM[m])
    rs = stats.spearmanr(a, b)[0]
    sl = np.polyfit(a, b, 1)[0]
    match_stats[key] = (r, ci, rs, sl, int(m.sum()))
    print(f"{key:>12s} {r:+7.3f}    [{ci[0]:+.3f}, {ci[1]:+.3f}] {rs:+9.3f} {sl:+7.3f} {m.sum():5d}")

    ax.scatter(a, b, s=7, alpha=0.35, rasterized=True, color=C.SUITE_COLORS["CV"])
    L = 1.05 * max(np.abs(a).max(), np.abs(b).max()); L = min(L, 0.6)
    ax.plot([-L, L], [-L, L], "k--", lw=1, zorder=1)
    ax.axhline(0, color="0.7", lw=0.6); ax.axvline(0, color="0.7", lw=0.6)
    xs = np.linspace(-L, L, 10)
    ax.plot(xs, sl * xs, "-", color="crimson", lw=1.4)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\Delta_{\rm true}$ [dex]")
    ax.set_ylabel(r"$\Delta_{\rm BIND}$ [dex]")
    ax.text(0.04, 0.96, rf"$r={r:.2f}^{{+{ci[1]-r:.2f}}}_{{-{r-ci[0]:.2f}}}$",
            transform=ax.transAxes, va="top", fontsize=11)
save_fig(fig, "fig_matched_residuals")
plt.close(fig)

# ── Headline: sign-corrected gas-star residual correlation ──────────────────
print("\n=== Headline r(Dgas|M200c, Dstar|M200c) ===")
for samp, msk in (("full", np.ones(len(tab), bool)), ("clean", clean)):
    cf = CFITS[samp]
    for lab in ("Truth", "BIND"):
        g, s = cf[lab]["Mgas|M200"], cf[lab]["Mstar|M200"]
        m = g["valid"] & s["valid"] & msk
        r = stats.pearsonr(g["resid"][m], s["resid"][m])[0]
        ci = block_boot_ci(g["resid"][m], s["resid"][m], SIM[m])
        print(f"  {samp:>5s} {lab:>5s}: r = {r:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]  (N={m.sum()})")
    # old mixed-x convention, for the sign-flip contrast
    fits = FITS[samp]
    for lab in ("Truth", "BIND"):
        g, s = fits[lab]["Mgas-Mstar"], fits[lab]["SHMR"]
        m = g["valid"] & s["valid"] & msk
        r = stats.pearsonr(g["resid"][m], s["resid"][m])[0]
        print(f"  {samp:>5s} {lab:>5s}: OLD mixed-x r(Dgas^(gas-star), Dstar^SHMR) = {r:+.3f}")

print("\ndone.")
