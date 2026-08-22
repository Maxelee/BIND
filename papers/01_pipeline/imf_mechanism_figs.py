#!/usr/bin/env python
"""imf_mechanism_figs.py -- two DIRECT mechanism figures for the IMFslope check,
companions to fig21 (which compares observable response *shapes* only).

fig21b  PER-HALO MASS-SCALE FINGERPRINT (physical channel discriminator).
        Spearman rho across the 256 Sobol runs between each parameter and the
        run-mean halo gas content, per M200 bin, z=0.034 snapshot of
        analysis_cache/integrated.parquet. SN winds act in low-mass halos,
        AGN at group/cluster scale, so WHERE a parameter moves f_gas / Y is a
        channel fingerprint that needs no shape normalization. Marginal rho
        (all other 29 params vary; same convention as the fig05 importance
        matrix). Shape stamps use sign-aligned profiles (fig21 convention).

fig21c  SOBOL INTERACTION CORNER (causal non-additivity test).
        Top: the Sobol design, x=IMFslope vs y=partner, colored by the WL
        suppression deficit D = 1 - <S(ell)>_{1e3..1e4} at z_s=1. Sobol axes
        are independent by construction, so the information is in the COLOR
        FIELD: a gradient along x that changes with y is an IMFslope-partner
        interaction. Bottom: that interaction quantified for all 29 partners --
        OLS of D (and of the band Cl_yy) on all 30 centered mains + one
        IMFslope*partner product, bootstrap t-statistic of the product term.
        If IMFslope acts THROUGH the AGN channel, its effect must be modulated
        by AGN knobs; shape resemblance (fig21) cannot distinguish "acts
        through" from "independently similar", this can.

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python imf_mechanism_figs.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, save, panel_label, COLORS, BAND_ALPHA, TWO_COL, TWO_COL_TALL  # noqa: E402
from param_labels import short_label  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
PARQUET = CEPH / "bind_sb35/analysis_cache/integrated.parquet"
DS_PATH = CEPH / "bind_sb35/emulator_dataset_xpkfix.npz"
ACSV = "/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv"
assert Path.cwd().name == "01_pipeline", "run from papers/01_pipeline/"

RNG = np.random.default_rng(21)

# fig21 family/color conventions, byte-identical
PCOL = {"IMFslope": "#111111",
        "BlackHoleRadiativeEfficiency": COLORS["highlight"],
        "VariableWindVelFactor": COLORS["bind"]}


def family(name):
    if any(k in name for k in ("BlackHole", "Quasar", "Radio")):
        return "agn"
    if "Wind" in name or "SN" in name:
        return "wind"
    return "other"


FAMCOL = {"wind": COLORS["bind"], "agn": COLORS["highlight"], "other": "#555555"}


def spearman(P, Y):
    """(n,p),(n,b) -> (p,b) Spearman rank correlation."""
    rp = rankdata(P, axis=0).astype(float)
    ry = rankdata(Y, axis=0).astype(float)
    rp = (rp - rp.mean(0)) / rp.std(0)
    ry = (ry - ry.mean(0)) / np.maximum(ry.std(0), 1e-12)
    return rp.T @ ry / len(P)


def _align(v):
    """Sign-align so the peak-|.| entry is positive (fig21 convention)."""
    return v * np.sign(v[np.argmax(np.abs(v))])


# ════════════════════════════════ fig21b: per-halo mass fingerprint ═════════
print("=== fig21b: per-halo mass-scale fingerprint ===")
pnames30 = [str(s) for s in np.load(DS_PATH, allow_pickle=True)["param_names"]]
# native astro columns exist in the parquet; read the z=0.034 snapshot only
cols = ["run", "M200", "f_gas_200", "Y_200"] + pnames30
tab = pq.read_table(PARQUET, columns=cols, filters=[("z", "<", 0.05)])
df = tab.to_pandas()
print(f"z=0.034 rows: {len(df)}  runs: {df['run'].nunique()}")

EDGES = np.arange(13.0, 14.61, 0.2)
CEN = 0.5 * (EDGES[:-1] + EDGES[1:])
df["lm"] = np.log10(df["M200"])
df["bin"] = np.digitize(df["lm"], EDGES) - 1
df = df[(df["bin"] >= 0) & (df["bin"] < len(CEN))]

# per-run parameter vector (constant within a run)
P = df.drop_duplicates("run").sort_values("run")[pnames30].to_numpy(float)
runs = np.sort(df["run"].unique())
assert P.shape == (len(runs), 30)

g = df.groupby(["run", "bin"])
FG = g["f_gas_200"].mean().unstack().reindex(index=runs, columns=range(len(CEN)))
cnt = g.size().unstack().reindex(index=runs, columns=range(len(CEN))).fillna(0)
df["logY"] = np.log10(df["Y_200"].where(df["Y_200"] > 0))
LY = df.groupby(["run", "bin"])["logY"].mean().unstack().reindex(
    index=runs, columns=range(len(CEN)))
keep = (cnt.min(0) >= 5).to_numpy()            # every run populates the bin
print(f"bins kept: {keep.sum()}/{len(CEN)}  (min per-run counts: "
      f"{cnt.min(0).to_numpy().astype(int)})")
CENk = CEN[keep]
FGk, LYk = FG.to_numpy()[:, keep], LY.to_numpy()[:, keep]

RHO = {"fg": spearman(P, FGk), "Y": spearman(P, LYk)}        # (30, nb)
NBOOT = 1000
boots = {k: np.empty((NBOOT,) + RHO[k].shape) for k in RHO}
for b in range(NBOOT):
    idx = RNG.integers(0, len(runs), len(runs))
    boots["fg"][b] = spearman(P[idx], FGk[idx])
    boots["Y"][b] = spearman(P[idx], LYk[idx])
ERR = {k: boots[k].std(0) for k in RHO}

FOCAL5 = ["IMFslope", "BlackHoleRadiativeEfficiency", "QuasarThreshold",
          "VariableWindVelFactor", "WindEnergyIn1e51erg"]
STYLE = {"IMFslope": dict(color=PCOL["IMFslope"], lw=1.8, ls="-", zorder=5),
         "BlackHoleRadiativeEfficiency": dict(color=COLORS["highlight"], lw=1.1, ls="-", zorder=4),
         "QuasarThreshold": dict(color=COLORS["highlight"], lw=1.1, ls="--", zorder=4),
         "VariableWindVelFactor": dict(color=COLORS["bind"], lw=1.1, ls="-", zorder=3),
         "WindEnergyIn1e51erg": dict(color=COLORS["bind"], lw=1.1, ls="--", zorder=3)}
iP = {n: pnames30.index(n) for n in pnames30}

fig, axes = plt.subplots(1, 2, figsize=TWO_COL, sharex=True)
for ax, key, lab in [(axes[0], "fg", r"$\rho\,(\theta,\ \langle f_{\rm gas,200}\rangle)$"),
                     (axes[1], "Y", r"$\rho\,(\theta,\ \langle \log Y_{200}\rangle)$")]:
    for nme in FOCAL5:
        r, e = RHO[key][iP[nme]], ERR[key][iP[nme]]
        ax.plot(CENk, r, label=short_label(nme), **STYLE[nme])
        ax.fill_between(CENk, r - e, r + e, color=STYLE[nme]["color"],
                        alpha=BAND_ALPHA, lw=0)
    ax.axhline(0, color="0.75", lw=0.6, zorder=1)
    ax.set_xlabel(r"$\log_{10} M_{200}\,[M_\odot/h]$")
    ax.set_ylabel(lab)
panel_label(axes[0], "(a) gas mass")
panel_label(axes[1], "(b) thermal (Compton $Y$)")
axes[0].legend(loc="lower right", fontsize=5.8, handlelength=1.8)
fig.tight_layout()
save(fig, "figs_v2/fig21b_mechanism_perhalo")
plt.close(fig)

# NB: with 8 near-monotone bins, profile-Pearson "shape r" stamps are not
# discriminative (all monotone profiles correlate at |r|~1) and peak-sign
# alignment is unstable on near-antisymmetric profiles. The robust per-halo
# discriminators are (i) WHERE |rho| concentrates in mass and (ii) whether
# the profile flips sign (the AGN self-regulation signature). Report those.
print("\nmass-profile diagnostics per parameter:")
for key in ("fg", "Y"):
    for nme in FOCAL5:
        r = RHO[key][iP[nme]]
        cen = float((CENk * np.abs(r)).sum() / np.abs(r).sum())
        flip = "sign-flip" if (r.min() < -2 * ERR[key][iP[nme]].mean()
                               and r.max() > 2 * ERR[key][iP[nme]].mean()) else "monotone-sign"
        print(f"  [{key:2s}] {short_label(nme):18s} |rho|-centroid logM={cen:.2f}  "
              f"range [{r.min():+.2f},{r.max():+.2f}]  {flip}")

# ════════════════════════════════ fig21c: interaction corner ════════════════
print("\n=== fig21c: Sobol interaction corner ===")
d = np.load(DS_PATH, allow_pickle=True)
X = np.asarray(d["X_unit"], float)                       # (256, 30) unit prior
pnames = [str(s) for s in d["param_names"]]
assert pnames == pnames30, "param order mismatch parquet CSV vs dataset"
zsrc = np.asarray(d["source_redshifts"], float)
ZI = int(np.argmin(np.abs(zsrc - 1.0)))
ell = np.asarray(d["a__suppression__ell"], float)
S = np.asarray(d["t__suppression__value"], float)[:, ZI, :]
YY = np.asarray(d["t__cl_yy__value"], float)
band = (ell >= 1e3) & (ell <= 1e4)
D = 1.0 - np.nanmean(S[:, band], axis=1)                 # >0 = suppressed
dYY = np.log10(np.nanmean(YY[:, band], axis=1))
dYY -= np.median(dYY)
ok = np.isfinite(D) & np.isfinite(dYY)
X, D, dYY = X[ok], D[ok], dYY[ok]
print(f"z_s={zsrc[ZI]:g}, band ell in [1e3,1e4]; runs kept {ok.sum()}/{len(ok)}; "
      f"D range [{D.min():+.3f}, {D.max():+.3f}]")

mm = pd.read_csv(ACSV).set_index("ParamName")
iIMF = pnames.index("IMFslope")
Xc = X - X.mean(0)                                       # centered mains


def interaction_scan(y):
    """t-stat of the IMFslope*partner product term, per partner, with all 30
    centered mains in the model; bootstrap std over runs."""
    t, c, s = {}, {}, {}
    ones = np.ones((len(Xc), 1))
    for j, nme in enumerate(pnames):
        if j == iIMF:
            continue
        A = np.hstack([ones, Xc, (Xc[:, iIMF] * Xc[:, j])[:, None]])
        coef = np.linalg.lstsq(A, y, rcond=None)[0][-1]
        bs = np.empty(2000)
        for b in range(2000):
            k = RNG.integers(0, len(y), len(y))
            bs[b] = np.linalg.lstsq(A[k], y[k], rcond=None)[0][-1]
        c[nme], s[nme] = coef, bs.std()
        t[nme] = coef / bs.std()
    return t, c, s


tD, cD, sD = interaction_scan(D)
tY, cY, sY = interaction_scan(dYY)

PARTNERS = ["BlackHoleRadiativeEfficiency", "VariableWindVelFactor", "WindEnergyIn1e51erg"]
fig = plt.figure(figsize=(7.2, 8.2))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.55, 0.55], hspace=0.52, wspace=0.32)

vmax = np.percentile(np.abs(D), 98)
sc = None
for k, nme in enumerate(PARTNERS):
    ax = fig.add_subplot(gs[0, k])
    j = pnames.index(nme)
    sc = ax.scatter(X[:, iIMF], X[:, j], c=D, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=13, edgecolors="0.6", linewidths=0.25)
    # conditional-slope stamp: Spearman rho(IMF, D) in lower/upper half of partner
    lo, hi = X[:, j] < np.median(X[:, j]), X[:, j] >= np.median(X[:, j])
    r_lo = spearman(X[lo, iIMF:iIMF + 1], D[lo, None])[0, 0]
    r_hi = spearman(X[hi, iIMF:iIMF + 1], D[hi, None])[0, 0]
    ax.text(0.03, 0.97, f"$\\rho_{{\\rm low\\,{{y}}}}={r_lo:+.2f}$\n"
            f"$\\rho_{{\\rm high\\,{{y}}}}={r_hi:+.2f}$",
            transform=ax.transAxes, fontsize=6.0, va="top",
            bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.2))
    row = mm.loc[nme]
    ax.set_xticks([0, 1], [f"{mm.loc['IMFslope','MinVal']:g}",
                           f"{mm.loc['IMFslope','MaxVal']:g}"])
    ax.set_yticks([0, 1], [f"{row['MinVal']:g}", f"{row['MaxVal']:g}"])
    ax.set_xlabel("IMFslope  (steep $\\to$ top-heavy)")
    ax.set_ylabel(short_label(nme), color=FAMCOL[family(nme)])
    panel_label(ax, f"({'abc'[k]})", loc="lower left")
cax = fig.add_axes([0.92, 0.68, 0.013, 0.22])
cb = fig.colorbar(sc, cax=cax)
cb.set_label(r"$D = 1 - \langle S(\ell)\rangle_{10^3-10^4}$", fontsize=7)
cb.ax.tick_params(labelsize=6)

# bottom: interaction t-stat bars, both metrics, one full-width row each
for k, (tt, lab) in enumerate([(tD, r"WL deficit $D$"),
                               (tY, r"band $\log_{10} C_\ell^{yy}$")]):
    ax = fig.add_subplot(gs[1 + k, :])
    order = sorted(tt, key=lambda n: -abs(tt[n]))
    vals = [tt[n] for n in order]
    cols = [FAMCOL[family(n)] for n in order]
    ax.bar(range(len(order)), vals, color=cols, width=0.75)
    ax.axhline(0, color="k", lw=0.6)
    for y0 in (-2, 2):
        ax.axhline(y0, color="0.75", lw=0.6, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short_label(n) for n in order], rotation=55, ha="right",
                       fontsize=5.5)
    for tick, n in zip(ax.get_xticklabels(), order):
        tick.set_color(FAMCOL[family(n)])
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylabel(f"$t$: IMF$\\times\\theta_j$ on {lab}", fontsize=6.5)
    panel_label(ax, f"({'de'[k]})", loc="upper right")
save(fig, "figs_v2/fig21c_mechanism_interaction")
plt.close(fig)

print("\ntop-6 |t| interactions with IMFslope:")
for tt, lab in [(tD, "WL D"), (tY, "log Cl_yy")]:
    top = sorted(tt, key=lambda n: -abs(tt[n]))[:6]
    print(f"  [{lab}] " + "; ".join(f"{short_label(n)} {tt[n]:+.1f}" for n in top))
agn = [n for n in tD if family(n) == "agn"]
wnd = [n for n in tD if family(n) == "wind"]
for tt, lab in [(tD, "WL D"), (tY, "log Cl_yy")]:
    print(f"  [{lab}] mean |t|: AGN {np.mean([abs(tt[n]) for n in agn]):.2f}  "
          f"wind {np.mean([abs(tt[n]) for n in wnd]):.2f}  "
          f"other {np.mean([abs(tt[n]) for n in tD if family(n) == 'other']):.2f}")
