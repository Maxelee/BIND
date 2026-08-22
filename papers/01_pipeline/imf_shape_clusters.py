#!/usr/bin/env python
"""imf_shape_clusters.py -- extend the Delta-shape hierarchical clustering
(quick_deltacl_clusters.png, C_ell^kappa) to the other twobound statistics:

    pdf   kappa PDF (nongaussian_stats.npz; raw hi-lo difference; no paired
          errors stored, so params are S/N-gated by their V0 response instead)
    clyy  C_ell^yy (Cl_kappa_y.npz; fractional vs fiducial run_0000 -- ratio
          use only, the pre-xpkfix norm cancels)
    V0/V1/V2  Minkowski functionals (paired_stats.npz resp differences)

Peaks/minima are excluded: the bound-to-bound count differences are S/N<3 for
(nearly) all params (see imf_mechanism_blend.py output). Per statistic:
unit-peak-normalized Delta shapes for all S/N-passing params, average-linkage
clustering on 1-r, one panel per multi-member cluster with member names,
IMFslope in black. Writes figs_preview/quick_clusters_<stat>.png.

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python imf_shape_clusters.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, panel_label, COLORS  # noqa: E402
from param_labels import short_label  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
TB = CEPH / "bind_science/runs/twobound"
FID = CEPH / "bind_science/runs/bind/run_0000"
assert Path.cwd().name == "01_pipeline", "run from papers/01_pipeline/"

ZI = 1
SNR_MIN = 3.0
T_CLUST = 0.15


def family(name):
    if any(k in name for k in ("BlackHole", "Quasar", "Radio")):
        return "agn"
    if "Wind" in name or "SN" in name:
        return "wind"
    return "other"


FAMCOL = {"wind": COLORS["bind"], "agn": COLORS["highlight"], "other": "#999999"}

names35 = list(pd.read_csv(
    "/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv")["ParamName"])
tbp = np.load(TB / "twobound_params.npy")
pairs = {}
for i in range(30):
    d = np.where(tbp[2 * i] != tbp[2 * i + 1])[0]
    assert len(d) == 1
    j = int(d[0])
    rr = [2 * i, 2 * i + 1]
    vv = [float(tbp[r, j]) for r in rr]
    o = np.argsort(vv)
    pairs[names35[j]] = [rr[k] for k in o]

P = {r: np.load(TB / f"run_{r:04d}/paired_stats.npz") for r in range(60)}
NG = {r: np.load(TB / f"run_{r:04d}/nongaussian_stats.npz") for r in range(60)}
CY = {r: np.load(TB / f"run_{r:04d}/Cl_kappa_y.npz") for r in range(60)}
cy_fid = np.load(FID / "Cl_kappa_y.npz")
ell, mf_nu = P[0]["ell"], P[0]["mf_nu"]
pdf_bins = NG[0]["pdf_bins"]

EDGES = np.geomspace(100.0, 3e4, 27)


def logbin(y):
    ib = np.digitize(ell, EDGES) - 1
    L, Y = [], []
    for b in range(len(EDGES) - 1):
        s = ib == b
        if s.any():
            L.append(np.exp(np.log(ell[s]).mean()))
            Y.append(np.nanmean(y[s]))
    return np.array(L), np.array(Y)


def mf_delta(nm, key):
    rlo, rhi = pairs[nm]
    d = P[rhi][f"{key}_resp"][ZI] - P[rlo][f"{key}_resp"][ZI]
    e = np.sqrt(P[rhi][f"{key}_err"][ZI] ** 2 + P[rlo][f"{key}_err"][ZI] ** 2)
    return mf_nu, d, e


def get(nm, stat):
    """(x, Delta, err_or_None) for one parameter and statistic."""
    rlo, rhi = pairs[nm]
    if stat == "pdf":
        d = NG[rhi]["pdf"][ZI] - NG[rlo]["pdf"][ZI]
        return pdf_bins, d, None
    if stat == "clyy":
        f = cy_fid["cl_yy"]
        x, d = logbin((CY[rhi]["cl_yy"] - CY[rlo]["cl_yy"]) / f)
        _, e = logbin(np.sqrt(CY[rhi]["cl_yy_err"] ** 2 + CY[rlo]["cl_yy_err"] ** 2) / f)
        return x, d, e / np.sqrt(np.diff(np.searchsorted(ell, EDGES)).clip(1)[:len(x)])
    return mf_delta(nm, stat)


def snr_of(nm, stat):
    """peak-bin |Delta|/err; pdf uses the V0 response as its S/N proxy."""
    if stat == "pdf":
        _, d, e = mf_delta(nm, "V0")
    else:
        _, d, e = get(nm, stat)
    k = int(np.nanargmax(np.abs(d)))
    return float(np.abs(d[k]) / e[k])


STATS = [("pdf", r"$\kappa$ PDF", r"$\kappa$"),
         ("clyy", r"$C_\ell^{yy}$", r"$\ell$"),
         ("V0", r"$V_0(\nu)$", r"$\nu$"),
         ("V1", r"$V_1(\nu)$", r"$\nu$"),
         ("V2", r"$V_2(\nu)$", r"$\nu$")]

summary = {}
for stat, slab, xlab in STATS:
    CUR, excl = {}, []
    for nm in pairs:
        if snr_of(nm, stat) < SNR_MIN:
            excl.append(nm)
            continue
        x, d, _ = get(nm, stat)
        CUR[nm] = d / d[np.nanargmax(np.abs(d))]
    NAMES = list(CUR)
    M = np.corrcoef([CUR[n] for n in NAMES])
    Z = linkage(squareform(np.clip(1 - M, 0, None), checks=False), method="average")
    lab = fcluster(Z, t=T_CLUST, criterion="distance")
    clusters = []
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        mem = [NAMES[i] for i in idx]
        rbar = (M[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
                if len(idx) > 1 else np.nan)
        clusters.append((mem, rbar))
    clusters.sort(key=lambda c: -len(c[0]))
    multi = [c for c in clusters if len(c[0]) > 1]
    single = [c[0][0] for c in clusters if len(c[0]) == 1]

    print(f"\n=== {stat} (excluded S/N<{SNR_MIN:g}: "
          f"{', '.join(short_label(n) for n in excl) or 'none'}) ===")
    for k, (mem, rbar) in enumerate(multi):
        star = " <-- IMFslope" if "IMFslope" in mem else ""
        print(f"  C{k+1} (n={len(mem)}, r̄={rbar:.2f}): "
              + ", ".join(short_label(m) for m in mem) + star)
    if single:
        print(f"  singletons: {', '.join(short_label(m) for m in single)}")
    summary[stat] = next((mem for mem, _ in multi if "IMFslope" in mem),
                         ["(singleton)"] if "IMFslope" in single else ["(excluded)"])

    npan = len(multi) + (1 if single else 0)
    ncol = 3
    nrow = max(1, int(np.ceil(npan / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.3 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    panels = [(f"C{k+1}  ($\\bar r$={rbar:.2f})", mem)
              for k, (mem, rbar) in enumerate(multi)]
    if single:
        panels.append(("singletons", single))
    for k, (title, mem) in enumerate(panels):
        ax = axes[k]
        for nm in NAMES:
            ax.plot(x, CUR[nm], color="0.88", lw=0.5, zorder=1)
        for nm in mem:
            lw = 2.0 if nm == "IMFslope" else 1.1
            col = "#111111" if nm == "IMFslope" else FAMCOL[family(nm)]
            ax.plot(x, CUR[nm], color=col, lw=lw,
                    zorder=6 if nm == "IMFslope" else 4)
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        if stat == "clyy":
            ax.set_xscale("log")
        panel_label(ax, title)
        ax.text(0.97, 0.05, "\n".join(short_label(m) for m in mem),
                transform=ax.transAxes, fontsize=5.0, ha="right", va="bottom",
                bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.0))
    for ax in axes[npan:]:
        ax.set_visible(False)
    for k in range(npan):
        axes[k].tick_params(labelsize=6)
        if k % ncol == 0:
            axes[k].set_ylabel(f"$\\Delta$ {slab} / peak", fontsize=6.5)
        if k >= npan - ncol:
            axes[k].set_xlabel(xlab, fontsize=7)
    fig.tight_layout()
    out = f"figs_preview/quick_clusters_{stat}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")

print("\n================ IMFslope's shape family per statistic ================")
print(f"  {'clk':5s}: RadioFdbkFactor, RadioFdbkReorient, IMFslope, UVBH0beta, "
      "UVBH0Deltaz   (from quick_deltacl_clusters)")
for stat, mem in summary.items():
    print(f"  {stat:5s}: " + ", ".join(short_label(m) for m in mem))
