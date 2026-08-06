#!/usr/bin/env python
"""paper_s3b_clusters.py -- publication versions of the response-shape
cluster figures for the paper's S3b (2026-08-06 figure plan).

MAIN   (pfig_s3b_cl_clusters):  Delta C_ell^kappakappa shape families C1-C4 +
       singletons with per-cluster mean correlations rbar -- the families
       result; the C3 panel shows IMFslope (black) riding the AGN family,
       the mechanism conclusion in one glance.
APPENDIX (pfig_s3b_pdf_clusters): the same clustering on the kappa-PDF
       responses -- family consistency across statistics.

Machinery is imf_shape_clusters.py's, unchanged (same S/N gate, average-
linkage on 1-r of unit-peak-normalized twobound bound-to-bound response
differences, same family colors, IMFslope in black); this script only adds
the clk statistic (the scratch quick_deltacl_clusters.png leg, now
regenerated at publication grade) and renders through paper_style.save().
Supersedes for the paper: the single-panel IMF-highlight overlay and old
fig 21 (cut -- cluster membership carries both).

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python paper_s3b_clusters.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402
from param_labels import short_label  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
TB = CEPH / "bind_science/runs/twobound"
assert Path.cwd().name == "01_pipeline", "run from papers/01_pipeline/"

ZI = 1
SNR_MIN = 3.0
T_CLUST = 0.15
EDGES = np.geomspace(100.0, 3e4, 27)


def family(name):
    if any(k in name for k in ("BlackHole", "Quasar", "Radio")):
        return "agn"
    if "Wind" in name or "SN" in name:
        return "wind"
    return "other"


FAMCOL = {"wind": COLORS["bind"], "agn": COLORS["highlight"],
          "other": "#999999"}

names35 = list(pd.read_csv(
    "/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv")["ParamName"])
tbp = np.load(TB / "twobound_params.npy")
pairs = {}
for i in range(30):
    dcol = np.where(tbp[2 * i] != tbp[2 * i + 1])[0]
    assert len(dcol) == 1
    rr = [2 * i, 2 * i + 1]
    vv = [float(tbp[r, int(dcol[0])]) for r in rr]
    o = np.argsort(vv)
    pairs[names35[int(dcol[0])]] = [rr[k] for k in o]

P = {r: np.load(TB / f"run_{r:04d}/paired_stats.npz") for r in range(60)}
NG = {r: np.load(TB / f"run_{r:04d}/nongaussian_stats.npz") for r in range(60)}
ell = P[0]["ell"]
pdf_bins = NG[0]["pdf_bins"]


def logbin(y):
    ib = np.digitize(ell, EDGES) - 1
    L, Y = [], []
    for b in range(len(EDGES) - 1):
        s = ib == b
        if s.any():
            L.append(np.exp(np.log(ell[s]).mean()))
            Y.append(np.nanmean(y[s]))
    return np.array(L), np.array(Y)


def get(nm, stat):
    """(x, Delta, err_or_None) for one parameter and statistic."""
    rlo, rhi = pairs[nm]
    if stat == "clk":
        x, d = logbin(P[rhi]["clk_resp"][ZI] - P[rlo]["clk_resp"][ZI])
        _, e = logbin(np.sqrt(P[rhi]["clk_err"][ZI] ** 2
                              + P[rlo]["clk_err"][ZI] ** 2))
        nb = np.diff(np.searchsorted(ell, EDGES)).clip(1)[:len(x)]
        return x, d, e / np.sqrt(nb)
    if stat == "pdf":
        d = NG[rhi]["pdf"][ZI] - NG[rlo]["pdf"][ZI]
        return pdf_bins, d, None
    raise KeyError(stat)


def snr_of(nm, stat):
    """peak-bin |Delta|/err; pdf uses the V0 response as its S/N proxy
    (identical to imf_shape_clusters.py -- no paired pdf errors stored)."""
    if stat == "pdf":
        rlo, rhi = pairs[nm]
        d = P[rhi]["V0_resp"][ZI] - P[rlo]["V0_resp"][ZI]
        e = np.sqrt(P[rhi]["V0_err"][ZI] ** 2 + P[rlo]["V0_err"][ZI] ** 2)
    else:
        _, d, e = get(nm, stat)
    k = int(np.nanargmax(np.abs(d)))
    return float(np.abs(d[k]) / e[k])


for stat, slab, xlab, logx, outname in [
        ("clk", r"$\Delta C_\ell^{\kappa\kappa}$", r"$\ell$", True,
         "figs_v2/pfig_s3b_cl_clusters"),
        ("pdf", r"$\Delta$ $\kappa$ PDF", r"$\kappa/\sigma_\kappa$", False,
         "figs_v2/pfig_s3b_pdf_clusters")]:
    CUR, excl = {}, []
    for nm in pairs:
        if snr_of(nm, stat) < SNR_MIN:
            excl.append(nm)
            continue
        x, d, _ = get(nm, stat)
        CUR[nm] = d / d[np.nanargmax(np.abs(d))]
    NAMES = list(CUR)
    M = np.corrcoef([CUR[n] for n in NAMES])
    Z = linkage(squareform(np.clip(1 - M, 0, None), checks=False),
                method="average")
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

    print(f"\n=== {stat} (S/N>={SNR_MIN:g}: {len(NAMES)}/30 params; "
          f"excluded: {', '.join(short_label(n) for n in excl) or 'none'}) ===")
    for k, (mem, rbar) in enumerate(multi):
        star = "  <-- IMFslope" if "IMFslope" in mem else ""
        print(f"  C{k+1} (n={len(mem)}, rbar={rbar:.2f}): "
              + ", ".join(short_label(m) for m in mem) + star)
    if single:
        print(f"  singletons: {', '.join(short_label(m) for m in single)}")

    npan = len(multi) + (1 if single else 0)
    ncol = 3
    nrow = max(1, int(np.ceil(npan / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TWO_COL[0], 2.3 * nrow),
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
        if logx:
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
            axes[k].set_ylabel(f"{slab} / peak", fontsize=6.5)
        if k >= npan - ncol:
            axes[k].set_xlabel(xlab, fontsize=7)
    fig.tight_layout()
    save(fig, outname)
    plt.close(fig)
    print(f"  wrote {outname}.pdf")
