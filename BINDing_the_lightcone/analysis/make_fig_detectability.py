"""Rebuild the section-4 opener with a self-consistent error budget.

Differences from the previous version of this figure, both of which the
referee report made necessary:

  * the shape-noise leg and the cosmic-variance leg are now evaluated in the
    SAME band-powers (Dl/l = 0.15).  Previously the measured leg was per
    native l bin and the Knox leg assumed Dl/l = 0.15, 54x wider.
  * panel (b) overlays the analytic model's own predictive scatter
    sigma_pred(l) on the survey precision (referee point II-9).

The denominator of S(l) is the seed-paired 50-realization DMO prefix, never
the shipped 550-realization mean (see _build_figures_nb.dmo_paired_cl).

Writes imgs/fig07_detectability.{png,pdf} and prints every number the caption
and the section prose quote.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
# BIND_CAMPAIGN=n1000 -> Sobol dataset assembled from the campaign runs, the
# fiducial/DMO spectra from the 1000-realization tree, output to imgs_1000.
CAMPAIGN = os.environ.get("BIND_CAMPAIGN", "sci50")
_N1K = CEPH / "bind_n1000"
HERE = Path(__file__).resolve().parent
IMGS = HERE.parent / ("imgs_1000" if CAMPAIGN == "n1000" else "imgs")
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline")
from family_model import FamilyModel  # noqa: E402

ZI, DL = 1, 0.15
A2SR = (np.pi / 180 / 60) ** 2
AREA_BOX = 25.0 * (np.pi / 180) ** 2
SURVEYS = (("LSST-Y10", 27.0, 0.26, 0.44, "#2F4B99", "--"),
           ("Euclid-like", 30.0, 0.30, 0.36, "#1E6E58", "-."))
ELL_MAX = 36864

d = np.load(_N1K / "emulator_dataset_n1000.npz" if CAMPAIGN == "n1000"
            else CEPH / "bind_sb35/emulator_dataset_nu05.npz", allow_pickle=True)
ELL = np.asarray(d["a__suppression__ell"], float)
S_all = d["t__suppression__value"][:, ZI, :]
if CAMPAIGN == "n1000":
    cl_fid = np.load(_N1K / "twobound/run_0049/Cl_kappa.npz")["cl"][ZI, ZI]
    # same 1000 realizations on one seed ladder -> plain DMO mean is the paired
    # denominator (the 50-real tree needs a prefix cache; the campaign does not)
    cl_dmo = np.load(_N1K / "dmo/run_0000/Cl_kappa.npz")["cl"][ZI, ZI]
else:
    cl_fid = np.load(CEPH / "bind_science/runs/twobound/run_0049/Cl_kappa.npz")["cl"][ZI, ZI]
    cl_dmo = np.load(CEPH / "bind_science/runs/dmo/run_0000/Cl_kappa_paired.npz")["cl_real"].mean(0)[ZI]
kk = np.load(CEPH / "bind_n1000/analysis/stream_stats_truth_run_0000.npz")["cl"][:, ZI, :]
N_COV = kk.shape[0]

edges = np.exp(np.arange(np.log(ELL[0]), np.log(52200) + DL, DL))
idx = np.digitize(ELL, edges) - 1
NB = idx.max() + 1


def rb(a):
    out = np.full(a.shape[:-1] + (NB,), np.nan)
    for b in range(NB):
        m = idx == b
        if m.sum():
            out[..., b] = np.nanmean(a[..., m], axis=-1)
    return out


ELLb, S_b, Sfid_b, kkb, clf_b = rb(ELL), rb(S_all), rb(cl_fid / cl_dmo), rb(kk), rb(cl_fid)
ok = np.isfinite(ELLb) & (ELLb <= ELL_MAX)
lo, hi = np.nanpercentile(S_b, [5, 95], axis=0)

SIG, Z, RAT = {}, {}, {}
for nm, ng, se, fs, _c, _ls in SURVEYS:
    cvk = np.sqrt(2.0 / ((2 * ELLb + 1) * np.maximum(ELLb * DL, 1.0) * fs))
    shot = cvk * (se ** 2 * A2SR / ng / clf_b)
    meas = np.nanstd(np.log(np.where(kkb > 0, kkb, np.nan)), 0) * np.sqrt(AREA_BOX / (fs * 4 * np.pi))
    SIG[nm] = np.sqrt(np.maximum(meas, cvk) ** 2 + shot ** 2)
    RAT[nm] = (hi - lo) / (2 * SIG[nm] * Sfid_b)
    Z[nm] = np.abs(S_b - Sfid_b) / (SIG[nm] * Sfid_b)

fm = FamilyModel()
xg, sp = fm.grid("clk"), fm.predictive_sigma("clk")

fig, axs = plt.subplots(2, 1, figsize=(3.6, 4.0), sharex=True,
                        gridspec_kw=dict(height_ratios=[2.2, 1.1], hspace=0.09))
ax = axs[0]
ax.fill_between(ELLb[ok], lo[ok], hi[ok], color="0.72", alpha=0.55, lw=0,
                label=f"Sobol 5–95% ({len(S_all)} nodes)", zorder=1)
for nm, ng, se, fs, c, ls in SURVEYS[::-1]:
    ax.fill_between(ELLb[ok], (Sfid_b * (1 - SIG[nm]))[ok], (Sfid_b * (1 + SIG[nm]))[ok],
                    color=c, alpha=0.9, lw=0, zorder=2 + (nm == "LSST-Y10"),
                    label=rf"{nm} $\pm1\sigma$")
ax.plot(ELLb[ok], Sfid_b[ok], color="k", lw=0.9, zorder=4, label="TNG fiducial")
ax.axhline(1, color="0.5", ls=":", lw=0.8, zorder=0)
ax.set_ylabel(r"$S(\ell)$ at $z_s=1$", fontsize=8)
ax.set_ylim(0.79, 1.46)
ax.legend(loc="upper left", fontsize=5.0, ncol=2, columnspacing=0.9, handlelength=1.7,
          frameon=False)
ax.text(0.02, 0.05, "(a)", transform=ax.transAxes, fontsize=7)

axz = axs[1]
for nm, ng, se, fs, c, ls in SURVEYS:
    p5, p50, p95 = np.nanpercentile(Z[nm], [5, 50, 95], axis=0)
    axz.fill_between(ELLb[ok], p5[ok], p95[ok], color=c, alpha=0.25, lw=0)
    axz.plot(ELLb[ok], p50[ok], color=c, lw=1.2, ls=ls, label=nm)
axz.axhline(1, color="k", ls="--", lw=0.8)
axz.axhline(5, color="k", ls=":", lw=0.6)
axz.set_yscale("log")
axz.set_ylim(0.02, 200)
axz.set_ylabel(r"$|\Delta S|/\sigma$", fontsize=8)
axz.legend(loc="upper left", fontsize=5.0, ncol=2, frameon=False)
axz.text(0.02, 0.08, "(b)", transform=axz.transAxes, fontsize=7)
axz.set_xscale("log")
axz.set_xlim(90, ELL_MAX)
axz.set_xlabel(r"$\ell$", fontsize=8)
for a in axs:
    a.tick_params(labelsize=6.5)
IMGS.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(IMGS / f"fig07_detectability.{ext}", dpi=300, bbox_inches="tight")
print(f"wrote {IMGS / 'fig07_detectability.png'}")

# ── the numbers the caption quotes ──────────────────────────────────────────
res = {"N_cov": int(N_COV), "dlnl": DL, "n_bands": int(ok.sum())}
for nm, *_ in SURVEYS:
    r, z = RAT[nm], Z[nm]
    g = ELLb[ok & (r > 10)]
    mtr = ok & (ELLb >= 300)
    pk = np.nanmax(z[:, mtr], axis=1)
    res[nm] = {"peak": round(float(np.nanmax(np.where(ok, r, np.nan))), 1),
               "peak_ell": int(ELLb[np.nanargmax(np.where(ok, r, -1))]),
               "window": [int(g.min()), int(g.max())] if g.size else None,
               "median_peak": round(float(np.median(pk)), 1),
               "pct_gt1": round(100 * float((pk > 1).mean()), 1),
               "pct_gt5": round(100 * float((pk > 5).mean()), 1)}
    print(f"  {nm}: peak {res[nm]['peak']} at ell={res[nm]['peak_ell']}, "
          f">10x over {res[nm]['window']}, {res[nm]['pct_gt5']}% of nodes >5sigma")
sig_on_grid = np.interp(xg, ELLb[ok], (SIG["LSST-Y10"] * Sfid_b)[ok])
cross = xg[sp > sig_on_grid]
res["sigma_pred_crossover_ell"] = int(cross.min()) if cross.size else None
res["sigma_pred_max_ratio"] = round(float(np.nanmax(sp / sig_on_grid)), 2)
print(f"  sigma_pred exceeds the LSST-Y10 sigma above ell={res['sigma_pred_crossover_ell']}, "
      f"peaking at {res['sigma_pred_max_ratio']}x")
# campaign-tagged so alternating sci50/n1000 runs cannot clobber each other's
# caption numbers (the untagged name was exactly that trap)
_tag = "_n1000" if CAMPAIGN == "n1000" else ""
(HERE / f"fig_detectability_numbers{_tag}.json").write_text(json.dumps(res, indent=1))
