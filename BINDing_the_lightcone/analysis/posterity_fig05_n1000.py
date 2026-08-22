"""Posterity: the fig-5 field-level validation at N = 1000 realizations.

Built from the bind_n1000 campaign's stream caches, which store PER-REALIZATION
draws of every statistic (cl 1000x5x724; peaks/minima 1000x5x68; V0-V2
1000x5x29; pdf 1000x5x41) for the bind, truth and dmo arms on the shared
1992+7r seed ladder.

Two things make this a posterity figure rather than a paper figure:

  * the BIND arm is twobound/run_0018: an independent paint replica at the
    fiducial parameters with the CORRECT TNG300 conditioning, traced to 1000
    realizations on the shared 1992+7r ladder (per-real stats from
    stream_stats_tb18.py, whose MFs are computed directly on the extended
    nu grid). The truth/dmo arms are the campaign's originals.
  * the nu statistics follow the paper's canonical nu05 convention
    (papers/01_pipeline/nu_grid.py): ONE 22-centre grid -2.75..7.75
    (0.5-wide bins) for all six, peaks/minima at 2' with per-map norm,
    MFs at 1' evaluated at the centres, and the PDF of the UNSMOOTHED map
    in its own S/N units -- recomputed per realization from the n1000
    kappa cubes by n1000_nu05_stream.py. Residual panels carry the
    LSST-Y10 precision band: the truth arm's 1000-realization scatter
    area-scaled to f_sky = 0.44, as a percent of the truth mean.

Writes figs_preview/fig05_field_validation_n1000.{png,pdf}.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python posterity_fig05_n1000.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
AN = CEPH / "bind_n1000/analysis"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figs_preview"
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

ZI = 1
ELL_LO, ELL_MAX = 100.0, 3.0e4
CB, CH, CG = COLORS["bind"], COLORS["truth"], COLORS["secondary"]
ZCOLS = [plt.get_cmap("plasma")(v) for v in (0.02, 0.25, 0.5, 0.7, 0.88)]

S_ = {"bind": np.load(HERE / "stream_stats_twobound_run_0018.npz"),
      "truth": np.load(AN / "stream_stats_truth_run_0000.npz"),
      "dmo": np.load(AN / "stream_stats_dmo_run_0000.npz")}
# pair to the shortest arm (the tb18 stream may be mid-checkpoint)
N = min(int(S_[a]["cl"].shape[0]) for a in S_)
S_ = {a: {k: (np.asarray(f[k], float)[:N] if f[k].ndim >= 2 else np.asarray(f[k]))
          for k in f.files} for a, f in S_.items()}
ELL = np.asarray(S_["bind"]["ell"], float)
ZS = [0.5, 1.0, 1.5, 2.0, 2.44]

# nu-domain statistics in the paper's nu05 convention (n1000_nu05_stream.py)
_n5 = np.load(HERE / "n1000_nu05_stats.npz")
for _a in ("bind", "truth"):
    assert int(_n5[f"n_done_{_a}"]) >= N, f"nu05 stream incomplete for {_a}"
NU = np.asarray(_n5["nu"], float)
NU05_B = {k: np.asarray(_n5[f"bind_{k}"], float)[:N] for k in
          ("pdf", "peaks", "minima", "v0", "v1", "v2")}
NU05_T = {k: np.asarray(_n5[f"truth_{k}"], float)[:N] for k in
          ("pdf", "peaks", "minima", "v0", "v1", "v2")}
print(f"N = {N} paired realizations (bind arm = twobound/run_0018, correct fiducial)")

# LSST-Y10 precision bands from the 1000-map covariance: the per-bin scatter
# of the truth arm across its 1000 seed-paired realizations, scaled from the
# 25 deg^2 footprint to the LSST-Y10 area (f_sky = 0.44), as a percent of the
# truth mean -- the same construction as the paper's Stage-IV comparison.
LSST_SCALE = np.sqrt(25.0 / (0.44 * 41252.961))


def lsst_band(R):
    """percent band from per-real draws R (n_real, nbin), area-scaled."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * np.nanstd(R, 0) / np.abs(np.nanmean(R, 0)) * LSST_SCALE


mell = (ELL >= ELL_LO) & (ELL <= ELL_MAX)
f_ell = ELL * (ELL + 1) / (2 * np.pi)


def sem(R):
    return np.nanstd(R, 0) / np.sqrt(R.shape[0])


def paired_resid(bR, tR):
    """mean and +-1sigma/sqrt(N) of 100*(bind/truth - 1), per bin (paired)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 * (bR / np.where(np.abs(tR) > 0, tR, np.nan) - 1)
    return np.nanmean(r, 0), np.nanstd(r, 0) / np.sqrt(np.isfinite(r).sum(0).clip(1))


fig = plt.figure(figsize=(TWO_COL[0], 5.9))
gs = fig.add_gridspec(5, 4, height_ratios=[2.2, 1, 0.55, 2.2, 1], hspace=0.18, wspace=0.45)


def panels_axes(j):
    r0 = (j // 4) * 3
    a = fig.add_subplot(gs[r0, j % 4])
    ar = fig.add_subplot(gs[r0 + 1, j % 4], sharex=a)
    return a, ar


# ── (a) Cl^kk and (b) S(ell) ────────────────────────────────────────────────
cl_b, cl_t, cl_d = (np.asarray(S_[a]["cl"], float) for a in ("bind", "truth", "dmo"))
for j, tag in ((0, "cl"), (1, "S")):
    a, ar = panels_axes(j)
    for zi in range(5):
        y = (f_ell * cl_b[:, zi].mean(0)) if tag == "cl" else \
            (cl_b[:, zi] / cl_d[:, zi]).mean(0)
        a.plot(ELL[mell], y[mell], color=ZCOLS[zi], lw=1.0)
    yt = (f_ell * cl_t[:, ZI].mean(0)) if tag == "cl" else (cl_t[:, ZI] / cl_d[:, ZI]).mean(0)
    a.plot(ELL[mell], yt[mell], color=CH, ls="--", lw=0.9)
    # +-1sigma/sqrt(N) bands on the two z1 means (marginal for Cl, paired for S)
    if tag == "cl":
        sb, st_ = f_ell * sem(cl_b[:, ZI]), f_ell * sem(cl_t[:, ZI])
        y1 = f_ell * cl_b[:, ZI].mean(0)
    else:
        sb, st_ = sem(cl_b[:, ZI] / cl_d[:, ZI]), sem(cl_t[:, ZI] / cl_d[:, ZI])
        y1 = (cl_b[:, ZI] / cl_d[:, ZI]).mean(0)
    a.fill_between(ELL[mell], (y1 - sb)[mell], (y1 + sb)[mell], color=ZCOLS[ZI],
                   alpha=0.35, lw=0, zorder=1)
    a.fill_between(ELL[mell], (yt - st_)[mell], (yt + st_)[mell], color=CH,
                   alpha=0.30, lw=0, zorder=1)
    a.set_xscale("log")
    if tag == "cl":
        a.set_yscale("log")
        a.set_ylabel(r"$\ell(\ell{+}1)C_\ell^{\kappa\kappa}/2\pi$", fontsize=7)
        a.legend(handles=[plt.Line2D([], [], color=ZCOLS[zi], lw=1.0,
                                     label=rf"$z_s={ZS[zi]:.2f}$") for zi in range(5)],
                 loc="upper left", fontsize=4.4, ncol=2, handlelength=1.1,
                 columnspacing=0.7, labelspacing=0.25, borderpad=0.2, handletextpad=0.4)
    else:
        a.set_ylabel(r"$S(\ell)=C_\ell^{\rm bind}/C_\ell^{\rm dmo}$", fontsize=7)
        a.legend(handles=[plt.Line2D([], [], color="0.35", lw=1.0, label="BIND (fiducial)"),
                          plt.Line2D([], [], color=CH, lw=0.9, ls="--",
                                     label="hydro-pasted"),
                          plt.Rectangle((0, 0), 1, 1, fc=COLORS["dmo"], alpha=0.35,
                                        ec="none", label="LSST-Y10")],
                 loc="lower left", fontsize=5.0, handlelength=1.6)
    plt.setp(a.get_xticklabels(), visible=False)
    a.tick_params(labelsize=6)
    ar.axhline(0, color=COLORS["dmo"], lw=0.6)
    bl = lsst_band(cl_t[:, ZI])   # relative Cl error = relative S error
    ar.fill_between(ELL[mell], -bl[mell], bl[mell], color=COLORS["dmo"],
                    alpha=0.35, lw=0)
    res, band = paired_resid(cl_b[:, ZI], cl_t[:, ZI])
    ar.fill_between(ELL[mell], (res - band)[mell], (res + band)[mell],
                    color=CB, alpha=0.28, lw=0)
    ar.plot(ELL[mell], res[mell], color=CB, lw=0.9)
    ar.set_ylim(-8, 8)
    ar.set_xscale("log")
    ar.set_xlabel(r"$\ell$", fontsize=7)
    ar.set_ylabel("resid. [%]" if j % 4 == 0 else "", fontsize=7)
    ar.tick_params(labelsize=6)
    panel_label(a, f"({'ab'[j]})", loc="upper right")

# ── nu-domain panels (nu05 convention, one 22-centre grid for all six) ──────
NUP = [("pdf", NU, r"PDF$(\nu)$", r"$\nu$", "log"),
       ("peaks", NU, r"$N_{\rm pk}(\nu)$", r"$\nu$", "log"),
       ("minima", NU, r"$N_{\rm min}(\nu)$", r"$\nu$", "log"),
       ("v0", NU, r"$V_0(\nu)$", r"$\nu$", "log"),
       ("v1", NU, r"$V_1(\nu)$", r"$\nu$", "log"),
       ("v2", NU, r"$V_2(\nu)$", r"$\nu$", "symlog")]
for k, (key, xg, ylab, xlab, ysc) in enumerate(NUP):
    j = k + 2
    a, ar = panels_axes(j)
    Ball = NU05_B[key]
    T = NU05_T[key]
    B = Ball[:, ZI]
    bm, tm = B.mean(0), T.mean(0)
    for zi in range(5):
        a.plot(xg, Ball[:, zi].mean(0), color=ZCOLS[zi], lw=1.0)
    a.plot(xg, tm, color=CH, ls="--", lw=0.9)
    sb, st_ = sem(B), sem(T)
    a.fill_between(xg, bm - sb, bm + sb, color=ZCOLS[ZI], alpha=0.35, lw=0, zorder=1)
    a.fill_between(xg, tm - st_, tm + st_, color=CH, alpha=0.25, lw=0, zorder=1)
    if ysc == "symlog":
        a.set_yscale("symlog", linthresh=max(1e-5, 0.02 * np.nanmax(np.abs(tm))))
    else:
        a.set_yscale("log")
    a.set_ylabel(ylab, fontsize=7)
    plt.setp(a.get_xticklabels(), visible=False)
    a.tick_params(labelsize=6)
    ar.axhline(0, color=COLORS["dmo"], lw=0.6)
    good = np.abs(tm) > 1e-3 * np.nanmax(np.abs(tm))
    bl = lsst_band(T)
    _gb = good & np.isfinite(bl)
    ar.fill_between(xg[_gb], -bl[_gb], bl[_gb], color=COLORS["dmo"],
                    alpha=0.35, lw=0)
    res, band = paired_resid(np.where(good, B, np.nan), np.where(good, T, np.nan))
    ar.errorbar(xg[good], res[good], yerr=band[good], fmt="o", ms=2.0, mew=0,
                color=CB, lw=0.7, elinewidth=0.7, capsize=1.3)
    ar.set_ylim(-8, 8)
    ar.set_xlim(-3, 8)
    ar.set_xlabel(xlab, fontsize=7)
    ar.set_ylabel("resid. [%]" if j % 4 == 0 else "", fontsize=7)
    ar.tick_params(labelsize=6)
    panel_label(a, f"({'cdefgh'[k]})", loc="upper right")

save(fig, str(OUT / "fig05_field_validation_n1000"))
plt.close(fig)
print("wrote fig05_field_validation_n1000")

res, band = paired_resid(cl_b[:, ZI], cl_t[:, ZI])
tr = (ELL >= 300) & (ELL <= 5000)
print(f"  Cl^kk median |resid| 300<l<5000: {np.nanmedian(np.abs(res[tr])):.2f}%  "
      f"(paired band {np.nanmedian(band[tr]):.3f}% at N={N})")
for key in ("pdf", "peaks", "minima", "v0", "v1", "v2"):
    B = NU05_B[key][:, ZI]
    T = NU05_T[key]
    tm = T.mean(0)
    good = np.abs(tm) > 0.01 * np.nanmax(np.abs(tm))
    r, b = paired_resid(np.where(good, B, np.nan), np.where(good, T, np.nan))
    print(f"  {key:7s} median |resid| = {np.nanmedian(np.abs(r)):.2f}%  "
          f"(band {np.nanmedian(b):.3f}%)")
