"""fidswap re-render: paper Fig 6 (spectra validation) == notebook fig04b_spectra.

Cell body copied verbatim from _build_figures_nb.py (including the live per-plane
C^yy / C^tautau / C^ytau recompute, the cross guard, the measured wide-survey
bands and every print).  Fiducial inlets re-pointed, and ONLY those:

  RB   = SCI/'runs/bind/run_0000'   -> FID_RUN (twobound/run_0049)
  BTAU = LC/'tau_maps.npz'          -> FID_RUN/'tau_maps.npz'
        (the replica carries its OWN seed-paired tau cube, so the fiducial tau
         leg stops being borrowed from a different run dir)
  fig-4's paired legs               -> referee_work/fidswap field cache
Truth-side inlets are untouched; runs/truth/run_0000/tau_maps.npz exists, so the
tau-family panels (c, e, f) render as CLOSURE panels, not Sobol-spread panels.
"""
from __future__ import annotations

import gc
import os

import numpy as np
from fsfig_common import (
    CAMPAIGN,
    COLORS,
    ELL_LO,
    ELL_MAX_PLOT,
    ELL_TRUST,
    FID_FC,
    FID_RUN,
    OLD_FC,
    REPLICA,
    RT,
    TWO_COL,
    ZI,
    ZS,
    dataset,
    ell_trust_marker,
    load_maps_prefix,
    panel_label,
    plt,
    rebin,
    save_imgs,
    tee,
)
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

tee("fig06_spectra")
from bind.inference.stats import power_spectrum  # noqa: E402

d = dataset()
_fc_raw = np.load(FID_FC)
# NR drives the panels that RE-COMPUTE spectra from the map cubes; the cached
# panels (kk/yy/ky/tt/kt) always use every realization in the cache.  At n1000
# an uncapped NR would recompute ~30k spectra and hold ~8 x 21 GB of maps, so
# cap it (BIND_FIG_NR) -- the cap only widens those panels' error bars, it does
# not bias them.
NR_CACHE = int(_fc_raw["kk_bind"].shape[0])
# Under n1000 nothing is recomputed from map cubes (every leg, including y-tau,
# comes from a per-realization cache), so the full cache is always used.
NR = (NR_CACHE if CAMPAIGN == "n1000"
      else min(NR_CACHE, int(os.environ.get("BIND_FIG_NR", "50"))))
# The figure's internal guards compare a cached leg against the same quantity
# RE-COMPUTED from the map cubes, so both must average the SAME realizations.
# Slicing the cache to the first NR keeps that identity exact; raising
# BIND_FIG_NR (up to the full cache) tightens every band at the cost of
# recomputing more spectra from maps.
_fc = {k: (v[:NR] if getattr(v, "ndim", 0) == 3 and v.shape[0] == NR_CACHE else v)
       for k, v in ((k, _fc_raw[k]) for k in _fc_raw.files)}
if NR < NR_CACHE:
    print(f"using NR={NR} of {NR_CACHE} cached realizations for every leg "
          f"(cache sliced to match the map recompute; set BIND_FIG_NR to change)")
ell_f = _fc["ell"]
f_ell = ell_f * (ell_f + 1) / (2 * np.pi)
mt = (ell_f >= ELL_LO) & (ell_f <= ELL_TRUST)
cB, cH = COLORS["bind"], COLORS["truth"]
ZCOLS = plt.cm.plasma(np.linspace(0.02, 0.82, 5))


def paired(bR, tR):
    n = min(bR.shape[0], tR.shape[0])
    rat = bR / np.where(tR > 0, tR, np.nan)
    res = 100 * (np.nanmean(bR, 0) / np.nanmean(tR, 0) - 1)
    band = 100 * np.nanstd(rat, 0) / np.sqrt(n)
    chi2 = np.nanmean((res[mt] / np.where(band[mt] > 0, band[mt], np.nan)) ** 2)
    return np.nanmean(bR, 0), np.nanmean(tR, 0), res, band, chi2


kk_b, kk_t = _fc["kk_bind"][:, ZI], _fc["kk_truth"][:, ZI]
ky_b, ky_t = _fc["ky_bind"][:, ZI], _fc["ky_truth"][:, ZI]
yy_b, yy_t = _fc["yy_bind"][:, -1], _fc["yy_truth"][:, -1]
KKb, KKt, res_kk, band_kk, x2_kk = paired(kk_b, kk_t)
KYb, KYt, res_ky, band_ky, x2_ky = paired(ky_b, ky_t)
YYb, YYt, res_yy, band_yy, x2_yy = paired(yy_b, yy_t)
b_cl0, t_cl0 = np.load(FID_RUN / "Cl_kappa.npz"), np.load(RT / "Cl_kappa.npz")
KK5b = np.array([b_cl0["cl"][zi, zi] for zi in range(5)], dtype=float)
KK5b[ZI] = KKb

# fig-4's LSST-Y10 kappa band (v2 recipe), recomputed here so panel (a) matches
A2SR = (np.pi / 180 / 60) ** 2
logcl_b_rb = rebin(np.log(np.where(kk_b > 0, kk_b, np.nan)), 8)
ell_f_rb = rebin(ell_f[None, :], 8)[0]
AREA_25_SR, AREA_LSST_SR = 25.0 * (np.pi / 180.0) ** 2, 0.44 * 4 * np.pi
sig_meas = np.interp(ell_f, ell_f_rb,
                     np.nanstd(logcl_b_rb, axis=0) * np.sqrt(AREA_25_SR / AREA_LSST_SR))


def _relerr(cl, ngal, sige, fsky, dlnl=0.15):
    Nl = sige ** 2 * A2SR / ngal
    dl = np.maximum(ell_f * dlnl, 1.0)
    return np.sqrt(2.0 / ((2 * ell_f + 1) * dl * fsky)) * (1 + Nl / cl)


def _cvonly(fsky, dlnl=0.15):
    dl = np.maximum(ell_f * dlnl, 1.0)
    return np.sqrt(2.0 / ((2 * ell_f + 1) * dl * fsky))


lsst_kk = 100 * np.sqrt(sig_meas ** 2 + (_relerr(KKt, 27.0, 0.26, 0.44) - _cvonly(0.44)) ** 2)

# ── the cell body ────────────────────────────────────────────────────────────
RT_TAU_PATH = RT / "tau_maps.npz"
HAS_TRUTH_TAU = RT_TAU_PATH.exists()
print(f"tau-family truth guard: {RT_TAU_PATH} exists? {HAS_TRUTH_TAU}")

_ck = d["t__cl_kappa__value"][:, ZI, ZI, :]
_sp = d["t__suppression__value"][:, ZI, :]
_dmo_ref = np.nanmedian(_ck / np.where(_sp > 0, _sp, np.nan), 0)
print("identity check: Cl_kappa == S(ell) x Cl_dmo -> allclose = "
      f"{np.allclose(_ck, _sp*_dmo_ref, rtol=1e-6)}")

NELL = len(ell_f)

if CAMPAIGN == "n1000":
    # ── every leg from the compute-once caches; zero map streaming ───────────
    # yy/tt/ky/kt per-realization legs live in the n1000 field cache
    # (build_field_cache_n1000.py); the y-tau cross lives in the yt cache
    # (build_yt_cache_n1000.py).  The cross-checks below compare every cached
    # mean against the INDEPENDENT per-run products of the MPI stats CLI
    # (Cl_kappa_y.npz / Cl_tau.npz), i.e. two separate code paths over the
    # same 1000 realizations.
    from pathlib import Path as _Path
    _YT_DIR = _Path("/mnt/home/mlee1/ceph/bind_n1000/field_cache")
    _ytb_p = _YT_DIR / f"yt_stats_bind_{REPLICA}_n1000.npz"
    _ytt_p = _YT_DIR / "yt_stats_truth_n1000.npz"
    for _p in (_ytb_p, _ytt_p):
        if not _p.exists():
            raise SystemExit(
                f"missing {_p}\nbuild it once with:\n"
                "  python papers/01_pipeline/build_yt_cache_n1000.py --side bind\n"
                "  python papers/01_pipeline/build_yt_cache_n1000.py --side truth")
    with np.load(_ytb_p) as _z:
        assert int(_z["n_done"]) == NR_CACHE, f"{_ytb_p.name} incomplete: {int(_z['n_done'])}/{NR_CACHE}"
        ytR5b = np.asarray(_z["yt_real"], float)
    with np.load(_ytt_p) as _z:
        assert int(_z["n_done"]) == NR_CACHE, f"{_ytt_p.name} incomplete: {int(_z['n_done'])}/{NR_CACHE}"
        ytR5t = np.asarray(_z["yt_real"], float)

    YY5b, YY5t = _fc["yy_bind"].mean(0), _fc["yy_truth"].mean(0)
    TT5b, TT5t = _fc["tt_bind"].mean(0), _fc["tt_truth"].mean(0)
    KY5b, KY5t = _fc["ky_bind"].mean(0), _fc["ky_truth"].mean(0)
    KT5b, KT5t = _fc["kt_bind"].mean(0), _fc["kt_truth"].mean(0)
    YT5b, YT5t = ytR5b.mean(0), ytR5t.mean(0)

    YYb_z, YYt_z, res_yy_z, band_yy_z, x2_yy_z = paired(_fc["yy_bind"][:, ZI],
                                                        _fc["yy_truth"][:, ZI])
    TTb_z, TTt_z, res_tt_z, band_tt_z, x2_tt_z = paired(_fc["tt_bind"][:, ZI],
                                                        _fc["tt_truth"][:, ZI])
    KTb_z, KTt_z, res_kt_z, band_kt_z, x2_kt_z = paired(_fc["kt_bind"][:, ZI],
                                                        _fc["kt_truth"][:, ZI])
    YTb_z, YTt_z, res_yt_z, band_yt_z, x2_yt_z = paired(ytR5b[:, ZI], ytR5t[:, ZI])
    yyR_z, ttR_z = _fc["yy_bind"][:, ZI], _fc["tt_bind"][:, ZI]
    ytR_z, ktb_z = ytR5b[:, ZI], _fc["kt_bind"][:, ZI]

    # cache <-> MPI-stats cross-validation (independent estimator runs)
    _bky = np.load(FID_RUN / "Cl_kappa_y.npz")
    _bkt = np.load(FID_RUN / "Cl_tau.npz")
    _tky = np.load(RT / "Cl_kappa_y.npz")
    _tkt = np.load(RT / "Cl_tau.npz")
    for _lab, _cache, _mpi in (
            ("ky bind", KY5b, _bky["cl_ky"]), ("ky truth", KY5t, _tky["cl_ky"]),
            ("kt bind", KT5b, _bkt["cl_kt"]), ("kt truth", KT5t, _tkt["cl_kt"]),
            ("yy bind (total)", YY5b[-1], _bky["cl_yy"]),
            ("yy truth (total)", YY5t[-1], _tky["cl_yy"]),
            ("tt bind (total)", TT5b[-1], _bkt["cl_tt"]),
            ("tt truth (total)", TT5t[-1], _tkt["cl_tt"]),
            ("yt bind (total)", YT5b[-1], _bkt["cl_yt"]),
            ("yt truth (total)", YT5t[-1], _tkt["cl_yt"])):
        _mpi = np.asarray(_mpi, float)
        _dd = float(np.nanmax(np.abs(np.asarray(_cache) / np.where(_mpi != 0, _mpi, np.nan) - 1)))
        print(f"cache-vs-MPI guard C^{_lab}: max |ratio-1| = {_dd:.2e}")
        assert _dd < 1e-3, f"C^{_lab}: field/yt cache disagrees with the per-run stats product"
else:
    YY5b, YY5t = np.empty((5, NELL)), np.empty((5, NELL))
    TT5b, YT5b = np.empty((5, NELL)), np.empty((5, NELL))
    TT5t, YT5t = np.full((5, NELL), np.nan), np.full((5, NELL), np.nan)

    BY = load_maps_prefix(FID_RUN / "y_maps.npz", "y", NR)
    TY = load_maps_prefix(RT / "y_maps.npz", "y", NR)
    BTAU = load_maps_prefix(FID_RUN / "tau_maps.npz", "tau", NR)
    if HAS_TRUTH_TAU:
        TTAU = load_maps_prefix(RT_TAU_PATH, "tau", NR)
    print("map cubes streamed")

    for zi in range(5):
        cb = np.array([power_spectrum(BY[r, zi])[1] for r in range(NR)])
        ct = np.array([power_spectrum(TY[r, zi])[1] for r in range(NR)])
        YY5b[zi], YY5t[zi] = cb.mean(0), ct.mean(0)
        if zi == ZI:
            YYb_z, YYt_z, res_yy_z, band_yy_z, x2_yy_z = paired(cb, ct)
            yyR_z = cb.copy()
        del cb, ct
        gc.collect()

        tb = np.array([power_spectrum(BTAU[r, zi])[1] for r in range(NR)])
        ytb = np.array([power_spectrum(BY[r, zi], BTAU[r, zi])[1] for r in range(NR)])
        TT5b[zi], YT5b[zi] = tb.mean(0), ytb.mean(0)
        if zi == ZI:
            ttR_z, ytR_z = tb.copy(), ytb.copy()
        if HAS_TRUTH_TAU:
            tt = np.array([power_spectrum(TTAU[r, zi])[1] for r in range(NR)])
            ytt = np.array([power_spectrum(TY[r, zi], TTAU[r, zi])[1] for r in range(NR)])
            TT5t[zi], YT5t[zi] = tt.mean(0), ytt.mean(0)
            if zi == ZI:
                TTb_z, TTt_z, res_tt_z, band_tt_z, x2_tt_z = paired(tb, tt)
                YTb_z, YTt_z, res_yt_z, band_yt_z, x2_yt_z = paired(ytb, ytt)
            del tt, ytt
            gc.collect()
        del tb, ytb
        gc.collect()
        print(f"  plane {zi} done")

    BT_tot = BTAU[:, -1].copy()
    if HAS_TRUTH_TAU:
        TT_tot = TTAU[:, -1].copy()
        del TTAU
    del BY, TY, BTAU
    gc.collect()
    print("cumulative-column guard: per-plane C^yy at the LAST plane vs the total-column "
          f"C^yy: median |ratio-1| = "
          f"{100*np.nanmedian(np.abs(YY5b[-1]/np.where(YYb > 0, YYb, np.nan) - 1)):.3f}%")

    def _cross5(K, F):
        return np.array([np.mean([power_spectrum(K[r, zi], F[r])[1] for r in range(NR)], 0)
                         for zi in range(5)])

    BK5 = load_maps_prefix(FID_RUN / "kappa_maps.npz", "kappa", NR)
    BY_tot = load_maps_prefix(FID_RUN / "y_maps.npz", "y", NR, -1)
    KY5b = _cross5(BK5, BY_tot)
    del BY_tot
    gc.collect()
    KT5b = _cross5(BK5, BT_tot)
    gc.collect()

    TK5 = load_maps_prefix(RT / "kappa_maps.npz", "kappa", NR)
    TY_tot = load_maps_prefix(RT / "y_maps.npz", "y", NR, -1)
    KY5t = _cross5(TK5, TY_tot)
    del TY_tot
    gc.collect()

    if HAS_TRUTH_TAU:
        KT5t = _cross5(TK5, TT_tot)
        ktb_z = np.array([power_spectrum(BK5[r, ZI], BT_tot[r])[1] for r in range(NR)])
        ktt_z = np.array([power_spectrum(TK5[r, ZI], TT_tot[r])[1] for r in range(NR)])
        KTb_z, KTt_z, res_kt_z, band_kt_z, x2_kt_z = paired(ktb_z, ktt_z)
        del ktt_z
        gc.collect()

    del BK5, TK5, BT_tot
    gc.collect()
    if HAS_TRUTH_TAU:
        del TT_tot
        gc.collect()

    for _lab, _a, _b in (("bind", KY5b[ZI], KYb), ("truth", KY5t[ZI], KYt)):
        _dd = float(np.nanmax(np.abs(_a / np.where(_b != 0, _b, np.nan) - 1)))
        print(f"cross guard C^ky {_lab} z_s=1 (5-plane vs field-cache paired leg): "
              f"max |ratio-1| = {_dd:.2e}")
        assert _dd < 1e-6, f"C^ky {_lab} z_s=1 disagrees with the paired leg"

CV_FSKY = 0.44


def cv_floor(fsky=CV_FSKY, dlnl=0.15):
    dl = np.maximum(ell_f * dlnl, 1.0)
    return 100 * np.sqrt(2.0 / ((2 * ell_f + 1) * dl * fsky))


_A25_SR, _AWIDE_SR = 25.0 * (np.pi / 180.0) ** 2, CV_FSKY * 4 * np.pi
_ell_rb4b = rebin(ell_f[None, :], 8)[0]


def measured_wide_band(clR):
    rb = rebin(clR, 8)
    rel = np.nanstd(rb, 0) / np.abs(np.nanmean(rb, 0))
    return 100 * np.interp(ell_f, _ell_rb4b, rel) * np.sqrt(_A25_SR / _AWIDE_SR)


band_meas = {"yy": measured_wide_band(yyR_z), "ky": measured_wide_band(ky_b),
             "tt": measured_wide_band(ttR_z), "yt": measured_wide_band(ytR_z)}
if HAS_TRUTH_TAU:
    band_meas["kt"] = measured_wide_band(ktb_z)

ELL_TT = d["a__cl_tt__ell"]


def sobol_spread(V):
    med = np.nanmedian(V, 0)
    r = 100 * (V / np.where(med > 0, med, np.nan) - 1)
    return np.nanpercentile(r, 16, 0), np.nanpercentile(r, 84, 0)


lo_tt, hi_tt = sobol_spread(d["t__cl_tt__value"])
lo_kt, hi_kt = sobol_spread(d["t__cl_kappa_tau__value"][:, ZI])
lo_yt, hi_yt = sobol_spread(d["t__cl_yt__value"])

kk_panel = dict(x=ell_f, yb_z=f_ell * KK5b, yt=f_ell * KKt, res=res_kk, band=band_kk,
                lsst=lsst_kk, yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\kappa}/2\pi$",
                tag="(a)", rlim=25)
yy_panel = dict(x=ell_f, yb_z=f_ell * YY5b, yt=f_ell * YYt_z, res=res_yy_z, band=band_yy_z,
                lsst=band_meas["yy"], yl=r"$\ell(\ell{+}1)C_\ell^{yy}/2\pi$", tag="(b)", rlim=100)
ky_panel = dict(x=ell_f, yb_z=f_ell * KY5b, yt=f_ell * KYt, res=res_ky, band=band_ky,
                lsst=band_meas["ky"], yl=r"$\ell(\ell{+}1)C_\ell^{\kappa y}/2\pi$",
                tag="(d)", rlim=50)
tt_panel = dict(x=ell_f, yb_z=f_ell * TT5b, yt=f_ell * TTt_z, res=res_tt_z, band=band_tt_z,
                lsst=band_meas["tt"], yl=r"$\ell(\ell{+}1)C_\ell^{\tau\tau}/2\pi$",
                tag="(c)", rlim=100)
kt_panel = dict(x=ell_f, yb_z=f_ell * KT5b, yt=f_ell * KTt_z, res=res_kt_z, band=band_kt_z,
                lsst=band_meas["kt"], yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\tau}/2\pi$",
                tag="(e)", rlim=50)
yt_panel = dict(x=ell_f, yb_z=f_ell * YT5b, yt=f_ell * YTt_z, res=res_yt_z, band=band_yt_z,
                lsst=band_meas["yt"], yl=r"$\ell(\ell{+}1)C_\ell^{y\tau}/2\pi$",
                tag="(f)", rlim=100)
panels4b = [kk_panel, yy_panel, tt_panel, ky_panel, kt_panel, yt_panel]

fig = plt.figure(figsize=(TWO_COL[0], 6.9))
gs = fig.add_gridspec(6, 3, height_ratios=[2.2, 1, 0.55, 2.2, 1, 0.9], hspace=0.18, wspace=0.42)
for j, p in enumerate(panels4b):
    r0 = (j // 3) * 3
    a = fig.add_subplot(gs[r0, j % 3])
    ar = fig.add_subplot(gs[r0 + 1, j % 3], sharex=a)
    for zi in range(5):
        a.plot(p["x"], p["yb_z"][zi], color=ZCOLS[zi], lw=1.0)
    if p["yt"] is not None:
        a.plot(p["x"], p["yt"], color=cH, ls="--", lw=0.9)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylabel(p["yl"], fontsize=7)
    a.set_xlim(ELL_LO, ELL_MAX_PLOT)
    ell_trust_marker(a, label=False)
    a.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
    a.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(3, 0.4)))
    plt.setp(a.get_xticklabels(), visible=False)
    a.tick_params(labelsize=6)

    ar.axhline(0, color=COLORS["dmo"], lw=0.6)
    ar.plot(p["x"], p["res"], color=cB, lw=0.9)
    ar.set_ylabel("resid. [%]" if j % 3 == 0 else "", fontsize=6)
    ar.set_ylim(-50, 50)   # shared across all six residual panels
    ar.set_xscale("log")
    ar.set_xlabel(r"$\ell$", fontsize=7)
    ar.tick_params(labelsize=6)
    panel_label(a, p["tag"], loc="upper right")

alg = fig.add_subplot(gs[5, :])
alg.axis("off")
_leg = [plt.Line2D([], [], color=ZCOLS[zi], lw=1.0, label=rf"$z_s={ZS[zi]:.2f}$")
        for zi in range(5)]
_leg += [plt.Line2D([], [], color=cH, lw=0.9, ls="--", label=r"hydro-pasted, $z_s=1$")]
alg.legend(handles=_leg, loc="center", ncol=4, fontsize=5.4, handlelength=1.6,
           labelspacing=0.5, columnspacing=1.4)
save_imgs(fig, "fig06_spectra_validation")
plt.close(fig)

mb = (ell_f >= 300) & (ell_f <= 5000)
print("fig 6 median |resid| in ell=300-5000:  "
      f"Cl_kk {np.nanmedian(np.abs(res_kk[mb])):.1f}%  Cl_ky {np.nanmedian(np.abs(res_ky[mb])):.1f}%  "
      f"Cl_yy(total column) {np.nanmedian(np.abs(res_yy[mb])):.1f}%  "
      f"Cl_yy(z_s=1 column) {np.nanmedian(np.abs(res_yy_z[mb])):.1f}%")
print(f"chi2/dof (seed-paired, ell 100-{ELL_TRUST:.0f}): kk {x2_kk:.0f}  ky {x2_ky:.0f}  "
      f"yy(total) {x2_yy:.0f}  yy(z_s=1) {x2_yy_z:.0f}  tt {x2_tt_z:.0f}  kt {x2_kt_z:.0f}  "
      f"yt {x2_yt_z:.0f}")
_cvk = cv_floor()
print("wide-survey MEASURED sample-variance bands, median % in the trusted range "
      "[x Knox floor]: " + "  ".join(
          f"{k} {np.nanmedian(band_meas[k][mt]):.2f}% [x{np.nanmedian(band_meas[k][mt]/_cvk[mt]):.1f}]"
          for k in band_meas))
i1k = int(np.argmin(np.abs(ell_f - 1000.0)))
print("tau-family spectra at ell~1000: l(l+1)C/2pi = "
      f"{f_ell[i1k]*TT5b[ZI][i1k]:.3e} (tt BIND, column to z=1)  "
      f"{f_ell[i1k]*TT5b[-1][i1k]:.3e} (tt BIND, total)  "
      f"{f_ell[i1k]*KT5b[ZI][i1k]:.3e} (kt BIND, z_s=1)  "
      f"{f_ell[i1k]*YT5b[ZI][i1k]:.3e} (yt BIND, column to z=1)")
print("tau-family truth-validated closure: median |resid| ell=300-5000: "
      f"tt {np.nanmedian(np.abs(res_tt_z[mb])):.1f}%  kt {np.nanmedian(np.abs(res_kt_z[mb])):.1f}%  "
      f"yt {np.nanmedian(np.abs(res_yt_z[mb])):.1f}%")
print("SIGNED tau/y closure (BIND/truth, mean over ell=300-1000 and 300-5000):")
m310 = (ell_f >= 300) & (ell_f <= 1000)
for lab, B, T in (("tt", TTb_z, TTt_z), ("kt", KTb_z, KTt_z), ("yt", YTb_z, YTt_z),
                  ("yy(z_s=1)", YYb_z, YYt_z), ("yy(total)", YYb, YYt),
                  ("ky", KYb, KYt), ("kk", KKb, KKt)):
    print(f"  {lab:10s} 300-1000: {np.nanmean(B[m310]/T[m310]):.4f}   "
          f"300-5000: {np.nanmean(B[mb]/T[mb]):.4f}")

# ── before/after: the same tau/y legs computed on the RETIRED fiducial ────────
if CAMPAIGN == "n1000":
    # The retired fiducial's caches are 50-real products; pairing them against
    # 1000-real NEW legs is the mixed-N comparison this package guards against
    # (same rule as fig05/fig10).
    np.savez_compressed(
        "/mnt/home/mlee1/ceph/bind_n1000/analysis/fig06_numbers_n1000.npz",
        ell=ell_f, TT5b=TT5b, TT5t=TT5t, YT5b=YT5b, YT5t=YT5t,
        KY5b=KY5b, KY5t=KY5t, KT5b=KT5b, KT5t=KT5t, YY5b=YY5b, YY5t=YY5t,
        res_tt=res_tt_z, res_kt=res_kt_z, res_yt=res_yt_z, res_yy_z=res_yy_z)
    print("\n(OLD vs NEW fiducial table skipped: BIND_CAMPAIGN=n1000 — the retired "
          "fiducial has no 1000-realization cache)")
    print("\nDONE fig06")
    raise SystemExit(0)

print("\n### OLD vs NEW fiducial, fig-6 legs")
_ofc = np.load(OLD_FC)
o_res_kk = 100 * (_ofc["kk_bind"][:, ZI].mean(0) / kk_t.mean(0) - 1)
o_res_ky = 100 * (_ofc["ky_bind"][:, ZI].mean(0) / ky_t.mean(0) - 1)
o_res_yy = 100 * (_ofc["yy_bind"][:, -1].mean(0) / yy_t.mean(0) - 1)
o_tt = _ofc["tt_bind"][:, ZI].mean(0)
o_kt = _ofc["kt_bind"][:, ZI].mean(0)
print(f"{'leg':12s} {'OLD':>10s} {'NEW':>10s}   (median |BIND/truth-1| over ell 300-5000)")
print(f"{'kk':12s} {np.nanmedian(np.abs(o_res_kk[mb])):9.2f}% {np.nanmedian(np.abs(res_kk[mb])):9.2f}%")
print(f"{'ky':12s} {np.nanmedian(np.abs(o_res_ky[mb])):9.2f}% {np.nanmedian(np.abs(res_ky[mb])):9.2f}%")
print(f"{'yy(total)':12s} {np.nanmedian(np.abs(o_res_yy[mb])):9.2f}% {np.nanmedian(np.abs(res_yy[mb])):9.2f}%")
print(f"{'tt':12s} {100*np.nanmedian(np.abs(o_tt[mb]/TTt_z[mb]-1)):9.2f}% "
      f"{100*np.nanmedian(np.abs(TTb_z[mb]/TTt_z[mb]-1)):9.2f}%")
print(f"{'kt':12s} {100*np.nanmedian(np.abs(o_kt[mb]/KTt_z[mb]-1)):9.2f}% "
      f"{100*np.nanmedian(np.abs(KTb_z[mb]/KTt_z[mb]-1)):9.2f}%")
print("SIGNED, OLD vs NEW, mean BIND/truth over ell=300-1000 / 300-5000:")
print(f"  tt  OLD {np.nanmean(o_tt[m310]/TTt_z[m310]):.4f} / {np.nanmean(o_tt[mb]/TTt_z[mb]):.4f}"
      f"   NEW {np.nanmean(TTb_z[m310]/TTt_z[m310]):.4f} / {np.nanmean(TTb_z[mb]/TTt_z[mb]):.4f}")
print(f"  kt  OLD {np.nanmean(o_kt[m310]/KTt_z[m310]):.4f} / {np.nanmean(o_kt[mb]/KTt_z[mb]):.4f}"
      f"   NEW {np.nanmean(KTb_z[m310]/KTt_z[m310]):.4f} / {np.nanmean(KTb_z[mb]/KTt_z[mb]):.4f}")
print("C_l^tautau NEW/OLD (the (Omega_b/Omega_m)^2 = 0.9279 prediction): "
      f"{np.nanmean(TTb_z[mb]/o_tt[mb]):.4f} over ell 300-5000, "
      f"{np.nanmean(TTb_z[m310]/o_tt[m310]):.4f} over 300-1000")
np.savez_compressed("/mnt/home/mlee1/ceph/referee_work/fidswap/fig06_numbers.npz",
                    ell=ell_f, TT5b=TT5b, TT5t=TT5t, YT5b=YT5b, YT5t=YT5t,
                    KY5b=KY5b, KY5t=KY5t, KT5b=KT5b, KT5t=KT5t, YY5b=YY5b, YY5t=YY5t,
                    res_tt=res_tt_z, res_kt=res_kt_z, res_yt=res_yt_z,
                    res_yy_z=res_yy_z, o_tt=o_tt, o_kt=o_kt)
print("\nDONE fig06")
