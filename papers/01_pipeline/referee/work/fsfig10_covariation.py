"""fidswap re-render: paper Fig 10 (covariation) == notebook fig07_covariation.

Every one of the 12 panels is "Sobol curve / FIDUCIAL", so this is THE figure
where a partial swap would silently mix fiducials across panels.  All four
fiducial inlets are switched together:

  RB7  = SCI/'runs/bind/run_0000/Cl_kappa.npz'  -> FID_RUN/'Cl_kappa.npz'
  field_cache (ky/kt/yy/tt legs)                -> referee_work/fidswap field cache
  LC/'tau_maps.npz' (the live cl_yt leg)        -> FID_RUN/'tau_maps.npz'
  nu05_shards/sci_bind.npz (6 nu statistics)    -> referee_work/fidswap nu05 shard

The 253-node Sobol numerators, the STATS table, the Spearman ranking and IMPg
are unaffected by the swap and are rebuilt verbatim from _build_figures_nb.py.
"""
from __future__ import annotations

import gc
import warnings

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import rankdata

from fsfig_common import (CAMPAIGN, COLORS, ELL, ELL_MAX_PLOT, ELL_TRUST, FID_FC,
                          FID_NU05, FID_RUN, LC, OLD_FC, OLD_FID, OLD_NU05,
                          SB35, SCI, TWO_COL, ZI, dataset, dmo_paired_cl,
                          load_maps_prefix, panel_label, plt, save_imgs, tee)

tee("fig10_covariation")
import sys  # noqa: E402

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from param_labels import short_label  # noqa: E402
from bind.inference.stats import power_spectrum  # noqa: E402

d = dataset()
X_unit = d["X_unit"]
pnames = [str(s) for s in d["param_names"]]


def _stat(key):
    return np.asarray(d[key], float)


def rebin(a, k):
    n = (a.shape[-1] // k) * k
    return a[..., :n].reshape(*a.shape[:-1], n // k, k).mean(-1)


def rebin_nan(a, k):
    n = (a.shape[-1] // k) * k
    return np.nanmean(a[..., :n].reshape(*a.shape[:-1], n // k, k), -1)


def spearman(P, Y):
    rp = rankdata(P, axis=0).astype(float)
    ry = rankdata(Y, axis=0).astype(float)
    rp = (rp - rp.mean(0)) / rp.std(0)
    ry = (ry - ry.mean(0)) / np.maximum(ry.std(0), 1e-12)
    return rp.T @ ry / len(P)


def spearman_masked(P, Y, min_n=100):
    P = np.asarray(P, float)
    Y = np.asarray(Y, float)
    R = np.full((P.shape[1], Y.shape[1]), np.nan)
    fin = np.isfinite(Y)
    allrows = fin.all(0)
    if allrows.any():
        R[:, allrows] = spearman(P, Y[:, allrows])
    for j in np.where(~allrows)[0]:
        m = fin[:, j]
        if m.sum() >= min_n:
            R[:, j:j + 1] = spearman(P[m], Y[m, j][:, None])
    return R


STATS = [
    dict(k="suppression", lab=r"$S(\ell)$", A=_stat("t__suppression__value")[:, ZI, :],
         x=ELL, rb=16, grp="WL"),
    dict(k="pdf", lab=r"PDF$(\nu)$", A=_stat("t__pdf__value")[:, ZI, :],
         x=_stat("a__pdf__pdf_bins"), rb=1, grp="WL"),
    dict(k="peak_counts", lab=r"$N_{\rm pk}$", A=_stat("t__peak_counts__value")[:, ZI, :],
         x=_stat("a__peak_counts__nu"), rb=1, grp="WL"),
    dict(k="minima_counts", lab=r"$N_{\rm min}$",
         A=_stat("t__minima_counts__value")[:, ZI, :],
         x=_stat("a__minima_counts__nu"), rb=1, grp="WL"),
    dict(k="mf_v0", lab=r"$V_0$", A=_stat("t__mf_v0__value")[:, ZI, :],
         x=_stat("a__mf_v0__mf_nu"), rb=1, grp="WL"),
    dict(k="mf_v1", lab=r"$V_1$", A=_stat("t__mf_v1__value")[:, ZI, :],
         x=_stat("a__mf_v1__mf_nu"), rb=1, grp="WL"),
    dict(k="mf_v2", lab=r"$V_2$", A=_stat("t__mf_v2__value")[:, ZI, :],
         x=_stat("a__mf_v2__mf_nu"), rb=1, grp="WL"),
    dict(k="cl_yy", lab=r"$C_\ell^{yy}$", A=_stat("t__cl_yy__value"), x=ELL, rb=16, grp="auto"),
    dict(k="cl_tt", lab=r"$C_\ell^{\tau\tau}$", A=_stat("t__cl_tt__value"), x=ELL,
         rb=16, grp="auto"),
    dict(k="cl_kappa_y", lab=r"$C_\ell^{\kappa y}$",
         A=_stat("t__cl_kappa_y__value")[:, ZI, :], x=ELL, rb=16, grp="cross"),
    dict(k="cl_kappa_tau", lab=r"$C_\ell^{\kappa\tau}$",
         A=_stat("t__cl_kappa_tau__value")[:, ZI, :], x=ELL, rb=16, grp="cross"),
    dict(k="cl_yt", lab=r"$C_\ell^{y\tau}$", A=_stat("t__cl_yt__value"), x=ELL,
         rb=16, grp="cross"),
]
ELL_ROWS = {"suppression", "cl_yy", "cl_tt", "cl_kappa_y", "cl_kappa_tau", "cl_yt"}
LNYQ = ELL <= ELL_MAX_PLOT


def _ranked(s):
    return s["A"][..., LNYQ] if s["k"] in ELL_ROWS else s["A"]


Ys = {s["k"]: rebin_nan(_ranked(s), s["rb"]) for s in STATS}
Rs = {s["k"]: spearman_masked(X_unit, Ys[s["k"]]) for s in STATS}
IMPg = np.array([np.nanmax(np.abs(Rs[s["k"]]), 1) for s in STATS])


def color_curves(ax, x, Y, cvals, solid=None, drawn=None):
    for i in np.argsort(cvals):
        c = plt.cm.coolwarm(cvals[i])
        if solid is None:
            ax.plot(x, Y[i], lw=0.4, alpha=0.55, color=c)
            continue
        ax.plot(x, np.where(solid, Y[i], np.nan), lw=0.4, alpha=0.55, color=c)
        ax.plot(x, np.where(drawn & ~solid, Y[i], np.nan), lw=0.4, alpha=0.25, color=c)


def spread16_84(Y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(Y, 0)
        rat = Y / np.where(np.abs(med) > 0, med, np.nan)
        p16, p84 = np.nanpercentile(rat, [16, 84], axis=0)
        return float(np.nanmax(p84 - p16))


PCOL = 2
assert pnames[PCOL] == "VariableWindVelFactor", pnames[PCOL]
cvals = X_unit[:, PCOL]
mS = ELL <= ELL_TRUST
mE = ELL <= ELL_TRUST
SPEC = ("cl_yy", "cl_tt", "cl_kappa_y", "cl_kappa_tau", "cl_yt")
NU7 = (-3, 8)


# ── the fiducial reference vector, SWAPPED ──────────────────────────────────
def build_fid(run_dir, fc_path, nu05_path, label):
    fc = np.load(fc_path)
    assert np.allclose(fc["ell"], ELL), "field cache ell grid != Sobol dataset ell grid"
    bcl0 = np.load(run_dir / "Cl_kappa.npz")
    # the numerator is this run's OWN Cl mean, so the DMO denominator must be
    # averaged over the same realization count (50 in the 50-real tree, 1000 in
    # the campaign) -- dmo_paired_cl enforces that and raises on a mismatch.
    _nr_num = int(np.load(FID_RUN / "kappa_maps.npz")["n_real"])
    F = {"suppression": bcl0["cl"][ZI, ZI] / dmo_paired_cl(_nr_num)[ZI]}
    F["cl_kappa_y"] = fc["ky_bind"][:, ZI].mean(0)
    F["cl_kappa_tau"] = fc["kt_bind"][:, ZI].mean(0)
    F["cl_yy"] = fc["yy_bind"][:, -1].mean(0)
    F["cl_tt"] = fc["tt_bind"][:, -1].mean(0)
    if CAMPAIGN == "n1000":
        # compute-once y-tau cache (build_yt_cache_n1000.py): full-N total-column
        # mean, replacing the sci50 branch's first-50-realizations live recompute
        # so this leg carries the same realization count as the other 11 panels.
        from fsfig_common import REPLICA as _REPL
        _ytp = (run_dir.parent.parent / "field_cache" / f"yt_stats_bind_{_REPL}_n1000.npz")
        if not _ytp.exists():
            raise SystemExit(
                f"missing {_ytp}\nbuild it once with:\n"
                "  python papers/01_pipeline/build_yt_cache_n1000.py --side bind")
        with np.load(_ytp) as _z:
            _nr = int(_z["yt_real"].shape[0])
            assert int(_z["n_done"]) == _nr, f"{_ytp.name} incomplete"
            assert _nr == _nr_num, (
                f"yt cache holds {_nr} reals but the numerator run has {_nr_num} -- "
                "rebuild the yt cache so every panel averages the same realizations")
            F["cl_yt"] = np.asarray(_z["yt_real"], float)[:, -1].mean(0)
        # guard: cache vs the independent per-run MPI stats product
        _mpi_yt = np.asarray(np.load(run_dir / "Cl_tau.npz")["cl_yt"], float)
        _dd = float(np.nanmax(np.abs(F["cl_yt"] / np.where(_mpi_yt != 0, _mpi_yt, np.nan) - 1)))
        print(f"cl_yt cache-vs-MPI guard: max |ratio-1| = {_dd:.2e}")
        assert _dd < 1e-3, "yt cache disagrees with the per-run Cl_tau.npz product"
    else:
        ypath = run_dir / "y_maps.npz"
        tpath = (run_dir / "tau_maps.npz") if (run_dir / "tau_maps.npz").exists() else LC / "tau_maps.npz"
        BY = load_maps_prefix(ypath, "y", 50, -1)
        BT = load_maps_prefix(tpath, "tau", 50, -1)
        F["cl_yt"] = np.mean([power_spectrum(BY[r], BT[r])[1] for r in range(50)], 0)
        del BY, BT
        gc.collect()
    sh = np.load(nu05_path)
    assert np.allclose(sh["nu"], d["a__peak_counts__nu"]), "nu05 shard grid != dataset"
    for _k in ("peak_counts", "minima_counts", "mf_v0", "mf_v1", "mf_v2", "pdf"):
        F[_k] = sh[_k][ZI]
    assert set(F) == {s["k"] for s in STATS}
    print(f"fiducial reference vector [{label}] built: "
          + ", ".join(f"{k} peak={np.nanmax(np.abs(v)):.3e}" for k, v in F.items()))
    return F


FID = build_fid(FID_RUN, FID_FC, FID_NU05, "NEW twobound replica")
# The "OLD retired fiducial" arm is a 50-real product (its field cache and nu
# shard exist only in the 50-real tree).  Pairing it against 1000-real NEW
# quantities would be the mixed-N comparison this package guards against, so
# under n1000 the figure is rendered with the new fiducial alone.
FID_OLD = (None if CAMPAIGN == "n1000"
           else build_fid(OLD_FID, OLD_FC, OLD_NU05, "OLD retired fiducial"))

SPEC_FLOOR = 1e-10


def response_ratio(A, fid, floor_frac=0.0):
    floor = floor_frac * np.nanmax(np.abs(fid))
    safe = np.where(np.abs(fid) > floor, fid, np.nan)
    return A / safe[None, :]


NU_KEYS = {"pdf", "peak_counts", "minima_counts", "mf_v0", "mf_v1", "mf_v2"}


def nu_solid_drawn(key, fid):
    fid = np.asarray(fid, float)
    if key in ("peak_counts", "minima_counts"):
        return fid >= 1.0, fid > 0.0
    drawn = np.abs(fid) > 0.0
    return drawn, drawn


YLAB_RATIO = {
    "suppression": r"$S(\ell)/S_{\rm fid}(\ell)$",
    "pdf": r"$\mathrm{PDF}/\mathrm{PDF}_{\rm fid}$",
    "peak_counts": r"$N_{\rm pk}/N_{\rm pk,fid}$",
    "minima_counts": r"$N_{\rm min}/N_{\rm min,fid}$",
    "mf_v0": r"$V_0/V_{0,\rm fid}$",
    "mf_v1": r"$V_1/V_{1,\rm fid}$",
    "mf_v2": r"$V_2/V_{2,\rm fid}$",
    "cl_yy": r"$C_\ell^{yy}/C_{\ell,\rm fid}^{yy}$",
    "cl_tt": r"$C_\ell^{\tau\tau}/C_{\ell,\rm fid}^{\tau\tau}$",
    "cl_kappa_y": r"$C_\ell^{\kappa y}/C_{\ell,\rm fid}^{\kappa y}$",
    "cl_kappa_tau": r"$C_\ell^{\kappa\tau}/C_{\ell,\rm fid}^{\kappa\tau}$",
    "cl_yt": r"$C_\ell^{y\tau}/C_{\ell,\rm fid}^{y\tau}$",
}


def panel_data(s, F):
    k, A = s["k"], s["A"]
    fid = F[k]
    yl = YLAB_RATIO[k]
    if k in SPEC:
        return (ELL[mE], response_ratio(A[:, mE], fid[mE], SPEC_FLOOR), r"$\ell$", yl,
                "log", (ELL[mE].min(), ELL_TRUST))
    if k == "suppression":
        return (ELL[mS], response_ratio(A[:, mS], fid[mS], SPEC_FLOOR), r"$\ell$", yl,
                "log", (ELL[mS].min(), ELL_TRUST))
    return (s["x"], response_ratio(A, fid, floor_frac=0.0), r"$\nu$", yl, "linear", NU7)


fig, ax = plt.subplots(3, 4, figsize=(TWO_COL[0], 6.6), layout="constrained")
spreads, nan_frac = {}, {}
for r, (s, a) in enumerate(zip(STATS, ax.ravel())):
    x, Y, xl, yl, xs, xlim = panel_data(s, FID)
    if s["k"] in NU_KEYS:
        solid, drawn = nu_solid_drawn(s["k"], FID[s["k"]])
        color_curves(a, x, Y, cvals, solid=solid, drawn=drawn)
        spreads[s["k"]] = spread16_84(np.where(drawn[None, :], Y, np.nan))
        nan_frac[s["k"]] = float(1.0 - drawn.mean())
    else:
        color_curves(a, x, Y, cvals)
        spreads[s["k"]] = spread16_84(Y)
        nan_frac[s["k"]] = float(np.mean(~np.isfinite(Y)))
    a.axhline(1, color=COLORS["dmo"], ls=":", lw=0.8)
    a.set_xscale(xs)
    if xlim is not None:
        a.set_xlim(*xlim)
    a.set_xlabel(xl, fontsize=7)
    a.set_ylabel(yl, fontsize=6.5)
    a.tick_params(labelsize=5.5)
    panel_label(a, f"({chr(97 + r)})")

sm = ScalarMappable(cmap="coolwarm", norm=Normalize(0, 1))
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", location="bottom",
                  fraction=0.030, pad=0.02, aspect=55)
cb.set_label(short_label(pnames[PCOL]) + " (prior units)", fontsize=6.5)
cb.ax.tick_params(labelsize=5.5)
save_imgs(fig, "fig10_covariation")
plt.close(fig)

print(f"all {len(STATS)} panels coloured by ONE parameter: {pnames[PCOL]} "
      f"(unit-cube range {cvals.min():.3f}-{cvals.max():.3f}); "
      f"top-ranked in {int((IMPg.argmax(1) == PCOL).sum())}/{len(STATS)} fig-5 rows")
print("per-panel peak |rho| for this parameter: "
      + ", ".join(f"{s['k']} {IMPg[r, PCOL]:.2f}" for r, s in enumerate(STATS)))

if FID_OLD is None:
    np.savez_compressed(
        "/mnt/home/mlee1/ceph/bind_n1000/analysis/fig10_numbers_n1000.npz",
        ell=ELL, **{f"fid_new__{k}": v for k, v in FID.items()})
    print("\n(OLD vs NEW fiducial table skipped: BIND_CAMPAIGN=n1000 — the retired "
          "fiducial has no 1000-realization cache)")
    print("\nDONE fig10")
    raise SystemExit(0)

print("\n### OLD vs NEW fiducial: what each panel's denominator does to it")
print(f"{'stat':16s} {'16-84 spread OLD':>17s} {'NEW':>10s} | "
      f"{'median curve/fid OLD':>21s} {'NEW':>10s} | {'fid NEW/OLD':>12s}")
for s in STATS:
    k = s["k"]
    _, Yo, *_ = panel_data(s, FID_OLD)
    _, Yn, *_ = panel_data(s, FID)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        so, sn = spread16_84(Yo), spread16_84(Yn)
        mo, mn = np.nanmedian(Yo), np.nanmedian(Yn)
        fr = np.nanmedian(np.asarray(FID[k], float)
                          / np.where(np.abs(FID_OLD[k]) > 0, FID_OLD[k], np.nan))
    print(f"{k:16s} {so:17.4f} {sn:10.4f} | {mo:21.4f} {mn:10.4f} | {fr:12.4f}")
np.savez_compressed("/mnt/home/mlee1/ceph/referee_work/fidswap/fig10_numbers.npz",
                    ell=ELL, **{f"fid_new__{k}": v for k, v in FID.items()},
                    **{f"fid_old__{k}": v for k, v in FID_OLD.items()})
print("\nDONE fig10")
