#!/usr/bin/env python
"""family_basis_all.py -- THE shipped family-basis model, ALL WL statistics
(2026-08-11 author scoping: the paper ships (i) the families, (ii) the
degeneracy/compression result, (iii) per-run amplitudes over the Sobol set,
and (iv) a GP fitting function theta -> amplitudes, packaged so anyone can
explore the statistics.  The amplitude-physics identification is descoped
to follow-up work.)

Statistics covered: the 7 WL field statistics with twobound response
measurements -- S(ell), PDF(nu), peak counts, minima counts, V0, V1, V2.
(The gas auto/cross spectra have no twobound leg and are out of scope of
this construction.)

Pipeline per statistic (machinery of family_basis_ship.py, generalized;
that script's clk/pdf legs are superseded by this one):
  1. cluster the 30 twobound bound-to-bound responses into shape families
     (S/N >= 3 gate, average linkage, t=0.15; clk/pdf in their established
     S3b spaces, new statistics directly in their canonical basis space);
  2. basis B_k = |peak|-weighted mean of unit-peak member curves, unit-peak
     renormalized, on the statistic's canonical dataset grid;
  3. free per-run amplitudes over the Sobol suite (plain OLS / minimum-norm
     pinv everywhere -- 2026-08-12 author decision: the ridge branch used
     for clk (cond 520 > 100) was removed once ridge-vs-OLS were shown
     e2e-equivalent, see FAMILY_BASIS_METHODS.md);
     compression table (conditioning-aware adoption); shipped subset = the
     noise-backed families (>=1 member above the fig-05 permutation null);
  4. the LAMBDA-ROUTE fitting function (2026-08-11 author ruling: the GP
     theta->a route is dropped entirely -- the linear map from MEASURED
     group/cluster-bin halo latents beats it for every statistic, e2e CV
     0.77-0.94 vs 0.27-0.70): a_k = M.(lambda - lambda_ref) + a_ref with
     lambda the 8-latent EXTENDED set (2026-08-12 bake-off promotion,
     see below), 5-fold CV reported;
  5. bundle everything (basis, mean, amplitudes, lambda maps) into
     figs_preview/family_model_bundle.npz -- consumed by the numpy-only
     loader family_model.py, which evaluates any statistic from eight
     measured halo numbers (any simulation or observation; no CAMELS
     parameters involved).

2026-08-12 EXTENDED-LAMBDA promotion: a verified bake-off
(scratchpad/bakeoff_extended_lambda.py) found that adding 4 more measured
halo latents -- group-bin log-pressure, group- and cluster-bin f_gas,200c,
cluster-bin log-pressure -- to the original shipped 4 (f_bar, f_star,
c_gas, logT, all group-bin) beats the 4-latent map on ALL 7 statistics
(clk e2e CV 0.96 vs 0.92; fiducial clk trough prediction 0.911 vs 0.920
vs measured 0.881; overfit gap +0.003). See FAMILY_BASIS_METHODS.md and
figs_preview/family_model_bundle_4lat.npz (preserved prior bundle).

Grid conventions (verified 2026-08-11): twobound pdf standardized per run
onto the canonical nu grid (grids drift with map sigma); peak/minima counts
aggregated exactly 2:1 from the twobound dnu=0.25 grid onto the canonical
dnu=0.5 grid (edges align); MFs interpolated from the 29-point (-3..4) grid
onto the canonical centers <= 3.75 (the twobound grid caps at nu=4).

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python family_basis_all.py
"""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, TWO_COL, panel_label, save, setup  # noqa: E402
from param_labels import short_label  # noqa: E402

setup()
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
TB = CEPH / "bind_science/runs/twobound"
SB35 = CEPH / "bind_sb35"
assert Path.cwd().name == "01_pipeline", "run from papers/01_pipeline/"

ZI = 1
SNR_MIN = 3.0
T_CLUST = 0.15
EDGES = np.geomspace(100.0, 3e4, 27)
TOL = 5e-4
ALPHA_REL = 1e-2              # kept only as ridge_amps()'s default; unused
                               # since 2026-08-12 (estimator is plain OLS
                               # everywhere -- see FAMILY_BASIS_METHODS.md)
NMF = 22                      # canonical MF bins -- FULL 22-bin grid as of
                               # 2026-08-12: the twobound MFs were remeasured
                               # on the canonical nu05 grid from the stored
                               # kappa_maps.npz (remeasure_twobound_nu05.py;
                               # engine identical to the Sobol reduction,
                               # V0 reproduces the old grid EXACTLY at shared
                               # thresholds). Previously 14 (<=3.75), capped
                               # by the old -3..4 nongaussian_stats grid.

# ── twobound pairs ──────────────────────────────────────────────────────────
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
# canonical-grid remeasurement (remeasure_twobound_nu05.py, 2026-08-12):
# the same nu_grid.compute engine as the Sobol nu05 dataset, run on the
# stored twobound kappa_maps.npz -- supplies the MFs on the full 22-bin grid
NGC = {r: np.load(TB / f"run_{r:04d}/nu05_stats.npz") for r in range(60)}
ell = P[0]["ell"]
mf_nu_tb = NG[0]["mf_nu"]

from predict_from_latents import CACHE, build_cache  # noqa: E402

if not CACHE.exists():
    build_cache()
coef = np.load(CACHE)
assert int(np.atleast_1d(coef["version"])[0]) == 2
EDGb = coef["ell_edges"]
ctr_b = coef["ell"]
NBc = len(ctr_b)

dsn = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
NUg = np.asarray(dsn["a__pdf__pdf_bins"], float)
assert np.allclose(np.asarray(NGC[0]["nu"], float), NUg), \
    "canonical MF remeasurement grid != dataset nu grid"
Xu = np.asarray(dsn["X_unit"], float)
pn_ds = [str(s) for s in dsn["param_names"]]
assert set(pn_ds) == set(pairs)
_elld = np.asarray(dsn["a__suppression__ell"], float)

# twobound dnu=0.25 grid aggregates EXACTLY 2:1 onto the canonical grid:
# centers -2.875,-2.625 (edges [-3.0,-2.5)) -> canonical bin [-3.0,-2.5)
nu_tb = P[0]["nu"]
assert len(nu_tb) == 68 and np.isclose(nu_tb[8], -2.875) \
    and np.isclose(nu_tb[1] - nu_tb[0], 0.25)
AGG = [(8 + 2 * j, 9 + 2 * j) for j in range(22)]


def bandS(r):
    y = P[r]["clk_resp"][ZI]
    return np.stack([np.nanmean(y[(ell >= EDGb[i]) & (ell < EDGb[i + 1])])
                     for i in range(NBc)])


def pdf_nu(r):
    kb = NG[r]["pdf_bins"]
    p = NG[r]["pdf"][ZI]
    norm = np.trapezoid(p, kb)
    mu = np.trapezoid(kb * p, kb) / norm
    sig = np.sqrt(np.trapezoid((kb - mu) ** 2 * p, kb) / norm)
    return sig * np.interp(mu + sig * NUg, kb, p, left=0.0, right=0.0)


def counts_canon(r, key):
    """realization-mean counts aggregated onto the canonical 22-bin grid."""
    c = np.nanmean(P[r][key][:, ZI, :], axis=0)
    return np.array([c[i] + c[j] for i, j in AGG])


def mf_canon(r, stat):
    """MFs natively on the canonical 22-bin grid (2026-08-12 remeasurement;
    the old path interpolated the -3..4 nongaussian_stats grid onto the
    first NMF=14 canonical centers)."""
    return np.asarray(NGC[r][f"mf_{stat}"], float)[ZI][:NMF]


def logbin(y):
    """returns (centers, binned y, per-bin mode counts) over NON-EMPTY bins
    only -- the counts are aligned with the kept bins (2026-08-11 verifier
    fix: the old `nb[:len(x)]` truncation misaligned counts past the first
    empty bin, understating clk S/N by ~1.4x uniformly; no gate flips)."""
    ib = np.digitize(ell, EDGES) - 1
    L, Y, N = [], [], []
    for b in range(len(EDGES) - 1):
        s = ib == b
        if s.any():
            L.append(np.exp(np.log(ell[s]).mean()))
            Y.append(np.nanmean(y[s]))
            N.append(int(s.sum()))
    return np.array(L), np.array(Y), np.array(N)


def snr_resp(nm, rkey, ekey):
    """peak-bin |Delta resp| / err on the native twobound grid -- the exact
    paper_s3b semantics (S/N evaluated AT the argmax-|Delta| bin), with a
    finite guard."""
    rlo, rhi = pairs[nm]
    d = P[rhi][rkey][ZI] - P[rlo][rkey][ZI]
    e = np.sqrt(P[rhi][ekey][ZI] ** 2 + P[rlo][ekey][ZI] ** 2)
    k = int(np.nanargmax(np.abs(d)))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.abs(d[k]) / e[k]
    return float(v) if np.isfinite(v) else 0.0


def snr_counts(nm, key):
    """peak-bin count-difference S/N on the canonical grid; errors from the
    50-realization scatter of the aggregated counts (the resp-ratio errors
    are meaningless where counts vanish)."""
    rlo, rhi = pairs[nm]

    def agg_reals(r):
        c = P[r][key][:, ZI, :]
        return np.stack([c[:, i] + c[:, j] for i, j in AGG], 1)

    # PAIRED per-realization differences (the twobound runs share RT seeds,
    # so cosmic variance cancels realization-by-realization); detect on the
    # WHOLE curve via the chi-square excess over its dof, in null sigmas
    # (single-bin count differences are Poisson-limited)
    dre = agg_reals(rhi) - agg_reals(rlo)
    d = dre.mean(0)
    e = dre.std(0) / np.sqrt(len(dre))
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = (d / e) ** 2
    z2 = z2[np.isfinite(z2)]
    if not len(z2):
        return 0.0
    return float((z2.sum() - len(z2)) / np.sqrt(2 * len(z2)))


def snr_clk(nm):
    rlo, rhi = pairs[nm]
    x, d, nb = logbin(P[rhi]["clk_resp"][ZI] - P[rlo]["clk_resp"][ZI])
    _, e, _ = logbin(np.sqrt(P[rhi]["clk_err"][ZI] ** 2
                             + P[rlo]["clk_err"][ZI] ** 2))
    k = int(np.nanargmax(np.abs(d)))
    return float(np.abs(d[k]) / (e / np.sqrt(nb))[k])


def member_curve(nm, stat):
    """the basis-space (canonical-grid) response difference."""
    rlo, rhi = pairs[nm]
    if stat == "clk":
        return bandS(rhi) - bandS(rlo)
    if stat == "pdf":
        return pdf_nu(rhi) - pdf_nu(rlo)
    if stat in ("pk", "mn"):
        key = "pk_real" if stat == "pk" else "min_real"
        return counts_canon(rhi, key) - counts_canon(rlo, key)
    return mf_canon(rhi, stat) - mf_canon(rlo, stat)


def cluster_curve(nm, stat):
    """the CLUSTERING-space curve: clk/pdf keep their established S3b
    spaces; the new statistics cluster directly in basis space."""
    if stat == "clk":
        rlo, rhi = pairs[nm]
        _, d, _ = logbin(P[rhi]["clk_resp"][ZI] - P[rlo]["clk_resp"][ZI])
        return d
    return member_curve(nm, stat)


SPEC = {
    "clk": dict(dkey="suppression", x=ctr_b, logx=True,
                slab=r"$S(\ell)$", xlab=r"$\ell$",
                snr=snr_clk, relres="truth"),
    "pdf": dict(dkey="pdf", x=NUg, logx=False,
                slab=r"PDF$(\nu)$", xlab=r"$\nu=\kappa/\sigma_\kappa$",
                snr=lambda nm: snr_resp(nm, "V0_resp", "V0_err"),
                relres="max"),
    "pk":  dict(dkey="peak_counts", x=NUg, logx=False,
                slab=r"$N_{\rm pk}(\nu)$", xlab=r"$\nu$",
                snr=lambda nm: snr_counts(nm, "pk_real"),
                relres="max"),
    "mn":  dict(dkey="minima_counts", x=NUg, logx=False,
                slab=r"$N_{\rm min}(\nu)$", xlab=r"$\nu$",
                snr=lambda nm: snr_counts(nm, "min_real"),
                relres="max"),
    "v0":  dict(dkey="mf_v0", x=NUg[:NMF], logx=False,
                slab=r"$V_0(\nu)$", xlab=r"$\nu$",
                snr=lambda nm: snr_resp(nm, "V0_resp", "V0_err"),
                relres="max"),
    "v1":  dict(dkey="mf_v1", x=NUg[:NMF], logx=False,
                slab=r"$V_1(\nu)$", xlab=r"$\nu$",
                snr=lambda nm: snr_resp(nm, "V1_resp", "V1_err"),
                relres="max"),
    "v2":  dict(dkey="mf_v2", x=NUg[:NMF], logx=False,
                slab=r"$V_2(\nu)$", xlab=r"$\nu$",
                snr=lambda nm: snr_resp(nm, "V2_resp", "V2_err"),
                relres="max"),
}


def sobol_Y(stat):
    s = SPEC[stat]
    A = np.asarray(dsn[f"t__{s['dkey']}__value"], float)
    A = A[:, ZI, :] if A.ndim == 3 else A
    if stat == "clk":
        return np.stack([np.nanmean(A[:, (_elld >= EDGb[i])
                                      & (_elld < EDGb[i + 1])], 1)
                         for i in range(NBc)], axis=1)
    if stat in ("v0", "v1", "v2"):
        return A[:, :NMF]
    return A


# ── fig-05 noise verdicts for all 7 rows (fig05a conventions) ──────────────
def _spearman(Pm, Ym):
    rp = rankdata(Pm, axis=0).astype(float)
    ry = rankdata(Ym, axis=0).astype(float)
    rp = (rp - rp.mean(0)) / rp.std(0)
    ry = (ry - ry.mean(0)) / np.maximum(ry.std(0), 1e-12)
    return rp.T @ ry / len(Pm)


_S16 = np.asarray(dsn["t__suppression__value"], float)[:, ZI, :][:, _elld <= 36864.0]
_S16 = _S16[:, :(_S16.shape[1] // 16) * 16].reshape(len(_S16), -1, 16).mean(2)
ROWS_SIG = {"clk": _S16}
for st, dk in (("pdf", "pdf"), ("pk", "peak_counts"), ("mn", "minima_counts"),
               ("v0", "mf_v0"), ("v1", "mf_v1"), ("v2", "mf_v2")):
    ROWS_SIG[st] = np.asarray(dsn[f"t__{dk}__value"], float)[:, ZI, :]
for k, Y in ROWS_SIG.items():
    assert np.isfinite(Y).all(), f"{k} significance row not finite"
_rngp = np.random.default_rng(4)
_null = {k: np.empty((200, Xu.shape[1])) for k in ROWS_SIG}
for _it in range(200):
    _perm = _rngp.permutation(len(Xu))
    for k, Y in ROWS_SIG.items():
        _null[k][_it] = np.abs(_spearman(Xu[_perm], Y)).max(1)
SOLID = {}
for k, Y in ROWS_SIG.items():
    imp = np.abs(_spearman(Xu, Y)).max(1)
    n95 = float(np.percentile(_null[k], 95))
    SOLID[k] = {nm: bool(v >= n95) for nm, v in zip(pn_ds, imp)}


def cluster_families(stat):
    CUR = {}
    for nm in pairs:
        if SPEC[stat]["snr"](nm) < SNR_MIN:
            continue
        d = cluster_curve(nm, stat)
        if not np.isfinite(d).all() or np.abs(d).max() == 0:
            continue
        CUR[nm] = d / d[np.nanargmax(np.abs(d))]
    names = list(CUR)
    if len(names) < 2:
        return [(f"C{i+1}", [n]) for i, n in enumerate(names)]
    M = np.corrcoef([CUR[n] for n in names])
    Z = linkage(squareform(np.clip(1 - M, 0, None), checks=False),
                method="average")
    lab = fcluster(Z, t=T_CLUST, criterion="distance")
    cl = [[names[i] for i in np.where(lab == c)[0]] for c in np.unique(lab)]
    cl.sort(key=len, reverse=True)
    fams = [(f"C{k+1}", mem) for k, mem in enumerate(
        [c for c in cl if len(c) > 1])]
    for s in (c[0] for c in cl if len(c) == 1):
        fams.append((f"C{len(fams)+1}", [s]))
    return fams


def build_basis(fams, stat):
    rows = []
    for fam, mem in fams:
        curves, ws = [], []
        for m in mem:
            d = member_curve(m, stat)
            pk = d[np.nanargmax(np.abs(d))]
            curves.append(d / pk)
            ws.append(abs(pk))
        b = np.average(curves, axis=0, weights=ws)
        rows.append(b / b[np.nanargmax(np.abs(b))])
    return np.array(rows)


def span_fit(Dm, Bmat):
    A = Dm @ np.linalg.pinv(Bmat)
    R = Dm - A @ Bmat
    return (A, 1 - (R ** 2).sum(1) / (Dm ** 2).sum(1),
            1 - (R ** 2).sum() / (Dm ** 2).sum())


def ridge_amps(Dm, Bmat, alpha_rel=ALPHA_REL):
    """Unused since 2026-08-12 (estimator is plain OLS everywhere, see
    FAMILY_BASIS_METHODS.md); kept for reference / off-line display re-
    projection of clk's ill-conditioned amplitudes if ever wanted."""
    BBt = Bmat @ Bmat.T
    al = alpha_rel * np.trace(BBt) / Bmat.shape[0]
    return Dm @ Bmat.T @ np.linalg.inv(BBt + al * np.eye(Bmat.shape[0]))


# ── the lambda-route fitting function (predict_from_latents convention) ────
# 2026-08-12 OBSERVABLE AGNOSTIC-LAMBDA promotion (author-adopted): a
# fully agnostic SFFS search (papers/01_pipeline/agnostic_lambda_search.py
# with OBS_ONLY=1; log audits/agnostic_lambda_search_obs_2026-08-12.log;
# results figs_preview/agnostic_lambda_results_obs.npz -- reference spec,
# do not re-derive) over 90 observationally accessible candidates
# (10 median quantities + 2 halo-to-halo widths x 7 mass bins 13.0..14.3+
# + 6 mass-trend slopes; DM-referencing candidates excluded; 10 shuffled
# decoys as tripwire, never selected) chose these 8 latents, stopping at
# its own +2e-3 pooled-gain threshold (first reject +1.3e-3). Selection
# targeted the POOLED 7-statistic e2e CV on selection folds (sklearn
# KFold rs=123), distinct from the rng(1) reporting folds used below.
# f_bar[13.0-13.2] is picked FIRST (20/20 bootstraps) and later evicted
# by the floating step as redundant with the same bin's Compton-Y +
# temperature (Y ~ gas mass x T). Beats the two-bin 8-latent map on all
# 7 statistics (pooled e2e CV 0.9157 vs 0.9068, rng(1) folds) and on the
# untouched out-of-design fiducial (clk trough 0.895 vs 0.908, measured
# 0.881). Column order = selection order. Prior bundle preserved at
# figs_preview/family_model_bundle_2bin8.npz. See FAMILY_BASIS_METHODS.md.
OB_OM = 0.0486 / 0.3089
cz = np.load(SB35 / "analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
_rows = dsn["run_ids"]
_mt = cz["sobol_m_tot_500c_bg"][_rows]          # log10-binning field, always
_mg = cz["sobol_m_gas_500c_bg"][_rows]
_g2 = cz["sobol_m_gas_200c_bg"][_rows]
_ms = cz["sobol_m_star_500c"][_rows]
_Tm = cz["sobol_T_mw_500c"][_rows]
_Pe = cz["sobol_Pe_mw_500c"][_rows]
_Ys = cz["sobol_Y_500c"][_rows]
_lm = np.log10(np.where(_mt > 0, _mt, np.nan))

MIN_N = 5
LAT_BINS = {"13.0-13.2": (13.0, 13.2), "13.2-13.4": (13.2, 13.4),
            "13.4-13.6": (13.4, 13.6), "14.0-14.3": (14.0, 14.3)}
LAT_NAMES = ["f_star[13.2-13.4]", "logT[13.0-13.2]", "logY[13.0-13.2]",
             "logPe[14.0-14.3]", "c_gas[14.0-14.3]", "logY_ss[13.4-13.6]",
             "c_gas[13.2-13.4]", "logPe[13.4-13.6]"]


def _row_bin_med(arr_row, mask, pos=False, log=False, min_n=MIN_N):
    """bake-off ``_bin_med`` reducer, one halo-catalog row: median of
    ``arr_row`` within the log10(Mtot) bin ``mask``; optional positivity
    guard (non-positive -> NaN) and log10 of the median; NaN if fewer
    than ``min_n`` finite values survive."""
    v = arr_row[mask]
    if pos:
        v = np.where(v > 0, v, np.nan)
    if np.isfinite(v).sum() < min_n:
        return np.nan
    m = np.nanmedian(v)
    return np.log10(m) if (log and m > 0) else m


LAT = np.full((len(_rows), 8), np.nan)
with np.errstate(divide="ignore", invalid="ignore"):
    for _q in range(len(_rows)):
        _b = {k: (_lm[_q] >= lo) & (_lm[_q] < hi)
              for k, (lo, hi) in LAT_BINS.items()}
        _cg = _mg[_q] / _g2[_q]                       # bg/bg gas concentration
        _yss = _Ys[_q] / np.where(_mt[_q] > 0, _mt[_q], np.nan) ** (5 / 3)
        LAT[_q, 0] = _row_bin_med(_ms[_q] / _mt[_q], _b["13.2-13.4"]) / OB_OM
        LAT[_q, 1] = _row_bin_med(_Tm[_q], _b["13.0-13.2"], pos=True, log=True)
        LAT[_q, 2] = _row_bin_med(_Ys[_q], _b["13.0-13.2"], pos=True, log=True)
        LAT[_q, 3] = _row_bin_med(_Pe[_q], _b["14.0-14.3"], pos=True, log=True)
        LAT[_q, 4] = _row_bin_med(_cg, _b["14.0-14.3"])
        LAT[_q, 5] = _row_bin_med(_yss, _b["13.4-13.6"], pos=True, log=True)
        LAT[_q, 6] = _row_bin_med(_cg, _b["13.2-13.4"])
        LAT[_q, 7] = _row_bin_med(_Pe[_q], _b["13.4-13.6"], pos=True, log=True)
assert np.isfinite(LAT).all(), "non-finite latent rows -- handle masking"
LAT_REF = LAT.mean(0)


def lam_fit_amps(As):
    """5-fold-CV predictions + the full-sample linear map for each
    amplitude column: a = M.(lambda - LAT_REF) + intercept."""
    Lc = LAT - LAT_REF
    order = np.random.default_rng(1).permutation(len(Lc))
    pred = np.full_like(As, np.nan)
    for f in np.array_split(order, 5):
        tr = np.setdiff1d(order, f)
        W, *_ = np.linalg.lstsq(np.c_[Lc[tr], np.ones(len(tr))], As[tr],
                                rcond=None)
        pred[f] = np.c_[Lc[f], np.ones(len(f))] @ W
    Wfull, *_ = np.linalg.lstsq(np.c_[Lc, np.ones(len(Lc))], As, rcond=None)
    r2 = 1 - ((As - pred) ** 2).sum(0) / ((As - As.mean(0)) ** 2).sum(0)
    return pred, Wfull, r2


FC = ["#2a78d6", "#eb6834", "#199e70", "#c98500", "#d55181"]
BUNDLE = {
    "lat_names": np.array(LAT_NAMES),
    "lat_convention": np.array(
        "8-latent OBSERVABLE AGNOSTIC-LAMBDA (2026-08-12 agnostic SFFS "
        "search, OBS_ONLY; every latent observationally accessible; "
        "all-bg 500c/200c convention; Ob/Om=0.0486/0.3089; medians over "
        "halos with min_n=5 in the quoted log10(M_500c,bg) bin): "
        "f_star[13.2-13.4]=median(Mstar500/Mtot500bg)/(Ob/Om); "
        "logT[13.0-13.2]=log10 median T_mw,500c [K] (T>0); "
        "logY[13.0-13.2]=log10 median Y_500c (Y>0); "
        "logPe[14.0-14.3]=log10 median Pe_mw,500c (Pe>0); "
        "c_gas[14.0-14.3]=median(Mgas500bg/Mgas200bg); "
        "logY_ss[13.4-13.6]=log10 median Y_500c/Mtot500bg^(5/3) (>0); "
        "c_gas[13.2-13.4]=median(Mgas500bg/Mgas200bg); "
        "logPe[13.4-13.6]=log10 median Pe_mw,500c (Pe>0). "
        "Order matches LAT columns 0-7 = the search's selection order; "
        "superseded the two-bin extended-lambda 8 (2026-08-12, preserved "
        "at figs_preview/family_model_bundle_2bin8.npz) and the shipped-4 "
        "(figs_preview/family_model_bundle_4lat.npz). See "
        "FAMILY_BASIS_METHODS.md and agnostic_lambda_search.py."),
}
SUMMARY = []

for stat in SPEC:
    s = SPEC[stat]
    xg = np.asarray(s["x"], float)
    tag = "" if stat == "clk" else f"_{stat}"
    print("\n" + "=" * 70 + f"\nFAMILY MODEL: {stat} ({s['dkey']})\n" + "=" * 70)

    fams = cluster_families(stat)
    K = len(fams)
    print("families: " + "; ".join(f"{f}({len(m)})" for f, m in fams))
    B = build_basis(fams, stat)

    Y = sobol_Y(stat)
    fin = np.isfinite(Y).all(1)
    Yf = Y[fin]
    Ybar = Yf.mean(0)
    D = Yf - Ybar
    Afull, r2r_full, r2_full = span_fit(D, B)
    print(f"full-{K} model: total R^2 {r2_full:.4f}, median per-run "
          f"{np.median(r2r_full):.4f}")

    backed = [i for i, (f, mem) in enumerate(fams)
              if any(SOLID[stat][m] for m in mem)]
    # SAFEGUARD (2026-08-11 verifier finding, minima counts): the fig-05
    # backing test is a confounded marginal-rank test with much less power
    # than the paired twobound contrast every family member has already
    # passed (all admitted at >=3 sigma) -- for mn it dropped the single
    # strongest family (BlackHoleFeedbackFactor, ~0.73 R^2 alone; Sobol
    # |rho| 0.142 vs null 0.177, a narrow power-limited miss).  If the
    # backed subset falls > MARGIN below the full-family span, add
    # twobound-detected families in order of marginal span gain, FLAGGED
    # as twobound-backed only.
    MARGIN = 0.05
    extra = []
    while (span_fit(D, B[sorted(backed + extra)])[2] if backed + extra
           else 0.0) < r2_full - MARGIN:
        rest = [i for i in range(K) if i not in backed + extra]
        if not rest:
            break
        extra.append(max(rest, key=lambda i: span_fit(
            D, B[sorted(backed + extra + [i])])[2]))
    ship_idx = sorted(backed + extra)
    shipnames = [fams[i][0] for i in ship_idx]
    if extra:
        print("  NB twobound-backed-only families added by the span "
              "safeguard: "
              + ", ".join(fams[i][0] for i in sorted(extra)))
    Bs = B[ship_idx]
    condS = float(np.linalg.cond(Bs))
    _, r2r_ship, r2_ship = span_fit(D, Bs)
    # 2026-08-12: estimator is plain OLS (minimum-norm pinv) everywhere --
    # the ridge branch (previously triggered only for clk, cond 520 > 100)
    # was removed; ridge-vs-OLS were measured e2e-equivalent (clk CV 0.9585
    # ridge vs 0.9593 OLS) and clk's ill-conditioning (kappa printed below)
    # is now an amplitude-interpretation caveat, not an estimator choice --
    # see FAMILY_BASIS_METHODS.md.
    As = span_fit(D, Bs)[0]
    r2_model = 1 - (((D - As @ Bs) ** 2).sum() / (D ** 2).sum())
    print(f"SHIPPED {{{','.join(shipnames)}}}: span R^2 {r2_ship:.4f}, "
          f"cond {condS:.0f} -> OLS "
          f"(model R^2 {r2_model:.4f})")

    adopted = {}
    for kk in range(1, K + 1):
        scored = [(span_fit(D, B[list(c)])[2],
                   float(np.linalg.cond(B[list(c)])), c)
                  for c in combinations(range(K), kk)]
        best = max(t[0] for t in scored)
        short = [t for t in scored if t[0] >= best - TOL]
        bck = [t for t in short if all(i in backed for i in t[2])]
        adopted[kk] = min(bck or short, key=lambda t: t[1])
        r2a, ca, sub = adopted[kk]
        print("  k=%d: best %.4f | adopted {%s} R^2 %.4f cond %.1f"
              % (kk, best, ",".join(fams[i][0] for i in sub), r2a, ca))

    # lambda-route fitting function: measured latents -> amplitudes
    assert As.shape[0] == LAT.shape[0], \
        "amplitude rows / latent rows misaligned -- a Sobol run was dropped"
    pred_cv, Wlat, r2amp = lam_fit_amps(As)
    D_pred = pred_cv @ Bs
    r2_e2e = 1 - ((D - D_pred) ** 2).sum() / (D ** 2).sum()
    rms_e2e = float(np.sqrt(((D - D_pred) ** 2).mean()))
    print("lambda->a (5-fold CV R^2): "
          + ", ".join(f"{n}:{v:.2f}" for n, v in zip(shipnames, r2amp)))
    print(f"END-TO-END lambda model: total R^2 {r2_e2e:.4f}, RMS "
          f"{rms_e2e:.4g} (free-amplitude ceiling {r2_ship:.4f})")
    SUMMARY.append((stat, K, len(ship_idx), r2_ship, r2_e2e))

    BUNDLE[f"{stat}__x"] = xg
    BUNDLE[f"{stat}__mean"] = Ybar
    BUNDLE[f"{stat}__basis"] = Bs
    BUNDLE[f"{stat}__fam_names"] = np.array(shipnames)
    # always False now (estimator is plain OLS everywhere); kept for
    # loader/tooling schema compat (no current reader depends on it).
    BUNDLE[f"{stat}__used_ridge"] = np.array(False)
    BUNDLE[f"{stat}__twobound_only"] = np.array(
        [i in extra for i in ship_idx])
    BUNDLE[f"{stat}__amps"] = As
    # r2 = [full-family span, shipped-span ceiling (free LSQ), shipped
    #      MODEL (the stored amplitudes, = ceiling unless ridge),
    #      lambda-route e2e CV]
    BUNDLE[f"{stat}__r2"] = np.array([r2_full, r2_ship, r2_model, r2_e2e])
    BUNDLE[f"{stat}__r2_amp_cv"] = r2amp
    BUNDLE[f"{stat}__lat_M"] = Wlat            # (9, K): 8 slopes + intercept
    BUNDLE[f"{stat}__lat_ref"] = LAT_REF
    # per-bin predictive sigma of the lambda route (CV residual scatter
    # across the design) -- the model's honest uncertainty; e.g. for S(ell)
    # it is ~0.004 at ell~1e3 but shrinks (not eliminated) at the ell>1e4
    # trough now that the 8-latent set includes cluster-bin pressure/f_gas
    BUNDLE[f"{stat}__sig_pred"] = (D - D_pred).std(0)

    # ── stages figure: all-run model array + relative residual subpanel ────
    mdl = Ybar + Afull @ B
    if stat == "clk":
        key = D.mean(1)
        klabel = "run mean deviation"
    else:
        v1 = np.linalg.svd(D, full_matrices=False)[2][0]
        key = D @ v1
        klabel = "leading-mode amplitude"
    vmax = float(np.abs(key).max())
    cmap = mpl.cm.RdBu_r
    order = np.argsort(np.abs(key))
    extv = [int(np.argmin(key)), int(np.argmax(key))]
    order = np.concatenate([[r for r in order if r not in extv], extv])
    fig, (ta, tb_) = plt.subplots(
        2, 1, figsize=(ONE_COL[0], 3.4), sharex=True,
        gridspec_kw=dict(height_ratios=[2.2, 1.0], hspace=0.06))
    denom = (None if s["relres"] == "truth" else
             np.abs(Yf).max(1, keepdims=True))
    for r in order:
        col = cmap(0.5 + 0.5 * key[r] / vmax)
        lw = 1.1 if r in extv else 0.4
        ta.plot(xg, mdl[r], color=col, lw=lw, alpha=0.85)
        rel = ((mdl[r] - Yf[r]) / Yf[r] if s["relres"] == "truth"
               else (mdl[r] - Yf[r]) / denom[r])
        tb_.plot(xg, 100 * rel, color=col, lw=lw, alpha=0.7)
    if s["logx"]:
        ta.set_xscale("log")
        ta.axhline(1, color="0.85", lw=0.5, zorder=0)
    ta.set_ylabel(s["slab"] + f"  (full {K}-family model)", fontsize=6.5)
    tb_.axhline(0, color="0.8", lw=0.5, zorder=0)
    tb_.set_ylabel("(model $-$ truth)" +
                   (" / truth [%]" if s["relres"] == "truth"
                    else " / max|truth| [%]"), fontsize=5.8)
    tb_.set_xlabel(s["xlab"], fontsize=7)
    sm = mpl.cm.ScalarMappable(cmap=cmap,
                               norm=mpl.colors.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=(ta, tb_), fraction=0.05, pad=0.02)
    cb.set_label(klabel, fontsize=5.8)
    cb.ax.tick_params(labelsize=5)
    save(fig, f"figs_v2/pfig_family_model_stages{tag}")
    plt.close(fig)

    # ── compression figure (no PCA ceiling) ────────────────────────────────
    fig, ax = plt.subplots(figsize=ONE_COL)
    ks = np.arange(1, K + 1)
    ax.semilogy(ks, [1 - adopted[k][0] for k in ks], "o-",
                color=COLORS["bind"], lw=1.4, ms=4.5)
    for k in ks:
        r2a, _, sub = adopted[k]
        last = k == K
        ax.annotate("{" + ",".join(fams[i][0] for i in sub) + "}",
                    (k, 1 - r2a), textcoords="offset points",
                    xytext=(-4, 6) if last else (5, 4),
                    ha="right" if last else "left", fontsize=5.2)
    ish = len(ship_idx)
    same = list(adopted[ish][2]) == ship_idx
    ax.plot([ish], [1 - r2_ship], "o", ms=9, mfc="none",
            mec=COLORS["highlight"], mew=1.4)
    ax.annotate("shipped" if same else
                "shipped {" + ",".join(shipnames) + "}",
                (ish, 1 - r2_ship), textcoords="offset points",
                xytext=(-14, -16), ha="center", fontsize=5.6,
                color=COLORS["highlight"],
                bbox=dict(fc="w", ec="none", alpha=0.85, pad=0.8))
    ax.set_xticks(ks)
    ax.set_xlabel("number of families kept", fontsize=7)
    ax.set_ylabel(r"$1-R^2$ (total)", fontsize=7)
    save(fig, f"figs_v2/pfig_family_model_compression{tag}")
    plt.close(fig)

np.savez("figs_preview/family_model_bundle.npz", **BUNDLE)

# ── summary figure: the whole product on one panel ──────────────────────────
fig, ax = plt.subplots(figsize=(ONE_COL[0], 2.4))
xs = np.arange(len(SUMMARY))
ax.bar(xs - 0.18, [t[3] for t in SUMMARY], 0.36, color=COLORS["bind"],
       label="free-amplitude ceiling")
ax.bar(xs + 0.18, [t[4] for t in SUMMARY], 0.36, color=COLORS["highlight"],
       label=r"$\lambda$ route: $a(\hat\lambda)$ (5-fold CV)")
for i, (stat, K, ks, r2s, r2e) in enumerate(SUMMARY):
    ax.text(i, 0.02, f"{ks}/{K}", ha="center", fontsize=5.5, color="w")
ax.set_xticks(xs)
ax.set_xticklabels([SPEC[t[0]]["slab"] for t in SUMMARY], fontsize=6)
ax.set_ylim(0, 1.0)
ax.set_ylabel(r"total $R^2$", fontsize=7)
ax.legend(fontsize=5.4, loc="lower right")
ax.set_title("shipped family models (labels: families kept / found)",
             fontsize=6.6)
save(fig, "figs_v2/pfig_family_model_summary")
plt.close(fig)

print("\n" + "=" * 70)
print("SUMMARY  (stat: shipped/found families, span R^2, lambda e2e CV)")
for stat, K, ks, r2s, r2e in SUMMARY:
    print(f"  {stat:>4s}: {ks}/{K} families, span {r2s:.4f}, "
          f"lambda e2e {r2e:.4f}")
print("wrote figs_preview/family_model_bundle.npz + per-stat figures "
      "+ pfig_family_model_summary")
