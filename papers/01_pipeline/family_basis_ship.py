#!/usr/bin/env python
"""family_basis_ship.py -- the SHIPPED family-basis models + figures for the
paper, for BOTH statistics (2026-08-11 author rulings):

  * C_ell^kappakappa suppression S(ell): ship the {C1,C2,C3,C4} model (all
    noise-backed families; C5's lone member SNII_MinMass is below the fig-05
    S(ell) permutation null).
  * lensing kappa-PDF: the same exact procedure -- families from the S3b
    pdf clustering, basis on the canonical nu05 grid, shipped model = the
    noise-backed families ({C1,C2,C3}; the ThermalWindFrac singleton is
    below the fig-05 PDF row null).

Per statistic this script produces:
  pfig_family_model_stages[_pdf]      -- the full-family model vs truth for
        ALL Sobol runs (array of lines, colored by run mean deviation) with
        a lower subpanel of the relative difference model-vs-truth.
  pfig_family_model_compression[_pdf] -- 1-R^2 vs number of families kept
        (conditioning-aware adopted subsets; no PCA ceiling).
  pfig_family_model_amps[_pdf]        -- per-amplitude identification
        scatters against the best-correlated measured halo latent.
  figs_preview/family_model_ship_{clk,pdf}.npz -- basis, amplitudes, tables.

EQUATIONS (documented in FAMILY_BASIS_METHODS.md; summary):
  member response   dS_m(x_b) = band/grid value of (high-bound - low-bound)
  signed peak       p_m = dS_m(b*_m),  b*_m = argmax_b |dS_m(b)|
  unit-peak curve   s_m = dS_m / p_m                  (sign-aligning)
  family basis     B_k = sum_m |p_m| s_m / sum_m |p_m|, renormalized to
                    unit peak: B_k <- B_k / B_k(argmax |B_k|)
  amplitudes        a_r = argmin_a || D_r - sum_k a_k B_k ||^2   (OLS), or
                    ridge a_r = D_r B^T (B B^T + alpha I)^-1 with
                    alpha = alpha_rel tr(B B^T)/K when cond(B) > 50
  where D_r = y_r - <y> over the Sobol suite, y = S(ell) bands or PDF(nu).
  PDF standardization (twobound -> canonical nu grid): the nu05 dataset PDF
  is a histogram of (map - mean)/std (nu_grid.py:119), so the twobound raw-
  kappa PDFs are mapped via PDF_nu(nu) = sigma_r PDF_kappa(mu_r + sigma_r nu)
  with mu_r, sigma_r the moments implied by the raw histogram itself (its
  +-9.6 sigma grid makes truncation negligible), linearly interpolated onto
  the canonical 22-bin nu grid.

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python family_basis_ship.py
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
# Ship ridge amplitudes only above this basis condition number.  Calibrated
# 2026-08-11: the clk {C1,C2,C3,C4} basis (cond 520) NEEDS ridge for
# identifiable amplitudes (OLS trades off wildly along the near-null
# combination) and loses only 0.0013 span R^2 to it; the pdf {C1,C2,C3}
# basis (cond 58) is moderate -- there OLS amplitudes are already usable
# and the same relative alpha would cost 40% of the model R^2.
COND_RIDGE = 100.0
ALPHA_REL = 1e-2

# ── twobound pairs (verbatim paper_s3b_clusters machinery) ──────────────────
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
# NB each twobound run's raw-kappa pdf grid is set from ITS OWN map sigma
# (bind.inference.stats), so pdf_bins drift run-to-run by up to ~1.5 bins
# between a bound pair -- every pdf operation below therefore uses
# NG[r]["pdf_bins"] per run and works on the COMMON canonical nu grid
# (2026-08-11 verifier finding; a shared-grid assumption here corrupts the
# 1-2%-level member difference curves at up to the 100% level).


def logbin(y):
    ib = np.digitize(ell, EDGES) - 1
    L, Y = [], []
    for b in range(len(EDGES) - 1):
        s = ib == b
        if s.any():
            L.append(np.exp(np.log(ell[s]).mean()))
            Y.append(np.nanmean(y[s]))
    return np.array(L), np.array(Y)


def get_stat(nm, stat):
    """cluster-space (x, Delta) for one parameter.  clk: paper_s3b logbin
    convention.  pdf: the difference of the two bounds' nu-standardized
    PDFs on the canonical grid (each run standardized on its OWN raw grid
    -- differencing the raw histograms index-by-index is invalid because
    the grids differ between the paired runs)."""
    rlo, rhi = pairs[nm]
    if stat == "clk":
        x, d = logbin(P[rhi]["clk_resp"][ZI] - P[rlo]["clk_resp"][ZI])
        return x, d
    return NUg, pdf_nu(rhi) - pdf_nu(rlo)


def snr_of(nm, stat):
    """peak-bin S/N; pdf uses the V0 response proxy (as paper_s3b does)."""
    rlo, rhi = pairs[nm]
    if stat == "pdf":
        d = P[rhi]["V0_resp"][ZI] - P[rlo]["V0_resp"][ZI]
        e = np.sqrt(P[rhi]["V0_err"][ZI] ** 2 + P[rlo]["V0_err"][ZI] ** 2)
    else:
        x, d = get_stat(nm, stat)
        _, e = logbin(np.sqrt(P[rhi]["clk_err"][ZI] ** 2
                              + P[rlo]["clk_err"][ZI] ** 2))
        nb = np.diff(np.searchsorted(ell, EDGES)).clip(1)[:len(x)]
        e = e / np.sqrt(nb)
    k = int(np.nanargmax(np.abs(d)))
    return float(np.abs(d[k]) / e[k])


def cluster_families(stat):
    """S/N gate + average-linkage clustering, identical to paper_s3b."""
    CUR = {}
    for nm in pairs:
        if snr_of(nm, stat) < SNR_MIN:
            continue
        _, d = get_stat(nm, stat)
        CUR[nm] = d / d[np.nanargmax(np.abs(d))]
    names = list(CUR)
    M = np.corrcoef([CUR[n] for n in names])
    Z = linkage(squareform(np.clip(1 - M, 0, None), checks=False),
                method="average")
    lab = fcluster(Z, t=T_CLUST, criterion="distance")
    cl = [[names[i] for i in np.where(lab == c)[0]] for c in np.unique(lab)]
    cl.sort(key=len, reverse=True)
    fams = [(f"C{k+1}", mem) for k, mem in enumerate(
        [c for c in cl if len(c) > 1])]
    # each singleton is its OWN family -- lumping them would average
    # mutually-uncorrelated shapes into one meaningless basis vector
    for s in (c[0] for c in cl if len(c) == 1):
        fams.append((f"C{len(fams)+1}", [s]))
    return fams


# ── basis-space member curves ───────────────────────────────────────────────
from predict_from_latents import CACHE, build_cache  # noqa: E402

if not CACHE.exists():
    build_cache()
coef = np.load(CACHE)
assert int(np.atleast_1d(coef["version"])[0]) == 2
EDGb = coef["ell_edges"]
ctr_b = coef["ell"]
NBc = len(ctr_b)

dsn = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
NUg = np.asarray(dsn["a__pdf__pdf_bins"], float)   # canonical 22-bin nu grid


def bandS(r):
    y = P[r]["clk_resp"][ZI]
    return np.stack([np.nanmean(y[(ell >= EDGb[i]) & (ell < EDGb[i + 1])])
                     for i in range(NBc)])


def pdf_nu(r):
    """twobound raw-kappa PDF standardized onto the canonical nu grid:
    PDF_nu(nu) = sigma PDF_kappa(mu + sigma nu), (mu, sigma) implied by the
    raw histogram itself (grid spans +-9.6 sigma -> negligible truncation),
    using THIS RUN'S OWN pdf_bins grid (they drift run-to-run with map
    sigma).  Matches the nu05 convention (nu_grid.py:119, histogram of
    (m-mean)/std)."""
    kb = NG[r]["pdf_bins"]
    p = NG[r]["pdf"][ZI]
    norm = np.trapezoid(p, kb)
    mu = np.trapezoid(kb * p, kb) / norm
    sig = np.sqrt(np.trapezoid((kb - mu) ** 2 * p, kb) / norm)
    return sig * np.interp(mu + sig * NUg, kb, p, left=0.0, right=0.0)


def member_curve(nm, stat):
    rlo, rhi = pairs[nm]
    if stat == "clk":
        return bandS(rhi) - bandS(rlo)
    return pdf_nu(rhi) - pdf_nu(rlo)


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
    BBt = Bmat @ Bmat.T
    al = alpha_rel * np.trace(BBt) / Bmat.shape[0]
    return Dm @ Bmat.T @ np.linalg.inv(BBt + al * np.eye(Bmat.shape[0]))


# ── fig-05 noise verdicts (verbatim paper_s3b_clusters significance) ────────
def _spearman(Pm, Ym):
    rp = rankdata(Pm, axis=0).astype(float)
    ry = rankdata(Ym, axis=0).astype(float)
    rp = (rp - rp.mean(0)) / rp.std(0)
    ry = (ry - ry.mean(0)) / np.maximum(ry.std(0), 1e-12)
    return rp.T @ ry / len(Pm)


Xu = np.asarray(dsn["X_unit"], float)
pn_ds = [str(s) for s in dsn["param_names"]]
assert set(pn_ds) == set(pairs)
_elld = np.asarray(dsn["a__suppression__ell"], float)
_S16 = np.asarray(dsn["t__suppression__value"], float)[:, ZI, :][:, _elld <= 36864.0]
_S16 = _S16[:, :(_S16.shape[1] // 16) * 16].reshape(len(_S16), -1, 16).mean(2)
ROWS_SIG = {"clk": _S16,
            "pdf": np.asarray(dsn["t__pdf__value"], float)[:, ZI, :]}
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

# ── measured halo latents (verbatim predict_from_latents convention) ────────
cz = np.load(SB35 / "analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
OB_OM = 0.0486 / 0.3089
rows_s = dsn["run_ids"]
mt = cz["sobol_m_tot_500c_bg"][rows_s]
mg = cz["sobol_m_gas_500c_bg"][rows_s]
g2 = cz["sobol_m_gas_200c_bg"][rows_s]
ms = cz["sobol_m_star_500c"][rows_s]
Tm = cz["sobol_T_mw_500c"][rows_s]
lm = np.log10(np.where(mt > 0, mt, np.nan))
LATsob = np.full((len(rows_s), 4), np.nan)
for q in range(len(rows_s)):
    s = (lm[q] >= 13.3) & (lm[q] < 13.6)
    sg = s & (g2[q] > 0)
    if s.sum() >= 5:
        LATsob[q, 0] = np.nanmedian(((mg + ms) / mt)[q, s]) / OB_OM
        LATsob[q, 1] = np.nanmedian((ms / mt)[q, s]) / OB_OM
        LATsob[q, 3] = np.log10(np.nanmedian(np.where(Tm[q, s] > 0,
                                                      Tm[q, s], np.nan)))
    if sg.sum() >= 5:
        LATsob[q, 2] = np.nanmedian((mg / g2)[q, sg])

LNAME = [r"$\tilde f_{\rm bar}$", r"$\tilde f_\star$", r"$c_{\rm gas}$",
         r"$\log\tilde T$"]
LWORD = ["baryon fraction", "stellar fraction", "gas concentration",
         "temperature"]
FC = ["#2a78d6", "#eb6834", "#199e70", "#c98500", "#d55181"]


# ═══ the pipeline, per statistic ════════════════════════════════════════════
def run_stat(stat):
    tag = "" if stat == "clk" else "_pdf"
    slab = (r"$S(\ell)$" if stat == "clk" else r"PDF$(\nu)$")
    xlab = (r"$\ell$" if stat == "clk" else r"$\nu=\kappa/\sigma_\kappa$")
    xg = ctr_b if stat == "clk" else NUg
    print("\n" + "=" * 70 + f"\nSHIPPED FAMILY-BASIS MODEL: {stat}\n" + "=" * 70)

    fams = cluster_families(stat)
    K = len(fams)
    print("families: " + "; ".join(
        f"{f}({len(m)})" for f, m in fams))
    B = build_basis(fams, stat)

    # Sobol test set
    if stat == "clk":
        raw = np.asarray(dsn["t__suppression__value"], float)[:, ZI, :]
        Y = np.stack([np.nanmean(raw[:, (_elld >= EDGb[i])
                                     & (_elld < EDGb[i + 1])], 1)
                      for i in range(NBc)], axis=1)
    else:
        Y = np.asarray(dsn["t__pdf__value"], float)[:, ZI, :]
    fin = np.isfinite(Y).all(1)
    Yf = Y[fin]
    Ybar = Yf.mean(0)
    D = Yf - Ybar
    LAT = LATsob[fin]
    okL = np.isfinite(LAT).all(1)

    # full-family model (stage 1)
    Afull, r2r_full, r2_full = span_fit(D, B)
    print(f"full-{K} model: total R^2 {r2_full:.4f}, median per-run "
          f"{np.median(r2r_full):.4f}")

    # noise-backed families -> the SHIPPED subset
    backed = [i for i, (f, mem) in enumerate(fams)
              if any(SOLID[stat][m] for m in mem)]
    shipnames = [fams[i][0] for i in backed]
    Bs = B[backed]
    svS = np.linalg.svd(Bs, compute_uv=False)
    condS = float(svS[0] / svS[-1])
    _, r2r_ship, r2_ship = span_fit(D, Bs)
    use_ridge = condS > COND_RIDGE
    As = ridge_amps(D, Bs) if use_ridge else span_fit(D, Bs)[0]
    r2_ship_amp = 1 - (((D - As @ Bs) ** 2).sum() / (D ** 2).sum())
    print(f"SHIPPED {{{','.join(shipnames)}}} (all noise-backed families): "
          f"span R^2 {r2_ship:.4f}, cond {condS:.0f} -> "
          f"{'ridge' if use_ridge else 'OLS'} amplitudes "
          f"(model R^2 {r2_ship_amp:.4f})")

    # compression table (conditioning-aware adoption; no PCA)
    adopted = {}
    for kk in range(1, K + 1):
        scored = [(span_fit(D, B[list(c)])[2],
                   float(np.linalg.cond(B[list(c)])), c)
                  for c in combinations(range(K), kk)]
        best = max(s[0] for s in scored)
        short = [s for s in scored if s[0] >= best - TOL]
        bck = [s for s in short
               if all(i in backed for i in s[2])]
        adopted[kk] = min(bck or short, key=lambda s: s[1])
        r2a, ca, sub = adopted[kk]
        print("  k=%d: best %.4f | adopted {%s} R^2 %.4f cond %.1f"
              % (kk, best, ",".join(fams[i][0] for i in sub), r2a, ca))

    # identification of the shipped amplitudes
    Aok = As[okL]
    Lok = LAT[okL]
    CORR = np.array([[np.corrcoef(Aok[:, j], Lok[:, i])[0, 1]
                      for i in range(4)] for j in range(len(backed))])
    Lc = Lok - Lok.mean(0)
    ordL = np.random.default_rng(1).permutation(len(Lc))
    pred = np.full_like(Aok, np.nan)
    for f in np.array_split(ordL, 5):
        tr = np.setdiff1d(ordL, f)
        W, *_ = np.linalg.lstsq(np.c_[Lc[tr], np.ones(len(tr))], Aok[tr],
                                rcond=None)
        pred[f] = np.c_[Lc[f], np.ones(len(f))] @ W
    r2cv = 1 - ((Aok - pred) ** 2).sum(0) / ((Aok - Aok.mean(0)) ** 2).sum(0)
    best_i = np.argmax(np.abs(CORR), 1)
    for j, nm in enumerate(shipnames):
        i = int(best_i[j])
        rest = ", ".join(f"{LWORD[q]} {CORR[j, q]:+.2f}"
                         for q in range(4) if q != i)
        print(f"  a_{nm}: {LWORD[i]} (r={CORR[j, i]:+.2f}); {rest}; "
              f"all-4 CV R^2 {r2cv[j]:.2f}")
    D_pred = pred @ Bs
    r2_e2e = 1 - ((D[okL] - D_pred) ** 2).sum() / (D[okL] ** 2).sum()
    print(f"  end-to-end lambda -> a[CV] -> {stat}: total R^2 {r2_e2e:.4f}")

    np.savez(f"figs_preview/family_model_ship_{stat}.npz",
             x=xg, basis=B, fam_names=[f for f, _ in fams],
             shipped_idx=backed, amps_shipped=As, used_ridge=use_ridge,
             r2=(r2_full, r2_ship, r2_ship_amp, r2_e2e),
             amp_corr_latents=CORR, r2cv_latents=r2cv)

    # ── FIGURE: stages (array of full-model lines + relative residuals) ─────
    mdl = Ybar + Afull @ B
    if stat == "clk":
        key = D.mean(1)                      # suppression <-> enhancement
        klabel = "run mean deviation"
    else:
        # PDFs integrate to 1, so mean(D) ~ 0 identically; key on the
        # leading PCA mode of the deviations instead
        v1 = np.linalg.svd(D, full_matrices=False)[2][0]
        key = D @ v1
        klabel = "leading-mode amplitude"
    vmax = float(np.abs(key).max())
    cmap = mpl.cm.RdBu_r
    order = np.argsort(np.abs(key))            # extremes drawn last (on top)
    fig, (ta, tb_) = plt.subplots(
        2, 1, figsize=(ONE_COL[0], 3.4), sharex=True,
        gridspec_kw=dict(height_ratios=[2.2, 1.0], hspace=0.06))
    ext = [int(np.argmin(key)), int(np.argmax(key))]
    order = np.concatenate([[r for r in order if r not in ext], ext])
    for r in order:
        col = cmap(0.5 + 0.5 * key[r] / vmax)
        lw = 1.1 if r in ext else 0.4
        ta.plot(xg, mdl[r], color=col, lw=lw, alpha=0.85)
        rel = ((mdl[r] - Yf[r]) / Yf[r] if stat == "clk"
               else (mdl[r] - Yf[r]) / Yf[r].max())
        tb_.plot(xg, 100 * rel, color=col, lw=lw, alpha=0.7)
    if stat == "clk":
        ta.set_xscale("log")
        ta.axhline(1, color="0.85", lw=0.5, zorder=0)
    ta.set_ylabel(slab + f"  (full {K}-family model)", fontsize=6.5)
    tb_.axhline(0, color="0.8", lw=0.5, zorder=0)
    tb_.set_ylabel("(model $-$ truth)" +
                   (" / truth [%]" if stat == "clk"
                    else " / max(truth) [%]"), fontsize=5.8)
    tb_.set_xlabel(xlab, fontsize=7)
    sm = mpl.cm.ScalarMappable(cmap=cmap,
                               norm=mpl.colors.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=(ta, tb_), fraction=0.05, pad=0.02)
    cb.set_label(klabel, fontsize=5.8)
    cb.ax.tick_params(labelsize=5)
    save(fig, f"figs_v2/pfig_family_model_stages{tag}")
    plt.close(fig)

    # ── FIGURE: compression (no PCA ceiling) ────────────────────────────────
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
    # the shipped model gets its OWN marker: for pdf the scree's k=3
    # conditioning-adopted subset ({C1,C2,C4}) is NOT the shipped set
    # ({C1,C2,C3} -- the noise-backed rule bars C4), so circling the scree
    # point would mislabel what ships
    ish = len(backed)
    same = list(adopted[ish][2]) == sorted(backed)
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

    # ── FIGURE: shipped-amplitude identification scatters ───────────────────
    npan = len(backed)
    ncol = 2
    nrow = int(np.ceil(npan / ncol))
    fig, AX = plt.subplots(nrow, ncol,
                           figsize=(TWO_COL[0], 2.35 * nrow))
    AX = np.atleast_1d(AX).ravel()
    for j, nm in enumerate(shipnames):
        ax = AX[j]
        i = int(best_i[j])
        ax.scatter(Lok[:, i], Aok[:, j], s=6, color=FC[backed[j] % len(FC)],
                   alpha=0.55, rasterized=True)
        ax.set_xlabel(LNAME[i] + f"  ({LWORD[i]})", fontsize=6.5)
        ax.set_ylabel(rf"$a_{{\rm {nm}}}$", fontsize=6.5)
        ax.set_title(f"$a_{{\\rm {nm}}}$ vs {LWORD[i]}: "
                     f"$r={CORR[j, i]:+.2f}$, all-4 CV $R^2$={r2cv[j]:.2f}",
                     fontsize=6.3)
        ax.tick_params(labelsize=5.5)
        panel_label(ax, f"({'abcdef'[j]})")
    for ax in AX[npan:]:
        ax.set_visible(False)
    fig.tight_layout()
    save(fig, f"figs_v2/pfig_family_model_amps{tag}")
    plt.close(fig)


run_stat("clk")
run_stat("pdf")
print("\ndone.")
