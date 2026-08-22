"""R1 referee response: the three-rung ladder (BIND / hydro-pasted / full-hydro TNG300).

    cd papers/01_pipeline/referee/work
    /mnt/home/mlee1/venvs/BIND_env/bin/python3 r1_ladder_fig.py            # tables + figure
    /mnt/home/mlee1/venvs/BIND_env/bin/python3 r1_ladder_fig.py --tables   # numbers only

Inputs are the per-realization arrays written by r1_stats.py (recomputed from the raw
map cubes with ONE code path for every rung -- no released Cl_*.npz cache is read).
Outputs: imgs/fig06b_full_hydro.{png,pdf} and referee/work/r1_numbers.json.

Conventions are the paper's (papers/01_pipeline/{field_cache,nu_grid}.py and the
_build_figures_nb.py setup cell): ELL_LO=100, ELL_TRUST=3e4 (=0.8 ell_Nyq),
ELL_MAX_PLOT=36864, the canonical 22-bin NU grid, the v2 measured-covariance
LSST-Y10 band, Hartlap-debiased full-covariance chi2, and C(A)=(A/25 deg^2)^-1 C.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import r1_common as C  # noqa: E402

WORK = C.WORK
IMGS = Path("/mnt/home/mlee1/BIND/imgs")
OUT_STEM = IMGS / "fig06b_full_hydro"

RUNGS = ("bind", "pasted", "diffuse", "hydro_full")
NU6 = ("pdf", "peak_counts", "minima_counts", "mf_v0", "mf_v1", "mf_v2")
BANDS = (("ell<1000", 100.0, 1000.0), ("1000-5000", 1000.0, 5000.0),
         ("5000-ELL_TRUST", 5000.0, C.ELL_TRUST), (">5000(all)", 5000.0, np.inf))
AREA_25_SR, AREA_LSST_SR = 25.0 * (np.pi / 180.0) ** 2, 0.44 * 4 * np.pi
AREA_SCALE = np.sqrt(25.0 / 18000.0)          # amplitude scaling of a 25 deg^2 error
A2SR = (np.pi / 180 / 60) ** 2


# ── helpers (paper conventions) ──────────────────────────────────────────────
def rebin(a, k):
    n = (a.shape[-1] // k) * k
    return a[..., :n].reshape(*a.shape[:-1], n // k, k).mean(-1)


def paired_ratio(num, den):
    """Per-realization ratio -> (mean, SE). Seeds are shared, so this is paired."""
    r = num / np.where(den > 0, den, np.nan)
    return np.nanmean(r, 0), np.nanstd(r, 0) / np.sqrt(r.shape[0])


def band_means(ratio, ell):
    return {lab: float(np.nanmean(ratio[(ell >= lo) & (ell < hi)])) for lab, lo, hi in BANDS}


def cl_relerr(cl, ell, ngal=27.0, sige=0.26, fsky=0.44, dlnl=0.15):
    Nl = sige ** 2 * A2SR / ngal
    dl = np.maximum(ell * dlnl, 1.0)
    return np.sqrt(2.0 / ((2 * ell + 1) * dl * fsky)) * (1 + Nl / cl)


def cl_relerr_cv(ell, fsky=0.44, dlnl=0.15):
    dl = np.maximum(ell * dlnl, 1.0)
    return np.sqrt(2.0 / ((2 * ell + 1) * dl * fsky))


def chi2_lsst(res_frac, X, mask=None, name=""):
    """Hartlap-debiased chi2 of a fractional residual under the LSST-Y10 covariance.

    X = (N, d) per-realization statistic of the hydro-pasted covariance set; the
    covariance is taken in FRACTIONAL units (X/<X>), area-scaled by (A/25)^-1 with
    A = 18000 deg^2, and debiased by (N-d-2)/(N-1) (Hartlap 2007).
    """
    m = np.nanmean(X, 0)
    ok = np.isfinite(res_frac) & np.isfinite(m) & (m != 0)
    if mask is not None:
        ok &= mask
    F = X[:, ok] / m[ok]
    ok2 = np.isfinite(F).all(0) & (np.nanstd(F, 0) > 0)
    idx = np.where(ok)[0][ok2]
    F, r = X[:, idx] / m[idx], res_frac[idx]
    N, d0 = F.shape
    assert d0 <= N - 3, f"chi2_lsst({name}): d={d0} vs N={N}"
    cov = np.cov(F, rowvar=False) * (25.0 / 18000.0)
    # Eigen-truncated inverse. The PDF data vector carries an EXACT linear
    # constraint (density=True => sum_i p_i dnu = 1), so its covariance is
    # singular and a plain solve() returns garbage (chi2 ~ -1e12). Modes below
    # 1e-10 of the largest eigenvalue are dropped for every statistic alike;
    # only the PDF actually loses any (its normalization direction).
    w, V = np.linalg.eigh(cov)
    keep = w > 1e-10 * w.max()
    d = int(keep.sum())
    h = (N - d - 2) / (N - 1)
    chi2 = float(np.sum((V[:, keep].T @ r) ** 2 / w[keep])) * h
    return {"chi2": chi2, "dof": d, "chi2_dof": chi2 / d, "N_cov": int(N),
            "hartlap": h, "n_bins": int(d0), "n_modes_dropped": int(d0 - d)}


def chi2_paired(num, den, mask):
    """chi2/dof of the paired residual against the +-1sigma/sqrt(N) band (fig 6 style)."""
    rat = num / np.where(den > 0, den, np.nan)
    res = np.nanmean(num, 0) / np.nanmean(den, 0) - 1
    band = np.nanstd(rat, 0) / np.sqrt(rat.shape[0])
    z = res[mask] / np.where(band[mask] > 0, band[mask], np.nan)
    return float(np.nanmean(z ** 2))


# ── load ─────────────────────────────────────────────────────────────────────
def load_all():
    d = {"kappa": {}, "y": {}, "tau": {}}
    for r in RUNGS + ("dmo",):
        p = WORK / f"kappa_{r}.npz"
        if p.exists():
            d["kappa"][r] = dict(np.load(p))
    for f in ("y", "tau"):
        for r in RUNGS:
            p = WORK / f"{f}_{r}.npz"
            if p.exists():
                d[f][r] = dict(np.load(p))
    d["cov"] = dict(np.load(WORK / "cov_pasted550.npz"))
    d["ell"] = d["kappa"]["bind"]["ell"]
    d["nu"] = d["kappa"]["bind"]["nu"]
    return d


# ── numbers ──────────────────────────────────────────────────────────────────
def compute(D, zi=C.ZI):
    ell, nu = D["ell"], D["nu"]
    out = {"zi": zi, "z_s": C.ZS[zi], "n_real": C.N_REAL, "ell": ell.tolist(),
           "nu": nu.tolist()}
    trust = (ell >= C.ELL_LO) & (ell <= C.ELL_TRUST)

    # ---- cross-check of the existing validation summary --------------------
    xc = {"mean_ratio": {}, "cl_band_ratio": {}}
    for f in ("y", "tau"):
        for r in ("pasted", "diffuse"):
            xc["mean_ratio"][f"{f}:{r}/hydro_full"] = [
                float(D[f][r]["mean"][:, z].mean() / D[f]["hydro_full"]["mean"][:, z].mean())
                for z in range(5)]
    for f, key in (("kappa", "cl_kk"), ("y", "cl_y"), ("tau", "cl_tau")):
        for r in ("pasted", "diffuse", "bind"):
            m, _ = paired_ratio(D[f][r][key][:, zi], D[f]["hydro_full"][key][:, zi])
            xc["cl_band_ratio"][f"{f}:{r}/hydro_full"] = band_means(m, ell)
    out["crosscheck"] = xc

    # ---- kappa sector: S(ell) and the paired ratios -------------------------
    dmo = np.nanmean(D["kappa"]["dmo"]["cl_kk"][:, zi], 0)
    S = {r: D["kappa"][r]["cl_kk"][:, zi] / np.where(dmo > 0, dmo, np.nan) for r in RUNGS}
    out["S_mean"] = {r: np.nanmean(v, 0).tolist() for r, v in S.items()}
    out["S_dmo_ref"] = dmo.tolist()

    # LSST-Y10 band on a Cl-type statistic: v2 recipe (measured covariance of the
    # pasted set, 8-bin block average, area-scaled) + analytic shape-noise excess.
    cov_cl = D["cov"]["cl_kk"]                                   # (550, n_ell)
    lrb = rebin(np.log(np.where(cov_cl > 0, cov_cl, np.nan)), 8)
    ell_rb = rebin(ell[None, :], 8)[0]
    sig_meas = np.interp(ell, ell_rb, np.nanstd(lrb, 0) * np.sqrt(AREA_25_SR / AREA_LSST_SR))
    kkt = np.nanmean(D["kappa"]["pasted"]["cl_kk"][:, zi], 0)
    shot = cl_relerr(kkt, ell) - cl_relerr_cv(ell)
    lsst_kk = 100 * np.sqrt(sig_meas ** 2 + shot ** 2)
    out["lsst_kk_pct"] = lsst_kk.tolist()
    out["lsst_kk_trace"] = {
        f"ell={ell[int(np.argmin(np.abs(ell - t)))]:.0f}": {
            "band_pct": float(lsst_kk[int(np.argmin(np.abs(ell - t)))]),
            "measured_leg_pct": float(100 * sig_meas[int(np.argmin(np.abs(ell - t)))]),
            "shot_excess_pct": float(100 * shot[int(np.argmin(np.abs(ell - t)))])}
        for t in (100, 300, 1000, 5000, 20000)}

    # ---- pairwise ratios for every statistic --------------------------------
    pairs = [("bind", "pasted"), ("bind", "hydro_full"), ("pasted", "hydro_full"),
             ("diffuse", "hydro_full"), ("bind", "diffuse")]
    tab = {}
    for f, key in (("kappa", "cl_kk"), ("y", "cl_y"), ("tau", "cl_tau")):
        for a, b in pairs:
            if a not in D[f] or b not in D[f]:
                continue
            m, se = paired_ratio(D[f][a][key][:, zi], D[f][b][key][:, zi])
            tab[f"Cl_{f}:{a}/{b}"] = {**band_means(m, ell),
                                      "trusted": float(np.nanmean(m[trust])),
                                      "se_trusted": float(np.nanmean(se[trust]))}
    for a, b in pairs:                                    # kappa x y cross
        k = f"cl_ky_z{zi}"
        if k in D["y"].get(a, {}) and k in D["y"].get(b, {}):
            m, _ = paired_ratio(D["y"][a][k], D["y"][b][k])
            tab[f"Cl_ky:{a}/{b}"] = {**band_means(m, ell), "trusted": float(np.nanmean(m[trust]))}
    out["cl_ratio_table"] = tab

    # nu-domain: mean statistic and ratios
    numean, nurat = {}, {}
    for k in NU6:
        for r in RUNGS + ("dmo",):
            numean[f"{k}:{r}"] = np.nanmean(D["kappa"][r][f"{k}_z{zi}"], 0).tolist()
        for a, b in pairs[:4]:
            A = np.nanmean(D["kappa"][a][f"{k}_z{zi}"], 0)
            B = np.nanmean(D["kappa"][b][f"{k}_z{zi}"], 0)
            nurat[f"{k}:{a}/{b}"] = (A / np.where(B != 0, B, np.nan)).tolist()
    out["nu_mean"], out["nu_ratio"] = numean, nurat

    # ---- RESPONSE CAPTURE: the number that replaces the cited "~90%" ---------
    # The DMO->hydro response of a statistic X is R_full = X_full - X_DMO. The
    # halo-replacement construction captures R_pasted = X_pasted - X_DMO of it;
    # BIND captures R_bind. We report (i) the band/least-squares projection
    # <R_x, R_full>/<R_full, R_full> and (ii) the median per-bin ratio.
    cap = {}
    Sm = {r: np.nanmean(S[r], 0) for r in RUNGS}
    for r in ("pasted", "bind"):
        row = {}
        for lab, lo, hi in BANDS:
            m = (ell >= lo) & (ell < hi)
            num, den = np.nansum(Sm[r][m] - 1), np.nansum(Sm["hydro_full"][m] - 1)
            row[lab] = float(num / den) if abs(den) > 1e-8 else np.nan
        m = trust
        row["trusted_proj"] = float(np.nansum((Sm[r][m] - 1) * (Sm["hydro_full"][m] - 1))
                                    / np.nansum((Sm["hydro_full"][m] - 1) ** 2))
        row["proj"] = row["trusted_proj"]        # same key as the nu statistics
        cap[f"S(ell):{r}"] = row
    for k in NU6:
        Dm = np.nanmean(D["kappa"]["dmo"][f"{k}_z{zi}"], 0)
        Fm = np.nanmean(D["kappa"]["hydro_full"][f"{k}_z{zi}"], 0)
        base = np.nanmean(D["cov"][k], 0) if k in D["cov"] else Fm
        use = np.isfinite(Fm - Dm)
        if k in ("peak_counts", "minima_counts"):
            use &= Fm > 5
        elif k == "pdf":
            use &= Fm > 1e-4
        else:
            use &= np.abs(Fm) > 0.02 * np.nanmax(np.abs(Fm))
        Rf = (Fm - Dm)[use]
        for r in ("pasted", "bind"):
            Rx = (np.nanmean(D["kappa"][r][f"{k}_z{zi}"], 0) - Dm)[use]
            cap[f"{k}:{r}"] = {
                "proj": float(np.nansum(Rx * Rf) / np.nansum(Rf ** 2)),
                "median_ratio": float(np.nanmedian(Rx / np.where(np.abs(Rf) > 0, Rf, np.nan))),
                "n_bins": int(use.sum()),
                "response_size_pct": float(100 * np.nanmedian(np.abs(Rf / Fm[use])))}
        _ = base
    # bootstrap CI on the projection, resampling the 50 realizations with the SAME
    # indices in every rung (the sets are seed-paired; breaking the pairing would
    # inject the cosmic variance the ladder is built to cancel).
    rng = np.random.default_rng(1992)
    nb, nr = 400, C.N_REAL
    for k in NU6 + ("S(ell)",):
        boots = {"pasted": [], "bind": []}
        for _ in range(nb):
            i = rng.integers(0, nr, nr)
            if k == "S(ell)":
                Fm = np.nanmean(S["hydro_full"][i], 0) - 1
                for r in ("pasted", "bind"):
                    Rx = np.nanmean(S[r][i], 0) - 1
                    boots[r].append(np.nansum(Rx[trust] * Fm[trust])
                                    / np.nansum(Fm[trust] ** 2))
                continue
            Dm = np.nanmean(D["kappa"]["dmo"][f"{k}_z{zi}"][i], 0)
            Fm = np.nanmean(D["kappa"]["hydro_full"][f"{k}_z{zi}"][i], 0) - Dm
            u = np.isfinite(Fm)
            if k in ("peak_counts", "minima_counts"):
                u &= np.nanmean(D["kappa"]["hydro_full"][f"{k}_z{zi}"], 0) > 5
            elif k == "pdf":
                u &= np.nanmean(D["kappa"]["hydro_full"][f"{k}_z{zi}"], 0) > 1e-4
            else:
                mx = np.nanmax(np.abs(np.nanmean(D["kappa"]["hydro_full"][f"{k}_z{zi}"], 0)))
                u &= np.abs(np.nanmean(D["kappa"]["hydro_full"][f"{k}_z{zi}"], 0)) > 0.02 * mx
            for r in ("pasted", "bind"):
                Rx = np.nanmean(D["kappa"][r][f"{k}_z{zi}"][i], 0) - Dm
                boots[r].append(np.nansum(Rx[u] * Fm[u]) / np.nansum(Fm[u] ** 2))
        for r in ("pasted", "bind"):
            key = f"S(ell):{r}" if k == "S(ell)" else f"{k}:{r}"
            lo, hi = np.percentile(boots[r], [16, 84])
            cap[key]["boot_p16"], cap[key]["boot_p84"] = float(lo), float(hi)
    out["response_capture"] = cap

    # LSST-Y10 band per nu statistic: sample variance of the 550-real pasted set,
    # area-scaled (paper convention (3); the ngal=10 shape-noise excess that the
    # paper adds for peaks/minima is quoted separately in the memo).
    lsst_nu = {}
    for k in NU6:
        X = D["cov"][k]
        m = np.nanmean(X, 0)
        lsst_nu[k] = (100 * np.nanstd(X, 0) * AREA_SCALE
                      / np.where(m != 0, np.abs(m), np.nan)).tolist()
    out["lsst_nu_pct"] = lsst_nu

    # ---- chi2 under the LSST-Y10 covariance ---------------------------------
    chi = {}
    for a, b in [("bind", "hydro_full"), ("pasted", "hydro_full"), ("bind", "pasted"),
                 ("diffuse", "hydro_full")]:
        rb = rebin(D["kappa"][a]["cl_kk"][:, zi][:, trust], 16)
        rt = rebin(D["kappa"][b]["cl_kk"][:, zi][:, trust], 16)
        res = np.nanmean(rb, 0) / np.nanmean(rt, 0) - 1
        Xc = rebin(D["cov"]["cl_kk"][:, trust], 16)
        chi[f"Cl_kappa:{a}-{b}"] = {**chi2_lsst(res, Xc, name=f"kk {a}{b}"),
                                    "chi2_dof_paired": chi2_paired(
                                        D["kappa"][a]["cl_kk"][:, zi],
                                        D["kappa"][b]["cl_kk"][:, zi], trust)}
        for k in NU6:
            A = np.nanmean(D["kappa"][a][f"{k}_z{zi}"], 0)
            B = np.nanmean(D["kappa"][b][f"{k}_z{zi}"], 0)
            res = A / np.where(B != 0, B, np.nan) - 1
            m = np.nanmean(D["cov"][k], 0)
            msk = np.isfinite(res) & (np.abs(m) > 1e-12)
            if k in ("peak_counts", "minima_counts"):
                msk &= m > 5                     # bins with a usable count
            chi[f"{k}:{a}-{b}"] = chi2_lsst(np.nan_to_num(res), D["cov"][k], msk, name=k)
    for f, key in (("y", "cl_y"), ("tau", "cl_tau")):
        for a, b in [("bind", "hydro_full"), ("pasted", "hydro_full"),
                     ("diffuse", "hydro_full"), ("bind", "pasted")]:
            chi[f"Cl_{f}:{a}-{b}"] = {"chi2_dof_paired": chi2_paired(
                D[f][a][key][:, zi], D[f][b][key][:, zi], trust)}
    out["chi2"] = chi
    return out


# ── figure ───────────────────────────────────────────────────────────────────
def make_figure(D, R, zi=C.ZI, nu_panels=("peak_counts", "mf_v2")):
    from paper_style import COLORS, TWO_COL, panel_label, setup
    setup()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator

    ell, nu = D["ell"], np.asarray(D["nu"])
    CB, CT = COLORS["bind"], COLORS["truth"]
    CF, CD, CG = COLORS["highlight"], COLORS["secondary"], COLORS["dmo"]
    STYLE = {"bind": dict(color=CB, lw=1.0, ls="-"),
             "pasted": dict(color=CT, lw=0.9, ls="--"),
             "diffuse": dict(color=CD, lw=0.9, ls="-."),
             "hydro_full": dict(color=CF, lw=1.1, ls="-")}
    LBL = {"bind": "BIND", "pasted": "hydro-pasted", "diffuse": "pasted + diffuse gas",
           "hydro_full": "full hydro TNG300"}
    lsst_kk = np.asarray(R["lsst_kk_pct"])
    trust = (ell >= C.ELL_LO) & (ell <= C.ELL_TRUST)

    # kappa is IDENTICAL in the pasted and diffuse traces by construction (the
    # diffuse gas carries no mass into the lensing planes; measured max
    # |dkappa| = 3e-8 vs max |kappa| = 0.82), so the kappa-sector panels omit
    # the diffuse rung -- it would draw exactly on top of the pasted curve.
    # draw order: the reference (full hydro, thick) FIRST, the two rungs on top,
    # so a rung that lands exactly on the reference is still visible.
    KRUNGS = ("hydro_full", "bind", "pasted")
    GRUNGS = ("hydro_full", "diffuse", "bind", "pasted")

    fig = plt.figure(figsize=(TWO_COL[0], 7.3))
    gs = fig.add_gridspec(6, 3, height_ratios=[2.0, 1.0, 0.42, 2.0, 1.0, 0.85],
                          hspace=0.16, wspace=0.40)

    def strip(ar, x, curves, band=None, ylab="", rlim=None, logx=True):
        ar.axhline(0, color=CG, lw=0.6)
        if band is not None:
            ar.fill_between(x, -band, band, color=CB, alpha=0.20, lw=0)
        for rung, y in curves:
            ar.plot(x, y, **STYLE[rung])
        if logx:
            ar.set_xscale("log")
        if rlim:
            ar.set_ylim(*rlim)
        ar.set_ylabel(ylab, fontsize=6)
        ar.tick_params(labelsize=6)

    # ── (a) S(ell) --------------------------------------------------------
    a = fig.add_subplot(gs[0, 0])
    ar = fig.add_subplot(gs[1, 0], sharex=a)
    for rung in KRUNGS:
        a.plot(ell, R["S_mean"][rung], **STYLE[rung], label=LBL[rung])
    a.axhline(1.0, color=CG, lw=0.6, zorder=0)
    a.set_xscale("log")
    a.set_xlim(C.ELL_LO, C.ELL_MAX_PLOT)
    a.set_ylim(0.80, 1.035)
    a.set_ylabel(r"$S(\ell)=C_\ell^{\kappa\kappa}/C_\ell^{\kappa\kappa,\rm DMO}$", fontsize=7)
    plt.setp(a.get_xticklabels(), visible=False)
    a.tick_params(labelsize=6)
    a.axvline(C.ELL_TRUST, color="0.55", lw=0.5, ls=":")
    a.text(C.ELL_TRUST, 0.985, r"$0.8\,\ell_{\rm Nyq}$", transform=a.get_xaxis_transform(),
           fontsize=4.6, color="0.4", ha="right", va="top")
    panel_label(a, "(a)", loc="lower left")
    cur = []
    for rung in ("bind", "pasted", "diffuse"):
        m, _ = paired_ratio(D["kappa"][rung]["cl_kk"][:, zi],
                            D["kappa"]["hydro_full"]["cl_kk"][:, zi])
        if rung != "diffuse":       # diffuse kappa == pasted kappa by construction
            cur.append((rung, 100 * (m - 1)))
    strip(ar, ell, cur, band=lsst_kk, ylab=r"$\Delta$ vs full hydro [%]", rlim=(-8, 26))
    ar.set_xlabel(r"$\ell$", fontsize=7)
    ar.set_xlim(C.ELL_LO, C.ELL_MAX_PLOT)
    ar.axvline(C.ELL_TRUST, color="0.55", lw=0.5, ls=":")

    # ── (b), (c) nu-domain -------------------------------------------------
    NULAB = {"peak_counts": r"$N_{\rm pk}(\nu)$", "minima_counts": r"$N_{\rm min}(\nu)$",
             "pdf": r"$p(\nu)$", "mf_v0": r"$V_0(\nu)$", "mf_v1": r"$V_1(\nu)$",
             "mf_v2": r"$V_2(\nu)$"}
    k = nu_panels[0]
    a = fig.add_subplot(gs[0, 1])
    ar = fig.add_subplot(gs[1, 1], sharex=a)
    for rung in KRUNGS:
        a.plot(nu, R["nu_mean"][f"{k}:{rung}"], **STYLE[rung])
    if k in ("peak_counts", "minima_counts", "pdf"):
        a.set_yscale("log")
    a.set_ylabel(NULAB[k], fontsize=7)
    plt.setp(a.get_xticklabels(), visible=False)
    a.tick_params(labelsize=6)
    panel_label(a, "(b)", loc="upper right")
    band = np.asarray(R["lsst_nu_pct"][k])
    cur = [(r, 100 * (np.asarray(R["nu_ratio"][f"{k}:{r}/hydro_full"]) - 1))
           for r in ("bind", "pasted")]
    strip(ar, nu, cur, band=np.where(np.isfinite(band), band, np.nan),
          ylab=r"$\Delta$ [%]", rlim=(-16, 16), logx=False)
    ar.set_xlabel(r"$\nu=\kappa/\sigma_0$", fontsize=7)
    ar.set_xlim(-2.0, 6.0)
    a.set_xlim(-2.0, 6.0)

    # ── (c) how much of the DMO -> full-hydro response each rung captures ----
    ac = fig.add_subplot(gs[0:2, 2])
    ROWS = [("S(ell)", r"$S(\ell)$"), ("pdf", r"$p(\nu)$"),
            ("peak_counts", r"$N_{\rm pk}$"), ("minima_counts", r"$N_{\rm min}$"),
            ("mf_v0", r"$V_0$"), ("mf_v1", r"$V_1$"), ("mf_v2", r"$V_2$")]
    yy = np.arange(len(ROWS))[::-1]
    ac.axvline(1.0, color=CF, lw=1.1, zorder=0)
    ac.axvline(0.0, color=CG, lw=0.8, ls=":", zorder=0)
    for rung, dx, mk in (("pasted", +0.16, "s"), ("bind", -0.16, "o")):
        v = [R["response_capture"][f"{r}:{rung}"] for r, _ in ROWS]
        c = [x["proj"] for x in v]
        lo = [x["proj"] - x["boot_p16"] for x in v]
        hi = [x["boot_p84"] - x["proj"] for x in v]
        ac.errorbar(c, yy + dx, xerr=[lo, hi], ls="none", marker=mk, ms=3.0,
                    color=STYLE[rung]["color"], elinewidth=0.8, capsize=1.4,
                    mfc=STYLE[rung]["color"], mec=STYLE[rung]["color"])
    ac.set_yticks(yy)
    ac.set_yticklabels([lab for _, lab in ROWS], fontsize=6.5)
    ac.set_ylim(-0.6, len(ROWS) - 0.4)
    ac.set_xlim(0.0, 1.65)
    ac.set_xlabel("captured fraction of the\nDMO $\\rightarrow$ full-hydro response", fontsize=6.5)
    ac.tick_params(labelsize=6)
    ac.text(1.0, len(ROWS) - 0.42, "full response", fontsize=5.6, color=CF,
            ha="center", va="bottom")
    for yv in yy:
        ac.axhline(yv, color="0.90", lw=0.4, zorder=-2)
    panel_label(ac, "(c)", loc="lower right")

    # ── (d), (e) gas spectra ------------------------------------------------
    f_ell = ell * (ell + 1) / (2 * np.pi)
    for j, (fld, key, ylab, rlim) in enumerate(
            [("y", "cl_y", r"$\ell(\ell{+}1)C_\ell^{yy}/2\pi$", (-32, 32)),
             ("tau", "cl_tau", r"$\ell(\ell{+}1)C_\ell^{\tau\tau}/2\pi$", (-80, 265))]):
        a = fig.add_subplot(gs[3, j])
        ar = fig.add_subplot(gs[4, j], sharex=a)
        for rung in GRUNGS:
            a.plot(ell, f_ell * np.nanmean(D[fld][rung][key][:, zi], 0), **STYLE[rung])
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(C.ELL_LO, C.ELL_MAX_PLOT)
        a.set_ylabel(ylab, fontsize=7)
        a.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
        a.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(3, 0.4)))
        plt.setp(a.get_xticklabels(), visible=False)
        a.tick_params(labelsize=6)
        a.axvline(C.ELL_TRUST, color="0.55", lw=0.5, ls=":")
        panel_label(a, f"({'de'[j]})", loc="lower left")
        cur, band = [], None
        for rung in ("bind", "pasted", "diffuse"):
            m, se = paired_ratio(D[fld][rung][key][:, zi], D[fld]["hydro_full"][key][:, zi])
            cur.append((rung, 100 * (m - 1)))
            if rung == "pasted":
                band = 100 * se
        strip(ar, ell, cur, band=None, ylab=r"$\Delta$ vs full hydro [%]" if j == 0 else "",
              rlim=rlim)
        ar.fill_between(ell, -band, band, color=CG, alpha=0.45, lw=0)
        ar.set_xlabel(r"$\ell$", fontsize=7)
        ar.set_xlim(C.ELL_LO, C.ELL_MAX_PLOT)
        ar.axvline(C.ELL_TRUST, color="0.55", lw=0.5, ls=":")

    # ── (f) mean-column ladder vs z_s ---------------------------------------
    a = fig.add_subplot(gs[3:5, 2])
    zs = np.asarray(C.ZS)
    for fld, mk, ls in (("y", "o", "-"), ("tau", "s", "--")):
        for rung in ("diffuse", "bind", "pasted"):
            v = np.asarray(R["crosscheck"]["mean_ratio"].get(f"{fld}:{rung}/hydro_full",
                                                             [np.nan] * 5)) \
                if rung != "bind" else np.array(
                [D[fld]["bind"]["mean"][:, z].mean() / D[fld]["hydro_full"]["mean"][:, z].mean()
                 for z in range(5)])
            a.plot(zs, v, marker=mk, ms=2.6, ls=ls, color=STYLE[rung]["color"], lw=0.9)
    a.axhline(1.0, color=CF, lw=1.1)
    a.set_yscale("log")
    a.set_ylim(0.12, 1.6)
    a.set_xlabel(r"$z_s$", fontsize=7)
    a.set_ylabel(r"mean map value / full hydro", fontsize=7)
    a.tick_params(labelsize=6)
    a.set_yticks([0.2, 0.3, 0.5, 0.7, 1.0, 1.5])
    a.set_yticklabels(["0.2", "0.3", "0.5", "0.7", "1.0", "1.5"])
    panel_label(a, "(f)", loc="lower left")
    a.text(0.5, 0.14, r"$\bar{y}$ (circles),  $\bar{\tau}$ (squares)", transform=a.transAxes,
           fontsize=5.6, ha="center", color="0.35")

    # ── legend row ----------------------------------------------------------
    alg = fig.add_subplot(gs[5, :])
    alg.axis("off")
    h = [plt.Line2D([], [], **STYLE[r], label=LBL[r]) for r in RUNGS]
    h += [plt.Rectangle((0, 0), 1, 1, fc=CB, alpha=0.20, ec="none",
                        label="LSST-Y10 precision (18000 deg$^2$)"),
          plt.Rectangle((0, 0), 1, 1, fc=CG, alpha=0.45, ec="none",
                        label=rf"paired $\pm1\sigma/\sqrt{{{C.N_REAL}}}$ (gas)")]
    alg.legend(handles=h, loc="center", ncol=3, fontsize=5.8, handlelength=2.0,
               labelspacing=0.45, columnspacing=1.6)
    fig.text(0.5, 0.905, "", fontsize=1)   # keep tight-bbox stable
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true", help="numbers only, no figure")
    ap.add_argument("--zi", type=int, default=C.ZI)
    ap.add_argument("--nu_panels", default="peak_counts,mf_v2")
    a = ap.parse_args()
    D = load_all()
    R = compute(D, a.zi)
    (Path(__file__).resolve().parent / f"r1_numbers_z{a.zi}.json").write_text(
        json.dumps(R, indent=1))

    print(f"\n=== cross-check: mean map value ratios (z_s = {C.ZS}) ===")
    for k, v in R["crosscheck"]["mean_ratio"].items():
        print(f"  {k:28s} " + " ".join(f"{x:.3f}" for x in v))
    print(f"\n=== cross-check: paired Cl band ratios at z_s={C.ZS[a.zi]} ===")
    for k, v in R["crosscheck"]["cl_band_ratio"].items():
        print(f"  {k:28s} " + " ".join(f"{lab}={v[lab]:.3f}" for lab in list(v)[:3]))
    print("\n=== Cl ratio table (band means) ===")
    for k, v in R["cl_ratio_table"].items():
        print(f"  {k:34s} " + " ".join(f"{lab}={v[lab]:7.3f}" for lab in
                                       ("ell<1000", "1000-5000", "5000-ELL_TRUST", "trusted")))
    print("\n=== chi2 (LSST-Y10 cov from 550 pasted reals, Hartlap) ===")
    for k, v in R["chi2"].items():
        if "chi2_dof" in v:
            print(f"  {k:34s} chi2/dof={v['chi2_dof']:10.2f} (d={v['dof']:2d})"
                  + (f"   paired chi2/dof={v['chi2_dof_paired']:8.2f}"
                     if "chi2_dof_paired" in v else ""))
        else:
            print(f"  {k:34s} paired chi2/dof={v['chi2_dof_paired']:10.2f}")
    print("\n=== response capture: (X - DMO)/(full hydro - DMO) ===")
    for k, v in R["response_capture"].items():
        if k.startswith("S(ell)"):
            print(f"  {k:24s} " + " ".join(f"{lab}={v[lab]:6.3f}" for lab in
                                           ("ell<1000", "1000-5000", "5000-ELL_TRUST",
                                            "trusted_proj")))
        else:
            print(f"  {k:24s} proj={v['proj']:6.3f}  median={v['median_ratio']:6.3f}  "
                  f"(n_bins={v['n_bins']}, |response| {v['response_size_pct']:.2f}% of signal)")

    print("\n=== nu-domain % residual vs full hydro (trusted nu<=4) ===")
    nu = np.asarray(R["nu"])
    m4 = nu <= 4.0
    for k in NU6:
        for r in ("bind", "pasted"):
            q = 100 * (np.asarray(R["nu_ratio"][f"{k}:{r}/hydro_full"]) - 1)
            print(f"  {k:14s} {r:8s} median={np.nanmedian(q[m4]):7.2f}%  "
                  f"max|.|={np.nanmax(np.abs(q[m4])):7.2f}%  "
                  f"LSST band median={np.nanmedian(np.asarray(R['lsst_nu_pct'][k])[m4]):6.2f}%")

    if not a.tables:
        fig = make_figure(D, R, a.zi, tuple(a.nu_panels.split(",")))
        IMGS.mkdir(exist_ok=True)
        fig.savefig(f"{OUT_STEM}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.02)
        print(f"\nwrote {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
