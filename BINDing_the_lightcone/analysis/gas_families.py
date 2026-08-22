"""Extend the response-family machinery to the gas auto- and cross-spectra.

The paper's family decomposition (S4.2) and family-basis model (S5) cover the
seven WL statistics. This script runs the identical pipeline on the five gas
spectra -- C_yy, C_tautau, C_ky, C_ktau, C_ytau -- with the same rules:

  1P responses     (Cl_hi - Cl_lo)/Cl_fid from the twobound pairs, banded
  entry gate       peak |dCl| / sigma_pair >= 3
  clustering       average linkage on 1 - r of extremum-normalized curves, t=0.15
  weights          peak |Spearman rho| per parameter across the 256 Sobol nodes
  retention        families with >=1 member above the 95th-pct permutation null
  span R^2         free-amplitude OLS on the 256 Sobol deviations
  CV R^2           the SHIPPED eight latents (selected on the WL statistics
                   only -- this is a transfer test, not a refit), rng(1) folds

Data notes (verified 2026-08-14): all 60 twobound runs carry Cl_kappa_y.npz
(cl_ky z_s-resolved + cl_yy integrated); runs 0000-0007 -- the WindEnergy,
RadioFeedbackFactor, VarWindVelFactor, RadioFeedbackReiorient pairs, traced
before the tau planes were added -- lack Cl_tau.npz, so the tau-sector
clustering runs on 26 of 30 parameters. Each affected family retains several
measured members, and the Sobol span R^2 measures directly whether the
missing legs cost basis coverage. Cross-norms are consistent across all 60
runs (max/min 1.14 at ell~500; no pre-xpkfix outliers).

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python gas_families.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

P1 = Path("/mnt/home/mlee1/BIND/papers/01_pipeline")
CEPH = Path("/mnt/home/mlee1/ceph")
# BIND_CAMPAIGN=n1000: per-run STATS come from the campaign tree; the design
# (twobound_params.npy, run_0049/params.npy) is campaign-independent and stays
# on the sci50 tree, which is the only place it is stored.
CAMPAIGN = os.environ.get("BIND_CAMPAIGN", "sci50")
TB_DESIGN = CEPH / "bind_science/runs/twobound"
TB = (CEPH / "bind_n1000/twobound" if CAMPAIGN == "n1000" else TB_DESIGN)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

ZI = 1
SNR_GATE, T_CLUST = 3.0, 0.15
RNG = np.random.default_rng(20260814)

# ── shared inputs ───────────────────────────────────────────────────────────
dsn = np.load(CEPH / ("bind_n1000/emulator_dataset_n1000.npz"
                      if CAMPAIGN == "n1000" else
                      "bind_sb35/emulator_dataset_nu05.npz"), allow_pickle=True)
PN = [str(x) for x in dsn["param_names"]]
XU = np.asarray(dsn["X_unit"], float)                     # (n_nodes, 30)
ELL = np.asarray(dsn["a__cl_yy__ell"], float)
# campaign row -> design row (run_id).  The sci50-derived model arrays (X_ALL)
# are row-indexed by run_id; the campaign dataset can hold a non-contiguous
# subset until tracing finishes, so every design-side array is subset by SEL
# before pairing it positionally with dsn rows.
SEL = np.asarray(dsn["run_ids"], int)
SEARCH = np.load(P1 / "figs_preview/agnostic_lambda_results_obs.npz", allow_pickle=True)
NAMES = [str(x) for x in SEARCH["names"]]
CHOSEN = list(SEARCH["chosen"])
X_ALL = SEARCH["X_ALL"]
LAT = (X_ALL - X_ALL.mean(0)) / X_ALL.std(0)   # standardized on the FULL design
LAT = LAT[:, CHOSEN][SEL]                      # campaign-aligned rows
CMAP = pd.read_csv(P1 / "figs_preview/param_color_map.csv")
CTX = dict(zip(CMAP["ParamName"], CMAP["Context"]))
CTXCOL = dict(zip(CMAP["Context"], CMAP["Color"]))

# the shipped clk band edges, for convention consistency
EDG = np.load(P1 / "latent_model_coeffs.npz")["ell_edges"]
print(f"bands: {len(EDG)-1} log bands over ell {EDG[0]:.0f}-{EDG[-1]:.0f} (shipped clk grid)")
BIDX = [np.where((ELL >= EDG[i]) & (ELL < EDG[i + 1]))[0] for i in range(len(EDG) - 1)]
XG = np.array([np.sqrt(EDG[i] * EDG[i + 1]) for i in range(len(EDG) - 1)])


def band(v):
    return np.array([np.nanmean(v[..., idx], axis=-1) for idx in BIDX]).T \
        if v.ndim > 1 else np.array([np.nanmean(v[idx]) for idx in BIDX])


def band_err(e):
    """band-average of a per-bin sigma, reduced by sqrt(n_bins_in_band)."""
    return np.array([np.nanmean(e[idx]) / np.sqrt(max(len(idx), 1)) for idx in BIDX])


# ── the twobound pairing (design tree: campaign-independent) ────────────────
tbp = np.load(TB_DESIGN / "twobound_params.npy")
fidp = np.load(TB_DESIGN / "run_0049/params.npy")
COSMO = {0, 1, 6, 7, 8}
AIDX = [i for i in range(35) if i not in COSMO]
pairs: dict[str, dict[str, int]] = {}
for r in range(60):
    d = np.where(~np.isclose(tbp[r], fidp))[0]
    if len(d) == 0:
        continue                                   # a fiducial replica (18/49/53)
    assert len(d) == 1, f"run {r} varies {len(d)} params"
    nm = PN[AIDX.index(d[0])]
    side = "hi" if tbp[r, d[0]] > fidp[d[0]] else "lo"
    pairs.setdefault(nm, {})[side] = r
print(f"pairs: {len(pairs)} parameters "
      f"({sum('hi' in v and 'lo' in v for v in pairs.values())} two-sided)")

# ── load the gas legs ───────────────────────────────────────────────────────
STATS = {
    "yy": dict(file="Cl_kappa_y.npz", key="cl_yy", ek="cl_yy_err", zsel=None,
               tgt="t__cl_yy__value", lab=r"$C_\ell^{yy}$"),
    "ky": dict(file="Cl_kappa_y.npz", key="cl_ky", ek="cl_ky_err", zsel=ZI,
               tgt="t__cl_kappa_y__value", lab=r"$C_\ell^{\kappa y}$"),
    "tt": dict(file="Cl_tau.npz", key="cl_tt", ek="cl_tt_err", zsel=None,
               tgt="t__cl_tt__value", lab=r"$C_\ell^{\tau\tau}$"),
    "kt": dict(file="Cl_tau.npz", key="cl_kt", ek="cl_kt_err", zsel=ZI,
               tgt="t__cl_kappa_tau__value", lab=r"$C_\ell^{\kappa\tau}$"),
    "yt": dict(file="Cl_tau.npz", key="cl_yt", ek="cl_yt_err", zsel=None,
               tgt="t__cl_yt__value", lab=r"$C_\ell^{y\tau}$"),
}


def run_leg(r, spec):
    f = TB / f"run_{r:04d}" / spec["file"]
    if not f.exists():
        return None, None
    d = np.load(f)
    v, e = d[spec["key"]], d[spec["ek"]]
    if spec["zsel"] is not None:
        v, e = v[spec["zsel"]], e[spec["zsel"]]
    return band(np.asarray(v, float)), band_err(np.asarray(e, float))


RESULTS = {}
for st, spec in STATS.items():
    fid_v, fid_e = run_leg(49, spec)
    resp, snr, present = {}, {}, []
    for nm, pr in pairs.items():
        hi = run_leg(pr["hi"], spec) if "hi" in pr else (fid_v, fid_e)
        lo = run_leg(pr["lo"], spec) if "lo" in pr else (fid_v, fid_e)
        if hi[0] is None or lo[0] is None:
            continue
        dv = hi[0] - lo[0]
        # gas responses are order-unity and multiplicative: the additive
        # variable is ln Cl (for small responses this reduces to dCl/Cl_fid)
        resp[nm] = np.log(hi[0]) - np.log(lo[0])
        snr[nm] = float(np.nanmax(np.abs(dv) / np.sqrt(hi[1] ** 2 + lo[1] ** 2)))
        present.append(nm)

    # Sobol targets on the same bands
    Y = np.asarray(dsn[spec["tgt"]], float)
    if Y.ndim == 3:
        Y = Y[:, ZI, :]
    Yb = np.array([band(Y[i]) for i in range(len(Y))])
    Lb = np.log(np.where(Yb > 0, Yb, np.nan))              # ln Cl: the additive variable
    D = Lb - np.nanmean(Lb, 0)
    SST = np.nansum(D ** 2)

    # Spearman peak |rho| per parameter + 200-shuffle retention null
    def peak_rho(X):
        rx = np.apply_along_axis(rankdata, 0, X)
        ry = np.apply_along_axis(rankdata, 0, Lb)
        rx = (rx - rx.mean(0)) / rx.std(0)
        ry = (ry - ry.mean(0)) / np.where(ry.std(0) > 0, ry.std(0), 1)
        return np.abs(rx.T @ ry / len(X)).max(1)
    rho = dict(zip(PN, peak_rho(XU)))
    null = np.array([peak_rho(XU[RNG.permutation(len(XU))]).max() for _ in range(200)])
    null95 = float(np.percentile(null, 95))

    # entry gate + clustering (paper rules)
    members = [nm for nm in present if snr[nm] >= SNR_GATE]
    curves = {}
    for nm in members:
        d_ = resp[nm]
        curves[nm] = d_ / d_[np.nanargmax(np.abs(d_))]
    fams = []
    if len(members) >= 2:
        M = np.corrcoef([curves[nm] for nm in members])
        Z = linkage(squareform(np.clip(1 - M, 0, None), checks=False), "average")
        lab = fcluster(Z, t=T_CLUST, criterion="distance")
        for k in sorted(set(lab)):
            fams.append([members[i] for i in range(len(members)) if lab[i] == k])
    elif members:
        fams = [[members[0]]]
    kept = [f for f in fams if any(rho[m] > null95 for m in f)]

    # basis: rho-weighted family means, extremum-normalized; drop
    # linearly-degenerate duplicates (the signed-clustering singlet pathology)
    basis = []
    for f in kept:
        w = np.array([rho[m] for m in f])
        b = np.average([curves[m] for m in f], axis=0, weights=w)
        b = b / b[np.nanargmax(np.abs(b))]
        if all(abs(np.corrcoef(b, b2)[0, 1]) < 0.98 for b2 in basis):
            basis.append(b)
    B = np.array(basis)
    K = len(B)

    # span R^2 (free amplitudes) + e2e CV with the shipped latents
    A = np.linalg.lstsq(B.T, D.T, rcond=None)[0].T           # (256, K)
    span = 1 - np.nansum((D - A @ B) ** 2) / SST
    perm = np.random.default_rng(1).permutation(len(D))
    folds = [(np.setdiff1d(perm, te), te) for te in np.array_split(perm, 5)]
    pred = np.empty_like(A)
    for tr, te in folds:
        Xtr = np.c_[LAT[tr] - LAT[tr].mean(0), np.ones(len(tr))]
        Xte = np.c_[LAT[te] - LAT[tr].mean(0), np.ones(len(te))]
        W, *_ = np.linalg.lstsq(Xtr, A[tr], rcond=None)
        pred[te] = Xte @ W
    cv = 1 - np.nansum((D - pred @ B) ** 2) / SST

    # the theta route (linear map from the 30 parameters), same folds
    TZ = (XU - XU.mean(0)) / XU.std(0)
    pred_t = np.empty_like(A)
    for tr, te in folds:
        Xtr = np.c_[TZ[tr] - TZ[tr].mean(0), np.ones(len(tr))]
        Xte = np.c_[TZ[te] - TZ[tr].mean(0), np.ones(len(te))]
        Wt, *_ = np.linalg.lstsq(Xtr, A[tr], rcond=None)
        pred_t[te] = Xte @ Wt
    cv_theta = 1 - np.nansum((D - pred_t @ B) ** 2) / SST

    # fiducial closure (out of design; latents measured, spectra from run_0049)
    lam_fid = ((SEARCH["X_FID"] - X_ALL.mean(0)) / X_ALL.std(0))[CHOSEN]
    W, *_ = np.linalg.lstsq(np.c_[LAT - LAT.mean(0), np.ones(len(LAT))], A, rcond=None)
    a_fid = np.r_[lam_fid - LAT.mean(0), 1.0] @ W
    fid_pred = np.nanmean(Lb, 0) + a_fid @ B
    # convention guard: the twobound per-run CROSS-spectrum caches carry a
    # pre-xpkfix normalization (~1.2e7 low, ell-dependent -- harmless for the
    # response RATIOS above, fatal for an absolute closure). Only report the
    # closure where the fiducial leg and the Sobol suite share a norm.
    off = float(np.nanmedian(np.log(fid_v) - np.nanmean(Lb, 0)))
    if abs(off) < 2.0:
        fid_dev = float(np.nanmedian(np.abs(np.exp(fid_pred - np.log(fid_v)) - 1)))
    else:
        fid_dev = np.nan
        print(f"    [{st}] fiducial closure SKIPPED: twobound leg offset "
              f"{off:+.1f} in ln Cl (pre-xpkfix cross norm)")

    RESULTS[st] = dict(K=K, span=span, cv=cv, cv_theta=cv_theta, fid_dev=fid_dev, fams=kept,
                       n_par=len(present), null95=null95, rho=rho, snr=snr,
                       curves=curves, members=members, basis=B, lab=spec["lab"])
    print(f"[{st}] params {len(present)}/30, gated {len(members)}, "
          f"families {len(fams)} -> kept {len(kept)} -> basis K={K} | "
          f"span R2 {span:.4f}  CV R2 {cv:.4f}  fid |dev| {100*fid_dev:.2f}%")
    for i, f in enumerate(kept):
        print(f"    G{i+1}: {f}")

# ── the appendix figure: tt + yy families, fig-13 style ─────────────────────
for st_fig, nrow in (("tt", 0), ("yy", 1)):
    pass
rows = ["tt", "yy"]
ncol = max(len(RESULTS[s]["fams"]) for s in rows)
fig, axs = plt.subplots(2, ncol, figsize=(TWO_COL[0], 4.2), sharex=True, sharey=True)
for ri, st in enumerate(rows):
    R = RESULTS[st]
    for ci in range(ncol):
        ax = axs[ri, ci]
        if ci >= len(R["fams"]):
            ax.axis("off")
            continue
        fam = R["fams"][ci]
        for nm in R["members"]:
            ax.plot(XG, R["curves"][nm], color="0.85", lw=0.6, zorder=1)
        for nm in fam:
            ok = R["rho"][nm] > R["null95"]
            ax.plot(XG, R["curves"][nm], color=CTXCOL[CTX[nm]],
                    lw=1.3 if ok else 0.9, ls="-" if ok else ":", zorder=3)
        ax.set_xscale("log")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.text(0.04, 0.92, f"G{ci+1}", transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")
        names = sorted(fam, key=lambda m: -R["rho"][m])[:6]
        ax.text(0.96, 0.05, "\n".join(names), transform=ax.transAxes, fontsize=4.0,
                ha="right", va="bottom",
                color="0.25")
        if ci == 0:
            ax.set_ylabel(rf"$\hat R$ [{R['lab']}]", fontsize=8)
        if ri == 1:
            ax.set_xlabel(r"$\ell$", fontsize=8)
        ax.tick_params(labelsize=6.5)
present_ctx = sorted({CTX[nm] for s in rows for f in RESULTS[s]["fams"] for nm in f})
fig.legend(handles=[plt.Line2D([], [], color=CTXCOL[c], lw=1.6, label=c)
                    for c in present_ctx],
           loc="lower center", ncol=min(len(present_ctx), 4), fontsize=6,
           frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.subplots_adjust(hspace=0.1, wspace=0.08)
_IMGS = "imgs_1000" if CAMPAIGN == "n1000" else "imgs"
_rtag = "_n1000" if CAMPAIGN == "n1000" else ""
save(fig, str(HERE.parent / _IMGS / "figA4_gas_families"))
plt.close(fig)
if CAMPAIGN == "n1000":
    # paper_style.save() drops the PNG preview at the UNTAGGED shared path
    # figs_preview/figA4_gas_families.png regardless of campaign; keep a
    # campaign-tagged copy so the two campaigns' previews cannot shadow each
    # other (the PDF/npz outputs are already separated).
    import shutil as _sh
    _prev = HERE.parent / "figs_preview/figA4_gas_families.png"
    _sh.copy2(_prev, _prev.with_name("figA4_gas_families_n1000.png"))

np.savez(HERE / f"gas_families_results{_rtag}.npz",
         **{f"{s}_{k}": RESULTS[s][k] for s in RESULTS
            for k in ("K", "span", "cv", "cv_theta", "fid_dev", "n_par")},
         xg=XG)
print(f"\nwrote {_IMGS}/figA4_gas_families + analysis/gas_families_results{_rtag}.npz")
print("\nTable block (K, span R2, CV R2 with the shipped WL-selected latents):")
for s in ("yy", "tt", "ky", "kt", "yt"):
    R = RESULTS[s]
    fc = f"{100*R['fid_dev']:.1f}%" if np.isfinite(R['fid_dev']) else "n/a (norm)"
    print(f"  {R['lab']:22s} K={R['K']}  span {R['span']:.3f}  CV {R['cv']:.3f}  "
          f"theta {R['cv_theta']:.3f}  (params {R['n_par']}/30, fid closure {fc})")
