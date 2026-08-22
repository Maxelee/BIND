"""fidswap re-render: paper Fig 5 (field validation) == notebook fig04_field_validation.

Cell body copied verbatim from _build_figures_nb.py (Part A audit + the 8-panel
figure + every print).  Fiducial inlets re-pointed, and ONLY those:

  RB = SCI/'runs/bind/run_0000'          -> FID_RUN  (twobound/run_0049)
  field_cache.load()                     -> referee_work/fidswap/field_cache/...
  nu05_shards/sci_bind.npz               -> referee_work/fidswap/nu05_shards/sci_bind_tb49.npz
  RB/paired_perreal_fid.npz  ['clk']     -> the field cache's own kk_bind
        (verified bit-identical on the OLD fiducial: max|clk/kk_bind - 1| = 0)
  RB/peak_counts_ngal10.npz              -> referee_work/fidswap/nu05_shards/peak_counts_ngal10_tb49.npz
Truth-side inlets (RT, sci_truth, truth peak_counts/ngal10) are untouched.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as _chi2dist, f as _fdist

from fsfig_common import (BAND_ALPHA, COLORS, ELL_LO, ELL_MAX_PLOT, ELL_TRUST,
                          FID_FC, FID_NG10, FID_NU05, FID_RUN, OLD_FC, OLD_NU05,
                          OLD_FID, RT, TRUTH_NU05, TWO_COL, ZI, ZS,
                          dmo_paired_cl, ell_trust_marker, panel_label, plt,
                          rebin, save_imgs, tee)

tee("fig05_field_hydrofull")

# ── loads ────────────────────────────────────────────────────────────────────
with np.load(FID_RUN / "kappa_maps.npz") as _f:
    NPIX_SIDE = int(np.asarray(_f["npix"]).ravel()[0])
    _N_BIND_KAPPA = int(_f["n_real"])
NPIX = NPIX_SIDE * NPIX_SIDE
_fc = np.load(FID_FC)
NR = int(_fc["kk_bind"].shape[0])
print(f"paired-realization availability: bind(new fiducial) kappa n_real={_N_BIND_KAPPA}, "
      f"field cache rows={NR} -> seed-paired N={NR} (+-1sigma/sqrt({NR}) paired bands)")

ell_f = _fc["ell"]
kk_b, kk_t = _fc["kk_bind"][:, ZI], _fc["kk_truth"][:, ZI]
ky_b, ky_t = _fc["ky_bind"][:, ZI], _fc["ky_truth"][:, ZI]
yy_b, yy_t = _fc["yy_bind"][:, -1], _fc["yy_truth"][:, -1]
print(f"fig-5 paired spectra: source=cache ({FID_FC.name}, fid_run={_fc['fid_run']})")

from nu_grid import NU, NU_EDGES, DNU  # noqa: E402

SB6 = np.load(FID_NU05)
ST6 = np.load(TRUTH_NU05)
# --- FULL-HYDRO TNG300 overlay (the construction ceiling) --------------------
from pathlib import Path as _P
_FSD = _P("/mnt/home/mlee1/ceph/referee_work/fidswap")
HF6 = np.load(_FSD / "nu05_shards/sci_hydrofull.npz")
_hfk = np.load(_FSD / "hydrofull_kk_perreal.npz")
HF_KK = _hfk["cl"]                       # (50, 724) per-realization, z_s=1
assert np.allclose(_hfk["ell"], ell_f), "hydro-full ell grid mismatch"
assert np.allclose(HF6["nu"], ST6["nu"]), "hydro-full nu grid mismatch"
print(f"full-hydro overlay: Cl {HF_KK.shape}, nu-stats {HF6['pdf_real'].shape}")
NU6 = ("pdf", "peak_counts", "minima_counts", "mf_v0", "mf_v1", "mf_v2")
for _nm, _sh in (("sci_bind_fidswap", SB6), ("sci_truth", ST6)):
    assert np.allclose(_sh["nu"], NU), f"{_nm} nu grid != nu_grid.NU"
    for k in NU6:
        assert f"{k}_real" in _sh.files, f"{_nm} missing {k}_real"
NR6 = int(SB6["pdf_real"].shape[0])
print(f"fig-5 nu-domain statistics: {FID_NU05.name} / sci_truth.npz ({NR6} realizations, "
      f"NU {NU[0]:.2f}..{NU[-1]:.2f}, {len(NU)} bins, d(nu)={DNU:.2f})")

mt = (ell_f >= ELL_LO) & (ell_f <= ELL_TRUST)


def paired(bR, tR):
    n = min(bR.shape[0], tR.shape[0])
    rat = bR / np.where(tR > 0, tR, np.nan)
    res = 100 * (np.nanmean(bR, 0) / np.nanmean(tR, 0) - 1)
    band = 100 * np.nanstd(rat, 0) / np.sqrt(n)
    chi2 = np.nanmean((res[mt] / np.where(band[mt] > 0, band[mt], np.nan)) ** 2)
    return np.nanmean(bR, 0), np.nanmean(tR, 0), res, band, chi2


def chi2_full(bR, tR):
    msk4 = (ell_f >= ELL_LO) & (ell_f <= ELL_TRUST)
    rat = rebin((bR / np.where(tR > 0, tR, np.nan))[:, msk4], 16)
    okb = np.isfinite(rat).all(0)
    rat = rat[:, okb]
    n, p = rat.shape
    assert p <= n - 3
    Cm = np.cov(rat, rowvar=False) / n
    corr = Cm / np.sqrt(np.outer(np.diag(Cm), np.diag(Cm)))
    rho_bar = np.nanmean(np.abs(corr[np.triu_indices(p, k=1)]))
    hart = (n - p - 2) / (n - 1)
    dv = rat.mean(0) - 1
    chi2_raw = float(dv @ np.linalg.solve(Cm, dv))
    return hart * chi2_raw / p, rho_bar, p, n, chi2_raw


# ── PART A: noise-accounting audit ───────────────────────────────────────────
def null_test(R, nhalf=25):
    m1, m2 = np.nanmean(R[:nhalf], 0), np.nanmean(R[nhalf:2 * nhalf], 0)
    s1 = np.nanstd(R[:nhalf], 0) / np.sqrt(nhalf)
    s2 = np.nanstd(R[nhalf:2 * nhalf], 0) / np.sqrt(nhalf)
    sig = np.sqrt(s1 ** 2 + s2 ** 2)
    return np.nanmean(((m1 - m2)[mt] / np.where(sig[mt] > 0, sig[mt], np.nan)) ** 2)


def rho_realization(R, N=None):
    if N is None:
        N = R.shape[0]
    X = R[:, mt]
    sd = np.nanstd(X, 0, keepdims=True)
    z = (X - np.nanmean(X, 0, keepdims=True)) / np.where(sd > 0, sd, np.nan)
    ok = np.isfinite(z).all(1)
    C = np.corrcoef(z[ok])
    rho_bar = np.nanmean(C[np.triu_indices(C.shape[0], k=1)])
    return rho_bar, N / (1 + (N - 1) * rho_bar)


print("=== Part A: noise-accounting audit ===")
for lab, Rt, Rb in [("kk", kk_t, kk_b), ("ky", ky_t, ky_b), ("yy", yy_t, yy_b)]:
    x2t, x2b = null_test(Rt), null_test(Rb)
    rho_t, _ = rho_realization(Rt)
    rho_b, _ = rho_realization(Rb)
    print(f"{lab}: null chi2/dof  truth-halves={x2t:.2f}  bind-halves={x2b:.2f}   |   "
          f"realization-realization rho_bar={max(rho_t, rho_b):+.3f} -> N_eff~{NR}")

# ── PART B: the figure ───────────────────────────────────────────────────────
b_cl0 = np.load(FID_RUN / "Cl_kappa.npz")
t_cl0 = np.load(RT / "Cl_kappa.npz")

f_ell = ell_f * (ell_f + 1) / (2 * np.pi)
KKb, KKt, res_kk, band_kk, x2_kk = paired(kk_b, kk_t)
KYb, KYt, res_ky, band_ky, x2_ky = paired(ky_b, ky_t)
YYb, YYt, res_yy, band_yy, x2_yy = paired(yy_b, yy_t)

KK5b = np.array([b_cl0["cl"][zi, zi] for zi in range(5)], dtype=float)
KK5t = np.array([t_cl0["cl"][zi, zi] for zi in range(5)], dtype=float)
print(f"cached vs live mean Cl_kk at z_s={ZS[ZI]:.1f}: median |ratio-1| = "
      f"{100*np.nanmedian(np.abs(KK5b[ZI]/np.where(KKb > 0, KKb, np.nan) - 1)):.4f}% (BIND), "
      f"{100*np.nanmedian(np.abs(KK5t[ZI]/np.where(KKt > 0, KKt, np.nan) - 1)):.4f}% (truth)")
KK5b[ZI], KK5t[ZI] = KKb, KKt

# S(ell): the BIND per-real clk is the field cache's own kk_bind (identical to
# paired_perreal_fid.npz['clk'] on the old fiducial, verified max|ratio-1| = 0).
CLK5 = _fc["kk_bind"]
dmo5 = dmo_paired_cl(NR, fid_clk=CLK5[:, ZI, :])
dmo_mean = dmo5[ZI]
S_b = CLK5[:, ZI, :] / np.where(dmo_mean > 0, dmo_mean, np.nan)
S_t = kk_t / np.where(dmo_mean > 0, dmo_mean, np.nan)
Sb_m, St_m, res_S, band_S, x2_S = paired(S_b, S_t)
S5b = CLK5.mean(0) / np.where(dmo5 > 0, dmo5, np.nan)
S5t = KK5t / np.where(dmo5 > 0, dmo5, np.nan)

# full-hydro: mean Cl, S(ell), and its residual against the SAME hydro-pasted truth
HF_KKm = np.nanmean(HF_KK, 0)
HF_S = np.nanmean(HF_KK / np.where(dmo_mean > 0, dmo_mean, np.nan), 0)
res_kk_hf = 100 * (HF_KKm / np.where(KKt > 0, KKt, np.nan) - 1)
res_S_hf = 100 * (HF_S / np.where(St_m > 0, St_m, np.nan) - 1)
res_nu_hf = {}
for _k in NU6:
    _h = np.nanmean(HF6[f"{_k}_real"][:, ZI], 0)
    _t = np.nanmean(ST6[f"{_k}_real"][:, ZI], 0)
    res_nu_hf[_k] = 100 * (_h / np.where(np.abs(_t) > 0, _t, np.nan) - 1)
print("full-hydro vs hydro-pasted, median |resid|: "
      + f"kk {np.nanmedian(np.abs(res_kk_hf[mt])):.2f}%  S {np.nanmedian(np.abs(res_S_hf[mt])):.2f}%  "
      + ", ".join(f"{k} {np.nanmedian(np.abs(v)):.2f}%" for k, v in res_nu_hf.items()))

fullcov = {lab: chi2_full(Rb, Rt) for lab, Rb, Rt in
           [("kk", kk_b, kk_t), ("ky", ky_b, ky_t), ("yy", yy_b, yy_t)]}

A2SR = (np.pi / 180 / 60) ** 2


def cl_relerr(cl, ngal, sige, fsky, dlnl=0.15, ell=None):
    ell = ell_f if ell is None else ell
    Nl = sige ** 2 * A2SR / ngal
    dl = np.maximum(ell * dlnl, 1.0)
    return np.sqrt(2.0 / ((2 * ell + 1) * dl * fsky)) * (1 + Nl / cl)


def cl_relerr_cv_only(fsky, dlnl=0.15, ell=None):
    ell = ell_f if ell is None else ell
    dl = np.maximum(ell * dlnl, 1.0)
    return np.sqrt(2.0 / ((2 * ell + 1) * dl * fsky))


logcl_b_rb = rebin(np.log(np.where(kk_b > 0, kk_b, np.nan)), 8)
ell_f_rb = rebin(ell_f[None, :], 8)[0]
sigma_lncl_25_kk = np.nanstd(logcl_b_rb, axis=0)
AREA_25_SR, AREA_LSST_SR = 25.0 * (np.pi / 180.0) ** 2, 0.44 * 4 * np.pi
sigma_meas_lsst_kk = np.interp(ell_f, ell_f_rb,
                               sigma_lncl_25_kk * np.sqrt(AREA_25_SR / AREA_LSST_SR))
shot_excess_kk = cl_relerr(KKt, 27.0, 0.26, 0.44) - cl_relerr_cv_only(0.44)
lsst_kk = 100 * np.sqrt(sigma_meas_lsst_kk ** 2 + shot_excess_kk ** 2)
lsst_S = lsst_kk

NU_LIM = (-3, 8)
AREA_SCALE_NU = np.sqrt(25.0 / 18000.0)

res, point_err, lsst_band = {}, {}, {}
for _k in NU6:
    _tm = ST6[_k][ZI]
    res[_k] = 100 * (SB6[_k][ZI] / np.where(_tm != 0, _tm, np.nan) - 1)

_pk_old_t = np.load(RT / "peak_counts.npz")
_ng10_b = np.load(FID_NG10)
_ng10_t = np.load(RT / "peak_counts_ngal10.npz")
assert np.allclose(_ng10_t["nu"], _pk_old_t["nu"]) and np.allclose(_ng10_b["nu"], _pk_old_t["nu"])
_nu_old = _pk_old_t["nu"]


def shape_noise_excess_pct(dst):
    rel_ng = _ng10_t[f"{dst}_err"][ZI] / np.where(_ng10_t[dst][ZI] > 0, _ng10_t[dst][ZI], np.nan)
    rel_plain = _pk_old_t[f"{dst}_err"][ZI] / np.where(_pk_old_t[dst][ZI] > 0,
                                                       _pk_old_t[dst][ZI], np.nan)
    excess = np.sqrt(np.clip(rel_ng ** 2 - rel_plain ** 2, 0, None))
    return 100 * np.interp(NU, _nu_old, np.nan_to_num(excess, nan=0.0))


for _k in NU6:
    _tm = ST6[_k][ZI]
    _sv = (100 * np.std(ST6[f"{_k}_real"][:, ZI, :], axis=0) * AREA_SCALE_NU
           / np.where(_tm != 0, np.abs(_tm), np.nan))
    lsst_band[_k] = (np.sqrt(_sv ** 2 + shape_noise_excess_pct(_k) ** 2)
                     if _k in ("peak_counts", "minima_counts") else _sv)

point_err["peak_counts"] = 100 / np.sqrt(np.where(ST6["peak_counts"][ZI] > 0,
                                                  ST6["peak_counts"][ZI], np.nan))
point_err["minima_counts"] = 100 / np.sqrt(np.where(ST6["minima_counts"][ZI] > 0,
                                                    ST6["minima_counts"][ZI], np.nan))
N_pdf_bin = NPIX * DNU * ST6["pdf"][ZI]
point_err["pdf"] = 100 / np.sqrt(np.where(N_pdf_bin > 0, N_pdf_bin, np.nan))
for _k in ("mf_v0", "mf_v1", "mf_v2"):
    _diff = SB6[f"{_k}_real"][:, ZI, :] - ST6[f"{_k}_real"][:, ZI, :]
    _tm = ST6[_k][ZI]
    point_err[_k] = (100 * np.std(_diff, axis=0) / np.sqrt(NR6)
                     / np.where(_tm != 0, np.abs(_tm), np.nan))


def chi2_nu(k):
    e = point_err[k]
    m = np.isfinite(res[k]) & np.isfinite(e) & (e > 0)
    return float(np.nanmean((res[k][m] / e[m]) ** 2)), int(m.sum())


x2_nu = {k: chi2_nu(k) for k in NU6}
print("chi2/dof (nu05 grid, 22 bins): "
      + ", ".join(f"{k} {v[0]:.2f} [{v[1]}/{len(NU)}]" for k, v in x2_nu.items()))

cB, cH = COLORS["bind"], COLORS["truth"]
C_HF = COLORS["secondary"]   # full-hydro TNG300 (outside the plasma z_s ramp)
ZCOLS = plt.cm.plasma(np.linspace(0.02, 0.82, 5))
_lt2 = max(1e-12, 0.02 * float(np.max(np.abs(ST6["mf_v2"][ZI]))))

panels = [
    dict(x=ell_f, yb_z=f_ell * KK5b, yt=f_ell * KKt, res=res_kk, band=band_kk, lsst=lsst_kk,
         xl=r"$\ell$", yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\kappa}/2\pi$",
         xs="log", ys="log", xlim=(ELL_LO, ELL_MAX_PLOT), tag="(a)",
         ctx="LSST-Y10", alias=True, alias_note=True, yhf=f_ell * HF_KKm, reshf=res_kk_hf),
    dict(x=ell_f, yb_z=S5b, yt=St_m, res=res_S, band=band_S, lsst=lsst_S,
         xl=r"$\ell$", yl=r"$S(\ell)=C_\ell^{\rm bind}/C_\ell^{\rm dmo}$",
         xs="log", ys="linear", xlim=(ELL_LO, ELL_MAX_PLOT), tag="(b)", ctx="LSST-Y10",
         yhf=HF_S, reshf=res_S_hf),
    dict(x=NU, yb_z=SB6["pdf"], yt=ST6["pdf"][ZI], res=res["pdf"],
         band=point_err["pdf"], lsst=lsst_band["pdf"], discrete=True,
         xl=r"$\nu$", yl=r"PDF$(\nu)$", xs="linear", ys="log", xlim=NU_LIM, tag="(c)", yhf=np.nanmean(HF6["pdf_real"][:, ZI], 0), reshf=res_nu_hf["pdf"], ylo=1e-5),
    dict(x=NU, yb_z=SB6["peak_counts"], yt=ST6["peak_counts"][ZI], res=res["peak_counts"],
         band=point_err["peak_counts"], lsst=lsst_band["peak_counts"], discrete=True,
         xl=r"$\nu$", yl=r"$N_{\rm peak}$", xs="linear", ys="log", xlim=NU_LIM, tag="(d)", yhf=np.nanmean(HF6["peak_counts_real"][:, ZI], 0), reshf=res_nu_hf["peak_counts"]),
    dict(x=NU, yb_z=SB6["minima_counts"], yt=ST6["minima_counts"][ZI],
         res=res["minima_counts"], band=point_err["minima_counts"],
         lsst=lsst_band["minima_counts"], discrete=True,
         xl=r"$\nu$", yl=r"$N_{\rm min}$", xs="linear", ys="log", xlim=NU_LIM, tag="(e)", yhf=np.nanmean(HF6["minima_counts_real"][:, ZI], 0), reshf=res_nu_hf["minima_counts"]),
    dict(x=NU, yb_z=SB6["mf_v0"], yt=ST6["mf_v0"][ZI], res=res["mf_v0"],
         band=point_err["mf_v0"], lsst=lsst_band["mf_v0"], discrete=True,
         xl=r"$\nu$", yl=r"$V_0(\nu)$", xs="linear", ys="log", xlim=NU_LIM, tag="(f)", yhf=np.nanmean(HF6["mf_v0_real"][:, ZI], 0), reshf=res_nu_hf["mf_v0"]),
    dict(x=NU, yb_z=SB6["mf_v1"], yt=ST6["mf_v1"][ZI], res=res["mf_v1"],
         band=point_err["mf_v1"], lsst=lsst_band["mf_v1"], discrete=True,
         xl=r"$\nu$", yl=r"$V_1(\nu)$", xs="linear", ys="log", xlim=NU_LIM, tag="(g)", yhf=np.nanmean(HF6["mf_v1_real"][:, ZI], 0), reshf=res_nu_hf["mf_v1"]),
    dict(x=NU, yb_z=SB6["mf_v2"], yt=ST6["mf_v2"][ZI], res=res["mf_v2"],
         band=point_err["mf_v2"], lsst=lsst_band["mf_v2"], discrete=True,
         xl=r"$\nu$", yl=r"$V_2(\nu)$", xs="linear", ys="symlog", linthresh=_lt2,
         xlim=NU_LIM, tag="(h)", yhf=np.nanmean(HF6["mf_v2_real"][:, ZI], 0), reshf=res_nu_hf["mf_v2"]),
]

fig = plt.figure(figsize=(TWO_COL[0], 5.9))
gs = fig.add_gridspec(5, 4, height_ratios=[2.2, 1, 0.55, 2.2, 1], hspace=0.18, wspace=0.45)
AX, AXR = [], []
for j, p in enumerate(panels):
    r0 = (j // 4) * 3
    a = fig.add_subplot(gs[r0, j % 4])
    ar = fig.add_subplot(gs[r0 + 1, j % 4], sharex=a)
    x1 = p["x"]
    for zi in range(5):
        a.plot(p["x"], p["yb_z"][zi], color=ZCOLS[zi], lw=1.0)
    a.plot(x1, p["yt"], color=cH, ls="--", lw=0.9)
    if p.get("yhf") is not None:
        a.plot(x1, p["yhf"], color=C_HF, ls=":", lw=1.1, zorder=5)
    a.set_xscale(p["xs"])
    if p["ys"] == "symlog":
        a.set_yscale("symlog", linthresh=p["linthresh"])
    else:
        a.set_yscale(p["ys"])
    a.set_ylabel(p["yl"], fontsize=7, labelpad=p.get("ylpad", 4.0))
    if p["xlim"]:
        a.set_xlim(*p["xlim"])
    if p.get("ylo"):
        a.set_ylim(bottom=p["ylo"])
    if p.get("alias"):
        ell_trust_marker(a, label=p.get("alias_note", False))
    plt.setp(a.get_xticklabels(), visible=False)
    a.tick_params(labelsize=6)

    ar.axhline(0, color=COLORS["dmo"], lw=0.6)
    if p.get("discrete"):
        if p["lsst"] is not None:
            ar.fill_between(p["x"], -p["lsst"], p["lsst"], color=cB, alpha=0.22, lw=0)
            ar.text(0.97, 0.05, "LSST-like", transform=ar.transAxes,
                    fontsize=4.2, color=cB, va="bottom", ha="right")
        if p.get("reshf") is not None:
            ar.plot(p["x"], p["reshf"], color=C_HF, ls=":", lw=1.1, zorder=2)
        ar.errorbar(p["x"], p["res"], yerr=p["band"], fmt="o", ms=2.0, mew=0,
                    color=cB, lw=0.7, elinewidth=0.7, capsize=1.3)
    else:
        if p["band"] is not None:
            ar.fill_between(x1, -p["band"], p["band"], color=COLORS["dmo"], alpha=0.35, lw=0)
        if p["lsst"] is not None:
            ar.fill_between(p["x"], -p["lsst"], p["lsst"], color=cB, alpha=0.22, lw=0)
            ar.text(0.97, 0.05, p.get("ctx", "LSST-Y10"), transform=ar.transAxes,
                    fontsize=4.2, color=cB, va="bottom", ha="right")
        if p.get("reshf") is not None:
            ar.plot(x1, p["reshf"], color=C_HF, ls=":", lw=1.1, zorder=2)
        ar.plot(x1, p["res"], color=cB, lw=0.9)
    rl = p.get("rlim", 25)
    ar.set_ylim(-rl, rl)
    ar.set_xscale(p["xs"])
    ar.set_xlabel(p["xl"], fontsize=7)
    ar.set_ylabel("resid. [%]" if j % 4 == 0 else "", fontsize=7)
    ar.tick_params(labelsize=6)
    panel_label(a, p["tag"], loc="upper right")
    AX.append(a)
    AXR.append(ar)

AXR[0].text(0.03, 0.06, r"residual: $z_s=1$", transform=AXR[0].transAxes,
            fontsize=4.2, color="0.3", va="bottom")
AX[3].text(0.03, 0.93, "Poisson err.", transform=AX[3].transAxes,
           fontsize=3.8, color=cB, ha="left", va="top")
AX[0].legend(handles=[plt.Line2D([], [], color=ZCOLS[zi], lw=1.0,
                                 label=rf"$z_s={ZS[zi]:.2f}$") for zi in range(5)],
             loc="upper left", fontsize=4.4, ncol=2, handlelength=1.1,
             columnspacing=0.7, labelspacing=0.25, borderpad=0.2, handletextpad=0.4)
AX[1].legend(handles=[plt.Line2D([], [], color="0.35", lw=1.0, label="BIND"),
                      plt.Line2D([], [], color=cH, lw=0.9, ls="--",
                                 label=r"hydro-pasted"),
                      plt.Line2D([], [], color=C_HF, lw=1.1, ls=":",
                                 label=r"full-hydro TNG300")],
             loc="lower left", fontsize=4.6, handlelength=1.3, labelspacing=0.25,
             borderpad=0.25, handletextpad=0.4)
save_imgs(fig, "fig05_field_validation_hydrofull")
plt.close(fig)

m = (ell_f >= 300) & (ell_f <= 5000)
print("median |resid| in ell=300-5000:  "
      f"Cl_kk {np.nanmedian(np.abs(res_kk[m])):.1f}%  "
      f"Cl_ky {np.nanmedian(np.abs(res_ky[m])):.1f}%  "
      f"Cl_yy {np.nanmedian(np.abs(res_yy[m])):.1f}%")
print(f"chi2/dof (seed-paired diagonal bands): kk {x2_kk:.0f}  S(ell) {x2_S:.0f}  "
      f"PDF {x2_nu['pdf'][0]:.2f}  N_pk {x2_nu['peak_counts'][0]:.2f}  "
      f"N_min {x2_nu['minima_counts'][0]:.2f}  V0 {x2_nu['mf_v0'][0]:.2f}  "
      f"V1 {x2_nu['mf_v1'][0]:.2f}  V2 {x2_nu['mf_v2'][0]:.2f}  "
      f"ky {x2_ky:.0f}  yy {x2_yy:.0f}")
print("nu-domain residual amplitude (percent of truth, full NU grid): "
      + ", ".join(f"{k} median {np.nanmedian(np.abs(res[k])):.2f}% max "
                  f"{np.nanmax(np.abs(res[k])):.2f}%" for k in NU6))

for lab in ("kk", "ky", "yy"):
    x2f, rho_bar, pbin, n_fc, chi2_raw = fullcov[lab]
    p_gauss = float(_chi2dist.sf(x2f * pbin, df=pbin))
    F_stat = ((n_fc - pbin) / (pbin * (n_fc - 1))) * chi2_raw
    p_sh = float(_fdist.sf(F_stat, pbin, n_fc - pbin))
    print(f"chi2/dof (FULL {pbin}-bin covariance, Hartlap) {lab}: {x2f:.0f}  "
          f"(mean |ell-bin corr|={rho_bar:.2f})  |  p Gaussian/Hartlap = {p_gauss:.2e}, "
          f"Sellentin-Heavens = {p_sh:.2e}")


def chi2_unpaired(bR, tR):
    msk4 = (ell_f >= ELL_LO) & (ell_f <= ELL_TRUST)
    Bc, Tc = rebin(bR[:, msk4], 16), rebin(tR[:, msk4], 16)
    okb = np.isfinite(Bc).all(0) & np.isfinite(Tc).all(0) & (np.abs(Tc.mean(0)) > 0)
    Bc, Tc = Bc[:, okb], Tc[:, okb]
    n, p = Bc.shape
    Tm = Tc.mean(0)
    Cf = (np.cov(Bc, rowvar=False) + np.cov(Tc, rowvar=False)) / n / np.outer(Tm, Tm)
    dv = Bc.mean(0) / Tm - 1
    hart = (n - p - 2) / (n - 1)
    return hart * float(dv @ np.linalg.solve(Cf, dv)) / p


print("chi2/dof T2 UNPAIRED: "
      f"kk {chi2_unpaired(kk_b, kk_t):.1f}  ky {chi2_unpaired(ky_b, ky_t):.1f}  "
      f"yy {chi2_unpaired(yy_b, yy_t):.1f}")
_okp10 = _ng10_t["peak_counts"][ZI] > 1
print("shape-noise (ngal=10) peaks (OLD 68-bin cache, diagnostic): median |resid| = "
      f"{100*np.nanmedian(np.abs(_ng10_b['peak_counts'][ZI][_okp10]/_ng10_t['peak_counts'][ZI][_okp10]-1)):.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# BEFORE/AFTER table -- the same estimators run on the RETIRED fiducial
# ═══════════════════════════════════════════════════════════════════════════
print("\n### OLD vs NEW fiducial, every number stamped on paper Fig 5")
_ofc = np.load(OLD_FC)
o_kk, o_ky = _ofc["kk_bind"][:, ZI], _ofc["ky_bind"][:, ZI]
o_yy = _ofc["yy_bind"][:, -1]
o_S = _ofc["kk_bind"][:, ZI, :] / np.where(dmo_mean > 0, dmo_mean, np.nan)
_, _, o_res_kk, o_band_kk, o_x2kk = paired(o_kk, kk_t)
_, _, o_res_ky, o_band_ky, o_x2ky = paired(o_ky, ky_t)
_, _, o_res_yy, o_band_yy, o_x2yy = paired(o_yy, yy_t)
_, _, o_res_S, o_band_S, o_x2S = paired(o_S, S_t)
OSB6 = np.load(OLD_NU05)
o_res = {k: 100 * (OSB6[k][ZI] / np.where(ST6[k][ZI] != 0, ST6[k][ZI], np.nan) - 1)
         for k in NU6}


def o_chi2_nu(k):
    e = point_err[k]
    if k in ("mf_v0", "mf_v1", "mf_v2"):
        dsub = OSB6[f"{k}_real"][:, ZI, :] - ST6[f"{k}_real"][:, ZI, :]
        e = (100 * np.std(dsub, axis=0) / np.sqrt(NR6)
             / np.where(ST6[k][ZI] != 0, np.abs(ST6[k][ZI]), np.nan))
    mm = np.isfinite(o_res[k]) & np.isfinite(e) & (e > 0)
    return float(np.nanmean((o_res[k][mm] / e[mm]) ** 2))


print(f"{'stat':14s} {'OLD':>12s} {'NEW':>12s}")
print(f"{'|resid| kk':14s} {np.nanmedian(np.abs(o_res_kk[m])):11.2f}% "
      f"{np.nanmedian(np.abs(res_kk[m])):11.2f}%")
print(f"{'|resid| ky':14s} {np.nanmedian(np.abs(o_res_ky[m])):11.2f}% "
      f"{np.nanmedian(np.abs(res_ky[m])):11.2f}%")
print(f"{'|resid| yy':14s} {np.nanmedian(np.abs(o_res_yy[m])):11.2f}% "
      f"{np.nanmedian(np.abs(res_yy[m])):11.2f}%")
for lab, ov, nv in (("chi2 kk", o_x2kk, x2_kk), ("chi2 S(ell)", o_x2S, x2_S),
                    ("chi2 ky", o_x2ky, x2_ky), ("chi2 yy", o_x2yy, x2_yy)):
    print(f"{lab:14s} {ov:12.1f} {nv:12.1f}")
for k in NU6:
    print(f"chi2 {k:9s} {o_chi2_nu(k):12.2f} {x2_nu[k][0]:12.2f}")
for k in NU6:
    print(f"med|res| {k:9s} {np.nanmedian(np.abs(o_res[k])):11.2f}% "
          f"{np.nanmedian(np.abs(res[k])):11.2f}%")
print(f"{'S(ell) trough':14s} {np.nanmin(np.nanmean(o_S, 0)):12.4f} "
      f"{np.nanmin(np.nanmean(S_b, 0)):12.4f}   (truth {np.nanmin(np.nanmean(S_t, 0)):.4f})")
np.savez_compressed("/mnt/home/mlee1/ceph/referee_work/fidswap/fig05_numbers.npz",
                    ell=ell_f, res_kk=res_kk, band_kk=band_kk, res_S=res_S, band_S=band_S,
                    res_ky=res_ky, res_yy=res_yy, o_res_kk=o_res_kk, o_res_S=o_res_S,
                    o_res_ky=o_res_ky, o_res_yy=o_res_yy, S_new=np.nanmean(S_b, 0),
                    S_old=np.nanmean(o_S, 0), S_truth=np.nanmean(S_t, 0), nu=NU,
                    **{f"res_{k}": res[k] for k in NU6},
                    **{f"o_res_{k}": o_res[k] for k in NU6})
print("\nDONE fig05")
