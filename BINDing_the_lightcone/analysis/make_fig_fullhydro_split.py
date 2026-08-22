"""The full-hydro anchor, split into two appendix figures.

Replaces the single six-panel fig06b_full_hydro ladder:

  figA1_fullhydro_wl.{png,pdf}   (a) S(ell) three rungs + residual vs full
                                 hydro, (b) N_pk(nu) + residual, (c) captured
                                 fraction of the DMO->full-hydro response.
  figA2_fullhydro_gas.{png,pdf}  (a) C_l^yy, (b) C_l^tautau, three rungs each
                                 with residual strips, (c) mean map value /
                                 full hydro vs z_s.

Changes from the retired composite figure (2026-08-14 author direction):
  * the "pasted + diffuse gas" rung is dropped everywhere -- the release
    carries no diffuse-gas variant;
  * the gas residual bands are the +-1sigma REALIZATION scatter of the
    hydro-pasted truth (its own covariance diagonal), not +-1sigma/sqrt(50);
  * the BIND rung is the corrected-cosmology fiducial, twobound/run_0049.

All inputs are the fidswap R1 caches (ceph/referee_work/fidswap/r1/), the
n1000 truth/dmo per-realization traces for the S(ell) band, and
r1_numbers_z1_tb49.json for the capture fractions.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python make_fig_fullhydro_split.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
# BIND_FH_VARIANT=oldfid renders the RETIRED (CAMELS-conditioned) fiducial's
# bind rung from the original R1 caches, for posterity comparison; output gets
# an _oldfid suffix into figs_preview/ and the gas figure is skipped.
import os
VARIANT = os.environ.get("BIND_FH_VARIANT", "")
R1 = CEPH / ("referee_work/r1" if VARIANT == "oldfid" else "referee_work/fidswap/r1")
HERE = Path(__file__).resolve().parent
IMGS = HERE.parent / ("figs_preview" if VARIANT == "oldfid" else "imgs")
SUF = "_oldfid" if VARIANT == "oldfid" else ""
ZI = 1                                        # z_s = 1.0
ELL_LO, ELL_TRUST = 300.0, 3.0e4
ZS = np.array([0.5, 1.0, 1.5, 2.0, 2.44])
A2SR = (np.pi / 180 / 60) ** 2
AREA_25_SR = 25.0 * (np.pi / 180) ** 2
AREA_LSST_SR = 0.44 * 4 * np.pi

K = {r: np.load(R1 / f"kappa_{r}.npz") for r in ("bind", "pasted", "hydro_full", "dmo")}
Y = {r: np.load(R1 / f"y_{r}.npz") for r in ("bind", "pasted", "hydro_full")}
T = {r: np.load(R1 / f"tau_{r}.npz") for r in ("bind", "pasted", "hydro_full")}
COV = np.load(R1 / "cov_pasted550.npz")
CAPF = (Path("/mnt/home/mlee1/BIND/papers/01_pipeline/referee/work/r1_numbers_z1.json")
        if VARIANT == "oldfid" else CEPH / "referee_work/fidswap/r1_numbers_z1_tb49.json")
CAP = json.load(open(CAPF))["response_capture"]
ELL = np.asarray(K["bind"]["ell"], float)
NU = np.asarray(K["bind"]["nu"], float)
trust = (ELL >= ELL_LO) & (ELL <= ELL_TRUST)

CB, CT, CF = COLORS["bind"], COLORS["truth"], COLORS["highlight"]
STYLE = {"bind": dict(color=CB, lw=1.0, ls="-"),
         "pasted": dict(color=CT, lw=0.9, ls="--"),
         "hydro_full": dict(color=CF, lw=1.1, ls="-")}
LBL = {"bind": r"\textsc{BIND}" if False else "BIND", "pasted": "hydro-pasted",
       "hydro_full": "full hydro TNG300"}
KRUNGS = ("hydro_full", "bind", "pasted")     # reference first, rungs on top


def s_of(r):
    """seed-paired mean S(ell) of rung r at z_s=1."""
    return np.nanmean(K[r]["cl_kk"][:, ZI] / K["dmo"]["cl_kk"][:, ZI], 0)


# ── the truth-covariance band on S(ell): n1000 truth/dmo paired ratios ──────
n1k_t = np.load(CEPH / "bind_n1000/analysis/stream_stats_truth_run_0000.npz")["cl"][:, ZI, :]
n1k_d = np.load(CEPH / "bind_n1000/analysis/stream_stats_dmo_run_0000.npz")["cl"][:, ZI, :]
n = min(len(n1k_t), len(n1k_d))
S_real = n1k_t[:n] / n1k_d[:n]
sig_S = np.nanstd(S_real, 0)                  # per-realization scatter, 25 deg^2

# LSST-Y10 band for the WL residual strips (measured leg area-scaled + shot)
lcov = np.log(np.where(COV["cl_kk"] > 0, COV["cl_kk"], np.nan))
sig_meas = np.nanstd(lcov, 0) * np.sqrt(AREA_25_SR / AREA_LSST_SR)
Nl = 0.26 ** 2 * A2SR / 27.0
clf = np.nanmean(K["hydro_full"]["cl_kk"][:, ZI], 0)
dl = np.maximum(ELL * 0.15, 1.0)
shot = np.sqrt(2.0 / ((2 * ELL + 1) * dl * 0.44)) * (Nl / clf)
lsst_kk = 100 * np.sqrt(sig_meas ** 2 + shot ** 2)
sig_pk = np.nanstd(COV["peak_counts"], 0) * np.sqrt(AREA_25_SR / AREA_LSST_SR)

# ═════════════════════════════════════════════════════════════════════════
# Figure A1 — the WL ladder
# ═════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(TWO_COL[0], 3.4))
gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.14, wspace=0.42)
axa = fig.add_subplot(gs[0, 0])
ra = fig.add_subplot(gs[1, 0], sharex=axa)
S = {r: s_of(r) for r in KRUNGS}
axa.fill_between(ELL[trust], (S["pasted"] - sig_S)[trust], (S["pasted"] + sig_S)[trust],
                 color=CT, alpha=0.15, lw=0,
                 label=r"truth $\pm1\sigma$ (per real.)")
for r in KRUNGS:
    axa.plot(ELL[trust], S[r][trust], label=LBL[r], **STYLE[r])
axa.axhline(1, color="0.75", lw=0.6)
axa.set_xscale("log")
axa.set_ylabel(r"$S(\ell) = C_\ell^{\kappa\kappa}/C_\ell^{\kappa\kappa,{\rm DMO}}$",
               fontsize=8)
axa.legend(fontsize=6, frameon=False, loc="lower left")
axa.tick_params(labelbottom=False, labelsize=7)
panel_label(axa, "(a)")
for r in ("bind", "pasted"):
    ra.plot(ELL[trust], 100 * (S[r] / S["hydro_full"] - 1)[trust], **STYLE[r])
ra.fill_between(ELL[trust], -lsst_kk[trust], lsst_kk[trust], color=CB, alpha=0.12,
                lw=0, label="LSST-Y10")
ra.axhline(0, color="0.75", lw=0.6)
ra.set_xscale("log")
ra.set_xlabel(r"$\ell$", fontsize=8)
ra.set_ylabel(r"$\Delta$ vs full hydro [\%]" if False else
              "$\\Delta$ vs full hydro [%]", fontsize=7)
ra.legend(fontsize=6, frameon=False, loc="upper left")
ra.tick_params(labelsize=7)

axb = fig.add_subplot(gs[0, 1])
rb = fig.add_subplot(gs[1, 1], sharex=axb)
for r in KRUNGS:
    axb.plot(NU, np.nanmean(K[r][f"peak_counts_z{ZI}"], 0), **STYLE[r])
axb.set_yscale("log")
axb.set_ylabel(r"$N_{\rm pk}(\nu)$", fontsize=8)
axb.tick_params(labelbottom=False, labelsize=7)
panel_label(axb, "(b)")
pk_full = np.nanmean(K["hydro_full"][f"peak_counts_z{ZI}"], 0)
ok = pk_full > 0.5
for r in ("bind", "pasted"):
    d = 100 * (np.nanmean(K[r][f"peak_counts_z{ZI}"], 0) / pk_full - 1)
    rb.plot(NU[ok], d[ok], **STYLE[r])
rb.fill_between(NU[ok], -100 * (sig_pk / pk_full)[ok], 100 * (sig_pk / pk_full)[ok],
                color=CB, alpha=0.12, lw=0)
rb.axhline(0, color="0.75", lw=0.6)
rb.set_ylim(-14, 14)
rb.set_xlabel(r"$\nu = \kappa/\sigma_0$", fontsize=8)
rb.set_ylabel("$\\Delta$ [%]", fontsize=7)
rb.tick_params(labelsize=7)

axc = fig.add_subplot(gs[:, 2])
ROWS = [("S(ell)", r"$S(\ell)$"), ("pdf", r"$p(\nu)$"), ("peak_counts", r"$N_{\rm pk}$"),
        ("minima_counts", r"$N_{\rm min}$"), ("mf_v0", r"$V_0$"),
        ("mf_v1", r"$V_1$"), ("mf_v2", r"$V_2$")]
yy = np.arange(len(ROWS))[::-1]
for dy, rung, c, m in ((0.16, "pasted", CT, "s"), (-0.16, "bind", CB, "o")):
    v = [CAP[f"{k}:{rung}"].get("trusted_proj", CAP[f"{k}:{rung}"].get("proj"))
         for k, _ in ROWS]
    e = [[CAP[f"{k}:{rung}"]["proj"] - CAP[f"{k}:{rung}"]["boot_p16"] for k, _ in ROWS],
         [CAP[f"{k}:{rung}"]["boot_p84"] - CAP[f"{k}:{rung}"]["proj"] for k, _ in ROWS]]
    axc.errorbar(v, yy + dy, xerr=e, fmt=m, color=c, ms=3.4, lw=0.8, capsize=1.5,
                 label=LBL[rung])
axc.axvline(1.0, color=CF, lw=1.4, zorder=0)
axc.text(1.0, len(ROWS) - 0.45, "full response", color=CF, fontsize=6,
         ha="center", va="bottom")
for y in yy:
    axc.axhline(y, color="0.92", lw=0.5, zorder=0)
axc.set_yticks(yy)
axc.set_yticklabels([lab for _, lab in ROWS], fontsize=7.5)
axc.set_xlim(0.0, 1.55)
axc.set_xlabel("captured fraction of the\nDMO $\\rightarrow$ full-hydro response",
               fontsize=7.5)
axc.legend(fontsize=6.5, frameon=False, loc="lower right")
axc.tick_params(labelsize=7)
panel_label(axc, "(c)")
save(fig, str(IMGS / f"figA1_fullhydro_wl{SUF}"))
plt.close(fig)
if VARIANT == "oldfid":
    print("oldfid variant: figA1 only; skipping the gas figure")
    raise SystemExit(0)
plt.close(fig)
print("wrote figA1_fullhydro_wl; capture S(ell): bind %.3f pasted %.3f" %
      (CAP["S(ell):bind"]["trusted_proj"], CAP["S(ell):pasted"]["trusted_proj"]))

# ═════════════════════════════════════════════════════════════════════════
# Figure A2 — the gas ladder
# ═════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(TWO_COL[0], 3.4))
gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.14, wspace=0.46)
pref = ELL * (ELL + 1) / (2 * np.pi)
for j, (cube, key, slab) in enumerate(
        ((Y, "cl_y", r"$\ell(\ell+1)C_\ell^{yy}/2\pi$"),
         (T, "cl_tau", r"$\ell(\ell+1)C_\ell^{\tau\tau}/2\pi$"))):
    ax = fig.add_subplot(gs[0, j])
    rx = fig.add_subplot(gs[1, j], sharex=ax)
    full = np.nanmean(cube["hydro_full"][key][:, ZI], 0)
    sig_t = np.nanstd(cube["pasted"][key][:, ZI], 0)          # truth realization scatter
    tru = np.nanmean(cube["pasted"][key][:, ZI], 0)
    ax.fill_between(ELL[trust], (pref * (tru - sig_t))[trust],
                    (pref * (tru + sig_t))[trust], color=CT, alpha=0.15, lw=0,
                    label=r"truth $\pm1\sigma$ (per real.)" if j == 0 else None)
    for r in KRUNGS:
        ax.plot(ELL[trust], (pref * np.nanmean(cube[r][key][:, ZI], 0))[trust],
                label=LBL[r] if j == 0 else None, **STYLE[r])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(slab, fontsize=8)
    if j == 0:
        ax.legend(fontsize=5.8, frameon=False, loc="lower center")
    ax.tick_params(labelbottom=False, labelsize=7)
    panel_label(ax, f"({'ab'[j]})")
    for r in ("bind", "pasted"):
        rx.plot(ELL[trust],
                100 * (np.nanmean(cube[r][key][:, ZI], 0) / full - 1)[trust],
                **STYLE[r])
    rx.fill_between(ELL[trust], -100 * (sig_t / full)[trust],
                    100 * (sig_t / full)[trust], color=CT, alpha=0.15, lw=0)
    rx.axhline(0, color="0.75", lw=0.6)
    rx.set_xscale("log")
    rx.set_xlabel(r"$\ell$", fontsize=8)
    rx.set_ylabel("$\\Delta$ vs full hydro [%]", fontsize=7)
    rx.set_ylim(-70, 70)
    rx.tick_params(labelsize=7)

axf = fig.add_subplot(gs[:, 2])
for cube, mk, lab in ((Y, "o", r"$\bar y$"), (T, "s", r"$\bar\tau$")):
    full_m = np.nanmean(cube["hydro_full"]["mean"], 0)
    for r, ls in (("bind", "-"), ("pasted", "--")):
        m = np.nanmean(cube[r]["mean"], 0) / full_m
        axf.plot(ZS, m, marker=mk, ms=3.5, ls=ls,
                 color=STYLE[r]["color"], lw=0.9)
axf.axhline(1, color=CF, lw=1.2, zorder=0)
axf.set_yscale("log")
axf.set_xlabel(r"$z_s$", fontsize=8)
axf.set_ylabel("mean map value / full hydro", fontsize=8)
axf.text(0.04, 0.06, r"$\bar y$ (circles),  $\bar\tau$ (squares)", fontsize=6.5,
         color="0.4", transform=axf.transAxes)
from matplotlib.lines import Line2D  # noqa: E402
axf.legend(handles=[Line2D([], [], color=CB, lw=0.9, ls="-", label="BIND"),
                    Line2D([], [], color=CT, lw=0.9, ls="--", label="hydro-pasted")],
           fontsize=6.5, frameon=False, loc="center left")
axf.tick_params(labelsize=7)
panel_label(axf, "(c)")
save(fig, str(IMGS / "figA2_fullhydro_gas"))
plt.close(fig)

# numbers the captions quote
ytab = {r: float(np.nanmean((np.nanmean(Y[r]["cl_y"][:, ZI], 0) / np.nanmean(
    Y["hydro_full"]["cl_y"][:, ZI], 0))[(ELL >= 300) & (ELL <= 1000)]))
    for r in ("bind", "pasted")}
ttab = {r: float(np.nanmean((np.nanmean(T[r]["cl_tau"][:, ZI], 0) / np.nanmean(
    T["hydro_full"]["cl_tau"][:, ZI], 0))[(ELL >= 300) & (ELL <= 1000)]))
    for r in ("bind", "pasted")}
taumean = {r: float((np.nanmean(T[r]["mean"], 0) / np.nanmean(
    T["hydro_full"]["mean"], 0))[ZI]) for r in ("bind", "pasted")}
ymean = {r: float((np.nanmean(Y[r]["mean"], 0) / np.nanmean(
    Y["hydro_full"]["mean"], 0))[ZI]) for r in ("bind", "pasted")}
print("wrote figA2_fullhydro_gas")
print(f"  Cl_yy/full  300-1000: bind {ytab['bind']:.3f} pasted {ytab['pasted']:.3f}")
print(f"  Cl_tt/full  300-1000: bind {ttab['bind']:.3f} pasted {ttab['pasted']:.3f}")
print(f"  mean y /full z_s=1: bind {ymean['bind']:.3f} pasted {ymean['pasted']:.3f}")
print(f"  mean tau/full z_s=1: bind {taumean['bind']:.3f} pasted {taumean['pasted']:.3f}")
print(f"  S band: n1000 pairs n={n}, sig_S at trough "
      f"{sig_S[int(np.argmin(np.abs(ELL - 1.3e4)))]:.4f}")
