"""Two posterity companions to pfig_fm_model_curves (not wired into main.tex).

1. pfig_fm_model_curves_fiducial : the same 7-statistic ratio layout, but
   evaluated at the OUT-OF-DESIGN fiducial (the 1P fiducial, twobound/run_0049)
   instead of the 7 leave-one-out Sobol nodes. Measured curves and shipped-
   bundle predictions come from the fidswap closure cache
   (ceph/referee_work/fidswap/closure_fidswap.npz); everything is shown as a
   ratio to the Sobol design mean, tail-masked at 5% of the reference peak.

2. pfig_fm_model_curves_gas : the leave-one-out validation of the family model
   extended to the ell-spectra -- kappa-kappa plus the five gas legs (yy,
   tautau, ky, ktau, ytau) -- for the same seven design-spanning Sobol nodes
   as the shipped WL figure. The gas legs use the log-space family bases of
   gas_families.py with the shipped eight latents (transfer, no re-search).

Both render into figs_preview/ only.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python posterity_model_curves.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np

P1 = Path("/mnt/home/mlee1/BIND/papers/01_pipeline")
CEPH = Path("/mnt/home/mlee1/ceph")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figs_preview"
sys.path.insert(0, str(P1))
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import os
os.chdir(P1)
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402
from family_model import FamilyModel  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CB = COLORS["bind"]
REF_FLOOR = 0.05
fm = FamilyModel()
# BIND_CAMPAIGN=n1000 propagates into the runpy'd gas_families.py (campaign
# stats + run_id-aligned latents) and tags this script's own outputs.
CAMPAIGN = os.environ.get("BIND_CAMPAIGN", "sci50")


def _tag(stem):
    return stem + ("_n1000" if CAMPAIGN == "n1000" else "")


AMPS = np.load(P1 / "figs_preview/amplitude_sets.npz", allow_pickle=True)
CLO = (None if CAMPAIGN == "n1000" else
       np.load(CEPH / "referee_work/fidswap/closure_fidswap.npz", allow_pickle=True))
STATS7 = ["clk", "pdf", "pk", "mn", "v0", "v1", "v2"]
SPECN = {"clk": (r"$S(\ell)$", r"$\ell$", True), "pdf": (r"$P(\nu)$", r"$\nu$", False),
         "pk": (r"$N_{\rm pk}$", r"$\nu$", False), "mn": (r"$N_{\rm min}$", r"$\nu$", False),
         "v0": (r"$V_0$", r"$\nu$", False), "v1": (r"$V_1$", r"$\nu$", False),
         "v2": (r"$V_2$", r"$\nu$", False)}


def masked_ref(ref):
    keep = np.abs(ref) >= REF_FLOOR * np.nanmax(np.abs(ref))
    return np.where(keep, ref, np.nan), keep


def shade_masked(axes, xg, keep):
    if (~keep).any():
        mid = np.r_[xg[0] - 0.5 * (xg[1] - xg[0]), 0.5 * (xg[1:] + xg[:-1]),
                    xg[-1] + 0.5 * (xg[-1] - xg[-2])]
        i = 0
        while i < len(keep):
            if not keep[i]:
                j = i
                while j + 1 < len(keep) and not keep[j + 1]:
                    j += 1
                for a in axes:
                    a.axvspan(mid[i], mid[j + 1], color="0.93", lw=0, zorder=0)
                i = j + 1
            else:
                i += 1


# ════════════════════════════════════════════════════════════════════════
# Figure 1: the out-of-design fiducial, 7-statistic ratio layout
# ════════════════════════════════════════════════════════════════════════
if CAMPAIGN == "n1000":
    print("pfig_fm_model_curves_fiducial SKIPPED under n1000: its closure cache "
          "(referee_work/fidswap/closure_fidswap.npz) is a sci50 product.")
else:
    fig = plt.figure(figsize=(TWO_COL[0], 6.4))
    gs = fig.add_gridspec(4, 4, height_ratios=[2.2, 1.0, 2.2, 1.0], hspace=0.45, wspace=0.42)
    summary = []
    for i, st in enumerate(STATS7):
        r0, c0 = 2 * (i // 4), i % 4
        ax = fig.add_subplot(gs[r0, c0])
        rx = fig.add_subplot(gs[r0 + 1, c0], sharex=ax)
        slab, xlab, logx = SPECN[st]
        xg = fm.grid(st)
        meas = np.asarray(CLO[f"new_meas_{st}"], float)
        pred = np.asarray(CLO[f"new_pred_{st}"], float)
        ref = np.asarray(AMPS[f"{st}__mean"], float)
        rr, keep = masked_ref(ref)
        ax.plot(xg, meas / rr, color="k", lw=1.2)
        ax.plot(xg, pred / rr, color=CB, lw=1.2, ls="--")
        dd = 100 * (pred - meas) / rr
        rx.plot(xg, dd, color=CB, lw=0.9)
        summary.append((st, float(np.nanmedian(np.abs(dd))), float(np.nanmax(np.abs(dd)))))
        shade_masked((ax, rx), xg, keep)
        if logx:
            ax.set_xscale("log")
        ax.axhline(1, color="0.6", lw=0.6, zorder=1)
        rx.axhline(0, color="0.6", lw=0.6)
        rx.set_ylim(*(lambda v: (-v, v))(max(0.5, 1.4 * float(np.nanpercentile(np.abs(dd), 99)))))
        ax.set_ylabel(slab + " / design mean", fontsize=6.5)
        rx.set_xlabel(xlab, fontsize=7)
        rx.set_ylabel(r"$\Delta$ [%]", fontsize=6)
        ax.tick_params(labelsize=6, labelbottom=False)
        rx.tick_params(labelsize=6)
        panel_label(ax, f"({'abcdefg'[i]})", loc="upper right")
    axl = fig.add_subplot(gs[2:4, 3])
    axl.axis("off")
    axl.legend(handles=[Line2D([], [], color="k", lw=1.2, label="measured (run_0049)"),
                        Line2D([], [], color=CB, lw=1.2, ls="--", label="model (shipped 8 latents)"),
                        Line2D([], [], color="0.93", lw=6, label="masked: |ref| < 5% of peak")],
               loc="upper center", fontsize=6.2, frameon=False, handlelength=2.2)
    axl.text(0.5, 0.55, "OUT-OF-DESIGN fiducial\n(median / max |$\\Delta$|, % of design mean)",
             ha="center", va="top", fontsize=6.2, transform=axl.transAxes)
    axl.text(0.5, 0.42, "\n".join(f"{SPECN[s][0]}:  {md:.2f} / {mx:.1f}"
                                  for s, md, mx in summary),
             ha="center", va="top", fontsize=6.0, transform=axl.transAxes, linespacing=1.5)
    save(fig, str(OUT / "pfig_fm_model_curves_fiducial"))
    plt.close(fig)
    print("wrote pfig_fm_model_curves_fiducial")
    for s_, md, mx in summary:
        print(f"  {s_:4s} median |D| = {md:.2f}%  max {mx:.1f}%")

# ════════════════════════════════════════════════════════════════════════
# Figure 2: LOO model curves for the ell-spectra (kk + the five gas legs)
# ════════════════════════════════════════════════════════════════════════
# gas machinery: re-run gas_families.py and reuse its RESULTS (bases, labels)
g = runpy.run_path(str(HERE / "gas_families.py"))
RES, band, EDG, dsn, ZI = g["RESULTS"], g["band"], g["EDG"], g["dsn"], g["ZI"]
LAT, XG = g["LAT"], g["XG"]
STATSG = {"yy": "t__cl_yy__value", "tt": "t__cl_tt__value", "ky": "t__cl_kappa_y__value",
          "kt": "t__cl_kappa_tau__value", "yt": "t__cl_yt__value"}

# the same 7 design-spanning nodes as the shipped WL figure
coef = np.load(P1 / "latent_model_coeffs.npz")
EDGb = coef["ell_edges"]
_eld = np.asarray(dsn["a__suppression__ell"], float)
Yclk = np.asarray(dsn["t__suppression__value"], float)[:, 1, :]
Yclk = np.stack([np.nanmean(Yclk[:, (_eld >= EDGb[i]) & (_eld < EDGb[i + 1])], 1)
                 for i in range(len(EDGb) - 1)], axis=1)[:, :fm.basis("clk").shape[1]]
key = (Yclk - Yclk.mean(0)).mean(1)
SHOW = [int(np.argsort(key)[int(q * (len(key) - 1))]) for q in np.linspace(0, 1, 7)]
print(f"LOO nodes (design-spanning, as shipped): {SHOW}")

PANELS = []
# kappa-kappa from the shipped WL machinery (S(ell) expansion).  Amplitudes
# are RE-MEASURED from the current dataset's curves on the fixed shipped basis
# (same free-lstsq definition as the stored a_sobol): the stored 256-row array
# must never be indexed with campaign-dataset positions, whose run_ids can be
# a non-contiguous subset until tracing completes.
B_kk, M_kk = np.asarray(AMPS["clk__basis"], float), np.asarray(AMPS["clk__mean"], float)
A_kk = np.linalg.lstsq(B_kk.T, (Yclk - M_kk).T, rcond=None)[0].T
PANELS.append(("kk", r"$C_\ell^{\kappa\kappa}\,(S)$", fm.grid("clk"),
               Yclk, A_kk, B_kk, M_kk, False))
for st, tgt in STATSG.items():
    Y = np.asarray(dsn[tgt], float)
    if Y.ndim == 3:
        Y = Y[:, ZI, :]
    Yb = np.array([band(Y[i]) for i in range(len(Y))])
    Lb = np.log(np.where(Yb > 0, Yb, np.nan))
    B = RES[st]["basis"]
    D = Lb - np.nanmean(Lb, 0)
    A = np.linalg.lstsq(B.T, np.where(np.isfinite(D), D, 0).T, rcond=None)[0].T
    PANELS.append((st, RES[st]["lab"], XG, Lb, A, B, np.nanmean(Lb, 0), True))

Lc = LAT - LAT.mean(0)
fig = plt.figure(figsize=(TWO_COL[0], 5.2))
gs = fig.add_gridspec(4, 3, height_ratios=[2.2, 1.0, 2.2, 1.0], hspace=0.45, wspace=0.40)
summary = []
for i, (st, slab, xg, Yv, A, B, MEAN, is_log) in enumerate(PANELS):
    r0, c0 = 2 * (i // 3), i % 3
    ax = fig.add_subplot(gs[r0, c0])
    rx = fig.add_subplot(gs[r0 + 1, c0], sharex=ax)
    # ratio panels: the reference never vanishes (Cl > 0 everywhere), so the
    # nu-style tail mask does not apply -- it would just mask the steep fall
    # of the raw spectrum
    ref = np.exp(MEAN) if is_log else MEAN
    rr, keep = ref, np.ones(len(ref), bool)
    resid = []
    for r in SHOW:
        keeprows = np.arange(len(LAT)) != r
        W, *_ = np.linalg.lstsq(np.c_[Lc[keeprows], np.ones(keeprows.sum())],
                                A[keeprows], rcond=None)
        a_pred = np.r_[Lc[r], 1.0] @ W
        if is_log:
            meas = np.exp(Yv[r])
            model = np.exp(MEAN + a_pred @ B)
        else:
            meas = Yv[r]
            model = MEAN + a_pred @ B
        ax.plot(xg, meas / rr, color="k", lw=1.0)
        ax.plot(xg, model / rr, color=CB, lw=1.0, ls="--")
        dd = 100 * (model / meas - 1)          # fractional model error per node
        rx.plot(xg, dd, color=CB, lw=0.7, alpha=0.8)
        resid.append(dd)
    resid = np.asarray(resid)
    summary.append((slab, float(np.sqrt(np.nanmean(resid ** 2))),
                    float(np.nanmax(np.abs(resid)))))
    ax.set_xscale("log")
    ax.axhline(1, color="0.6", lw=0.6, zorder=1)
    rx.axhline(0, color="0.6", lw=0.6)
    rx.set_xscale("log")
    rx.set_ylim(*(lambda v: (-v, v))(max(2.0, 1.3 * float(np.nanpercentile(np.abs(resid), 99)))))
    BARLAB = {"kk": r"$S_\ell/\bar{S}_\ell$",
              "yy": r"$C_\ell^{yy}/\bar{C}_\ell^{yy}$",
              "tt": r"$C_\ell^{\tau\tau}/\bar{C}_\ell^{\tau\tau}$",
              "ky": r"$C_\ell^{\kappa y}/\bar{C}_\ell^{\kappa y}$",
              "kt": r"$C_\ell^{\kappa\tau}/\bar{C}_\ell^{\kappa\tau}$",
              "yt": r"$C_\ell^{y\tau}/\bar{C}_\ell^{y\tau}$"}
    ax.set_ylabel(BARLAB[st], fontsize=6.5)
    rx.set_xlabel(r"$\ell$", fontsize=7)
    rx.set_ylabel(r"$\Delta$ [%]", fontsize=6)
    ax.tick_params(labelsize=6, labelbottom=False)
    rx.tick_params(labelsize=6)
    panel_label(ax, f"({'abcdef'[i]})", loc="upper right")
fig.legend(handles=[Line2D([], [], color="k", lw=1.0, label="measured"),
                    Line2D([], [], color=CB, lw=1.0, ls="--", label="model")],
           loc="lower center", ncol=2, fontsize=6.5, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
save(fig, str(OUT / _tag("pfig_fm_model_curves_gas")))
plt.close(fig)
print(f"wrote {_tag('pfig_fm_model_curves_gas')}")
for s_, rms, mx in summary:
    print(f"  {s_:24s} rms {rms:5.2f}%  max {mx:5.1f}%")
