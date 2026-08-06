#!/usr/bin/env python
"""paper_latent_measurement.py -- how the four latents are measured
(pfig_s4a_latent_measurement, 2026-08-06 arc: the latent-choice beat).

Top row: one real hinge-bin halo from the fiducial snap-096 composite slabs,
four channels (Sigma_tot, Sigma_gas, Sigma_star, T_mw) with the measurement
geometry drawn on the maps: R500c = 0.659 r200 (solid), r200 (dashed), the
fixed 2.5-3.0 Mpc/h background annulus (dotted ring). Bottom row: the
population step -- per-halo values across all fiducial halos in the hinge
bin (log10 m_tot_500c_bg in [13.3, 13.6)), median marked: ONE number per
universe per latent.

Conventions identical to the atlas chain (ANALYTIC_LATENT_MODEL.md S2.2-2.3):
aperture sums of pixel surface mass, total/gas background-subtracted with
the annulus mean, stars raw, T gas-mass-weighted with raw gas weights.

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python paper_latent_measurement.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
FIDLC = CEPH / "bind_lightcone_tng"
ATLAS = CEPH / "bind_science/halo_atlas/fid_snap096.npz"
assert Path.cwd().name == "01_pipeline", "run from papers/01_pipeline/"

OB_OM = 0.0486 / 0.3089
PIX = 6.25 / 128.0
R500_FAC = 0.659
BG_ANN = (2.5, 3.0)
HB = (13.3, 13.6)                                # hinge bin

# ── pick a display halo from the fiducial slabs (deterministic: the hinge-
# bin halo whose cylinder mass is closest to the bin center) ────────────────
best = None
for slab in sorted((FIDLC / "snap_096").glob("composite_slab*.npz")):
    dd = np.load(slab)
    if int(dd["n_halos"]) == 0:
        continue
    gen = dd["generated_patches"].astype(np.float64)
    thr = dd["thermo_patches"].astype(np.float64)
    r200 = np.asarray(dd["halo_r200"], float)
    P = gen.shape[-1]
    cen = P // 2
    yy, xx = np.mgrid[0:P, 0:P]
    rr = np.hypot(xx - cen, yy - cen) * PIX
    ann = (rr >= BG_ANN[0]) & (rr < BG_ANN[1])
    tot = gen.sum(1)
    sig_tot = tot[:, ann].mean(1)
    for h in range(len(r200)):
        m5 = rr <= R500_FAC * r200[h]
        mt = tot[h][m5].sum() - sig_tot[h] * m5.sum()
        if mt <= 0:
            continue
        lm = np.log10(mt)
        if HB[0] <= lm < HB[1]:
            dctr = abs(lm - 0.5 * (HB[0] + HB[1]))
            if best is None or dctr < best[0]:
                best = (dctr, gen[h], thr[h], float(r200[h]), lm)
assert best is not None, "no hinge-bin halo found in the fiducial slabs"
_, genh, thrh, r200h, lmh = best
print(f"display halo: log10 m_tot_500c_bg = {lmh:.2f}, r200 = {r200h:.2f} "
      f"Mpc/h (R500c = {R500_FAC*r200h:.2f})")

# ── per-halo population values from the fid atlas (identical reducers) ──────
fa = np.load(ATLAS)
lmf = np.log10(fa["m_tot_500c_bg"])
sel = (lmf >= HB[0]) & (lmf < HB[1])
fbar_h = ((fa["m_gas_500c_bg"] + fa["m_star_500c"])
          / fa["m_tot_500c_bg"])[sel] / OB_OM
fstar_h = (fa["m_star_500c"] / fa["m_tot_500c_bg"])[sel] / OB_OM
okg = sel & (fa["m_gas_200c_bg"] > 0)
cgas_h = (fa["m_gas_500c_bg"] / fa["m_gas_200c_bg"])[okg]
logT_h = np.log10(fa["T_mw_500c"][sel])

# ── figure ──────────────────────────────────────────────────────────────────
EXT = [-3.125, 3.125, -3.125, 3.125]
CHANS = [
    (np.log10(genh.sum(0) + 1e-30), "viridis",
     r"$\Sigma_{\rm tot}$", r"$m^{\rm bg}_{\rm tot}(<\!R)$: denominators",
     True),
    (np.log10(genh[1] + 1e-30), "cividis",
     r"$\Sigma_{\rm gas}$",
     r"$m^{\rm bg}_{\rm gas}(<\!R_{500c})$, $m^{\rm bg}_{\rm gas}(<\!R_{200})$",
     True),
    (np.log10(genh[2] + 1e-30), "magma",
     r"$\Sigma_\star$", r"$m_\star(<\!R_{500c})$ (no bg sub.)", False),
    (np.log10(np.where(thrh[1] > 0, thrh[1], np.nan)), "inferno",
     r"$T$", r"$T_{\rm mw}=\frac{\sum T\,\Sigma_{\rm gas}}"
             r"{\sum\Sigma_{\rm gas}}(<\!R_{500c})$", False),
]
fig, AX = plt.subplots(2, 4, figsize=(TWO_COL[0], 4.4))
for k, (img, cmap, cl, note, ann_on) in enumerate(CHANS):
    ax = AX[0, k]
    ax.imshow(img, cmap=cmap, extent=EXT, origin="lower")
    ax.add_patch(Circle((0, 0), R500_FAC * r200h, fill=False, ec="w",
                        lw=1.0))
    ax.add_patch(Circle((0, 0), r200h, fill=False, ec="w", lw=0.8,
                        ls="--"))
    if ann_on:
        for rad in BG_ANN:
            ax.add_patch(Circle((0, 0), rad, fill=False, ec="w", lw=0.6,
                                ls=":"))
    if k == 0:
        ax.annotate(r"$R_{500c}$", xy=(0, R500_FAC * r200h),
                    xytext=(-2.6, 1.1), color="w", fontsize=5.5,
                    arrowprops=dict(arrowstyle="-", color="w", lw=0.5))
        ax.annotate(r"$R_{200}$", xy=(r200h * 0.7, r200h * 0.7),
                    xytext=(1.2, 1.9), color="w", fontsize=5.5,
                    arrowprops=dict(arrowstyle="-", color="w", lw=0.5))
        ax.text(0, -2.82, "bg annulus", color="w", fontsize=5.0,
                ha="center")
        ax.set_ylabel(r"$y$ [$h^{-1}$Mpc]", fontsize=6.5)
    ax.set_title(f"{cl}\n{note}", fontsize=6.0)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=5)
    if k > 0:
        ax.set_yticklabels([])
    panel_label(ax, f"({chr(97 + k)})", color="w")

HISTS = [
    (fbar_h, r"$\tilde f_{\rm bar}$ per halo",
     r"med $\to \tilde f_{\rm bar}$", COLORS["bind"]),
    (fstar_h, r"$\tilde f_\star$ per halo",
     r"med $\to \tilde f_\star$", COLORS["truth"]),
    (cgas_h, r"$m^{\rm bg}_{\rm gas,500c}/m^{\rm bg}_{\rm gas,200c}$ per halo",
     r"med $\to c_{\rm gas}$", COLORS["highlight"]),
    (logT_h, r"$\log_{10}T_{\rm mw,500c}$ per halo",
     r"med $\to \log\tilde T$", COLORS["secondary"]),
]
for k, (vals, xlab, medlab, col) in enumerate(HISTS):
    ax = AX[1, k]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    ax.hist(vals, bins=np.linspace(lo, hi, 36), color=col, alpha=0.75)
    med = float(np.median(vals))
    ax.axvline(med, color="k", lw=1.2, ls="--")
    ax.text(0.97, 0.92, f"{medlab}\n= {med:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.6)
    ax.set_xlabel(xlab, fontsize=6.0)
    ax.set_yticks([])
    ax.tick_params(labelsize=5)
    if k == 0:
        ax.set_ylabel(f"halos in the hinge bin\n(n = {int(sel.sum())})",
                      fontsize=6.0)
    panel_label(ax, f"({chr(101 + k)})")
fig.tight_layout(h_pad=1.0)
save(fig, "figs_v2/pfig_s4a_latent_measurement")
plt.close(fig)
print(f"medians: fbar {np.median(fbar_h):.3f}, fstar {np.median(fstar_h):.3f},"
      f" cgas {np.median(cgas_h):.3f}, logT {np.median(logT_h):.3f} "
      f"(the fiducial's latent vector)")
