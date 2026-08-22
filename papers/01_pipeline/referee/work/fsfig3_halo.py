"""fidswap re-render: paper Fig 3 (halo-level validation).

Paper Fig 3 == notebook cell `fig03a_halo_raw` (md5-verified in the depmap scope),
so that is the figure written to imgs/fig03_halo_validation_fidswap.{png,pdf}.
The sibling notebook cell `fig03_halo_validation` (bg-subtracted f_gas scaling
relations) is ALSO re-rendered, to imgs/fig03_halo_scaling_fidswap.{png,pdf},
because its linear-in-logM calibration coefficients are quoted in the text.

Both cell bodies are copied verbatim from _build_figures_nb.py; the only edit is
    fa = np.load(SCI/'halo_atlas/fid_snap096.npz')
 -> fa = np.load(FID_ATLAS)                  (referee_work/fidswap, run_0049)
Every stamped number is printed for the OLD and the NEW fiducial side by side.
"""
from __future__ import annotations

import numpy as np

from fsfig_common import (BAND_ALPHA, COLORS, FID_ATLAS, ONE_COL, OLD_ATLAS,
                          TRUTH_ATLAS, panel_label, plt, running_stat,
                          save_imgs, tee)

tee("fig03_halo")

ta = np.load(TRUTH_ATLAS)
ATL = {"OLD (bind_lightcone_tng)": np.load(OLD_ATLAS), "NEW (twobound replica)": np.load(FID_ATLAS)}

PIV3 = 13.5
edges = np.arange(13.0, 15.01, 0.2)
MLAB = r"$\log_{10}\, M_{200c}/{\rm M}_\odot$"
OB_OM = 0.0486 / 0.3089

SIGMA_T, M_P, MSUN_G, MPC_CM, HH = 6.6524e-25, 1.6726e-24, 1.989e33, 3.0857e24, 0.6774
PIX_MPCH = 6.25 / 128.0


# ═══════════════════════════════════════════════════════════════════════════
# numbers first: OLD vs NEW for every quantity the text quotes
# ═══════════════════════════════════════════════════════════════════════════
def channels(fa):
    logM = np.log10(fa["M_fof"])
    a3a = 1.0 / (1.0 + float(fa["z"]))
    K = SIGMA_T * (0.88 / M_P) * MSUN_G / (HH * (PIX_MPCH * a3a / HH * MPC_CM) ** 2)
    return logM, a3a, K, [
        ("Y_200c", fa["Y_200c"], ta["Y_200c"]),
        ("tau_200c", K * PIX_MPCH**2 * fa["m_gas_200c"], K * PIX_MPCH**2 * ta["m_gas_200c"]),
        ("m_star", fa["m_star_200c"], ta["m_star_200c"]),
    ]


print("\n### RAW R200c aperture ratios BIND/hydro-pasted truth (paper Fig 3, main.tex:502)")
print(f"{'quantity':10s} {'source':26s} {'median':>8s} {'bootSE(ln)':>11s} {'z=|lnr|/SE':>11s} "
      f"{'Delta(logM) intercept':>22s} {'slope %/dex':>12s}")
RES = {}
for src, fa in ATL.items():
    logM, a3a, K, CH = channels(fa)
    for key, vf, vt in CH:
        ok = np.isfinite(vf) & np.isfinite(vt) & (vf > 0) & (vt > 0)
        rr = vf[ok] / vt[ok]
        dlz = np.log(rr)
        rngz = np.random.default_rng(7)
        se = np.std(np.median(dlz[rngz.integers(0, dlz.size, (2000, dlz.size))], axis=1))
        cenr, medr, _, _, ser = running_stat(logM[ok], 100 * (rr - 1), edges, boot=200, min_n=1)
        ft = np.polyfit(cenr - PIV3, medr, 1, w=1 / np.where(ser > 0, ser, np.nan))
        RES[key, src] = (np.median(rr), se, ft, medr, cenr)
        print(f"{key:10s} {src:26s} {np.median(rr):8.4f} {se:11.4f} "
              f"{abs(np.median(dlz))/se:11.1f} {ft[1]:+21.2f}% {ft[0]:+11.2f}")

print("\n### bg-subtracted f_gas,200c and the other atlas medians (notebook fig 3 / appendix)")
for src, fa in ATL.items():
    fg_f = fa["m_gas_200c_bg"] / fa["m_tot_200c_bg"]
    fg_t = ta["m_gas_200c_bg"] / ta["m_tot_200c_bg"]
    fg5_f = fa["m_gas_500c_bg"] / fa["m_tot_500c_bg"]
    fg5_t = ta["m_gas_500c_bg"] / ta["m_tot_500c_bg"]
    print(f"{src:26s} f_gas200c {np.nanmedian(fg_f/fg_t):.4f}  f_gas500c "
          f"{np.nanmedian(fg5_f/fg5_t):.4f}  Y_500c {np.nanmedian(fa['Y_500c']/ta['Y_500c']):.4f}  "
          f"T_mw500c {np.nanmedian(fa['T_mw_500c']/ta['T_mw_500c']):.4f}  "
          f"Pe_mw500c {np.nanmedian(fa['Pe_mw_500c']/ta['Pe_mw_500c']):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE A -- paper Fig 3 == notebook fig03a_halo_raw, NEW fiducial
# ═══════════════════════════════════════════════════════════════════════════
fa = np.load(FID_ATLAS)
logM, a3a, K_TAU3A, CH3A_v = channels(fa)
CH3A = [("Y_200c", CH3A_v[0][1], CH3A_v[0][2], True, r"$\log_{10} Y_{200c}$", "(a)"),
        ("tau_200c", CH3A_v[1][1], CH3A_v[1][2], True, r"$\log_{10}\, \tau_{200c}$", "(b)"),
        ("m_star", CH3A_v[2][1], CH3A_v[2][2], True,
         r"$\log_{10}\, M_{\star,200c}/{\rm M}_\odot$", "(c)")]

fig, ax = plt.subplots(6, 1, figsize=(ONE_COL[0], 8.8), sharex=True,
                       gridspec_kw=dict(height_ratios=[2.6, 1, 2.6, 1, 2.6, 1]))
fits_a = {}
for j, (key, vf, vt, logy, ylab, tag) in enumerate(CH3A):
    axM, axR = ax[2 * j], ax[2 * j + 1]
    for v, c, ls, lab in [(vf, COLORS["bind"], "-", "BIND"),
                          (vt, COLORS["truth"], "--", "hydro-pasted")]:
        okr = np.isfinite(v) & (v > 0)
        cen, med, lo, hi, _ = running_stat(logM[okr], np.log10(v[okr]) if logy else v[okr],
                                           edges, min_n=1)
        axM.fill_between(cen, lo, hi, color=c, alpha=BAND_ALPHA, lw=0)
        axM.plot(cen, med, color=c, ls=ls, lw=1.6, label=lab)
    axM.set_ylabel(ylab)
    panel_label(axM, tag)
    okb = np.isfinite(vf) & np.isfinite(vt) & (vf > 0) & (vt > 0)
    dln = np.log(vf[okb] / vt[okb])
    cend, medd, lod, hid, sed = running_stat(logM[okb], dln, edges, boot=400, min_n=1)
    to_pct = lambda v: 100 * (np.exp(v) - 1)  # noqa: E731
    axR.axhline(0, color="0.55", ls=":", lw=0.8)
    axR.fill_between(cend, to_pct(lod), to_pct(hid), color=COLORS["bind"],
                     alpha=BAND_ALPHA, lw=0)
    axR.plot(cend, to_pct(medd), color=COLORS["bind"], lw=1.3)
    axR.errorbar(cend, to_pct(medd), yerr=100 * np.exp(medd) * sed, fmt="none",
                 ecolor=COLORS["bind"], elinewidth=0.7, capsize=0, zorder=5)
    axR.set_ylim(-25, 25)
    axR.set_ylabel(r"$\Delta$ [%]")
    rr = vf[okb] / vt[okb]
    cenr, medr, _, _, ser = running_stat(logM[okb], 100 * (rr - 1), edges, boot=200, min_n=1)
    fits_a[key] = (np.polyfit(cenr - PIV3, medr, 1, w=1 / np.where(ser > 0, ser, np.nan)),
                   np.median(rr), medr, cenr)
ax[0].set_xlim(13.0, 15.05)
ax[0].legend(loc="lower right")
ax[-1].set_xlabel(MLAB)
fig.align_ylabels(ax)
fig.tight_layout(h_pad=0.25)
save_imgs(fig, "fig03_halo_validation")
plt.close(fig)

print("\n### paper Fig 3 stamps (NEW fiducial), notebook-format")
print("RAW R200c aperture sums (no background subtraction), BIND/truth:")
for key, lab in [("Y_200c", "Y_200c (raw sum)"),
                 ("tau_200c", "tau_200c (= K_tau x raw M_gas,200c)"),
                 ("m_star", "M_star,200c RAW")]:
    ft, mr, medr, cenr = fits_a[key]
    print(f"  {lab:36s} median ratio {mr:.3f}; Delta(logM) = {ft[1]:+.2f}% "
          f"{ft[0]:+.2f}%/dex x (logM-{PIV3}) [binned {medr.max():+.1f}% -> {medr.min():+.1f}%]")
print(f"tau_200c = {K_TAU3A:.6e}/px^2-mass x integrated Sigma_gas at a = {a3a:.5f}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE B -- notebook fig03_halo_validation (bg-subtracted scaling relations)
# ═══════════════════════════════════════════════════════════════════════════
fg_f = fa["m_gas_200c_bg"] / fa["m_tot_200c_bg"]
fg_t = ta["m_gas_200c_bg"] / ta["m_tot_200c_bg"]
SCT = (np.arange(logM.size) % 4 == 0) | (logM >= 14.2)
TAIL = logM >= 14.2

fig, ax = plt.subplots(3, 1, figsize=(ONE_COL[0], 7.2), sharex=True)
for atl, c, ls, lab in [(fa, COLORS["bind"], "-", "BIND"),
                        (ta, COLORS["truth"], "--", "hydro-pasted")]:
    ok = atl["Y_200c"] > 0
    cen, med, lo, hi, _ = running_stat(logM[ok], np.log10(atl["Y_200c"][ok]), edges, min_n=1)
    ax[0].fill_between(cen, lo, hi, color=c, alpha=BAND_ALPHA, lw=0)
    ax[0].plot(cen, med, color=c, ls=ls, lw=1.6, label=lab)
ax[0].scatter(logM[SCT], np.log10(np.where(fa["Y_200c"] > 0, fa["Y_200c"], np.nan))[SCT],
              s=1.0, color=COLORS["bind"], alpha=0.10, lw=0, rasterized=True)
ax[0].scatter(logM[TAIL], np.log10(np.where(fa["Y_200c"] > 0, fa["Y_200c"], np.nan))[TAIL],
              s=4.0, color=COLORS["bind"], alpha=0.5, lw=0, rasterized=True)
ax[0].set_ylabel(r"$\log_{10} Y_{200c}$")
ax[0].set_xlim(13.0, 15.05)
ax[0].legend(loc="lower right")
panel_label(ax[0], "(a)")

ok = (fa["Y_200c"] > 0) & (ta["Y_200c"] > 0)
ry = fa["Y_200c"][ok] / ta["Y_200c"][ok]
cen_y, med_y, _, _, se_y = running_stat(logM[ok], 100 * (ry - 1), edges, boot=200, min_n=1)
rng_b = np.random.default_rng(3)
se_y_all = np.std([np.median(ry[rng_b.integers(0, len(ry), len(ry))]) for _ in range(200)])
fit_y = np.polyfit(cen_y - PIV3, med_y, 1, w=1 / np.where(se_y > 0, se_y, np.nan))

for fg, c, ls in [(fg_f, COLORS["bind"], "-"), (fg_t, COLORS["truth"], "--")]:
    ok2 = np.isfinite(fg)
    cen, med, lo, hi, _ = running_stat(logM[ok2], fg[ok2], edges, min_n=1)
    ax[1].fill_between(cen, lo, hi, color=c, alpha=BAND_ALPHA, lw=0)
    ax[1].plot(cen, med, color=c, ls=ls, lw=1.6)
ax[1].axhline(OB_OM, color=COLORS["dmo"], ls=":", lw=1.0)
ax[1].text(14.99, OB_OM + 0.004, r"$\Omega_b/\Omega_m$", color=COLORS["dmo"],
           fontsize=6, ha="right")
ax[1].set_ylabel(r"$f_{\rm gas,200c}$")
ax[1].set_ylim(top=0.172)
panel_label(ax[1], "(b)")

okf = np.isfinite(fg_f) & np.isfinite(fg_t) & (fg_t != 0)
rg = fg_f[okf] / fg_t[okf]
cen_g, med_g, _, _, se_g = running_stat(logM[okf], 100 * (rg - 1), edges, boot=200, min_n=1)
fit_g = np.polyfit(cen_g - PIV3, med_g, 1, w=1 / np.where(se_g > 0, se_g, np.nan))

for atl, c, ls, lab in [(fa, COLORS["bind"], "-", "BIND"),
                        (ta, COLORS["truth"], "--", "hydro-pasted")]:
    oks = atl["m_star_200c"] > 0
    cen, med, lo, hi, _ = running_stat(logM[oks], np.log10(atl["m_star_200c"][oks]),
                                       edges, min_n=1)
    ax[2].fill_between(cen, lo, hi, color=c, alpha=BAND_ALPHA, lw=0)
    ax[2].plot(cen, med, color=c, ls=ls, lw=1.6, label=lab)
ax[2].scatter(logM[SCT], np.log10(fa["m_star_200c"])[SCT], s=1.0,
              color=COLORS["bind"], alpha=0.10, lw=0, rasterized=True)
ax[2].scatter(logM[TAIL], np.log10(fa["m_star_200c"])[TAIL], s=4.0,
              color=COLORS["bind"], alpha=0.5, lw=0, rasterized=True)
ax[2].set_ylabel(r"$\log_{10}\, M_{\star,200c}/{\rm M}_\odot$")
panel_label(ax[2], "(c)")

oks2 = (fa["m_star_200c"] > 0) & (ta["m_star_200c"] > 0)
rs_star = fa["m_star_200c"][oks2] / ta["m_star_200c"][oks2]
cen_s, med_s, _, _, se_s = running_stat(logM[oks2], 100 * (rs_star - 1), edges,
                                        boot=200, min_n=1)
fit_s = np.polyfit(cen_s - PIV3, med_s, 1, w=1 / np.where(se_s > 0, se_s, np.nan))

for a3 in ax:
    a3.set_xlabel(MLAB if a3 is ax[-1] else "")
fig.align_ylabels(ax)
fig.tight_layout(w_pad=0.9)
save_imgs(fig, "fig03_halo_scaling")
plt.close(fig)

print("\n### notebook fig03_halo_validation stamps (NEW fiducial)")
print(f"median per-halo Y_200c ratio BIND/truth = {np.median(ry):.3f} +/- {se_y_all:.3f} "
      f"(offset significance {abs(np.median(ry)-1)/se_y_all:.0f} sigma); "
      f"median f_gas,200c ratio = {np.nanmedian(fg_f/fg_t):.3f}; "
      f"median M_star,200c ratio = {np.median(rs_star):.3f}")
print("MASS-DEPENDENT amplitude offset -- recommended linear-in-logM calibration:")
print(f"  Delta_Y(logM)    = {fit_y[1]:+.2f}% {fit_y[0]:+.2f}%/dex x (logM-{PIV3})   "
      f"[binned-median range {med_y.max():+.1f}% -> {med_y.min():+.1f}%]")
print(f"  Delta_fgas(logM) = {fit_g[1]:+.2f}% {fit_g[0]:+.2f}%/dex x (logM-{PIV3})   "
      f"[binned-median range {med_g.max():+.1f}% -> {med_g.min():+.1f}%]")
print(f"  Delta_Mstar(logM)= {fit_s[1]:+.2f}% {fit_s[0]:+.2f}%/dex x (logM-{PIV3})   "
      f"[binned-median range {med_s.max():+.1f}% -> {med_s.min():+.1f}%]")
print("  after correction, binned-median residuals: "
      f"max|Y| = {np.max(np.abs(med_y - np.polyval(fit_y, cen_y - PIV3))):.1f}%, "
      f"max|fgas| = {np.max(np.abs(med_g - np.polyval(fit_g, cen_g - PIV3))):.1f}%")
print("\nDONE fig03")
