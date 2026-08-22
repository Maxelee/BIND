"""fidswap re-render: paper Fig 4 (per-halo radial profiles) == notebook fig06_radial_profiles.

Cell body copied verbatim from _build_figures_nb.py; the single edit is
    pf = np.load(SCI/'profiles/perhalo_fid_snap096.npz')
 -> pf = np.load(FID_PROF)   (referee_work/fidswap/profiles, run_0049 patches)
The OLD fiducial's mean-ratio table is printed alongside for the before/after.
"""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.lines import Line2D

from fsfig_common import (COLORS, FID_PROF, OLD_PROF, ONE_COL, TRUTH_PROF,
                          panel_label, plt, save_imgs, tee)

tee("fig04_profiles")

pt = np.load(TRUTH_PROF)
pf = np.load(FID_PROF)
assert np.allclose(pf["M"], pt["M"]) and np.array_equal(pf["npix"], pt["npix"]), (
    "fid/truth per-halo caches are not paired")

XE6 = pf["x_edges"]
xc6 = np.sqrt(XE6[1:] * XE6[:-1])
MEDG6 = np.array([1e13, 2.5e13, 6e13, 1.5e14, 1e15])
MB6 = np.log10(MEDG6)
ib6 = np.digitize(pf["M"], MEDG6) - 1
CNT6 = np.array([(ib6 == b).sum() for b in range(4)])
CB6 = plt.cm.viridis([0.78, 0.55, 0.32, 0.05])

SIGMA_T, M_P, MSUN_G, MPC_CM, HH = 6.6524e-25, 1.6726e-24, 1.989e33, 3.0857e24, 0.6774
PIX_MPCH = 6.25 / 128.0
a_snap = 1.0 / (1.0 + float(pf["z"]))
K_TAU = SIGMA_T * (0.88 / M_P) * MSUN_G / (HH * (PIX_MPCH * a_snap / HH * MPC_CM) ** 2)

CH6 = [("prof_y", 1.0, r"$y(r)$", "(a)", 25),
       ("prof_gas", K_TAU, r"$\tau(r)$", "(b)", 25),
       ("prof_star", 1.0, r"$\Sigma_\star(r)$ [$M_\odot h^{-1}$ px$^{-1}$]", "(c)", 45)]

fig, ax = plt.subplots(6, 1, figsize=(ONE_COL[0], 9.4), sharex=True,
                       gridspec_kw=dict(height_ratios=[2.6, 1, 2.6, 1, 2.6, 1]))
rng6 = np.random.default_rng(11)
Z6 = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for j, (key, sc, ylab, tag, rlim) in enumerate(CH6):
        axM, axR = ax[2 * j], ax[2 * j + 1]
        B, T = sc * pf[key], sc * pt[key]
        for b in range(4):
            s = ib6 == b
            res_ok = (np.isfinite(B[s]).mean(0) >= 0.8) & (np.median(pf["npix"][s], 0) >= 4)
            bad = np.where(~res_ok)[0]
            okx = np.zeros_like(res_ok)
            okx[(bad[-1] + 1 if len(bad) else 0):] = True
            medB = np.nanmedian(B[s], 0)
            medT = np.nanmedian(T[s], 0)
            loB, hiB = np.nanpercentile(B[s], [16, 84], axis=0)
            m = okx & (medB > 0) & (medT > 0)
            axM.fill_between(xc6[m], loB[m], hiB[m], color=CB6[b], alpha=0.15, lw=0)
            axM.loglog(xc6[m], medB[m], color=CB6[b], lw=1.4,
                       label=f"{MB6[b]:.1f}-{MB6[b+1]:.1f} ($N$={CNT6[b]})")
            axM.loglog(xc6[m], medT[m], color=CB6[b], ls="--", lw=1.1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dln = np.log(B[s] / T[s])
            dln[~(np.isfinite(B[s]) & np.isfinite(T[s]) & (B[s] > 0) & (T[s] > 0))] = np.nan
            medR = np.nanmedian(dln, 0)
            loR, hiR = np.nanpercentile(dln, [16, 84], axis=0)
            idxb = rng6.integers(0, int(s.sum()), (500, int(s.sum())))
            seR = np.full_like(medR, np.nan)
            for aa in np.where(m)[0]:
                seR[aa] = np.std(np.nanmedian(dln[idxb, aa], axis=1))
            to_pct = lambda v: 100 * (np.exp(v) - 1)  # noqa: E731
            axR.fill_between(xc6[m], to_pct(loR[m]), to_pct(hiR[m]),
                             color=CB6[b], alpha=0.12, lw=0)
            axR.plot(xc6[m], to_pct(medR[m]), color=CB6[b], lw=1.2)
            axR.errorbar(xc6[m], to_pct(medR[m]), yerr=(100 * np.exp(medR) * seR)[m],
                         fmt="none", ecolor=CB6[b], elinewidth=0.6, capsize=0, zorder=6)
            zin = np.abs(medR / seR)[m & (xc6 <= 1.0)]
            Z6[key, b] = (np.nanmin(zin), np.nanmax(zin))
        axR.axhline(0, color="0.55", ls=":", lw=0.8)
        axR.set_xscale("log")
        axR.set_ylim(-rlim, rlim)
        axR.set_ylabel(r"$\Delta$ [%]")
        axM.set_ylabel(ylab)
        panel_label(axM, tag, loc="upper right")
ax[0].legend(title=r"$\log_{10}\, M_{200c}/{\rm M}_\odot$", title_fontsize=5.6,
             fontsize=5.2, loc="lower left")
ax[2].legend(handles=[Line2D([], [], color="0.3", lw=1.4, label="BIND"),
                      Line2D([], [], color="0.3", ls="--", lw=1.1,
                             label="hydro-pasted")], fontsize=5.2, loc="lower left")
ax[-1].set_xlabel(r"$r/r_{200c}$")
fig.align_ylabels(ax)
fig.tight_layout(h_pad=0.25)
save_imgs(fig, "fig04_radial_profiles")
plt.close(fig)

print(f"per-halo profiles, snap {int(pf['snap'])} (z = {float(pf['z']):.5f}); native "
      f"x = r/R200c annuli {XE6[0]:.2f}-{XE6[-1]:.1f}; tau = {K_TAU:.6e} * Sigma_gas "
      f"at a = {a_snap:.5f}; RAW (no background subtraction)")
print("mass bins " + ", ".join(f"{MB6[b]:.2f}-{MB6[b+1]:.2f} (N={CNT6[b]})" for b in range(4)))
print("residual-offset significance |z| = |median ln ratio|/bootSE per annulus, "
      "range over drawn annuli inside r200c (NEW fiducial):")
for key, _, _, _, _ in CH6:
    print(f"  {key:10s} " + " | ".join(
        f"{MB6[b]:.1f}-{MB6[b+1]:.1f}: {Z6[key, b][0]:.1f}-{Z6[key, b][1]:.1f}" for b in range(4)))

print("\nmean BIND/truth of the median profiles over drawn annuli inside r200c "
      "-- OLD vs NEW fiducial (main.tex:513):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for src, P in (("OLD", np.load(OLD_PROF)), ("NEW", pf)):
        for key, lab in [("prof_y", "y"), ("prof_gas", "tau"), ("prof_star", "M_star"),
                         ("prof_DM", "DM (control)")]:
            row = []
            for b in range(4):
                s = ib6 == b
                res_ok = (np.isfinite(P[key][s]).mean(0) >= 0.8) & (np.median(P["npix"][s], 0) >= 4)
                bad = np.where(~res_ok)[0]
                okx = np.zeros_like(res_ok)
                okx[(bad[-1] + 1 if len(bad) else 0):] = True
                medB = np.nanmedian(P[key][s], 0)
                medT = np.nanmedian(pt[key][s], 0)
                m = okx & (medB > 0) & (medT > 0) & (xc6 <= 1.0)
                row.append(f"{MB6[b]:.2f}-{MB6[b+1]:.2f}: {np.mean(medB[m]/medT[m]):.3f}")
            print(f"  {src} {lab:14s} " + "; ".join(row))
print("\nDONE fig04")
