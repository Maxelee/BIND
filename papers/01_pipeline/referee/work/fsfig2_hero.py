"""fidswap re-render: paper Fig 2 (hero) == notebook fig00_hero.

Cell body copied verbatim from _build_figures_nb.py.  Two edits:
  * the three fiducial map panels (b) kappa, (c) tau, (d) y read FID_RUN
    (twobound/run_0049) instead of bind_lightcone_tng;
  * the bottom-row response panels subtract the NEW fiducial, so
    Delta-kappa = kappa_node - kappa_fiducial is still a seed-paired difference
    against the fiducial the paper actually ships.
Realization 0 is streamed with load_maps_prefix (one realization, one plane)
instead of np.load(...)[0, zi] -- an I/O change only; the arrays are identical.
"""
from __future__ import annotations

import gc

import matplotlib.patheffects as pe
import numpy as np
from fsfig_common import (
    CAMPAIGN,
    DMO_RUN,
    FID_RUN,
    LC,
    TB_PARAMS,
    TB_ROOT,
    TWO_COL,
    ZI,
    load_maps_prefix,
    plt,
    save_imgs,
    tee,
)
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from scipy.ndimage import gaussian_filter

tee("fig02_hero")

FOV = 5.0
TB = TB_ROOT                       # campaign-dependent (bind_science or bind_n1000)
tbp = np.load(TB_PARAMS)           # the 1P design itself is campaign-independent
A_LO, A_HI = float(tbp[0, 2]), float(tbp[1, 2])


def _hero_map(path, key, zi):
    m = load_maps_prefix(path, key, 1, zi)[0].astype(float)
    m = gaussian_filter(m, 1.0)
    gc.collect()
    return m


kap_dmo0 = _hero_map(DMO_RUN / "kappa_maps.npz", "kappa", ZI)
kap_fid0 = _hero_map(FID_RUN / "kappa_maps.npz", "kappa", ZI)
tau_fid0 = _hero_map(FID_RUN / "tau_maps.npz", "tau", -1)
y_fid0 = _hero_map(FID_RUN / "y_maps.npz", "y", -1)
kap_weak0 = _hero_map(TB / "run_0000/kappa_maps.npz", "kappa", ZI)
kap_str0 = _hero_map(TB / "run_0001/kappa_maps.npz", "kappa", ZI)

vk0 = float(np.percentile(np.abs(np.concatenate([kap_dmo0.ravel(), kap_fid0.ravel()])), 99.9))
knorm0 = Normalize(-vk0, vk0)
tau_hi0 = np.percentile(tau_fid0, 99.8)
tnorm0 = LogNorm(vmin=tau_hi0 / 30.0, vmax=tau_hi0)
ynorm0 = LogNorm(vmin=np.percentile(y_fid0[y_fid0 > 0], 50),
                 vmax=np.percentile(y_fid0, 99.8))

dkw0 = kap_weak0 - kap_fid0
dks0 = kap_str0 - kap_fid0
vr0 = float(np.percentile(np.abs(np.concatenate([dkw0.ravel(), dks0.ravel()])), 99.5))
rnorm0 = Normalize(-vr0, vr0)

fig = plt.figure(figsize=(TWO_COL[0], 5.0))
gs0 = fig.add_gridspec(2, 4, height_ratios=[1.0, 2.05], hspace=0.10, wspace=0.34,
                       left=0.015, right=0.955, top=0.985, bottom=0.012)
axd = fig.add_subplot(gs0[0, 0])
axk = fig.add_subplot(gs0[0, 1])
axt = fig.add_subplot(gs0[0, 2])
axy = fig.add_subplot(gs0[0, 3])
axw = fig.add_subplot(gs0[1, 0:2])
axs = fig.add_subplot(gs0[1, 2:4])
ext = [0, FOV, 0, FOV]
STK = [pe.withStroke(linewidth=2.0, foreground="black")]
BOX0 = dict(facecolor="w", edgecolor="none", alpha=0.78, pad=1.6)

panels0 = [(axd, kap_dmo0, "RdBu_r", knorm0, "(a) DMO $\\kappa$"),
           (axk, kap_fid0, "RdBu_r", knorm0, "(b) BIND $\\kappa$"),
           (axt, tau_fid0, "magma", tnorm0, "(c) BIND $\\tau$"),
           (axy, y_fid0, "inferno", ynorm0, "(d) BIND $y$")]
imk0 = None
for a, m, cmap, nrm, lab in panels0:
    im = a.imshow(m, origin="lower", extent=ext, cmap=cmap, norm=nrm, rasterized=True)
    a.set_xticks([])
    a.set_yticks([])
    a.text(0.5, 1.02, lab, transform=a.transAxes, fontsize=7.5, ha="center", va="bottom")
    if nrm is knorm0:
        imk0 = im
        continue
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.04)
    cb.set_label(r"$\tau$" if nrm is tnorm0 else r"$y$", fontsize=7, labelpad=1)
    cb.locator = LogLocator(base=10.0, subs=(1.0, 3.0), numticks=12)
    cb.formatter = LogFormatterSciNotation(minor_thresholds=(3, 0.4))
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6)
cbk0 = fig.colorbar(imk0, ax=[axd, axk], fraction=0.028, pad=0.025)
cbk0.set_label(r"$\kappa$", fontsize=7, labelpad=1)
vkt0 = np.floor(vk0 * 1000) / 1000
cbk0.set_ticks([-vkt0, 0.0, vkt0])
cbk0.ax.tick_params(labelsize=6)
axd.plot([0.35, 1.35], [0.30, 0.30], color="w", lw=1.6, path_effects=STK)
axd.text(0.85, 0.45, r"$1^\circ$", color="w", fontsize=7, ha="center", va="bottom",
         path_effects=STK)

imw = axw.imshow(dkw0, origin="lower", extent=ext, cmap="RdBu_r", norm=rnorm0, rasterized=True)
axs.imshow(dks0, origin="lower", extent=ext, cmap="RdBu_r", norm=rnorm0, rasterized=True)
for a, lab in [(axw, rf"(e) $A_{{\rm SN1}} = {A_LO:g}$"),
               (axs, rf"(f) $A_{{\rm SN1}} = {A_HI:g}$")]:
    a.set_xticks([])
    a.set_yticks([])
    a.text(0.025, 0.975, lab, transform=a.transAxes, fontsize=8, va="top", bbox=BOX0)
cbb = fig.colorbar(imw, ax=[axw, axs], fraction=0.024, pad=0.015, extend="both")
cbb.set_label(r"$\kappa_{\rm node} - \kappa_{\rm fiducial}$", fontsize=8, labelpad=2)
vrt0 = np.floor(vr0 * 10000) / 10000
cbb.set_ticks([-vrt0, 0.0, vrt0])
cbb.ax.tick_params(labelsize=7)

save_imgs(fig, "fig02_hero")
plt.close(fig)
print(f"hero: fiducial realization 0; one-parameter A_SN1 bracket "
      f"twobound/run_0000 (A_SN1={A_LO:g}) / run_0001 (A_SN1={A_HI:g}); "
      f"mean tau (fiducial, total column) = {tau_fid0.mean():.3e}; "
      f"mean y (fiducial) = {y_fid0.mean():.3e}; kappa rms DMO/painted = "
      f"{kap_dmo0.std():.4f}/{kap_fid0.std():.4f}; kappa norm +-{vk0:.4f}; "
      f"Delta-kappa rms weak/strong = {dkw0.std():.4f}/{dks0.std():.4f} "
      f"({100*dkw0.std()/kap_fid0.std():.1f}%/{100*dks0.std()/kap_fid0.std():.1f}% of "
      f"kappa rms); Delta-kappa norm +-{vr0:.4f} ({vrt0:.4f} plotted tick)")

# ── before/after on the stamped scalars ─────────────────────────────────────
# bind_lightcone_tng is the retired 50-real fiducial trace; comparing an n1000
# figure against it would mix realization counts, so skip it under n1000.
if CAMPAIGN == "n1000":
    print("\n(old-vs-new fiducial stamps skipped: BIND_CAMPAIGN=n1000)")
    print("\nDONE fig02")
    raise SystemExit(0)

o_kap = _hero_map(LC / "kappa_maps.npz", "kappa", ZI)
o_tau = _hero_map(LC / "tau_maps.npz", "tau", -1)
o_y = _hero_map(LC / "y_maps.npz", "y", -1)
o_dkw, o_dks = kap_weak0 - o_kap, kap_str0 - o_kap
print("\n### OLD vs NEW fiducial, hero stamps")
print(f"{'mean tau (total col)':28s} {o_tau.mean():.4e} -> {tau_fid0.mean():.4e} "
      f"(x{tau_fid0.mean()/o_tau.mean():.4f})")
print(f"{'mean y':28s} {o_y.mean():.4e} -> {y_fid0.mean():.4e} "
      f"(x{y_fid0.mean()/o_y.mean():.4f})")
print(f"{'kappa rms painted':28s} {o_kap.std():.5f} -> {kap_fid0.std():.5f} "
      f"(x{kap_fid0.std()/o_kap.std():.5f});  DMO {kap_dmo0.std():.5f}")
print(f"{'Delta-kappa rms weak/strong':28s} {o_dkw.std():.5f}/{o_dks.std():.5f} -> "
      f"{dkw0.std():.5f}/{dks0.std():.5f}")
print(f"{'% of kappa rms weak/strong':28s} {100*o_dkw.std()/o_kap.std():.2f}/"
      f"{100*o_dks.std()/o_kap.std():.2f} -> {100*dkw0.std()/kap_fid0.std():.2f}/"
      f"{100*dks0.std()/kap_fid0.std():.2f}")
print(f"{'pixel corr(new,old) kappa':28s} {np.corrcoef(kap_fid0.ravel(), o_kap.ravel())[0,1]:.6f}")
print(f"{'pixel corr(new,old) tau':28s} {np.corrcoef(tau_fid0.ravel(), o_tau.ravel())[0,1]:.6f}")
print("\nDONE fig02")
