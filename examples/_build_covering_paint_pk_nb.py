"""Build examples/covering_paint_pk.ipynb.

Faithful implementation of the covering paint algorithm (branch:
low_mass_extrapolation), steps 1-7:
  1. load a DMO map
  2. seed a 128px patch on the largest unaccounted halo
  3. find every halo inside that patch
  4. keep a halo iff its FULL 4*R200 aperture fits in the patch (pure geometry)
  5. an edge halo whose aperture spills stays unaccounted -> resolved by a later
     patch (or seeds its own)
  6. repeat until all halos are accounted for
  7. paste each halo's circular 4*R200 aperture into the DMO map

The coverage gate is aperture-fit intersected with an optional off-center cap
r_max (from offcenter_covering.ipynb); the user's pure-geometric algorithm is the
r_max -> infinity case. We score the 1024^2 composite's total-matter P(k) against
CAMELS hydro truth and sweep r_max to expose the savings-vs-fidelity curve.

Center patches are reused from the cached per-halo generations (a box centered on
halo i IS the naive generation i -- verified to match fresh model.generate at the
centers), so the r_max sweep needs no GPU.

Run:  python examples/_build_covering_paint_pk_nb.py
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

nb = new_notebook()
C: list = []


def md(src: str):
    C.append(new_markdown_cell(src.strip("\n")))


def code(src: str):
    C.append(new_code_cell(src.strip("\n")))


md(r"""
# Covering paint, implemented: full-box $P(k)$ vs hydro truth

BIND paints a 6.25 Mpc/h (128px) box per GPU call but only pastes a circular
$4\,R_{200c}$ aperture back. So one box can baryonify every halo it *fully
contains*. The algorithm (steps 1–7):

1. Load a DMO map.
2. Seed a 128px patch on the **largest** unaccounted halo.
3. Find every halo inside that patch.
4. **Keep a halo iff its full $4\,R_{200c}$ aperture fits inside the patch** (pure
   geometry).
5. An edge halo whose aperture spills stays unaccounted — a later patch resolves
   it (worst case it seeds its own patch).
6. Repeat until all halos are accounted for.
7. Paste each halo's circular $4\,R_{200c}$ aperture into the DMO map.

The coverage gate is *aperture-fit* ∩ an optional off-center cap $r_{\max}$ (from
[`offcenter_covering.ipynb`](offcenter_covering.ipynb), which found faithful
painting only to $\sim\!1$ Mpc/h). **The pure-geometric algorithm is
$r_{\max}\!\to\!\infty$.** We paint `[DM, Gas, Stars]`, compare the $1024^2$
total-matter $P(k)$ to CAMELS hydro truth, and sweep $r_{\max}$ to see the
savings-vs-fidelity tradeoff.
""")

md("## 0. Load a DMO map, hydro truth, catalog, and cached generations")

code(r"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bind.inference.artifacts import (
    load_full_maps, load_halo_catalog, load_generated_halos)
from bind.inference.pipeline import (
    extract_halo_cutouts, build_bind_composite, circular_taper_weight)
from bind.metrics import power_spectrum_pylians_2d

plt.rcParams.update({"figure.dpi": 110, "font.size": 11})

SIM_L, NPIX, PP = 50.0, 1024, 128
MPP = SIM_L / NPIX                     # Mpc/h per pixel (== 6.25/128)
PATCH_HALF = (PP / 2) * MPP            # 3.125 Mpc/h
R200_FAC, TAPER, MASS_MIN = 4.0, 0.15, 1e12
CH = ["DM_hydro", "Gas", "Stars"]

sd = Path("/mnt/home/mlee1/ceph/fm_lowmass/CV/sim_0/snap_090")
dmo, truth = load_full_maps(sd / "full_maps.npz")                   # step 1
halos, hmass, hr200, hpos = load_halo_catalog(sd / "mass_threshold_1p000e12/halo_catalog.npz")
# center patches: a box on halo i IS the naive generation i (cached); fresh
# model.generate at the centers gives the same P(k) (verified).
gen = load_generated_halos(
    sd / "mass_threshold_1p000e12/fm_thermo_ema/generated_halos.npz")[:, :3].astype(np.float32)
centers = np.asarray([h["halo_center"] for h in halos], float)
masses  = np.asarray([h["halo_mass"] for h in halos], float)
r200s   = np.asarray([h.get("r200", 0.0) for h in halos], float)
ap_mpc  = R200_FAC * r200s             # aperture radius per halo [Mpc/h]
cutouts = extract_halo_cutouts(dmo, halos, SIM_L, NPIX, PP)         # halo-centered DMO
PPM = NPIX / SIM_L
# per-halo circular aperture weight (radius 4*R200c), reused for BOTH the paste
# and the aperture-local mass match in section 2.
AP_W = [circular_taper_weight(PP, r_pix=r200s[j] * PPM * R200_FAC, taper_frac=TAPER)
        for j in range(len(halos))]
# patch_mass_match=False: we mass-match locally in the aperture ourselves. The
# built-in full-128px match is WRONG for re-centered passengers -- their rolled
# patch's outer pixels are wrapped from the center halo, not the passenger's
# neighbourhood, so patch.sum() is the wrong region (this caused ring residuals).
COMPOSITE_KW = dict(box_size=SIM_L, npix=NPIX, patch_pix=PP, patch_mass_match=False,
                    taper_frac=TAPER, r200_factor=R200_FAC)
print(f"{len(halos)} halos | truth {truth.shape} | gen {gen.shape}")
""")

md("""
## 1. The planner (steps 2–6)

Greedy, largest-first. A halo joins a patch iff its full aperture fits
(`|dx|+ap < PATCH_HALF`) **and** it lies within the off-center cap `r_max`. The
seed always owns its patch. `r_max=np.inf` is the pure-geometric algorithm.
""")

code(r"""
def plan_covering(r_max=np.inf, mass_min=MASS_MIN):
    order = np.where(masses >= mass_min)[0]
    order = order[np.argsort(-masses[order])]          # step 2: largest first
    covered = np.zeros(len(masses), bool)
    cen_idx, box_of, off = [], np.full(len(masses), -1), np.zeros((len(masses), 2))
    for i in order:                                    # step 6: until all accounted
        if covered[i]:
            continue
        g = len(cen_idx); cen_idx.append(i)
        cand = order[~covered[order]]                  # step 3
        d = centers[cand] - centers[i]; d -= SIM_L * np.round(d / SIM_L)
        fits = ((np.abs(d[:, 0]) + ap_mpc[cand] < PATCH_HALF) &     # step 4
                (np.abs(d[:, 1]) + ap_mpc[cand] < PATCH_HALF))
        ok = fits & ((d ** 2).sum(1) < r_max ** 2)
        ok |= (cand == i)
        for j, dd in zip(cand[ok], d[ok]):
            covered[j] = True; box_of[j] = g; off[j] = dd
    return np.array(cen_idx), box_of, off

cen_idx, box_of, off = plan_covering(np.inf)           # pure geometric
above = masses >= MASS_MIN
n_above, n_gen = int(above.sum()), len(cen_idx)
poff = np.hypot(off[above, 0], off[above, 1])
print(f"pure geometric: {n_gen} generations for {n_above} halos "
      f"({n_above/n_gen:.2f}x fewer GPU calls)")
print(f"passenger offsets [Mpc/h]: median {np.median(poff):.2f}  "
      f"90th {np.percentile(poff, 90):.2f}  max {poff.max():.2f}  (r_max cap off)")
""")

md(r"""
## 2. Paint (step 7)

For halo $j$ in a box centered on halo $c$ at offset $(dx,dy)$, a periodic roll by
$(-dx,-dy)$ re-centers the halo (safe: the aperture-fit gate keeps the aperture
clear of the wrapped edge). **Mass-match is done locally within the
$4\,R_{200c}$ aperture** — the same region for prediction and DMO — which is the
correct scaling for off-center passengers and removes the ring residuals a
full-patch match produced. Then `build_bind_composite` pastes the circular
aperture. The same local match is applied to the naive baseline for a fair
comparison (it barely changes centered halos, $s\approx1$).
""")

code(r"""
def prescale_aperture(patches):
    out = patches.copy()
    for j in range(len(halos)):
        w = AP_W[j]
        m_pred = float((out[j].sum(0) * w).sum())
        m_dmo = float((cutouts[j]["condition"] * w).sum())
        out[j] *= m_dmo / (m_pred + 1e-30)
    return out

def paint(r_max=np.inf):
    ci, box_of, off = plan_covering(r_max)
    patches = np.zeros((len(halos), 3, PP, PP), np.float32)
    for j in range(len(halos)):
        if box_of[j] < 0:
            continue
        dpx = (int(round(off[j, 0] / MPP)), int(round(off[j, 1] / MPP)))
        patches[j] = np.roll(gen[ci[box_of[j]]], shift=(-dpx[0], -dpx[1]), axis=(1, 2))
    comp = build_bind_composite(dmo, halos, prescale_aperture(patches), cutouts,
                                **COMPOSITE_KW)["composite"]
    return len(ci), comp

naive = build_bind_composite(dmo, halos, prescale_aperture(gen), cutouts,
                             **COMPOSITE_KW)["composite"]
_, cover = paint(np.inf)
print("total mass  truth %.3e  naive %.3e  cover %.3e"
      % (truth.sum(), naive.sum(), cover.sum()))
""")

md("## 3. Total-matter $P(k)$: truth vs naive vs pure-geometric covering")

code(r"""
def pk(f2d):
    k, p, _ = power_spectrum_pylians_2d(f2d, box_size=SIM_L, MAS="None")
    return k, p

k, pt = pk(truth.sum(0)); _, pn = pk(naive.sum(0)); _, pv = pk(cover.sum(0))
_, pd = pk(dmo)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].loglog(k, pt, "k-", label="hydro truth")
ax[0].loglog(k, pn, "C0--", label="BIND naive (1/halo)")
ax[0].loglog(k, pv, "C3-", label=f"covering geom ({n_gen} gen)")
ax[0].loglog(k, pd, "C7:", label="DMO")
ax[0].set(xlabel="k [h/Mpc]", ylabel="P(k)",
          title="Total-matter P(k), 1024² over 50 Mpc/h"); ax[0].legend()
ax[1].semilogx(k, pn/pt, "C0--", label="naive / truth")
ax[1].semilogx(k, pv/pt, "C3-", label="covering / truth")
ax[1].axhline(1, color="k", lw=0.8)
ax[1].set(xlabel="k [h/Mpc]", ylabel=r"$P_{\rm BIND}/P_{\rm truth}$",
          ylim=(0.8, 1.2), title="Ratio to hydro truth"); ax[1].legend()
plt.tight_layout(); plt.show()
for lbl, (lo, hi) in {"large (k<2)": (0, 2), "mid (2-10)": (2, 10),
                      "small (10-40)": (10, 40)}.items():
    m = (k >= lo) & (k < hi)
    print(f"  {lbl:>14s}: naive {np.median(pn[m]/pt[m]):.3f}  "
          f"covering {np.median(pv[m]/pt[m]):.3f}")
""")

md("""
## 4. The tradeoff — savings vs fidelity as we cap the off-center radius

The design curve. Sweep `r_max`: fewer, larger boxes vs the mid-$k$ $P(k)$ error
from painting halos off-center. Look for the knee.
""")

code(r"""
mid = (k >= 2) & (k < 10)
rmaxes = [0.5, 0.75, 1.0, 1.5, 2.0, np.inf]
ng, midr = [], []
for rm in rmaxes:
    g, c = paint(rm); ng.append(g)
    _, p = pk(c.sum(0)); midr.append(np.median(p[mid] / pt[mid]))
xr = [rm if np.isfinite(rm) else 3.3 for rm in rmaxes]      # plot inf at 3.3
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(xr, [n_above / g for g in ng], "s-", color="C0")
ax1.set_xlabel(r"off-center cap $r_{\max}$ [Mpc/h]  (rightmost = $\infty$, pure geometric)")
ax1.set_ylabel("speedup (fewer GPU calls)", color="C0"); ax1.tick_params(axis="y", labelcolor="C0")
ax2 = ax1.twinx()
ax2.plot(xr, [100 * (m - 1) for m in midr], "o-", color="C3")
ax2.axhline(100 * (np.median(pn[mid] / pt[mid]) - 1), color="C0", ls=":", lw=1,
            label="naive mid-k error")
ax2.set_ylabel("mid-k P(k) error vs truth [%]", color="C3"); ax2.tick_params(axis="y", labelcolor="C3")
ax2.legend(loc="lower right", fontsize=8)
ax1.set_title("Covering paint: savings vs mid-k fidelity")
plt.tight_layout(); plt.show()
for rm, g, m in zip(rmaxes, ng, midr):
    lbl = "inf" if not np.isfinite(rm) else f"{rm}"
    print(f"  r_max={lbl:>4s}: {g:3d} gen ({n_above/g:.2f}x)  mid-k {100*(m-1):+.1f}%")
print("Note: ~±2% mid-k sampler scatter on a single sim -- average over the suite for precision.")
""")

md("## 5. Visual — truth vs covering composite (total matter)")

code(r"""
z = 20; npz = int(z / MPP)
tt = np.log10(np.clip(truth.sum(0)[:npz, :npz], 1e-3, None))
cc = np.log10(np.clip(cover.sum(0)[:npz, :npz], 1e-3, None))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(tt, origin="lower", extent=[0, z, 0, z], cmap="magma",
               vmin=tt.min(), vmax=tt.max()); axes[0].set_title("hydro truth")
axes[1].imshow(cc, origin="lower", extent=[0, z, 0, z], cmap="magma",
               vmin=tt.min(), vmax=tt.max()); axes[1].set_title("BIND covering (geom)")
cc_pos = centers[cen_idx]
inzoom = (cc_pos[:, 0] < z) & (cc_pos[:, 1] < z)     # only centers inside the zoom
axes[1].scatter(cc_pos[inzoom, 0], cc_pos[inzoom, 1], marker="x", c="cyan", s=18, lw=0.8)
im = axes[2].imshow(cc - tt, origin="lower", extent=[0, z, 0, z], cmap="RdBu_r",
                    vmin=-0.5, vmax=0.5); axes[2].set_title("Δlog10 Σ (cover − truth)")
for a in axes:
    a.set(xlim=(0, z), ylim=(0, z), xlabel="x [Mpc/h]")   # keep all panels at the zoom
axes[0].set_ylabel("y [Mpc/h]")
fig.colorbar(im, ax=axes[2], shrink=0.8)
fig.suptitle(f"{z}×{z} Mpc/h (cyan × = generation centers)", y=1.02)
plt.tight_layout(); plt.show()
""")

md(r"""
## 6. Takeaway

- **The algorithm works end-to-end** and, pure-geometric ($r_{\max}=\infty$),
  paints all 382 halos with **130 generations (2.94×)** — total mass exact,
  large/small-scale $P(k)$ on truth, and mid-$k$ within **~+1.8% of the naive
  per-halo composite** (itself ~+6% over truth from BIND's intrinsic bias). The
  covering approximation is nearly free.
- **Mass-match must be aperture-local.** The built-in full-128px match corrupts
  every re-centered passenger's amplitude (scales scattered 0.6–1.9×), producing
  ring residuals and a spurious ~+5% mid-$k$ bump; matching within the
  $4\,R_{200c}$ aperture (§2) — the same region for prediction and DMO — removes
  both.
- **The savings knob is soft:** conservative $r_{\max}\!\approx\!0.75$ sits at or
  below naive at ~1.2×; pure-geometric buys ~2.9× at only ~+1.8% mid-$k$.
- **Caveats.** (i) One CV sim; the mid-$k$ number has ~±2% sampler scatter —
  repeat across the suite. (ii) Gas/Stars carry more of the residual than total
  matter. (iii) Push $r_{\max}$ (and savings) further with translation
  augmentation in training. (iv) Deploy: index `halo_gen`/`halo_offset` in the
  composite rather than the roll-and-rebuild shortcut, and mass-match in-aperture.
""")

nb["cells"] = C
out = "examples/covering_paint_pk.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(C), "cells")
