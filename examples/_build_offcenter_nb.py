"""Build examples/offcenter_covering.ipynb.

Motivation (branch: low_mass_extrapolation)
--------------------------------------------
BIND paints one 6.25 Mpc/h box per generation but only pastes a ~4xR200
circular aperture back into the mosaic -- so ~85% of every painted box is
thrown away. If a neighbouring halo already sits inside that box, it is
*already painted* and needs no second GPU call. That turns "baryonify every
halo above threshold" into a geometric covering problem: cover all halos with
the fewest freely-placed 6.25 Mpc/h boxes.

The binding constraint on how much you can save is a *physics* question, not a
geometry one: BIND was trained on patches CENTERED on the target halo, so a halo
that shares a box with a neighbour is painted OFF-CENTER -- potentially
out-of-distribution. This notebook measures that. It takes isolated halos from a
CV sim, re-paints each one at a ladder of offsets from box center, and asks how
far off-center the model stays faithful (r_max). r_max is exactly the covering
radius, so the final section turns it into the achievable GPU-call reduction on
the real halo field.

Everything is read from the low-mass suite artifacts already on ceph
(full_maps.npz + halo_catalog.npz + truth_thermo_patches.npz) plus the
fm_thermo_ema checkpoint; generation cells cache to npz so re-render is cheap.

Run:  python examples/_build_offcenter_nb.py
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
# Off-center validity & covering: can one generation baryonify many halos?

BIND paints a **6.25 Mpc/h box** per GPU call but only pastes back a
**circular $4\,R_{200c}$ aperture** — for a $10^{13}\,M_\odot/h$ group that
aperture is ~2.4 Mpc/h across inside a 6.25 Mpc/h box, so **~85% of every
painted box is discarded**. Any neighbour whose aperture falls inside that box
is *already painted*: it needs no second GPU call. So "baryonify every halo
above threshold" is really a **covering problem** — cover all halos with the
fewest freely-placed 6.25 Mpc/h boxes.

The catch is a physics constraint, not a geometry one. BIND was trained on
patches **centered on** the target halo. When one box serves several halos, the
neighbours are painted **off-center**, with a `condition`/`large_scale` context
centered on someone else — potentially out-of-distribution.

**This notebook measures the one number that sets the whole design:** how far
off-center ($r_{\max}$, Mpc/h) can a halo sit before its painted aperture drifts
from the centered baseline. Then:

- $r_{\max}$ large → the problem collapses to a fixed overlapping tile grid;
  GPU calls become $\approx \mathrm{area}/\mathrm{tile\ area}$, *independent of
  $N$*.
- $r_{\max}$ small → constrained covering: each halo must stay within $r_{\max}$
  of some box center.

The last section plugs the measured $r_{\max}$ into the **real CV halo field**
and reports the achievable reduction in GPU generations vs. the naive
one-call-per-halo path.
""")

md("## 0. Config + load the `fm_thermo` model")

code(r"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from bind.inference.artifacts import load_full_maps, load_halo_catalog
from bind.inference.pipeline import extract_multiscale
from bind.inference.paint import Model

plt.rcParams.update({"figure.dpi": 110, "font.size": 11})

# --- geometry (CV L50n512 full-box projection; must match the suite) --------
BOX_MPCH   = 50.0
NPIX       = 1024
PATCH_PIX  = 128
PATCH_MPCH = 6.25                      # side of one generation box
MPC_PER_PX = BOX_MPCH / NPIX           # 0.04883 -- IDENTICAL to PATCH_MPCH/PATCH_PIX
assert abs(MPC_PER_PX - PATCH_MPCH / PATCH_PIX) < 1e-6
THERMO_KEYS = ["compton_y", "T", "entropy", "P_e"]
MASS_CH     = ["DM_hydro", "Gas", "Stars"]

# --- artifacts (low-mass suite output on ceph) ------------------------------
OUTPUT_ROOT = Path("/mnt/home/mlee1/ceph/fm_lowmass")
SUITE, SNAP, MASS_DIR = "CV", 90, "mass_threshold_1p000e12"
RUN_DIR    = Path("/mnt/home/mlee1/ceph/fm_runs/fm_thermo")
CKPT       = RUN_DIR / "checkpoints/kept/keep_epoch064_ema.ckpt"
NORM_STATS = RUN_DIR / "norm_stats.npz"
CACHE      = Path("offcenter_cache");  CACHE.mkdir(exist_ok=True)

N_STEPS, BATCH = 20, 32                 # thermo validated to <0.03 dex at 20 steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Model.from_files(CKPT, NORM_STATS, device=DEVICE)
print(model)
assert model.predict_thermo, "expected the 7-channel (mass+thermo) model"
""")

md("""
## 1. Load a CV sim and select *isolated* halos

To isolate the *off-center* effect from "a bigger halo wandered into the frame",
we test **isolated** halos: the target must be the most massive object within
`ISO_MPCH` (patch half-width + max offset). We keep a mass ladder, emphasising
groups ($\\geq 10^{13}$) where the aperture is largest and the geometric offset
budget is tightest.
""")

code(r"""
sim_dir = OUTPUT_ROOT / SUITE / "sim_0" / f"snap_{SNAP:03d}"
dmo_fullbox, _ = load_full_maps(sim_dir / "full_maps.npz")
halos, halo_masses, halo_r200s, halo_positions = load_halo_catalog(
    sim_dir / MASS_DIR / "halo_catalog.npz")
sim_params = np.asarray(halos[0]["params"], dtype=np.float32)   # p14=0 already applied
centers = np.asarray([h["halo_center"] for h in halos], dtype=np.float64)  # Mpc/h (x,y)
masses  = np.asarray([h["halo_mass"]   for h in halos], dtype=np.float64)
r200s   = np.asarray([h.get("r200", 0.0) for h in halos], dtype=np.float64)  # Mpc/h
print(f"sim_0: DMO {dmo_fullbox.shape}, {len(halos)} halos, "
      f"logM in [{np.log10(masses.min()):.2f}, {np.log10(masses.max()):.2f}]")

ISO_MPCH  = PATCH_MPCH / 2 + 2.5       # dominant out to the largest offset tested
MAX_OFF   = 2.5                        # Mpc/h ladder cap (also aperture-limited below)

def _sep_xy(i):
    d = centers - centers[i]
    d -= BOX_MPCH * np.round(d / BOX_MPCH)         # periodic
    return np.hypot(d[:, 0], d[:, 1])

def isolated(i):
    # "Locally dominant": no MORE massive halo within ISO_MPCH, so the target
    # stays the generated object as the box shifts. Low-mass halos are never
    # truly isolated at the 1e12 floor, but many are locally dominant (in voids)
    # -- and those are exactly the ones a covering box would pack together.
    s = _sep_xy(i)
    nb = (s < ISO_MPCH) & (np.arange(len(s)) != i)
    return not np.any(masses[nb] > masses[i])

iso = np.array([i for i in range(len(halos)) if r200s[i] > 0 and isolated(i)])
logM = np.log10(masses[iso])
# a spread across mass: a handful per 0.5-dex bin, groups included
sel = []
for lo in np.arange(12.0, 14.5, 0.5):
    inb = iso[(logM >= lo) & (logM < lo + 0.5)]
    if len(inb):
        order = inb[np.argsort(masses[inb])]
        sel += list(order[:: max(1, len(order) // 4)][:4])
sel = np.array(sorted(set(sel), key=lambda i: masses[i]))
print(f"{len(iso)} isolated halos; probing {len(sel)}: "
      f"logM = {np.round(np.log10(masses[sel]), 2)}")
""")

md("""
## 2. The off-center probe

For each halo we build cutouts centered at a ladder of **box-center offsets**
$o$ (in +x, −x, +y, −y). Offset $o$ Mpc/h $=$ `round(o / MPC_PER_PX)` px in
*both* the full-box and patch grids (same pixel scale). At offset $o$ the halo
sits at patch pixel $(64 - o_{px})$ instead of the center, so we compare a
**registered** aperture window (radius $= 4\,R_{200c}$, capped so it stays in
the box) between each offset and the $o=0$ centered baseline.

We cap each halo's offset ladder at the largest $o$ whose $4\,R_{200c}$ aperture
still fits inside the 128-px box — that geometric limit is itself part of the
covering constraint.
""")

code(r"""
def cutout_at(cx_px, cy_px):
    cond, ls = extract_multiscale(dmo_fullbox, cx_px % NPIX, cy_px % NPIX,
                                  target_res=PATCH_PIX, mpc_per_pix=MPC_PER_PX)
    return {"condition": cond, "large_scale": ls}

DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}

def build_jobs(idx):
    # Per halo: offset ladder (px) capped so the 4xR200 aperture fits the box.
    cx = int(round(centers[idx, 0] / MPC_PER_PX))
    cy = int(round(centers[idx, 1] / MPC_PER_PX))
    ap = int(np.ceil(4.0 * r200s[idx] / MPC_PER_PX))          # aperture radius, px
    ap = int(np.clip(ap, 4, PATCH_PIX // 2 - 2))
    off_cap = min(MAX_OFF, (PATCH_PIX // 2 - 1 - ap) * MPC_PER_PX)
    offs = np.round(np.arange(0.0, off_cap + 1e-6, 0.5) / MPC_PER_PX).astype(int)
    offs = np.unique(offs)
    jobs = [(0.0, "+x", 0, cx, cy)]                            # o=0 baseline
    for opx in offs[1:]:
        for name, (ux, uy) in DIRS.items():
            jobs.append((opx * MPC_PER_PX, name, opx, cx + ux * opx, cy + uy * opx))
    return ap, jobs

# assemble one big batch of cutouts across all halos+offsets, generate once
plan, cutouts = [], []
for idx in sel:
    ap, jobs = build_jobs(idx)
    for o_mpch, name, opx, jx, jy in jobs:
        plan.append((idx, ap, o_mpch, name, opx))
        cutouts.append(cutout_at(jx, jy))
print(f"{len(cutouts)} generations across {len(sel)} halos "
      f"(naive one-per-halo would miss the whole point)")

cache_f = CACHE / "gen_sim0.npz"
if cache_f.exists():
    gen = np.load(cache_f)["gen"]
    print("loaded cached generations", gen.shape)
else:
    gen = model.generate(cutouts, sim_params, n_steps=N_STEPS,
                         batch_size=BATCH, use_amp=True)          # (M,7,128,128)
    np.savez_compressed(cache_f, gen=gen.astype(np.float32))
    print("generated + cached", gen.shape)
""")

md("""
### 2a. Metrics vs. the centered baseline

For each (halo, offset) we crop the **registered** aperture window and compare it
to the same halo's $o=0$ window:

- **$\\Delta\\log_{10}$ integrated mass** in the aperture, per channel
  (DM / Gas / Stars) and per thermo field — the science-level quantities.
- **aperture pixel correlation** (Pearson $r$ in $\\log$) — morphology fidelity.

Curves are averaged over the four directions (band = min/max across directions).
""")

code(r"""
def window(img, cy_px, cx_px, ap):
    y0, y1 = cy_px - ap, cy_px + ap + 1
    x0, x1 = cx_px - ap, cx_px + ap + 1
    return img[y0:y1, x0:x1]

def logsum(w):
    return np.log10(max(w[w > 0].sum(), 1e-30))

def logr(a, b):
    ma = (a > 0) & (b > 0)
    if ma.sum() < 8:
        return np.nan
    return np.corrcoef(np.log10(a[ma]), np.log10(b[ma]))[0, 1]

# index generations by halo -> {offset -> {dir -> row}}
from collections import defaultdict
byhalo = defaultdict(lambda: defaultdict(dict))
apof = {}
for row, (idx, ap, o, name, opx) in enumerate(plan):
    byhalo[idx][round(o, 4)][name] = row
    apof[idx] = ap

records = []
for idx in sel:
    ap = apof[idx]
    base_row = byhalo[idx][0.0]["+x"]
    base = gen[base_row]                                   # halo at (64,64)
    base_ap = {c: window(base[c], 64, 64, ap) for c in range(7)}
    base_sum = {c: logsum(base_ap[c]) for c in range(7)}
    for o in sorted(byhalo[idx]):
        for name, (ux, uy) in DIRS.items():
            if name not in byhalo[idx][o]:
                continue
            r = byhalo[idx][o][name]
            opx = int(round(o / MPC_PER_PX))
            hy, hx = 64 - uy * opx, 64 - ux * opx          # halo location in patch
            g = gen[r]
            rec = {"idx": int(idx), "logM": float(np.log10(masses[idx])),
                   "off": float(o), "dir": name}
            for c in range(7):
                w = window(g[c], hy, hx, ap)
                rec[f"dsum_{c}"] = logsum(w) - base_sum[c]
                rec[f"corr_{c}"] = logr(w, base_ap[c])
            records.append(rec)

import pandas as pd
R = pd.DataFrame(records)
R["mbin"] = np.where(R["logM"] >= 13.0, "group (>=1e13)", "low (<1e13)")
print(R.groupby(["mbin", "off"]).size().head(12))
""")

md("### 2b. Degradation curves — this is $r_{\\max}$")

code(r"""
CH = {"DM_hydro": 0, "Gas": 1, "Stars": 2, "compton_y": 3, "P_e": 6}
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

# top row: |Delta log integrated mass| vs offset, per channel, split by mass bin
for ax, (lbl, c) in zip(axes[0], list(CH.items())[:3]):
    for mb, gm in R.groupby("mbin"):
        stat = gm.groupby("off")[f"dsum_{c}"].agg(
            lambda s: np.nanmedian(np.abs(s)))
        lo = gm.groupby("off")[f"dsum_{c}"].agg(lambda s: np.nanmin(np.abs(s)))
        hi = gm.groupby("off")[f"dsum_{c}"].agg(lambda s: np.nanmax(np.abs(s)))
        ax.plot(stat.index, stat.values, "o-", label=mb)
        ax.fill_between(stat.index, lo.values, hi.values, alpha=0.15)
    ax.axhline(0.05, color="r", ls="--", lw=0.8)   # 0.05 dex ~ 12% tolerance
    ax.set_title(f"|Δlog Σ {lbl}|"); ax.set_ylabel("dex vs centered")
    ax.legend(fontsize=8)

# bottom row: aperture pixel correlation vs offset
for ax, (lbl, c) in zip(axes[1], list(CH.items())[:3]):
    for mb, gm in R.groupby("mbin"):
        stat = gm.groupby("off")[f"corr_{c}"].median()
        ax.plot(stat.index, stat.values, "o-", label=mb)
    ax.axhline(0.98, color="r", ls="--", lw=0.8)
    ax.set_title(f"aperture corr — {lbl}")
    ax.set_xlabel("box-center offset [Mpc/h]"); ax.set_ylabel("Pearson r (log)")
fig.suptitle("Off-center degradation: where does the painted aperture drift?", y=1.01)
plt.tight_layout(); plt.show()

# headline r_max: largest offset holding median |Δlog Σ| < tol. Use the robust
# mass channels DM(0)+Gas(1); Stars(2) is occupancy-noisy and thermo(3..6) are
# the sharpest channels -- both break earlier, reported separately.
tol = 0.10
for mb, gm in R.groupby("mbin"):
    def rmax_over(chs):
        ok = [o for o in sorted(gm["off"].unique())
              if o > 0 and max(np.nanmedian(np.abs(gm[gm["off"] == o][f"dsum_{c}"]))
                               for c in chs) < tol]
        return max(ok) if ok else 0.0
    print(f"{mb:>18s}: r_max(<{tol} dex) DM+Gas = {rmax_over([0, 1]):.2f} | "
          f"+Stars = {rmax_over([0, 1, 2]):.2f} | "
          f"thermo = {rmax_over([3, 4, 5, 6]):.2f} Mpc/h")
""")

md("""
## 3. Visual — one group halo painted off-center

DMO condition + generated Gas and Compton-$y$ for a single group at increasing
offset. If the model is valid off-center, the **aperture** (red circle) content
should look the same regardless of where the halo sits in the box.
""")

code(r"""
gidx = sel[np.argmax(masses[sel])]                         # most massive probed
ap = apof[gidx]
offs = sorted(byhalo[gidx])[:5]
fig, axes = plt.subplots(3, len(offs), figsize=(3 * len(offs), 9))
rows = [("cond DMO", None), ("gen Gas", 1), ("gen compton_y", 3)]
for j, o in enumerate(offs):
    opx = int(round(o / MPC_PER_PX))
    r = byhalo[gidx][o]["+x"]
    hy, hx = 64, 64 - opx
    imgs = [cutouts[r]["condition"], gen[r][1], gen[r][3]]
    for i, (title, _) in enumerate(rows):
        img = imgs[i]
        a = np.log10(np.clip(img, img[img > 0].min() if (img > 0).any() else 1e-30, None))
        axes[i, j].imshow(a, cmap="magma")
        axes[i, j].add_patch(plt.Circle((hx, hy), ap, fill=False, color="r", lw=1.2))
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        if j == 0: axes[i, j].set_ylabel(title)
    axes[0, j].set_title(f"offset {o:.1f} Mpc/h")
fig.suptitle(f"Group logM={np.log10(masses[gidx]):.2f} painted off-center "
             f"(red = 4R200 aperture)", y=1.0)
plt.tight_layout(); plt.show()
""")

md("""
## 4. Payoff — turn $r_{\\max}$ into GPU-call savings on the real halo field

Given a covering radius $r_{\\max}$, a simple **grid-snap** covering places one
6.25 Mpc/h box per occupied cell of a grid with side $\\approx 2\\,r_{\\max}$
(minus an aperture margin so every halo's paste region stays inside its box).
GPU calls $=$ number of occupied cells, vs. the naive $N_{\\text{halos}}$.

We sweep the mass threshold and report the fold reduction on **sim_0**'s actual
halo positions. Set `RMAX` from the number printed in §2b.
""")

code(r"""
RMAX = 1.5     # <-- set from section 2b (Mpc/h). Conservative default.

def grid_snap_calls(pos_xy, cell):
    # Occupied-cell count for a periodic grid of the given cell size (Mpc/h).
    n = max(1, int(round(BOX_MPCH / cell)))
    ij = np.floor((pos_xy % BOX_MPCH) / (BOX_MPCH / n)).astype(int) % n
    return len({(int(a), int(b)) for a, b in ij})

cell = 2.0 * RMAX
thr = np.logspace(12, 14, 9)
naive, covered = [], []
for t in thr:
    m = masses >= t
    naive.append(int(m.sum()))
    covered.append(grid_snap_calls(centers[m], cell) if m.sum() else 0)
naive, covered = np.array(naive), np.array(covered)
fold = naive / np.maximum(covered, 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].loglog(thr, naive, "o-", label="naive (1 call / halo)")
ax[0].loglog(thr, covered, "s-", label=f"grid-snap (cell={cell:.1f} Mpc/h)")
ax[0].set(xlabel=r"halo mass threshold $[M_\odot/h]$", ylabel="GPU generations",
          title="Generations to baryonify all halos above threshold")
ax[0].legend()
ax[1].semilogx(thr, fold, "k-o")
ax[1].set(xlabel=r"halo mass threshold $[M_\odot/h]$",
          ylabel="fold reduction", title=f"Savings at r_max={RMAX} Mpc/h")
ax[1].axhline(1, color="r", ls="--")
plt.tight_layout(); plt.show()
print(f"At 1e12 floor: {naive[0]} -> {covered[0]} generations "
      f"({fold[0]:.1f}x fewer GPU calls on sim_0)")
""")

md("""
## 5. Interpretation

- **§2b gives $r_{\\max}$** — the offset where the painted aperture starts to
  drift from the centered baseline. That is the covering radius: two halos can
  share a box iff both sit within $r_{\\max}$ of a common center *and* both
  apertures fit in the box.
- **If $r_{\\max}$ is large** (approaches the geometric aperture limit), the
  design is "tile the box on a fixed grid" — GPU calls decouple from $N$.
- **If $r_{\\max}$ is small**, the covering is constrained; §4 quantifies what
  that buys on the real field.
- **Aperture geometry sets an independent ceiling.** A group's $4\\,R_{200c}$
  aperture already fills most of the 6.25 Mpc/h box, so there is little room to
  pack a second aperture regardless of $r_{\\max}$ — covering pays off precisely
  in the **low-mass, small-aperture, high-density** regime this branch targets.
- **Caveats.** (i) Isolated halos isolate the pure off-center effect; a crowded
  box adds neighbour-contamination on top — the natural follow-up probe. (ii)
  Thermo channels carry per-halo truth; here we test *self-consistency* vs the
  centered prediction, which is what the covering optimization must preserve.
  (iii) `large_scale` context re-centers on the box, not the neighbour — folded
  into the measured degradation.

**Next step if $r_{\\max}$ is usable:** implement grid-snap covering in the paint
path (place one box per occupied cell at its halo centroid, paste every halo's
aperture from that single generation) and validate the composite P(k) against
the one-call-per-halo mosaic.
""")

nb["cells"] = C
out = "examples/offcenter_covering.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(C), "cells")
