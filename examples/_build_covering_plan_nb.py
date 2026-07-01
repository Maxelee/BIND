"""Build examples/covering_plan.ipynb.

Greedy geometric set-cover planner for multi-halo painting (branch:
low_mass_extrapolation). BIND paints a 6.25 Mpc/h box per GPU call but only
pastes a ~4xR200 aperture back, so a box centered on one halo already paints any
neighbour that (a) sits within the off-center validity radius r_max and (b) has
its aperture inside the box. This notebook plans the minimal set of generation
centers by processing halos in DESCENDING mass order -- which forces the
aperture-constrained massive halos to be perfectly centered and lets small,
flexible halos be swept up for free -- and emits the pre-computed "map": the
generation centers plus, for every halo, which generation paints it and at what
offset. That map decouples cheap CPU planning from expensive batched GPU
generation.

CPU-only (no model / no GPU); reads only the halo catalog + DMO map already on
ceph. r_max is a free parameter here, set from examples/offcenter_covering.ipynb.

Run:  python examples/_build_covering_plan_nb.py
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
# Covering plan: baryonify every halo with the fewest GPU generations

BIND paints a **6.25 Mpc/h box** per GPU call but only pastes back a circular
**$4\,R_{200c}$ aperture**. So a box centered on one halo *already paints* any
neighbour that

1. sits within the **off-center validity radius** $r_{\max}$ of the center
   (measured in [`offcenter_covering.ipynb`](offcenter_covering.ipynb)), **and**
2. has its $4\,R_{200c}$ aperture **inside** the box (no edge spill).

"Baryonify every halo above threshold" is then **greedy geometric set-cover**.
We process halos in **descending mass order**: massive halos have huge apertures
that fill the box, so they *cannot* be off-center passengers — they are forced to
be box centers, perfectly centered (best fidelity). Small halos are flexible and
get swept up for free.

The output is the **pre-computed map**: the list of generation centers, and for
every halo `(which generation paints it, its offset in that box)`. That map is
what decouples cheap CPU planning from expensive batched GPU generation — plan
all centers, build all cutouts, generate in batches, paste each halo's aperture
from *its assigned* generation.

This notebook is **CPU-only** — pure geometry on the real `sim_0` catalog.
""")

md("## 0. Config + load the halo catalog (no model needed)")

code(r"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from bind.inference.artifacts import load_full_maps, load_halo_catalog

plt.rcParams.update({"figure.dpi": 110, "font.size": 11})

SIM_L      = 50.0        # full sim box [Mpc/h] (periodic)
GEN_BOX    = 6.25        # side of one generation box [Mpc/h]
NPIX       = 1024
R200_FAC   = 4.0         # paste aperture radius = R200_FAC * R200c (pipeline default)
MASS_MIN   = 1e12

# r_max: how far off-center BIND stays faithful, measured in
# offcenter_covering.ipynb (CV sim_0, <0.1 dex). The catch: the packable
# population IS the low-mass passengers, and low-mass DM/Gas only holds to
# ~0.5 Mpc/h (groups ~1.0); low-mass thermo breaks even at 0.5. So r_max ~0.5-1.0
# for mass-only painting (~1.1-1.35x here) and thermo painting barely packs at
# all. The §4 sweep is the honest picture; this default is the mass-only upper end.
R_MAX      = 1.0

OUTPUT_ROOT = Path("/mnt/home/mlee1/ceph/fm_lowmass")
sim_dir = OUTPUT_ROOT / "CV" / "sim_0" / "snap_090"
dmo_fullbox, _ = load_full_maps(sim_dir / "full_maps.npz")
halos, halo_masses, halo_r200s, halo_positions = load_halo_catalog(
    sim_dir / "mass_threshold_1p000e12" / "halo_catalog.npz")
centers = np.asarray([h["halo_center"] for h in halos], dtype=np.float64)  # (N,2) Mpc/h
masses  = np.asarray([h["halo_mass"]   for h in halos], dtype=np.float64)
r200s   = np.asarray([h.get("r200", 0.0) for h in halos], dtype=np.float64)  # Mpc/h
print(f"sim_0: {len(halos)} halos, logM in "
      f"[{np.log10(masses.min()):.2f}, {np.log10(masses.max()):.2f}], "
      f"R200c median/max = {np.median(r200s[r200s>0]):.3f}/{r200s.max():.3f} Mpc/h")
""")

md("""
## 1. The greedy covering planner

Process uncovered halos in descending mass order. Each becomes a box center; a
still-uncovered halo joins that box iff it is within `r_max` **and** its aperture
fits inside the box (`|dx|, |dy| < GEN_BOX/2 - R200_FAC*R200c`). The center halo
always belongs to its own box (its aperture may clip the edge, exactly as the
current single-halo pipeline already tolerates).
""")

code(r"""
def plan_covering(centers, masses, r200s, *, r_max, gen_box=GEN_BOX,
                  r200_factor=R200_FAC, sim_l=SIM_L, mass_min=MASS_MIN):
    # Greedy set-cover. Returns (gen_centers (G,2), assign, covered).
    # assign[halo_id] = (gen_idx, dx, dy): the generation that paints the halo
    # and its periodic offset (Mpc/h) from that generation's center.
    half = gen_box / 2
    sel  = np.where(masses >= mass_min)[0]
    order = sel[np.argsort(-masses[sel])]              # mass-descending
    covered = np.zeros(len(masses), dtype=bool)
    gen_centers, assign = [], {}
    for i in order:
        if covered[i]:
            continue
        gi = len(gen_centers)
        c  = centers[i].copy()
        gen_centers.append(c)
        cand = order[~covered[order]]                  # uncovered, mass-desc
        d = centers[cand] - c
        d -= sim_l * np.round(d / sim_l)               # periodic (full box)
        dx, dy = d[:, 0], d[:, 1]
        margin = r200_factor * r200s[cand]
        fits = (np.abs(dx) < half - margin) & (np.abs(dy) < half - margin)
        ok   = ((dx**2 + dy**2) < r_max**2) & fits
        ok  |= (cand == i)                             # center owns its box
        for j, dxj, dyj in zip(cand[ok], dx[ok], dy[ok]):
            covered[j] = True
            assign[int(j)] = (gi, float(dxj), float(dyj))
    return np.asarray(gen_centers), assign, covered

gen_centers, assign, covered = plan_covering(centers, masses, r200s, r_max=R_MAX)

n_above = int((masses >= MASS_MIN).sum())
n_gen   = len(gen_centers)
assert covered[masses >= MASS_MIN].all(), "some halo above threshold left uncovered"
per_box = np.bincount([g for g, _, _ in assign.values()], minlength=n_gen)
print(f"r_max = {R_MAX} Mpc/h")
print(f"  halos to paint : {n_above}")
print(f"  GPU generations: {n_gen}   ->  {n_above / n_gen:.2f}x fewer calls")
print(f"  halos/box: mean {per_box.mean():.2f}, max {per_box.max()}, "
      f"solo boxes {int((per_box == 1).sum())}")
""")

md("""
## 2. The map, beforehand

The plan on top of the DMO field: box centers (white ×), the boxes they generate
(cyan squares), and halos colored by how many share their box — the "map before
we ever touch the GPU". A zoom panel makes the packing visible.
""")

code(r"""
box_of = np.full(len(masses), -1)
off_of = np.zeros((len(masses), 2))
for j, (g, dx, dy) in assign.items():
    box_of[j] = g; off_of[j] = (dx, dy)
mates = per_box[np.clip(box_of, 0, None)]              # #halos in each halo's box
above = masses >= MASS_MIN

def draw(ax, xlim, ylim, title):
    ax.imshow(np.log10(np.clip(dmo_fullbox, 1e-30, None)), origin="lower",
              extent=[0, SIM_L, 0, SIM_L], cmap="Greys", alpha=0.65)
    for c in gen_centers:
        ax.add_patch(Rectangle((c[0]-GEN_BOX/2, c[1]-GEN_BOX/2), GEN_BOX, GEN_BOX,
                     fill=False, ec="cyan", lw=0.6, alpha=0.7))
    sc = ax.scatter(centers[above, 0], centers[above, 1],
                    s=3 + 6*(np.log10(masses[above])-12),
                    c=mates[above], cmap="viridis", vmin=1, vmax=max(2, per_box.max()))
    ax.scatter(gen_centers[:, 0], gen_centers[:, 1], marker="x", c="white",
               s=14, lw=0.8)
    ax.set(xlim=xlim, ylim=ylim, title=title, xlabel="x [Mpc/h]", ylabel="y [Mpc/h]")
    return sc

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
sc = draw(axes[0], (0, SIM_L), (0, SIM_L), f"Full box — {n_gen} generations for {n_above} halos")
z = 12
draw(axes[1], (0, z), (0, z), f"Zoom {z}x{z} Mpc/h (× = center, cyan = 6.25 box)")
fig.colorbar(sc, ax=axes, label="halos sharing the box", shrink=0.8)
plt.show()
""")

md("""
## 3. Who gets packed, and who flies solo

Covering pays off in the **low-mass, small-aperture** regime: a group's aperture
fills the box so it can host few (or no) passengers, while low-mass halos pack
tightly. This shows the fraction of halos that are passengers (painted for free)
vs. box centers, as a function of mass.
""")

code(r"""
is_center = np.zeros(len(masses), bool)
# a halo is a "center" if its own position seeded a generation
cen_ids = {int(np.argmin(np.hypot(centers[:,0]-c[0], centers[:,1]-c[1])))
           for c in gen_centers}
is_center[list(cen_ids)] = True

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
mbins = np.arange(12.0, 14.6, 0.25)
mc = 0.5*(mbins[1:]+mbins[:-1])
lm = np.log10(masses[above])
frac_solo, frac_pass = [], []
for lo, hi in zip(mbins[:-1], mbins[1:]):
    m = (lm >= lo) & (lm < hi)
    if m.sum() == 0:
        frac_solo.append(np.nan); frac_pass.append(np.nan); continue
    ids = np.where(above)[0][m]
    frac_pass.append(np.mean(~is_center[ids]))          # painted for free
    frac_solo.append(np.mean(per_box[box_of[ids]] == 1))
ax[0].plot(mc, frac_pass, "o-")
ax[0].set(xlabel=r"$\log_{10}M_{200c}$", ylabel="fraction that are passengers",
          title="Painted for free (not a box center)", ylim=(-0.05, 1.05))
ax[1].hist(per_box, bins=np.arange(0.5, per_box.max()+1.5), color="#2c3e50")
ax[1].set(xlabel="halos per generation", ylabel="# generations",
          title=f"Packing (mean {per_box.mean():.2f} halos/box)")
plt.tight_layout(); plt.show()
""")

md("""
## 4. Savings vs. $r_{\\max}$ and mass threshold

How the GPU-call count depends on the off-center radius (from the probe) and the
mass floor. The adaptive greedy is compared against a fixed grid-snap baseline
(one box per occupied cell of side $2r_{\\max}$) — centering on real halos does
strictly better.
""")

code(r"""
def grid_snap_calls(pos_xy, cell):
    n = max(1, int(round(SIM_L / cell)))
    ij = np.floor((pos_xy % SIM_L) / (SIM_L / n)).astype(int) % n
    return len({(int(a), int(b)) for a, b in ij})

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) vs r_max at the 1e12 floor
rmaxes = np.array([0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
greedy, grid = [], []
for rm in rmaxes:
    gc, _, cov = plan_covering(centers, masses, r200s, r_max=rm)
    greedy.append(len(gc))
    grid.append(grid_snap_calls(centers[masses >= MASS_MIN], 2*rm))
ax[0].plot(rmaxes, greedy, "o-", label="greedy (centered on halos)")
ax[0].plot(rmaxes, grid, "s--", label=f"grid-snap (cell 2·r_max)")
ax[0].axhline(n_above, color="r", ls=":", label=f"naive = {n_above}")
ax[0].set(xlabel=r"$r_{\max}$ [Mpc/h]", ylabel="GPU generations",
          title="Cost vs off-center radius (1e12 floor)")
ax[0].legend(fontsize=8)

# (b) vs mass threshold at the current R_MAX
thr = np.logspace(12, 14, 9)
naive_n, greedy_n = [], []
for t in thr:
    naive_n.append(int((masses >= t).sum()))
    gc, _, _ = plan_covering(centers, masses, r200s, r_max=R_MAX, mass_min=t)
    greedy_n.append(len(gc))
naive_n, greedy_n = np.array(naive_n), np.array(greedy_n)
ax[1].loglog(thr, naive_n, "o-", label="naive (1/halo)")
ax[1].loglog(thr, greedy_n, "s-", label=f"greedy (r_max={R_MAX})")
ax[1].set(xlabel=r"mass threshold $[M_\odot/h]$", ylabel="GPU generations",
          title="Cost vs mass floor")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print("fold reduction at 1e12 floor vs r_max:",
      dict(zip(rmaxes, np.round(n_above/np.array(greedy), 2))))
""")

md("""
## 5. Export the map — the hand-off to generation

The plan as arrays, ready to drive batched generation: `gen_centers` are the box
centers to extract cutouts at; `halo_gen` / `halo_offset` tell the composite
which generation paints each halo and where it sits in that box (so the aperture
is pasted from the right patch at the right location).
""")

code(r"""
halo_gen    = box_of.copy()                 # (N,) generation index per halo (-1 if below floor)
halo_offset = off_of.copy()                 # (N,2) Mpc/h offset from its gen center
out = Path("covering_plan_sim0.npz")
np.savez(out, gen_centers=gen_centers, halo_gen=halo_gen, halo_offset=halo_offset,
         masses=masses, r200s=r200s, centers=centers,
         r_max=R_MAX, gen_box=GEN_BOX, r200_factor=R200_FAC, sim_l=SIM_L)
print(f"wrote {out}: {len(gen_centers)} generations, "
      f"{int((halo_gen>=0).sum())} halos assigned")

# next step: for gi, c in enumerate(gen_centers): cutout = extract_multiscale(dmo, c)
#            gen = model.generate(cutouts, params)
#            for each halo j: paste 4*R200 aperture of gen[halo_gen[j]] at halo_offset[j]
""")

md("""
## 6. Interpretation & next step

- The **map** (§5) is the whole payoff: planning is cheap CPU set-cover; the GPU
  only runs once per generation center, and the composite pastes each halo's
  aperture from its assigned box.
- Savings are set by $r_{\\max}$ (§4). The **honest input** is the probe in
  [`offcenter_covering.ipynb`](offcenter_covering.ipynb) — plug its measured
  $r_{\\max}$ into `R_MAX` above.
- The greedy beats grid-snap because it centers on real halos, which also gives
  the massive, aperture-constrained halos perfect centering for free.
- **Caveats.** (i) Passenger apertures must fit the box — so groups host few
  passengers; the win is at low mass. (ii) The probe used *isolated* halos; a
  packed box adds neighbour contamination, the natural validation before
  deploying. (iii) To wire into production: place one generation per
  `gen_centers` row and teach the composite to index `halo_gen`/`halo_offset`
  instead of one-generation-per-halo, then validate the composite $P(k)$ against
  the naive mosaic.
""")

nb["cells"] = C
out = "examples/covering_plan.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(C), "cells")
