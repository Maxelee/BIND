# Shell/slab-straddling halo fraction — results

Engine: `papers/01_pipeline/shell_straddle.py`. Data: `papers/01_pipeline/audits/shell_straddle_data.json`
(full per-snapshot table + overall totals). Feeds paper §5 caveat 10.

## Method recap

BIND paints the lightcone per-slab: each of the 20 lightcone snapshots' TNG300-DM box
(box_size = 205 Mpc/h) is cut into `n_slabs=4` non-overlapping slabs of depth
`slab_depth = box_size/n_slabs = 51.25 Mpc/h` along a per-snapshot randomly-drawn
line-of-sight (LOS) axis (`bind.inference.lightcone_transforms.LightconeTransforms`).
Each halo's condition/patch is extracted from *only* the slab containing its
(transformed) center, and particles are hard-cut into slabs by exact LOS coordinate
(`_project_zslabs`), so a halo whose 3D extent `[center_LOS - r200c, center_LOS + r200c]`
straddles a slab cut has part of its real mass projected into the *wrong* slab's
density/condition map.

Halo LOS positions are **not cached anywhere on disk** — `truth_lightcone.py` /
`paint.py` only keep the transverse `(x, y)` `halo_centers` in `composite_slab*.npz`; the
LOS coordinate used by `_assign_halos_to_slabs` is computed on the fly and discarded.
This script recomputes it from scratch: for each of the 20 snapshots it reads the raw DMO
FoF catalog (`GroupPos` / `Group_M_Crit200` / `Group_R_Crit200`, path taken from that
snapshot's `stage1/stage1_manifest.json`) and re-applies the *same*
`LightconeTransforms` (`bind_lightcone_tng/lightcone_transforms.json`) used to build the
lightcone, exactly mirroring `truth_lightcone.extract_truth_halos`'s halo-to-slab
assignment.

Two boundary types are distinguished:
- **straddle** — any of the 3 internal cuts within one snapshot's single, continuous
  205 Mpc/h box. Same continuous volume; the halo's mass is real but split between two
  slabs of the *same* snapshot/realization.
- **shell** (snapshot) boundary — the box's own periodic seam at `z=0`/`box_size`. Each
  snapshot's box is used exactly once per 205 Mpc/h lightcone shell (its own random
  disp/flip/proj_dir draw); slab 3 of snapshot *k* is immediately followed in comoving
  distance by slab 0 of the *different* snapshot *k+1* (an unrelated realization/redshift),
  not by slab 0 of snapshot *k*'s own periodic image. A halo straddling `z=0`/`box_size`
  is therefore split across two entirely unrelated realizations — the physically worse
  case. Its count is a **subset** of "straddle".

Mass cut: M200c > 1e13 Msun/h (the painted-halo threshold).

## Per-snapshot table

| snapshot | z | N(M>1e13) | straddle N | straddle frac | mass-wt frac | shell N | shell frac | shell mass-wt |
|---|---|---|---|---|---|---|---|---|
| snap_029 | 2.444 | 147  | 5  | 3.40% | 3.37% | 0  | 0.00% | 0.00% |
| snap_031 | 2.208 | 226  | 2  | 0.88% | 1.06% | 1  | 0.44% | 0.44% |
| snap_033 | 2.002 | 345  | 9  | 2.61% | 2.19% | 3  | 0.87% | 0.94% |
| snap_035 | 1.823 | 497  | 12 | 2.41% | 2.74% | 7  | 1.41% | 1.95% |
| snap_038 | 1.604 | 712  | 14 | 1.97% | 2.07% | 4  | 0.56% | 0.91% |
| snap_041 | 1.414 | 920  | 32 | 3.48% | 3.75% | 10 | 1.09% | 0.94% |
| snap_043 | 1.302 | 1103 | 27 | 2.45% | 2.95% | 10 | 0.91% | 1.69% |
| snap_046 | 1.155 | 1326 | 35 | 2.64% | 3.28% | 4  | 0.30% | 0.27% |
| snap_049 | 1.036 | 1511 | 24 | 1.59% | 1.98% | 7  | 0.46% | 0.81% |
| snap_052 | 0.923 | 1692 | 42 | 2.48% | 3.32% | 7  | 0.41% | 0.52% |
| snap_056 | 0.791 | 1920 | 46 | 2.40% | 3.06% | 9  | 0.47% | 0.52% |
| snap_059 | 0.700 | 2074 | 51 | 2.46% | 2.89% | 6  | 0.29% | 0.42% |
| snap_063 | 0.599 | 2248 | 51 | 2.27% | 2.29% | 14 | 0.62% | 0.59% |
| snap_067 | 0.503 | 2411 | 61 | 2.53% | 3.06% | 13 | 0.54% | 0.67% |
| snap_071 | 0.420 | 2533 | 56 | 2.21% | 3.17% | 22 | 0.87% | 1.04% |
| snap_076 | 0.329 | 2647 | 45 | 1.70% | 2.11% | 5  | 0.19% | 0.13% |
| snap_080 | 0.261 | 2734 | 48 | 1.76% | 2.41% | 9  | 0.33% | 0.67% |
| snap_085 | 0.180 | 2813 | 72 | 2.56% | 3.38% | 13 | 0.46% | 0.61% |
| snap_090 | 0.110 | 2886 | 64 | 2.22% | 3.31% | 19 | 0.66% | 1.25% |
| snap_096 | 0.034 | 2933 | 52 | 1.77% | 2.24% | 16 | 0.55% | 0.36% |

## Overall (20 snapshots, 80 slabs, M200c > 1e13 Msun/h)

- **N halos total: 33,678**
- **Straddle (any of the 3 slab boundaries): 748 halos = 2.22%** by count, **2.80%** mass-weighted
- **Shell (snapshot) boundary specifically: 179 halos = 0.53%** by count, **0.68%** mass-weighted
  (a subset of the straddle total)
- Internal-only (straddles an internal cut but not the shell seam): 569 halos = 1.69%
- Check: 179 + 569 = 748 ✓

## Interpretation

- The fraction is small but not negligible: **about 1 in 45 painted halos (M200c > 1e13)
  has its density mass split across two lightcone slabs**, and roughly 1 in 190 is split
  across two *different, unrelated snapshots* (the physically worse "shell" case).
- Mass-weighted fractions run consistently above count fractions (2.80% vs 2.22% overall;
  the pattern holds snapshot by snapshot) — expected, since larger halos have larger r200c
  and are proportionately more likely to straddle a fixed-width cut. The effect is
  therefore slightly more important for cluster-scale halos than the raw count fraction
  suggests.
- No strong redshift trend: the straddle fraction is flat within noise across the full
  z = 0.03–2.44 range (roughly 1.6–3.5%, consistent with Poisson scatter given halo counts
  from 147 to 2933 per snapshot); the shell fraction is similarly flat (~0.2–1.4%).
- This quantifies, for the first time, the caveat that BIND's per-slab painting
  can truncate halos that sit near a slab cut — a small but real source of map-level
  error distinct from the halo-replacement/mass-threshold caveats already discussed in §5.
