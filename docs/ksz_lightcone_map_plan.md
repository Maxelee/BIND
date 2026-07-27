# P4b — Map-level DESI×ACT confrontation on the SB35 lightcones — execution plan

Created 2026-07-24 (branch `analysis/ksz-desi-act-v2`). Successor to
`docs/ksz_desi_act_plan.md` (P4, per-halo — complete through D10). This plan
repeats the headline P4 result — *which TNG feedback regimes are consistent
with the DESI×ACT kSZ data* — but on the **ray-traced lightcone maps**
(tau/y/kappa, 50 realizations × 25 deg² per Sobol node) instead of individual
painted halo patches. New physics gained: real line-of-sight projection,
2-halo term, correlated LSS "noise" in the stack, and a realization-based
covariance — i.e. the analysis is performed the way the observers perform it.

Written to be **enacted by Sonnet/Haiku agents at minimal token cost**: §1 is
the distilled geometry knowledge (already verified against the lux source, the
BIND source, and the on-disk products — do NOT re-derive it); §3 is the phase
list with per-phase owner, exact inputs/outputs, validation gates, and figures.
Every phase ends with a PNG figure + a small `verdict` JSON; subsequent phases
read only the verdict. A companion notebook (§5) assembles all validation and
science figures.

---

## 0. Goal and relation to the per-halo result

The per-halo capstone (paper2 §6/§D): stacked CAP f̃_gas from painted patches
shows 49/256 SB35 nodes consistent with Ried Guachalla+25 BGS kSZ (fiducial
TNG ~1.4–1.8× too gas-rich); the tSZ y-CAP leg (Liu+2025) was computed on
individual halos and is NOT trusted (shape/aperture systematics). Here:

1. Recover where every halo lands in every traced map (all 50 realizations).
2. Build mock DESI (BGS/LRG/ELG) catalogs on the lightcone.
3. Stack CAP filters on the traced tau/y/kappa maps at those positions.
4. Re-make the two money plots (f̃_gas vs data; y-CAP vs data) at map level,
   with a realization-based covariance and the same 256-node Sobol envelope.

---

## 1. Established facts — geometry, formats, paths (verified 2026-07-24)

**Do not re-verify these; cite this section.** Sources: `/mnt/home/mlee1/lux`
(`main.cpp`, `raytracing.cpp`), `src/bind/inference/{lightcone_transforms,
paint,lensplane}.py`, `src/bind/cli/paint_{project,lensplane,tauplane,yplane}.py`,
on-disk npz/json inspection.

### 1.1 Paths

- `FID = /mnt/home/mlee1/ceph/bind_lightcone_tng` — fiducial lightcone.
  Retains `lensplanes/` (incl. binary `config.dat`), `rt_output/run001..NNN/`
  (102 dirs; the first 50 correspond to the collected maps), per-snap
  `snap_SSS/{stage1/, composite_slab*.npz, summary.json}`, top-level
  `{kappa,y,tau}_maps.npz`, `lightcone_transforms.json`, `lux_tau.ini`.
- `RUNS = /mnt/home/mlee1/ceph/bind_sb35/runs/run_0000..0255` — per Sobol node:
  `params.npy` (35,), `{kappa,y,tau}_maps.npz`, `Cl_*.npz`, stats npz, and
  `snap_SSS/composite_slab*.npz` (per-halo `generated_patches` (n,3,128,128),
  `thermo_patches` (n,4,128,128), `halo_centers` (n,2), `halo_masses`,
  `halo_r200`).
- `DESIGN = /mnt/home/mlee1/ceph/bind_sb35/design/` — `astro_params_sobol.npy`
  (256,35), `design.json` (param names/indices; cosmology fixed at TNG300).
- `PARQUET = /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet`
  — 8.6M rows × 55 cols, per-halo instance keyed (`run`,`snap`,`slab`,`idx`).
- `TNGDM = /mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output` —
  `groups_SSS/fof_subhalo_tab_SSS.*.hdf5` (75 files/snap). Header: BoxSize
  205000 ckpc/h. `/Group`: `GroupPos` (ckpc/h), `Group_M_Crit200` (1e10
  Msun/h), `Group_R_Crit200` (ckpc/h), `GroupVel`, `Group_M_Crit500`, … .
  Reader: `bind.inference.io_gadget.read_fof_catalog`. **Never recursively
  search this mount; address files by exact path.**
- `KS = /mnt/home/mlee1/ceph/bind_science/ksz_confront` — kSZ data + prior
  reductions. New products go under `KS/lightcone/{catalogs,shards,figs,verdicts}/`.
- Data vectors: `KS/desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm11.{00,25}.npz`
  (keys `th, ratio, yerr, cov_ksz`; Ried Guachalla+25 f̃_gas(θ)),
  `examples/figures_ksz2/tsz_liu2025_official.npz` (Liu+2025 y-CAP; regenerate
  via `_build_ksz_paper2_nb.py` §F if missing).

### 1.2 The halo → map-pixel chain (four deterministic steps)

**Step A — snapshot-level box transform** (per snapshot, shared by all runs
and all realizations). `LightconeTransforms.load(FID/lightcone_transforms.json)`
(seed 2020, 20 entries). `.apply(pos_mpch, snap_idx, 205.0)` = axis permutation
by `proj_dirs[snap_idx]` (`PROJ_DIR_AXES: 0:(1,2,0), 1:(2,0,1), 2:(0,1,2)` →
(transverse_x, transverse_y, LOS)), + `disp`, negate where `flip`, mod 205.
Snapshot order (snap_idx 0..19, low-z first):
`96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29`.
Known z anchors: 96→0.0337, 85→0.18, 67→0.50, 49→1.04, 46→1.16, 38→1.60,
29→2.44 (full 20-row table: read `scale_factor`/`redshift` from each
`FID/snap_SSS/stage1/stage1_manifest.json` — P0 emits it).

**Step B — slab / lens-plane assignment.** `slab = clip(floor(LOS/(205/4)),
0, 3)` (`bind.inference.paint._assign_halos_to_slabs`; slab thickness 51.25
Mpc/h — NB the manifest's `slab_depth: 50.0` is the patch-cutout depth, not
the slab thickness). 1-indexed lux plane `p = 4*snap_idx + slab + 1` ∈ [1,80].
Plane files `lenspot{p:02d}.dat` / `tauplane{p:02d}.dat` / `yplane{p:02d}.dat`.

**Step C — plane pixelization.** Native stage-1 grid 4198 px × 0.048828125
Mpc/h (≈205 Mpc/h). `paint_lensplane` **center-crops** to `LP_grid = 4096` px
= 200.0 Mpc/h, so plane frame origin sits at `x0 = ((4198-4096)/2) *
0.048828125 ≈ 2.4902` Mpc/h in transverse box coords. Plane pixel
`(i,j) = floor((xy - x0)/dLt)` with `dLt = 200/4096 = 0.048828125` Mpc/h (the
**native** stage-1 pixel size) — **verified in P1**: this is a literal numpy
index crop (`mass_map[51:51+4096, 51:51+4096]`, no resample), so the original
guess was numerically right. Do not confuse `dLt` here with `config.dat`'s
`Lt=205.0`, a *different*, coarser pixel size (`205/4096≈0.050049` Mpc/h) used
only in Step D's sky-angle conversion — using it for Step C instead fails
badly (P1 check 2: median offset ~12.8px vs ~0.7px against the raw
`tauplane*.dat`). Halos with transformed transverse coords outside the crop
fall off the plane (`1-(4096/4198)² ≈ 4.8%` of area, correcting the ~2.4%
estimated pre-P1).
`stage1_slab*.npz:halo_centers` are (transverse_x, transverse_y) Mpc/h in the
**native** (uncropped) frame — use them as ground truth to pin conventions.

**Step D — per-realization RT randomization + ray geometry** (lux; identical
for tau/y/lensing planes, one transform per snapshot shared by its 4 planes):

- Realization `r = 1..50` (`runNNN`, 1-indexed) used GSL-MT19937 seed
  `1992 + 7*r`; drawn per snapshot s in order (rot, disp_x, disp_y):
  `rot[s] ∈ {0,1,2,3}` (0/90/180/270°), `disp[s] ∈ [0,4096)²` integer pixels.
  **Do NOT re-implement the RNG.** The drawn values are recorded in the tail
  of `FID/rt_output/run{r:03d}/config.dat` (little-endian binary):
  `int32 Np(=80), int32 Ns(=20), f8 a[81], f8 chi[81], f8 chi_out[80],
  f8 Ll[20], f8 Lt[20], int32 rot[20], int32 disp[40] (x0,y0,x1,y1,…)`.
  Because every Sobol run used the same `RT_SEED=1992`, **these 50 records
  apply verbatim to all 256 runs' maps**.
- Scatter map, applied literally (note `4096-j`, NOT `4096-1-j` — reproduce
  the quirk): value at plane pixel (i,j) lands at
  `rot 0: (i, j); 1: (4096-j, i); 2: (4096-i, 4096-j); 3: (j, 4096-i)`,
  then `+ (disp_x, disp_y) mod 4096`.
- Ray geometry (Born, small-angle): FOV θ = 5° centered on the optical axis,
  `dθ = θ/1024` (0.29297′/px). The axis pierces the plane at `0.5*Lt`;
  a randomized plane position (X,Y) = ((I+0.5)·dLt, (J+0.5)·dLt) maps to
  `β = (X − 0.5*Lt)/chi[p]` and map pixel `i_map = β_x/dθ + 512` (same for j).
  The plane tiles periodically across the FOV (`W = chi·θ` vs `Lt=200` Mpc/h):
  a halo appears at every periodic image `X + n·Lt` that lands in [0,1024)² —
  keep **all copies**. Use `chi[p]` (plane center) and `Lt[s]`, both read from
  `config.dat`. Map arrays are flat `j + 1024*i` (i slow); the empirical
  axis-order/transpose check is part of the P1 gate.
- Cumulative maps: npz key `tau`/`y`/`kappa`, shape (50, 5, 1024, 1024) f4,
  axis 1 = source planes `{26,45,59,70,78}` → `source_redshifts ≈
  {0.5, 1.0, 1.5, 2.0, 2.44}` (source χ = `chi_out[m-1]`). tau/y at source
  index k include only planes p ≤ that source — default science slice:
  **index 4 (z_s=2.44, full LOS)** for tau/y; kappa mass-anchor at the source
  plane just behind the sample (BGS → z_s=1.0). Realization axis 0 is in
  `run001..run050` order — aligned with the config.dat records.
- Caveat (quantified in P1): lux samples tau/y along the *deflected* ray; the
  Born prediction ignores accumulated deflection (≲1 px low z, potentially
  ~1 px at z≳1). Measured empirically as stack-centroid offset/width.

### 1.3 The already-started empirical alternative (fallback / gold standard)

`src/bind/cli/paint_haloplane.py` + `paint_massplane.py` (untracked, working):
paint (a) a halo-indicator plane (`log10 M200` in R200 disks, ≥1e12 FoF halos)
into the y-slot and (b) a total-matter column into the tau-slot, then run
**one** lux trace of the fiducial with the same seed/geometry → traced
halo-ID and mass maps pixel-aligned with every existing map, deflection
included. Recover halo pixels by peak-finding. One SLURM job (adapt
`run_sobol_lightcone.sh`; **user submits** — agents must never run sbatch).
Also yields the shared CAP_mat denominator (§3, P2). The analytic chain
(§1.2) is the primary path; this is the validation anchor and the fallback if
the P1 gate fails.

### 1.4 Analysis constants (mirror paper2 — read them from
`examples/_build_ksz_paper2_nb.py`, do not invent)

- CAP filter: disk(<θ_d) minus equal-area ring(θ_d..√2·θ_d); map form exists
  (`cap_on_map(m2d, px, py, theta_pix)` in `examples/_reduce_fgas_lightcone.py`).
- f̃_gas = CAP_tau/(σ_T x_e per-mass) / CAP_mat / F_B, F_B = 0.049/0.3089
  (keep paper2's constant for consistency).
- Samples: BGS M*>10^{11.0} and >10^{11.25} (painted central M* within
  50 kpc/h from the Stars channel; mean logM200 13.60/13.82, z_c≈0.26);
  ELG z≈1.16, hosts logM200≈12.2 (patch-reuse regime); LRG (z 0.45–0.9)
  optional — Part I data (2604.19744) not on disk.
- kSZ-consistency: χ² vs `cov_ksz` in the 1-halo regime θ ≤ 1.4·θ(r200);
  consistent if χ² < N_dof + 2√(2N_dof). Per-halo counts to reproduce:
  49/256 (M*>11.0), 65/256 (M*>11.25).
- Angular scale of a halo on its plane: θ(r200) = r200_comoving/chi[p]
  (both comoving; Group_R_Crit200 is ckpc/h → convert).
- Beam: ACT DR6 y-map ≈1.6′ FWHM Gaussian on BIND maps before y-CAP
  (pixel 0.293′). For the kSZ f̃_gas ratio, compare beamless first (matches
  the per-halo treatment) and show a 1.6′-beam variant panel (D10 lesson:
  never publish a data overlay without a beam-sensitivity check).

### 1.5 Known caveats inherited at map level

- **Patch reuse below 1e13**: BIND painted only M200 ≥ 1e13 halos; smaller
  halos are physical only where they fall inside a painted patch (capture
  ~29% at 4×R200 — memory `lowmass-reuse-capture`); elsewhere the map is
  scaled DMO background. ELG results carry the same grey/green shading as the
  per-halo figure.
- Realization scatter shares one box → underestimates true sample variance;
  say so in the covariance caption.
- 50 realizations share the fiducial DMO geometry across all runs → run-to-run
  differences are purely parametric (by design; cosmic variance cancels).
- SB35 ≠ CV: the p14 caveat does NOT apply.

---

## 2. Deliverables

| # | artifact | where |
|---|---|---|
| E1 | `src/bind/inference/lux_geometry.py` — config.dat reader + halo→map-pixel engine (unit-tested) | repo |
| E2 | `examples/lightcone_desi_catalog.py` — mock-survey catalog builder | repo |
| E3 | `examples/lightcone_cap_stack.py` — CAP stacker over runs (shardable CLI) | repo |
| E4 | `examples/_build_ksz_lightcone_nb.py` → `examples/paper_ksz_lightcone.ipynb` | repo |
| P | catalogs/shards/verdicts/figs under `KS/lightcone/` | ceph |
| F | validation figs V0–V5 + money plots M1 (f̃_gas), M2 (y-CAP), M3 (κ anchor) | ceph + notebook |

---

## 3. Phases

Owner key: **S** = Sonnet (engine code, validation judgment), **H** = Haiku
(mechanical: run CLIs, batch sweeps, regenerate figs). Heavy compute runs as
plain Python on the workstation (many cores; stacking is numpy/IO-bound) or —
for the 256-run sweep — disBatch inside a user-submitted allocation. Agents
never execute sbatch/srun.

### P0 — transform + geometry tables (H, ~1 short session) — ✅ PASS 2026-07-24

Write `KS/lightcone/geometry.npz` + `verdicts/P0.json`:
1. Parse `FID/rt_output/run{001..050}/config.dat` → `rot (50,20)`,
   `disp (50,20,2)`; parse geometry (`a, chi, chi_out, Ll, Lt`) once.
2. Snap↔z table from the 20 stage1 manifests (snap, snap_idx, z, a, chi range).
3. Copy `lightcone_transforms.json` content in.
Gate: 50 records parsed; rot∈{0..3}, disp∈[0,4096); chi strictly increasing;
chi_out[{25,44,58,69,77}] → z within 2% of `source_redshifts` stored in the
map npz. **Fig V0**: chi(z) curve with the 20 plane bands + the 5 source
planes; table of per-realization (rot, disp) for r=1..3 printed in the verdict.

### P1 — halo→pixel engine + empirical validation (S; the load-bearing phase) — ✅ PASS 2026-07-24 (verdict: **P2 required** — Born centroid drifts 0.27→~1.7 px from z 0.18→1.2, real deflection)

Build E1 (`lux_geometry.py`): `load_rt_transforms(FID)`,
`halo_map_pixels(pos_box, snap_idx, realization) -> [(i,j,plane,copy)]`
implementing §1.2 A–D, vectorized.
Validation ladder (each a subplot of **Fig V1**, `verdicts/P1.json`):
1. **Map-free unit check**: from raw FoF (snap 96, M200≥1e13) reproduce
   `stage1_slab*.npz:halo_centers` — max |Δ| < 1e-3 Mpc/h, per-slab counts
   666/723/743/801. Pins Step A+B conventions exactly.
2. **Plane check**: read `FID/lensplanes/tauplane{p:02d}.dat` for snap 96's 4
   planes; overlay predicted plane pixels of the 30 most massive halos on the
   plane image → visual + centroid: measured tau peak within 1 px of
   prediction. Pins Step C (the crop convention) with zero lux involvement.
3. **Per-realization map check**: for realizations r ∈ {1, 17, 50} separately,
   stack 21×21-px cutouts of the fiducial `tau_maps.npz[r-1, 4]` at predicted
   pixels (M200 ≥ 10^13.5; snaps 96, 85, 67, 46) vs 500 random positions.
   Gate: peak/random ≥ 5, |centroid| ≤ 0.5 px, all four snaps, all three
   realizations. Resolve the i/j-transpose ambiguity empirically here, then
   freeze it in E1's docstring.
4. **Deflection budget**: stack-centroid offset + width vs snap z → recorded
   in the verdict (expected ≲1 px; this is the Born error).
Failure mode: if 3 fails only at high z or the width inflates > 1 px →
mark P2 *required* and switch the catalog's pixel source to the traced
haloplane; if 3 fails everywhere → a convention bug, fix before proceeding.

### P2 — fiducial co-trace: haloplane + massplane (S prepares, **user submits**) — ✅ COMPLETE 2026-07-24: job cancelled at 47/50 realizations (sufficient), collected locally → FID/haloplane_trace/{haloplane,massplane,kappa_retrace}_maps.npz (47,5,1024,1024); kappa retrace BIT-EXACT over all 47×5 (max|diff|=0) → geometry certified; deflection 0.79 px @BGS-z / 0.76 px @ELG-z → ELG uses Born positions + σ=0.22′ smearing (stacker beam flag FWHM 0.52′). _haloplane_work/ + mass_lensplanes/ (~30 GB) can be deleted once P6b lands.

Adapt `run_sobol_lightcone.sh` → `run_fiducial_haloplane.sh`: paint
`bind-paint-haloplane` (y-slot) + `bind-paint-massplane` (tau-slot) from the
retained fiducial composites/FoF, then one lux trace (same ini, seed 1992,
n_realizations 50, compute_tsz/tau True) → collect to
`FID/haloplane_maps.npz` + `massplane_maps.npz`.
Purpose: (a) gold-standard halo pixels incl. deflection (peak-find the traced
indicator, match to P1 predictions — **Fig V2**: match rate, offset
histogram; gate ≥95% within 1 px); (b) `CAP_mat` denominator for f̃_gas,
shared across all 256 runs (total matter is feedback-insensitive at aperture
scale — verify ≤2% by comparing CAP_mat from the fiducial massplane vs the
kappa-derived Σ for one snap). Fallback denominator if the user opts not to
run it: kappa-CAP at the nearest-behind source plane divided by the lensing
efficiency W(chi_halo, chi_s) — validate that conversion against the
per-halo CAP_mat at snap 85 instead.

### P3 — mock DESI catalogs (H after S specs E2, ~1 session) — 🟡 in progress (P3prep ✅: per-halo central M* for 256 runs @snap085, counts match paper exactly)

E2 builds `KS/lightcone/catalogs/desi_mock.parquet`: one row per (halo,
snapshot-shell) with M200, r200, chi, z_shell, plane, native plane xy, painted
central M* per Sobol run where available (from `RUNS/*/snap_SSS/
composite_slab*.npz` Stars patches — reuse the paper2 §mstar reduction if its
shards exist), FoF v_LOS (GroupVel along the shell's LOS axis, for the
optional kSZ surrogate), and selection flags: BGS_11.0 / BGS_11.25 (z<0.45),
LRG (0.45–0.9), ELG (0.8–1.6, logM200 12.0–12.6 mass-proxy + patch-capture
flag). Halos down to 1e12 (ELG) — mark `in_patch` via distance to nearest
≥1e13 patch center (< 4×R200_patch, the reuse criterion). Map pixels are NOT
stored (recomputed on the fly from geometry.npz — 60 ints/realization).
**Fig V3**: dN/dz per sample vs DESI targets (qualitative), logM200 and M*
distributions, sky scatter of one realization over the tau map.
Gate: counts per sample per shell recorded; BGS mean logM200 within 0.1 dex
of the per-halo samples' 13.60/13.82.

### P4 — CAP stacking engine + fiducial closure (S) — ✅ PASS 2026-07-24 (map≈patch at 0.25 r200: τ 13%, y 5-8%; smooth 2-halo excess beyond; ~12 s/combo → P5 fits on the workstation)

E3: for one run × one sample × one map type: load `{tau,y,kappa}_maps.npz`
once, loop realizations, compute per-galaxy CAP(θ) on the θ grids (data `th`
grid + θ(r200)-scaled), average copies, stack → mean profile + realization
scatter + galaxy bootstrap. Beam convolution optional flag. Shard per
(run, map, sample) → `KS/lightcone/shards/`.
**Fig V4 (closure)**: fiducial map-level CAP f̃_gas(θ) for BGS_11.0 at snap-85
shell vs the per-halo `fgas_cap_mstar_snap085.npz` — agreement at θ ≤ θ(r200)
(quantify; expect ≤10–15%), documented divergence at larger θ (2-halo +
LOS — the *feature* of this analysis). Same closure for y-CAP vs the
per-halo y stack. Gate: small-aperture agreement; if the map-level is
systematically off at ALL θ, suspect the tau/(σ_T x_e) or mass normalization
— cross-check against `_reduce_fgas_lightcone.py`'s Born pipeline before
touching E1.

### P5 — 256-run sweep (H; disBatch task file, user submits allocation) — ✅ COMPLETE 2026-07-24 (2016 shards = 253 traced runs × 8 combos − 8 empty-sample [runs 64/87 have 0 galaxies at M*>11.25, matching the per-halo cnts exactly]; runs 0114/0115/0117 were never ray-traced — the known 253-node suite)

One task = one run: E3 over {tau, y, kappa} × {BGS_11.0, BGS_11.25, ELG}.
I/O ~2.9 GB/run; estimate ≲5 min/run/worker → a 32-worker disBatch clears
256 runs in ~1 h. Idempotent shards; `--merge` step → 
`KS/lightcone/{fgas_cap,ycap,kcap}_lightcone.npz` (grids × 256 nodes ×
samples, + realization covs). Gate: 256/256 shards, spot-check 3 nodes
against direct recompute. Note dropped runs explicitly if any.

### P6 — science figures + notebook (S) — ✅ COMPLETE 2026-07-24 (after one real defect, user-caught)
- P6a ✅ merge + M3 (κ spread only ~10-22% below τ — M*-selection leaks feedback into the "mass anchor"); M2 needed the mass-matched LRG mock (`desi_mock_snap067.npz`, mean logM200 13.181 vs Sailer+24 13.18; Liu fig3.csv photo-z bins bit-identical = export bug). ELG shards ✅ (incl. 0.52′ deflection-smearing variant — suppresses ELG τ-CAP 14-20%, θ200 is tiny) + LRG shards ✅.
- **M1 defect (user-caught: envelope f̃ 4-16)**: per-node M*-selected numerator ÷ fiducial-sample CAP_mat = mismatched halo populations. Fixed: CAP_mat per node at its OWN sample (node 64: 8.1× correction ≈ its 8.7× mass offset). Lesson recorded: **numerator and denominator must always be stacked on identical samples**.
- **Corrected M1**: fiducial f̃(θ200)=0.80/0.85 (per-halo 0.77/0.84, <5%); envelope [0.19,1.09]/[0.21,1.19] in the 1-halo window. Consistent counts, DESI-precision (headline): **31/253 / 48/251** vs per-halo 49/256 / 65/256 — same story, map-level slightly stricter (real 2-halo/LOS). Survey-variance variant: 234/253, 246/251 (one 25 deg² footprint can't exclude BIND).
- M2-LRG: fiducial/Liu = 1.9-4.2× at matched mass (was "32-165×" pre-matching). Notebook `examples/paper_ksz_lightcone.ipynb` executes clean.
- 🟡 M5 (the user's headline f̃_gas–M200 figure, map-level) in progress: massbin stacker + fiducial/denominators local → 506-task disBatch (user) → figure + notebook section.

- **M1**: map-level f̃_gas(θ) — fiducial + 256-node envelope + kSZ-consistent
  subset (χ² with the map-level realization covariance ⊕ data `cov_ksz`) vs
  Ried Guachalla points; side panel: map-level vs per-halo consistent-node
  count (the 49 → N_map migration is a headline number).
- **M2**: y-CAP(θ) with 1.6′ beam vs Liu+2025 (the trustworthy replacement for
  the distrusted per-halo tSZ figure), same node coloring.
- **M3**: kappa-CAP mass anchor (feedback-blind check that the mock samples
  have the right masses; source plane z_s=1.0).
- E4 notebook `paper_ksz_lightcone.ipynb` (builder script, repo convention):
  §1 geometry + V0/V1/V2, §2 catalog + V3, §3 closure V4, §4 M1–M3,
  §5 caveats. Notebook consumes ONLY `KS/lightcone/*.npz` + verdicts
  (< 200 MB, no raw map loads in heavy cells; gate any full-map cell behind
  `RUN_HEAVY`).

### P7 (optional, clearly-labelled surrogate) — kSZ temperature stack

ΔT_kSZ per galaxy ∝ tau-CAP × v_LOS(DMO)/c with sign-flip stacking, v_LOS
from the catalog's GroupVel — the P4-plan Phase-3 surrogate, now nearly free.
Only if a reviewer needs the literal estimator.

---

## 4. Agent protocol / token economy

- Each phase = one agent with: this file's §1 + its own §3 entry + the
  previous verdict JSON. Never paste map data or notebook JSON into context.
- Engines are self-documenting CLIs (`--help`, docstring with the §1.2 chain
  summary); later agents read the engine docstring, not this plan.
- Heavy loops run as scripts (`python examples/... &` on the workstation);
  agents poll logs, never hold arrays in context. SLURM/disBatch submission
  lines are printed for the user, never executed.
- Verdict JSON schema: `{phase, pass: bool, metrics: {...}, figs: [...],
  notes, next}` — ≤1 KB.
- Figures: matplotlib, paper_style if available; every validation figure has
  the gate criterion in its title (e.g. "centroid 0.21 px ≤ 0.5 px PASS").
- Model assignment: S = engine phases (P1, P2-prep, P4, P6); H = P0, P3-run,
  P5, fig regeneration. Escalate H→S only on a failed gate.

## 5. Companion notebook

`examples/paper_ksz_lightcone.ipynb` — built by `_build_ksz_lightcone_nb.py`
(idempotent, executes end-to-end from reduced products). Sections mirror
phases; every section renders its validation figure(s) with a one-paragraph
"what would look wrong" note. The two money plots reproduce the layout of the
per-halo figures (SB35 thin lines + fiducial + data points) so the map-level
vs per-halo comparison is visual at a glance.

## 6. Kill-gates

- **KG1 (after P1/P2)**: if halo positions cannot be recovered to ≤1 px by
  either path, stop — everything downstream is untrustworthy. (Very unlikely:
  two independent paths, both validated stepwise.)
- **KG2 (after P4)**: if the fiducial map-level f̃_gas disagrees with the
  per-halo result at small aperture beyond ~20% after normalization checks,
  diagnose before the sweep — do not burn the 256-run pass on a broken
  estimator.
- **KG3 (after P6-M1)**: if the map-level consistent-node set is wildly
  inconsistent with the per-halo 49 (e.g. <10 or >150), treat as a finding to
  explain (2-halo/LOS covariance effects), not as an error to hide — but rule
  out covariance bugs first.
