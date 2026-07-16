# Figure data map — Paper V (`05_anisotropy`)

Provenance for every `\includegraphics` in `main.tex`. All four figures are
currently *notebook/engine-output placeholders copied byte-for-byte* into
`figs/` (verified with `md5sum` below) — none are fabricated, but per
`FIGURE_STYLE.md` they still need to be re-plotted as standalone
`fig_scripts/figNN_*.py` scripts using `paper_style.py`. This document
records, for each figure, exactly which cached arrays reproduce it.

**Headline gotcha (read before writing fig01/fig03 scripts):** the field
directory named `sph/` under `/mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/`
was **redefined** between when `wl_anisotropy_paper.ipynb` cells 20 and 24
were executed (producing the frozen output PNGs `figs_raw/.../cell020_out0.png`,
`cell024_out0.png`, still embedded in the notebook) and today. At render
time `sph` held the crude azimuthal-average ("mono") construction; it now
holds the faithful radial-BCM-warp construction (`run_spherical_maps.sh`
comments confirm `sph/` = radial-warp BCM, `mono/` = circular monopole, as
of the current script). Loading the *current* `fid/sph/kappa_maps.npz` for
those two figures would silently swap in different data and violate the
"same data as the placeholder" rule. I verified this numerically (below) —
**use `fid/mono/` wherever cells 20/24 read `fid/sph/`.**

---

## fig01_maps.png — `\label{fig:maps}` (line 209)

- **Caption topic**: 4-panel convergence maps at z_s=1 — DMO, "spherical BCM"
  (crude mono control), BIND full, and the difference κ_bind − κ_mono.
  Caption itself already states this uses the *superseded* mono control, not
  bcm_warp.
- **Placeholder file**: `figs/fig01_maps.png`, md5 `c614257dd98dc75fab1eb85cf9820daf`
  — **byte-identical** to `figs_raw/wl_anisotropy_paper/cell020_out0.png`.
- **Notebook source**: `examples/wl_anisotropy_paper.ipynb` (worktree copy at
  `/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/wt/wl-anisotropy/examples/wl_anisotropy_paper.ipynb`),
  **cell 20**, heading "Field 1 — the maps". Full cell body reproduced in
  `figs_raw/wl_anisotropy_paper/manifest.json` (truncated `source_head`) and
  in the notebook dump at
  `/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/nbdump/wl_anisotropy_paper_full.txt:613-634`.
- **Data files** (all confirmed present, `kappa` key shape `(8 real, 4 zsrc, 1024, 1024)`):
  - `/mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/dmo/kappa_maps.npz` → `kappa`
  - `/mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/mono/kappa_maps.npz` → `kappa`  **(use this, NOT `fid/sph/`, for the "spherical BCM" panel — see gotcha above)**
  - `/mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/bind/kappa_maps.npz` → `kappa`
  - index `[0, isrc=1]` (realization 0, z_s=1.0, since `source_redshifts=[0.5,1,1.5,2]`)
  - 4th panel = `bind − mono` (labelled "anisotropy" in the plot, `κ_bind−κ_sph` in the
    cell's own variable naming, but built from the mono array as loaded)
- **Verification performed**: printed diagnostic in the notebook's cell-20
  stdout — `rms kappa dmo=2.34e-02 sph=2.32e-02 bind=2.29e-02 | anisotropy
  residual rms = 3.65e-03 (16.0% of bind rms)` — matches computing
  `std()` directly from the **current** `fid/mono` array (`0.023160→2.32e-02`,
  residual rms fraction `15.97%→16.0%`) almost exactly, and does **not**
  match current `fid/sph` (bcm_warp; `0.022926→2.29e-02`, `16.08%→16.1%`,
  collides with `bind`'s own value). Confirms `mono` is the correct
  substitute.
- **Regeneration**: straightforward — `imshow` 4 panels (`cividis` for the
  first 3, `RdBu_r` diverging centered at 0 for the residual), same as cell
  20's plotting code (lines 625-634 of the dump). Difficulty: **low**, plotting-only.
- **Style note**: cell 20 used `plt.subplots(1,4, figsize=(17,4.4))` with a
  colorbar per panel and a `suptitle` — the `suptitle`/per-panel titles must
  be dropped/replaced with `panel_label` per `FIGURE_STYLE.md` rule 3.

## fig02_decomposition.png — `\label{fig:decomposition}` (line 364, headline figure)

- **Caption topic**: the corrected, faithful 3-way comparison (BIND vs
  radial-BCM-warp `sph` vs pure-spherical `mono`) at z_s=1 — suppression
  S(ℓ), residual-field power, and the f_aniso/f_tri/f_mono decomposition
  fractions vs ℓ. This is the paper's headline result.
- **Placeholder file**: `figs/fig02_decomposition.png`, md5
  `777520967d7c5a7714a8a656cc8a2342` — **byte-identical** to
  `/mnt/home/mlee1/BIND/examples/figures_lightcone/bcm_warp_comparison_fid.png`
  (a real, already fully-computed engine output, not a fabricated
  placeholder; verified 2026-07-16 that this PNG is an untracked/gitignored
  generated artifact sitting in the current working tree, not a file
  committed to `main` — `git show main:examples/figures_lightcone/bcm_warp_comparison_fid.png`
  fails).
- **Source script**: `examples/bcm_warp_comparison.py` — verified 2026-07-16
  this is tracked on branch `analysis/wl-anisotropy` (`git show main:examples/bcm_warp_comparison.py`
  fails; `git branch -a --contains <blob-commit>` shows only
  `analysis/wl-anisotropy`), **not** on `main`; the copy read for this
  figure was the worktree mirror at
  `/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/wt/wl-anisotropy/examples/bcm_warp_comparison.py`.
  `main()` at bottom of file: loads the 4 field dirs, calls
  `compare()`, `print_table()`, `make_figure()`.
- **Data files** (all confirmed present, same shapes as fig01):
  - `/mnt/home/mlee1/ceph/bind_science/wl_anisotropy/fid/{bind,sph,mono,dmo}/kappa_maps.npz`
    → `kappa`, `source_redshifts`, `fov_deg`
  - Uses `bind.inference.stats.power_spectrum` on each realization/field,
    then bins fractions `f_aniso=(Cb−Cs)/(Cb−Cd)`, `f_tri=(Cs−Cm)/(Cb−Cd)`,
    `f_mono=(Cb−Cm)/(Cb−Cd)` (script lines ~110-120).
- **Regeneration**: the script already runs `power_spectrum` per realization
  (8 real × 4 field types) — a real but cheap FFT-based computation, not an
  engine/MPI/training run. Difficulty: **low-medium** (needs to be ported
  into `paper_style.py` conventions: no titles/suptitle, `panel_label`,
  `save()`); the 3-panel layout and math already match the caption exactly.
  Since the placeholder is already numerically correct and current, the new
  script should reproduce it pixel-for-pixel modulo styling.
- **Note**: this is the ONE figure of the four that already reflects the
  *current* (bcm_warp) `sph/` and `mono/` data — no gotcha here, unlike
  fig01/fig03.

## fig03_crossauto.png — `\label{fig:crossauto}` (line 266)

- **Caption topic**: exact cross/auto decomposition of ΔP = P_bind − P_sph;
  left panel raw ΔC_ℓ contributions, right panel fraction of ΔP. Caption
  explicitly flags this as measured with the superseded mono control.
- **Placeholder file**: `figs/fig03_crossauto.png`, md5
  `c10afed603d81cac813c833d4edee6e0` — **byte-identical** to
  `figs_raw/wl_anisotropy_paper/cell024_out0.png`.
- **Notebook source**: `examples/wl_anisotropy_paper.ipynb`, **cell 24**,
  heading "Why the anisotropy is *first order*: the cross/auto
  decomposition". Full source in
  `/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/nbdump/wl_anisotropy_paper_full.txt:682-738`.
- **Data files**: same `kappa_maps.npz` triplet as fig01 —
  `fid/bind`, `fid/dmo`, and **`fid/mono`** (not current `fid/sph`; same
  gotcha as fig01) — `kappa` array, index `[:, isrc=1]` over all 8
  realizations.
- **Computation** (deterministic FFT/binning, reproduced exactly from the
  cell, not an "engine" — safe to re-derive from cached maps): per
  realization, 2D FFT of `(dmo − mean)`, `(bind−dmo − mean)`,
  `(mono−dmo − mean)`; `cross = 2·Re[F(dmo)* · (F(bind−dmo)−F(mono−dmo))]`,
  `auto = |F(bind−dmo)|² − |F(mono−dmo)|²`; radially bin in `|k|→ℓ` with
  `fov_deg=5.0`, 25 log-spaced bins from ℓ∈[200,3e4]; average over the 8
  realizations.
- **Verification performed**: re-ran this exact computation (numpy, cached
  npz only, no engine/MPI) against both current `fid/mono` and current
  `fid/sph`. Band `ℓ∈[3000,20000]` result:
  - `mono` → cross/ΔP=**+1.39**, auto/ΔP=**−0.39**, closure=1.000 — **exact
    match** to the notebook's printed cell-24 output
    (`cross/dP = +1.39 ... auto/dP = -0.39 ... closure = 1.000`).
  - current `sph` (bcm_warp) → cross/ΔP=+1.45, auto/ΔP=−0.45 — does **not**
    match.
  This confirms unambiguously that `mono` is the correct substitute for
  regenerating this figure with the same data as the placeholder.
- **Regeneration**: **low difficulty** — cell code is ~35 lines of pure
  numpy on 3 cached npz arrays (see line numbers above); no dependency on
  any engine/notebook helper beyond `numpy.fft`. Restyle per
  `FIGURE_STYLE.md` (drop `suptitle`, use `panel_label`, `COLORS`).

## fig04_ejection_profiles.png — `\label{fig:ejection}` (line 455)

- **Caption topic**: radial profiles of the aligned quadrupole fraction
  q2 = |c2|/c0 for the WL convergence residual (left) and tSZ pressure
  (right), group/cluster mass bins, BIND fid (solid) vs TNG truth (dashed).
- **Placeholder file**: `figs/fig04_ejection_profiles.png`, md5
  `fb0b23708aa8ffdc866b9dd1f63826f9` — **byte-identical** to
  `figs_raw/ejection_anisotropy/cell005_out0.png`.
- **Notebook source**: `examples/ejection_anisotropy.ipynb` (worktree copy),
  **cell 5** ("Figure 1 — the radial anisotropy of κ and y (fiducial vs TNG
  truth)"), depending on `fid = reduce_run('fid')` and `truth =
  reduce_run('truth')` defined/called in cell 3. Full source in
  `/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/nbdump/ejection_anisotropy_full.txt:49-164`.
- **Data files** (confirmed present, small — a few KB each, already the
  reduced/cached form, no need to touch raw slabs):
  - `/mnt/home/mlee1/ceph/bind_science/ejection/fid_snap096.npz`
  - `/mnt/home/mlee1/ceph/bind_science/ejection/truth_snap096.npz`
  - keys in both: `q2k` (2,15), `q2y` (2,15), `c0dmo` (2,15), `q2k_ann` (2,),
    `q2y_ann` (2,), `align` (2,), `counts` (2,), `rmid` (15,). Axis 0 = mass
    bin (group, cluster); axis 1 (`q2k`/`q2y`) = radial bin in units of r200c
    (`rmid`, 15 points over `[0,3] r200c`).
  - `reduce_run()` in cell 3 is itself cache-checking (`cf = OUT /
    f'{run}_snap{snap:03d}.npz'`; returns cached npz directly if present,
    only touches raw composite slabs if absent) — the cache already exists
    so no slab access is needed to regenerate this figure.
- **Verification performed**: loaded `fid_snap096.npz` directly and computed
  `counts`, `q2k_ann`, `q2y_ann`, `align` — **exact match** to the
  notebook's cell-3 printed stdout (`fid counts per mass bin: [2512 416]`;
  `q2_kappa (ejection annulus): [0.0431 0.0349]`; `q2_y (ejection annulus):
  [0.1312 0.1634]`; `alignment: [-0.773 -0.895]`). `truth_snap096.npz` has
  the matching structure (`counts=[2512,416]` too, since fid/truth share
  halo catalogs by construction).
- **Regeneration**: **low difficulty** — pure 2-panel line plot,
  `RMID` vs `100*q2k`/`100*q2y` per mass bin, solid (fid) + dashed (truth),
  shaded ejection annulus `axvspan(0.5,2.0)` (cell 5 lines 145-164 of the
  dump). No dependency beyond the two cached npz files.

---

## Summary table

| figure | placeholder = | data | current-vs-placeholder gotcha | difficulty |
|---|---|---|---|---|
| fig01_maps.png | `wl_anisotropy_paper.ipynb` cell 20 | `fid/{dmo,mono,bind}/kappa_maps.npz` | **use `mono`, not current `sph`** | low |
| fig02_decomposition.png | `examples/figures_lightcone/bcm_warp_comparison_fid.png` (already real, current) | `fid/{bind,sph,mono,dmo}/kappa_maps.npz` via `bcm_warp_comparison.py` | none — already current data | low-medium |
| fig03_crossauto.png | `wl_anisotropy_paper.ipynb` cell 24 | `fid/{bind,dmo,mono}/kappa_maps.npz` | **use `mono`, not current `sph`** | low |
| fig04_ejection_profiles.png | `ejection_anisotropy.ipynb` cell 5 | `bind_science/ejection/{fid,truth}_snap096.npz` | none | low |

All four figures are feasible to regenerate from cached data with no gaps.
None require re-running the lightcone pipeline, MPI, or Slurm — every
number needed is already sitting in an `.npz` on ceph, and I verified each
one numerically against the frozen placeholder's own printed diagnostics
before writing this map.
