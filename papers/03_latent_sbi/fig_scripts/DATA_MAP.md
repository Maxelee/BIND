# DATA_MAP — Paper III (`03_latent_sbi`)

For every `\includegraphics` in `main.tex`. All 14 figures were verified feasible: the
placeholder PNG/PDF in `figs/` is byte-identical (md5-checked) to a file already produced
by a notebook cell or script in the corresponding source-branch worktree, and the exact
cached array(s) that cell reads are confirmed present on disk with the expected shapes.
Every underlying computation needed to reproduce the plotted arrays is either (a) a direct
read of a cached `.npz`/`.parquet` array, or (b) a lightweight closed-form reduction
(SVD/PCA, Spearman/Pearson correlation, CCA, `np.polyfit`, 5-fold linear ridge via
`lstsq`) applied to a cached array — never model training, MPI, or Slurm. Two figures
(fig06/fig07/fig08/fig09's NPE/SBC panels) additionally rely on **already-materialized NPE
posterior-sample caches** (`npe_cl.npz`, `npe_progressive.npz`, `sbc_cl.npz`,
`emcee_cl.npz`) rather than re-running `sbi` training — flagged explicitly below.

Common engines/paths referenced repeatedly:
- **Engine `wl_latent_sbi.py`** (worktree `sobol-sb35/examples/wl_latent_sbi.py`, notebook
  `sobol-sb35/examples/wl_latent_sbi.ipynb`) — `E.load_dataset()` reads
  `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` (the 253-run SB35 Sobol sweep: WL
  `C_ell`, peaks/minima/PDF/Minkowski, tSZ/kSZ cross-spectra, `f_gas`/`f_star` scaling
  relations, `X_unit` 30-d design, `run_ids`). Companion module `wl_stat_latents.py`
  (`load_stat`/`reduce_stat`, sklearn PCA/CCA) and `sobol_ml.py` (IOB/active-subspace,
  only used for the non-`main.tex` `f2a_effdim.png`/`f1_global_sobol.png` asides).
- **Notebook `paper_lightcone_figs2.ipynb`** (worktree `sobol-sb35/examples/`) — preamble
  loads the SAME `emulator_dataset.npz` (as `SOBOL`) plus per-run halo atlases at
  `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/halo_atlas/run_{rid:04d}_snap096.npz`
  (5120 files present) and 1P/"twobound" cached reductions under
  `/mnt/home/mlee1/ceph/bind_science/{dashboard_cache,runs,halo_atlas}`.
- **Notebook `paper_ksz_field.ipynb`** (worktree `ksz-desi-act/examples/`) — the
  field-level kSZ/tSZ companion paper; loads the same `emulator_dataset.npz` (as `E`) plus
  real DES Y3×ACT data at
  `/mnt/home/mlee1/ceph/bind_science/ksz_confront/desact_data.npz`. fig10/fig12 are
  **cross-paper provenance** (dossier.md Gaps #2): produced for that sibling paper, reused
  here verbatim.
- **Notebook `shmr_accretion_scatter_sb35.ipynb`** (worktree `sobol-sb35/examples/`) —
  reads the index-aligned per-halo atlas cube
  `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz`.

All paths above were confirmed to exist read-only (`ls`/`np.load` key-listing); none were
searched for — they are exact paths named in the engine/notebook source.

---

## fig01_latent_axes_zones.png

- **Placeholder**: `figs/fig01_latent_axes_zones.png` (md5 `68ad25...`) — byte-identical to
  `examples/wl_latent_sbi_figs/f2b_wl_latent_gaszones.png` on disk today.
- **Shows**: (a) linear PCA scree of the WL suppression latent (2 comps ≈98.7–99%); (b) bar
  chart of which of the 30 params load most on latent-1/latent-2; (c)/(d) the latent plane
  colored by inner (`<R500`) and outer (`R500→R200`) halo gas fraction, r=+0.59/+0.71
  annotated.
- **Original plotting code**: `sobol-sb35/examples/wl_latent_sbi.ipynb` cell 8 (§2a/§2b),
  engine calls `E.latent_svd`, `E.orient_to_gradient`, `E.latent_loadings`,
  `E.gas_zone_labels` (all in `wl_latent_sbi.py`).
- **Data files + keys**:
  - `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` → `t__cl_kappa__value` (253,5,5,724),
    `cl_dmo` (5,724), `a__cl_kappa__ell` (724,), `t__cl_kappa__valid`, `X_unit` (253,30),
    `param_names`, `run_ids` — builds `R, vR = E.wl_suppression_stack(d)` then
    `Z, lam, Vt = E.latent_svd(R[vR])` (plain SVD, no training).
  - `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet` (columns `run, snap,
    M200, M_gas_500, M_gas_200, M_tot_500, M_tot_200`) → `E.gas_zone_labels(run_ids)` computes
    per-run median inner/outer `f_gas` at snap 85, mass bin [13.4,13.8] (pandas groupby
    median — lightweight).
- **Regeneration**: straightforward — load the two files above, call the three
  `wl_latent_sbi.py` helper functions verbatim (pure numpy/pandas, no NN/GP). All confirmed
  present with matching shapes.
- **feasible**: true.

## fig02_latent_physical.png

- **Placeholder**: `figs/fig02_latent_physical.png` (md5 `11aaa0...`) — byte-identical to
  `papers/03_latent_sbi/figs_raw/paper_lightcone_figs2/cell026_out1.png`.
- **Shows**: 11-panel grid — the same (latent-1, latent-2) suppression-SVD plane, each panel
  colored by a different physical quantity (`f_gas`, `Y`, `T`, gas/DM/star concentration,
  `P_e`, entropy `K`, `f_star`, `y`-map, `τ`-map), with Pearson `r1`/`r2` annotated per panel.
- **Original plotting code**: `sobol-sb35/examples/paper_lightcone_figs2.ipynb` cell 26
  (§6.1 `fig_latent_colored`); manifest entry confirms cell/heading
  (`figs_raw/paper_lightcone_figs2/manifest.json`).
- **Data files + keys**:
  - `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` → `t__suppression__value` (253,5,724),
    `t__suppression__valid`, `a__suppression__ell`, `run_ids`, plus `t__cl_yy__value` /
    `t__cl_tt__value` (+ `a__cl_yy__ell`/`a__cl_tt__ell`) for the y-map/τ-map panels — SVD of
    `log10(S(ell))` gives the 2-D latent (`Zl`).
  - `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/halo_atlas/run_{rid:04d}_snap096.npz` (one
    file per of the 253 valid Sobol run IDs; 5120 files present, confirmed keys `M_fof,
    m_gas_500c_bg, m_star_500c, m_tot_500c_bg, T_mw_500c, Y_500c, m_gas_500c, m_gas_200c,
    m_dm_500c, m_dm_200c, m_star_200c, Pe_mw_500c, K_mw_500c`) → per-run group–cluster
    (10^13.4–14.3) medians, one scalar per physical quantity per run.
- **Regeneration**: load `emulator_dataset.npz` once + loop the 253 halo-atlas npz files
  (all present); everything else is `np.linalg.svd` + `np.corrcoef`. No engine re-run needed
  beyond reading cached arrays.
- **feasible**: true.

## fig02b_latent_plane.png

- **Placeholder**: `figs/fig02b_latent_plane.png` (md5 `063403...`) — byte-identical to
  `papers/03_latent_sbi/figs_raw/paper_lightcone_figs2/cell022_out1.png`.
- **Shows**: (a) scree bar+cumulative line (PC1≈89%, PC2≈11%, cum 99.2%); (b) the 253 Sobol
  nodes on the (latent-1, latent-2) plane colored by group–cluster `f_gas`, fiducial (black
  star), and the 1P AGN (red)/SN-wind (blue) spokes projected through the same SVD basis.
- **Original plotting code**: `sobol-sb35/examples/paper_lightcone_figs2.ipynb` cell 22
  (§5.1 `fig_latent_plane`).
- **Data files + keys**:
  - `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` → `t__suppression__value`,
    `t__suppression__valid`, `a__suppression__ell`, `a__scaling_f_gas__log_mass_bins` (7,),
    `t__scaling_f_star__value`/`t__scaling_f_gas__value` (253,7) for the sign-convention
    orientation.
  - `/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json` (the 1P/"twobound"
    run→parameter table) + `g1_stats.npz` (dict of per-1P-run `<run>_clk` response arrays,
    confirmed present, 1.86 MB) → the "1P spokes" `RUNS_OK`/`GS`.
  - `/mnt/home/mlee1/ceph/bind_science/runs/bind/run_0000/Cl_kappa.npz` and
    `/mnt/home/mlee1/ceph/bind_science/runs/dmo/run_0000/Cl_kappa.npz` → fiducial suppression
    `S_FID = CL_BIND[1,1]/CL_DMO[1,1]`, projected into the same latent basis via the notebook's
    explicit `project()` closure.
- **Regeneration**: all five files confirmed present; computation is SVD + a closed-form
  linear projector (`Vt[:2].T`) — no retraining.
- **feasible**: true.

## fig03_wl_ksz_geometry.png

- **Placeholder**: `figs/fig03_wl_ksz_geometry.png` (md5 `b36708...`) — byte-identical to
  `examples/wl_latent_sbi_figs/f2c_wl_vs_ksz_latent.png`.
- **Shows**: two side-by-side scatter panels (WL suppression latent; kSZ τ/y latent), both
  colored by inner-halo `f_gas`, with the canonical-correlation numbers `[0.98, 0.79]` and
  principal angles `[12°, 38°]` in the figure title.
- **Original plotting code**: `wl_latent_sbi.ipynb` cell 10 (§2c), engine calls
  `E.ksz_latent()`, `E.gas_zone_labels()`, `E.orient_to_gradient()`, `E.align_planes()`
  (sklearn `CCA` + QR-based principal angles — lightweight, not trained).
- **Data files + keys**:
  - Same `emulator_dataset.npz` + `integrated.parquet` as fig01 (WL side).
  - `/mnt/home/mlee1/ceph/bind_science/ksz_confront/bind_tauy_xprof_snap085.npz` (confirmed
    present, 290 KB) → `x, tau, y, nodes` radial τ/y profiles feeding `E.ksz_latent()`.
- **Regeneration**: straightforward, all inputs present; CCA/QR are `sklearn`/`numpy`
  one-shot calls on ~250 rows.
- **feasible**: true.

## fig04_multistat_align.png

- **Placeholder**: `figs/fig04_multistat_align.png` (md5 `bf4391...`) — byte-identical to
  `examples/wl_latent_sbi_figs/f2d_multistat_align.png`.
- **Shows**: bar chart, one pair of bars per WL/tSZ summary statistic (peaks, minima, PDF,
  3 Minkowski functionals, `C_ell^{kappa y}`, `C_ell^{yy}`), showing each statistic's own
  top-2 PCA latent's CCA alignment to the `kappa`-`C_ell` 2-D latent.
- **Original plotting code**: `wl_latent_sbi.ipynb` cell 12 (§2d), uses
  `wl_stat_latents.load_stat`/`reduce_stat` (sklearn `PCA`) + sklearn `CCA`.
- **Data files + keys**: `emulator_dataset.npz` → `t__cl_kappa__value`,
  `t__peak_counts__value`, `t__minima_counts__value`, `t__pdf__value`, `t__mf_v0/v1/v2__value`,
  `t__cl_kappa_y__value`, `t__cl_yy__value` (+ matching `__valid`/axis arrays) — all confirmed
  present as `t__<name>__value`/`__valid` pairs in the dataset.
- **Regeneration**: straightforward; PCA(n≤6) + CCA(n=2) per statistic, all closed-form.
- **feasible**: true.

## fig05_spearman_response.png

- **Placeholder**: `figs/fig05_spearman_response.png` (md5 `7a0046...`) — byte-identical to
  `examples/wl_latent_sbi_figs/f1_corr_heatmaps.png`.
- **Shows**: 2×4 grid of heatmaps (7 WL statistics + 1 blank), x-axis = 30 astro params
  (ranked by peak |Spearman r| with the suppression), y-axis = statistic's native bin
  (`ell`/`nu`/`kappa`), model-free per-bin Spearman correlation over the 253 measured Sobol
  runs (deliberately emulator-free).
- **Original plotting code**: `wl_latent_sbi.ipynb` cell 3 (§1), engine
  `E.bin_param_correlation` + `E.param_importance` (rank-transform + Pearson on ranks —
  `scipy.stats.rankdata`, no fitting).
- **Data files + keys**: `emulator_dataset.npz` → `t__suppression__value`,
  `t__peak_counts__value`, `t__minima_counts__value`, `t__pdf__value`, `t__mf_v0/v1/v2__value`
  (+ valid/axis arrays) and `X_unit`/`param_names` for the 30 params. All present (same file
  as fig04).
- **Regeneration**: straightforward; identical machinery to fig04's data loading, just a
  correlation map instead of PCA+CCA.
- **feasible**: true.

## fig06_npe_corner.png

- **Placeholder**: `figs/fig06_npe_corner.png` (md5 `2d74c0...`) — byte-identical to
  `examples/wl_latent_sbi_figs/f3b_corner_cl.png`.
- **Shows**: `corner`-package corner plot, the 6 best-constrained of 30 params (posterior
  std/prior std smallest), NPE posterior samples vs the TNG fiducial (truth = prior centre,
  red).
- **Original plotting code**: `wl_latent_sbi.ipynb` cell 17 (§3b), samples from cell 16's
  `post_cl = cached("npe_cl", ...)`.
- **Data files + keys**: `examples/wl_latent_sbi_figs/npe_cl.npz` → `samples` (24000, 30)
  **[already-materialized NPE posterior cache — confirmed present, no NPE training needed]**.
  Physical-unit mapping uses `bind.inference.design.ASTRO_PARAM_INDICES`/`_unit_to_native`
  and `bind.params.PARAM_LOG_FLAG/PARAM_MIN/PARAM_MAX` (installed-package constants, not
  data files). `shrink = samples.std(0)/pr` with `pr=1/sqrt(12)` (flat-prior constant) picks
  the 6-param subset; `pn` (param display names) from `E.param_names(d)` on the same
  `emulator_dataset.npz`.
- **Regeneration**: draw directly from the cached `samples` array + `corner.corner(...)`; no
  simulation, no `sbi` training invoked.
- **feasible**: true.

## fig07_latent_posterior.png

- **Placeholder**: `figs/fig07_latent_posterior.png` (md5 `6351aa...`) — byte-identical to
  `examples/wl_latent_sbi_figs/f3c_latent_posterior.png`.
- **Shows**: (a) `chi^2` filled-contour over the 2-D suppression latent plane with the MAP
  starred and the 253 Sobol nodes overlaid; (b) the same latent plane with the 30-d NPE
  posterior (cell 16's `samples`) projected onto it via the latent basis, contrasted with the
  Sobol node cloud.
- **Original plotting code**: `wl_latent_sbi.ipynb` cell 19 (§3c), engine
  `E.latent_chi2_grid(sim_cl, Ze, rid[vR], d, ng=140)` for panel (a); a notebook-local
  `to_latent()` closure for panel (b).
- **Data files + keys**: reuses `Ze` (oriented latent scores) and `Vt`/`e1,e2` basis computed
  earlier in the SAME notebook from `emulator_dataset.npz` (as in fig01), plus `samples` from
  `npe_cl.npz` (as in fig06).
- **Caveat — NOT a pure cached-array plot**: unlike fig06, both `E.latent_chi2_grid` (panel
  a) and `to_latent()` (panel b) call `em.predict(...)` on the **cached GP emulator
  checkpoint** `/mnt/home/mlee1/ceph/bind_sb35/emulator/lightcone_emulator_gp.pt`
  (`Emulator.load(EMU_GP)`, confirmed present, 15 MB) to forward-evaluate `S(ell)` at
  ~140×140 grid points / at each of ~4000 posterior samples. This is **inference on an
  already-fitted, cached model** (docstring: "cached GP bundle, ~3 s" to load), not training,
  MPI, or re-running the analysis engine's science computation — but it is more than a flat
  `np.load` of a pre-existing array, since no cached `chi2`-grid or latent-projected-sample
  npz exists on disk today. Flagging explicitly per the "no cached data → keep placeholder"
  rule's spirit: the regenerator should decide whether querying the cached GP checkpoint
  counts as permitted "plotting from cached files" (it never retrains/refits) or should be
  treated as out of scope and left as the placeholder.
- **feasible**: true (data + cached model checkpoint present), with the forward-emulator
  caveat above flagged for the regenerating agent.

## fig08_sbc_crosschecks.png

- **Placeholder**: `figs/fig08_sbc_crosschecks.png` (md5 `3e9260...`) — byte-identical to
  `examples/wl_latent_sbi_figs/f3e_crosschecks.png`.
- **Shows**: (a) NPE (solid) vs `emcee` (dashed) marginals for the 3 best-constrained params;
  (b) SBC rank histogram (should be ≈uniform if calibrated); (c) per-param 68% coverage bars,
  mean 0.70 (red dashed reference line).
- **Original plotting code**: `wl_latent_sbi.ipynb` cell 23 (§3, SBC calibration).
- **Data files + keys**: `examples/wl_latent_sbi_figs/emcee_cl.npz` → `chain` (115200, 30)
  and `examples/wl_latent_sbi_figs/sbc_cl.npz` → `ranks` (120, 30) **[both already-materialized
  caches, confirmed present]**, plus `samples`/`shrink`/`pn` from fig06's cell 16/1.
- **Regeneration**: pure histogram/bar plotting of the three cached arrays; no `emcee` or SBC
  re-run needed.
- **feasible**: true.

## fig09_progressive_stacking.png

- **Placeholder**: `figs/fig09_progressive_stacking.png` (md5 `06d0fb...`) — byte-identical to
  `examples/wl_latent_sbi_figs/f3d_progressive.png`.
- **Shows**: (a) bar chart of mean NPE posterior/prior width for 3 progressively larger data
  vectors (`C_ell`; `C_ell`+peaks; `C_ell`+peaks+`kappa×y`), degrading 0.96→1.01→1.03; (b)
  per-parameter posterior/prior width for the 10 params best-constrained by `C_ell` alone,
  compared across the 3 data vectors.
- **Original plotting code**: `wl_latent_sbi.ipynb` cell 21 (§3d, cautionary result).
- **Data files + keys**: `examples/wl_latent_sbi_figs/npe_progressive.npz` → `dv0`, `dv1`,
  `dv2` (each (24000, 30)) **[already-materialized cache, confirmed present]** → `shr[lab] =
  prog[key].std(0)/pr`. The console-only `obsz` diagnostic (max-|z| of the measured fiducial
  vs the emulated manifold) is NOT plotted (print-only), so it does not gate this figure.
- **Regeneration**: bar/barh plots of `shr` derived purely from the 3 cached sample arrays +
  the constant `pr=1/sqrt(12)`; no emulator call needed for the plotted panels.
- **feasible**: true.

## fig10_kxy_driver.pdf

- **Placeholder**: `figs/fig10_kxy_driver.pdf` (md5 `611277...`) — byte-identical to
  `examples/figures_field/g6_kxy_driver.pdf`.
- **Shows**: (a) band-integrated feedback response `D` (=median of std/|median| over
  `300<ell<8000`) ranked by probe (`y×τ > τ×τ > y×y > κ×y ≫ κ×κ`); (b) node-spread/fiducial
  vs `ell` per probe — `κκ` (WL auto) stays flat/dark, cross/tSZ probes "light up" earlier.
- **Original plotting code**: `ksz-desi-act/examples/paper_ksz_field.ipynb` cell 14 (§6) —
  **cross-paper provenance**: produced for the sibling field-level kSZ paper, reused verbatim
  here per dossier.md Gaps #2 / dossier.json figure #9.
- **Data files + keys**: `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` (same file, via
  `E = np.load(DS)` in that notebook's preamble) → `t__cl_kappa__value`, `t__cl_kappa_y__value`,
  `t__cl_yy__value`, `t__cl_tt__value`, `t__cl_yt__value` (+ their `a__..__ell` axes) — all
  confirmed present.
- **Regeneration**: `D(arr) = nanmedian(nanstd(arr[:,band],0)/(|nanmedian(arr[:,band],0)|+eps))`
  per probe, then two simple plots — no fitting.
- **feasible**: true.

## fig11_lightcone_population.pdf

- **Placeholder**: `figs/fig11_lightcone_population.pdf` (md5 `196166...`) — byte-identical to
  `examples/figures_lightcone/fig_lightcone_population.pdf`.
- **Shows**: 3-panel — (a) `z_s=1` lensing-kernel weight `W(z)` (bars) vs the directly-measured
  per-halo Born `|Δκ|` line, with "old §6 epoch (z=0.034): 3%" annotated; (b) single-epoch↔
  lightcone-weighted proxy fidelity per feature (amplitude vs redistribution, colored bars);
  (c) hexbin of per-halo Born `Δκ` (norm.) vs halo `f_gas(<r500c)`, r=+0.62 annotated.
- **Original plotting code**: `sobol-sb35/examples/paper_lightcone_figs2.ipynb` cell 32 (§6.4
  `fig_lightcone_population`), built from `examples/lightcone_halo_{catalog,observables,
  dkappa}.py` per the cell's own comment (those engines are NOT re-run — only their cached
  output is read).
- **Data files + keys**: single file
  `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/lightcone_halo_section6.npz` (confirmed
  present, 639 KB) → `z_shell`, `w_shell_frac`, `dk_z_shell`, `dk_z_frac`, `feature_names`,
  `proxy_run_corr`, `dk_perhalo_dkappa`, `dk_perhalo_fgas`, `dk_perhalo_M`, `dk_fgas_r` — every
  array the plotting cell reads is present with the expected role.
- **Regeneration**: a single `np.load` + direct plotting, no recomputation at all.
- **feasible**: true.

## fig12_desact_sheary.pdf

- **Placeholder**: `figs/fig12_desact_sheary.pdf` (md5 `1ebdc0...`) — byte-identical to
  `examples/figures_field/g5_desact_confront.pdf`.
- **Shows**: 2-panel `xi_{gamma y}(theta)` — BIND Sobol 16-84% band (red) + median, vs real DES
  Y3×ACT data points (black) for 2 source bins (z̄≈0.74, 0.94), with the 2.4′ ACT beam
  applied, robust 8′–40′ aperture BIND/data ratio annotated (≈2.5×/2.3×), grey shading =
  aperture-limited scales excluded from the robust comparison.
- **Original plotting code**: `ksz-desi-act/examples/paper_ksz_field.ipynb` cell 12 (§5) —
  **cross-paper provenance** (same caveat as fig10).
- **Data files + keys**:
  - `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` → `t__cl_kappa_y__value`,
    `a__cl_kappa_y__ell`, `source_redshifts` (for the `ZS`-weighted `n(z)` projection).
  - `/mnt/home/mlee1/ceph/bind_science/ksz_confront/desact_data.npz` (confirmed present,
    1.87 MB) → `nz_z`, `nz_bins`, `covmat`, `cs_bin1`, `cs_ang`, `cs_value` — the real DES
    Y3×ACT shear×y data vector + covariance + source `n(z)`.
  - Constant `F_XPK = 1.2016e7` (Pylians `XPk_plane` normalization fix, hardcoded in the
    notebook, documented in dossier.md §7) and `Y_BEAM=2.4` (arcmin, physical ACT beam).
- **Regeneration**: Hankel-transform (`scipy.special.jv`) of the cached `C_ell^{kappa y}`
  cube with the real `n(z)` weighting and beam — closed-form, no fitting; all inputs present.
- **feasible**: true.

## fig13_shmr_scatter_budget.png

- **Placeholder**: `figs/fig13_shmr_scatter_budget.png` (md5 `b12c50...`) — byte-identical to
  `papers/03_latent_sbi/figs_raw/shmr_accretion_scatter_sb35/cell011_out1.png`.
- **Shows**: bar chart, SHMR residual-variance budget on the SB35 Sobol suite — "feedback
  (Sobol prior)" ≈0.149 dex² vs "halo-to-halo" ≈0.005 dex², with the twobound one-at-a-time
  marginal spread (≈0.026 dex²) as a dashed reference line.
- **Original plotting code**: `sobol-sb35/examples/shmr_accretion_scatter_sb35.ipynb` cells
  9 (builds `r`, the mass-trend-removed SHMR residual) → 11 (the scatter-budget bar chart);
  manifest cell 11 confirms (`figs_raw/shmr_accretion_scatter_sb35/manifest.json`).
- **Data files + keys**:
  `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz`
  (confirmed present, 53 MB) → `M_fof`, `sobol_valid`, `sobol_m_star_500c` (256×N, only the
  Sobol columns are needed — NOT the twobound/history columns for this specific figure: the
  dashed "twobound" reference line reads `tb_m_star_500c`/`tb_valid`, also in the same file).
  `atlas_design.npz` is loaded in the notebook preamble but not needed for THIS figure
  specifically (only for the later §8 parameter-response section).
- **Regeneration**: `np.polyfit` (quadratic mass-trend removal) + `.var()`/`.mean()` variance
  decomposition — closed-form, no accretion-history file needed (the `HISTORY` `.hdf5`
  dependency only gates §3/§6/§7 formation-time cells, not §4/§5 which produce this figure).
- **feasible**: true.

---

## Summary

| # | figure | feasible | source file(s) confirmed present |
|---|--------|----------|-----------------------------------|
| 1 | fig01_latent_axes_zones.png | yes | emulator_dataset.npz, integrated.parquet |
| 2 | fig02_latent_physical.png | yes | emulator_dataset.npz, 253× halo_atlas/run_*.npz |
| 3 | fig02b_latent_plane.png | yes | emulator_dataset.npz, g1_design.json, g1_stats.npz, Cl_kappa.npz×2 |
| 4 | fig03_wl_ksz_geometry.png | yes | emulator_dataset.npz, integrated.parquet, bind_tauy_xprof_snap085.npz |
| 5 | fig04_multistat_align.png | yes | emulator_dataset.npz |
| 6 | fig05_spearman_response.png | yes | emulator_dataset.npz |
| 7 | fig06_npe_corner.png | yes | wl_latent_sbi_figs/npe_cl.npz (cached NPE samples) |
| 8 | fig07_latent_posterior.png | yes* | emulator_dataset.npz, npe_cl.npz + cached GP checkpoint (forward-eval only — see caveat) |
| 9 | fig08_sbc_crosschecks.png | yes | emcee_cl.npz, sbc_cl.npz, npe_cl.npz (all cached) |
| 10 | fig09_progressive_stacking.png | yes | npe_progressive.npz (cached) |
| 11 | fig10_kxy_driver.pdf | yes | emulator_dataset.npz (cross-paper reuse) |
| 12 | fig11_lightcone_population.pdf | yes | lightcone_halo_section6.npz (single file) |
| 13 | fig12_desact_sheary.pdf | yes | emulator_dataset.npz, desact_data.npz (cross-paper reuse) |
| 14 | fig13_shmr_scatter_budget.png | yes | atlas_cube_snap096.npz |

**n_feasible = 14 / 14. n_unmapped = 0.** No placeholder needs to stay a placeholder for
lack of data — every figure's underlying array(s) exist on disk today. The one nuance is
**fig07**, whose two panels need a forward pass through the cached (not retrained) GP
emulator checkpoint rather than a flat array read; flagged above so the regenerating agent
can make an informed call rather than silently treating it as identical in kind to the
other 13.
