# REPO_MAP.md — Phase 0 reconnaissance for BIND Scatter-Residual Analysis

This is the inventory of artefacts located inside the BIND repo for the
scatter-residual cross-correlation task (see `BIND Scatter-Residual Analysis.md`).

## 1. CV simulation truth data
- Root: `/mnt/home/mlee1/ceph/fm_testsuite/CV/`
- 27 simulations: `sim_0` … `sim_26`
- Per sim:
  - `snap_090/full_maps.npz` — `dmo_fullbox` (1024×1024) and `truth_maps` (3, 1024, 1024)
    in physical surface-density units (M_sun/h / pixel) for DM, gas, stars.
  - `snap_090/mass_threshold_1p000e13/halo_catalog.npz` — keys: `centers` (N,2 in Mpc/h),
    `params` (N,35), `masses` / `halo_masses` (N, M_sun/h), `radii` (N, in kpc/h).
  - `snap_090/mass_threshold_1p000e13/halo_cutouts.npz` — keys: `condition` (N, 128, 128),
    `large_scale` (N, 3, 128, 128). 128×128 patches centered on each halo.
- Box: 50 Mpc/h, full grid 1024², patch 128² → `MPC_PER_PIX = 50/1024`.
- Halo IDs: implicit (index within a sim). Use `(sim_id, halo_idx)` as the composite key.
- 1154 halos with `M200c > 1e13 M_sun/h` across all 27 sims (≥ 100 required ✓).

## 2. BIND inference entry point
- Function: `scatter.measure_scatter.measure_scatter(...)`
  ([scatter/measure_scatter.py](scatter/measure_scatter.py))
  - Takes the model, norm stats, theta_norm, dmo_conds, ls_conds, masses, r200_pix, K,
    n_steps, batch_size, dmo_raw, omega_m, seed.
  - Generates K BIND samples per halo, denormalises to physical units, computes 16
    observables per (halo, sample) pair, returns the full obs tensor and variance
    decomposition.
- Lower-level FM sampler: `fd_jacobian_cv._sample_fixed_noise` — fixed-step Euler
  integrator that accepts an explicit `noise` tensor.
- Checkpoint: `/mnt/home/mlee1/ceph/fm_runs/fm_two_head/checkpoints/last.ckpt`
- NormStats: `/mnt/home/mlee1/ceph/fm_runs/fm_two_head/norm_stats.npz`

## 3. Observable computation utilities
- `fd_jacobian_cv.observables_from_phys(phys_3HW, r200_pix, f_b_cosmic, q_DMO_const)`
  returns a dict with: `M_dm, M_gas, M_star, f_b, f_b_norm, Rc_over_R200, q_DM, q_gas,
  q_star, dq_DM, Sigma_gas_c`. Aperture is `min(R200c, 62 pix)`.
- `scatter.measure_scatter._compute_all_obs(...)` extends the above with five
  log-spaced gas-surface-density profile bins (`Sigma_gas_r0…r4`).
- Coverage for the brief's 8-observable set (within R200c):
  - `log10_M_DM`        ← `log10(M_dm)`
  - `log10_M_gas`       ← `log10(M_gas)`
  - `log10_M_star`      ← `log10(M_star)`
  - `log10_Sigma_gas_c` ← `log10(Sigma_gas_c)` (mean Σ_gas inside 0.1·R200c, matches brief §3.2)
  - `q_DM, q_gas, q_star` ← direct
  - `log10_f_b`         ← `log10(f_b)` (supplementary only)
- No re-implementation needed; reuse the existing functions.

## 4. Halo catalogue loader
- `fd_jacobian_cv.load_cv_halos(cv_root: Path)`
  - Concatenates all 27 sims' halo catalogues and cutouts.
  - Returns `cond_raw, ls_raw, masses, sim_id, params, radii_pix`.
- Centers are not returned by default; `scatter/calibration_cv.py` reloads them separately.

## 5. Per-channel normalisation / unit conversion
- `data.NormStats` + `data.log_transform` (z-score + log10 transform of inputs).
- `test_suite.pipeline._denormalize_to_physical(x_gen, norm_stats)` converts model output
  back to physical surface-density units (M_sun/h per pixel).
- `fd_jacobian_cv.normalize_inputs` and `fd_jacobian_cv.normalize_params_fid` build the
  normalised conditioning inputs for the model.

## 6. Sample-seed audit (§2.2 of the brief)
- The model integrator (`_sample_fixed_noise`) accepts an explicit `noise` tensor.
- `measure_scatter` builds noise via `torch.randn(B*K, ...)` from a `torch.Generator(device).manual_seed(seed)`, where `seed=42` is the function-level default.
- **Case (a)/(b)** — the seed *is* a function argument that can be set externally. K=10
  independent noise draws per halo are produced in a single deterministic pass; rerunning
  with the same `seed` reproduces the cached outputs bit-exactly.
- No code modification needed in the inference path.

## 7. Existing scatter / covariance / LOWESS code
- `scatter/measure_scatter.py` — multi-sample inference + per-halo σ_inter/σ_intra/σ_total
  variance decomposition (no covariance matrix yet).
- `scatter/calibration_cv.py` — produces `cv_truth_obs.npz` (truth observables for all
  CV halos) and `cv_bind_obs_K10.npz` (BIND K=10 multi-sample observable tensor).
- `scatter/scatter_jacobian.py` — parameter-Jacobian per halo (different statistic; not
  relevant here).
- No prior implementation of: LOWESS mean fit on observables, residual correlation
  matrices, or Frobenius-norm null tests.

## 8. Existing cached artefacts (eliminate Phase 1 inference cost)
- `scatter/cv_truth_obs.npz` — `truth_obs` (1154, 16), `q_dmo_arr` (1154,).
- `scatter/cv_bind_obs_K10.npz` — `obs_tensor` (1154, 10, 16), `masses` (1154,),
  `sigma_inter`, `sigma_intra`, `sigma_total`, `Y_bar`.
- Halo ordering is identical between the two files and matches `load_cv_halos`
  concatenation order. Median `log10(M_DM)` matches between truth and BIND to
  **−0.0014 dex** (well within Gate 1's 0.05 dex tolerance ✓).

## 9. Deviations from the brief
- Package layout: brief uses `bind/analysis/`, `bind/scripts/`. This repo uses a flat
  layout with `scatter/` containing analysis modules and `run_scatter_*.sh` SLURM
  scripts at the root. New code follows the repo convention:
  - `scatter/residual.py` — LOWESS, residuals, correlation matrices (per-§4–§5).
  - `scatter/residual_pipeline.py` — driver that builds the table, runs Gate 1, fits
    residuals (Gate 2), computes matrices (Gate 3), writes figures (Gate 4).
  - `scatter/residual_figures.py` — fig1/fig2/fig3 + summary table.
- File format: brief asks for `observables.parquet` and `residuals.parquet`. `pyarrow`
  is not installed in `/mnt/home/mlee1/venvs/torch3`. Use compressed `.npz` instead
  (consistent with the rest of the repo), plus a CSV mirror for quick inspection.
- LOWESS implementation: `statsmodels` is not installed. Implement a local-linear
  tricube-kernel smoother in numpy (matches the algorithm `statsmodels.nonparametric.lowess`
  uses; one Gaussian smoothing pass at `frac=0.4`).
- Outputs land in `scatter/scatter_residual/` (data) and `paper_figures/scatter_residual/`
  (figures) — mirrors existing scatter-paper convention. The brief's `outputs/...` paths
  are remapped 1:1.
- The K=10 cached BIND inference data is reused (seed=42 deterministic). No new
  inference work is needed for Phase 1.

## 10. Gate 0 status
All required artefacts in §2.1 of the brief are present.
**Gate 0: PASS.**
