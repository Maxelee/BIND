# REPORT — BIND scatter-residual analysis run

- Run completed in 0.0s wall-clock.
- Output directory: `/mnt/home/mlee1/vdm_bind2/scatter/scatter_residual`
- Figure directory: `/mnt/home/mlee1/vdm_bind2/paper_figures/scatter_residual`

## Scripts that ran

1. `scatter/residual_pipeline.py` — phases 1–3.
2. `scatter/residual_figures.py`  — phase 4 (this script).

## Gates

- **Gate 1**: PASS=True
  - n_halos_post_cut: 1154
  - n_halos_required: 100
  - halo_count_ok: True
  - K: 10
  - K_required: 10
  - K_ok: True
  - median_log10_M_DM_truth: 13.325231919493994
  - median_log10_M_DM_bind: 13.32662028705134
  - median_diff: -0.0013883675573449494
  - median_diff_ok: True
  - no_nan_inf_primary: True
- **Gate 2**: PASS=True
  - n_halos: 1154
  - K: 10
  - cardinality_ok: True
  - fit_pool_mean_zero_ok: True
  - any_diff_above_0p10_dex: False
- **Gate 3**: PASS=True
  - sanity_qdm_qstar: True
  - rho_mass_trend: False
  - frob_finite: True
  - eig_positive: True

## Headline numbers

- N halos: 1154, K = 10
- ‖C^T − C^G‖_F (7×7): 0.6543, split-half null p = 0.0045
- Leading eigenvector angle: 7.05°
- Mean P_aa across 7 obs: +0.828
- ρ_truth(ΔM*,ΔMgas) [low→high mass]: +0.343 → +0.464
- ρ_BIND (ΔM*,ΔMgas) [low→high mass]: +0.494 → +0.588

## Warnings

- truth ρ(ΔM*, ΔM_gas) trend reversed: bin_lo=0.343, bin_hi=0.464
- 8 pair(s) flagged at |z| > 2 — see summary table.
