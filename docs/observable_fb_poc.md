# Observable → f_b Mapping — Proof-of-Concept (repo-adapted plan)

*Branch `analysis/observable-fb-map`. Adapted 2026-06-03 from the handoff agent
plan ("Non-parametric Observable → f_b Mapping") to the actual state of this
repo. Notebook: `observable_fb_map.ipynb`. All outputs are ⚠️ provisional.*

## Scientific question (unchanged)
BIND paints, for each cluster-scale halo, both **observables** (tSZ Compton-Y,
kSZ optical depth, X-ray surface brightness, gas temperature/entropy/pressure)
and the **unmeasurable physical state** (baryon fraction f_b). Can we map
observables → f_b *without ever estimating the subgrid feedback parameters θ*?
If observables are a near-sufficient statistic for f_b at fixed mass, BIND
becomes a tool to read off unmeasurable physical state from real data.

**Deliverable:** residual scatter σ(f_b | observables) per mass bin, on
**held-out θ-designs**, and how it shrinks as observables are added; plus a
degeneracy diagnostic and conditional mutual information. **FAILS** (publishable
null) if σ(f_b|Y) ≈ σ(f_b)_marginal within a mass bin. **SUCCEEDS** if observables
substantially reduce scatter and a second observable collapses residual degeneracy.

## What changed vs. the original handoff (the "improvement")
1. **The data exists — Phase 0 / synthetic fallback is dropped.** The 256×1111
   factorial is already generated and reduced at
   `/mnt/home/mlee1/ceph/sobol_ss_cv/cube.npz`
   (`obs (256,1111,8)` = `[Y200,T,S,P,f_gas,m_gen,supp_k10,supp_prof]`, plus
   `M200,R200,sim_id,design_*`). See memory `project_sobol_ss_calibration`.
2. **Data structure confirmed: true factorial.** `M200/R200` are per-halo (1111)
   and shared across all 256 designs; `f_gas` varies design-to-design for a fixed
   halo (same halos repainted under different feedback). ⇒ **the cross-validation
   split is `GroupKFold(groups=design_index)`** — the parameter-free test. Halo
   axis is **lexicographic** sim order (`project_sobol_cube_halo_ordering`).
3. **Cosmology is fixed** (Ω_m=0.3, σ8=0.8, Ω_b=0.049, h=0.671, n_s=0.962), so
   f_cosmic = Ω_b/Ω_m = 0.1633 is a constant — used for the scaled target.
4. **Target = f_b within R200.** The cube's `f_gas` is the gas fraction
   (gas/total in the aperture); f_b additionally includes stars. We read the
   Stars map channel to form **f_b = (Gas+Stars)/total** and use it as the target,
   reporting f_gas alongside (stars are sub-dominant at these masses).
5. **Observable ladder, mapped to what BIND actually emits:**
   - `{Y}` — tSZ Compton-Y (the cube's `Y200`).
   - `{Y, SX}` — add the **X-ray surface-brightness proxy**
     `SX = Σ_ap Gas² √T` (emission-measure × bremsstrahlung emissivity),
     derived from the maps by `tools/observable_fb_reduce.py`. This is the
     genuinely *independent* second observable (density² + T information that the
     pressure-like Y does not carry).
   - `{Y, SX, T, S, P}` — add the gas-mass-weighted thermodynamics already in
     the cube.
   - **kSZ caveat:** the natural kSZ proxy is the optical depth τ ∝ projected gas
     mass = `f_gas·m_gen`. That is (numerator of) the f_b target itself, so
     `{Y, kSZ}` is **near-circular** and only reported as an *upper-bound anchor*
     (it shows the pipeline recovers f_b when handed the gas mass), never as the
     headline degeneracy-breaker. We use **SX** for the honest second-observable test.
6. **MI resampling unit = the design, not the halo** (memory
   `project_mutual_information_methodology`): uncertainties come from bootstrapping
   over θ-designs, mirroring the GroupKFold logic; per-halo bootstrap would
   manufacture a phantom information floor.

## Definitions (exact)
- **Mass bins** (log10 M200 [M_⊙/h]): `[13.0,13.5)`, `[13.5,14.0)`, `[14.0,14.5]`.
  (A few halos reach 14.79; the top bin's right edge is opened to include them
  and the count is logged.) Work *within* bins so f_b's mass trend can't pose as
  observable information.
- **Conditional scatter** σ(f_b|𝒪): fit ĝ(𝒪) on training designs, predict on
  held-out designs (GroupKFold, 5 folds), σ = RMS(f_b − f̂_b) over held-out rows
  in the bin. Report in linear f_b and in dex; report marginal σ(f_b) (in-bin std)
  as the no-information baseline and the **reduction fraction** 1 − σ_cond/σ_marg.
- **Regressor:** `HistGradientBoostingRegressor` (sklearn defaults). Features = the
  observable set; mass handled by binning, with M200 optionally added as a feature
  (both reported). No hyperparameter tuning — scatter is the science, not the fit.
- **Degeneracy diagnostic:** within each mass bin, bin halos into Y-deciles and
  measure mean over cells of std(f_b) at fixed (mass, Y); repeat at fixed (mass,
  Y, SX). Collapse on adding SX ⇒ Y alone is degenerate and SX breaks it. Plus an
  f_b–Y scatter coloured by SX.
- **Conditional MI** I(f_b; 𝒪 | mass bin) via `mutual_info_regression`, converted
  to R²_info = 1 − e^(−2I); bootstrap over designs for the spread.
- **Scaled units (secondary):** Y/Y^SS (Y/M^{5/3}) and f_b/f_cosmic.

## Phases (as executed in the notebook)
- **P1** single observable Y, held-out-θ scatter + pred-vs-true plot.
- **P2** observable ladder table {Y}→{Y,SX}→{Y,SX,T,S,P} (+ kSZ anchor) ×
  mass bin; degeneracy diagnostic.
- **P3** conditional MI / R²_info table; optional repeat with target = T and = S
  (entropy) to show the map generalises to other unmeasurable state.
- **Write-up** `docs/observable_fb_results_2026-06-03.md` with the (a) what the
  numbers say / (b) what they might mean / (c) what must be true + still untested
  structure, and a clear verdict on well-posedness.

## Scope guardrails (unchanged)
Scalar integrated quantities only (no profiles/images yet); fixed cosmology; one
subgrid model (CAMELS-TNG) — no off-grid generalisation; no hyperparameter chasing;
every result ⚠️ provisional with fold-to-fold / bootstrap uncertainty, never bare
point estimates.
