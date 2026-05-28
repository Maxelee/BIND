# BIND × ACT-DR6 kSZ — Paper 2 of the BIND program

> **One-line pitch.** Use BIND's halo-resolved, parameter-conditioned gas
> predictions to identify *which* CAMELS-TNG subgrid parameter (or low-dim
> combination) reconciles the ACT-DR6 × DESI stacked kSZ optical-depth deficit
> at log M ≈ 13 with IllustrisTNG-fiducial.
>
> **Status.** Scoping. Fully accessible within CAMELS-25 + z = 0, no BIND
> retrain required. Six-month paper target. Spun out from the
> Committee Meeting Plan 2026-05-29 v2 §Paper 2.


## 1. Motivation

### 1.1 The observational landscape
- **kSZ measurement.** ACT-DR6 × DESI stacked kSZ (Hadzhiyska et al. line of
  work, Calafut et al., Schaan et al. earlier ACT-DR4) measures the integrated
  electron column density τ around galaxies and groups by stacking kSZ
  temperature decrements against galaxy positions with velocity reconstruction.
  The signal is T_kSZ = −(σ_T / c) ∫ n_e v_los dl, linear in electron density,
  *no temperature dependence*.
- **The headline tension.** At log M ≈ 13 (galaxy groups), measured τ is
  roughly a factor of ~2 below IllustrisTNG-fiducial prediction. The deficit
  grows toward smaller masses. FLAMINGO and BAHAMAS variants with stronger
  feedback match better — suggesting TNG's AGN/wind treatment under-evicts
  baryons at group scale.
- **Independent direction-confirmation.** Same direction is independently seen
  in eROSITA stacked X-ray (lower f_gas vs TNG), SZ stacks (lower Y_int vs TNG
  at fixed M), and FLAMINGO's S8 calibration. **kSZ is the cleanest single
  observable** because it is linear in n_e and decoupled from temperature.

### 1.2 Why this matters
- **Stage-IV WL implications.** The Universe's actual baryon distribution is
  the dominant systematic for Stage-IV WL cluster cosmology and small-scale
  shear. If the AGN feedback is stronger than TNG-fiducial, current
  baryonic-correction priors (BCM, SP(k), bahamas-style) are mis-centered —
  directly biasing S8 / σ8 inference.
- **Direct link to the hydro_replace paper.** hydro_replace showed BCMs cannot
  match power-spectrum suppression even with 10^12 halos out to 5 R_P — i.e.
  the *spatial distribution* of baryons matters at the small scales of WL
  summary statistics. kSZ probes exactly that distribution at the relevant
  mass scale, *independently of WL*.
- **Connects the BIND program to a live observational tension.** Paper 1
  (methods) demonstrates BIND works; Paper 2 *uses* BIND to do something a
  profile-level emulator cannot.

### 1.3 Why current approaches are limited
- **Profile-level SBI** (CARPoolGP, TEMU, ANP, my own X-GAP pipeline) averages
  out the cross-channel and morphology information that pins down which
  subgrid axis is doing the work. Posteriors are diffuse.
- **Hand-tuned recalibrations** (BAHAMAS, FLAMINGO) match observations but
  cannot tell you which physical mechanism is responsible — they retune
  multiple parameters simultaneously without identifiability constraints.
- **No halo-resolved, parameter-conditioned τ prediction exists** across the
  CAMELS subgrid space. This is the gap BIND fills.

### 1.4 Why BIND uniquely
- **ρ_gas per halo across 35 subgrid params** → τ_halo(M, c_DMO, θ) =
  σ_T ∫ n_e dl available as a forward map.
- **Differentiable in θ** via backprop → posterior gradient information that
  profile emulators lack.
- **Conditional scatter physics captured** (~83 % of per-halo joint residual
  variance per the methods paper) → mass-binned stacks won't be biased by
  ignored covariance.
- **Halo-by-halo stochastic samples (K = 10)** → quantifies the
  chaotic-feedback floor à la Genel+2019, separates deterministic feedback
  from stochasticity.


## 2. Scientific question + falsifiable deliverable

**Scientific question.** Given the ACT-DR6 × DESI stacked kSZ measurement of
τ(M) at z ≲ 0.3, what is the posterior P(θ_35 | data) on the CAMELS-TNG
subgrid parameters under field-level forward modeling with BIND?

**Falsifiable deliverable.** Two mutually exclusive outcomes, both publishable:
1. **A single low-dim subgrid direction closes the kSZ deficit.** Report
   θ_best and the physical mapping (e.g. "ε_AGN,kinetic at 60 %, η_wind at
   25 %, IMF at 10 %"). This is a *concrete prior* for cluster-cosmology
   baryon nuisance models in Stage-IV.
2. **No single CAMELS subgrid direction closes the deficit.** kSZ is in
   genuine tension with the TNG architecture itself — implies the missing
   physics is structural (e.g. cosmic-ray driven outflows, AGN duty cycle,
   IMF variability) rather than parametric. Sharper and more interesting
   claim, with implications for the next generation of hydro sims.


## 3. Step-by-step procedure

### 3.1 Pre-paper validation (these support the methods paper too)
1. **Verify BIND τ recovery on held-out halos.** For 1P / CV / SB35-holdout:
   compute true τ_halo (from the hydro sim, integrated through the cluster's
   actual depth) vs BIND τ_halo. Quantify bias, scatter as a function of mass.
2. **Validate the projection convention.** BIND outputs 2D maps projected over
   a fixed 50 Mpc/h LOS depth (CAMELS-25 box). The relevant kSZ comparison
   projects through the cluster's own depth (≪ 50 Mpc/h) plus 1-halo and
   2-halo contamination. **Decide and document the LOS-projection
   methodology** (see §6.1 risks).

### 3.2 Build the mock observable
3. **Match ACT-DR6 stacking geometry.** Use the published aperture R_aperture
   (arcmin → physical at lens redshift) and mass-bin definitions. Replicate
   the compensated-aperture filter used by ACT-DR6 to suppress backgrounds.
4. **Predict τ_BIND(M, θ) on the CAMELS-25 grid.** For each θ in the SB35 /
   CV / 1P grids, generate BIND maps for ~1000 halos spanning the relevant
   mass range; integrate within the matched aperture; bin by M_halo.
5. **Include velocity treatment.** kSZ requires v_los. Two options:
   (a) use the DMO velocity field directly (clean, simulation truth, requires
   verifying DMO velocity reconstruction methodology matches ACT-DR6's), or
   (b) marginalize over v_los as the ACT-DR6 likelihood already does — use τ
   alone, treating velocity as part of the data-side reconstruction.

### 3.3 Inference
6. **Build the population-level likelihood.** Following Matt Ho's
   population-SBI framework: each θ → predicted τ(M) curve → compare to
   observed ACT-DR6 stacks via the published error covariance.
7. **Run SBI.** Sample θ from the CAMELS prior; run BIND; compute the
   likelihood; recover P(θ | data) via NPE / NLE.
8. **Posterior diagnostics.** Coverage tests on synthetic data (§4.E); rank
   statistics; posterior predictive checks against held-out simulation.

### 3.4 Cross-checks (in-paper, not separate paper)
9. **X-ray cross-validation.** Run the same SBI but on the eROSITA-stacked
   f_gas (or stacked L_X-T). Same forward model (BIND), different observable
   channel. Compare posteriors: consistent or in tension?
10. **tSZ cross-validation.** Same forward model on ACT-DR6 stacked Y_int.
    Third independent observable.
11. **Probe-consistency plot.** Triangle of kSZ-alone, X-ray-alone,
    tSZ-alone, joint posteriors. The plot answers: "do the three observables
    identify the same subgrid lever?"

### 3.5 Physical interpretation
12. **Map the constrained direction to physical mechanisms.** SVD on the
    posterior covariance; project onto canonical physical axes (mass loading
    η, ε_AGN,kinetic, ε_AGN,thermal, IMF slope, wind escape velocity). Report
    the dominant axis and its physical interpretation in the discussion.
13. **Forward predictions at θ_best.** Predict K(r) entropy profiles, Y(M)
    tSZ scaling, f_gas(M) at θ_best. These are *not* fit observables in
    Paper 2; they are falsifiable predictions for the next observational
    round (and the entry point for the BIND entropy floor follow-up).


## 4. Validation plots — confidence in the approach

Show all of these before any inference. If A–C fail, BIND isn't fit for
purpose at the τ observable; stop.

> **What the current pipeline establishes (read before trusting the plots).**
> A–C are BIND-vs-truth *emulator-fidelity* checks computed at the projection
> depth of the maps fed in (50 Mpc/h for the 2D `fm_two_head` model). They
> answer "does BIND reproduce the simulation's projected gas in an aperture?",
> **not** "does BIND reproduce the cluster optical depth ACT measures" — that
> requires the cube/3D forward model or a depth correction (see §6.1 Decision).
> D/E/F use the CAP estimator (uniform LOS background subtracted). E is a real
> coverage/SBC test: the synthetic observation is the *true held-out stack*, not
> the emulator's own mean, so it can actually detect mis-specification. C uses
> sim-level aggregation (one stacked τ per sim), so its Spearman p-values are
> honest (N = N_sims, not N_halos).

- **A. Per-halo τ recovery scatter.** True τ_halo vs BIND τ_halo for
  1P / CV / SB35-holdout, colored by M_halo bin. Want: diagonal, low scatter,
  no mass-dependent bias.
- **B. Annular τ profile match.** τ in concentric annuli for true vs BIND
  across (M_halo, θ) cells. Addresses whether the *spatial distribution* of
  gas matches, not just the integrated content. Show fiducial + extreme-AGN +
  extreme-SN slices.
- **C. Spearman τ–parameter sensitivity.** Spearman correlation between
  τ_halo and each of the 35 parameters: BIND vs truth. Want both to agree.
  Shows BIND responds to subgrid parameters *the right way for τ
  specifically*, not just for integrated mass.
- **D. Aperture-matched mock stack vs simulation ground truth.** Stack τ in
  ACT-DR6-like apertures on TNG-Cluster (z = 0) or held-out CAMELS sims.
  Compare to BIND's stacked prediction at the same θ. The "we can do mocks
  correctly" plot.
- **E. SBI coverage on synthetic data.** Generate synthetic ACT-DR6-like data
  at known θ_truth. Run the full inference. Coverage test: do 68 % CIs
  contain truth 68 % of the time? Across 100 mock realizations. Standard SBI
  sanity check.
- **F. Velocity-field robustness.** Show that the posterior is insensitive to
  the v_los reconstruction methodology (option a vs b in §3.2.5). If it's
  not, the paper has a v_los systematic — flag honestly.
- **G. HMF resolution + mass-range constraint.** At the mass bins ACT-DR6
  actually measures (log M ≈ 12.5–14), confirm BIND is well-resolved by the
  CAMELS-25 HMF. Show HMF-weighted error bars on the BIND τ(M) curve. This
  is the scope-constraint check — flag any mass bin where BIND is
  extrapolating.

### 4.H Information-content gate (pre-SBI go/no-go, 2026-05-27)

Run *before* building any NPE/NLE pipeline: `analysis/ksz/sensitivity_1p.py`
(Gate 1) and `analysis/ksz/information_content.py` (Gates 2+3). Result on the
cube model (`fm_cube_two_head`, cluster-depth τ):

- **Gate 1 — sensitivity (1P, isolated variation): strong GO.** Sweeping each
  parameter across its full CAMELS range, τ responds strongly to *feedback*,
  not just cosmology: A_SN1 |Δτ/τ|≈0.81, IMF_slope 0.62, A_SN2 0.47, A_AGN2
  0.39, BH_radeff 0.33. 23/34 probed params responsive (18 feedback), and BIND
  reproduces the **sign** of the truth response on every top responder
  (magnitudes within ~30 %, except a few BIND under-responds to badly, e.g.
  QuasarThresholdPower 0.41→0.09).
- **Gates 2+3 — inverse recoverability (SB35, simultaneous variation):
  degenerate.** Held-out R²(θ_j | x) from the stacked observable. **Zero
  feedback params reach R² ≥ 0.1** in any variant; the best is IMF_slope at
  0.098 — and its *truth ceiling* is also 0.097, so the limit is the
  observable, not BIND. Nonlinearity (RandomForest) does **not** beat the linear
  inverse (data-starved at ~100 sims); the rich observable (+ annular profiles
  + per-halo scatter, dim 3→18) adds only ~1 recoverable param. Only the two
  cosmological axes (Ω_m, Ω_b) clear the bar.
- **Reconciliation + verdict.** τ is *sensitive* to feedback (Gate 1) but the
  stacked observable is *degenerate* across feedback axes (Gates 2+3) — the
  classic "sensitive but not individually invertible" regime. SBI is therefore
  worth running, **but the deliverable is a constrained low-dim feedback
  *direction* (§3.5.12 SVD / Fig 4), not per-parameter identification** —
  exactly §2 outcome 1. Practical consequences: (i) the kSZ-only corner plot
  (Fig 2) will show degeneracy bananas, not sharp marginals; (ii) breaking the
  degeneracy needs the multi-probe channels (X-ray f_gas, tSZ Y; §3.4) — rerun
  this gate on the *joint* observable to confirm before promising per-param
  constraints; (iii) ~100 SB35 sims are too few for high-dim NPE — train the
  inference on the LH (1000-sim) suite.


## 5. Results figures — what we'll find

- **Figure 1: The headline measurement.** Stacked kSZ τ vs M_halo. Black
  points: ACT-DR6 data. Solid red: TNG-fiducial prediction. Dashed blue:
  BIND-fiducial prediction (should match TNG-fid). Solid blue: BIND best-fit
  prediction. Answers *can a subgrid combination close the deficit*.
- **Figure 2: The subgrid identification.** Corner plot of
  P(θ_35 | ACT-DR6 kSZ), with the CAMELS prior as a contour. Likely a few
  axes will be sharply constrained (η_wind, ε_AGN,kin, maybe IMF). Answers
  *which subgrid direction*.
- **Figure 3: Probe consistency.** Triangle: posteriors from kSZ-alone,
  X-ray-alone, tSZ-alone, joint. Answers *do independent observables agree on
  the lever*. If yes → strong claim. If no → "probe-level tension diagnosis"
  claim instead.
- **Figure 4: Physical mapping.** The dominant constrained direction
  projected onto canonical physical axes. Title: "The kSZ-closing direction
  is dominated by [Springel/Oppenheimer parameter] at the N % level." Bar
  chart or stacked-fraction visualization. Connects the statistical posterior
  to galaxy-formation physics.
- **Figure 5: Halo-by-halo mechanism.** Per-halo τ_BIND at θ_fid vs θ_best,
  with color-coding by M_halo. Shows *what changed* at the per-halo level —
  did the gas get pushed out further? More cleared from the core? Sample
  butterfly plots (5-7 representative halos) showing the τ-map difference.
- **Figure 6: Falsifiable forward predictions.** At θ_best, predict (a) K(r)
  for log M = 13 halos, (b) Y_int(M) tSZ scaling, (c) f_gas(M) X-ray
  scaling. These are not fit; they are predictions. Answers *what does this
  paper predict for the next observational round*.


## 6. Risks / failure modes

### 6.1 LOS-projection systematics — the biggest risk
- **The issue.** BIND's 2D output projects over a fixed 50 Mpc/h LOS. ACT-DR6
  measures the line-of-sight integrated optical depth through the cluster's
  own gas distribution (∼ a few Mpc) plus 1-halo + 2-halo contamination.
  **BIND's training projection is wider than the relevant cluster scale by
  ~10×.** Naively integrating BIND's 2D map over-counts ~45 Mpc/h of
  "background" gas.
- **Mitigation paths.** Either (a) use BIND's prelim 3D mode to compute τ at
  the cluster's own depth, requiring 3D validation work, or (b) build a
  background-subtracted aperture estimator (compensated-filter formalism,
  which is what ACT-DR6 does data-side), or (c) explicitly model the
  projection with a per-tile depth correction.
- **Decision point.** Pick the method early; validate via plot D.
- **Decision (2026-05-27).** The aperture τ in `analysis/ksz/` integrates the
  *full projection depth* of whatever maps it is given. We adopt a two-track
  resolution:
  1. **Canonical estimator = compensated aperture (CAP).** All stacked-observable
     plots (C/D/E/F) default to CAP (`tau_utils.per_halo_tau(..., estimator="cap")`).
     Because the CAP weights satisfy ∑w = 0, a spatially *uniform* line-of-sight
     background cancels exactly — this is mitigation (b) and mirrors what ACT-DR6
     does data-side. It does **not** remove correlated 2-halo structure within
     √2·R_ap, which remains a flagged systematic for the full-box (2D) model.
  2. **Preferred forward model = cube/3D mode (mitigation a).** The cube model
     (`--no_large_scale`) projects each halo over a ~6.25 Mpc/h thin slab
     (≈ cluster depth), written as `truth_halos_cube.npz`. The validation loader
     auto-detects this and reports `truth_source`/`los_depth_mpc_h`; every
     observable script prints a `[LOS]` banner stating the integrated depth so
     the 2D-vs-cube distinction can't be silently conflated.
  - **Status (cube run done, 2026-05-27).** Both forward models are now
    validated:
    - **2D `fm_two_head` (50 Mpc/h LOS)** — `fm_testsuite/`. A–C are
      emulator-fidelity at the 50 Mpc/h projection; D's CAP is
      background-subtracted with residual 2-halo.
    - **Cube `fm_cube_two_head` (6.25 Mpc/h LOS ≈ cluster depth)** —
      `fm_testsuite_cube/`. This is the ACT-comparable observable (LOS banner
      confirms `truth_source=cube → 6.25 Mpc/h`). Run the validation by setting
      `TESTSUITE_ROOT=…/fm_testsuite_cube MODEL_NAME=fm_cube_two_head`.
  - **Headline fidelity result.** At cluster depth the per-halo signal is no
    longer diluted by ~45 Mpc/h of background, so BIND's gas **over-prediction
    is more visible**: D's stacked CAP τ(M) sits **+11.6 % above truth at
    logM≈13.15** (vs +3.2 % for the 2D model), tapering to +5–8 % at higher
    mass; A's per-halo bias is +0.03 dex (vs +0.01 dex for 2D). The +12 % at
    logM≈13.15 — exactly the ACT-DR6 deficit scale — is a BIND-fidelity floor:
    sub-dominant to the factor-~2 deficit but a systematic to carry into the
    inference error budget. (Per-tile depth correction, mitigation c, remains
    an option if a tighter floor is needed.)

### 6.2 Velocity-field treatment
- **The issue.** kSZ = τ × v_los. ACT-DR6 reconstructs v_los from galaxy
  density + RSD; we need a consistent treatment in the forward model. If the
  v_los reconstruction has biases, those mix with τ in inference.
- **Mitigation.** Treat τ as the observable (with v_los marginalized as in
  the ACT-DR6 likelihood), or include the v_los reconstruction in the
  forward model with a known transfer function. Validate via plot F.

### 6.3 1-halo vs 2-halo contamination
- **The issue.** ACT-DR6 stacks measure τ around galaxies/groups but pick up
  2-halo contributions from nearby structure. BIND, as a per-halo model,
  does not natively include 2-halo terms.
- **Mitigation.** Either pad BIND-painted halos into a CAMELS-DMO realisation
  (preserves 2-halo) or marginalize the 2-halo term using its known scale
  dependence (linear-bias subtraction in the data side). The first is closer
  to "field-level" forward modeling.

### 6.4 Mass-range / HMF resolution
- **The issue.** ACT-DR6 stacks span log M ≈ 12.5–14. CAMELS-25 has limited
  statistics at log M > 13.5; HMF imbalance in BIND training (per the
  methods paper) also disfavors high-mass halos.
- **Mitigation.** Quantify the HMF-weighted error budget per mass bin;
  restrict the inference to bins where BIND is resolved; possibly use the
  low/high-mass ensemble fix already on the BIND to-do list.

### 6.5 The "single-parameter" framing might fail
- **The issue.** If no single subgrid direction closes the kSZ gap, the
  paper has to pivot to the "tension with TNG architecture" framing. That's
  a valid result but requires careful posterior diagnostics to defend.
- **Mitigation.** Pre-commit to both framings in §6 of the manuscript;
  coverage tests and posterior predictive checks defend either outcome
  rigorously.


## 7. Data + simulation inputs

- **CAMELS-25 SB35 + CV + 1P + LH** — for BIND prediction + training.
- **TNG-Cluster + TNG300-Hydro** — for high-mass cross-validation; tests
  projection methodology and HMF coverage.
- **ACT-DR6 × DESI stacked kSZ data** — primary observable. Need to obtain
  mass-bin definitions, aperture sizes, error covariance directly from
  Boryana / Hadzhiyska / collaborators.
- **eROSITA stacked f_gas** (X-GAP via Erwin Lau / Eckert) — cross-check
  observable, §3.4.9. Already in flight via the CPGP_Xray project.
- **ACT-DR6 stacked Y_int** — second cross-check observable.
- **BIND model** — current trained version with T, Y, P, S channels.
  Possibly retrain on velocity field if §6.2 path (a) chosen.
