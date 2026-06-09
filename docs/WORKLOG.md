# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

---

## 2026-06-09 — Circular paste aperture is now the standard (`r200_factor=4.0`)

The BIND composite now defaults to a **circular `4×R200c` paste aperture** instead
of the legacy square Hann taper. Over 26 CV sims (`fm_two_head`), circular removes
the small-scale total-matter `P(k)` deficit: BIND/Truth at `k` 40–70 h/Mpc goes
−10.6% (square) → −0.8% (circular), at the cost of mild +7–11% over-production at
`k` 10–40. The hydro-replaced control shows the square-aperture high-`k` deficit is
an over-smooth-core *model* issue that the tight aperture compensates geometrically.

- Engine: default `r200_factor` 0.0→4.0 in `RunConfig` (`schemas.py`), both CLIs
  (`camels_suite`, `paint`), and `build_bind_composite`. `load_halo_catalog` now reads
  R200c from cached catalogs (legacy `radii` kpc/h as well as `r200s` Mpc/h), so a
  `--repaste` no longer needs the FOF files and preserves halo↔patch ordering.
- Note + example figure: `docs/circular_aperture.md`. Full study (also scale_global
  P(k)-invariance + taper sweeps): `experiments/composite_study/FINDINGS.md`.
- Reversible: `--r200_factor 0` restores square; circular composites rebuild cheaply
  from cached `generated_halos.npz`.

## 2026-06-07 (pm-4) — S(k) local-FD box driver (`analysis/observable-fb-map`)

The WL-target side of the chain on local-FD footing. S(k) is full-box, so it reuses
`box_supp_sobol` paste+Pk (NOT the per-halo FD script). Discovery: `box_supp_sobol.npz`
already exists — `S_true (256,724)` = per-design BOX suppression (the real WL systematic,
better than the per-halo supp_k10) + `k_box`, `S_truth`. `tools/fd_sk_box.py`:
- `--validate` (NO GPU): recompute box S(k) from existing cube maps via composite_pk →
  **reproduces cached S_true EXACTLY (max|dS|=0.00e+00)** for designs 0,1,128 → box reuse
  verified. (design 128 S(k~10)=0.85.)
- compute (GPU): generate fm_thermo mass-channel patches at θ_fid±Δ (fixed noise, reuses
  fd_jacobian_thermo machinery) → central-diff dS(k)/dθ (35 params), saves S_fid + J_Sk(k).
  `run_fd_sk_box.sh` (array 0-6, n_steps=20). User submits + merges.
**Synthesis J_S upgraded to box S_true + §E added:** J_S now = ∂(box S(k=10))/∂θ from
`box_supp_sobol.S_true` (the real WL systematic; prior σ(S)=0.094) instead of the per-halo
supp_k10 mean. Headline shifted (more correct): σ(S(k)) reduction Y 1% / Y+SX 9% / all-gas
**46%** (was 41%); top S(k) drivers IMFslope, BlackHoleRadiativeEfficiency, WindEnergy,
QuasarThreshold (AGN still poorly constrained: detect 0.41/0.08, gas 8/3%). §D Y+SX now
Sobol 9% / local-FD 4% (still r=0.93). New §E: when `fd_sk_box_fm_thermo.npz` lands, overplots
local-FD vs box-Sobol J_S per param (same normalized-θ space) — graceful fallback tested.
§C mass-decade decomposition still uses per-halo supp_k10 (separate attribution, kept).

## 2026-06-07 (pm-3) — found pre-existing FD Jacobian + synthesis (`analysis/observable-fb-map`)

User had ALREADY computed the rigorous local FD Jacobian (the refinement I flagged):
`fd_jacobian_cv.py` on **analysis/2d** (fixed-noise central differences at CV fiducial,
35 params, mass+scaling-relation obs, fm_two_head). Artifacts in working tree
`analysis_physics_cache/`: `proj6_cv_fd_fm_two_head*.npz` (per-halo J, 1154×35) +
`jacobian_inference_artifacts/cv_scatter_jacobian_inference_latest.npz` (Fisher summary:
detectability_all, f_theta_all, c_theta_all, k90=14). **Param order: FD columns = cube
param_names (SB35/NormStats) order — VERIFIED (CV fiducial inside every Sobol column range
except p14 col14 + the two Δz fiducials=0); the artifact CSV's w0/wa/Mν labels are a WRONG
template — ignore.**

**(a) Synthesis** (`tools/build_synthesis_nb.py` → `synthesis_feedback_constraints.ipynb`,
no new compute): unifies the rigorous FD mass-obs Fisher + the Sobol gas→S(k) chain.
- **Cross-validation:** FD |∂f_b/∂θ| vs logM astro-only r=−0.73 ≡ Sobol Var_d[f_b] r=−0.73
  → cheap Sobol Jacobian CERTIFIED. Cosmo/feedback dichotomy: cosmo r=+0.73 (massive
  clusters cosmology-standard), feedback r=−0.73 (groups feedback-dominated).
- **Money plot:** S(k) sensitivity vs constrainability per param. SN winds (WindEnergy,
  IMFslope) drive S(k) AND well-constrained (mass detect 0.88, gas 15%); AGN
  (BlackHoleRadiativeEfficiency=top S(k) driver, QuasarThreshold) drive S(k) but poorly
  constrained → residual WL systematic is AGN. k90=14 effective feedback dim.
- σ(S(k)) reduction: Y 1% / +SX 6% / all-gas 41% (carried over from fisher_chain).

**(b) SCOPED + observables validated; GPU campaign remains (user submits).** Extending
the rigorous local FD to the WL chain splits into two pieces:
- **Thermo Y/SX FD: BUILT + CPU-validated.** Thermo weights ARE available at
  `/mnt/home/mlee1/ceph/fm_runs/fm_thermo/checkpoints/last.ckpt` (predict_thermo=True,
  out_channels=8 → 7 physical via pipeline._denormalize_to_physical). `tools/fd_jacobian_thermo.py`
  = port of analysis/2d `fd_jacobian_cv.py` to bind.* imports + adds `thermo_observables`
  (Y200, SX aperture) into PER_HALO_KEYS; observable functions VALIDATED vs cube
  (`tools/fd_thermo_observables.py`: Y200/SX corr=1.0000). CPU smoke test (2 halos,
  param 0, n_steps=4) ran end-to-end → finite J_Y200/J_SX. `run_fd_jacobian_thermo.sh`
  (sbatch array 0-6, fm_thermo, n_steps=20). NO GPU on this workstation (cuda False) → user
  submits the SLURM array, then merges. Script ALSO saves fiducial observables (`Ffid_*`,
  one extra fixed-noise pass) so the output is self-contained for the log-Jacobian
  dlnY/dθ = J/Ffid (CPU smoke: dlnY/dθ ≈ −0.85/−0.33, sensible O(1)). **Synthesis notebook
  WIRED** (`build_synthesis_nb.py` §D): if `analysis_physics_cache/thermo_cv_fd_fm_thermo.npz`
  exists it recomputes the Y+SX σ(S(k)) reduction with the local-FD Jacobian and compares to
  Sobol; else prints the run instruction (graceful fallback, tested). Convert raw dO/dθ→log
  via Ffid (now saved). **User ran the 7-shard array** → `thermo_cv_fd_fm_thermo.npz` merged
  (35/35 params, J_Y200/J_SX finite 1154×35) — but those shards PREDATE the Ffid feature, so
  Ffid is absent. Fix: added `--fiducial_only` mode (one fast pass, saves Ffid_*) →
  `thermo_cv_fd_fiducial.npz`; §D now reads Ffid from merged-or-fiducial file (graceful skip
  if neither). Merge gotcha: quote `--shard_glob "...*.npz"` (shell-glob expansion bug).
  **Ffid obtained with NO GPU:** the CV fiducial EMA generation already exists at
  `CV/sim_i/snap_090/.../fm_thermo_ema/generated_halos.npz` ('generated' (n,7,128,128)).
  `tools/build_thermo_fiducial.py` reads it across the FD-loader sim order (radii fallback
  r200c_mpc_h when 'radii' absent, matching the FD), computes Y200/SX apertures →
  `thermo_cv_fd_fiducial.npz` (1154 halos, masses VERIFIED == FD masses_use). Caveat: EMA
  fiducial vs non-EMA FD J → small per-obs normalisation offset, negligible for the relative
  reduction. **§D RESULT: local-FD Y+SX σ(S(k)) reduction = 4% ≈ Sobol 6%** → the cheap
  Sobol gas Jacobian is validated against the rigorous local FD on the WL-relevant quantity.
  (Headline 41% needs all thermo Y+SX+T+S+P; FD covered only Y,SX.) Gas side now on local-FD
  footing; only J_S (=dS(k)/dθ) remains Sobol-based (needs the box ±Δ driver).
  **§D visualized + base bug fixed:** the FD log-Jacobian was natural-log while halo_cov/Sobol
  jac are log10 → fixed (÷ln10). New figD (2 panels): (left) per-param gas Jacobian Sobol vs
  local-FD scatter on 1:1, **r=0.93** — the real validation (same response per param); (right)
  σ(S(k)) reduction bars Y+SX Sobol 6% / local-FD 2% / all-gas Sobol 41%. Honest framing: Y+SX
  is a WEAK constraint so the 6-vs-2 gap is small-Jacobian amplification (+ linear-vs-local,
  EMA-vs-nonEMA), NOT a response disagreement; the 41% is the full thermo set (T,S,P), still
  Sobol, now backed by the r=0.93 Y,SX Jacobian validation.
- **S(k) local FD:** S(k) is a FULL-BOX quantity (box_supp_sobol pastes patches into the
  50 Mpc/h box → P_hydro/P_DMO), NOT per-halo — so it does NOT go in fd_jacobian_cv.py.
  Rigorous dS(k)/dθ = generate fm_two_head patches (fixed noise) at θ_fid±Δ for the 35
  params (70 points) → box_supp_sobol paste+Pk → central difference. No new weights;
  reuses tested box_supp machinery + fd_jacobian's fixed-noise generation. GPU+CPU SLURM.
Both are GPU campaigns the user launches; Sobol J_S (R²~0.7) already substitutes in the
synthesis until the local-FD versions land.

## 2026-06-07 (pm-2) — Fisher chain: gas obs → σ(S(k)) via BIND Jacobian (`analysis/observable-fb-map`)

The headline *use* of capability #1 (per the reframe below). BIND's clean parameter
Jacobian (finite-difference at fixed DMO+noise — the derivative regime, BIND-only;
backprop-through-ODE is the wrong tool for a flow model) links what surveys observe
(gas) to what limits cosmology (S(k)). `tools/build_fisher_chain_nb.py` →
`fisher_chain.ipynb` (fast, cube.npz + obs_fb_extra.npz; linear-response Jacobian over
the Sobol prior, R²~0.68–0.78):
- J_O = ∂log(stacked gas obs)/∂θ, J_S = ∂S(k=10)/∂θ; Fisher F=J_Oᵀ Σ_O⁻¹ J_O (Σ_O =
  halo stacking cov /N_cl + noise); posterior C_θ=(F+12·I)⁻¹; propagate to σ(S(k)).
- **Result (N_cl=1000, 10% noise):** reduces σ(S(k=10)) by **tSZ alone 1%, +X-ray 6%,
  +kSZ 7%, all-gas (Y,SX,τ,T,S,P) 41%** (noiseless ceiling 78%). Saturates by N_cl~100s
  (intrinsic-scatter limited, not noise). → the constraining power is in the JOINT
  multi-probe thermodynamic field, not any single observable — the honest, correct form
  of "field-level matters" (about constraining feedback that drives WL, not a regression).
- Alignment fig: top S(k) drivers = WindEnergy, IMFslope, BlackHoleRadiativeEfficiency,
  QuasarThreshold. SN-wind params both drive S(k) AND are well-constrained by gas (~14%);
  AGN radiative efficiency is the TOP S(k) driver but only ~8% constrained → residual WL
  systematic is AGN-dominated (interpretable result).
- Honest scope: linear/Gaussian, single subgrid family, fixed cosmo, projected S(k),
  representative covariance; connect at P(k) (projection-invariant sensitivities), no
  shear forecast. Local finite-difference Jacobian would refine the linearisation.

## 2026-06-07 (pm) — thesis reframe + controlled-experiment proof (`analysis/observable-fb-map`)

**Key reframe (user-driven).** The "field beats profile [at the f_b inverse]" claim is a
strawman: if the target is a stacked summary, you can forward-model the CAMELS training
sims and regress — no generative emulator needed. The field-vs-profile experiments
(below, am session) demonstrate the value of *forward-modeling*, which CAMELS gives for
free, NOT of a generative field emulator. So most of `paper.ipynb` §3 (six-functionals)
is vulnerable.

**New thesis.** BIND is a fast, faithful generative baryon forward model; its irreducible
value is (i) **controlled same-halo feedback experiments** no hydro suite can run →
attribute the WL baryon systematic to halo populations; (ii) **transport** CAMELS feedback
onto arbitrary N-body volumes; (iii) the **joint field** for field-level statistics. The
f_b inverse is demoted to a fidelity application (with the explicit concession). Plan:
gut §3, promote §4.2/§4.3 (S(k) attribution) to headline, keep §4.1 fidelity.

**Point-1 experiment built** (`tools/build_controlled_experiment_nb.py` →
`controlled_experiment.ipynb`, fast, from cube.npz + pk_supp_extra.npz; supp_k10 is
per-halo S(k=10) with exact cross-design correspondence):
- **A. Variance efficiency** (paired same-halo vs unpaired varying-IC): gain G=1/(1−ρ),
  median G=3 (f_b)/6 (S(k)), rising to **100–400× for small feedback steps** (the
  derivative/sensitivity regime). Answers "control cosmic variance efficiently": yes,
  quantified — modest for large feedback diffs, enormous for sensitivities; plus exact
  (no back-reaction) + joint-space, beyond CAMELS 1P.
- **B. Attribution** (needs correspondence): f_b per-halo response Var_d[f_b] strongly
  mass-dependent (r=−0.73; low-mass halos respond ~100× more — evacuation); shuffle
  control destroys it (r=0, inflates quiet halos to pooled scatter). Honest finding:
  S(k) per-halo response is mass-INDEPENDENT (r=0.06) → S(k) attribution is POPULATION
  level: group decade [13,13.5) carries **55.7%** of Var[S(k=10)] (≈ the §4.3 result).
- **WL connection (honest):** connect at P(k) (van Daalen 2020 interface); which halos
  source Var[S(k)] is projection-invariant → no shear lightcone needed for attribution;
  a survey shear-bias forecast is explicitly NOT claimed.

## 2026-06-07 — simple field-vs-profile demonstration (`analysis/observable-fb-map`)

The §3 "six-functionals" case in `paper.ipynb` is correct but diffuse — a taxonomy,
not a single observer-relevant number. Built a **standalone** one-experiment
demonstration that a *field*-level emulator beats a *profile*-level one for reading
$f_b$ off a realistic stacked observation.

The trap (the user's instinct, corrected): a straight stacked-$Y(r)\to f_b$ regression
IS the paper's existing "foil" and it *works* (69–80% reduction) → would prove the
profile *sufficient*. The distinction is not the regressor; it is that **only a field
emulator can forward-model the actual measurement** so training and data are processed
identically. Cast as one controlled inverse problem (leave-one-design-out over the
256-design × 1111-halo CV Sobol cube; each design = a held-out mock observation):

- `tools/forward_model_reduce.py` → `ceph/sobol_ss_cv/forward_model_stacks.npz`:
  streams per-halo 2D fields; per design builds a **clean** stack (profile-emulator
  output) and a **processed** stack run through a real pipeline — steep mass function +
  flux-limited **Eddington selection** + **beam** (1.6′ ACT-equiv, σ≈53 kpc/h) +
  **miscentering** (150 kpc/h) + **core mask** (0.15 R₂₀₀). Target = parent
  (unselected) mass-weighted $f_b$. ~30 min streaming; resumable (--start/--end).
- `tools/build_fieldvsprofile_nb.py` → `field_vs_profile.ipynb` (12 cells): three
  estimators differing ONLY in the training observable — profile/vacuum (clean→clean),
  profile/real-obs (clean→**processed**), field/real-obs (processed→processed).

**Result.** Pipeline reshapes the stack by a large radius-dependent factor (core
0.59×, intermediate ~1.9× — non-commuting beam/miscenter, Fig 1). Headline (Fig 2):
profile-on-real-obs is biased **−10%** in $f_b$ (RMS 0.0176); field path **unbiased**
(RMS 0.0108); profile-in-vacuum unbiased (RMS 0.0081) → the failure is the
forward-model mismatch, not the profile→$f_b$ link. Per-bin (fixed composition) the
shape bias is −0.3/−5.5/−1.3%; the larger aggregate −10% includes the
selection-driven mass-mix shift (also a field-level effect). Caveat: mock obs is BIND's
own field (isolates the *forward-model* bias, not BIND-vs-CAMELS fidelity); CAMELS-truth
obs is the natural follow-up. Not yet committed.

## 2026-06-04 — paper.ipynb full paper build-out (`analysis/observable-fb-map`)

Expanded `paper.ipynb` from the Intro+demonstrations note into a complete paper
with six top-level sections: **§1 Introduction → §2 Methods → §3 The case for
field-level emulation → §4 Results → §5 Discussion → §6 Conclusion**. All edits
in `tools/build_paper_nb.py` (paper.ipynb is generated); rebuilt + executed
clean (42 cells, 16 figures, **zero** errors).

- **§2 Methods (new):** 2.1 the BIND field emulator (flow-matching $p_\theta(x|c)$,
  7 channels, AdaGroupNorm, common random numbers), 2.2 the controlled Sobol
  feedback suite (1111 fixed halos × 256 designs, feedback-vs-intrinsic split),
  2.3 observables + instrument forward model (azimuthal average, aperture scalars,
  beam $\star$ field + miscentering), 2.4 the old §0 setup retitled + a new
  **Figure M** (Sobol design coverage + ACT/Planck beam kernels).
- **§3 (reframe):** moved the old top "Overview" block down to lead §3 and
  renumbered the existing demonstrations §1–§8 → §3.1–§3.8 (with all in-body
  cross-refs updated).
- **§4 Results (new):** Fig10 baryon-fraction scaling fidelity vs hydro truth
  (~5.4% median error, from `obs_fm_two_head.npz`); Fig11 painted gas → box
  $S(k)$ suppression (Spearman ρ≈0.53, `box_supp_sobol.npz`+cube); Fig12 Shapley
  variance decomposition (group decade ~57% of Var[S]; per-halo leverage
  [0.069, 0.151, 0.407] %/halo, `partial_supp_sobol.npz`).
- **§5 Discussion (new):** Fig13 $f_b(r)$ recovery from mock (Y,SX) vs CAMELS
  truth (`realdata_results.json`); Fig14 feature-ladder robustness (Y vs Y+SX vs
  full thermo — full set over-fits the domain gap by orders of magnitude); 5.3
  caveats (one sub-grid family, proxies, projection, domain gap).
- **§6 Conclusion (new):** replaces old §9, wrapping Methods/§3/§4/§5.
- All headline numbers verified against on-disk data products before writing the
  narrative; figures saved to `figures/paper/` (gitignored).

---

## 2026-06-04 — paper.ipynb science Introduction (`analysis/observable-fb-map`)

Reframed `paper.ipynb` from a pure "why field-level" validation note into a
science paper by adding a result-focused **Introduction** (§1) following the
problem → standard-approach-limits → field-updates → novel-direction → results
arc. The problem is baryonic suppression of $S(k)$ as the leading Stage-IV
weak-lensing systematic and the unobservable/uncomputable baryon budget $f_b$;
the novel direction is BIND's field-level + common-random-numbers repainting,
enabling controlled feedback experiments and a parameter-free observable$\to f_b$
inverse. The results paragraph threads the companion notebooks
(`scaling_relations` validation → this notebook's necessity proof →
`profile_fb_realdata` reading $f_b(r)$ at RMS≈0.004 on held-out feedback →
`pk_suppression_decomposition` cosmology payoff), framing the sky-data step as
"one of data, not of method."

- Inserted intro as the new top cell; demoted the old opening to an unnumbered
  `## Overview` so there is a single H1/abstract and no clash with the §0–§9 body.
- Mirrored both edits into `tools/build_paper_nb.py` (first two `md(...)` blocks)
  so a rebuild is reproducible — verified by building into a temp dir and diffing:
  25 cells, **zero** source mismatches vs the executed live notebook. Live
  notebook's embedded figures were left intact (builder not run in place).

## 2026-06-04 — paper.ipynb §7 rigor fix: transparent gas asymmetry (`analysis/observable-fb-map`)

Reviewer (human) distrusted §7 of `paper.ipynb`: (i) the same-halo claim was
illustrated with **two different halos** sorted by the opaque shard `morph`
scalar (circular), and (ii) the "important parameters" came from a marginal
Pearson corr of the *design-mean* of that black-box scalar — weak (0.2–0.37) and
including implausible drivers (`WindFreeTravelDensFac`, `VariableWindVelFactor`).
Both complaints were correct. Rebuilt the section from scratch:

- New `tools/compute_gas_asymmetry.py` → `ceph/sobol_ss_cv/gas_asymmetry.npz`:
  transparent CAS 180° rotational asymmetry of the Gas channel within R200,
  centroid-centered (`A=0` for any axisymmetric field, so exactly orthogonal to
  `A_r`). Caches `A (256,1111)` + the demo halo's gas under all 256 designs.
  Note: this transparent `A` correlates only **0.47** with the old `morph` shard
  scalar — confirming the black box was a poor "asymmetry".
- **Same-halo figure** (`fig8_feedback_same_halo.png`): halo 870 (logM 13.47),
  weak vs strong feedback (32 lowest/highest composite-score designs), identical
  DMO input; A rises 0.299→0.440, ratio panel localizes change inside R200.
- **Controlled sensitivity** (`fig8b_morphology_sensitivity.png`): per-halo OLS
  of standardized `A(θ)` on standardized θ, averaged over 1111 halos with
  halo-bootstrap 95% CIs (`β_p = ⟨(ΘᵀΘ)⁻¹ΘᵀA_h⟩_h`). New ranking is physical —
  **WindEnergyIn1e51erg (+0.33), BlackHoleRadiativeEfficiency (+0.24)**, IMFslope,
  wind momentum, quasar threshold; the spurious wind-travel knobs drop out.
  Variance split: feedback = **34%** of main-effect variance (intrinsic assembly
  dominates). Suite **varies 30 astro params at fixed cosmology**, so the 5
  cosmology params are *placebo* regressors (guarded standardization → exactly
  `β=0`, max |β|=0.000), framed as placebo not a varied null.
- Lesson: marginal corr of a design-mean over 256 noisy points is confounded by
  inter-parameter collinearity and can promote spurious drivers; the
  same-halo-across-designs estimator controls for it and beats noise by √N_h.

## 2026-06-04 — Field-vs-profile paper notebook (`analysis/observable-fb-map`)

Synthesized the observable→f_b work into `paper.ipynb` (built by
`tools/build_paper_nb.py`, executed with the `torch3` kernel): a paper-style,
rigorously-argued case for why BIND must be a **field-level** generative emulator
rather than a profile→f_b regression. Academic style throughout (scienceplots
`['science','no-latex']`, no titles, no hand-set fontsizes); figures → `figures/paper/`.

- Framing: any summary is a functional `O[x]`; a profile is the azimuthal-average
  operator `A_r`; the regression emulator learns one object, `E[f_b | A_r x]` — the
  first moment of a single marginal of `p_θ(x|c)`. Six live-computed demos
  (`ceph/sobol_ss_cv/`): (A) the foil works — stacked Y→f_b OOF `R`≈0.82/0.81/0.66
  (Y), 0.90/0.87/0.79 (+kSZ); (B) intrinsic `σ(f_b)` only 20/16/2% reduced by Y →
  irreducible stack covariance; (C) **beam/instrument** is a 2D operator — isotropic
  beam commutes with `A_r` only for centered/axisymmetric halos, so per-halo
  axisymmetry error reaches 43% (vs small stack mean) and the aperture-Y shift
  spread is 16.6/7.7/3.8% by mass; beam-aware field forward model holds `f_b` RMS
  ≈5×10⁻³ where naive degrades to ≈40 (reuses `beam_aware_results.json`); (E)
  flux selection biases stacked f_b +2.8/+1.4% (Eddington); (F) corr(Y,τ)≈0.96 →
  probes near-redundant, joint covariance required; morphology corr +0.37 with
  BlackHoleRadiativeEfficiency — feedback info `A_r` discards.
- Honest reframings baked in: kSZ τ is a gas-mass anchor (not independent);
  morphology is a feedback-sensitive observable, not a strong f_b predictor;
  multi-probe message is redundancy, not added independent power.



Tested whether BIND observables predict the unmeasurable baryon fraction f_b
*without estimating feedback θ*, on the existing 256-design × 1111-halo Sobol
factorial (`ceph/sobol_ss_cv/cube.npz`). Held-out-θ via `GroupKFold(design)`.

- `tools/observable_fb_reduce.py` — derives SX (X-ray ∝ Σ Gas²√T), τ (kSZ gas-mass
  proxy), f_star, **f_b** from the 7-channel maps in the *same* R200 aperture as
  `sobol_ss_generation.reduce_design` (self-check: Y200/f_gas reproduce the cube
  to 0). → `ceph/sobol_ss_cv/obs_fb_extra.npz`.
- `tools/build_observable_fb_nb.py` → `observable_fb_map.ipynb`; plan in
  `docs/observable_fb_poc.md`, results in `docs/observable_fb_results_2026-06-03.md`.
- **Finding (⚠️ provisional):** map is well-posed as a *vector*, not from Y alone.
  σ(f_b|Y,SX) is ~half σ_marginal at logM 13–14 (−57/43/22%); full Y,SX,T,S,P
  reaches −64/56/49%. **Y alone is weak (−26/16/9%)** and degenerate — SX (n_e²√T)
  is the degeneracy-breaker, beating the τ gas-mass proxy (so not circular).
  Untested: off-grid model, BIND-vs-CAMELS-truth f_b, observational systematics,
  cosmology marginalisation.
- **Profile edition.** `tools/stack_profiles_reduce.py` → field-level per-(design,
  bin) stacked profiles Y(r), SX(r) (= Gas²√T per-halo *then* stacked), gas-weighted
  T/S/P(r), projected f_b(r)=Σ_b/Σ_tot, 12 pixel-based radial bins (57–1595 kpc/h) →
  `stacked_profiles.npz`. `tools/build_profile_fb_nb.py` → `profile_fb_map.ipynb`
  (reviewer-mode docs: Sobol design, emulator provenance, stacking, Ridge method).
  Predict f_b(r) profile from observable profiles, multi-output Ridge, KFold over
  256 designs. Results `docs/observable_fb_profile_results_2026-06-03.md`.
  **Finding:** at the population/profile level Y is a STRONG predictor (69/73/67%
  reduction) — opposite of the per-halo scalar where Y was weak — because stacking +
  shape disambiguate feedback. Profile beats integrated scalar (54%→79%, low bin).
  Full Y,SX,T,S,P → 75–80%; reduction peaks ~90% at 100–200 kpc (feedback core),
  fades to f_cosmic outskirts. α-robust. §3.5 joint-distribution view
  (`prof_joint_*`): 256 designs at fixed (r,mass) = sample of p(f_b,Y); Ridge =
  conditional mean; ρ(Y,f_b)=0.94 @141 kpc, loosens core/outskirts.
- **Real-data validation** (the key test). `tools/stack_profiles_truth.py` stacks
  the 27-sim CV **hydro truth** (`full_maps.truth_maps`=[DM,Gas,Stars] cut at halo
  centres — crop validated to reproduce stored DMO `condition` to corr=1;
  `truth_thermo_patches`=[y,T,S,P]) into truth Y/SX/f_b(r) →
  `truth_stacked_profiles.npz`. `tools/build_realdata_nb.py` →
  `profile_fb_realdata.ipynb`: train Ridge on 256 BIND designs, predict CAMELS-truth
  f_b(r) from truth observables + realistic log-normal obs noise. Results
  `docs/observable_fb_realdata_results_2026-06-03.md`.
  **PASSES:** truth (Y,f_b) lands on the BIND conditional mean (`real_truth_on_joint`);
  Y+SX predicts truth f_b(r) to RMS ~0.004 (≈4% of range), recovers the truth
  evacuated core (not the BIND mean); robust to 20% obs noise. **Honest finding:**
  the full Y,SX,T,S,P (best in-distribution) transfers WORSE to truth (overfits the
  BIND→truth gap, esp. sparse top bin) → Y+SX is the robust real-data set. Caveats:
  single feedback point (interpolation), shared DMO halos, projected f_b, proxy SX.
- **Projection/LOS sanity check** (`projection_fb_check.ipynb`,
  `docs/observable_fb_projection_2026-06-03.md`, `proj_*.png`): the stacked f_b(r)>cosmic
  at ~400 kpc is NOT a projection artifact — `fm_testsuite_cube` 6.25 Mpc/h truth
  (`truth_halos_cube.npz`, same halos) gives ~identical differential f_b (LOS effect
  <0.02, only r≳600 kpc, dilutes toward cosmic). It's the DIFFERENTIAL profile
  overshooting (gas pushed out of core); ENCLOSED f_b(<R200)=0.11/0.15/0.16 stays below
  cosmic (missing baryons, deficit largest at low mass). tSZ Y IS LOS-sensitive (flattens
  to uniform bg at large r) unlike mass-f_b; observers use CAP (disk−annulus)/matched
  filter/deprojection — demo'd. Next data-gen: 6.25 Mpc/h THERMO cubes to quantify Y's LOS.
- **CAP-filtered validation** (`profile_fb_cap_validation.ipynb`,
  `docs/observable_fb_cap_2026-06-03.md`, `cap_*.png`): re-ran BIND→truth with
  Compensated-Aperture-Photometry observables (CAP(θ)=disk−√2-annulus, nulls uniform LOS
  bg; computed exactly from profiles+pixel counts, no re-reduction). **Validation holds**:
  CAP predicts truth f_b(r) to ~0.005–0.007 (vs raw 0.004), modestly worse because CAP
  discards the absolute zero-point an observation can't measure anyway → result no longer
  depends on unmeasurable Y normalisation (more credible for real data). **Hypothesis
  overturned:** CAP is NOT more noise-robust — per-measurement error grows ~linearly for
  both raw & CAP (CAP slightly worse); CAP buys robustness vs background, not per-bin
  noise. Used honest median-per-measurement metric (RMS-of-mean was a regression artifact).

## 2026-06-03 — Two-stage paint (CPU/MPI project → GPU generate); fixes TNG-box OOM

`run_paint_tng.sh` (one-shot `bind.paint` on one A100 node) OOMed: the box load
path (`io_gadget.read_dmo_particles`) concatenates **all** ~33 GB of TNG300-Dark
particle positions on one process, then `Simulation.project()` masks them per
z-slab. Split painting into two SLURM jobs so neither holds the full particle set:

- **Stage 1 — `bind.cli.paint_project` (`run_paint_tng_project.sh`, CPU/MPI).**
  New `bind.inference.paint_stages.project_and_extract`. Each MPI rank reads only
  `files[rank::size]` snapshot chunks, **streams them one at a time** into the
  small z-slab maps (~70 MB/slab), then partials are `MPI.SUM`-reduced onto rank 0,
  which reads the FoF catalog, extracts per-halo DMO cutouts, and writes
  `stage1_slab{NN}.npz` + `stage1_manifest.json`. Streaming alone fixes the OOM
  (peak ≈ one chunk + slab maps); MPI just adds multi-node speed. Runs serially
  with 1 task / no mpi4py.
- **Stage 2 — `bind.cli.paint_generate` (`run_paint_tng_generate.sh`, GPU).**
  `paint_stages.generate_from_stage1` loads the intermediate, runs the sampler,
  composites, and writes the same `composite_slab{NN}.npz`/`summary.json` as
  `bind.paint`. Light on memory; no particle I/O.

**Gotcha (cost real time if forgotten):** Pylians `MASL.MA` is **not** additive
onto a pre-filled field — calling it repeatedly to accumulate chunks silently
*loses mass* (measured ~25%). Deposit each chunk into a fresh zero field and sum
with numpy (`_accumulate_chunk_into_slabs` reuses `_project_zslabs` per chunk).
Verified: streaming == production one-shot to float precision, mass conserved.

**Multiscale physical-scale bug fixed (the important one — correctness, not just a
crash):** the network's 4 input channels are fixed *physical* scales
`[6.25,12.5,25,50] Mpc/h` (training: `data_generation/process_simulations2_cpu.py`
`extract_multiscale_cutouts`, `scales_mpc`). But inference `pipeline.extract_multiscale`
used fixed *pixel* scales `[128,256,512,full_res]` — so its 4th (largest) context
channel was the **whole box**: fine at the native 50 Mpc/h / npix=1024 grid, but on
the 205 Mpc/h TNG box it became a 205 Mpc/h window (OOD for the UNet) **and** crashed
(`npix=4198` not a multiple of 128 → reshape `ValueError`). The first real TNG stage-1
run hit exactly this — *after* a fully successful MPI projection (64 ranks, 92.7s, 2951
halos at M>1e13, confirming mpi4py works). Fix: `extract_multiscale` now takes
`mpc_per_pix` and cuts the `MULTISCALE_MPC=(6.25,12.5,25,50)` windows (capped at the
box), resampled to 128² via `_downsample_square` (exact block-mean when divisible —
**CAMELS box=50/npix=1024 bit-for-bit unchanged** — else area-avg down / bilinear up).
Network inputs stay 128² at the trained scales; only the slab *background* spans the
full box, which is correct (that's where patches get pasted). Both
`extract_halo_cutouts` callers pass `mpc_per_pix=box_size/npix`.

**Stage 3 — re-composite without regenerating (`bind-paint-recomposite` /
`recomposite_slab` / `recomposite_from_saved`).** The sampler output is already
saved (`generated_patches` per `composite_slab*.npz`), so re-blending with new
`taper_frac` / `r200_factor` / `patch_mass_match` re-runs only
`build_bind_composite` — **no GPU**. `recomposite_slab(stage1_npz, generated_npz,
**settings)` returns the bundle for interactive notebook sweeps; verified
idempotent (same settings reproduce the saved composite to max|diff|=0) and
mass-conserving under a circular `r200_factor` paste. Generate now also writes
`condition_sums` (per-halo cutout mass) so future composites are self-recompositable.
Shared `_save_composite_slab` helper used by both generate + recomposite.

**Validated on the real TNG300-Dark box (snap 099).** Stage 1: mass conserved
*exactly* (projected = `pmass × 2500³` to ratio 1.0000), Ωm=0.3090 from the maps
(fiducial 0.3089), 2951 halos M200c 1.0e13–1.0e15 (154 >1e14), R200↔M200c
consistent, cutouts centered on density peaks (98–99%), multiscale context at the
correct [6.25,12.5,25,50] Mpc/h. Stage 2 (fm_two_head, 50 steps, ~5 s/16-halo
batch on one A100): composite mass-conserved per slab, DM_hydro reproduces the
DMO web, Gas/Stars painted only in halos, f_b≈0.10 (feedback-depleted, below
cosmic 0.157). Notebook `examples/paint_tng_results.ipynb` does these checks +
figures; stage-2 cells are race-safe (`safe_load`) so they populate as slabs land.
§7 computes the **matter-power suppression**: sum the z-slabs (masses additive) →
full-box DMO + painted (DM+Gas+Stars) grids → 2D `Pk_plane` ratio (capped at
`k_Nyq=π·npix/L`). Textbook curve: S→1 for k<2, knee at k~3, **~17–20% suppression
(S≈0.80–0.83) by k~20–40 h/Mpc** — consistent with TNG AGN feedback (from the
M>1e13 painted population). Runs in the `bind_env` Jupyter kernel
(`python -m ipykernel install --user --name bind_env`).

**Env:** runs in `~/venvs/BIND_env` — a Python-3.11 `--system-site-packages` venv
built from the module python view (inherits the view's numpy 2.2.4 / torch 2.6
cuda12.5 / h5py; only `bind` + Pylians are pip-installed locally). mpi4py comes
from the `python-mpi` module (no build), matching the view's numpy, so stage 1
loads `module load python openmpi python-mpi` (same modules at create + run time
so PYTHONPATH exposes mpi4py); fallback `module load openmpi && pip install mpi4py`
into BIND_env. Stage 2 just activates BIND_env (torch bundles CUDA). Submit gated:
`jid=$(sbatch --parse run_paint_tng_project.sh);
sbatch --dependency=afterok:$jid run_paint_tng_generate.sh`.

---

## 2026-06-02 — New branch `analysis/pk-decomposition`: field-level halo-masking decomposition of P(k) suppression

Started the "which halos drive the matter-power suppression?" experiment (the
natural next paper after the `scaling_relations` figures). Field-level halo
masking: for each of the 256 Sobol designs, re-composite halo subsets (3 mass
decades × 3 gas-fraction-at-fixed-mass terciles) back into the 50 Mpc/h box and
measure the partial S(k) — reusing the precomputed BIND patches
(`sobol_ss_cv/maps/`, **no re-emulation, no hydro**). New CPU-only tools:
- `tools/partial_supp_sobol.py` — array-ready/resumable; per design stores
  `S_full`, `S_sub` (subset-only paste), `S_loo` (leave-one-out). `full_pk`
  reproduces `box_supp_sobol` `S_true` exactly.
- `tools/fig_partial_supp.py` — variance attribution + 4-panel figure.
- `run_partial_supp.sh` — 16-way CPU SLURM array (**user submits**; org policy).

**Methodology (validated on a 32-design thin slice).** Subset-only paste
OVER-counts a subset's contribution (~1.8×: mass-renorm + patch-overlap
cross-term); LOO UNDER-counts (~0.55×). Their mean — the 2-bracket Shapley
contribution `c_s = ½[(S_sub−1)+(S_full−S_loo)]` — is ~additive (Σ_s c_s ≈
S_full−1, slope 1.11, Σ shares ≈ 115%), giving defensible **absolute** Var[S]
shares via `share_s = Cov(c_s, ΔS_full)/Var(ΔS_full)`. A naive LMG/regression
split washes out to uniform ~11% (the 9 subset partials are collinear — all driven
by the same knobs); use the covariance/Shapley partition, not LMG.

**Prototype result** (k=10, n=32; full 256 pending user Slurm): group decade
**[13,13.5) ≈ 48% of Var[S]**, [13.5,14) ≈ 30%, clusters [14,15) ≈ 21% (the last
from only **51 halos** — high per-halo weight; ~0 mean contribution though). At
fixed mass, **gas-rich halos dominate the variance** (~43–49%) over gas-poor
(~25–30%) — gas content, not just mass, structures Var[S] (direction 2 confirmed).
High WindEnergy *deepens* the group-decade contribution (group dominance
strengthens, doesn't shift to clusters — direction 1). Stable k=3↔k=10. Direction
3 (beyond-R_vir redistribution radius vs θ) not yet built.

**Publication-grade notebook** `pk_suppression_decomposition.ipynb` (Question →
Methods → Results → Discussion, **11 figures**, executed 0 errors on the prototype;
auto-upgrades to the full cache when present). Built to disarm a skeptic: Fig 1
motivates the variance + **honestly places the fiducial** (it is the *8th-pct
strong-feedback tail*, not the floor — fixed a misleading "typical member" framing
after human pushback); Fig 2 pure validation (masking composite == independent box
pipeline to **machine precision**, all designs & k); Fig 4 the Shapley-additivity
credibility plot; Fig 5 headline heatmap; Figs 6–7,9 results w/ bootstrap CIs;
**Fig 8 the param→halo sensitivity map** + **Fig 10 the feedback-strength-axis
robustness** (the 30-D treatment: ~4 of 30 params drive S(k), group/gas-rich
dominance holds weak→strong — replaced the misleading single-knob split); Fig 11 the
actionable synthesis (group $f_{\rm gas}$ most constrains the WL baryonic prior); §5
referee Q&A table. Branch carries the uncommitted `scaling_relations` fixes too;
nothing committed yet.

**Full 256-design run + mechanism check (2026-06-02 eve).** Ran the campaign
(`run_partial_supp.sh` → `--reduce` → `partial_supp_sobol.npz`, 256 designs, clean);
notebook auto-upgraded. Results **robust** vs the 32-pt prototype: by mass
57/35/21% of Var[S(k=10)] (groups/[13.5,14]/clusters), by gas 32/36/44%
(poor/mid/rich), additivity slope 1.12, drivers IMFslope/BHRadEff/WindEnergy
($R^2{\sim}0.7$). **Honest mechanism check (Fig 11) overturned the naive
"swing-voter" reading I'd floated:** at the halo level the per-patch suppression
swing is *uncorrelated* with gas ($\rho{=}0.00$, even at fixed mass). The gas-rich
excess is a **reservoir/magnitude** effect — all subsets are ~coherent with the total
($r{\sim}0.99$) so variance tracks contribution *magnitude* ($\rho{=}0.97$), and it is
**group-scale only** (gas rich/poor var ratio 2.68/1.72/0.89). The genuinely
variance-specific result is **clusters: $\sim$0 mean contribution but $\sim$21% of the
variance** (6× per-halo). Re-framed the Discussion accordingly + fixed a 3× scale bug
in Fig 6B per-halo. Figures `paper_figures/pk_decomp_fig{1..12}_*.png`.

## 2026-06-02 — scaling_relations.ipynb: fixed silent halo-ordering bug + unified into one 4-act story (assembly dropped)

Two-part pass on `scaling_relations.ipynb` (still on `main`, the human's active
notebook; per convention belongs on a topic branch). **(1) Fixed a silent
correctness bug:** the CV scaling-relation loader (cell 10) iterated sims in
*numeric* order (`sim_0,sim_1,sim_2,…`) but the Sobol cube stores its 1111 halos
in *lexicographic* dir order (`sim_0,sim_1,sim_10,…`; sim_17 dropped/no-radii,
sim_27 absent). Both total 1111, so the old `assert len==1111` passed while
per-halo identity was scrambled — every "same halos across designs" result was
wrong. Loader now iterates `sorted(CV_ROOT.iterdir())` (== cube order) and a hard
`np.allclose(_cube['M200'],halo_mass)` assert enforces it (mirrors
`tools/box_supp_sobol.py`). **(2) Made it run + tell one story:** restored the
missing Sobol-cube loader cell (was a NameError cascade:
`_cube/D_norm/ASTRO_NAMES/OBS_NAMES/N_HALO/SUPP/spearmanr`), defined `hi_m/lo_m`,
trimmed the dangling assembly/`S_massonly` NOTE. Rewrote the narrative as a 4-act
arc on two pillars (same halos ⇒ cosmic variance differenced away; field-level
emulator ⇒ paste→P(k)): I Fidelity (Figs 1–2) → II structured scatter / two
populations (Fig 3) → III controlled feedback experiment (Figs 4–5: MI of the 30
knobs on the fiducially-classified gas-rich/poor tails) → IV power spectrum (Fig 6
design fan + new **Fig 7** capstone). Fig 7 (`perhalo_box_synthesis`) shows the
*same* knobs (IMFslope, WindEnergy, BHRadEff) drive both per-halo gas content and
box S(k=10), and design-mean log f_gas predicts box S(k=10) at ρ=+0.54 (p~1e-20).
**Assembly (DMO history) dropped per the human** — this supersedes the prior
entry's assembly "Fig 4"; that analysis is not in the current notebook.
Re-executed end-to-end (torch3 kernel + gcc-13 libstdc++ on `LD_LIBRARY_PATH`),
0 errors, 7 figures.

## 2026-06-02 — scaling_relations.ipynb: "assembly as a hidden 2nd parameter of feedback" (new Fig 4)

Mined the Sobol feedback cube (`/mnt/home/mlee1/ceph/sobol_ss_cv/`: `cube.npz`
256 designs × 1111 CV halos × 8 obs, common-random-noise; `pk_supp_extra.npz`
supp at k=5/10/band; `assembly_table.npz` 3D DMO assembly c_V/λ/σ_v/rhalf/z_form)
+ assembly histories. Added Figure 4 to `scaling_relations.ipynb` (on `main` —
the notebook the human is actively editing; flagged that per convention this
belongs on a topic branch).

Result (executed, fig written to `paper_figures/assembly_feedback_susceptibility.{pdf,png}`):
per-halo OLS response of P_hydro/P_DMO to the 30 astro knobs (median R²=0.68);
dominant drivers BHRadiativeEfficiency / IMFslope / WindEnergy. **At fixed mass
the *mean* suppression is ~assembly-independent (|ρ|≲0.07), but the
*susceptibility* (response derivative) is significantly set by assembly**: early-
forming/concentrated/compact halos resist feedback — z_form ρ=−0.15 (p~1e-6),
c_V ρ=−0.15 (p~1e-7), rhalf ρ=+0.16 (p~1e-7); k=5 even stronger (ρ=−0.26,
p~1e-18). Mass-matched tercile split: high-conc ~10% less susceptible in 8/9
bins. Framed honestly as hypothesis-grade (emulator + 2D + CV cosmology; modest
ρ); motivates a direct 3D hydro/DMO P(k) split-by-formation-time test. The ICM-
observable analogue already exists on `analysis/tsz-icm`
(`assembly_feedback_susceptibility.ipynb`); this is the matter-power / weak-
lensing version.

## 2026-06-01 — Mutual-information notebook: rebuilt twice to per-sim, null-calibrated + expanded viz

`examples/mutual_information.ipynb` (split out of `paper_figures.ipynb`). Went
through TWO corrections driven by the human's skepticism, both proven in-notebook
(executed end-to-end, 0 errors, 16 figures, 4.1 MB; torch3 venv — from a bare
shell needs `LD_LIBRARY_PATH` → a gcc-13 libstdc++).

1. **Per-sim-means → per-halo** (fixed small-N noise). Then the human flagged a
   strong, unphysical dependence on **UVBHepDeltaz** (HeII-reion redshift width)
   on z=0 stellar mass. Diagnosis: **per-halo MI is pseudo-replicated** — only
   ~101 independent parameter draws (Test/SB35 LH) but ~4272 halos broadcast the
   same params, so KSG reports a ~0.4-bit (Stars)/~0.15 (Gas) *phantom floor on
   every parameter*. UVBHepDeltaz was pure floor.
2. **Per-halo → per-sim, null-calibrated** (the correct fix): aggregate halos →
   per-sim statistic (N=independent sims), report **excess over a shuffle-null**
   with 3σ significance + bootstrap error bars. UVBHepDeltaz excess → **0.000**
   under both per-sim and a block-preserving per-halo null; the surviving signals
   are physical: **Ωm→DM ≈1.79 bit, Ωb→Gas ≈0.79, VariableWindVel/σ8→Stars**.
   BIND reproduces the real excess (Ωb→Gas: truth 0.79 / BIND 0.83). The
   independent unit for parameter MI is the **simulation, not the halo**.

Also expanded from single colorbar heatmaps to a full battery (per the human's
request): §7 per-param profile (bar/sorted/cumulative — 80% of info in ~10
params), §8 param–param MI (LH independence check) + interaction info
(synergy/redundancy) + graph, §9 pointwise/specific information, §10
compressed-rep MI (PCA latent×param + t-SNE), §11 field-space per-pixel MI maps
(Ωb→Gas shows a feedback-regulated central hole; truth≈BIND). Multivariate KSG
estimator added (`ksg_mi`). Caches under `examples/paper_figures/mi_cache/`
(`halo_features_<model>.npz`, `sim_stacked_patches_<model>.npz`).
`paper_figures.ipynb` MI cells left in place. Per branch convention may belong on
`analysis/*`.

## 2026-05-27 — Repo hygiene, branch reorganization, and agent instructions

**Repo cleanup.** The repo had no `.gitignore`, so ~304 untracked items
(2 GB of caches/outputs/figures, committed `.pyc`) were noise. Added a
`.gitignore` (caches, `outputs/`, figures, `*.npz`/`*.npy`/`*.log`, pycache,
notebook checkpoints, machine-local `.claude/settings.local.json`), untracked
the committed `.pyc` files, and refreshed the tracked paper figures. Untracked
count: 304 → 0.

**Branch reorganization.** Decision: keep `main` a clean trunk and park distinct
analyses on topic branches instead of dumping everything on `main`.
- `main` — core engine (`data/model/train/metrics`, `test_suite/`) + the
  ~890-line engine evolution since the last working-model commit + refreshed
  `paper_figures.ipynb`.
- `feature/3d-cube` — 3D / cube-projection extension.
- `analysis/2d` — scatter package, observables, `project1-7`, CV derivatives.
- `wip` — scratch notebooks, parameter-injection experiments, planning notes.

Notebooks are committed with outputs (per preference). No git remote — local-only.

**Agent instructions.** Added `CLAUDE.md` (architecture + commands + conventions
+ data caveats), this `docs/WORKLOG.md`, and `.github/copilot-instructions.md`
mirroring the project context for GitHub Copilot. Then merged `main` into each
topic branch so they all carry the shared docs, and appended a tailored
`## This branch: …` section to `CLAUDE.md` + the Copilot file on each
(`feature/3d-cube`, `analysis/2d`, `wip`) describing that branch's projects.
`main`'s copy stays generic.
