# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

---

## 2026-07-15 — feature/wl-emu: community WL statistics emulator (`bind.wlemu`)

The paper's community-tool section: `bind.wlemu.WLEmulator` maps the 30 SB35
astro params (unit cube or physical dict) + source z ∈ {0.5,1,1.5,2,2.44} to
all κ summary statistics of a 5×5 deg 1024² field (Cl/pdf/peak/min/V0–V2/
scattering/moments = 383 dims) with GP σ and a single-field covariance.
Trained on `ceph/wlemu_cache_1024/stats_cache.npz` (253 Sobol pts × 50
noise-paired realizations × 5 z; raytraced BIND-baryonified TNG-DMO
lightcones, built in the `kappa_emu` sibling project). Design = the winner of
kappa_emu's 9-design shootout (per-z PCA-32 + batched exact ARD-Matérn-5/2
GPs; beats the field-level FM map generator 5–100×/statistic); the port
reproduces the historical fixed-split table *exactly* and the numpy predictor
matches gpytorch to 7e-11. **Inference is numpy-only** from the committed
artifact `src/bind/assets/wlemu_gp.npz` (3.3 MB); refits via
`python -m bind.wlemu.fit` (gpytorch, `bind[wlemu-fit]`). Validation =
13-fold CV over all 253 pts (`examples/data/wlemu_validation.npz`, committed):
frac err 0.09–2% (peaks 5%, minima 7% — empty-tail dominated), median
|err|/SEM 0.09–0.7 (typ. ≈0.2); GP σ approximately calibrated (68%→54–69%);
known systematic: last Cl bins (ℓ≳10⁴) ~+2 SEM, covered by GP σ. Toy
2-param inference (ASN2 × BH rad. eff., Cl+peaks, Hartlap + GP-var budget):
11/12 held-out points inside the 68% contour. Tutorial
`examples/wlemu_tutorial.ipynb` (executed; generator
`examples/_build_wlemu_tutorial.py`), docs `docs/wl_emulator.md`, CLI
`bind-wlemu`, sample maps `examples/data/kappa_sample.npz`. Gotchas learned:
realizations are noise-paired across runs so SEM-based dilution/reliability
math misleads (truth cross-stat corrs only mildly diluted); the naive
max-bin/σ sensitivity ranking is contaminated by zero-variance bins;
peak-count information in a 50-realization covariance is eaten by the
Hartlap factor. Pre-existing ruff violations on the integration branch
(16, none in wlemu) left untouched.

## 2026-07-06 — Paper restricted to the trained regime (≥1e13); hand-edits folded back into the notebook builder

Decision: the paper drops the low-mass (<1e13) halos entirely — all analysis now
uses the trained regime $M_{200c}\ge 10^{13}$ only. `examples/_build_paper_nbs.py`
first absorbed every hand-edit that had accumulated in `paper_figures2.ipynb`
(new Fig 1 layout without residual row, consolidated Fig 2 scatter+distribution
per bin, KS/median-bias table, Fig 3a-bis per-bin parameter correlations, pk.npz
diagnostic, pk_fixed Fig 5, §6 shape-illustration cell, `eps`-metric filter), so
**regenerating is safe again** — the builder is the source of truth. Then the
≥1e13 restriction: shared SETUP defines `BINS` (mass bins ≥13.0) and every per-bin
loop (main + thermo T4) iterates over it; §7 covering-paint section and the
`paper_fig_lowmass.ipynb` appendix were removed from the builder (the generated
lowmass ipynb is left on disk, delete when ready). Fig 5 P(k) now shows only the
BIND ≥1e13 shared-paste line, and `tools/paper_cache/build_pk_fixed.py` gained an
`hr_ge13` control — truth patches of the *same* ≥1e13 halos, same shared paste —
replacing the legacy all-catalog (≥1e12) hydro-replace; stale partials are
recomputed automatically and `--reduce` tolerates missing keys. Validated on
CV/sim_0 (hr_ge13/truth ≈ 1.04 at k=40–64 vs BIND 1.16); full rebuild
(`build_pk_fixed.py --metric fixed --pool N` then `--reduce`) still to run.

## 2026-07-02 (later) — T6 outlier physics: +1σ ridge diagnosed as aperture contamination; section rebuilt on clean halos

Why does the T6 corner plot correlate strongly only in the **+1σ** tails? Diagnosis:
the eval maps are full-box-depth projections, so R200c apertures also sum every
*projected* neighbor — an additive (strictly positive) boost lifting M★/Mgas/Y
coherently. Joint (+SHMR,+Y–M) outliers have median external catalog mass ≈ 1×M200
(70% >0.2×; half have a more-massive projected companion; only ~16% 3D-associated →
mostly chance LOS superposition); joint negative outliers have none. T6 was rebuilt
on **clean halos** (M_ext < 0.2 M200, drops ~6%): fits tighten (σ_SHMR 0.13→0.09,
Y–M 0.19→0.14) and two new figures added — `figT6c` (contamination diagnostic:
r_SHMR vs r_Y–M colored by aperture M_ext; ρ₊ = +0.84 contaminated vs +0.26 clean,
ρ₋ ≈ 0) and `figT6d` (surviving physics: star-rich halos are *gas-rich* at fixed
M200 → SZ-bright, gas-mass-mediated ρ(r_Mgas, r_Y)=0.82; star-poor halos are NOT
gas-poor → −1σ tails decorrelate; BIND reproduces the tail-conditional structure).
Caveat recorded: ALL R200-aperture quantities in the pipeline carry this projection
contamination — filter via catalog projected neighbors for clean-halo statements.

## 2026-07-02 — T6: joint mass+SZ scaling-relation scatter & residual corner plot (from analysis/2d)

Promoted the `analysis/2d` `scatter.ipynb` figures (residual-colored scaling
relations + ±1σ outlier residual corner plot) into the paper pipeline as the
previously missing **T6** section of `examples/paper_fig_thermo.ipynb`, extended
with the SZ relations **Y–M200** and **Y–T** (both in the scatter grid and as
corner-plot dimensions). Pure load-only: joins the existing `mass_table.pkl` +
`thermo_scaling.pkl` caches per halo (row-aligned by (suite, sim, row), asserted
via the shared log-mass column) — no new cache builder. CV suite, ≥1e13
(`C.PARAM_RESPONSE_MASS_MIN`). Cells inserted **in place** into the existing
notebook (outputs preserved; `_build_paper_nbs.py` deliberately *not* rerun — user
has custom edits in `paper_figures2.ipynb`); the generator source was updated to
match so a future deliberate regen keeps T6. Headline numbers (truth vs BIND):
Y–M α 1.94/1.87 σ 0.19/0.21; Y–T α 2.49/2.41 σ 0.14/0.14 (self-similar 5/3, 5/2);
BIND reproduces the truth outlier ridge in r_Y–M vs r_Mbar–M200. Note: this dev
suite's Test/SB35 *does* have thermo truth in `thermo_scaling.pkl` (truth-thermo
projection was kept on), though T6 uses CV only.

## 2026-07-02 — Fig 5 high-k P(k) suppression diagnosed: overlap-averaging paste artifact (+ SB35 content deficit)

The new ≥1e12 Fig 5 (`pk.npz`, fm_lowmass suite) rolls off at k>30 (CV −3%,
SB35 −13% vs truth at k40–70) where the old main-branch ≥1e13 circular figure was
near-perfect (reproduced on `fm_testsuite/fm_two_head`: 1.002). Full decomposition
(mass subsets × content × paste mode, 57 sims) in
`ceph/paper_cache/fm_thermo/pk_diagnostics/` (scripts + partials + figures):

- **Dominant artifact — the paste, not the low-mass content.** `paste_halos_2d`
  weight-*averages* overlapping patches; with the 1e12 floor 30–50% of painted area
  is multi-covered, and averaging *independent* FM realizations destroys their
  stochastic small-scale power: −10% (CV) / −12% (SB35) at k40–70, Spearman −0.75
  vs multi-covered fraction across sims. The hydro-replace control is immune
  (overlapping truth patches are identical pixels), which is why it stays flat.
  1e12–1e13-only content is fine (b/hr ≈ 1.00 CV).
- **Secondary, honest signal — SB35 content deficit.** ≥1e13 BIND/HR content ratio
  at k40–70: CV 1.011 [0.96–1.07], Test 0.927 [0.84–0.99] (worst sim 0.49). Not an
  n_steps artifact (20 vs 50 vs 100 identical within realization noise) — genuine
  extrapolation error; already hinted in the old SB35 panel's droop+wide bands.
- **Fix validated — covering-style shared content.** "Adoption repaste" (greedy set
  cover over the *existing* `generated_halos.npz`; satellites adopt the host patch's
  realization; aperture-local mass match; CPU-only) ≈ fresh covering paint (GPU) and
  removes the averaging loss: CV 0.969→1.06 (4R200) / 1.04 (Rc), SB35 0.872→0.94 at
  k40–70; SB35 mid-k 0.963→1.00. Exclusive first-wins paste also works but has
  ordering/seam artifacts (+8% CV). Remaining deviations = model content (the same
  ~+8% mid-k CV hump the original figure had).
- **Engine fix (same day, follow-up):** shared-content paste is now the package
  default. `bind.inference.pipeline.share_overlap_content()` + `paste_mode`
  {"shared" (default) | "average" (legacy, bit-identical to old composites)} on
  `build_bind_composite`, wired through `RunConfig`/runner, `bind-camels-suite
  --paste_mode`, `bind.paint()` (whose `r200_factor` default also fixed 0.0→4.0),
  `bind-paint`, and all three `paint_stages` entry points; documented in
  `docs/circular_aperture.md` ("Shared-content overlap handling") + CLAUDE.md.
  In shared mode `patch_mass_match` is aperture-local (covering-paint standard).
  Validated: `average` reproduces cached composites bit-for-bit; `shared` matches
  the adoption diagnostic to 0.03%.
- **Corrected Fig 5**: `tools/paper_cache/build_pk_fixed.py` (resumable, env-flips
  to the fm_redshift spine like the other builders) computes per-sim P(k) for
  shared-paste ≥1e13 / ≥1e12 composites (CPU) and the TRUE covering paint (GPU:
  all FoF halos ≥1e10, fresh generation per set-cover box, Rc closure apertures
  with per-sim f_cos=Ω_b/Ω_m, gas background; CAMELS 1P N-body routing for the
  shared-DMO 1P sims) over CV+Test+1P; `--reduce` → `pk_fixed.npz` (+ hydro-replace
  stacks carried over from `pk.npz`). Notebook cell:
  `tools/paper_cache/fig5_pk_fixed_cell.py` (paste into `paper_figures2.ipynb`).
  Two data-handling gotchas fixed en route: (1) f_cos must be per-sim — SB35
  spans Ω_b/Ω_m ≈ 0.06–0.68 (a first-pass clip at 0.35 silently distorted 9
  low-Ω_m sims); (2) 1P astro-param `n1`/`n2` sims fail `_resolve_1p_nbody_sim`'s
  int parse and need the build_1p_specs-style fallback to the `1P_p1_0` DMO FoF.

---

## 2026-07-01 — `integration/paper-figures`: parallel cache pipeline + load-only notebooks; reframe around one model

Reworked the paper-figure workflow from a slow, all-inline `paper_figures2.ipynb`
(re-loads the same suite `.npz` and re-runs per-halo loops / Pylians FFTs in every
figure; narratively a mass-only high-mass model with low-mass/thermo/redshift/VDM/
observable variants tacked on) into **parallel cache builders + load-only notebooks**,
reframed around **one model** (mass + thermo as a function of redshift & parameters,
`fm_redshift` @ z=0) validated **by halo-mass bin**. User decisions: spine =
`fm_redshift`@z=0 (needs one new z=0 suite eval, user submits); lean main + thin
companion notebooks over one shared cache.

- **`tools/paper_cache/`** (new): `paper_config.py` (single source of truth — paths,
  discovery, loaders; env-flip dev→spine), `build_metric.py` (7 resumable per-sim CPU
  metrics + `--reduce`, mirroring `tools/partial_supp_sobol.py`: `mass profiles
  profiles_r200 pk shapes thermo field1p`), `build_gpu_insets.py` (`covering redshift
  vdm obs`), `README.md`. Launcher `run_paper_cache.sh` (local pool **or** SLURM array +
  dependent reduce).
- **Two latent bugs fixed as caches:** Fig 3b (param→profile Spearman, referenced an
  undefined `rho_prof`) is computed in `profiles_r200` reduce; shape **parameter
  response** (absent before) added in `shapes` reduce. P(k) hydro-replace control is
  **reconstructed on CPU** (`build_bind_composite` with truth patches; settings from
  `summary.json`) so it no longer needs the missing `hydro_replace.npy`.
- **4 load-only notebooks** built by `examples/_build_paper_nbs.py`: `paper_figures2`
  (main: §1 showcase · §2 mass+response by bin · §3 profiles+response by bin · §4 field
  1P · §5 P(k) · §6 shapes+response · §7 low-mass covering paint), `paper_fig_thermo`,
  `paper_fig_redshift`, `paper_fig_models` (App A VDM, App B observables). Old
  `paper_figures2.ipynb` (inline) preserved in git history (`f59139a`); `paper_figures2_lowmass.ipynb`
  superseded.
- **Validated end-to-end on the on-disk `fm_lowmass` (fm_thermo, CV/1P/Test ≥1e12,
  identical layout to the spine eval):** all 7 CPU metrics + 4 GPU insets build/reduce/
  load clean; per-sim ~6-12 s (full 268-sim run ~20 min pooled); mass full-patch ratios
  1.00/0.99/1.03; P(k) BIND/DMO 0.99→0.85→0.85; covering 24252 halos→183 gens (133×).
  All 4 notebooks execute headlessly **0 errors, 22 figures**.
- **Two figure fixes from review:** (1) Fig 1 residual used the blended `composite`,
  whose paste-weight `alpha` tapers to 0 at each 4×R200 aperture edge → spurious −1 ring
  around every halo; switched to the taper-free `hydro_canvas`, and now emit both a patches
  variant (`fig1_showcase`) and a full-box variant (`fig1_showcase_composite`). (2) Fig 3a/3b
  parameter response averaged each sim over the full ≥1e12 population → diluted by the
  extrapolation halos (True−BIND residual ~0.2); made param-response **per mass window**
  (`PARAM_WINDOWS` trained ≥1e13 / lowmass 1e12–1e13) for mass/profile/thermo/shape → main
  figs use `trained` (residual ~0.01, all astro params shown). Added `paper_fig_lowmass.ipynb`
  (the 1e12–1e13 breakdown: integrated-mass & profile fidelity + per-window param response).
  Finding: BIND reproduces the low-mass feedback *response* (residual ~0.014); its low-mass
  error is absolute *amplitude* (Gas over-produced), not response.
- **Data caveats found:** Test/SB35 has no truth-thermo in `fm_lowmass` → thermo
  validation is CV+1P (keep truth-thermo projection on for the spine eval's Test suite);
  CV has fixed params so param-response figures populate only with 1P+Test; the notebook's
  old hardcoded `SB35_param_minmax.csv` path is dead — config reads the bundled asset.
- **Remaining (Phase II/III, user):** submit the `fm_redshift`@z=0 suite eval (parameter
  swap of `run_lowmass_suite.sh`), then flip `PAPER_SUITE_ROOT/MODEL_SUBDIR/MODEL_TAG`
  and rerun the cache. See `tools/paper_cache/README.md`.

## 2026-07-01 — `integration/paper-figures`: consolidate all model families for one paper notebook

Goal: one `examples/paper_figures2.ipynb` that generates every paper figure. New
branch off `low_mass_extrapolation`; merged `feature/vdm`, `feature/redshift`,
`feature/observable-conditioning` (union-resolved `docs/WORKLOG.md`; hand-resolved
overlapping additive conflicts in `model.py` docstring, `data.py` `__getitem__`,
`train.py` ×9 — kept both features' hparam-gated paths). **All four checkpoints now
load from one code path** (verified): `fm_thermo` (FM+thermo), `vdm`
(VariationalDiffusion), `fm_redshift` (FM+thermo, condition_redshift), 
`fm_observables_masked` (FM+thermo, condition_observables, n_params=14). Added
paper_figures2 §Fig 8–11: low-mass covering to 1e10 + closure-radius + connected
gas (Fig 8); redshift response z=0–2 (Fig 9, `fm_redshift`); VDM-vs-FM sharpness +
high-k patch P(k) (Fig 10, `vdm` vs `fm_redshift`); observable-conditioned paint on
the M+Y+Tx subset (Fig 11, `fm_observables_masked` via `fb_predict` helpers). All
five sections smoke-tested (load+generate). Consolidation fix: `Model.generate`
passed `scale_factor=` unconditionally (redshift feature), incompatible with the
VDM sampler → now passed only when set. Observable models can't use `Model.generate`
(35-param assumption); use `fb_predict.generate`/`build_observable_vectors`.

## 2026-07-01 — Multi-halo "covering" paint + generated closure-radius aperture (`low_mass_extrapolation`)

Idea: baryonify *many* halos per GPU call by covering the box with the fewest
6.25 Mpc/h generations (greedy geometric set-cover, largest-first; a halo joins a
box iff its 4·R200 aperture fits). New notebooks (+ `_build_*_nb.py` builders) in
`examples/`:
- `offcenter_covering.ipynb` — GPU probe of BIND's off-center validity radius:
  DM/Gas hold to r_max≈1 Mpc/h (<0.1 dex); thermo + low-mass tighter.
- `covering_plan.ipynb` — CPU planner + GPU-call savings; `covering_paint_pk.ipynb`
  — deploy + full-box P(k) vs hydro truth, r_max sweep.
- `covering_lowmass_closure.ipynb` — covering down to **1e10** (all ~24k halos in
  **183 generations, 132×**); **generated closure-radius aperture** (paste each
  halo to where *its generated* f_b(<r)=Ω_b/Ω_m — deployable, no truth) fixes the
  1e10 total-matter P(k) overshoot; **inter-halo gas background** (f_gas·smooth(DMO),
  total conserved) removes the hard gas edges.
- Numbers firmed over **8 CV sims** (1e10 floor): R_c beats fixed 4·R200 by
  +2.9/+3.2 pp at mid/small-k; covering→total-matter P(k) ≈ +0.5/+4.6/+6.8%.
- Gotchas: mass-match MUST be aperture-local (full-128px match on roll-recentered
  passengers → 0.6–1.9× amplitude scatter → ring residuals); projected (2D)
  closure radius is unusable (background column already = cosmic). Dwarfs match
  truth (small dwarf R_c is a projection/resolution artifact); the real fidelity
  gap is BIND slightly under-depleting group cores. See memory `project_covering_paint`.

## 2026-06-25 — `low_mass_extrapolation`: zero-shot test of BIND below its 1e13 training cut

New branch off `main`. Question: BIND trains on M200c > 1e13 halos, but the
6.25 Mpc/h patches contain smaller halos — can it paint sub-1e13 halos (goal:
TNG300 down to ~1e10)? Decision: **z=0 zero-shot validation first** with the
redshift-free `fm_thermo` model (`/mnt/home/mlee1/ceph/fm_runs/fm_thermo`,
EMA epoch064); user later narrowed the floor to **1e12** (1e10 is sub-pixel —
R200≈39 kpc/h < 48.8 kpc/h pixel; 1e12 is ~3.7 px, resolved).

- Fix (`0e2f9fa`): new `/mnt/home/mlee1/Sims/IllustrisTNG/L50n512` hydro
  snapshots are `snapshot_NNN.*`, loaders only globbed `snap_NNN.*`. Added
  `_resolve_hydro_snap_files()` (both prefixes), routed all 3 hydro call sites.
- Dry run (CV_0, 1e12, 382 halos): pipeline clean; mass conserved to 0.07%;
  **thermo log10 bias small & flat across mass** (y/T/K/P_e ≈ −0.04/−0.03/−0.05/0.00
  dex at 1e12–3e12, no worse than the >3e13 training regime); no channel collapse.
- Staging (`213d270`): `run_lowmass_suite.sh` (SLURM array, `HALO_MASS_MIN=1e12`,
  new L50n512 paths, explicit CV/1P roots, held-out SB35 `Test` manifest) +
  `examples/lowmass_extrapolation.ipynb` (coverage = 8.5× more halos at 1e12,
  thermo bias vs mass, per-halo mass conservation, painted low-mass halo).
  CV + 1P paths smoke-validated; user submits the arrays.
- Caveat: per-halo truth saved for thermo only; mass channels via mass
  conservation + composite. `fm_thermo` is z=0 — redshift dependence (TNG300
  goal) needs `feature/redshift` merged later.
## 2026-06-18 — f_b prediction, Stage 1: sim-validation suite

Turned the masked observable model into a baryon-fraction predictor: condition on a survey
subset (headline `M+Y+Tx`, withholding the baryon-mass observables) and read f_b off the
generated field. New reusable module `examples/fb_predict.py` (subset_keep / pack_cond /
generate / aperture_fb[_profile] / measure_obs_from_maps / **predict_fb_marginal** = the
DMO-template marginalization used for real halos w/o a DMO image). New notebook
`examples/fb_prediction_validation.ipynb`: subset ablation, f_b–M trend, feedback recovery
(SN `WindEnergyIn1e51erg` / AGN `BlackHoleFeedbackFactor`), **DMO-swap viability + coverage**
(real-data proxy), f_b(<r) profile. Smoke (16 halos, `fm_observables_masked`, n_params=14):
ablation scatter M 0.063 → M+Y 0.047 → **M+Y+Tx 0.047 (bias −0.016)** → full 0.016; **Y is the
degeneracy-breaker, not Tx**; full tightest b/c Mgas is in it (ceiling). DMO-swap 0.047→0.059 —
viability holds. Decisions: real-data target = **eROSITA/X-ray groups**, headline subset = **M+Y+Tx**.
Next (Stage 2): mock-observation calibration (survey→BIND-observable converters) + eROSITA ingest.

## 2026-06-17 — Observable input-dropout (`--mask_observables`)

Engine support so an observable-conditioned model tolerates a **missing subset** of
the 7 observables — the enabler for predicting f_b from a real survey's partial set
(`Y,Tx,M` → Mgas) and for cross-suite use. Conditioning vector packs to `2·N_OBS`
`[obs·mask, mask]` (`bind.data.pack_observable_conditioning`); training draws a random
keep-mask per sample (`sample_observable_keep_mask`: 25% full, else Bernoulli(0.5)).
CFG's whole-vector zero coincides with the empty subset, so it falls out for free.
`NormStats.mask_observables` (back-compat default False) records the mode for inference
auto-detect; `n_params=2·N_OBS`; model/`ParamEncoder` unchanged but for the width (so a
masked model is a fresh run, not a fine-tune). Touched `data.py` (helpers + NormStats
field + dataset packing), `train.py` (`--mask_observables`, validation, n_params, DataModule
persist), `inference/pipeline.py` (`build_observable_vectors` packs all-ones mask),
`run_train.sh` (`MASK=1`). Validated on CPU: helpers (25% full / 0.62 per-obs keep), NormStats
round-trip, dataset (7,)→(14,), UNet fwd+bwd grad through param_emb(14) + full & empty-subset
sample. Launch: `MASK=1 OBS=1 sbatch run_train.sh` (run `fm_observables_masked`). Docs: `docs/observables.md`.
Both observable notebooks made mask-aware (auto-detect `ns.mask_observables`, pack `[obs*mask, mask]`
via a `make_cond` helper, all-ones=full obs): `examples/analysis_observables.ipynb` (+ new §4
subset-conditioning — predict withheld `Mgas,Mstar→f_b` from `Y,Tx,M`, guarded to skip on unmasked
ckpt; also fixed its kernelspec `python3`→`torch3`) and `examples/observable_paint_literature.ipynb`.
Both re-validated headlessly on the unmasked `fm_observables` (run clean, §4 prints its skip msg).

## 2026-06-17 — Experiment A: paint baryons from observed scaling relations

First science use of the observable-conditioned z=0 model. `examples/observable_paint_literature.ipynb`
(GPU-explore notebook; `.py` script variant alongside) re-paints held-out halos twice from the SAME DMO + SAME noise: baseline (TNG-native
observables) vs "observed" (each observable multiplied by the literature/TNG fractional
offset at that halo's M200 — **ratio-anchoring**, to sidestep the projected-aperture vs
spherical-`_500` unit mismatch). Headline knob = X-ray group gas deficit (Sun+09/Lovisari+15/
Eckert+16/eROSITA); secondary Y/P (SZ ~0.85), Tx/K/Mstar; M200 fixed (lensing anchor).
Reports the *non-circular* outputs (Mgas/Mstar/M200 are inputs, so R200-integrated f_b is
~tautological): f_b(<r) at r≠R200, Gas/Stars profile shapes, and hydro-DM contraction
(DM_hydro/DMO, DM is an output). 4-halo smoke test on A100/torch3 validated end-to-end:
observed relations lower f_b by ~0.02 dex at all r and push DM_hydro closer to DMO (less
adiabatic contraction with fewer baryons) — both physically correct, both genuine predictions.
Decision: real f_b *prediction* (drop Mgas/Mstar from conditioning) needs an **input-dropout
retrain** — the enabler for cross-survey/cross-sim deployment too; queued, not done.

## 2026-06-16 — Observable conditioning (`feature/observable-conditioning`)

New conditioning mode: instead of the 35 cosmology+astrophysics params, condition
the emulator on **aperture-integrated observables within R200** measured (in
projection) from each halo's own maps — the quantities a survey reports. The DMO
image conditioning and the outputs (mass + thermo fields) are unchanged; only the
conditioning vector changes, so the UNet/`ParamEncoder` are untouched apart from
`n_params = N_OBS`. The observable vector flows through the existing `params`
batch slot.

- `OBSERVABLE_KEYS` (7): `Y_200, Mgas_200, Mstar_200, Tx_200, K_200, P_200, M_200`
  — integrated Compton-y, projected gas/stellar mass, gas-mass-weighted T/K/P,
  and M200c (lensing-like anchor). Per-feature log10+floor+standardize (mirrors
  the thermo transform). `compute_observables()` / `compute_norm_stats(...,
  condition_observables=True)` in `bind.data`.
- R200c is derived deterministically from the saved M200c
  (`m200c_to_r200c`, h cancels in h-units at z=0; matches FOF `Group_R_Crit200`).
  `data_generation/add_r200.py` optionally persists it as an `r200` key; the loader
  derives it on the fly when absent, so no data regen is required to train.
- CLI: `bind.train --condition_observables` (needs `--interpolant fm` + the thermo
  rotated2_128 path; pair with `--predict_thermo` for mass+thermo output). Launcher:
  `OBS=1 sbatch run_train.sh` (implies `--predict_thermo`, run_name `fm_observables`).
- Validated end-to-end on CPU (norm stats, dataset `target=(8,…)`/`params=(7,)`,
  NormStats round-trip, UNet forward+backward, sample). Reference: `docs/observables.md`.
- **Inference wired** (same session): `generate_halo_patches(..., cond_vectors=...)`
  takes per-halo normalized conditioning; `build_observable_vectors` /
  `extract_truth_mass_patches` (pipeline.py) measure per-halo observables from the
  truth maps; `load_model_bundle`/runner auto-detect `condition_observables` from
  norm_stats and thread it through `bind-camels-suite` (validation-by-reconstruction,
  needs truth+thermo). Verified the inference-side observable build is **bit-identical**
  to training (max |Δ|=0.0) and the full path runs against the live `fm_observables`
  checkpoint (n_params=7, out_ch=8).
- Notebook `examples/analysis_observables.ipynb`: load a checkpoint → reconstruct
  held-out halos (gen-vs-truth, dex error, radial profiles) + conditioning-response
  sweep (perturb one observable, fixed noise, watch field respond). Validated against
  `fm_observables/last.ckpt`.

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
## 2026-06-07 — `feature/redshift`: publish the redshift+thermo model to Hugging Face

Released the trained multi-z model (`/mnt/home/mlee1/ceph/fm_runs/fm_redshift`,
`last.ckpt` = epoch 199; `stars_two_head` + `predict_thermo` + `condition_redshift`)
as run **`fm_redshift_thermo`** on the HF weights repo. Slimmed the checkpoint
3988→1994 MB via `bind.tools.slim_checkpoint` (drops optimizer state, keeps
`state_dict`+`ema_state_dict`+hparams; verified it reloads through
`FlowMatchingLit`), then uploaded `last.ckpt`+`norm_stats.npz` to
`mel2260/BIND/fm_redshift_thermo/`.

Corrected a stale repo pointer: the real HF weights repo is **`mel2260/BIND`**
(already hosting `fm_two_head`/`fm_thermo`), not `Maxelee/BIND2`. Updated
`download_weights.py` (`DEFAULT_REPO`, registered the new run in `KNOWN_RUNS`),
`pyproject.toml`, `README.md`, `docs/index.md`, `docs/baryonify.md`. GitHub URLs
(`Maxelee/BIND`) left unchanged. ⚠ z>0 thermo physics still unvalidated — model
card warning not yet added.

## 2026-06-04 — `feature/redshift`: stage + launch the multi-z conditioned training

The multi-z dataset (`/mnt/home/mlee1/ceph/train_data_multiz_128_cpu`,
`process_simulations_multiz.py` output) is on disk: 922 train / 101 test sims,
**7** snapshots each (snap 024/z=4 has no halos >1e13, so it dropped out;
nominal z = 0, 0.21, 0.47, 1.05, 1.48, 2.00, 3.01). Verified sample npz carry
`redshift`/`scale_factor` + 4 thermo maps across z.

- **Prep** (`data_generation/prep_redshift_training.py`, new): one-time
  single-process build of the two recursive file caches
  (`file_list_cache_multiz.txt`: **151,685** train / **15,789** test) +
  `fm_redshift/norm_stats.npz` (stars_two_head + predict_thermo). Rationale:
  on a fresh run all 8 DDP ranks would each rglob ceph + compute stats and race
  on both writes. Gotcha: the interactive Slurm job has a **17.5 GB** cgroup cap;
  the default 10k-sample float64 stack OOM-killed (~12 GB) — prep now defaults to
  `n_stats_samples=4000` (~65M px/channel; the 1 TB training node keeps 10k).
- **Launch**: `REDSHIFT=1 sbatch run_train.sh` → `--condition_redshift
  --predict_thermo --stars_two_head`, out_ch=8, writes `fm_runs/fm_redshift/`.
  (Run by the user; sbatch isn't run from here.)
- **Notebook** (`examples/analysis_redshift.ipynb`, new): mirrors
  `analysis_thermo.ipynb` + a redshift-dependence section — per-z fidelity
  scorecard, amplitude evolution truth-vs-BIND, Y–M evolution, and a
  conditioning-response test (fix structure+params, sweep only the conditioning
  a). Same z>0-thermo-physics caveat applies to absolute high-z amplitudes.

## 2026-05-31 — `feature/redshift`: multi-redshift data + redshift conditioning

New branch off `main` to make redshift a continuous conditioning variable.
Design (decided with the user): condition on **scale factor a=1/(1+z)** via a
dedicated summed embedding (mirroring the time embedding), kept **optional**
(back-compatible); new dataset directory; inference accepts redshift *or* scale
factor.

- **Model/data/train** (`eda1c63`): `UNet(condition_redshift=)` adds a
  sinusoidal→MLP `redshift_emb` summed into AdaGroupNorm conditioning;
  `forward(x,t,params,scale_factor=None)` (defaults a=1 for a redshift model);
  `FlowMatching.loss/sample` thread `scale_factor` (held fixed under CFG).
  `data.py`: `z_to_a`/`a_to_z`, `SNAPSHOT_REDSHIFTS`, `AstroDataset(condition_redshift=)`
  emits per-sample `scale_factor`, `load_file_list(recursive=)` for the nested
  layout. `train.py`: `--condition_redshift`.
- **Data gen** (`457a088`): `data_generation/process_simulations_multiz.py`
  merges the mass + thermo pipelines into one MPI pass over 8 snapshots
  (z=0..4), 1 rotation/halo, nested `train/sim_i/snap_NNN/` with `redshift`/
  `scale_factor` stored. + `run_mpi_multiz.sh`.
- **Inference** (`ff5b237`): `bind.paint(..., redshift=/scale_factor=)` and the
  `bind-paint` CLI flags; `Model` reads `condition_redshift` from the checkpoint.

**Open / needs the user:** (1) ⚠ the z>0 **thermo** comoving→physical factors
(physical density ∝ a⁻³ → pressure/entropy; physical pixel area ∝ a² →
Compton-y) are implemented but **unvalidated** against an independent reference.
(2) Data generation + training are the user's compute steps (MPI/Pylians/GPU —
not runnable here); code is syntax-checked + unit-smoke-tested only.

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
