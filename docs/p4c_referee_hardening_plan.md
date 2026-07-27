# P4c referee hardening — critical report + remediation plan

Created 2026-07-28 (branch `analysis/ksz-desi-act-v2`). Successor to
`docs/tsz_des_data_plan.md` (P4c, complete through Phase L). Purpose: subject
the kSZ+tSZ selection/inference chain (M1 → T1–T3 → L) to a hostile-referee
audit and fix everything that is not research-grade. Same enactment protocol
(verdicts to `KS/lightcone/verdicts/R*.json`, figures to `KS/lightcone/figs/`,
no Slurm execution by agents, disBatch task files prepared for the user).

Current headline under audit: *"25/31 kSZ-consistent SB35 nodes survive the
real ACT y-CAP test (P(tSZ|kSZ)=0.81 vs 0.12, ρ=0.90); joint gas constraint
f̃_gas(<R500)=0.469+0.031/−0.005, shell 0.881+0.066/−0.009; TNG300 fiducial
rejected at χ²/dof=4.1 in the strong-feedback direction."*

---

## A. Referee report

### Major concerns (each could plausibly change a headline number)

**M-1. Satellite galaxies / miscentering are ignored (most dangerous).**
The BIND mock stacks at exact FoF halo centers of mass-selected *central*
halos. The data stack at DESI LRG positions: ~10–15% of LRGs are satellites
(HOD literature), and even centrals are miscentered w.r.t. the gas peak at
the few-tens-of-kpc level. Both effects *suppress the small-aperture CAP
signal in the data only* — i.e. they mimic exactly the observed "data below
fiducial at small xb" signature that the analysis attributes to strong
feedback. Until the satellite/miscentering term is modeled or bounded, the
5.7×→1.7× fiducial-over-data ratios and therefore the tSZ-consistent node
set are not secure.

**M-2. Covariance statistics are biased and partially ad hoc.**
(a) No Hartlap/Kaufman correction anywhere: the T3 χ² inverts a 30-cell
jackknife covariance for 6 bins → χ² biased high by ×1.32; the BIND
realization covariance (50 realizations, 6 bins) adds ×1.17; M1 (kSZ) and
the DES patch covariances have the same defect. Direction: we *under*-count
consistent nodes, but every quoted χ² is wrong.
(b) The CIB systematic is 1.6–3.2× the statistical error on the fit columns
— it *dominates* the covariance — yet is treated as diagonal, though CIB
residuals are strongly correlated across apertures. A correlated treatment
changes χ² nontrivially.
(c) 9 of the 11 CIB variants enter the fine-resolution band only via a
coarse-resolution transfer factor rather than being measured.
(d) 30 jackknife cells is marginal for a 27-column vector (singular regime
for any full-grid use) and the bootstrap-vs-jackknife comparison is recorded
but never propagated.

**M-3. The data-side aperture physics uses a single mass/redshift anchor.**
θ200(z) for every LRG assumes logM200 = 13.18 exactly. (a) The Sailer+24
anchor has an uncertainty (mass calibration error → coherent aperture and
amplitude shift; ±0.1 dex ⇒ ±8% in θ200). (b) The data sample has a mass
*distribution*, the mock has its own (σ=0.126 dex, median 13.161 — note also
the 0.02 dex median offset vs the anchor); stacked CAP over a mass
distribution ≠ CAP at the median mass. (c) The mock is one snapshot
(z=0.503) vs the data z∈[0.4,0.6] distribution; the measured z-0.45–0.9
robustness window was never used in the comparison.

**M-4. The two-halo / unpainted-gas floor inside the fit range is asserted,
not quantified.** The xb≤1.4 cut and the "conservative direction" argument
are reasonable but unquantified: sub-10¹³ halos and diffuse IGM contribute
*some* y at xb<1.4. If that contribution is a non-negligible fraction of the
small-aperture data signal, the true 1-halo data points are lower and the
feedback inference shifts further; if it is negligible, say so with a number.

**M-5. The kSZ leg was never audited to the T3 standard.** The tSZ leg was
rebuilt after the pixel-scale/aperture-convention/CIB findings; the kSZ M1
comparison (which defines the selection that the whole multi-probe story
rests on) inherits P4b conventions: were data and model apertures measured
with the same discrete estimator at the same pixel scale? Beam treatment?
Same Hartlap defect (M-2a)? Additionally, BIND's "kSZ" is a velocity-free τ
column; the published DESI×ACT f̃_gas data carry their own velocity
reconstruction, correlation coefficient corrections and modeling — that
transfer (and its uncertainty) is currently invisible in the M1 error
budget.

**M-6. The inference (Phase L) is importance sampling at ESS≈6.**
(a) Contours and 30-dim intervals hang on ≲10 design nodes; the corner's
lumps are individual nodes. (b) No continuous posterior (GP/emulator + MCMC)
despite the repo having exactly this machinery. (c) No coverage validation
(leave-run-out). (d) No look-elsewhere null: with 30 parameters and ESS~6,
some "constraints" arise from random weights — the expected null
distribution of width ratios was never computed. (e) The joint weight
assumes kSZ⊥tSZ; the data are independent but the *model* errors share BIND
realizations and the single TNG300 halo population. (f) The "posterior"
medians inherit the Sobol design measure (flat in the mixed lin/log box) —
prior-dependence untested.

**M-7. Emulator/model fidelity for this specific observable is indirect.**
BIND is validated against TNG300 hydro at the fiducial for WL/tSZ *maps* and
per-halo scalings, but there is no direct CAP-y closure at the mock LRG
halos (composite painting vs TNG truth patches at the same halos, same
estimator). The single-box provenance (all 253 nodes repaint the SAME TNG300
halo sample) means cosmic/super-sample variance of the box is common-mode
and unbudgeted. The 0.5′→0.293′ cubic resampling of the data map slightly
smooths the field; no closure test demonstrates the resampled CAP is
unbiased.

**M-8. The CIB/dust treatment is decisive and post hoc.** Switching the data
vector from baseline to deproj-CIB moves the consistent count from 0 to 30 —
the result is CIB-treatment-limited. The choice of deproj-β=1.7 as primary,
made after seeing the comparison, needs a pre-registered justification
(Liu+2025 precedent), a principled marginalization (dust-amplitude nuisance
or correlated CIB covariance), and the unused variants (EBV<0.15, z-window,
data−random subtraction) folded into a single systematic budget. The −3σ
large-θ random-null failure is flagged but uninvestigated.

### Minor concerns

- **m-1.** The consistency threshold χ²<dof+2√(2dof) is arbitrary; report
  p-values and a threshold-sensitivity curve (the P(tSZ|kSZ) enrichment
  should be shown as a function of threshold).
- **m-2.** KDE bandwidth choices (0.35/0.5) shape the ESS-6 contours; show
  bandwidth sensitivity or histogram-based contours.
- **m-3.** KG-T was re-scoped to the 1-halo range after it failed at large θ
  (documented, but formalize: our stack can be compared to Liu's *own*
  photometric sample measured with *our* pipeline — the dr9_lrg_pzbins
  catalog is on disk; this closes the 2–5× amplitude question properly).
- **m-4.** DES WL: dropping it is defensible (D4 numbers) but the paper needs
  the appendix + the ρ=0.78 co-ranking presented as corroboration, not
  constraint.
- **m-5.** Reproducibility: the entire campaign is uncommitted; no tag, no
  environment lock, seeds scattered.
- **m-6.** The beam is treated as an exact 1.6′ Gaussian (verified <0.1% on
  b_ℓ); the optional exact-b_ℓ filtering variant (plan §1.1, ≲0.3%) was
  never run — cheap to close.
- **m-7.** Blinding: several analysis choices (fit range, primary map,
  KG-T scope) were made after seeing data/model comparisons. A variants
  table with all pre/post-hoc decisions labeled is needed.
- **m-8.** The z0.45–0.9 and EBV variants were measured but never appear in
  any figure or budget.

---

## B. Remediation plan

Phases ordered by risk-to-headline. Effort tags: [h]=hours, [d]=day.

### R0 [h] — statistics hygiene (fixes M-2a, m-1; touches every χ²)
Add Hartlap factors ((n−p−2)/(n−1)) to every inverse-covariance: T3
(jk n=30), M1 kSZ (audit its n), BIND realization covs (n=50), DES legs.
Increase jackknife to ~100 cells (10×10 quantile grid) for the T2f
measurements so the correction is mild and the 27-column covariance is
non-singular; cross-check against bootstrap. Report per-node p-values and
consistency counts as a function of threshold. Regenerate T3/L/X numbers.
**Gate:** consistent-node counts stable to <±20% under jk→boot swap.

### R1 [h] — estimator closure tests (fixes M-7c, m-6, part of M-2c)
(a) Resampling closure: take BIND fiducial y maps, bin to a 0.5′ grid,
cubic-resample back to 0.293′, re-measure CAP at the mock positions vs
native — the bias curve IS the resampling systematic (expect ≪ CIB band).
(b) Exact-b_ℓ filter variant of the T2f baseline (ilc_beam.txt vs Gaussian).
(c) Re-measure ALL 11 CIB variants at 0.293′ via a disBatch task file
(11 × ~12 min) — retire the coarse-transfer approximation.
**Gate:** closure bias <2% on all valid xb columns.

### R2 [d] — HOD-population stacking (fixes M-1 + the aperture-rule mismatch)
Framing (refined 2026-07-28 discussion): the satellites' GAS is already in
the painted 6.25 Mpc/h fields on both sides — the mismatch is purely the
*stacking-center population* and the *aperture-assignment rule*, so the fix
is a forward model on the EXISTING maps, no external "correction" and no new
painting:
(a) Central miscentering is negligible and is hereby retired: BCG-vs-gas
offsets (~40 kpc) under the 1.6' beam (σ≈0.22 Mpc at z=0.5) suppress the
stack by ~(0.04/0.22)²/2 ≈ 2% ≪ the CIB band. Record the number in the
verdict.
(b) Satellite term, forward-modeled: build the mock LRG sample as an HOD
population on the existing halo catalog — centrals at halo centers per
N_cen(M), satellites at NFW-drawn projected offsets inside hosts per
N_sat(M) (hosts are typically MORE massive, so the sign of the net CAP
shift is not obvious a priori: offset dilution vs richer-host boost) —
and re-stack the SAME per-node y maps at that mixed population.
f_sat as a ±50%-prior nuisance. **User decision item: HOD reference
(default Yuan+23-class, f_sat=0.13±0.05).**
(c) Aperture-rule matching (new finding from this discussion): the data
assign θ_d = xb·θ200(z; logM=13.18 anchor) — quasi-fixed apertures — while
the current mock shards scale by each halo's TRUE per-halo θ200. These are
different measurement rules even for a pure-central population. The HOD
re-stack must assign mock apertures by the DATA's rule (anchor-based,
z-only). Quantify the rule-mismatch bias on the current T3 result as part
of the deliverable.
(d) Re-derive the tSZ-consistent set and Phase L with (b)+(c).
**Gate:** headline P(tSZ|kSZ) enrichment survives the f_sat=0.18 end-member
AND the aperture-rule swap; if not, re-scope the claim accordingly.

### R3 [h–d] — mass/redshift anchor propagation (fixes M-3)
(a) Propagate the mass-anchor uncertainty (adopt Sailer+24 σ_logM; user to
confirm value) as a coherent θ200 rescaling nuisance: re-measure the data xb
grid at logM ∈ {13.08, 13.18, 13.28} (three engine runs) → ∂CAP/∂logM
template → marginalize.
(b) Mass-distribution matching: importance-reweight mock halos to a
lognormal mass distribution centered on the anchor with the HOD-implied
width; quantify the stack shift vs the current cut sample (also fixes the
0.02 dex median offset).
(c) Use the z0.45–0.9 window + a second snapshot (snap 063 or 071) to bound
the z-distribution effect.
**Gate:** combined M-3 budget < CIB band on every fit column, else
marginalize alongside R2's nuisance.

### R4 [h] — two-halo floor quantification (fixes M-4)
Halo-model estimate of the 2-halo CAP-y at the sample mass/z (standard
P_e profile + linear bias; pyccl has the pieces), cross-checked empirically:
fit A_2h from the xb>1.4 excess (data − best-node) and extrapolate the CAP
kernel inward. Report the xb≤1.4 contamination fraction; if >10% of the data
signal at any fit column, add as a +template with free amplitude.
**Gate:** quantified number in the T3 verdict; conclusion direction
re-verified.

### R5 [d] — CIB/dust systematic, done properly (fixes M-8, m-3, m-8)
(a) Build the correlated CIB covariance empirically: Σ_CIB = cov across the
11 fine-res variant curves (rank-limited, add as a correlated block, not
diagonal); alternatively/additionally marginalize a single dust-template
amplitude (the baseline−deproj difference curve as the template shape).
(b) Fold in: EBV<0.15 shift, data−random subtraction, z-window shift → one
systematic table (m-8, m-7's variants table).
(c) Measure Liu's own sample: run the engine on the dr9_lrg_pzbins
photometric catalog (pz_bin 1–4, Main), same pipeline/resolution → direct
apples-to-apples with the Liu csv → resolves the 2–5× KG-T question and
gives an independent published-value validation point.
(d) Investigate the −3σ large-θ random-null (correlate null residual with
EBV / Galactic latitude / mask distance; if footprint-systematic, adopt
data−random as the default vector).
**Gate:** tSZ-consistent count stable within ±30% across (a)-treatments;
the baseline-vs-deproj dichotomy replaced by one marginalized result.

### R6 [d] — kSZ-leg audit (fixes M-5; protects the joint claim)
Re-open M1 with the T3 checklist: (a) confirm the data/model aperture
estimator match (pixel scale, discrete gates, beam) for the kSZ f̃_gas
comparison; re-measure the BIND side at the data's convention if needed.
(b) Document exactly what the published kSZ f̃_gas assumed (velocity
reconstruction correlation r_v, τ→f_gas conversion, mass anchor) and attach
an inherited-systematics term to the kSZ covariance. (c) Apply R0 statistics.
Re-derive `ksz_consistent_nodes.npz`; propagate through T3/L/X (the 31-node
set may change).
**Gate:** if the kSZ-consistent set changes by >30%, all downstream numbers
regenerate automatically (scripts are idempotent); document the delta.

### R7 [h–d] — model fidelity closures (fixes M-7a,b)
(a) Painting-completeness: CAP-y of the BIND composite vs the TNG300 hydro
truth patches at the SAME mock halos (truth products exist —
`bind-truth-halos` pattern), identical estimator; this is the direct
emulator-fidelity number for the exact observable used.
(b) Single-box variance: halo-subsample jackknife of the mock stack (already
possible from per-halo CAP values) + an explicit statement that box-scale
modes are common to all nodes (they cancel in node *ranking* but not in the
absolute fiducial-vs-data ratio).
(c) Verify thermo painting z-scaling at snap067 (the feature/redshift a-factor
caveat does not apply to the z=0.5 lightcone painting path — confirm).
**Gate:** fidelity bias < the R5 systematic band; else added to the budget.

### R8 [d] — inference upgrade (fixes M-6; makes the corner referee-proof)
(a) GP emulators of the two data-vector predictions over the 30-dim design
(or 2D-latent + nuisance dims): kSZ f̃_gas(θ) and tSZ CAP(xb≤1.4) — the
`bind.emulator`/`wl_latent_sbi` patterns. (b) MCMC (emcee) over 30 params ×
{R2 f_sat, R3 mass, R4 A_2h, R5 dust} nuisances with the R0-corrected
likelihood → continuous posterior; corner regenerated with real contours.
(c) Validation: leave-run-out coverage (perhalo-population-sbi pattern) +
GP-error inflation. (d) Look-elsewhere null: random-weight (ESS-matched)
draws → null distribution of 30-dim width ratios; report which parameters
exceed the null at >95%. (e) Prior-measure sensitivity: flat-lin vs flat-log
for the LogFlag params. (f) Joint-model-error covariance from the shared
realization structure.
**Gate:** GP leave-run-out coverage within 1σ bands; parameters claimed
"constrained" must beat the random-weight null.

### R9 [h] — paper hygiene (fixes m-4, m-5, m-7)
Variants/decision table (pre- vs post-hoc labeled); DES appendix framing;
commit the branch with per-phase commits + tag; pin the environment
(`pip freeze`, module list) into `docs/`; seed registry; final regeneration
of M2R/L/X from the R0–R8 outputs; WORKLOG + verdicts.

---

## C. What could actually kill the result (triage)

1. **R2 satellites/miscentering** — the only effect that mimics the
   headline signature with known-plausible amplitude. Do first.
2. **R5 CIB marginalization** — the 0↔30 baseline/deproj dichotomy shows the
   answer currently depends on a discrete map choice.
3. **R6 kSZ audit** — the selection that everything conditions on has not
   passed the standards its partner probe now meets.
4. **R0/R1** — quick, mechanical, and every quoted number moves by ~30% in
   χ² units until done.
5. R3/R4/R7 are budget lines expected to be subdominant — but must exist as
   numbers, not sentences.
6. R8 converts "importance-sampling grade" into publishable posteriors.

## D. Decision items (user)

1. R2: which DESI-LRG HOD reference for f_sat + radial profile (default
   Yuan+23-class, f_sat prior 0.13±0.05)?
2. R3: adopted σ_logM for the Sailer+24 anchor (default ±0.1 dex)?
3. R5c: run our pipeline on Liu's 12.4M-object photometric catalog (Main
   sample; ~90 min disBatch) — yes/no (default yes)?
4. R6: if the kSZ set shifts materially, do we re-freeze the headline before
   or after R8's continuous posterior (default: after)?
5. R8 scope: 30-dim GP+MCMC (default) vs 2D-latent-only inference?
