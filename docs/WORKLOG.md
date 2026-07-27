# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

---

## 2026-07-27 — R8: GP+MCMC inference upgrade, replaces the ESS~6 pseudo-posterior

Executed `docs/p4c_referee_hardening_plan.md` R8 (`examples/_r8_gp_mcmc.py`,
verdict `KS/lightcone/verdicts/R8.json`, figs `R8_posterior_corner.png` /
`R8_param_forest.png`, chains `KS/lightcone/r8_posterior.npz`). Replaces
Phase L's `exp(-dchi2/2)` importance weights over 253 Sobol nodes (ESS~6)
with continuous posteriors: per-column GP emulators (Matern+White, shared
ARD hyperparameters per physically-correlated column group, K-fold CV with
a temperature/calibration factor on the GP sigma) feeding an emcee MCMC
over 33 params (30 astro + f_sat + dlogM + **A_2h**, DE+snooker moves —
default StretchMove gave ~5% acceptance/huge tau at this dimensionality).

**Mid-session addendum**: R4 (two-halo/unpainted-gas floor, completed by a
sibling agent) quantified a 16–30%-of-data-value analytic 2-halo CAP-y
contamination exceeding its own 10% action gate; folded into the tSZ/joint
likelihood as a template with free amplitude `A_2h ~ N(1.0,0.3)`
truncated≥0 (R4's own analytic prediction is exactly 1). A_2h posterior
tracks its prior almost exactly (16/50/84 ≈ 0.70/0.95/1.21 vs prior
0.70/1.00/1.30) — the tSZ fit range doesn't strongly update it — and a
secondary `joint_noA2h` comparison chain confirms <0.03 (normalized-unit)
shifts in the top-8 constrained params with/without the term.

**GP validation**: post-calibration coverage lands close to nominal (1σ
≈0.67–0.73, 2σ≈0.93–0.96) after a 1.2–1.5× temperature factor; RMSE/
node-to-node-spread 0.45–0.86 (real signal, not noise) — gate passed.
**MCMC convergence**: did NOT reach the 50-tau target for any of the 3
primary chains (nsteps/tau_max ≈ 20–29; default StretchMove was far worse
before the DE-move fix) — a shared, contended 1-CPU box made a fully
gold-standard run infeasible in the available wall-clock; documented
honestly (`pass: false` on the convergence sub-gate) rather than faked.
**Gas-plane vs L**: joint f_in 0.39/0.45/0.51 (L: 0.464/0.468/0.498), f_out
0.90/0.95/1.00 (L: 0.867/0.880/0.919) — R8's contours are broad, smooth
regions that CONTAIN L's tiny lumpy ESS~6 contour, i.e. L's point estimate
was a real but overconfident/undersampled snapshot of a genuinely wider
posterior (see the corner fig). **Look-elsewhere null** (200 Dirichlet
draws, α tuned to ESS≈6.0 over 253 nodes): only 1/30 params
(`WindFreeTravelDensFac`) beats the 5th-percentile null threshold — neither
of L's own "robustness-stable" pair (`RadioFeedbackReiorientationFactor`,
`ThermalWindFraction`) survives a proper null test, though the top-ranked
directions (IMFslope, WindEnergyIn1e51erg) agree qualitatively with L.
**Prior-measure sensitivity**: flat-linear reweight of the 19 LogFlag dims
(ESS=53 of 5184 joint samples) shifts medians by ≤0.24 (normalized units,
largest for BlackHoleRadiativeEfficiency) — mostly mild.

**Engineering note** (for future long-running-compute sessions): a 1-CPU
sandbox shared with another agent made simple "launch in background, wait
for notification" unreliable in practice this session. Restructured the
script into a checkpointed CLI (`--stage {gp,chain,finalize}`,
`emcee.backends.HDFBackend` per leg under a scratch dir, `cloudpickle` to
hand the fitted-GP closures between processes) so the whole campaign runs
as a series of bounded (<10 min), independently resumable foreground calls
— worth reusing for the next long MCMC/training run in this environment.

## 2026-07-28d — R6: kSZ-leg audit finds a major data-covariance unit bug

Executed `docs/p4c_referee_hardening_plan.md` R6 (`examples/_r6_ksz_audit.py`,
verdict `KS/lightcone/verdicts/R6.json`, fig `figs/R6_ksz_audit.png`). Audited
M1 (`lightcone_m1_fgas_desiact.py`) to the T3 standard.

**Headline finding**: the published Zenodo 19160138 (Ried Guachalla+25)
`Fig8_*.npz['cov_ksz']` is **byte-identical to `Fig6_*.npz['cov']`** for the
logm11.00 cut — i.e. it is the covariance of the *raw T^CAP amplitude*
profile, not of the `ratio` (f̃_gas) column the M1/per-halo chi2 has always
been built from (confirmed by a second tell: `yerr` vs `sqrt(diag(cov_ksz))`
disagree by 3.26× too small at θ=1′ → 31× too large at θ=10.75′). Every kSZ
chi2 this repo has ever computed (M1 **and** the per-halo `_build_ksz_paper_nb.py`
§6/§6a/§7 money plots) inherited this. Fixed via a diagonal-rescale
(congruence transform onto `yerr**2`, preserving cov_ksz's correlation
shape) — the only correction possible without re-measuring the ACT map.
Also applied the missing Hartlap factor to the BIND-side realization block
(`h_bind=(n_r-1)/(n_r-ndof-2)`, T3's own convention) — cov_ksz's *own*
jackknife/bootstrap N is undocumented in the repo and NOT corrected
(flagged, not guessed).

**Result**: bgs110 31/253→27/253 (25/31 survive, Spearman ρ=0.97 old-vs-new);
bgs1125 48/251→**5/251** (only 3/48 survive, ρ=0.84) — bgs1125 blows through
the ±30% gate (89.6% change), because its one extra 1-halo-regime bin
(ndof=7 vs 6) was exactly the large-θ bin the bug had made vacuously loose.
kSZ-vs-tSZ ranking coherence is **robust**: Spearman(kSZ_r6, chi2_tsz) =
0.81 (bgs110) / 0.69 (bgs1125), vs T3's own 0.82 pre-correction — the paper's
ranking-coherence claim survives; raw consistent-node counts do not.
New product `KS/lightcone/ksz_consistent_nodes_r6.npz` (same schema as the
original + `_original`-suffixed companions; original file untouched).
**Next**: R7 painting-fidelity closure, R8 GP+MCMC; propagate `_r6` through
T3/L/X only if the campaign re-freezes on the corrected kSZ set (open
decision, see R6.json `next`).

## 2026-07-28d — R6+R7 (parallel Sonnet agents) + final chain regeneration

- **R7 (painting fidelity, agent)**: full TNG300-hydro TRUTH lightcone found on
  disk (bind-truth-halos, identical lux trace/realizations/seed) → closure at
  the exact 1691 LRG halos/apertures with the production estimator: **BIND
  over-paints CAP-y by +15–18% (mean +16.4%, ~10σ)** — explains ~16% of the 2×
  data deficit → tension NOT emulator bias. Bonus: this validates the
  redshift-conditioned (a-factor) production path empirically at z=0.5 —
  the plan's "does not apply" note was wrong (`weights/fm_redshift_thermo`
  generated the lightcone). `examples/_r7_fidelity_closure.py`, R7.json.
- **R6 (kSZ audit, agent)**: data vector = Ried Guachalla+25 Part II Zenodo
  f̃_gas (velocity-marginalised by their pipeline; cancels in BIND's ratio).
  **NEW BUG: the release's `cov_ksz` is byte-identical to the RAW T^CAP
  amplitude covariance (Fig6), not the f̃_gas ratio covariance** — unit fix
  applied by congruence rescale onto yerr²; + Hartlap on the BIND block.
  Corrected sets: bgs110 31→27 (25/31 survive, rank ρ_old,new=0.97);
  **bgs1125 48→5** (the fixed covariance actually constrains the 7th bin).
  Corrected kSZ-vs-tSZ coherence: ρ=0.81 (bgs110). Also affects the per-halo
  §6/§6a/§7 companion figures of `_build_ksz_paper_nb.py`.
  `examples/_r6_ksz_audit.py`, `ksz_consistent_nodes_r6.npz`, R6.json.
- **Final chain** (T3/L/X consume `ksz_consistent_nodes_r6` + R7 correction;
  UnboundLocalError shadowing bug fixed in T3): fid/data 4.96→2.26 over the
  fit range, fid χ²/dof=4.16 (p=3e-4), best node χ²=15.9/6, 4 nodes at
  p>0.01, ρ(kSZ,tSZ)=0.823. Joint gas latent: f̃_gas(<R500)=0.468+0.030/−0.004,
  shell=0.880+0.039/−0.013. **Headline unchanged: strong-feedback tension +
  ranking coherence; consistent-node counts demoted (unstable under both
  T- and k-leg corrections).**
- Remaining: R8 (GP+MCMC), R4 (2-halo number), R9 (commit/tag/variants table).

## 2026-07-28c — R3+R5: the error budget completed; the headline becomes a TENSION

- **R5a (correlated CIB)**: the diagonal half-band was absorbing non-CIB-shaped
  residuals (χ²_med 13.5 → 40 when the CIB freedom is restricted to its actual
  correlated shape, mean off-diag r=0.73). Diagonal treatment retired.
- **R3 (mass anchor)**: ±0.1 dex template built (model ±17–19%/0.1 dex via
  ±window stacks; data via aperture rescale) — `examples/_r3_mass_anchor.py`.
- **Final budget** (T3 primary): jk(Hartlap) ⊕ resample ⊕ per-node satellites ⊕
  correlated Σ_CIB ⊕ mass anchor. Under it **no SB35 node fits**: best node
  χ²=16.9/6 (p≈0.01), fid 4.36/6 (p=2e-4); the data sit ~2× below even the
  strongest-feedback node with a SHALLOWER profile (3–5σ/column), and no
  amplitude freedom fixes it (shape-orthogonal residual). Per-variant range
  (each deprojection map as truth): all reject. **The paper's headline is the
  tension** — map-level counterpart of the P4 eROSITA finding — plus the
  invariant kSZ–tSZ ranking coherence (ρ=0.82–0.90 through every budget).
  Consistency counts (0↔119 across treatments) are demoted to a treatment
  table. T3/L/X regenerated under the full budget.
- **R5c (Liu reproduction, `examples/_r5c_liu_figure.py` + engine `--mode
  liu/rotated_liu`)**: our pipeline on Liu's own 12.4M-object catalog
  (400k/bin) REPRODUCES their published curve at ~1.1σ mean — once one
  realizes **the release csv's duplicated column is the pz4 (z~0.9) bin**
  (χ²/9: pz4 1.7, pz3 4.3, pz2 14, pz1 46): the bins-bit-identical bug is now
  fully characterized, our measurement resolves all four bins (2–4σ/pt,
  correct z-ordering), the rotated null is ~0 (no hidden background
  subtraction). Data side validated end-to-end at the fit scales.
- **Open**: R7 painting-fidelity closure is now the critical item for the
  tension claim (is BIND CAP-y faithful to TNG truth at these halos at the
  10–20% level?); then R6 kSZ audit, R8 GP+MCMC.

## 2026-07-28b — Referee hardening R0–R2: statistics, closures, and the satellite forward model

Executed `docs/p4c_referee_hardening_plan.md` R0–R2 (report + plan written the
same session; R2 reframed after discussion: satellites' GAS is already painted
on both sides — the mismatch is the stacking-center population + the
aperture-assignment rule).

- **R0**: inverse-Hartlap inflation on every sample covariance block; engine
  jackknife 30→100 cells + per-object CAP/position/weight storage
  (`T2f_*` re-measured, 15-task disBatch `run_t2f_disbatch.sh`); p-values +
  threshold scans in verdicts.
- **R1** (`examples/_r1_closure.py`): beam-shape retired (≤0.014%); the
  0.5'→0.293' resampling path biases stacked CAP −4.5→−1.9% at θ<1.6'
  (mimics feedback) → measured template divided out of the data
  (`R1_resample_correction.npz`), 50% kept as systematic; fine-res 11-variant
  CIB band direct (transfer approximation retired).
- **R2** (`examples/lightcone_hod_stack.py`): aperture-rule correction is
  node-independent (±2%) and adopted; naive satellites-on-top double-counts
  the lensing anchor (kills all nodes — recorded as unphysical bound);
  median-mass recalibration is a no-op; the defensible calibration matches
  the mixed population's CAP-κ stack on the lightcone κ maps to the
  pure-central 13.18 reference. Satellite boost is strongly NODE-dependent
  (±35–39%) → final = per-node HOD stacks (254-task disBatch
  `run_hod_disbatch.sh`), f_eff ∈ {0.04,0.08,0.12} (true f_sat×0.73;
  sub-1e13 hosts hold 27% of satellites but 0.2% of the Y-boost).
- **Final numbers** (per-node HOD model, all corrections): 119/253
  tSZ-consistent (P=0.47); **P(tSZ|kSZ)=1.00 at every threshold**;
  ρ=0.879; TNG300 fiducial now CONSISTENT (p=0.18) — the "fiducial
  rejected" claim is retired; count evolution 30→53→119 shows raw counts
  are treatment-dependent — lead with conditional/enrichment + ranking.
  Baseline-map data still 0/253 → R5 (correlated CIB) is the top remaining
  systematic, then R6 (kSZ audit), R8 (GP+MCMC inference).

## 2026-07-27/28 — P4c executed end-to-end: real ACT DR6 y-CAP + DES Y3 WL vs the kSZ subspace

Full enactment of `docs/tsz_des_data_plan.md` (T0–T3, D0–D4, Capstone X);
verdicts in `KS/lightcone/verdicts/{T0..T3,D0..D4,X}.json`, figures
`figs/{VT0,VT1,VT2,VD1,VD2}_*.png` + money plots `M2R_ycap_real.png`,
`M7_des_wl.png`, `X_multiprobe.png`; notebook §7 addendum
(`examples/_build_ksz_lightcone_nb.py`).

- **Stream T** — new engine `examples/act_ycap_measure.py` (batched-gnomonic
  thumbnails ~800 obj/s single-core, pixell-exact to ≤0.09%; DR5-cluster
  validation S/N 22–32, nulls pass). Measured 160k DESI DR1 SGC spec-LRGs
  (z 0.4–0.6) on baseline + 10 CIB-variant maps + EBV/z-window variants +
  random/rotated nulls. **Estimator lessons (first M2R draft was wrong —
  caught in review):** (1) the discrete CAP filter is pixel-scale-dependent →
  data re-measured at BIND's 0.29296875'/px (`--res_arcmin`); (2) compare
  xb=θ/θ200-scaled apertures on both sides, never fixed-arcmin vs scaled;
  (3) baseline ILC is CIB/dust-biased (negative y!) at small θ → deproj-CIB
  is the primary data vector, 11-map spread in the covariance; (4) χ² cut to
  xb≤1.4 (M1's 1-halo convention) — beyond it the data climb through the
  envelope (2-halo + sub-1e13/IGM gas BIND doesn't paint).
  **Result (T3/M2R): 30/253 nodes tSZ-consistent; P(tSZ|kSZ)=0.81 vs
  P(tSZ)=0.12 (~7× enrichment); ρ(kSZ χ², tSZ χ²)=0.90. TNG300 fiducial
  over-predicts small-aperture y up to ~5.7× (χ²/dof=4.1) — the real sky
  prefers the strong-feedback edge, same direction as kSZ.**
- **Stream D** — D1 killed direct Hankel ξ± (ℓ<87 truncation of the 5° FOV
  loses 10–130%) → forward model = pyccl theory Cl (exact DES n(z), TNG300
  cosmo) × paired-DMO suppression S_ab(ℓ) (`examples/des_bind_forward.py`);
  no mas_correct re-trace needed (W² ≤4.3% at ℓ≤6000, corrected; upturn
  ℓ≳1.2e4). D2: official 2pt vector + COVMAT + 1000 n(z) reals; 45
  hole-tolerant 5° GLIMPSE/Wiener patches (plan's 150–200 impossible —
  Swiss-cheese mask; GLIMPSE holes filled from Wiener).
  **Result (D4/M7): raw fixed-cosmology rejects all 253 (B_fid=0.79 ≈ the S8
  offset); amplitude-marginalized accepts all 253 (ξ+ envelope ≤0.4% vs
  11–19% errors — DES 2pt has no residual feedback discrimination); but the
  χ² ranking tracks the kSZ selection (ρ=0.78). Map leg uniformly rejected →
  reconstruction-filter dominated, consistency-check-grade only.**
- **Capstone X**: 25/31 kSZ-consistent nodes survive real-tSZ (12% base
  rate); survivors sit at flatter IMFslope (−2.08 vs −2.34) and lower
  log₁₀WindEnergy (0.28 vs 0.62) — the M6 directions.
- Ops: session cgroup ~10 GB + 1 CPU (OOM lessons: per-galaxy `DA()`
  linspace; fine-res chunk scaling); monolithic per-node map stats hit a 6 h
  Slurm wall → redesigned as 253 disBatch shards
  (`examples/_mapstats_tasks.disbatch` + `run_mapstats_disbatch.sh`, ~25 min
  total); `run_tsz_des_overnight.sh` = the idempotent T2 battery + tail.
- **Phase L (post-X pivot, user decision: drop DES WL as a constraint)** —
  `examples/lightcone_latent_corner.py`: kSZ/tSZ/joint χ² → importance
  weights over the Sobol design, shown in the parent paper's rotated 2-D gas
  latent (rebuilt verbatim, λ=0.51/0.46, r=+0.95/+0.95) →
  `figs/L_latent_corner.png` (corner: both probes select the low-inner-gas
  edge; joint f̃_gas(<R500)=0.469+0.031/−0.005, shell 0.881+0.066/−0.009) +
  `figs/L_param_backtrack.png` (30-dim weighted 68% forest). ESS 9/13/6 —
  pseudo-posterior, robustness block in `verdicts/L.json`: width-stable top
  set {IMFslope↑, RadioFeedbackReiorientationFactor↑, ThermalWindFraction↓,
  UVBHepbeta}; median-pull directions stable 16/18 (wind/SN energy LOW,
  IMFslope/WindFreeTravelDensFac HIGH, BHEddington LOW). Notebook §7
  extended (55 cells).

## 2026-07-24 — M6: SB35 astro-param constraints from the kSZ-consistent node sets

Closed out `docs/ksz_lightcone_map_plan.md` phase M6 — turns M1's
kSZ-consistent node SETS around to ask what they say about the 30 SB35 astro
params relative to their Sobol prior (a selection test, not a posterior).

- Extended `examples/lightcone_m1_fgas_desiact.py::main()` (new `skip_fig`
  arg / `--skip-fig` CLI flag) to also write
  `KS/lightcone/ksz_consistent_nodes.npz` — `node_ids_all` (253,, positionally
  aligned run IDs), `node_ids_bgs110`/`node_ids_bgs1125` (31,/48,, the
  DESI-precision-consistent run IDs), `chi2_desi_bgs110`/`chi2_desi_bgs1125`
  (253, each, NaN at bgs1125's 2 empty-sample nodes), `readme`. Ran with
  `--skip-fig` so the existing `M1_fgas_vs_desiact.png` was not touched
  (md5-verified identical); reproduced the P6b headline 31/253, 48/251
  exactly. Finding: bgs110-consistent is an exact subset of
  bgs1125-consistent (overlap 31/31) for this data.
- New `examples/lightcone_m6_params.py` → `KS/lightcone/figs/
  M6_ksz_param_constraints.png`: panel (a) all 30 astro params (prior-
  normalized to the empirical 256-row Sobol box, log10-space for the 19/30
  `LogFlag==1` params per `bind.params.PARAM_LOG_FLAG`), full-253 grid vs.
  both consistent sets' 16/50/84 percentiles, sorted by KS p-value (smaller
  of the two cuts, annotated per row); panel (b) pairwise scatter of the
  top-3. Verified the design is a genuinely filled box (max 3.3% relative
  deviation of per-param min/max over random 200/256 subsets).
- **Result**: 4/30 params cross p<0.05 (vs. ~1-2 expected by chance over 30
  tests), 3/30 survive Bonferroni. Top 5: `IMFslope` (p=0.0004, new signal),
  `WindEnergyIn1e51erg` (p=0.0015), `VariableWindSpecMomentum` (p=0.0015, new
  signal), `VariableWindVelFactor` (p=0.012), `WindFreeTravelDensFac`
  (p=0.056). Cross-check vs. the per-halo D5/D5-CAP "wind/SN sector"
  (`VariableWindVelFactor`, `WindEnergyIn1e51erg`, `WindFreeTravelDensFac`,
  `MinWindVel` — `docs/ksz_desi_act_plan.md` lines 165-199): 3/4 land in
  M6's top 5 (agreement) — `MinWindVel` does not (rank 17/30). `IMFslope`
  and `VariableWindSpecMomentum` are signals the per-halo GP posterior did
  not flag, a genuine new finding from the map-level selection test, not a
  contradiction (different statistic: hard selection on 253 discrete draws
  vs. a smooth GP-emulated posterior).
- Added an M6 subsection to `examples/_build_ksz_lightcone_nb.py` (end of
  §4, after M5) with its own "what would look wrong" note; rebuilt +
  re-executed `examples/paper_ksz_lightcone.ipynb` end-to-end — 47 cells,
  zero errors. Verdict `KS/lightcone/verdicts/M6.json`
  (`next: "feeds DES stream (ksz_consistent_nodes.npz)"`).

## 2026-07-24 — M5 finisher: massbin notebook section + verdict metrics

Closed out `docs/ksz_lightcone_map_plan.md` phase M5 (the map-level
recreation of the per-halo headline f̃_gas–logM200 figure), whose
companion-notebook integration was lost in a session restart even though the
506-shard `_p5_massbin_tasks.disbatch` sweep and `M5_fgas_mass.png` were
already complete on disk. Added an M5 subsection to
`examples/_build_ksz_lightcone_nb.py` (end of §4, after M3): prose on the
≥1e13 closure (1-6% vs. per-halo, inside the P4 gate) and the sub-1e13
reuse-regime story (pooled curve diluted by unpainted background; the
dashed in_patch substack — ELG in_patch 0.578 ≈1.3× the data vs. per-halo's
≈1.9× — is the honest read), a re-plot cell calling
`lightcone_m5_fgas_mass.main()` directly (mirrors the other money-plot
cells; ~75MB of per-node shard `.npz` files, the notebook's single largest
data load but still well under the plan's 200MB budget), and its own "what
would look wrong" note. Rebuilt + re-executed
`examples/paper_ksz_lightcone.ipynb` end-to-end — 42 cells, zero errors.
Added `metrics.envelope_range_per_bin` (253-node max−min per mass bin) and
`metrics.n_nodes_touching_bgs_data` (1σ overlap count at the ≥1e13 bins
where all 5 BGS data points live) to `KS/lightcone/verdicts/M5.json`:
48/253 nodes touch at least one of the 5 BGS points (0/253 touch all 5,
driven by the M★=10.50 cut having zero touching nodes at any node) — close
to, but not the same statistic as, the per-halo capstone's headline 49/256.

## 2026-07-24 — P6 stage B (final assembly): map-level money plots + notebook

Completed `docs/ksz_lightcone_map_plan.md` phase P6 (verdict
`KS/lightcone/verdicts/P6b.json`, `KS = bind_science/ksz_confront`). New
products/scripts:

- `examples/lightcone_capmat_merge.py` (new) + a minimal `--map massplane`
  branch in `lightcone_cap_stack.py` (reads
  `bind_lightcone_tng/haloplane_trace/massplane_maps.npz` key `'tau'`, capped
  to its 47 realizations) → `KS/lightcone/capmat_lightcone.npz`, the f̃_gas
  CAP_mat denominator (fiducial-shared across all 253 Sobol nodes; error
  convention: f̃_gas's realization scatter is carried entirely by the
  numerator, denominator fixed at its 47-real. mean — documented in the
  script docstring).
- `examples/lightcone_cap_merge.py` generalized (per-product `samples` list)
  to also merge the ELG/LRG shard sweeps → 9 new products
  (`elgtau[_smear]/elgy[_beam]/elgkappa/lrgtau/lrgy[_beam]/lrgkappa_lightcone.npz`).
- `examples/lightcone_m1_fgas_desiact.py` (new) — **M1, the headline**:
  map-level f̃_gas(θ) for both BGS cuts vs Ried Guachalla+25. First pass used
  a single **fiducial-shared** CAP_mat denominator (per the plan's literal
  wording) and found 217/253 (bgs110) / 245/251 (bgs1125) map-level
  consistent — far above the per-halo 49/256, 65/256 — plus individual SB35
  node curves reaching unphysical f̃_gas≈4–16. A mid-session message flagged
  the f̃_gas>1 shape itself as a bug; checked explicitly and found it was
  the expected, already-documented "cosmic" f̃_gas→1 asymptote at large θ
  (fiducial stayed ≤0.87 throughout the 1-halo regime) — that specific claim
  was not correct and was not acted on. A **follow-up** message correctly
  identified the real defect: BGS's per-node M★ cut selects a *different*
  halo subset per Sobol node (feedback changes which halos cross the
  threshold — verified independently against `mstar_matrix_snap085.npz`
  before acting: node 64 selects 11 halos at mean logM200=14.38 vs the
  fiducial's 1753 halos at 13.44), so dividing a node's own gas column by
  the *fiducial's* matter column inflated f̃_gas for mass-mismatched
  extreme-feedback nodes — the actual source of the unphysical tails. Fixed
  by rewriting `lightcone_capmat_merge.py` to stack the (still
  fiducial-shared, single-co-trace) massplane field at **each node's own**
  M★-selected positions (253×2 in-memory CAP stacks, ~30 min, no Slurm);
  independently verified ELG/LRG selection is node-independent (byte-
  identical across nodes) so their shared CAP_mat was never affected. Post-fix
  the SB35 envelope is fully physical (no node >1.5 anywhere in the 1-halo
  regime; max observed 1.09/1.19) and M1 reports two χ² variants: (a)
  survey-variance (`cov_ksz + this node's single-realization covariance`,
  still 92–98% consistent — a real, KG3-anticipated finding about one 25
  deg² realization's small landing-galaxy count, not a bug) and (b)
  **DESI-precision** (`cov_ksz + covariance of the realization mean`, the
  headline/plotted classification): 31/253 (12.3%) and 48/251 (19.1%) — the
  same order of magnitude and qualitative story as the per-halo 49/256
  (19.1%) and 65/256 (25.4%). Fiducial f̃_gas at θ(r200) unchanged throughout
  (0.80/0.85, still <5% from the per-halo reference 0.77/0.84). Full
  defect/fix narrative in `KS/lightcone/verdicts/P6b.json`'s
  `metrics.defect_and_fix`.
- `examples/lightcone_m2_ycap_liu.py` (regenerated) — LRG (mass-matched to
  Liu's logM200=13.18) promoted to the PRIMARY panel: fid/Liu = 1.9-4.2×, an
  O(1) offset (vs 32-165× for the old BGS-vs-LRG mass-mismatched
  comparison), confirming most of the old gap was mass selection, consistent
  in direction/magnitude with the trusted per-halo LRG result (~1.5×).
- `examples/lightcone_m4_elg_zshell.py` (new) — ELG (z=1.16) τ-CAP/f̃_gas in
  the patch-reuse regime (in_patch=29.2-29.4%) + a deflection-smearing
  sensitivity panel: measured 14% mean / 20% max suppression (larger than
  the plan's rough "~%-level" prior guess — reported as measured, not
  adjusted to match), because the P2 deflection σ=0.222′ is a large
  fraction of ELG's own θ200≈0.36′. Confirmed unaffected by the M1 CAP_mat
  defect above (ELG selection is node-independent); denominator's own 96.6%
  realization-scatter noise floor now explicitly annotated on the figure.
- `examples/_build_ksz_lightcone_nb.py` → `examples/paper_ksz_lightcone.ipynb`
  (executes end-to-end, <25MB of npz loaded, one optional raw-map cell gated
  behind `RUN_HEAVY=False`). 6 sections (§0 overview, §1 geometry, §2
  catalogs, §3 closure, §4 money plots regenerated in-notebook, §5 caveats,
  §6 verdict), each with a "what would look wrong" paragraph.

## 2026-07-24 — P4b plan: map-level DESI×ACT confrontation on the SB35 lightcones

Reconnaissance + execution plan for redoing the P4 kSZ/tSZ headline (per-halo
f̃_gas + y-CAP vs DESI×ACT) on the ray-traced lightcone maps
(`bind_sb35/runs/*/{tau,y,kappa}_maps.npz`, 50 realizations × 25 deg²/node).
Plan: `docs/ksz_lightcone_map_plan.md` (phases P0–P7, agent-enactable).

Key findings that unblock it (verified against lux + BIND source + disk):
- lux's per-realization randomization = per-snapshot 90°-rotation + integer
  pixel shift, seed `1992+7*r`, and the drawn values are **persisted in the
  tail of `rt_output/runNNN/config.dat`** — the fiducial retains all 50, and
  the same records apply to every Sobol run (shared RT_SEED). No RNG
  re-implementation needed; full halo→map-pixel chain is deterministic
  (plan §1.2, incl. the `4096-j` scatter quirk and ray geometry).
- Untracked `bind.cli.paint_{haloplane,massplane}` (co-trace a halo-indicator
  + total-mass plane through one fiducial lux run) = gold-standard validation
  + shared CAP_mat denominator (plan §1.3, P2).
- No instrument-beam handling exists anywhere in the repo — must be added for
  the y-CAP vs Liu+2025 comparison (per-halo tSZ figure stays untrusted).

Researched + prototyped extending the Sobol suite from 30 astro params to the full 35 (adding
Om, s8, Ob, h, ns) by Angulo & White (2010) rescaling of TNG300-Dark instead of new N-body runs.
**Verdict: feasible**; full plan with validated error budget in `docs/cosmo_rescaling_plan.md`.

- `examples/cosmo_rescale.py` — (s, z\*) solver on linear sigma(R) (pyccl+CAMB; camb pip-installed
  into BIND_env). Identity exact; typical SB35 draws rms(sigma) 0.03-2%; joint one-s-per-lightcone
  fit costs nothing. Extreme corner (Om=0.1 ∧ s8=1.0) pins s=5 → degrades (decision D2 open).
- `examples/rescale_validation.py` — real-data test on TNG300-3-Dark (625³): P(k) + M200m HMF of
  rescaled snapshots vs halofit/Tinker08 at the target, unrescaled-vs-theory as control band.
  Mild targets ≤3.4%; large shifts show the two known AW10 residuals (BAO misalignment wiggles at
  k~0.1-0.3, concentration deficit at k≳1). **Halofit-level D(k) predicts the measured error
  point-by-point** → theory scan = per-run quality flag, and a map-level Fourier reweighting can
  cancel the bulk (v1.5). Figure: `examples/rescale_validation.png`.
- `examples/rescale_sobol_scan.py` — 128-draw error scan over the SB35 cosmology box (numbers in
  the plan doc).
- Pipeline audit: stage-1 maps/catalogs are cosmology-independent (relabel, don't recompute);
  lux is cosmology-clean in PreProjected mode (config.dat carries chi/a); `generate_from_stage1`
  already takes a scale_factor override; velocities never used. Engine changes needed: rescale
  design CLI, per-run "stage-1b" cutout extraction (target-unit manifests → GPU side unchanged),
  per-run lightcone geometry (N' replications, plane indices), paired per-run DMO trace.

## 2026-06-29 — §6 of `paper_lightcone_figs2.ipynb` fixed: the *lightcone* halo population, not the z=0.034 box

§6 connected the WL-suppression latent to halo baryon structure using **only the z=0.034 snapshot**
atlas — wrong population for a ray-traced z_s≥0.5 statistic. New engines under `examples/`:

- `lightcone_halo_catalog.py` — per-run **lightcone halo catalog**: joins `halo_centers`+`slab_idx`
  (transverse position, **already in the lightcone frame** since the per-snapshot transform is applied
  at stage-1 projection) from the `composite_slab*.npz` to the cached per-halo atlas
  (`bind_sb35/analysis_cache/halo_atlas/run_XXXX_snapSSS.npz`), in the same sorted-slab order so rows
  align 1:1 (asserted via `M_fof`). Adds z (snapshot shell), comoving χ + angular θ from the lux
  `config.dat`, and lensing-kernel weights W(z;z_s). Cache `analysis_cache/lightcone_halo_catalog/`
  (253 runs + fid, ~34k halos/run over 20 shells z=0.034–2.444). Built with `--jobs N` (multiproc).
- `lightcone_halo_observables.py` — reproduces the §5 latent and compares the z=0.034 readout vs the
  kernel-weighted lightcone readout. **Result:** the z=0.034 shell carries only **~3%** of the z_s=1
  lensing weight (kernel peaks z≈0.42), yet the latent↔physical mapping is **robust** — because feedback
  acts *coherently across cosmic time*, run-ranking of the **amplitude** features (f_gas/f_star/Y/T/K)
  at z=0.034 matches the lightcone at **r>0.95** (latent-2 validated). The **redistribution** features
  (gas conc/ejection) are *not* proxied (r≈0.47); on the lightcone latent-1 CV R² improves 0.94→0.95.
- `lightcone_halo_dkappa.py` — per-halo **Born Δκ** from the fiducial composites (full `composite`+`dmo`
  4198² maps, exactly registered to `halo_centers`; κ=Σ/Σ_cr): the baryonic imprint peaks at z≈0.33 and
  per-halo tracks **f_gas (r=+0.62)** + T. Direct halo-property→observable link.

Notebook: new **§6.4** (`fig_lightcone_population`) inserted after §6.3, loads
`analysis_cache/lightcone_halo_section6.npz` (built by `_observables.py --cache`). On branch `lightcone`.
Also rebuilt the stale fiducial atlas (`bind_science/halo_atlas/fid_snap{067..029}` were missing the
bg-subtracted fields). Sobol runs have **no** lensplanes (only randomized ray-traced `kappa_maps.npz`),
so the per-halo map link is fiducial-only.

---

## 2026-06-29 — NEW `examples/wl_latent_sbi.ipynb`: WL feedback **global response · latent space · rethought SBI**

Re-evaluation of `examples/wl_baryon_response.ipynb` (one-param-at-a-time, dashboard cache) now that the
256-pt SB35 Sobol sweep + `bind.emulator` exist. Engine `examples/wl_latent_sbi.py`, builder
`examples/_build_wl_latent_sbi_nb.py`, notebook `examples/wl_latent_sbi.ipynb` (26 cells, builds clean,
runs on GPU). Reuses `bind.emulator`, `wl_stat_latents.py`, `wl_feedback_vae.py`, `sobol_ml.py`, and the
kSZ §4 latent kernel from `paper_ksz_desi_act.ipynb`.

- **§1 Global response** — **model-free** bin-by-bin **Spearman** correlation of each measured statistic
  with the 30 params, read straight off the Sobol runs (deliberately emulator-free: per §3d the emulator is
  only percent-accurate for the suppression, so it is not trusted to *rank* sensitivity). One heat map per
  statistic (param × ℓ/ν/κ bin). Top C_l-suppression drivers (|Spearman r|): IMFslope 0.49, VariableWindVel
  0.48, BHRadEff 0.43, WindEnergy 0.34 (the wind/SN/BH feedback sector); peak-function sensitivity
  concentrates in the high-ν tail. (Emulator-pushed Sobol indices kept in the engine for reference —
  `global_sensitivity`/`sobol_indices` — but no longer drive §1.)
- **§2 Latent + the kSZ bridge (headline)** — WL feedback is **2-D** (PCA 0.884/0.103, 98.7% in 2 comps).
  **It is NOT the same plane the kSZ gas sees:** WL↔kSZ canonical corr **[0.98, 0.79]**, principal angles
  **[12°, 38°]** — dominant axis *shared* with the kSZ inner-gas/suppression axis, **second axis rotated ~38°**;
  WL tracks the inner/outer f_gas zones only moderately (r=0.59/0.71 vs kSZ's 0.95). Interpretation: WL κ is the
  projected *total-mass* perturbation → responds to a *blend* of gas zones, not their separation. Answer to
  "does the latent change?": **same leading direction, rotated/blended second.**
- **§3 SBI, rethought** — `EmulatorSimulator` = emulator-as-simulator (unlimited (θ,x) + calibrated noise),
  escaping the n=253 starvation of `lightcone_cl_sbi.py`. Target = measured fiducial (`bind/run_0000`, θ_fid = prior
  centre). From **C_l alone**: ~5 params constrained (IMFslope 0.66, WindEnergy 0.72, VarWindVel 0.75, BHRadEff
  0.83), rest near-prior (data are rank-2) → report the **2-D latent posterior**, which at full scale is
  **tight (~0.17–0.18 of the node-cloud width)** — C_l pins the suppression-latent hard even though the
  30-param marginals stay broad. emcee (explicit-likelihood) + SBC (68% coverage 0.70, calibrated) cross-checks.
  **Cautionary result (§3d):** *stacking* emulated peaks/κ×y onto C_l **degrades** the fit (mean shrink
  0.97→1.01→1.03, informed 5→0) — because the measured fiducial is in-distribution for C_l (max|z|=2.1σ) but a
  **5.2σ outlier** once HOS are appended. Cause = **emulator HOS/cross fidelity**, not missing info (§1 shows
  peaks/κy respond strongly). ⇒ C_l/suppression is the trustworthy SBI data vector here; new directions need an
  *orthogonal, well-emulated* tSZ probe.
- Two engine bugs fixed mid-session: Sobol summary on low-variance scalars exploded → use **PC1 projection**;
  the `log10` clip floor (1e-6) silently zeroed the tiny-valued κ×y (~1e-21) → **raw** for cross/auto spectra.
- GPU: NPE flow training + SBC run on the node's **V100S** (`device` threaded through `run_npe`/`sbc_ranks`;
  notebook auto-detects). The `gp` emulator backend kept (predict already cheap at n=253). `WL_SBI_FAST=1`
  switch + `wl_latent_sbi_figs/*.npz` caching for cheap re-runs. On `lightcone` branch; figs/cache gitignored.

---

## 2026-06-27 — NEW `bind.wlemu`: **field-level** WL emulator (θ → κ map, conditional flow matching)

User wants a *generative* WL emulator (vs the summary-stat `bind.emulator`): feed 30 SB35 astro params
+ z_s, **draw fresh κ maps**. Surveyed the 4 proposed methods (CFM / latent-CFM / score / wavelet-flow) →
chose **Conditional Flow Matching θ→κ**: BIND is already a param-conditioned FM U-Net so the conditioning
machinery is validated here; ~20–50 NFE sampling matters for drawing many maps; latent/wavelet add
compression/scale-coupling risk that would wash out the small-scale baryon response (the science point).

New module `src/bind/wlemu/` (engine, mirrors `bind.emulator` layout):
- `model.py` — `FieldUNet` (resolution-aware sibling of `bind.model.UNet`: single-channel, **no DMO
  concat**, 30-param + z_s conditioning reusing the AdaGroupNorm/sinusoidal/`redshift_emb` blocks) +
  `FieldCFM` (OT flow matching, CFG). `default_arch()` keeps attention ≤32² at any resolution (1024² →
  ch_mult (1,1,2,2,4,8)); `--grad_checkpoint` for native res.
- `data.py` — pooled **memmap cache** (`kappa.npy`+`meta.npz`+`norm.npz`) so training isn't I/O-bound on
  256 GB npz; stores *raw* pooled κ, applies normalisation on the fly. `KappaNorm` = per-z_s
  `arcsinh(κ/λ_z)` then standardise (tail is ~70σ). Split **by run** (param generalisation).
- `train.py` — plain torch + DDP (torchrun), bf16, EMA, warmup→cosine, grad-accum; portable checkpoint
  embeds arch+norm. `sample.py` — `WLEmulator.load(...).generate(params, z_s, n)`.
- CLIs: `bind-wlemu-cache` (`bind.cli.wlemu_cache`), `bind-wlemu-train` (`bind.wlemu.train`).
- Cache build is a **standalone OpenMPI script** `build_wlemu_cache.py` (pure numpy + mpi4py, NO bind/CLI
  import — like `bind_mpi_worker.py`): each rank pools `[rank::size]` runs into disjoint memmap rows; the
  κ norm is a streamed two-pass `MPI.Allreduce` (never materialises the 130 GB array). **Idempotent** — if
  `kappa.npy` is already written it skips the fill and only (re)computes `norm.npz`, so it also *finishes* a
  partial cache. `run_wlemu_cache.sh` mirrors **`data_generation/run_mpi_cpu.sh`** (`-p cca --constraint=rome
  -N 4 -n 64 --exclusive`, `module load python openmpi python-mpi`, `srun python build_wlemu_cache.py`).
  `run_wlemu_train.sh` mirrors **`run_train.sh`** (partition=gpu/constraint=h100, one `srun` task per GPU,
  `srun python -m bind.wlemu.train` — no torchrun; `_ddp_info` reads SLURM env).
  ⚠ Bug found in first 1024² run: the original `KappaNorm.from_raw` cast the whole 132 GB cache to float64
  *per redshift* → finalize hung/OOM-risk. Fixed to a chunked streaming reduction (verified byte-identical).
  An earlier over-engineered CLI/SPMD-phase cache path (`bind.cli.wlemu_cache`, init/fill/finalize) was
  removed in favour of the standalone script; `bind.wlemu.build_cache` remains as the single-process path
  for the notebook/tests.
- `examples/wl_field_emulator.ipynb` (`_build_wl_field_emulator_nb.py`): QUICK toggle trains a tiny model
  inline; generates maps, validates vs held-out truth (C_ℓ/PDF/peaks at matched res from the cache),
  feedback-response sweep + CFG. **Decision: kept native 1024² as a first-class flag** (user asked) —
  default arch + grad-ckpt make it tractable; 256² is the fast-iteration path.

Built on `lightcone` branch. Engine smoke-tested end-to-end on CPU (cache→train→generate); 512²/1024²
shape-checked; ruff clean. Not yet trained at scale — that's the SLURM run. Caveat unchanged: 256 design
points in 30-D → param interpolation is the hard part (50 realisations only teach the noise model); trust
feedback *ratios* over absolute high-ℓ ([[lightcone-kappa-upturn-aliasing]] CIC artifact lives in the maps).
Memory [[wl-field-emulator-cfm]]; relates to [[lightcone-emulator]] (the stat-level sibling),
[[wl-feedback-2d-latent]], [[sb35-sobol-cache]].

## 2026-06-26 — NEW **field-level** companion paper `paper_ksz_field.ipynb` (uses the ray-traced data)

User: "did we ever use the ray-traced data?" → no (the CAP paper uses per-halo patches). So built a
**sibling paper** that does (Track B): `examples/_build_ksz_field_nb.py` → `paper_ksz_field.ipynb`,
mirroring the 8-section arc but **field-level**, from the lux multi-plane ray-traced κ/τ/y maps. All
field stats are pre-assembled in `bind_sb35/emulator_dataset.npz` (253 nodes: tomographic Cl_κκ, **Cl_κy**,
Cl_yy, Cl_ττ, ready-made suppression S(ℓ,z_s) vs the paired DMO baseline, peaks, PDF/MF/WST, scaling
relations) — no new reductions. **Key finding (honest, differs from the per-halo result):** the field
response is **~1-D** (λ₁≈0.95, robust across every probe combo) — a single gas-ejection amplitude — the
**contrast** to the per-halo CGM's clean 2-d (inner+outer gas); the LOS projection + lensing kernel wash
out the radial 2nd dimension. §5 confronts **real DES Y3 × ACT** shear×y (`desact_data.npz`, ξ_γy via
n(z)-weight + ACT beam + J₂ Hankel): **BIND ~1.5–2× high** (TNG too gas-bound), the field-level
missing-baryon tension. **User caught a §5 bug**: an earlier draft used a 10′ "effective beam" that
over-smoothed BIND (peak shifted to 11′ vs data 4.5′) — replaced with the **physical 2.4′ ACT beam**;
exposed that BIND is also **too steep** (over-bound gas + the 5° field cuts low-ℓ at large θ). §6: the
**κ×y cross is the feedback driver** (response D: κy/yy/yτ ≫ κκ auto). §7/§8: Fisher → ~1 dominant
constrained direction (latent-limited), forecast pins it (dir1 99% in the §4 latent plane). 11 figs,
executes clean. ⚠ the lux RT *field* maps are randomized 5° realizations (not halo-indexed), so the CAP
paper still uses per-halo patches — a halo-stacked ray-traced CAP (Track A) is the open follow-up.
Memory [[ksz-p4-tau-not-velocity]], [[wl-sz-kappay-feedback]], [[wl-feedback-2d-latent]].

## 2026-06-26 — kSZ paper: **figure-revision pass** (user feedback on `paper_ksz_desi_act.ipynb`)

Reworked 8 figures of `examples/_build_ksz_paper_nb.py` (→ 13 figs, executes clean, 0 errors) per a
detailed read-through. **f1**: Sobol panel now in **physical (log) units** (ASN1/ASN2 with fiducial
★, read from `assets/SB35_param_minmax.csv` `LogFlag`); ELG promoted to first-class (snap-46 marker
in panel a; **both** BGS+ELG mass-function slices + the patch-reuse band in panel c). **f2**: panel
(a) switched from median-over-nodes to the **fiducial** stack (consistent w/ f3, no longer pre-empts
the response fan). **f3**: black line is now the **fiducial** (not the median); clean band labelled;
**new bottom row** = node-spread/fiducial vs R → quantifies that the feedback response is
**core-dominated** (core/outskirt spread ratio **13.4×**), confirming the user's read. **f4** (two iterations on user challenge): first labelled ê₂ via a gas-blind inner/outer-τ shape
residual, but the full decomposition showed that was an **overclaim** (in TNG gas content & profile
concentration are ~87% degenerate; the τ-residual gave only partial r=0.70). The user pushed to "find
what is true" → searched a battery of *physical* per-halo observables and found the **clean physical
labelling**: the 2-d manifold is **(inner gas, outer gas)** — ê₁ = inner $f_{\rm gas}({<}R_{500})$
(r=0.95, the ejection axis), ê₂ = outer $f_{\rm gas}(R_{500}{\to}R_{200})$ (r=0.95). The two zones are
only **r=0.24 correlated** (genuinely independent) and **reconstruct the plane** (R² 0.90/0.96); total
mass is common-mode across nodes (CV 0.0008) so f_gas variation = gas variation. Panels (c),(d) now
color the same plane by the two physical f_gas zones; labels propagated to f6e/§8/abstract.
**(key gotcha: f1(c) had named histogram counts `h`, clobbering the global Hubble `h` in the shared
kernel → broke r200phys/DA downstream.)**
**f6a (NEW)**: "which feedback fits" — prior-normalized params of the 49 kSZ-consistent nodes vs prior
+ fiducial; only the **wind/SN+IMF sector** is pulled (a ~6-knob *combination*; individual signs are
non-naive — corrected an initial markdown that wrongly claimed "higher wind energy"). **f6d** (reworked twice): the user stayed "sketched out" by the
"amplitude window" band-aid, so replaced it with an honest **2-panel** — (a) amplitude with the
$\pm0.1$-dex mass-syst band (Y∝M^5/3) that **dominates** the 1.5×; (b) shape *normalized at 2.25′*
showing BIND's painted pressure is genuinely **more extended** (CAP peaks 4.75′ vs data 3.5′).
Verified (loading the snap-67 y patches) that a **flat pedestal cancels exactly in the CAP** so bg-sub
does nothing — the extension is real, not removable. kSZ-consistent nodes (orange) + **χ²(kSZ) vs
χ²(tSZ) rank corr ρ=0.80** (same nodes fit both legs) retained. **f6e**: spelled out the linear-emulator χ² method; **circled the actual
Sobol node nearest the joint MAP** (run 230) + labelled its extreme knobs. **f8**: axes now carry
**`[prior σ]` units** + markdown explaining 0=prior centre, ν<1=tighter-than-prior, dir1/dir2 span the
§4 inner+outer-gas plane. **§7 expanded** (user: "we have 6 BGS + 3 ELG cuts, why use only 2?"): new
`examples/ksz_posterior_allcuts.py` builds the GP+emcee posterior from the CAP-ratio f̃_gas at **all**
DESI×ACT M* cuts (reusing the §6b per-node f̃_gas–M caches — no new painting), three nested configs
`ksz_posterior_cap_{bgs2,bgsall,all}.npz`. Result: 2→5 BGS cuts **does** open the wind/SN sector
(#params<0.85× prior 0→6, min ratio 0.87→0.75 — the low cuts are gas-poorer data, f̃_gas 0.31–0.35,
not copies) but **nothing is pinned**, and +3 ELG (8 pts) barely moves it → **latent-limited**, the §4
prediction. Honest caveat kept: low BGS cuts are host-mass floor-limited (logM 13.4) so they fold a
mass-modelling systematic, which is why §6 leads with the 2 high-M* bins. New helper
`examples/_reduce_tauy_fiducial.py` builds the fiducial τ/y profile (snap 85+46) the response-fan cache
lacked. **"Ray-traced" overclaim corrected** (user: "did we ever use the ray-traced data?"): we did
NOT — every confrontation (κ, f̃_gas, y-CAP, τ) reads **per-halo painted patches** (`generated_patches`/
`thermo_patches`) from `composite_slab*.npz` at a single snapshot per tracer; κ is even single-plane
(Σ_crit at z=0.18), no LOS integration. Replaced all 9 "ray-traced" claims (abstract, §1, §8, closing)
with "painted / 1-halo projected"; the real enabling novelty is **differentiability** (continuous in
feedback → Sobol → GP), not ray-tracing. Noted the 1-halo patch is *well-matched* to the CAP (which
cancels the 2-halo background anyway), so it's appropriate, not just a shortcut. Memory
[[ksz-p4-tau-not-velocity]].

## 2026-06-26 — kSZ paper: **tSZ mass-selection bug** (the "2.6×" was Msun-vs-Msun/h) + CAP propagation

User: "the tSZ is soooo far off, something is wrong." It was. §6d selected LRG hosts by `logM200 ∈
[13.2,13.5]` in **Msun/h**, but the DESI-LRG host mass from ACT CMB-lensing is **logM200 = 13.18 Msun/h**
(Sailer+24) — the old band was ~0.2 dex too high (an Msun-vs-Msun/h slip). Because **Y ∝ M^5/3**, that
inflated the y-CAP to a spurious **2.6×**. At the correct mass (mean-stacked to match the data stack)
BIND y-CAP = **1.51× the data**, with the **±0.1-dex mass systematic alone spanning 1.35–2.05×** → the
tSZ amplitude is **mass-selection-dominated, NOT a clean feedback probe**. Fix: `_reduce_ycap_lrg.py`
band → [13.0,13.35] Msun/h, median→**mean** stack, + a ±0.1-dex mass-syst band saved (`fid_msys_lo/hi`).
**§6d reframed**: dropped the broken cross-sample "density×mass×T" decomposition; new right panel shows
the Y∝M^5/3 mass-sensitivity (data matched within the mass uncertainty); headline "mostly too much gas,
mildly too hot" → "tSZ confirms the kSZ gas excess (~1.5×) but is mass-systematic-dominated". **§6e
reframed**: with the tSZ mass-corrected the kSZ & tSZ legs now **agree** at TNG's strong-fb edge — the
earlier "tSZ pulls beyond / inter-probe tension" was the logM bug; removed. **CAP propagation** (user
asked): `_reduce_fgas_lowmass.py` §6b switched its 1.4–2.2 r200 bg-annulus → the **exact CAP** (disk
minus equal-area ring), reran snap085+046 (centrals now match §6's 0.77 at logM 13.6). §6c κ correctly
left as a profile (its data, Fig7 κ≈0.01, is a convergence profile, not a CAP-ratio). Intro/closing
headlines updated. Lesson: **tSZ Y is hostage to halo-mass selection (M^5/3); always confirm host mass
in Msun/h and mean-stack.** **Tooling**: `_reduce_fgas_lowmass.py` gained `--nproc` (multiprocessing
Pool over the 256 independent nodes, fork-inherited FoF) + `run_fgas_lowmass.sh` (SLURM array 0–1 =
BGS/ELG snaps, exclusive cascadelake node, `--nproc=$SLURM_CPUS_ON_NODE`) — the ~1.5 h serial reduction
runs in ~1–2 min. (Claude's shell is cpuset-pinned to 1 CPU, so local parallelism there is a no-op; run
on an unpinned login shell or via SLURM.) `_reduce_ycap_lrg.py` likewise gained `--nproc` + bumped to all
256 nodes, with `run_ycap_lrg.sh` (single exclusive node, snap067). **Plot style** (user pref): §6b and §6d
now draw **every one of the 256 runs as faint light-blue lines** + a thick fiducial (instead of a 16–84%
band); §6b dropped the truth line; §6d collapsed to a single panel (mass systematic reported in a text box,
not a side panel). Final notebook = **12 figs, 30 cells, executes clean**; §6e: kSZ & tSZ MAPs agree at
ê₁≈−7 (implied f̃_gas≈0.53), inside the manifold. Memory [[ksz-p4-tau-not-velocity]].

## 2026-06-25 — kSZ paper: **CAP-aperture reconciliation** (headline correction) + §6e latent-from-data

Two user catches collapsed the kSZ "0/256 / 2× / beyond the manifold" headline. **(1) The §6b dotted
"strongest fb" line was a per-bin MINIMUM** over 256 nodes — a non-coherent stitch (different nodes per
mass: `[106,205,205,205,173,42,42]`), not a feedback scenario. Removed from §6/§6b. Checking a *single
coherent* node instead: in the compensated aperture a node fits the BGS data near-perfectly → flagged the
"0/256" as aperture-dependent. **(2) Checked the paper** (Ried Guachalla+25 Part II, arXiv:2604.19745
§III.2.1): their f̃_gas is a **CAP-ratio**, "the CAP filter for the gas at the virial radius … [over] the
CAP filter for the matter" — *compensated*, not cumulative. So the cumulative aperture used in §6/§6e was
**wrong** (it keeps the ~50 Mpc/h LOS the CAP cancels, inflating BIND 0.86→ and manufacturing the tension).
**Fix = exact-CAP recompute** (`examples/_reduce_fgas_cap.py` → `fgas_cap_mstar_snap085.npz`, disk minus
**equal-area** ring, CAP_gas/CAP_mat at r_vir, 256 nodes). **CAP-correct result: fiducial CAP f̃_gas(r200)=
0.77/0.84 (M\*>11.0/11.25) vs data 0.44/0.60 → 1.76×/1.40× (~2σ); ~20% of nodes (49/65 of 256) consistent
within the (large) errors; best single node χ²≈0.1 → feedback/data-limited, NOT "0/256".** Swapped §6 +
§6e's kSZ leg to the CAP cache. **§6e reframed**: kSZ-only pins the ejection latent to TNG's strong-fb edge
**INSIDE** the manifold (MAP ê₁=−6.7, f̃_gas 0.54, 19/256 in 95%); **tSZ pulls BEYOND** (pressure wants more
ejection than density) → **inter-probe tension = §6d in latent form**. **§6d updated**: density 2.0×→1.7×
(CAP) ⇒ T_e 1.0×→**1.2×**, so "too much gas, not too hot" → "**mostly too much gas, mildly too hot**".
Updated the intro/closing headlines accordingly. ⚠ Remaining ripples (flagged, not done): §6c κ uses a
smoothed bg-sub, not the exact CAP filter (decomposition mass leg 1.3×); §6b still uses a 1.4–2.2 r200
annulus (amplitude ≈, trend is the point); §5 eROSITA is a separate (tighter) X-ray constraint. Notebook
12 figs / 30 cells, executes clean. Memory [[ksz-p4-tau-not-velocity]].

## 2026-06-25 — `lightcone`: low-mass (10¹²–10¹³) baryons via **reuse-only**, not new generation

Goal: get sub-10¹³ halos into the lightcone maps. Naive per-halo extraction → generation
is infeasible: at 10¹¹ there are ~228k halos/snap (78× the 2,933 at 10¹³) ≈ 1.2 TB export
and ~78× GPU cost; at 10¹¹ halos *fill space* (~53 per 6.25 Mpc/h patch) so per-halo
generation is ~50× redundant. Quantified (snap_096, scratch scripts) that every existing
10¹³ patch is a 6.25 Mpc/h cutout already containing the painted/true hydro of the smaller
halos sharing its column → **capture fraction** of the 10¹²–10¹³ band: **28.7%** within the
production circular paste (4×R200, median 1.76 Mpc/h), **51.6%** at the full 6.25 Mpc/h
footprint; the remaining ~48% are field halos too far from any cluster.

Decision (user): **reuse-only, accept ~49%, band 10¹²–10¹³, mirrored on truth** (not 10¹¹,
not full-box tiling, not new generation). Symmetry that makes it clean: neither side needs a
new low-mass job — BIND patches are on disk; `bind-truth-halos` at the *same* 10¹³ halos
projects the full hydro so its patches carry the true low-mass hydro in the same footprints.

Built [examples/lightcone_lowmass_reuse.py](../examples/lightcone_lowmass_reuse.py):
cross-matches the 10¹²–10¹³ FoF halos to existing patch centres (same transform/slabs),
tags captured ones, reads each captured halo's aperture (gas/Y/star/T) straight out of its
host patch — works on BIND and truth dirs, matches by `halo_id` → per-halo BIND-vs-truth
table. Validated on fiducial BIND snap_096: reproduces 28.7%/51.6%, clean Y–M & Gas–M
(median host distance 1.87 Mpc/h → these are *off-center* halos, the regime the truth
comparison validates); self-comparison smoke test = ratio 1.000 / 0.000 dex. Next:
`sbatch --array=0 run_truth_halos.sh` (note default array `1-19` **skips IDX 0 = snap_096**)
→ truth patches in `bind_science/runs/truth/run_0000/snap_096`, then rerun with `--truth_dir`.

Per-halo BIND-vs-truth (snap_096, captured 1e12–1e13): **gas 1.034× / 0.074 dex**, **Y 0.986× /
0.223 dex** (the WL/tSZ-relevant fields, well recovered off-center), stars 0.75× / 0.54 dex
(two-head channel, expected; irrelevant to maps). Map-level (full truth lightcone now built,
all 20 snaps): re-composite at widened footprint = `--r200_factor 99` (circular taper clamps
to half-patch 3.125 Mpc/h = the 51.6% capture). snap_096 totals: **widening boosts total tSZ y
+4.0% identically for BIND & truth**; **BIND/truth y = 0.963 at *both* footprints** (no new
low-mass bias); matter total unchanged (patch mass-match) → WL low-mass signal lives in the
*power-spectrum redistribution*, only visible after lux. → full ray-trace recipe: expose the
BIND-fiducial + truth patches as 4 symlinked "designs" (`bind_prod/bind_wide/truth_prod/
truth_wide`) under `bind_science/runs/`, run all through `run_sobol_{paste,lux,stats}.sh`
(identical RT seed 1992 → paired realizations) with `DESIGN`/`R200_FACTOR` envs; non-destructive
(writes to new run dirs). Widened ≠ complete low-mass baryonification (~48% field halos stay raw
DMO) — BIND/truth comparison still clean, absolute effect a lower bound.

## 2026-06-25 — `lightcone`: refocused the P4 paper notebook on the *feedback-latent* story; methods-forward (10→8 figs)

The 2026-06-24 multi-probe build had grown `examples/paper_ksz_desi_act.ipynb` to 10
figures spanning kSZ→tSZ→field-level κ×y→shear×y→cosmology-anchor→real-DES×ACT — two
papers' worth, hard to follow. Per the user ("sell that BIND has continuous lightcones
over the 30-d astro space; the latent direction is right; the multiprobe is the
confusion"; then "set up the BGS bin better, detail the methods, add plots") I rewrote
the builder `examples/_build_ksz_paper_nb.py` into a single linear **kSZ+tSZ latent**
argument, **methods-forward** (8 figs, executes clean ~30s, scienceplots/no-titles,
explicit Question/Math/Citations scaffolding per section):

- **§1 DATA** (Fig f1_lightcone, 3-panel) — TNG300 DMO lightcone (20 snaps z=0.03–2.44,
  shared halos), 256-node Sobol design, halo mass function with the BGS science bin +
  DESI BGS/LRG/ELG host masses vs the 1e13 floor (ELG below floor).
- **§2 METHODS** (Fig f2_pipeline, 3-panel) — per-halo paint→radial stack by mass + clean
  band 0.3<x<1.5; M\*-matching via BIND SHMR (M\*>11.0/11.25 → logM200 **13.56/13.81**);
  the CAP filter schematic.
- **§3 Response fan** (τ,y across 256 nodes, BGS bin, colored by f̃_gas).
- **§4 Latent** (3-panel) — τ+y response is **2-d (97% var**, λ=0.51/0.46, sharp knee);
  PCA plane **oriented to the f_gas gradient** → **ejection axis (corr f_gas =0.94**,
  wind/SN+IMF) ⊥ **shape axis (corr =0.00)**. Replaces the raw-PCA labels (degenerate at
  this bin) with a variance-preserving target rotation.
- **§5** f_gas vs eROSITA (0% of the box reaches strong-fb). **§6** CAP + σ_v-free f̃_gas
  vs DESI×ACT (BIND ~2.7× too gas-rich within r200). **§7** GP CAP posterior
  (latent-limited = the §4 prediction).
- **§8 MONEY (forecast)** — SO/CMB-S4×DESI, now **kSZ vs tSZ vs joint**: each probe alone
  leaves dir2 prior-wide (var ~0.9), joint pins both (ν≈**0.10/0.16**) → complementary,
  not redundant. Best-constrained forecast dir lies **0.81** inside the §4 latent plane
  (random 0.26). Closing cell states question/novelty/why-BIND explicitly.

Key fix: the science **BGS mass bin is now logM200∈[13.4,13.8] (center 13.6**, ~700
halos), the honest match to the M\*-selected DESI host mass — the earlier build used the
[13.8,14.2] bin with a hardcoded "logM200~13.6" fallback print (the tauy npz has no
logM200).

**§6 f̃_gas confrontation reworked (user push-back: interior CAP bias too high; right
panel only 2 points; show the fiducial not the median).** Diagnosis: the ~5× interior CAP
offset is NOT clean feedback — θ(r200)≈2–2.4′ so the smallest aperture is the total column,
τ^CAP ∝ M_gas=f_gas·M_halo and τ^data ∝ 1/σ_v, neither cancels (both cancel in the σ_v-free
f̃_gas, only ~2.7× high) → CAP interior = robust 2.7× × O(2×) mass(SHMR)/σ_v systematic; the
**ACT beam is ruled out** (2D-convolve τ FWHM 1.6′ → <10%, ratio 6.0→5.4). Swapped panels:
LEFT now the robust σ_v-free f̃_gas, RIGHT the CAP with a σ_v=200–400 band + systematics
caveat. New `examples/_reduce_fgas_radial.py` computes f̃_gas(R/r200) as a **continuous
curve from the painted per-halo patches** (`generated_patches` [DM,Gas,Stars], 128px @
0.0488 Mpc/h native = 6.25 Mpc/h; r200=halo_r200/pix; projected ΣM_gas(<R)/ΣM_tot(<R)), for
the **fiducial** (`bind_lightcone_tng`) + **all 256 SB35 nodes**, **selected by BIND's painted
central M\* matching the TWO DESI cuts** (M\*>11.0/11.25, Stars within ~1px≈50kpc/h → mean
logM200 13.60/13.82, apples-to-apples — same selection as the data + the CAP panel) → cache
`ksz_confront/fgas_radial_mstar_snap085.npz`. Validated: projected f̃_gas(r200)=0.86 vs parquet
3D 0.88. Shown as **per-cut fiducial curve + SB35 feedback band** (NOT the Sobol median): at
r200 fiducial 0.86/0.90, min-of-256 0.52/0.53, data 0.28/0.36 → fiducial 2.5–3.1×, every node
≳1.5–2× too gas-rich, **0/256 reach the data**. (Fixed the misleading "mass-independent"
label — f̃_gas IS selection-dependent — to "σ_v-free · M\*-matched". The mgas_cum/mtot_prof
arrays in the mstar npz are NOT physical f_gas — recompute from patches.)

**§6b — low-mass extension via patch reuse (reaches the ELG regime).** New
`examples/_reduce_fgas_lowmass.py`: the 6.25 Mpc/h painted patches already contain the
smaller halos sharing their column, so cross-matching the 10¹²–10¹³ FoF halos
(`/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output/groups_085`, 26k in band,
~49% captured) to existing patches recovers them with NO new generation
(reuses `examples/lightcone_lowmass_reuse.py`). ⚠ **bg-subtraction is essential**: the raw
projected aperture → f̃_gas≈1.1 flat (the 50 Mpc/h LOS dominates a small off-centre
aperture); a compensated disk−(1.4–2.2 r200 annulus) recovers the halo. **Validated vs
truth** (true TNG hydro at the same footprints, snap 85, `runs/truth/run_0000`): BIND/truth
1.04–1.16 across 10¹²–10¹⁴·⁵ → off-centre painting faithful to ~10%. Result (cache
`fgas_lowmass_snap085.npz`, fiducial+truth+24 nodes): f̃_gas–M rises 0.6→0.9, smooth across
the reuse↔central boundary; the **SB35 feedback band fans open at low mass** (±0.05 @10¹⁴·⁵
→ 0.42–0.79 @10¹²; strongest-fb reaches the data ~0.3 at low/mid mass, while no node comes
close at group scale) → low mass/ELG is the discriminating regime. TNG ~3× too gas-rich vs
kSZ BGS across 2.5 decades. Caveats: captured halos near-cluster (median 1.9 Mpc/h), not
field (ELGs are field); BIND ~10% gas-rich vs truth; flat within the decade. Notebook now
9 figs (added f6b_lowmass).

**Real DESI×ACT data + an angular-distance bug (the user's units question).** (a) Got the
actual measurements — Zenodo 19160138 (Ried Guachalla+25 Part II, BGS+ELG, w/ covariances) →
`ksz_confront/desact_zenodo/` (Fig8=f̃_gas(θ), Fig6=T^CAP(θ)) and wired the **real values+errors**
into Fig 6 + §6b, replacing the hand-digitized 2-cut FGDAT. (b) **Found+fixed a real bug**: the
notebook's `DA(z)` had a spurious extra `/h` → angular-diameter distance 1/h=1.476× too big →
BIND's r200→θ ~0.68× too small; the data's own R/θ proves D_A(0.26)=866, D_A(1.17)=1761 (matches
no-/h). Masses are Msun/h (handled by r200phys's `/h`); (1+z) was fine. **Impact: θ(r200) 2.0'→3.0',
data f̃_gas(r200) 0.28→0.44–0.60, robust tension ~3×→~2× (1.5–2.0× over the BGS M\* cuts).** With the
correct DA the CAP absolute ratio balloons to ~8–20× → CAP amplitude is normalization-dominated
(σ_v · velocity-fidelity r/r_fid · T_CMB · M_halo). (c) **Fig 6 CAP panel DROPPED**: a naive
τ=T^CAP/(T_CMB σ_v/c) conversion is INVALID (kSZ measures a velocity-weighted T, not τ; folds in
σ_v+M_halo → spurious ~10×). The data's f̃_gas (Fig8 `ratio`) is ALREADY the authors'
velocity-marginalised, M_halo-robust gas fraction = the right comparison. Fig 6 is now a **single
f̃_gas(θ) panel** (BIND fiducial+256 band+strongest-fb vs real Zenodo f̃_gas, ~2× at r200).
(d) **ELG done — at its own redshift.** BIND IS painted at snap 46 (z=1.155 ≈ ELG z=1.17;
`_reduce_fgas_lowmass.py --snap 046 --snap_idx 12`, 256 nodes). §6b is now a **two-redshift**
figure: z=0.18/BGS (r200) | z=1.16/ELG. ELG halos are tiny so the kSZ only reaches ~2.8 r200;
at that **matched aperture** BIND fid f̃_gas(2.8 r200)=0.88 (truth 0.77) vs ELG data ~0.46 → **~2×**.
So the tension holds across mass (clusters→ELG) AND redshift (z=0.2→1.2). Notebook 9 figs, executes
clean. Env-bias check (full TNG300 hydro catalog): near-host vs field f̃_gas 0.408 vs 0.409 (<0.3%) →
reuse sample representative.

**(e) All DESI M\* cuts** in §6b: BGS 5 cuts (9.5–11.25, the 3 lowest pile up near the 1e13 floor —
floor-limited host mass), ELG 3 cuts (9.0–10.0); host logM200 per cut from BIND painted central M*.
**(f) §6c MASS ANCHOR — CMB-lensing κ** (`examples/_reduce_kappa_bgs.py`, cache `kappa_bgs_snap085.npz`).
The LRG Part I paper (2604.19744, TABLE III) shows TNG's kSZ needs a 0.367 amplitude rescaling (2.7× too
high) but flags that this is **mass-degenerate** (would imply implausibly low mass). So we test the mass
directly: κ ∝ total mass. BIND κ = Σ_tot/Σ_crit from the painted DM+Gas+Stars patches (CMB src z_s=1100,
Σ_crit=2.72e15 Msun/Mpc²), L<3000-smoothed (σ_θ~1/L), bg-subtracted (outer annulus), M*-matched, vs the
BGS×ACT κ(θ) (Part II Fig.7 = `Fig7_bgs_y3` — it's **κ, NOT tSZ**; the ~0.01 values are convergence).
**Result: BIND κ ~1.31×/1.29× the data at the 1-halo scale (θ~2.25'), vs ~2× in f̃_gas → masses right to
~30%, the f̃_gas tension is GAS not mass** (de-degenerates TABLE III). Caveats: BIND patch is 1-halo only
(falls below data at θ≳3', the 2-halo); inner θ<1' high (resolution+L-cut) → compare at 1-halo only.
**(g) §6d tSZ PRESSURE LEG — ADDED (the 3-probe closure).** Real ACT×DESI photometric-**LRG** y-CAP
(arXiv:2502.08850, Zenodo 14706729, `tsz_zenodo/fig3.csv` = cumulative Compton-y CAP vs R_ap 1–6'),
confronted at one representative LRG snapshot (snap 067, z=0.50, logM200 13.2–13.5). ⚠ the per-pz columns in
the Zenodo CSVs are **identical across pz1–4** (export collapse — means genuinely duplicated, max diff 0.00;
only the per-z covariance npy's fig6/fig9 differ), so we confront a single z, not the 4-bin tomography. BIND y
from the painted thermo patches (`thermo_patches[:,0]`), ACT-beam-convolved (1.6' FWHM Gaussian), compensated
aperture (disk − equal-area ring). Reducer `examples/_reduce_ycap_lrg.py` → `ycap_lrg_snap067.npz`.
**Result: BIND pressure ~2.6× the data at the 1-halo scale**; the 3-probe closure **y = f̃_gas · κ · T_e**
decomposes density 2.0× × mass 1.3× × pressure 2.6× ⇒ **T_e ≈ 1.0×** → "**too much GAS, not too HOT**"
(an ejection problem, not a heating problem — answers the heating-vs-ejection question the f̃_gas tension alone
couldn't). Notebook now **11 figs** (added f6d_tsz_pressure), **28 cells**, executes clean. Only ADDED §6d
(existing plots untouched).

**(h) §6e — THE 2-D LATENT CONSTRAINT FROM THE DATA (the capstone; `f6e_latent_data`).** Projects the REAL
DESI×ACT data into the §4 feedback manifold — the paper's thesis made quantitative. Per Sobol node we have its
§4 latent coord **E=(ejection ê₁≈f̃_gas, shape ê₂)** *and* its painted observable; fit the linear feedback
response o(E)=a+b·ê₁+c·ê₂ per bin (a 2-d emulator in the latent), then χ²(E)=[d−o(E)]ᵀC⁻¹[d−o(E)] against the
real measurement+covariance. Two legs: **kSZ** = the full f̃_gas(θ) profile (Zenodo 19160138, BGS M\*>11.0,
**full 13×13 cov**, restricted to 1-halo θ≤1.4·θ(r200)); **tSZ** = the y-CAP 1-halo amplitude (R≤2.5', diagonal
err — amplitude only, NOT shape). **Result: joint MAP ê₁=−13.2 vs node-cloud range [−9.4, +3.1] → the data sit
PAST the gas-poor edge** (implied f̃_gas=0.30 vs node min 0.48; ~1.4 node-σ beyond), **0/256 nodes inside the
joint 95%**, χ²_min/dof=23/5. kSZ-only = an open band along ê₂ (pins ejection, slides on shape); tSZ-only tilts
differently and closes it; the joint is a tight ellipse at the gas-poor extreme → **today's data pin ~1 of the
2 latents (ejection); the shape latent is left for the §8 forecast.** This is the latent-space form of the
"0/256 reach the data" result. **§6d y-CAP SHAPE explained:** BIND is high at every R but *more* at small R
(4.5×@1' → 2.6×@3.5') ⇒ BIND y is both higher AND more centrally peaked; the data is flatter/peaks at ~1.7 r200
because the **photometric-LRG stack is smoothed by miscentering + photo-z LOS spread + the ACT beam** (which the
authors model and marginalise) — hence we compare only the 1-halo amplitude, not the profile shape. Notebook now
**12 figs** (added f6e_latent_data), **30 cells**, executes clean. Only ADDED §6e.

Cut Figs 7–10 (WL κ×y, shear×y, cosmology anchor, real DES×ACT) — that's the
separate WL×SZ program (`docs/wl_tsz_plan.md`). Old builder/notebook backed up in
scratchpad; orphaned figure PDFs removed from `examples/figures_ksz/`. Underlying
reductions/engines unchanged (`ksz_confront/*.npz`, parquet). Memory
[[ksz-p4-tau-not-velocity]], [[lowmass-reuse-capture]], [[sobol-feedback-latent]], [[wl-feedback-2d-latent]].

---

## 2026-06-24 — `lightcone`: P4 → multi-probe WL×SZ (tSZ + field-level κ + shear×y) — overnight autonomous

Extended P4 from kSZ-only to a multi-probe WL×SZ feedback program (8-figure paper
notebook `examples/paper_ksz_desi_act.ipynb`, scienceplots/no-titles, executes clean).
New: `ksz_posterior_multiprobe.py` (per-halo kSZ τ + tSZ y forecast, realistic AR(1)
cov, constrained-DIRECTIONS corner — feedback is ~2-D so per-param corners are
prior-dominated); `ksz_thermo_decomp.py` ((κ,τ,y) decomposition — per-halo κ = mass
anchor); `wl_sz_multiprobe.py` (FIELD-LEVEL WL×SZ on the assembled
`emulator_dataset.npz`); `wl_sheary_fit.py` (cosmology-robust tomographic shear×y fit,
real-data hook); `perhalo_plus_kappay.py`. GP training parallelised (joblib, no CV)
→ ~20× faster; node HAS a V100S (CPU posteriors don't use it).

**Findings.** (1) WL κ AUTO-stats (suppression S(ℓ)+peaks) barely constrain feedback;
the **κ×y cross + tSZ y dominate** (the cross is the most feedback-sensitive observable,
Sobol spread 0.76) → at fixed cosmology pins 2 dirs to var 0.06/0.11. (2) But the κ×y
signal is AMPLITUDE-dominated → degenerate with σ8/Ωm; amplitude-marginalised shear×y is
modest (var 0.25) → **WL×tSZ feedback needs a cosmology anchor**; tomography helps. (3)
κ×y is REDUNDANT with per-halo y (both pressure) → no gain folding it in. **Meta:
multi-probe feedback gains saturate (strong probes redundant + feedback ~2-D); levers =
SNR, cosmology prior, pressure×independent-density.** Per-halo κ stack = feedback-blind
mass anchor (user's insight). Plan `docs/ksz_desi_act_plan.md` D6-D8; memory
[[wl-sz-kappay-feedback]].

**FIRST REAL-DATA fit (D9-D10).** Cosmology anchor (`wl_sz_cosmo_anchor.py`): κ-auto
(∝A²) pins the amplitude (A 1.00±0.06→0.999±0.008), freeing κ×y (∝A¹) → feedback dirs
var 0.25→0.15 (toward the fixed-cosmo 0.06). Then got REAL DES Y3 × ACT shear×y data
(Pandey `shivampcosmo/ACTxDESY3` `DES_ACT.fits` → `desact_data.npz`, 26.5σ, joint cov,
DES n(z)). **Found+fixed a real bug:** Pylians `XPk_plane` norm ≠ `Pk_plane` (const
F≈1.2016e7) → stored κ×y cross ~1e7 too low (autos fine, pixel corr(κ,y)=0.52);
patched `bind.inference.stats.power_spectrum` (forecasts unaffected; stored cross +
emulator dataset need regen). **Result (self-corrected):** a first "BIND ~2× =
feedback too weak" figure was a PIPELINE ERROR (omitted the y-map beam — BIND y is
beam-free, data y ~10' effective — + no scale cuts). With y-beam+θ>8' cut, BIND matches
the data's intermediate-θ peak but over-predicts large θ ~1.5-2×, a residual entangled
with cosmology (TNG vs DES σ8), beam/cuts, and κ×y norm → NO clean feedback claim from
shear×y yet; the robust feedback signal stays inner-CGM (kSZ-CAP, X-ray f_gas).
(`desact_sheary_realfit.py`, paper Fig 10 corrected.) Notebook 10 figs. Plan D9-D10.

## 2026-06-24 — `lightcone`: P4 D4-D5 — kSZ amplitude resolved (CAP) + first feedback posterior + σ_v-free f~gas

Closed the amplitude leg in the DESI×ACT *observable* and built the posterior.
`examples/ksz_cap_compare.py` + `ksz_posterior_cap.py` + `ksz_fgas_profile.py`; M*-
matched stacking via BIND **painted central M*** (Stars<50 kpc/h, varies per node)
with BIND SHMR setting M200c (`run_ksz_tau.sh MODE=mstar`).

**Resolved the spurious ~30×:** the data are the compensated-aperture signal
T_kSZ^CAP (τ in arcmin²), NOT local τ vs the GNFW. Paper-confirmed normalization:
(r/r_fid) is the velocity-recon *fidelity* coeff (=1 at fiducial), CAP=disk/equal-
area-ring, σ_v^true≈300 km/s (T_CMB σ_v/c=2727 µK). Proper CAP comparison: **BIND
~2-6× above the data at small/mid aperture, ~1× at the largest** = gas too centrally
concentrated, feedback too weak (~2-4σ inner). **σ_v-FREE capstone (f~gas):** BIND 3D
f~gas(<r200)=0.88 vs data ≈0.32 → **BIND ~2.7× too gas-rich within r200**; data reach
cosmic only at ~4 r200 (gas ejected). Coherent across f_gas/CAP/f~gas + Siegel/Bigwood.

**First kSZ feedback posterior** (GP on CAP, CV R²≈0.83 — emulator works): single BGS
bin data-limited (top 0.86× prior); **joint M*>11.0+>11.25** modestly better (0.77, 5
params<0.85, wind/SN sector). Verdict: pipeline succeeds, per-param constraint is
kSZ-SNR-limited not BIND-limited; needs published cov + LRG (Part I) + ELG (below
BIND's 1e13 floor → unreachable). Figs `ksz_{cap_matched,fgas_profile,posterior_cap_*}`.
Plan `docs/ksz_desi_act_plan.md` D4-D5. Memory [[ksz-p4-tau-not-velocity]].

## 2026-06-24 — `lightcone`: P4 D2-D4 — tau/y profiles vs Hadzhiyska+26 GNFW (full 256-node, clean-α)

`examples/ksz_tau_gnfw.py` + `run_ksz_tau.sh` (openmpi/srun MPI batch, restart-safe
per-(snap,node) shards, the `run_sobol_atlas.sh` idiom). Encodes the Hadzhiyska+26
Table II (user-supplied) and forward-projects their density GNFW (Eq 26-27, fixed
γ=-0.5 x_c=0.7, free {ρ0,α,β}, r/r200c) into projected τ(R/r200c). One reduce pass
stacks BIND τ (electron column) AND y (tSZ) by **halo mass** across all 256 Sobol
nodes at the BGS-z (snap 85, z0.18) and ELG-z (snap 46, z1.16) slices.

**Results (256 nodes).** D2 shape: in the clean band x∈[0.3,1.5] BIND's τ shape is
consistent with the massive BGS bins (M*>11, β≈4.8-5.2) and ELG (β≈4.5), shallower
than steep BGS_all (β=7.3); α–β plane β overlaps (BIND 4-6 vs data 4.5-7.3) but α
offset (BIND 1-3 vs ~0.2) — **α≈0.2 is a near-singular GNFW corner, degenerate with
the fixed core, so the valid comparison is the profile overlay + β, not (α,β)**.
D3 y/T: ELG-z pressure y > BGS-z at fixed mass (ρ_cr(z)); T_e proxy ≈1-2×10^7 K.
D4 amplitude: absolute τ rises with halo mass, Sobol feedback spread; BIND
f_gas/f_b≈0.78-0.80 within r500 (gas-rich vs kSZ "low BGS f_gas", cf D1).
**⚠ kSZ-inferred f_gas from the GNFW table is unphysical as transcribed** (BGS_all→0
via (3ρ0/200)∫x²GNFW) — ρ0 units wrong; defer absolute kSZ-f_gas to the paper's
reported table. Figs `tau_shape_vs_hadzhiyska,alpha_beta_plane,y_and_temperature,
tau_amplitude.png`. Plan: `docs/ksz_desi_act_plan.md` D2-D4. Memory [[ksz-p4-tau-not-velocity]].

## 2026-06-24 — `lightcone`: P4 kicked off — kSZ/gas vs DESI×ACT + eROSITA (feasibility + D1)

Assessed feasibility of "Project 4" (BIND kSZ profiles vs DESI DR2 × ACT DR6) against
the actual code/data and corrected the spec's premise: **BIND has no gas velocity
field** — its "kSZ" is the velocity-free electron-column `tau` (`paint_tauplane.py`,
`lightcone_maps.py`); a real ΔT_kSZ map is unbuilt (needs DMO-velocity surrogate). But
`bind_sb35` **is** the TNG300-L205 ray-traced 256-node Sobol set, so the velocity-free
legs (f_gas, τ/y profiles, κ×τ) are the headline and need no new physics. Reframed P4 to
"baryon fractions + τ/y profiles vs DESI×ACT kSZ + eROSITA". Plan + phased deliverables +
kill-gates in `docs/ksz_desi_act_plan.md`.

**D1 done** (`examples/ksz_fgas_confront.py`): f_gas(M500,z) for BGS/LRG/ELG windows
across the 256 Sobol nodes vs Eckert+19 / Popesso+24-eROSITA, off
`analysis_cache/integrated.parquet` (8.6M halo-instances, pure query, CPU). Result: the
whole Sobol band sits at f_gas≈0.11–0.15 at logM500~13.5; **0% of nodes reach the eROSITA
strong-fb band** even at the 2.5% feedback edge → reproduces the Siegel/Bigwood
(2509.10455) tension across a continuous parameter space. Caveat to carry: BIND f_gas =
total gas, eROSITA = hot X-ray gas, kSZ = total electrons (literal Hadzhiyska GNFW
comparison = D2). ELG leg mass-floor-limited (M200≥1e13). Figs `fgas_M500_vs_erosita.png`,
`bgs_elg_dichotomy.png`. Memory [[ksz-p4-tau-not-velocity]].

## 2026-06-24 — `lightcone`: capstone — 2 baryon templates necessary & sufficient for S8

**`examples/cosmo_bias_capstone.py`** (extends `lightcone_cosmo_bias.py`): the headline
cosmology result tying the whole investigation together. Schneider+2020 Fisher bias on
LSST-Y10 cosmic shear (5 tomographic κ autos, CCL halofit derivs, Gaussian cov+shape
noise) using BIND ΔC=C_fid·(S−1) for all **216** runs, plus N baryon-template nuisance
params (leading SVD modes of the 216 ΔC = the WL feedback manifold) swept N=0…10.

**Result — exactly 2 templates are necessary & sufficient.** Ignoring baryons biases S8
by up to **2.0/3.6/6.0σ** (ℓmax 2000/3000/5000), bidirectionally. The ΔC manifold is 2D
(PC1 89-92%, PC2 5-10%, PC3 <1%). N=1 leaves up to 0.67σ residual (insufficient); **N=2
nulls it to ≤0.07σ for all 216 runs, held-out validated** (templates from even runs kill
the bias on odd runs, ≤0.08σ); N>2 adds nothing. Cost: the 2-template floor inflates
σ_S8 ~3× (shear-auto only); over-modeling (N=6 →×4.3, N=10 →×5.8) is pure cost ⇒ a
minimal 2-param baryon model beats a 6-7 param BCM. tSZ pins one template (its SN/thermal
axis) → prior that buys back part of the floor. Fig `cosmo_bias_capstone.png`. Caveats:
TNG-only amplitude (2D generality from Lin+2026), shear-auto only, Gaussian/linear,
P_hyd=P_dmo·T² separability. See memory [[baryon-cosmo-bias-2template]].

## 2026-06-24 — `lightcone`: WL feedback latents (Lin+2026 for WL) + multi-probe test

Built the WL analogue of Lin+2026 ("One latent to fit them all", arXiv:2509.01881)
on the 216-run SB35 suite (TNG, fixed cosmology — so across-run variation of any
statistic *is* the feedback response; no cosmology conditioning needed). Two scripts:

- **`examples/wl_feedback_vae.py`** — a small β-TCVAE (Chen+2019 TC decomposition,
  full-batch, KL annealing) on the binned κ suppression S(ℓ,z_s) → disentangled 2D
  latent (RMSE 1.3%). Reproduces their Fig 1 & 3 for WL: **Latent 0 = BH axis**
  (BHRadEff −0.43, QuasarThreshold, BH accr/Edd) acts ~uniformly across z_s;
  **Latent 1 = SN/wind axis** (VarWindVel −0.46, WindEnergy, IMFslope) evolves with
  z_s — exactly their scale/time split. Figs `wl_vae_{latent_perturbation,param_corr,
  latent_params}.png`.
- **`examples/wl_stat_latents.py`** — do other WL stats share the 2 latents, and do
  orthogonal probes add a 3rd? (PCA latents + CCA alignment + 5-fold CV-ridge param
  recoverability.)

**Findings.** (1) The 2D feedback manifold is **universal across WL summaries**: every
statistic (peaks/minima/PDF/MF/moments) shares the *first* κ-Cl latent (canon corr
≈1.0); the *second* is shared by peaks/PDF (~0.7–0.8) but **rotated for Minkowski
functionals & moments** (0.2–0.4) — morphology sees a slightly different 2nd combo.
Each stat is individually ~2–3D in feedback (PC1/PC2 param-R²~0.6–0.7, PC3 drops; the
rest of their PCA variance is ray-trace noise). (2) **Orthogonal tSZ probes don't
expand the dimensionality** much (κ×y/yy/κ×τ align 0.9–0.98 with κ's 2D) — # params
recoverable (CV-R²>0.1) stays **~5/30** for κ-2pt vs all-WL vs WL+tSZ. BUT they
**sharpen and rotate** the accessible axes: IMFslope 0.33→0.56→0.64, and **tSZ
specifically unlocks the SN/thermal-energy axis** — WindEnergy 0.13→0.33,
RadioFeedbackReorient →0.16 (physical: tSZ ∝ thermal energy, cf [[wl-tsz-science-plan]]).
**Punchline:** in a single TNG suite feedback is intrinsically ~2D, so multi-probe
WL+tSZ buys precision + the thermal direction, not a big jump in # constrained params.
Figs `wl_stat_latent_summary.png`, `wl_recoverability_byparam.png`. (Caveats: 216
sims/ridge = finite power; WST excluded, only 40 valid runs; cross-suite/cosmology not
testable with our data.)

## 2026-06-24 — `lightcone`: auto-Cl NPE on 216 runs + all-modes/Nyquist summary

Re-ran `examples/lightcone_cl_sbi.py` after the SB35 Sobol suite grew to **216
completed runs** (from 123). Re-assembled `emulator_dataset.npz` via
`bind-emulator-assemble` first (the stale Jun-23 file still held 123; old kept as
`emulator_dataset.bak123.npz`). Also reworked the summary per a question about the
coarse binning: the script now takes **`--n_ell_bins 0`** (every native multipole,
no binning) and **`--n_pca -1`** (no PCA), and the default ℓ cut is the map
**Nyquist** ℓ≈36864 (=π/pixel for the 5°/1024 κ maps) instead of 5000 — the
suppression S=Cl/Cl_dmo cancels the CIC aliasing that contaminates the *raw* Cl,
so high-ℓ is usable, and the run-to-run S spread (the baryon signal) *grows* with
ℓ (~0.13 at ℓ~3k → ~0.6 at ℓ~45k).

**Findings (3 configs, all coverage≈0.90, well-calibrated):** (A) old 10-bin/ℓ<5k
→ mean shrink 0.994; (B) all-modes/ℓ<Nyq + PCA → **0.986** (best); (C) all-modes,
no PCA (2550 feats, 216 sims) → 1.004 (worst). Takeaways: (1) more runs + smaller
scales help only marginally — auto-Cl is intrinsically weakly constraining; (2)
**the info is ~3-dimensional**: all 510×5 modes PCA-compress to **3 components =
99.5% var** (0.863/0.126/0.006), the 0.126 one a genuine high-ℓ direction the old
ℓ<5k cut missed; (3) **PCA is effectively free and slightly helps** — skipping it
just dilutes the flow with correlated noise. Best params now VariableWindVelFactor
(0.89) + BlackHoleRadiativeEfficiency (0.92). B is the new default; headline corner
re-saved to `examples/figures_lightcone/cl_sbi_{corner,posterior}.*`.

**Same session — physical-units corner + raw-Cl + full-tomography.** (1) Corner now
plots **physical parameter values** (was the [0,1] cube): each param shown in the
space where its Sobol prior is flat — `log10(value)` for `LogFlag` params (`log `
label prefix), linear value otherwise — via `design._unit_to_native` + the bounds
from `bind.params`; ranges set to the prior box. Saved arrays gain `samples_phys`,
`truth_phys`, `param_min/max`, `log_flag` (shrink metric stays in unit space). (2)
New `--raw_cl` (raw `log10(Cl)` vs suppression) and `--tomo {auto,full}` (5 autos
vs 15 unique auto+cross) flags; feature builder refactored to a pair-based
`feat_from_cube`. **Findings:** raw Cl ≡ suppression **exactly** (max |Δshrink|=0.000)
— with cosmology fixed across the suite the shared `Cl_dmo` is just a per-mode
constant that standardization removes (suppression only matters if cosmology
varies). Full tomography does **not** help (mean shrink 0.992 vs 0.986; 7650 feats
still → **3 PCA comps**, same 0.864/0.125/0.006) — the cross-spectra are redundant
with the autos for the baryon response. Confirms the ~3-direction ceiling is
physical, not a summary/statistic choice. Exploratory runs in
`figures_lightcone/{D_rawcl,E_fulltomo}/`.

**Same session — diagnosed *why* (rank-2 forward map).** The κ suppression data
vector is rank ~2: **PC1=86.3% + PC2=12.6% = 98.9%** of all Cl variation across the
216 runs (≈ suppression amplitude + ℓ-tilt — the van Daalen/BCM 2-param family). Many
params drive the *same* 2 directions (forward |corr| 0.37–0.45 for VarWindVel/BHRadEff/
WindFreeTravelDens/IMFslope) ⇒ degenerate. Model-free proof (no flow): the 40 runs with
Cl nearest the fiducial span ~100% of the prior in θ (kNN std/prior ≈1.01); linear
inverse R² ≤0.43, mean 0.06. The 2 informed dirs are only ~75% param-driven (rest =
ray-trace/painting scatter). ⇒ ~2 combos constrained, 28 → prior. The corner "U-shapes"
are a minor logit+ensemble artifact (avg outer-10% mass 7.7% < flat 10%); informed
params are correctly center-peaked on truth (`figures_lightcone/cl_sbi_marginals.png`).
Real lever = add orthogonal probes (κ×y/tSZ, peaks/PDF). Not a bug.

## 2026-06-23 — `lightcone`: NPE on the WL auto-Cl → 30-param posterior

New **`examples/lightcone_cl_sbi.py`** (+ GPU `run_cl_sbi.sh`): `sbi` NPE that
maps the tomographic κ **auto-spectrum suppression** S(ℓ)=Cl/Cl_dmo (5 source
planes × 10 quantile ℓ-bins, ℓ∈[100,5000], shared-DMO denominator cancels cosmic
variance) → P(θ | x) over the 30 SB35 astro params, evaluated at the fiducial
`bind_science/runs/bind/run_0000`. Trained on the 123 valid Sobol runs in
`emulator_dataset.npz` (`X_unit`, already on the [0,1] cube).

**Two non-obvious fixes.** (1) The bounded `[0,1]^30` prior makes NPE rejection
sampling stall (acceptance ~0.8³⁰≈1e-3) — train the flow in **logit space** with
a wide box prior, sigmoid back. (2) With only 123 sims a single flow is unstable
(VarWindVel shrink 0.97 vs 1.12 across draws) → **pool a 5-flow ensemble**.

**Result: WL auto-Cl alone barely constrains individual baryon params.** Ensemble
posterior std/prior ≈ 0.93–1.06 (mean 0.99); most-informed = VariableWindVelFactor
(0.93), which is also the top param in an independent linear sensitivity test
(R²=0.36 predicting θ_i from x; mean R² 0.05). Held-out 90%-CI coverage 0.93
(well-calibrated). Info lives in ~2 weak combined directions, capped by
irreducible ray-trace/painting scatter (only ~58%/77% of the 2 x-PCs are
θ-explained). Consistent with the suite's active-subspace finding — tighter
constraints need κ×y / peaks / tSZ added to the data vector. Corner +
samples in `examples/figures_lightcone/cl_sbi_{corner.png,posterior.npz}`.
Also ran on the workstation's **local V100S** (the old "no GPU" note is stale).

## 2026-06-23 — `lightcone`: `bind.emulator` — instant field-level statistics emulator

New importable package **`bind.emulator`** (mirrors `bind.paint`): maps the 30
SB35 astro params (+ a source redshift `z_s`) to every ray-traced lightcone
statistic in ~10 ms, with calibrated 1σ. The TNG analogue of the BCM
"instant-`Cl`" emulators, built on the SB35 Sobol suite
(`/mnt/ceph/users/mlee1/bind_sb35/runs`, 121/256 done and growing; 50 real × 5
source planes × 1024², maps retained on disk).

**Pipeline (4 stages).** (1) `bind.inference.stats` gained `wst` (kymatio 2-D
scattering — needs the `sph_harm` shim for scipy≥1.13), `dm_stats` (DM PDF/σ_DM/F
from the τ maps), and `scaling_relations` (binned Y–M/f_gas–M/T–M); wired into
`bind-lightcone-stats` (+ a dedicated skip-if-exists backfill CLI
`bind-emulator-stats` + GPU array `run_emulator_stats.sh`, since WST is GPU-bound:
95 s for 2 maps on CPU). (2) `bind-emulator-assemble` walks the suite →
`emulator_dataset.npz` (21 statistics, per-stat **validity mask** so a
partially-backfilled stat never shrinks the others; Y–M from the pre-reduced
`integrated.parquet`). (3) the engine: per-stat `StatCompressor` (transform →
NaN-impute → standardize → variance-capped PCA, truncation residual = noise floor)
+ a pluggable backend — `GPBackend` (sklearn, CPU), **`GPTorchBackend` (gpgpu —
GPU exact GP batched over PCs via gpytorch, the recommended backend)**,
`MLPEnsemble`, and a conditional-flow `FlowBackend` (zuko NSF, generative
p(stats|θ,z_s) → mock vectors). `backend="auto"` picks gpgpu on a GPU node, gp on
CPU; `device` threads through. `Emulator.fit/save/load/predict`; `predict`
interpolates the source-plane stats to any `z_s`. (4)
`examples/emulator_validation.py` + `examples/lightcone_emulator.ipynb` (one
runnable notebook) — k-fold CV + the 60 `twobound` corners as OOD (Stage-B, plan §5).

**Two bugs caught in validation (both important):** (a) emulating **raw `Cl_κκ`**
nails it (R²=1) but **washes out the baryon suppression** — the few-% ratio is
buried under the fixed-cosmology shape; fix = emulate **S(ℓ)=C/C_DMO directly** (a
first-class target, cosmic-variance-cancelled) and reconstruct C_auto=S·C_DMO.
(b) The **MLP ensemble regresses to the mean** in 30-d with ~100 points (a single
net fits, but averaging diverse over-fit nets → constant; amplitude 50–200× too
small) → **GP is the default**. Also: pooled per-element R²/frac-err are dominated
by the shared bin-shape and gave a false R²=1.0 — validation now uses
**response-aware** metrics (across-run R² per bin + amplitude ratio).

**Status.** GP recovers the suppression response (dip-corr ~0.82, amplitude within
~10% on held-out runs at 121 runs); gpgpu matches it on GPU; ~ms/eval; save/load
bit-identical for all backends. WST/DM still need the GPU backfill (then re-run
`bind-emulator-assemble`); production training via `run_emulator_train.sh`
(`BACKENDS="mlp gpgpu flow"`). Deps `kymatio`+`zuko`+`gpytorch` = `[emulator]`
extra (numpy stayed 2.4.6 — raytracing unaffected). Code on `lightcone`; bundles
gitignored. Realizes the Stage-B emulator of `docs/wl_tsz_plan.md` §5.

**User control + presentation (follow-up).** `predict(params, z_s, stats=[...],
grids={axis: values})` — select statistics and resample onto custom ell/nu/mass
grids (per-statistic axis values, so the peak_counts-nu/peak_R-nu name collision is
handled); `em.describe()` + `bind-emulate predict --list/--stats/--ell/--nu/--mass`.
`examples/emulator_showcase.py` makes the two paper figures (all-statistics
emulator-vs-truth at ~100 ms/predict; per-parameter feedback response inferred from
the OAT offset). The **1P** comparison is the ideal response test but its lightcones
aren't generated yet (`DESIGN=1P sbatch run_sobol_{paste,lux,stats}.sh`); the tool
demos on the 60 twobound corners (dip(S) corr 0.84 — good ranking, conservative at
the extreme corners, as expected for GP extrapolation). Confirmed the workstation
*does* have a GPU, so `backend="auto"` trains `gpgpu` on-device. Trimmed the Y–M
mass bins to the populated 10^13–10^14.7 band (empty edge bins were imputation
artifacts).

**Full-set rebuild + backend verdict + tutorial (2026-06-24).** Rebuilt on the
**237-run** suite (WST now ≥ threshold, 23 statistics, all on gpgpu/GPU). Measured
backends on a held-out interior split (190/190+47): **GP wins** — S(ℓ) median frac
err **2.0%**, dip-corr 0.87, 4 s; the **normalizing flow is worse** (2.4%, 60 s —
generative density model, data-hungry, its use is mock-sampling not point accuracy);
MLP collapses (4.9%, amp→0). The showcase/ratio plots look "off" only because they
compare against the **extreme twobound corners** (hardest extrapolation, GP
conservative there); on interior held-out the emulator tracks both suppression and
enhancement (`emulator_heldout_interior.png`). Error budget: the 50-realization
scatter (cosmic var ⊕ BIND generative noise) ≈ 6% **dominates** the GP epistemic
1σ ≈ 1.4% on S(ℓ) — emulator is below the data noise floor. New
`examples/bind_tutorial.ipynb` (via `_build_bind_tutorial_nb.py`): end-to-end
walkthrough — live BIND paint (weights present, 3.9 s/patch → 7 channels) →
κ/y/τ maps → live statistics → emulator API → error propagation (BIND ⊗ GP ⊗
cosmic variance) → the fixed-cosmology science case (feedback inference + baryon
marginalization for cosmology).

## 2026-06-22 — `lightcone`: ⚠ CORRECTION to Step 4 self-calibration claim

The Step-4 "rank-1 ⇒ tomography barely self-calibrates baryons" conclusion was
**WRONG** (flagged by the human). The flaw: it collapsed each source plane to a
single amplitude (discarding the in-bin ℓ-shape that drives self-cal) and used 5
nested δ-sources of one lightcone (near-rank-1 by construction); and rank-1 of the
run-to-run variation answers "do feedback models bias in the same z-direction?",
not "can tomography separate baryons from cosmology?".

Proper test in `examples/lightcone_selfcal.py` → `figures_lightcone/selfcal.png`:
joint Fisher (S8, Ωm, A_bary) on the full-ℓ tomographic auto-spectra, A_bary
scaling the measured suppression template. **Tomography DOES self-calibrate**: the
baryon bias z-signature *declines* 83% from z_s=0.5→2.44 (a true S8 shift is FLAT)
→ separable. 1 bin: σ(S8) degrades ×2.37 marginalising A_bary, r(S8,A)=−0.91
(degenerate). **≥2 well-separated bins: degradation ×1.01, r→+0.12 (degeneracy
broken, marginalisation ~free).** Honest nuance (multi-mode): the feedback *shape*
variation beyond the mean is itself ~rank-1 (leading mode = 98% of residual var);
5-bin tomography absorbs the leading amplitude for free but marginalising the 2nd
(shape) mode costs **×3.4** in σ(S8). So tomography nails the dominant baryon mode,
but the residual feedback shape (the part the 1-param SP(k)/f_gas relation misses,
Step 5) stays partly degenerate. Caveat: δ-sources give more z-leverage than a
realistic overlapping n(z). Supersedes the Step-4 entry below.

## 2026-06-22 — `lightcone`: do analytic baryon models span BIND? (Step 5)

`examples/lightcone_model_comparison.py` → `figures_lightcone/model_comparison.png`.
Compared the BIND S(ℓ) ensemble against 3 analytic baryon families, all projected
through the *same* WL kernel via CCL (P_hydro=P_dmo·S → C_ℓ ratio): **BCM**
(CCL `BaryonsSchneider15`, 3-param, spanned log10Mc/eta_b), **van Daalen 2019**
(CCL `BaryonsvanDaalen19`, 1-param f_bar = the SP(k)/f_gas-proxy family), **BCemu
2025** (spanned its dominant log10Mc). Per-model coverage = fraction of BIND runs
whose S(ℓ) falls in the model's full-parameter span.

**Findings:** (1) the **1-param van Daalen/SP(k) relation spans only ~22–54%** of
the BIND ensemble → a single baryon-fraction number is insufficient; higher-order
feedback structure breaks the f_bar→S(k) relation (echoes the moderate f_gas↔S
Spearman in Step 1). (2) Multi-param BCM/BCemu cover more (60–96%) but
**systematically miss the enhancement corner**: BIND produces intermediate-scale
power *enhancement* up to S≈1.10–1.21 (cooling/condensation, gas-rich runs) where
the analytic models top out at S≈1.01–1.05 — that gas-rich/cooling region is the
"gap" next-gen correction models must capture. Caveats: BCemu spanned along
log10Mc only (lower bound on its coverage); **HMcode-AGN not included** (needs
CAMB/Fortran toolchain — future); the enhancement is TNG-specific baryon physics.

Env: installed `pyccl` 3.3.4 + `BCemu` 2.0.5 + `smt` 2.14.1 (+deps) into BIND_env;
numpy bumped 2.2.4→2.4.6. **Verified pyccl, Pylians, and bind.inference.stats all
work under 2.4.6** so the live raytracing SLURM array is unaffected.

## 2026-06-22 — `lightcone`: redshift tomography of the bias (Step 4) + refresh to 99 runs

**Step 4 — ΔS8(z_s).** `examples/lightcone_bias_tomography.py` →
`analysis_cache/bias_tomography.npz` + `figures_lightcone/bias_tomography.png`.
Per source plane, the 1-param S8 bias (Ωm fixed — one spectrum can't break the
degeneracy) → ΔS8_i(z_s) per run. **The z-evolution is rank-1: PC1 = 99.7% of the
variance** ⇒ baryons act as essentially *one* nuisance amplitude across the 5
tomographic bins (universal z-shape × per-run amplitude). Bias dilutes with source
z (median +0.0019→+0.0011 z_s=0.5→2.44). Survey-design takeaway: WL tomography
alone barely self-calibrates baryons — external priors on the Step-3 knobs are what
matter. The faint PC2 (0.3%) *does* carry a feedback-timing imprint: the per-run
z-slope correlates with IMFslope (ρ=+0.58), WindEnergy (−0.32), BHRadEff (−0.31)
— stellar vs AGN/SN leave slightly different z-evolution. Caveat: 5 delta-source
planes with heavily overlapping lensing kernels limit z-leverage; a broad n(z)
might do better.

**Refresh.** Re-ran Steps 1–4 on the 99 runs done so far (array still running, 14
tasks live). Results **stable** vs 96: Step 2 unchanged; Step 3 top-2 now IMFslope
(S_T=0.36) ≈ BHRadEff (0.32) (they swap rank within noise, ~2/3 variance together),
CV R²=0.38; Step 4 rank-1 99.7%. Robustness check passed; numbers lock at 256.

## 2026-06-22 — `lightcone`: which knobs drive the WL bias + non-Gaussian feasibility (Step 3)

**Non-Gaussian bias (peaks/minima/MFs) — investigated, deferred.** The Fisher bias
needs a cosmology derivative ∂μ/∂(Ωm,σ8); the Sobol suite fixes cosmology, and
there's no analytic model for these stats. Tested the only proxy (rescale DMO maps
κ→(1+ε)κ ≈ S8 shift, recompute stats): on C_ℓ it's **rank-exact (r=1.0000) but
scale-miscalibrated ×1.38** because real ∂lnC/∂lnS8≈2.6 (nonlinear growth) not the
rescaling's 2.0. That miscalibration factor *is* the morphology effect and differs
per statistic (unknown for peaks/MFs) → a cross-statistic bias comparison is not
trustworthy. Rigorous unlock = ray-trace a small DMO lightcone grid at varied
Ωm/σ8 (then peaks/MFs drop into the same machinery). Deferred.

**Step 3 — Sobol sensitivity of ΔS8(θ_astro).** `examples/lightcone_bias_sensitivity.py`
→ `analysis_cache/bias_sensitivity.npz` + `figures_lightcone/bias_sensitivity.png`.
Target = ΔS8(ℓ_max=3000) from Step 2; 30 astro params normalised to [0,1] (log
flags). Three estimators agree on the ranking: binned main-effect S_i, GB-surrogate
Saltelli/Jansen S_i+S_Ti (CV R²=0.34 at 96 runs — modest, firms up at 256),
permutation importance. **Top "lensing-dangerous" knobs: BlackHoleRadiativeEfficiency
(S_T=0.38), IMFslope (0.29)** — ~2/3 of the variance — then RadioFeedbackReorientation
(0.14), WindEnergyIn1e51erg (0.10). Strong interaction component (S_T≈2×S_i) ⇒ the
bias lives in parameter *combinations* (the f_gas-controlling AGN+IMF sector),
consistent with the ~2 effective dims of [[sobol-feedback-latent]]; same AGN+IMF
sector that sets gas content. Next: Step 4 (z-tomography of ΔS8) / Step 5 (vs
BCemu/HMcode-AGN).

## 2026-06-22 — `lightcone`: effective cosmological bias (ΔS8, ΔΩm) per run (paper Step 2)

Pushed Step 1's S(ℓ) ensemble through a linearised Fisher bias (Schneider+2020):
`Δθ=F⁻¹B`, baryonic residual `ΔC^ii_ℓ=C^ii,fid_ℓ(S_i(ℓ)−1)`, full 5×5 Gaussian
tomographic covariance (auto data vector + cross terms + shape noise). Cosmology
derivatives ∂C/∂(Ωm,σ8) from **pyccl** (installed: `pip install --only-binary`
pyccl 3.3.4; EH transfer + halofit — **validated 2–5% vs the sim DMO C_ℓ** across
ℓ=200–4000, all 5 source planes → confirms ℓ↔multipole mapping + lensing norm).
Engine `examples/lightcone_cosmo_bias.py` → `analysis_cache/cosmo_bias.npz` + fig
`figures_lightcone/cosmo_bias.png`. LSST-Y10 (f_sky 0.4, n_gal 27, σ_e 0.26) for
significance only — bias Δθ is itself f_sky-independent (F⁻¹B).

**Findings (96 runs):** (1) the bias is a **tight 1-D locus** in (ΔΩm,ΔS8),
slope dS8/dΩm≈0.4–0.67, rotating with ℓ_max — the single effective baryon
degeneracy direction. (2) **group f_gas orders the bias far better than it ordered
S(ℓ)**: Spearman ρ(f_gas,ΔS8)≈+0.51–0.58 (vs +0.27 for S itself) — projecting onto
cosmology amplifies the gas signal; gas-poor/strong-feedback → ΔS8<0 (infer S8 low,
correct sign). (3) fiducial/median bias is **small** (sub-σ at ℓ_max=3000, TNG is
gentle), but the dangerous tail grows **steeply with ℓ_max**: frac |ΔS8|>2σ =
1%/9%/22% at ℓ_max=2000/3000/5000, max 6σ, ΔS8 95% CI ±0.01–0.014. Caveat: the
σ-significance is statistical-only/2-param (no h,ns,IA,photo-z marginalisation) →
optimistic; the ΔS8 *magnitudes* are the robust headline. Step 3 = Sobol
sensitivity of ΔS8 over the 30 astro params.

## 2026-06-22 — `lightcone`: baryonic WL transfer-function ensemble (paper Step 1)

First science cut on the 256-pt Sobol raytrace: the *surgical* suppression ratio
`S_i(ℓ,z_s) = C_ℓ^κκ(θ_astro^i)/C_ℓ^κκ(DMO)` per tomographic source plane. DMO
reference is the paired lux trace at `/mnt/home/mlee1/ceph/bind_science/runs/dmo/run_0000`
(same RT_SEED=1992 + transforms as every BIND run — verified realization-by-realization,
map correlation r≈0.983–0.986; cosmic variance cancels so the ratio isolates baryons).
Engine `examples/lightcone_transfer.py` builds S from the stored per-run `Cl_kappa.npz`
(no recompute), caches `bind_sb35/analysis_cache/transfer_Sell.npz`, colours the ensemble
by group f_gas (atlas cube snap096). 96/256 runs done at time of writing (array still running).

**Findings (96 runs):** (1) ensemble brackets both suppression *and* enhancement — at z_s=0.5,
S spans ~0.80→1.24 in 2e3<ℓ<1e4; **59%** of runs *enhance* WL power at ℓ~500–2000 (cooling
bump). (2) Clean redshift dilution: median band-suppression 0.938→0.982 from z_s=0.5→2.44;
run-to-run σ[S(ℓ)] grows with ℓ and is largest at low z_s (feedback info ∝ high-ℓ, low-z).
(3) group f_gas orders the curves but only moderately (Spearman +0.27→+0.34) — SP(k)-style
single-f_gas → S(k) is *partial*, the WL TF carries feedback structure beyond one gas number
(preview of Step 5). Caveats: ratio-of-means (pairing makes it ≈ paired); ℓ capped 1.5e4
(CIC aliasing above, cancels in ratio); numbers firm up at 256. Figs in
`examples/figures_lightcone/transfer_function_{ensemble,tomography}.png`.

## 2026-06-22 — `lightcone`: SHMR scatter (feedback vs accretion) on the SB35 Sobol suite

Replicated `examples/shmr_accretion_scatter.ipynb` (the 60-run twobound SHMR
feedback-vs-accretion-history analysis) on the 256-pt Sobol suite →
`examples/shmr_accretion_scatter_sb35.ipynb` (built by `examples/_build_shmr_sb35_nb.py`,
repo nb-build convention). Sobol/twobound/fiducial paint the **same** snap-96 DMO halos,
so the analysis is a drop-in: same 2933 halos, reuses `tng300dm_mass_history.hdf5` verbatim
(matches 2933/2933), reuses `bind_science/.../truth_snap096.npz` as the TNG300-hydro
reference (identical `M_fof` order). Source is the index-aligned atlas cube
(`bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz` + `atlas_design.npz`), same
files as `sobol_atlas.ipynb`. §8's per-knob lo/hi panels → per-param low/high **tercile
main-effects** (no single-knob runs in a Sobol design; terciles marginalise the other 29).

**Finding that corrects the twobound notebook's caveat:** the joint Sobol feedback scatter
(σ≈0.386 dex, 97% of total SHMR variance at fixed M) is **larger** than the twobound
one-at-a-time spread (σ≈0.161 dex) — varying all 30 knobs jointly *compounds*, so single-knob
extremes are **not** an upper bound on the joint feedback variance. Accretion history (t_form
on the DMO tree) explains ~10% of the halo-to-halo (non-feedback) residual; early-forming
halos sit above the mean SHMR (ρ(t_form,resid)<0). Executed headless via `bind_env` kernel,
0 errors, 5 figs.

## 2026-06-22 — `lightcone`: SB35 Sobol latent space + SBI (ML-forward, 2 notebooks)

With the cache complete (256 runs, 8.6M halos), built the "latent space of feedback"
analysis on the same-halo Sobol sweep. Engine `examples/sobol_ml.py` (torch+sklearn, no
new deps): per-run observable fingerprint `(X 256×D, THETA 256×30 in [0,1])` from
`integrated.parquet` (reads only needed columns — full table is ~4 GB);
**Information-Ordered Bottleneck** (nested-mask latent → info curve → effective dim);
**active subspace** (autograd Jacobian E[JᵀJ] eigendecomp → stiff/sloppy param combos);
neural posterior p(θ|x). Two notebook builders →
`examples/sobol_latent.ipynb` (Part A: dimensionality + degeneracy distillery + Isomap
topology) and `examples/sobol_sbi.ipynb` (Part B). Both validated headless (peak ~1.8 GB).

Part B reworked after a trust check: the first diagonal-MDN-in-[0,1] posterior produced
**fake spikes at the parameter bounds** (Gaussian samples clipped to [0,1]). Replaced with
`GaussNPE` — full-cov conditional Gaussian in **logit space** (no boundary artifact),
evaluated at the **real TNG300 fiducial** x_fid (reduced from `bind_science/runs/truth/
run_0000` with the identical binning+scaler via `S.fiducial_fingerprint`; θ_fid known).
Validated by leave-out rank coverage (calibrated). Honest finding: at 256 design points
the inverse posterior is **data-starved/broad**, so width-based "constrained" rankings are
noisy (defer to forward active subspace); degeneracies trusted only where NPE agrees with a
forward-model Fisher (`S.fisher_degeneracy`) — robust pair = AGN
BlackHoleRadiativeEfficiency↔QuasarThresholdPower (+0.27). Fixes: more Sobol points (GPU
`run_sb35_generate.sh`, needs `fm_redshift_thermo` weights, absent from bundle) / per-halo
SBI / lightcone κ,y observables.

**Result:** the 30-d feedback space collapses to **~2 effective directions** (95% of the
observable response). Stiffest = wind/SN energy sector (VariableWindVelFactor, WindEnergy,
WindFreeTravelDensFac) — the [[fgas-bridge-anatomy]] energy-budget axis; 2nd =
IMFslope/BlackHoleRadiativeEfficiency. Sloppiest = SofteningComovingType01 (numerical),
SNIa rate. **Active subspace and the SBI posterior independently agree** on
constrained-vs-degenerate params. Next: add lightcone κ/y map stats as observables; swap
MDN→sbi flow + SBC; survey-restricted forecasts (`wl_tsz_plan.md`).

## 2026-06-22 — `lightcone`: SB35 Sobol halo-aperture cache, ported to `preempt`

Made the per-halo aperture reduction runnable on this system's `preempt` queue. The
generation is **complete** (256 runs × 20 snaps = 5120 keys, 4 slabs each on disk); only
the aperture cache was partial (2971/5120 done). Two portability blockers fixed: (1) the
worker/notebook read snapshot metadata from `BUNDLE/conditions` which doesn't exist here —
it lives with the shared fiducial conditions at
`/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_*/stage1/stage1_manifest.json` (uniform
`pixel_size=50/1024`, `n_slabs=4`); (2) the old design gathered all results to rank 0 and
wrote the 2.2 GB pickle once at the end → preemption lost everything.

Rewrote `bind_mpi_worker.py` as preempt-safe + restartable: per-key parquet shards under
`analysis_cache/shards/`, skip-if-exists. Modes `migrate` (seed shards from the existing
pickle, no patch reads — bit-for-bit verified vs cache), `build`, `merge`
(→ `integrated.parquet`/`.pkl`), `status`. Dropped the mpi4py dependency: the split uses
srun's `SLURM_PROCID`/`SLURM_NTASKS` over the *full deterministic key grid* (race-free,
load-balanced 32–34 keys/rank at 64-way), sidestepping the mpi4py↔launcher mismatch
([[mpi-venv-h5py-clash]]). `submit_bind_mpi.sh` → `preempt`, `--requeue`, no MPI modules.
`examples/figures_lightcone/sb35_sobol_analysis.ipynb` reads the prebuilt cache + the
fiducial conditions. Workflow: `migrate` (done) → `sbatch submit_bind_mpi.sh` → `merge`.
~2131 keys remain (~6 min/rank at 64-way). Science targets next: effective dimensionality
of the 30-d astro space (same-halo Sobol sweep) + SBI param constraints from halo
observables — see [[wl-tsz-science-plan]], [[fgas-bridge-anatomy]].

## 2026-06-21 — `lightcone`: SB35 Sobol lightcone pipeline (256 runs)

The 256-point Sobol astro-parameter design (generated off-cluster, halos transferred to
`/mnt/home/mlee1/ceph/bind_sb35/runs/run_NNNN/snap_*/composite_slab*.npz`) is the exact
twobound setup at scale: halos painted from the *same* fiducial DMO conditions
(`run_lightcone_project.sh` stage1), verified halo-for-halo (`halo_centers` identical,
n_halos=666, box=205). New `run_sobol_lightcone.sh` is a verbatim clone of
`run_twobound_lightcone.sh` repointed at the sb35 bundle (`RUNS_ROOT`, `--array=0-255%8`):
per run, recomposite patches onto the shared stage1 → paint lenspot/y/tau planes (shared
`lightcone_transforms.json`) → lux trace at the fiducial `RT_SEED=1992` → collect
{kappa,y,tau}_maps.npz → Cl stats; `_work` cleaned. Same seed/transforms ⇒ per-run
kappa/y/tau differences are purely parametric (cosmic variance cancels). Idempotent on
`tau_maps.npz`.

## 2026-06-21 — `lightcone`: SB35 lightcone map-statistics atlas (MPI, preempt)

Map-level analog of the per-halo atlas, for the ray-traced κ/y/τ maps. The raytrace
already writes uniform per-run stats, but with map-ν peaks (absorbs the σ_κ response);
new `examples/lightcone_atlas_mpi.py` recomputes a response-correct vector from each
run's `{kappa,y,tau}_maps.npz` (50,5,1024,1024) via `bind.inference.stats`: power
spectra (κκ tomographic 5×5, κy, yy, κτ, ττ, yτ), non-Gaussian (PDF, moments,
Minkowski V0/V1/V2), peak/minima counts and tSZ-at-peaks R(ν) — peaks use
`nu_norm='fixed'` referenced to the FIDUCIAL map rms. `--recompute` (mpi4py, 256
sobol+60 tb+fid, restart-safe + incremental as raytrace lands runs) + `--consolidate`
→ one `lightcone_cubes/lightcone_atlas.npz` (sobol_<K>(256,…)/tb_<K>(60,…)/fid_<K> +
valid masks; design reused from atlas_cubes). Batch `run_lightcone_atlas.sh`
(preempt, openmpi-only, 128 ranks; ~11 min/run measured so ~30 min full). Viewer
`examples/lightcone_atlas.ipynb` (`_build_lightcone_atlas_nb.py`): per-probe twobound-
envelope-vs-Sobol-cloud, ratio-to-fid response, and a param→statistic Spearman heatmap.
Caveats: no `kappa_dmo` ⇒ no S(ℓ); some twobound runs predate τ (NaN-handled).

## 2026-06-21 — `lightcone`: SB35 Sobol per-halo atlas (MPI, preempt)

Per-halo analysis of the 256 Sobol points while the lightcone job runs. Sobol,
twobound (60 extremes), and fiducial all paint the *same* fiducial DMO halos at the
same 20 snapshots (verified halo-for-halo), so the per-halo arrays are index-aligned
across all parameter points. New `examples/sobol_atlas_mpi.py` reuses
`halo_atlas.reduce_run_snap` (so Sobol caches are bit-comparable to the existing
`bind_science/halo_atlas` twobound caches): `--reduce` (mpi4py, 256×20=5120 tasks,
restart-safe per-(run,snap) npz) + `--consolidate` (serial → notebook-ready per-snap
cubes `sobol_<K>(256,n)/tb_<K>(60,n)/fid_<K>(n)` halo-aligned + `atlas_design.npz`).
Batch `run_sobol_atlas.sh` (preempt/qos=preempt, openmpi-only per [[mpi-venv-h5py-clash]],
64 ranks ≈ 8 min). Viewer `examples/sobol_atlas.ipynb` (built by
`_build_sobol_atlas_nb.py`): f_gas(M)/Y–M twobound-envelope-vs-Sobol-cloud, per-halo
response distribution, global Spearman parameter sensitivity (AGN+wind drive f_gas),
z-evolution. Validated end-to-end on a 10-run/2-snap subset; cubes live in
`bind_sb35/analysis_cache/atlas_cubes/`.

## 2026-06-18 — `lightcone`: anatomy of the f_gas→S(ℓ) bridge

Dug into the van-Daalen-style bridge from `paper_decomp_figs.ipynb` §7.3 (group f_gas
predicts WL suppression S(ℓ)=C_bind/C_dmo, r≈0.91). New notebook
`examples/fgas_bridge_explore.ipynb` (built by `examples/_build_fgas_bridge_explore_nb.py`),
six figures, all from the same caches (`dashboard_cache` + `halo_atlas` + `runs/{bind,dmo}`):

1. **Parameter directionality** (`fig_bridge_param_directions`) — quiver of each 1P run's
   (Δf_gas, ΔS) from fiducial + per-parameter tornado/lever. Runs slide *along* the gas axis;
   top levers are the energy budgets (WindEnergyIn1e51erg, BlackHoleRadiativeEfficiency, IMFslope).
2. **Mass dependence** (`fig_bridge_mass`) — bridge refit in group/intermediate/cluster M200c bins.
   Slope steepens 0.59→0.86→1.12 toward clusters but r loosens 0.91→0.84→0.59; group scale is the
   sweet spot (atlas floor is 1e13, so group→cluster only). **§2b** (`fig_bridge_heatmap`):
   r[f_gas,S(ℓ)] over the full (M200c × ℓ) grid — single bright ridge peaking r=0.92 at
   M~1.9e13, ℓ~1037; dark stripe at ℓ~400 is the S→1 suppression node (no variance to correlate).
   **§2c** (`fig_bridge_panels`): the bridge unrolled into 16 S(ℓ_i)-vs-f_gas scatter panels
   (3 mass-bin fits each) — slopes steepen + tip off S=1 with ℓ. NB every point = one of the 57
   1P runs (the whole feedback library), not fiducial.
3. **Compton-Y** (`fig_bridge_compton_y`) — **Y_500c is a tighter S predictor than f_gas
   (r=0.935 vs 0.914)**; partial corr shows Y subsumes f_gas (r[f_gas,S|Y]≈0) while keeping
   residual power (r[Y,S|f_gas]≈0.49 = the heating leg). WL-relevant variable is thermal gas
   content M_gas·T, directly tSZ-observable → eROSITA f_gas tension and WL suppression are one Y.
4. **When the bridge breaks** (`fig_bridge_residual`) — low r has 3 causes; only the high-ℓ
   one (at fixed group mass) is physics: f_gas is the monopole (large scales), small scales need
   the profile. partial r[S,X|f_gas] vs ℓ shows a crossover — thermal (Y,K) below ℓ~1.5k →
   structural (c_dm, c_gas, **f_star→0.93**) above. f_gas+c_dm 2-var bridge restores r 0.75→0.92
   at ℓ~8k (so it's a deterministic 2nd DOF, not noise). Off-bridge residual is feedback-specific:
   AGN above the bridge, SN/wind below at high ℓ → residual = feedback-MODE diagnostic. Quad≈lin
   (not curvature, scatter). High-mass low-r is just sampling noise.
5. **van Daalen+2020 for WL** (`fig_bridge_vandaalen`) — reframed from r to the *functional
   relation* (their Fig.16). x=renormalized BARYON frac f̃_bar=(gas+star)/M500c/(Ω_b/Ω_m), M500c
   6e13-2e14, vd fit −exp(−5.990 f̃−0.5107). **TNG-CAMELS spans only f̃~0.81-0.98 = the flat
   saturated top** (can't reach obs band 0.55-0.76 or the steep part) → THAT'S why every fit looked
   linear (we sample the linear tail of a global exponential). WL ΔC_ℓ/C_ℓ vs f̃ is tight (r~0.95),
   steepens with ℓ (higher ℓ=higher k_eff=deeper on curve), crosses into ENHANCEMENT (>0) at small
   scales (adiabatic contraction/stars, beyond vd's suppression-only range).

## 2026-06-18 — `lightcone`: twobound full-RT campaign + probes notebook

**Probes notebook.** `examples/lightcone_probes.ipynb` (built by `examples/_build_lightcone_probes_nb.py`):
κ/y/τ at a glance, tomographic autos, cross-spectra + r_ℓ (all 3 pairs), κ-peak-stacked y/τ
(R(ν) + y/τ temperature proxy), and the twobound parameter response (κ,y). Loaders fall back to
raw `rt_output/` so it runs mid-pipeline. **Bug found:** `stats.power_spectrum` normalizes autos
(`Pk_plane`) vs cross (`XPk_plane`) differently → naive r_ℓ≈0; the notebook gets cross+autos+r from a
single `XPk_plane` call (κ×τ r≈0.86, y×τ r≈0.70, κ×y r≈0.66). Logic validated; needs a final exec pass.

**twobound full ray-trace.** `run_twobound_lightcone.sh` (array 0-59) ray-traces every twobound run
exactly like the fiducial. The twobound runs are **patch-only** (just `generated_patches`/`thermo_patches`
— no pasted composite, no stage1, no lensplanes), so each task: recomposite (reusing the FIDUCIAL
`snap_*/stage1`; DMO lightcone + halos are identical across runs, verified halo-for-halo) →
lenspot/yplane/tauplane (reusing fiducial transforms) → lux (same `RT_SEED=1992`/transforms so the
response is purely parametric) → collect → stats → cleanup. Painting chain validated end-to-end
(lenspot corr 0.997 vs fiducial); lux+sbatch are the user's to submit. Decisions: all 60, N_REAL=50.

## 2026-06-18 — `lightcone`: lux-RT kSZ/FRB tau plane + per-run Cl_yy

Enacted the two §7 gaps as real pipeline, not scaffold.

**lux tau plane (full RT).** Cloned the y machinery in `/mnt/home/mlee1/lux/` (separate git repo):
`compute_tau`/`tau_input_dir` (params.cpp), `tauplane`/`taumap0`/`taumap_out` + `load_tau`/`calc_tau`
in raytracing.{hpp,cpp}, writes `tau{plane}.dat`; **rebuilt** (`module restore lux && make`, links
openmpi-4.0.7/hdf5-1.12.3/fftw-3.3.10/gsl-2.7.1). tau is additive along the (deflected) ray, no kernel —
the RT counterpart of the Born tau already in `assemble_lightcone`. BIND side: `lensplane.write_tauplane`,
`bind-paint-tauplane` (electron column σ_T·x_e·Σ_gas/m_p from the gas channel via `_tau_per_gas_pixel`),
`lux_io.load_tau_realizations`, `stats.cl_kappa_tau` (κ×τ/τ×τ/+y×τ), `lux_collect`→`tau_maps.npz`,
`lightcone_stats`→`Cl_tau.npz`, `dashboard_precompute` per-run `cltt`/`ktau`/`ytau`. Scripts:
`run_sobol_lux.sh` (ini `compute_tau`), `run_sobol_paste.sh` (paints tauplanes), new
`run_lightcone_tauplane.sh` + **`run_lightcone_tau.sh`** (one-allocation paint→lux→collect→stats).
Validated: paint on real snap_096 composites → (4096², matches yplanes, τ_max≈0.018, slab mean≈1.3e-5).

**Cl_yy.** Per-run `{run}_clyy` already existed (my earlier "absent" note was wrong — wrong key); it is a
clean ±5–7% pure-tSZ leg in ℓ∈[1000,2000]. Wired it into the decomp-notebook §7.1 orthogonality matrix
(+ auto-activating κτ/ττ legs once the tau run lands): WL block redundant (|cos|=0.92, 23°), WL↔gas
secondaries **62°** (was 56° with ky/R only); C_ℓ^yy is the most WL-decorrelated leg (|cos|≈0.2–0.4).

## 2026-06-18 — `lightcone`: split the figure notebook into Paper I / Paper II; added the "matches → measures" gaps

`examples/paper_lightcone_figs.ipynb` had drifted into a 15-figure kitchen sink — half validation
("BIND matches hydro"), half catalog ("everything that responds") — with no thesis spine, so it read
as "BIND can match a lot of things" rather than a result. Partitioned it (pure cell-split, nothing
recomputes) along the [wl_tsz_plan.md](wl_tsz_plan.md) Paper I/II line:
- **[examples/paper_methods_figs.ipynb](examples/paper_methods_figs.ipynb)** (Paper I, methods/validation):
  skymaps, halo + field closure vs TNG300, gas fractions, baryon budget. Job is tool-faithfulness — the
  "no new physics" feeling there is correct and fine.
- **[examples/paper_decomp_figs.ipynb](examples/paper_decomp_figs.ipynb)** (Paper II, flagship): the
  §4–§6 response-library / bridge / WL×tSZ / forecast spine, pressure+morphology demoted to Appendix A,
  **plus a new §7 closing the three gaps between "matches" and "measures"**:
  - **§7.1 `fig_orthogonality`** — the G1 response-vector angle, measured not asserted: WL block
    (S(ℓ),N_pk,N_min,V1,V2) mutually redundant (|cos|=0.92, 23°) but **56° from the tSZ legs**
    (C_ℓ^κy, R) → tSZ carries independent feedback info.
  - **§7.2 `fig_decomp`** — peak response separates into mass (Δκ), gas/τ-leg (Δln M_gas), heat (Δln T);
    heating anti-correlates with mass r=−0.73, and pressure **closes** ΔlnY≈ΔlnM_gas+ΔlnT at r=0.96
    (y = gas × temperature). Uses the per-peak thermo channels in `g1_peakhalo.npz` (Mg/Y/T); the
    field-level lux τ-map is still the pending step (plan §8.7).
  - **§7.3 `fig_bridge_predict`** — promotes the f_gas↔S(ℓ) correlation to a falsifiable prediction:
    bridge S=0.57·f_gas+0.95 (r=0.91); projects X-ray-standard vs eROSITA-low f_gas → predicted WL
    suppression (eROSITA-low ⇒ 3.4% at ℓ~1500).
  All three new cells validated against the cached `bind_science/` reductions. Original
  `paper_lightcone_figs.ipynb` left in place; the two new notebooks supersede it for figure generation.

## 2026-06-17 — `lightcone`: three-way kappa comparison confirms triaxiality-tracing inflated f_aniso

Built [examples/bcm_warp_comparison.py](examples/bcm_warp_comparison.py) — paired (matched-seed)
decomposition of the four WL fields (bind/sph=radial-BCM/mono=pure-spherical/dmo) in real space
(residual rms + Pearson r) and harmonic space (suppression S(ℓ), residual-field power C_ℓ^ΔΔ, and
fractions f_aniso=(C_b−C_s)/(C_b−C_d), f_tri=(C_s−C_m)/(C_b−C_d), f_mono=f_aniso+f_tri). Ran on fid.
**Result (fid, z_s=1):** the old "40–65% anisotropic" number = f_mono (pure-spherical error) ≈ 0.5
flat in ℓ. The corrected **f_aniso (what a faithful radial BCM misses) = 0.45 (ℓ=3e3) → 0.24 (1e4) →
~0 (2e4)**; triaxiality-tracing f_tri rises 0.08→0.25→0.61 over the same range. So at small scales the
entire apparent anisotropy is triaxiality a real BCM reproduces — the headline dropped as predicted.
Real space: r(bind,sph)=r(bind,mono)=0.987–0.988; the warp barely beats mono globally (16% vs 16% of
κ rms) — its advantage is concentrated at high ℓ. Fractions below ℓ≈2000 are 0/0 noise; N_real=8 of one
box. Fig: examples/figures_lightcone/bcm_warp_comparison_fid.png. TODO: rerun `--run tbNNNN` once the
twobound maps finish for the param dependence of f_aniso.

## 2026-06-17 — `lightcone`: SHMR scatter — feedback vs accretion history

New `examples/shmr_accretion_scatter.ipynb` (+ generator `examples/_build_shmr_nb.py`), built from a
colleague's `examples/merger_tree_11_tng300 (1).ipynb` + `examples/halo_atlas.py`. Studies the stellar–halo
mass scatter for M200c>1e13 over the 60-run `twobound` feedback ensemble (index-aligned per-halo atlas
caches, snap 096, 2933 halos). Balanced variance split at fixed halo mass: `σ²_tot = E_i[Var_p] + Var_i[E_p]`
→ **feedback ≈ 84.5%** of variance (the 60 extreme single-knob ± runs, ±0.4 dex), **halo-to-halo ≈ 15.5%**
(σ_tot=0.175 dex, exact closure). The accretion-history fraction (regress the halo-mean residual on
SubLink formation times t_form_10/25/50/90) is gated on a one-time merger-tree build, adapted to run on the
**DMO** tree (`L205n2500TNG_DM`, `base_snap=96`) so `GroupNumber`+mass join BIND's DMO halos exactly —
needs `illustris_python` on a node. Caveat: twobound bounds are extreme, so the feedback term is an upper
bound; reweight/Sobol for a prior.

## 2026-06-17 — `lightcone`: faithful radial-BCM counterfactual for the WL-anisotropy study

Fixed a methodological flaw in the spherical control of `examples/wl_anisotropy_paper.ipynb`. The old
`sph` azimuthally averaged the *projected correction* (`azimuthal_average_patch(ΔΣ)`), i.e.
"monopole-only correction in projection" — which forces the **correction** circular and so credits the
BCM with *none* of the anisotropy a real Schneider–Teyssier BCM reproduces (a purely radial displacement
of a triaxial halo still yields an anisotropic ΔΣ). That makes `dX_aniso = X_bind − X_sph` an **upper
bound**: it lumps "triaxiality-tracing" in with genuine feedback anisotropy, inflating `f_aniso`.
- New primary control `bcm_warp_patch` (`mode="bcm_warp"`) in [spherical.py](src/bind/inference/spherical.py):
  builds the radial displacement `r→r'` matching `M_dmo(<r)=M_hydro(<r')` and **pushes each DMO pixel
  radially** (CIC, mass-exact), so triaxiality/substructure **co-move with the mass flow**. `dX_aniso`
  now isolates feedback-specific angular structure. Verified: mass conserved to 1e-5, monopole tracks
  hydro, DMO annular RMS preserved.
- Kept the circular monopole as a secondary curve (`mode="monopole"` → `composite_mono`); the gap
  `X_sph − X_mono` *measures* the triaxiality-tracing term. `write_spherical_slab` now emits three
  totals: `composite` (BCM-warp → `kappa_sph`), `composite_mono` (→ `kappa_mono`), `composite_bind`.
- **TODO**: re-run `bind-spherical-slabs` to regenerate the tree (the `composite` key changed meaning),
  then `bind-lightcone-maps --mass_key {composite,composite_mono,composite_bind}`; update the notebook's
  decomposition to add the `mono` curve and re-derive `f_aniso`. The headline "40–65% anisotropic"
  number is expected to **drop** once triaxiality-tracing is removed.

## 2026-06-17 — `lightcone`: WL-anisotropy notebook turned into a paper draft (cross-term = first-order)

Restructured `examples/wl_anisotropy_paper.ipynb` from preliminary results into a paper draft
(additive: kept working analysis cells, rewrote narrative, added Methods+stats; 30→41 cells).
**Key new science (verified numerically on the fid maps):** the exact identity
`ΔP = P_bind − P_sph = cross + auto`, with `cross = 2 Re⟨κ_dmo* 𝓕(κ_bind−κ_sph)⟩` (1st order) and
`auto = ⟨|𝓕δ_bind|²⟩−⟨|𝓕δ_sph|²⟩` (2nd order), gives band ℓ∈[3e3,2e4] **cross/ΔP≈1.39, auto/ΔP≈−0.39,
closure 1.000**. The anisotropic WL suppression is **first-order**, carried by the *alignment* of the
ejected gas with its own halo's DMO field — this **overturns the draft's "second-order on P_κ" claim**
(Part I's `f_quad` measured only the subdominant, wrong-sign auto term). This resolves the Part I/II
narrative contradiction and is the engine of the BCM conclusion.
- Added: **Methods** + a self-contained analytic **toy** (validates the multipole/Hankel estimator,
  the cross/auto sign, a null, and the `cos2Δθ` alignment dilution — runs with no ceph data);
  validation-rigour cell (`bind−sph` is pure `m≥2`); **bootstrap significance + null tests** for the
  Cℓ `f_aniso` (paired vs unpaired cosmic-variance check); peaks/PDF significance (conservative
  marginal errors → lower bound); parameter-response **significance gating** vs the fid noise floor.
- **Determination (Discussion):** a spherical BCM has zero cross term → fits `P(ℓ)` only by
  over-ejecting radially → biased gas profile → breaks multi-probe (tSZ/X-ray/kSZ/cluster-lensing)
  consistency and HOS (the first-principles cause of BCM peaks failing >ν4, 2201.08320). BCMs must add
  an aligned `m=2` term; BIND supplies the per-halo `c2(r)`. One-line takeaway in cell 0 + Conclusions.
- Also fixed the alignment-dilution comment (`0.55 ↔ Δθ≈28°`, not 35°: `cos70°=0.34`).
- New cells **written but not executed** (user chose "write code only"); toy + cross-term validated
  standalone. See [[wl-anisotropy-fieldlevel]] memory.

## 2026-06-17 — `lightcone`: WL-anisotropy notebook Part II built out (all 60 OAT runs done)

All spherical-BCM runs finished (fid, truth, tb0000–59 minus tb0032/tb0043, which need a maps
re-run). Expanded `examples/wl_anisotropy_paper.ipynb` Part II into a full field-level story,
every cell executed against the real data (headless):
- **Validation/profile proof** — stacked-cluster `c0(r)` of BIND vs spherical lie on top (L2 2.2%,
  corr 0.99998; residual is the inner zero-crossing), with the 2D `ΔΣ_bind−ΔΣ_sph` quadrupole map.
  Proves "same radial profile, different field" (the spherical model is a fair monopole control).
- **Field maps + parameter dependence** — `κ_bind−κ_sph` at a parameter's min/fid/max bound.
- **Cℓ & peaks vs parameter** — anisotropic part (`S_sph−S_bind`, `N_bind−N_sph`) as line families
  colored by normalized param value `u`; top responders are AGN BH + wind params.
- **Field 4 replaces the bar chart** with response curves `A(u)` (per-family) + a scale-resolved
  `∂(S_sph−S_bind)/∂u` parameter×ℓ heatmap.
- Data-layout fixes this session: twobound/truth composites are lightweight (no `composite`/`dmo`)
  → dual-total slabs (`composite`=spherical, `composite_bind`=BIND) + `assemble_lightcone(mass_key=)`;
  `kappa_dmo` reused from fid (run-independent). See [[wl-anisotropy-fieldlevel]] memory.

## 2026-06-16 — `lightcone`: spherical-BCM control for the WL-anisotropy paper (field-level test)

Added the machinery to answer "how much does **anisotropic** baryon redistribution matter for WL
*statistics* (Cℓ, peaks/minima, PDF/MFs)?" by building a spherical-BCM counterfactual lightcone and
decomposing against it. Key realization: a projected spherical BCM = the **azimuthal average of each
mass-matched halo patch** (the monopole `c0(r)` of `wl_anisotropy_paper.ipynb` Part I), so it is built
directly from the per-halo arrays already in `composite_slab*.npz` — **no 3D particles, no N-body
re-run**. BIND/spherical/DMO κ are assembled on **matched realization seeds** so cosmic variance + shape
noise cancel in `ΔX_aniso = X_bind − X_sph`.

- New `bind.inference.spherical` (`azimuthal_average_patch`, `spherical_total_slab`,
  `write_spherical_tree`) + CLI `bind-spherical-slabs`. Validated on snap_096: non-symmetrized re-paste
  reproduces stored `composite.sum(0)` to ~2e-6, mass conserved to 1e-6, biggest-halo quadrupole
  18.2%→0.0% with patch mass preserved.
- SLURM: `run_spherical_slabs.sh` (array over 20 snaps, `RUN=fid|truth|tbNNNN`) →
  `run_spherical_maps.sh` (assemble bind/sph/dmo on shared seeds, split DMO, run `bind-lightcone-stats`;
  peaks/minima `nu` fixed to the BIND σ0 so the three fields are comparable). Outputs →
  `ceph/bind_science/wl_anisotropy/<run>/{bind,sph,dmo}/`.
- Notebook: added **Part II — field-level decomposition** (Fields 1–4: example maps + anisotropy
  residual; Cℓ suppression & `f_aniso(ℓ)` (paired, expected small — Cℓ near-blind); peaks/minima/PDF
  `f_aniso(ν)` (where anisotropy should dominate); OAT `∂A_aniso/∂param`, field-level analogue of Pillar 3).
  Cells guard on missing overnight outputs and print the submit commands.

## 2026-06-16 — `lightcone`: from-scratch draft figure notebook (`examples/paper_lightcone_figs.ipynb`)

Built a single, self-contained figure notebook for `examples/draft.tex` (the "BINDing the lightcone"
feedback-atlas paper), one cell per draft section, **all plotting written fresh** (no reuse of the
`examples/*.py` generators' plotting code) in a uniform `scienceplots` `["science","no-latex"]` style with
**no titles**. 17 figures → `examples/figures_lightcone/*.pdf`, each loading a cached reduction from
`ceph/bind_science/` (no lightcone recompute); the last cell prints the figure→draft-section map.
Assembled by `examples/_build_paper_nb.py` (nbformat builder; kernelspec pinned to `bind_env`) and
verified end-to-end via `nbconvert --execute` (0 errors, all 17 render). Headline `r=0.91` group
f_gas→S(ℓ) reproduces (0.912).

Data-fidelity decisions worth remembering:
- f_gas(M,z) evolution uses the **raw R500c component ratio** (bg-subtracted fields exist only for snaps
  96/90/85; bg≈raw at z~0, 0.086 vs 0.093); the z=0 atlas + the r=0.91 correlation use bg-subtracted.
- §5.4 detectability was **reframed to a unit-safe fractional-response figure** (RMS ΔC/C across runs for
  κκ/yy/κy + LSST-Y10 σ(C_ℓ^κκ)). The cached **absolute C_ℓ^κy normalization is unreliable** (implied
  κ–y correlation coeff r_ℓ≈5e-8, unphysical) → an absolute κ×y S/N forecast needs an external SO/S4
  y-map noise curve; not fabricated. Noted in the notebook markdown.
- peaks/minima/cross y-ranges clipped (5 deg² high-ν tail is shot-noise dominated) and the
  "dominant knob" / §6.2 hierarchy use **band-averaged** |response| (not the noisy high-ν max), which
  fixed the spurious N_peak saturation.

**Revision pass (same day, per author feedback):** validation_field → 2-row (stats + relative residual)
× 6 cols incl. V0/V1/V2; fgas right panel → Δf_gas vs fiducial; pressure → shared x + envelope filtered
(≥30 halos, positive) — confirmed the deep-core envelope edge is the IMFslope excursion (real, printed);
budget → added BIND/hydro residual panel; **new `fig_morphology_fields`** (per-halo DM/gas/star patches
with measured centroids+ellipses, reproducing the morphology.py 2nd-moment estimator); cl → editable
`FAMILY_OVERRIDE` + printed/LaTeX family table (`param_families.tex`) + documented LSST-Y10/Euclid Knox
bands + right panel now r(f_gas,S) **vs ℓ** (peaks 0.92 at ℓ≈1037) with scatter inset; peaks/minima →
**cumulative** N(>ν)/N(<ν) to ν≈10 across all 5 source planes + RMS-across-runs feedback-sensitivity row
(grows to high ν, stronger at low z_s); MF → per-plane RMS sensitivity (V2≫V1≫V0) + sensitivity-vs-z_s;
cross → extended to the map Nyquist (ℓ≈3.7e4) with the CIC-aliasing regime shaded. **detect made
defensible**: absolute κκ/yy/κy now measured from the fiducial z_s=1 maps (FFT, physical r_ℓ≈0.6), signal
from the unit-independent fractional responses, LSST-Y10 κ-noise + stated SO-baseline y-noise (signal-dom
to ℓ≈3000), f_sky=0.4 → C_κy detection ≈766σ anchor; for distinguishing extreme feedback, C_yy/C_κy S/N
(≈300/190) exceed C_κκ (≈55) — the tSZ legs add power. hierarchy → family-coloured labels, annotated
values, leader-ringed, block-ordered.

## 2026-06-16 — `lightcone`: argument-first paper notebook (`examples/paper_wl_tsz.ipynb`)

Reorganized the existing science products into a single paper-structured figure notebook around the
thesis *"which feedback knobs control the Stage-IV WL/tSZ observables, and does WL×tSZ break
ejection-vs-heating."* Sections 1–7 follow the rebuilt outline; the **r=0.91 group f_gas→S(ℓ)** result
is the explicit spine (halos → field). The notebook is an *orchestration + narrative* layer: each
figure cell re-runs a validated `examples/*.py` generator from its **cached reductions** on
`ceph/bind_science/` (seconds each, no lightcone recompute) and displays inline; a `REGEN` toggle
switches to view-cached-only. Only the §1 hero (κ_dmo/κ_bind/y from `demo_maps.npz`) is plotted inline.
Verified end-to-end via `nbconvert --execute` (18 code cells, 0 errors, all figures render). No new
data products or generator changes — purely a presentation layer over the existing
halo_atlas/profiles/morphology/bridge/synthesis/fisher caches.

## 2026-06-16 — `lightcone`: pivot off Fisher → f_gas response model, feedback rankings, unified e-column, τ maps

Reviewed the Rung-0/1/2 science figures with the human and acted on four asks (Fisher set aside as
untrusted; focus on the gas observables we trust). Diagnosed the existing plots first: the
`baryon_budget`/`gas_vs_dm_extent` "feedback envelope" is min/max over the 60 twobound OAT runs (NOT
centred on truth — that guess was wrong); the fiducial pokes above it because the all-median point is
not a member of one-at-a-time perturbations + per-run generative noise (fix: include fid or use the
median twobound run). `morphology` panel-2 gas–DM offsets are sub-pixel at 50 kpc/h (resolution floor →
motivates raytracing). `tau_profiles` ≡ `frb_dm` (same electron column up to σ_T·pc).

- **(a) `examples/fgas_fit.py`** — calibrated `log10(f_gas/f_b)=L0(x,z)+Σ_k β_k(x,z)(u_k−u_fid)` from the
  halo_atlas caches + OAT design (CSV-normalized knobs, log-aware). Baseline RMS 0.012 dex, first-order
  reconstruction 0.025 dex over 5814 (run,M,z) pts. **WindEnergyIn1e51erg is the dominant f_gas knob,
  group response ≈2.6× cluster, growing toward low z** — the groups-bite-harder physics as fitted
  coefficients, replacing the envelope. Coeffs → `halo_atlas/fgas_model.{json,npz}`; fig `fgas_fit.png`.
  (Calibration bias: TNG truth ≈0.94× BIND fid, the known low-mass gas excess; baseline turns over above
  the data range — quadratic artifact, interpolation only.)
- **(b) `examples/feedback_response.py`** — OAT response rankings for gas/DM extent (@2r500, norm 0.3r500)
  and gas–DM centroid offset, group vs cluster, family-coloured. Confirms group ≫ cluster response.
- **(c) `examples/electron_column.py`** — collapses kSZ τ + FRB DM + gas column into ONE probe
  (τ/DM=σ_T·pc=2.05e-6), triple-unit-axis profile + BIND/truth validation (groups +11%) + column–mass
  scaling. Supersedes `tau_profiles.py`/`morphology.do_frb`.
- **(d) τ lightcone maps** — `bind.inference.lightcone_maps.assemble_lightcone` now emits a `tau`
  electron-column map (gas channel, per-plane a_l area factor, additive LOS, cumulative-to-source like y),
  pixel-aligned with κ/y → κ×τ, y×τ immediate. `bind-lightcone-maps` writes `tau_maps.npz`
  (`--no_tau` to skip; `DM=tau/TAU_PER_DM`). Born/additive projection is exact to first order for
  τ/y; lux deflection only matters for κ. Validated on snap_096 (τ_max~1e-3, DM_field-mean~57 pc/cm³/snap).
  T/entropy maps deferred — thermo channels are per-halo `thermo_patches` only (not composited), and
  intensive quantities need a defensible projected definition.

All on `lightcone`, untracked. CI ruff not run locally (no binary in venv); py_compile + import smoke pass.

## 2026-06-15 (pm) — `lightcone`: Rung 3 FINALIZED — true 2000-map hydro WL covariance

Replaced the 50-map BIND covariance (Hartlap 0.39, source of the spurious BHRadEff inflation) with the
**2000-realization hydro-replacement κ covariance** (Hartlap 0.99). New `examples/fisher_cov_hydro.py`
+ `run_fisher_cov_hydro.sh` (SLURM array over the 20 LP sets of `hydro_replace_RT/.../Ml_1e13_Mu_1e15_
Ri_0_Ro_5.0`, z_s=1 = plane 23, FOV 5 deg, 1024² — same binning/σ0 as fisher_precompute) → 2000 maps
of the 18 κ-only bins; `--combine` → `fisher_cov_hydro.npz`. Wired into `fisher_probes.py` as a combined
covariance: **hydro κ-block + BIND SZ-block & κ–SZ cross (via correlation, rescaled to the hydro κ
variances + hydro fiducial)**. κ-only — the tSZ block is still the 50-map BIND estimate.

- **BHRadEff dominance confirmed an artifact:** with the clean cov it only *marginally* leads WL
  (P(ℓ) 1.8 vs IMF 1.6; MFs 2.7 vs 2.4) — the WL field is genuinely AGN-tilted but undiscriminating.
  **Wind/IMF lead tSZ (WindEnergy 3.4) and the gas/peaks** — the physical channels (your expectation).
- **The true cov RECOVERS information the 50-map estimate washed out:** leading eigen-S/N 8→11; κ+tSZ
  now constrains **3 feedback combinations at 5 deg²** (modes 1→1→1→1→3; tSZ supplies modes 2-3).
- Scaling (`fisher_scaling.py`, auto-updated): mode 3 already at 5 deg²; mode 4 at 16, mode 5 at 26 deg²;
  LSST(18000)→23 combos (κ alone 14), 15 params pinned <½ prior. Degeneracies are **S/N-limited, not
  fundamental.** Individual params still degenerate at 5 deg² (1 pinned <0.8 prior) → measurables are
  COMBINATIONS.
- Caveats: tSZ block still 50-map (needs y maps for these fields to reach 2000); the 2000 maps are box
  rotations (stabilizes C⁻¹; absolute amplitude still box-limited). See docs/wl_tsz_plan.md.
- **`fisher_forecast.py` deprecated** (the naive 50-map/aliasing-bin version that produced the artifact);
  its `fisher_forecast.png`/`fisher_modes.png` removed. **Current Rung-3 figure set = `fisher_probes.png`,
  `fisher_probe_addition.png`, `fisher_corner.png`, `fisher_corr.png`, `fisher_scaling.png`** (all hydro cov).

## 2026-06-15 (pm) — `lightcone`: Rung 3 — area/realization scaling to break the degeneracies

`examples/fisher_scaling.py` (new, on fisher_probes): F ∝ area ⇒ eigen-S/N ∝ √A, so feedback mode k
(eigen-S/N ν_k at 5 deg²) crosses S/N=1 at A = 5/ν_k². Results (κ+tSZ):
- modes 1–2 constrained at 5 deg²; **mode 3 at 7 deg², mode 5 at ~30 deg²**; **LSST (18000 deg²) →
  23 constrained feedback combinations (κ alone 14), 12 params pinned <½ prior** (VariableWindVel,
  IMFslope, BHRadEff, WindFreeTravel, WindEnergy, Quasar…). ⇒ the degeneracies are **S/N-limited, not
  fundamental** — area breaks them; tSZ helps throughout (14→23 at LSST).
- **Realizations** (separate axis): Hartlap 50→0.39, 100→0.71, 300→0.90 — need **~300 realizations**
  (or a theory covariance) for <10% C⁻¹ bias; currently 50 (free 2× → 100). And the small modes'
  reliability *needs* those realizations before the area extrapolation can be trusted.
- Caveats: naive √A (no super-sample covariance; the 50 "realizations" are rotations of ONE 205 box,
  so large-area extrapolation is optimistic). `fisher_scaling.png`. See docs/wl_tsz_plan.md.

## 2026-06-15 (pm) — `lightcone`: Rung 3 CORRECTED — BHRadEff dominance was an artifact

User skepticism (expected Wind/IMF dominant, not BlackHoleRadiativeEfficiency) was right. Diagnosed the
naive forecast (`fisher_forecast.py`): BHRadEff #1 came from (a) the highest-ℓ C_ℓ^κκ bin (ℓ 8600-20000)
in the **CIC-aliasing regime** contributing 21 of its S/N², and (b) the noisy 50-real C⁻¹ overweighting
the WL-power bins, where **BHRadEff/IMF/Wind respond ~EQUALLY** (3.1/2.7/2.6% @ℓ1500 — power can't
separate them). The physical group f_gas response AND the WL peaks both rank **Wind/IMF top, BHRadEff #5**.

New `examples/fisher_probes.py` (robust forecast): drops the aliasing C_ℓ bin, shrinkage-regularizes the
covariance (α=0.3, toward its diagonal), and splits the data vector into probes.
- `fisher_probes.png` (per-probe S/N heatmap): WL P(ℓ)+MFs undiscriminating (BHRadEff≈IMF≈Wind); WL
  peaks/minima weak (5 deg² peak-starved, S/N~0.3); **tSZ discriminates → WindEnergy(3.5)/IMF(2.7)/
  VariableWindVel top** = physical.
- `fisher_probe_addition.png`: WL constrains **1 feedback COMBINATION** (leading eigen-S/N≈8), **tSZ adds
  the 2nd** (modes 1→1→1→1→2). The WindEnergy–IMFslope marginalized contour stays ~prior (degenerate).
- `fisher_corner.png` (27×27) + `fisher_corr.png`: marginalizing over all 27 params, **only ~1 is
  constrained <0.8 prior** — the measurable quantities are COMBINATIONS, not individual knobs.

**Corrected forecast:** the BIND WL+tSZ lightcone (5 deg², z_s=1) measures **~1–2 feedback combinations**
(the feedback-strength axis + a tSZ thermal direction); the earlier "BHRadEff #1 / 21-of-27 / 3 modes /
S-N 19" was inflated by the aliasing bin + noisy C⁻¹ — superseded. Adding area/realizations, the kSZ/FRB
field legs, and SBI (Sobol design) would add modes. See docs/wl_tsz_plan.md.

## 2026-06-15 (pm) — `lightcone`: Rung 3 kickoff — Fisher forecast (feedback constraints + tSZ value)

`examples/fisher_forecast.py` (new): the constraint forecast from the precomputed 29-bin WL+tSZ data
vector + 50-realization covariance + per-parameter responses (`fisher_precompute.py`; rebuilt
`fisher_resp` 9→27 params now that all runs are in g1_stats). F = dDᵀC⁻¹dD (Hartlap κ=0.61,
κ+SZ=0.39), in half-prior units → S/N = √F_ii.

- **21/27 feedback params detectable (S/N>1)** at z_s=1, 5 deg², no shape noise. Top: BH radiative
  efficiency 11.6, IMFslope 9.0, WindEnergyIn1e51erg 7.1, QuasarThreshold 6.6, VariableWindVel 6.2.
- **tSZ complementarity:** median 1.25× S/N gain, but **up to 3.3× for thermal/wind params**
  (WindEnergyReductionExponent, UVB, SNII_MinMass) that κ is blind to but y sees via temperature —
  the G1 complementarity realized as a forecast. `fisher_forecast.png` (detectability bars + a
  2-param Fisher ellipse: tSZ tightens & rotates the κ-only contour).
- Caveats: Hartlap 0.39 (29 bins / 50 real — collect the free 2× extra fiducial realizations via
  lux_collect to sharpen C⁻¹); 2-param *conditional* ellipse (others fixed); PoC z_s=1/5 deg²/noiseless.
- **Realistic forecast (`do_modes`, `fisher_modes.png`):** the 27-param Fisher is degenerate, so the
  honest answer is the eigenmode spectrum — **κ-only constrains 2 feedback COMBINATIONS, κ+SZ
  constrains 3** (eigen-S/N κ+SZ = 18.9, 11.1, 1.6; sharp drop after) — tSZ boosts the top two AND
  adds the 3rd (thermal) mode; **robust to LSST shape noise** (n=10, still 3, via `fisher_cov_ngal10`).
  Leading combination (S/N≈19) = BHRadEff − IMFslope − Quasar − wind ≈ the "feedback strength" axis
  the synthesis identified. Prior-marginalized per-param errors are near-prior (degeneracy) → the
  measurable quantities are the COMBINATIONS, not individual knobs. Next: SBI (needs the Sobol design),
  the τ field leg. See docs/wl_tsz_plan.md.

## 2026-06-15 (pm) — `lightcone`: cross-rung synthesis — one feedback axis, every probe

`examples/synthesis.py` (new; pure viewer over halo_atlas + profiles + morphology + dashboard_cache):
the cohesive story tying Rungs 0–2 + the field. Master axis = **group Δln f_gas** (the eROSITA-
contested R0 quantity); across the 57 runs at z~0 EVERY probe correlates with it: gas extent (R1,
kSZ) r=−0.96, baryon budget (R1) +0.92, gas–DM offset (R2) −0.98, halo FRB DM (R2) +0.96, WL
C_ℓ^κκ suppression +0.92, tSZ R(ν) +0.92 — AGN + SN/wind families both on the relation. ⇒ the
f_gas tension is not one number but the **master knob** propagating to the kSZ extent, morphology,
FRB DM, and the WL/tSZ field signals. `bind_story.png` (2×3). Next: Rung 3 (Fisher/SBI forecasts).

## 2026-06-15 (pm) — `lightcone`: Rung 2 — component morphology (shapes/offsets) + FRB DM

`examples/morphology.py` + `run_morphology.sh` (new): per-halo LOS-bg-subtracted 2nd moments within
1.5 r500 → median per-M200c-bin distortion |e| of gas/DM/star, gas–DM centroid offset, gas–DM
alignment; reduce+plot, SLURM array 0-61. FRB-DM figure reuses the Rung-1 profiles cache.

- **Shapes (`morphology.png`):** gas is ROUNDER than DM — |e| gas≈0.07, DM≈0.18, star≈0.5 (most
  elongated, the BCG+ICL); BIND=truth ~1-3% (the classic gas-vs-DM shape result, arXiv:1003.2270).
  gas–DM centroid offset ≈0.06 r500 (group) → 0.03 (cluster); feedback envelope WIDE at group scale
  (strong fb decouples the gas centroid). Panel 3 = which knobs offset the gas (IMFslope, BHRadEff,
  winds, Quasar/Radio).
- **FRB DM (`frb_dm.png`):** halo dispersion measure = the electron column of the τ leg. DM(b)
  profile + DM–M relation; DM(b=0.5 r500) = 145 (group) → 741 (cluster) pc/cm³, BIND=truth, feedback
  envelope wide (gas ejection lowers halo DM). Magnitudes sensible (cluster cores ~1000 pc/cm³).
- Caveat: per-halo shapes are resolution+latent noisy (group r500~5.5 px) — population median +
  feedback response carry the signal (like the bridge hero). Rungs 0–2 now done at z~0; Rung 3 =
  the lightcone forecasts (docs/wl_tsz_plan.md). See [[baryon-atlas-rung0]].

## 2026-06-15 (pm) — `lightcone`: Rung 1 finished (pressure + budget) + bridge τ leg folded

- **Rung 1 complete** (`examples/profiles.py --plot`, from the snap096 caches, no new reduction):
  - `pressure_profile.png`: scaled projected pressure (Compton-y) vs the projected universal Arnaud+10
    GNFW (`_arnaud_proj`) per mass bin — clusters follow, groups show a core deficit (AGN); BIND≈truth + envelope.
  - `baryon_budget.png`: cumulative f_bar(<r)/(Ω_b/Ω_m) vs r/r500 — groups baryon-deficient in cores
    (~50% cosmic, ejected), recover toward cosmic at large r; clusters close; envelope wide at groups;
    BIND≈truth. (bg-subtracted Σ cumulated with r·dr; pixel-area cancels in the ratio.)
- **Bridge IV — τ leg folded** (`bind_bridge.py --fig 4` → `bind_bridge_decomp.png`): the (κ,τ,y)→(M,f_gas,T)
  thermodynamic decomposition at peak-halos. κ∝M_tot (WL) barely moves, τ∝M_gas (kSZ = the Rung-1 gas)
  moves most, y∝Y (tSZ) intermediate — the **τ–y gap = heating**. f_gas=τ/κ (ejection), T=y/τ (heating) →
  the three projected probes jointly separate gas fraction from temperature (the plan's flagship
  decomposition). All three legs now in hand: κ←mass, τ←gas, y←gas×T. See [[bind-halo-field-bridge]].

## 2026-06-15 (pm) — `lightcone`: Rung 1 — gas-vs-DM extent profiles (the kSZ τ leg)

`examples/profiles.py` + `run_profiles.sh` (new): stacked, **LOS-background-subtracted** radial
profiles Σ(r/r500) of DM/gas/star/y/Pe per M200c mass bin (bg from the 2.5-3 Mpc/h annulus, as in
the f_gas atlas — patches are full-slab projections). Reduce+plot, per-(run,snap) cache in
`bind_science/profiles/`; SLURM array 0-61 for the full z=0→2.5 grid (same pattern as the atlas).

- **Headline (`gas_vs_dm_extent.png`, z~0):** gas IS more extended than dark matter (kSZ;
  ACT×DESI / Schaan+21 / Amodeo+21). gas/DM normalized-profile ratio @2 r500 = **4.2** (group
  1-2e13) → 1.3 (6e13-2e14) → **1.04** (cluster 2-5e14): feedback puffs gas out dramatically at the
  group scale (shallow potential), barely at clusters. **BIND = TNG truth to ~5%** across all bins.
- **Feedback envelope** (57 runs) is WIDE at group scale (strong fb → gas ~3× more extended than
  DM at large r) and narrow at clusters → "feedback extends the gas, most at group scale."
- f_gas (Rung 0) = *how much* gas; the extent = *where* it is — the two halves of the kSZ story;
  adds the **τ leg** to the halo↔field bridge. Pressure-vs-Arnaud + baryon-budget = stretch (TODO).
  See [[baryon-atlas-rung0]], [[bind-halo-field-bridge]].

## 2026-06-15 (pm) — `lightcone`: halo↔field bridge — 3 figures for the unified "BIND-TNG" paper

Connected the per-halo baryon atlas to the WL lightcone stats — the spine of a unified Paper I
("BIND applied to IllustrisTNG: a halo-to-field baryon model linking gas fractions to weak
lensing"). Bridge currency: BIND paints per-halo Δρ + thermo; every observable is that same
object integrated with a different kernel (aperture→atlas, lensing→κ, LOS pressure→y). Atlas +
lightcone are the SAME painted halos. New `examples/bind_bridge.py` → `bind_science/bridge/`,
all from existing caches (dashboard_cache + halo_atlas):

- **Fig 1 hero** (`bind_bridge_hero.png`): per peak-halo (42 peaks↔halos), per-run aggregated —
  **Δκ/κ vs Δln M_gas slope = 0.080 = the projected gas fraction** (that's why the slope is
  "shallow": gas is ~8% of the lensing mass — the slope IS the bridge coefficient). Per-run r=0.50;
  per-peak scatter ~100% of variance = latent morphology **+ multi-halo projection** (corrected an
  earlier "19% latent" claim). Panel B = ejection–heating plane (κ sees Δln M_gas; tSZ sees
  Δln Y = Δln M_gas + Δln T).
- **Fig 2 van Daalen** (`bind_bridge_fgas_sofl.png`): group f_gas (M200c 1-3e13, R500c, bg-sub)
  predicts the measured suppression S(ℓ)=C_ℓ^bind/C_ℓ^DMO at ℓ~1-2k with **r=0.91**. Group scale
  tightest (lightcone is group-dominated) = the SAME f_gas in tension with eROSITA → one quantity.
  Each point = 1 of 57 feedback lightcones, coloured by family (AGN/SN-wind/other).
- **Fig 3 correlation bars** (`bind_bridge_response.png`, was a saturated heatmap — replaced):
  |corr(field probe, halo Δf_gas group)| across 57 runs — **S(ℓ) 0.92, R(ν) 0.92, N_min 0.90,
  C_κy 0.87, V1 0.83, V2 0.82 all driven by halo f_gas; N_pk the exception (0.28, peak counts
  noisy)**. Panel B = R(ν) vs Δf_gas by family.

Gotchas: peak HEIGHTS / N_pk are noise-dominated (latent + multi-halo projection = the field
covariance); global stats S(ℓ)/R(ν) average it away (r~0.9). f_gas→S tightest at group scale +
ℓ~1-2k. See [[bind-halo-field-bridge]].

## 2026-06-15 (pm) — `lightcone`: eROSITA f_gas confrontation + projection-contamination fix

Sharpened the Rung-0 f_gas result into a publication tension figure vs Popesso+24/eROSITA —
which exposed that the earlier "saturates ~41% cosmic" was a **projection artifact**.

- **Finding:** per-halo patches are projected through the FULL slab (`pipeline.py` standard
  `extract_halo_cutouts` = full-box depth), so the aperture column carries a cosmic-f_b LOS
  background (whole-patch f_gas → 0.158 = Ω_b/Ω_m; Σ(R) plateaus to cosmic by ~2.5 Mpc/h).
  Raw aperture f_gas was LOS-contaminated. **Fix:** observer-style background subtraction from an
  outer annulus (2.5-3 Mpc/h), now in `halo_atlas.reduce_file` (`m_{gas,tot}_{ap}_bg`).
- **Mass def confirmed:** `halo_masses`=M200c (Group_M_Crit200), r200=r200c; added NFW+Duffy
  `m500c_from_m200c` (M500c/M200c≈0.72→0.67). Cross-check: bg-subtracted M(<R500c)≈0.8 M200c ✓.
- **Obs is a BAND, not one curve** (user caught that eROSITA looked "insanely low"): eROSITA/Popesso+24
  (slope 0.39 → f_gas,500=0.026@1e13, the contested strong-feedback low value) vs mainstream X-ray
  (Eckert+16/+19 slope 0.21 norm 0.131@2e14, Lovisari+15 slope 0.16 → 0.070@1e13). ~2.7× apart at
  group scale. Added `eckert_fgas`; figure plots both + shaded band.
- **Corrected result (`fgas_erosita.png`, `--plot_erosita`, z<0.2, M500c~1.3e13):** BIND fid
  f_gas,500=0.076 ≈ TNG truth ≈ **mainstream X-ray (Eckert) to ~9%** — BIND/TNG f_gas is NOT
  anomalous. **2.9× above eROSITA-Popesso** (declining to ~1.2× at clusters); strongest single knob
  → 0.047 (1.8× eROSITA, closes ~40% of the gap). ⇒ reaching eROSITA-low needs combined knobs
  (Sobol) or a non-TNG model. New: `--plot_erosita` (+ FORCE env in `run_halo_atlas.sh`). The earlier
  raw "41%" + raw f_gas-evolution absolutes are LOS-contaminated (relative trends robust). See
  [[baryon-atlas-rung0]].

## 2026-06-15 (pm) — `lightcone`: per-halo baryon-feedback atlas (Rung 0) + z=0 f_gas saturation result

New **per-halo** science axis complementary to the lightcone-stats plan (`docs/wl_tsz_plan.md`):
mine the twobound per-halo catalog (60 runs × 20 snaps × 4 slabs of `composite_slab*.npz`;
7 fields Mdm/Mg/Mstar/Y/T/S/Pe as 128² stamps; halos **identical across runs** → paired in
feedback; matched TNG300-hydro truth on same halos) into scaling relations the 2024-26
"strong feedback consensus" is built on (eROSITA f_gas, ACT×DESI kSZ τ, tSZ Y–M).

- **`examples/halo_atlas.py`** (new): generalizes `stats.halo_scaling` to per-(run,snap),
  vectorized aperture sums (R500c=0.659·r200, R200c) → per-halo Mdm/Mg/Mstar, f_gas, Y_500c,
  T_mw/K_mw/Pe_mw. `--reduce` caches one npz per run×snap in `bind_science/halo_atlas/`
  (restart-safe, atomic); `--plot` decodes params (median-of-runs + SB35 csv) → f_gas(M|θ),
  Y–M+scatter, group-scale saturation ranking. ~19 s/run×snap (4 slabs); full grid ~1.3 TB I/O.
- **`run_halo_atlas.sh`** (new): SLURM array 0-61 (tasks 0-59 twobound, 60 fid, 61 truth),
  one run/task all 20 snaps, restart-safe. `SNAPS=...` env for a z-subset. (User submits; the
  z~0 headline ran on the workstation — snap_096 × 62 runs.)
- **Validation:** BIND f_gas,500(>5e13)=0.137 vs TNG truth 0.132 (~4%); f_star ~2%; Y–M relation
  overlaps truth; σ(logY|M) 0.12–0.22 dex (BIND slightly over-dispersed at group scale =
  calibrated generative scatter, vs deterministic BCMs). `fgas_atlas_z0.png`, `YM_scatter_z0.png`.
- **Result (z~0, group scale 1-3e13, R500c):** cosmic f_b=0.159; BIND fid f_gas=0.093 (58%) ≈
  truth 0.086 (54%). **Strongest single-param depletion = 0.065 (41% cosmic), WindFreeTravelDensFac-hi**;
  SN/wind/IMF knobs dominate (not AGN) at group scale. **No single-parameter TNG excursion reaches
  the eROSITA strong-feedback band (~20-40% cosmic) — saturates at ~41%.** Combined-knob pushes
  (Sobol emulator) the open question. Caveats: single-param design, R500c-vs-obs-aperture not yet
  sample-matched, z~0.03. See [[baryon-atlas-rung0]].

## 2026-06-15 (pm) — `lightcone`: empirical feedback atlas notebook (`examples/wl_feedback_atlas.ipynb`)

New standalone notebook answering "how do non-Gaussian WL statistics depend on feedback"
as a pure empirical atlas (no editorializing) — built because `wl_baryon_response.ipynb`
only showed 9 params / only V2 and leaned on a "power spectrum is faint" thesis the data
don't support. Pure viewer of `dashboard_cache/g1_stats.npz`; computes nothing new.

- **Diagnosis:** the cache already has *all* 27 two-bound params with *every* statistic
  landed (`clk, pk, min, V0, V1, V2, sk` + paired errors, 57 runs). The "9 params / only
  V2 / auto-expands as restats land" notes in the old notebook are stale — the limits
  were hardcoded `[:9]` slices and V2-only plotting, not data. Tight MF error bars are
  *correct*: they're paired shared-sky errors (`bind-paired-stats`, ~10× < marginal).
- **Findings (paired σ, sample-variance only):** ~6 knobs carry all the signal —
  IMFslope, BlackHoleRadiativeEfficiency, WindEnergyIn1e51erg, QuasarThreshold,
  RadioFeedbackReorientation, SeedBlackHoleMass (AGN + IMF + SN energy); rest sub-few-σ.
  Small-scale C^κκ (ℓ>3000) responds ±8% (40–56σ) — *not* faint. Among MFs **V1
  (perimeter) > V2 (genus) > V0 (area ≈ null <0.05%)** — V0 null is a clean consistency
  check. Signs coherent across each row (one physical ejection/energy axis).
- Notebook sections (scaffolded build): (1) **fiducial** statistics multipanel
  (Cl/Npk/Nmin/V0/V1/V2, absolute — Cl from `g1_quick._fid_Pk`), (2) **R(θ)=S/S_fid** for
  the top-3 knobs (rows) × statistics (cols), blue=lo/red=hi bound, (3) 27×12 response
  matrix, (4) robustness check. §7-style per-halo mechanism deliberately dropped (user:
  "get the basics right").
- **"Why is BHRadiativeEfficiency so dominant / why is the response one-sided?"** Resolved
  via the SB35 **`LogFlag`**: most feedback knobs are sampled in **log**, with the fiducial
  at the **geometric mean** of [min,max] (verified: BHRadEff 0.2=√(0.05·0.8), etc.). So in
  the native coord (log10 θ for LogFlag=1) the two bounds are *equidistant* from fid — the
  asymmetry |δS(hi)|≠|δS(lo)| is genuine **curvature of S(log θ)** (saturation: a sensitive
  + an insensitive feedback regime), not a sampling/prior-width artifact. `IMFslope`
  (LogFlag=0, arithmetic-centered) is the near-linear exception. Caught that
  `dashboard_precompute.py` reads the CSV (`SB35_param_minmax.csv`) **for names only** and
  ignores LogFlag; the notebook now reads LogFlag directly. §4 reworked into "why is the
  response one-sided?" with native-axis 3-point curves; earlier prior-width caveat dropped.

## 2026-06-15 (pm) — `lightcone`: literal κ-peak ↔ halo association (§7 of wl_baryon_response)

Reworked §7 ("ejection vs heating") to tie the WL response to the *actual* peak-making
halos instead of all M>5e13 in one slab. Key enabler: all 60 twobound runs share the
lightcone geometry (rotations/displacements/lens planes) and differ only in painted
baryons — so the peak↔halo map is solved **once** on the fiducial and reused.

- Reproduced lux's Born ray-trace geometry to place every lightcone halo on the κ map:
  parse per-realization `rot`/`disp` from `rt_output/run001/config.dat`
  (raytracing.cpp::write_config), apply `load_phi` randomization + `calc_alpha`
  sampling, β=(pix·dLt−Lt/2)/χ → RT pixel. Orientation locked by correlating predicted
  halo-convergence vs real `kappa45.dat` (r≈0.6; all other transpose/flips ≈0). Top κ
  peaks match cluster halos logM 14.2–14.9 within a few px.
- `dashboard_precompute.py`: new `_peakhalo_map` (→`g1_peakhalo_map.npz`: peak table +
  unique peak-halo catalogue; 42 peaks ν>3 at 2′) and `g1_peakhalo`
  (→`g1_peakhalo.npz`: per-run R500 sums of all 7 painted fields Mdm/Mg/Mstar/Y/T/S/Pe,
  incremental, ~43s/run). Fiducial ref = LC lightcone (not twobound/run_0000).
- `g1_peakhalo_nu`: the lensing observable that *actually* changes between runs — each
  peak's HEIGHT. Samples each run's shared-sky `kappa_maps.npz[0, z_s=1]` (realization 0
  == fiducial `rt_output/run001` exactly) at the fixed peak pixels, 2′-smoothed,
  ν=κ_sm/σ0_fid → `{rn}_nu`. Reference κ map = `runs/bind/run_0000` (NOT twobound/run_0000).
- Notebook `examples/wl_baryon_response.ipynb` §7 rewritten: 7a (peak-halo
  ejection–heating plane + distance-from-1:1 vs high-ν peak response), 7b (all-7-field
  response fingerprint matrix), 7c (**Δν per peak** = (ν_hi−ν_lo)/2 vs the same halo's
  ΔM_gas and ΔY, small-multiples per parameter — the peak height changes between runs,
  same halo). Panels auto-expand as extraction lands; corr printed with N as preliminary.
- Honest result with all 27 knobs: high-ν peak-count response correlates with ΔM_gas
  (+0.49) AND the perpendicular heating distance (−0.65) at comparable strength →
  thermal-energy-like (f_gas·T), not pure ejection (matches the R(ν) finding). Narrative
  + Takeaway #4 reworded accordingly (was overclaiming "κ blind to heating").

## 2026-06-15 (pm) — `lightcone`: paired (shared-sky) response errors

Implemented the paired-error re-reduction (the real fix for "50 maps isn't enough":
the error, not the map count, was wrong). Realizations are rotations of the same box,
so the per-realization ratio S_r(run)/S_r(fid) cancels cosmic variance → paired error
~10× smaller than the stored marginal one (IMFslope high-ν peaks: 1σ→~11σ at 50 maps).

- `bind.inference.stats`: added `return_realizations` to `peak_counts` /
  `nongaussian_stats` (exposes the per-realization cubes they already compute).
- New CLI `bind-paired-stats` (`src/bind/cli/paired_stats.py`) + `run_paired_stats.sh`
  (array 0-60; task 0 builds the shared fiducial per-realization cache
  `paired_perreal_fid.npz`, atomic write). Writes `<run>/paired_stats.npz` with
  paired response+error for clk/pk/min/V0-V2/sk (z_s bins).
- `dashboard_precompute.g1_stats` prefers `paired_stats.npz` when present (overrides
  the marginal `{rn}_*`/`{rn}_*_err` + adds `{rn}_clk_err`, `_V*_err`, `_sk_err`,
  `paired_runs`). Notebook `wl_baryon_response.ipynb` auto-uses paired errors (matrix
  dots, V2 error bands); also fixes the V2 fake-∞ significance (errkey was None).
- Answered the map-count questions: 50 maps is plenty for paired responses; the
  fiducial has 100 ray-traced but only 50 collected (free 2× via lux_collect); more
  lensplanes only matter for absolute (unpaired) stats, not these responses.

## 2026-06-15 — `lightcone`: clean baryon→WL-NG-stats notebook + finished nufid restats

New scaffolded analysis notebook **`examples/wl_baryon_response.ipynb`** —
replaces the buried science inside the convoluted `wl_tsz_dashboard.ipynb` with a
linear argument: *30 subgrid knobs → sensitivity matrix → N(ν) → high-ν tail →
minima → V₂ → the (ΔM_gas, ΔY) ejection-vs-heating plane*. Pure viewer of
`bind_science/dashboard_cache`; generated by a script (kept in /tmp, not in repo).
Design choices (user): stop the physics rung at the ejection/heating plane (no
regression); sweep peaks→minima→morphology. Panels rank by **response amplitude**
(not >2σ) and auto-expand via `avail()` as products land.

Findings: (1) `g1_stats` cache was stale (23/60 runs) — refreshed via
`dashboard_precompute.py` → 57. (2) Fixed-ν peak/minima (`*_nufid.npz`) had only
landed for 20/60 runs (9 params); `run_nufid_restats.sh` was incomplete. Each
restat task ≈1m47s (verified locally). User submitting `sbatch --array=21-60
run_nufid_restats.sh` for the remaining 40; ran run_0019 locally. (3) At z_s=1 /
50 realizations, per-parameter peak responses are marginal (≤1.6σ, IMFslope) vs
the *conservative marginal* error stored in the cache — the paired shared-sky
error is smaller, so cross-parameter patterns + the many-σ halo-level (M_gas,Y)
plane carry the physics.

## 2026-06-11 — `lightcone`: WL×tSZ publication plan + three findings

Wrote the research plan for the WL-NG-stats × tSZ program →
**`docs/wl_tsz_plan.md`** (positioning vs 2025–26 literature, systematics tiers,
inference design, validation ledger, paper I/II/III split, decision gates).
Findings while auditing the pipeline:

- **R(ν) reframed** (user): `R∝f_gas·T_mw` = specific thermal energy, *not* a gas
  fraction (ν-slope is T∝M^⅔ thermometry; AGN cancels in f_gas↓·T↑). Plan: add
  τ/kSZ/DM planes (Gas channel × DMO velocities) → (κ,τ,y)=(M,f_gas,T̄)
  decomposition at peaks. Retire "projected gas fraction" wording.
- **lux already ray-traces tSZ**: y-planes sampled at *deflected* ray positions
  (`lux/raytracing.cpp:272`, `calc_y` :876) — only a Born-vs-RT quantification
  remains; same mechanism will serve DM/τ planes.
- **High-ν peak deficit is partly self-inflicted**: ν-histogram caps in
  `src/bind/inference/stats.py` (peak_counts ν≤6 :172; peak_cross *discards*
  ν≥6 :228/:252), plus no shape noise + single-box volume.

**Implemented (same session, after user review):** `stats.py` peak stats
upgraded — ν bins to 12 (peak_cross edges to 13, discard removed), multi-scale
`smoothing_arcmin`, shape noise (`shape_noise_ngal`/`sigma_e`, deterministic
per-(real,src) seeding) with `nu_norm={"map","noise"}` (analytic smoothed-noise
rms, verified 0.2% vs empirical); CLI gains the flags + `--peaks_only` +
suffixed outputs (`_ngal<N>`, custom); `run_sobol_stats.sh` gains
SMOOTHING/NGAL/SIGMA_E/STATS_EXTRA. Fiducial bind/truth/dmo peak stats
regenerated (legacy + `_multi` + `_ngal10`). **Stochasticity audit** (existing
halo_scaling npz): σ(logY|M) BIND 0.23–0.44 vs hydro 0.24–0.41 dex (+3–6% at
groups, matched at clusters); σ(f_gas|M) identical. kSZ/FRB/big-box feasibility
decisions recorded in `docs/wl_tsz_plan.md` §2/§3.6 (τ-first kSZ, κ-style DM
composite, Quijote ruled out → TNG300-2/3-Dark resolution gate).

---

## 2026-06-10 — `lightcone`: BIND validated vs hydro + R(ν) gas-fraction deep-dive

**BIND reproduces TNG300 hydro** (50 realizations, identical geometry): Cℓκκ
BIND/hydro≈1.00, suppression tracks hydro, **R(ν) BIND/hydro=0.98–1.06** (rises ×2
with ν), y-profile-at-peaks ~5%. R shown to be a gas fraction: `R_halo=Y/M ∝
f_gas·T` (corr 0.79), BIND f_gas(M) matches truth 2–11%. See
[[bind-validated-vs-hydro]].

- **Lightweight comparison notebook** `examples/lightcone_comparison.ipynb` loads
  only the stat npz (not maps) — instant, no kernel kill. Has BIND/hydro/DMO for
  all stats + an **R(ν) deep-dive** section (R∝f_gas·T, f_gas(M), R(ν), y-profile).
- `run_sobol_stats.sh` gained `RT_ROOT`/`SNAP_ROOT`/`N_REAL`/`KEEP_RAW` env so the
  BIND fiducial (maps in `bind_lightcone_tng/rt_output`) gets stats as a batch job
  without deleting the maps. `lux_collect` got `--n_real` + frees κ before y (OOM
  fix). Stats code verified: saved Cℓ == fresh recompute.

---

## 2026-06-10 — `lightcone`: BIND-vs-hydro validation notebook + DMO ray-traced baseline

- Truth patches complete (20 snaps × 4 slabs); per-halo check: BIND vs hydro
  compton_y (max 1.06e-4 vs 9.86e-5) and mass/halo (2.592e14 vs 2.588e14) agree to
  a few % on identical halos — apples-to-apples confirmed.
- **DMO baseline:** `bind-paint-lensplane --dmo` builds lensplanes straight from
  the stage-1 DMO maps (same geometry); `run_dmo_lensplane.sh` + lux
  (`COMPUTE_TSZ=False`, new env in `run_sobol_lux.sh`) + stats → ray-traced DMO κ
  for Cℓ-level suppression.
- **Notebook Part II** (`fiducial_lightcone_stats.ipynb`): BIND vs hydro for every
  stat (Cℓκκ, peaks, minima, PDF/skew/Minkowski, Cℓyy/Cℓκy, R(ν)) + ratio panels;
  baryonic suppression both projected-matter (recomposites truth on the fly) and
  ray-traced Cℓ (DMO/BIND/hydro); validation summary table. Guards on data
  availability so it runs incrementally as truth/DMO maps land.

Run order now: truth = `DESIGN=truth run_sobol_{paste,lux,stats}.sh --array=0-0`;
DMO = `run_dmo_lensplane.sh` → `COMPUTE_TSZ=False DESIGN=dmo run_sobol_lux.sh` →
`DESIGN=dmo run_sobol_stats.sh`.

---

## 2026-06-10 — `lightcone`: truth lightcone (TNG300 hydro) for apples-to-apples with BIND

Built a truth-lightcone path so the fiducial BIND lightcone can be compared to the
actual IllustrisTNG hydro on **identical geometry**.

- **`truth_lightcone.extract_truth_halos`** + **`bind-truth-halos`** (MPI): projects
  the real hydro fields (DM_hydro, Gas, Stars + compton_y/T/entropy/P_e) at the
  **same M≥10¹³ DMO-FoF halos** + lightcone transform BIND uses, into the BIND
  `composite_slab` patch format → drop-in for recomposite→lensplane→lux→stats.
  Per-particle thermo + comoving→physical a-factors match
  `process_simulations_multiz.py`, so truth compton_y is in BIND's units.
  Single-chunk validation: compton_y max 2.5e-4 / mean 3.8e-6, T_mw ~4e7 K — same
  scale as BIND's composite_thermo. (Overflow fix: divide y-integrand by physical
  pixel area per-particle before float32 cast.) See [[tsz-ymap-normalization]].
- **`run_truth_halos.sh`** (cca/MPI, array over 20 snaps) → `runs/truth/run_0000`.
  Truth lightcone then built with the SAME pipeline:
  `DESIGN=truth run_sobol_{paste,lux,stats}.sh --array=0-0`. Truth composite =
  DMO background (stage-1) + truth hydro patches (same as BIND's structure).
- hydro_replace2 reused for physics + TNG300 paths only (its pps=2/40-plane
  geometry isn't directly comparable to BIND's pps=4).

---

## 2026-06-10 — `lightcone`: prior-bound (1P) design + portable off-site generation bundle

Per user: start with one-parameter prior-bound variations (not Sobol) and make
generation Globus-portable to a GPU-rich machine.

- **`design.twobound_design`**: 30 astro params × {min, max} = **60 lightcones**,
  one param varied per row, cosmo at TNG300. Validated.
- **`bind-make-portable`**: builds a self-contained bundle (`design/`, per-run
  `params.npy`, `conditions/` = stage-1 stripped of DMO maps [compressed],
  `weights/`, `generate_halos_portable.sh`, `README.md`). Workflow: Globus bundle
  → GPU box runs `sbatch --array=0-59 generate_halos_portable.sh` (BUNDLE-relative
  paths, needs only `pip install -e BIND` + GPU) → Globus `runs/` back → CPU side
  `DESIGN=twobound` `run_sobol_{paste,lux,stats}.sh --array=0-59`.
  Built `/mnt/home/mlee1/ceph/bind_portable_twobound` (~20 GB: conditions
  dominate; DMO background stays home for compositing).

---

## 2026-06-10 — `lightcone`: GPU/CPU split + staged SLURM scripts matching conventions

Replaced the monolithic `run_science_run.sh` with **staged, convention-matching**
array jobs (cf. run_lightcone_*.sh / lux/run_lux_bind.sh), and split the GPU-heavy
generation from the CPU raytracing so the Sobol generation can run on a GPU-rich
machine:

- **`bind-generate-halos`** (`paint_stages.generate_halos`): GPU generation ONLY —
  model.generate on the stage-1 *cutouts*, saves per-halo patches (no compositing,
  no DMO). Portable: ship cutouts+params+weights to the GPU machine, generate,
  ship the halos back; compositing (DMO background) is CPU-side via
  `bind-paint-recomposite`.
- **Staged scripts** (array index = run index): `run_sobol_generate.sh` (gpu/a100,
  ~4 GPU-hr/run), `run_sobol_paste.sh` (cca: recomposite→lensplane+yplane, deletes
  composites), `run_sobol_lux.sh` (cca, `module restore lux`, `srun -n` scalable —
  raytracing speeds up with cores; uses lux `n_realizations`), `run_sobol_stats.sh`
  (collect→stats→delete lensplanes/raw). Keeps halos/kappa/y/stats only.
- Storage confirmed ample (100 TB); kept full halo patches (no scalars-only mode).
  GPU budget here (~497 hr ≈ 100 pts) → arrays default 0-99 (Sobol is incremental;
  the 256-pt design's first 100 are a valid sub-design). Removed monolithic
  run_science_run.sh / run_science_generate.sh.

---

## 2026-06-10 — `lightcone`: canonical (ray-traced) per-run pipeline + lean data management

Dropped the Born approximation as the production path (kept `assemble_lightcone`
only for quick diagnostics). The science runs now follow the canonical chain
end-to-end per run: **generate halos → composite → lensplanes + y-planes → lux
ray-trace → collect → statistics → delete transients.**

- **Driver** `run_science_run.sh` (SLURM-array, one task per run; resolves
  `RUN_DIR` from `DESIGN`+`SLURM_ARRAY_TASK_ID`). **Keeps** `snap_*/halos.npz`,
  `kappa_maps.npz`, `y_maps.npz`, stats; **deletes** composites, lensplanes, raw
  lux `.dat` (large 4096²-class, regenerable).
- New CLIs: `bind-extract-halos` (slim per-halo arrays — patches+metadata, no
  maps) and `bind-lux-collect` (stack lux `kappa{p}.dat`/`y{p}.dat` →
  `kappa_maps.npz`/`y_maps.npz`, `--delete_raw`).
- lux: `n_realizations` param (Sobol uses fewer than the fiducial 100);
  `main.cpp` loop now reads it.
- Orchestration: `science.plan_runs` writes per-run `params.npy` + the array
  submit command; `bind-science-plan` repointed to it (Born `plan_generate`/
  `plan_maps_stats` retained as library functions, not the default path).

---

## 2026-06-10 — `lightcone`: fiducial stats notebook + R(ν) tSZ-at-WL-peaks data vector

- **Notebook** `examples/fiducial_lightcone_stats.ipynb`: end-to-end fiducial WL+tSZ
  stats from the 100 ray-traced lux κ maps (+ lux y once landed) + composites —
  Cℓκκ tomographic, baryonic suppression (same-pipeline projected matter), peaks/
  minima, PDF/moments/Minkowski, κ×y / Cℓyy, halo scaling, and R(ν).
- **`stats.peak_cross_stats`**: stacks the (geometry-consistent) lux y at WL peaks →
  **R(ν,z_s)=⟨y⟩/⟨κ⟩** (projected gas-fraction proxy ∝ Y/M ∝ f_gas·T_mw), peak
  counts, stacked radial y profile (feedback shape), and Cov(κ_peak,y_peak).
  Validated on real maps: R rises ~2× with ν (z_s=1) and falls with z_s — exactly
  the expected gas-fraction behaviour; radial y profile declines 2e-5→1e-6 core→9′.
- Wired into `bind-lightcone-stats` (writes `peak_cross.npz` when per-source-bin y
  present) → R(ν,θ_astro) is now an emulator target. Made the Born assembler
  (`assemble_lightcone`) emit **tomographic** y (per source plane, like lux) so the
  Sobol runs get R(ν) too. `lux_io.load_y_realizations` reads lux `y{plane}.dat`.
  See [[tsz-ymap-normalization]] (y is physical, summed — no 1/a² weight).

---

## 2026-06-10 — `lightcone`: stats upgrades (Pylians/MFs) + lux tSZ y-emission

Per user requests on the stats and the fiducial end-to-end run:

- **Stats** (`bind.inference.stats`): power spectra now via **Pylians**
  `Pk_plane`/`XPk_plane` (BoxSize=fov_rad → k=ell); `peak_counts` returns peaks
  **and minima** (8-neighbour); replaced Betti with **Minkowski functionals**
  `V0,V1,V2` (differential estimator) in `nongaussian_stats`. Validated on real
  lux maps: Cl rises with z_s, skewness 2.0→0.54 with smoothing, V0 1→0,
  V2(Euler) +ve at high ν, peaks≫minima at |ν|>2.
- **lux ray-traced output reader** `bind.inference.lux_io` (kappa/gamma 1024²
  float64 Fortran-record; planes 26/45/59/70/78 → z_s 0.5/1/1.5/2/2.44).
- **Cleanup:** deleted the 900 non-{26,45,59,70,78} kappa/gamma maps across the
  100 fiducial RT runs (kept 1000).
- **tSZ via lux** (chosen approach): modified `/mnt/home/mlee1/lux`
  (`params.{hpp,cpp}` add `compute_tsz`/`tsz_input_dir`; `raytracing.{hpp,cpp}`
  add `load_y`/`calc_y` + a line-of-sight y accumulator written as `y{plane}.dat`
  with the SAME per-snapshot rot/disp randomization as the lensing planes →
  pixel-consistent with kappa). Syntax-checked (g++ stubs); user recompiles
  (`module restore lux && make`) + reruns. y-planes generated by
  `bind-paint-yplane` (single-field Fortran record, validated lux format) /
  `run_lightcone_yplane.sh`; tSZ config `lux/lux_bind_tsz.ini`.
- New CLIs: `bind-paint-yplane`. ⚠ z>0 thermo a-factors in calc_y still
  unvalidated (see [[lightcone-kappa-upturn-aliasing]] sibling caveat).

---

## 2026-06-10 — `lightcone`: Stage 3 summary statistics + maps/stats orchestration

Completed the 1a/1b analysis path: per-run summary statistics + wiring map
assembly and stats into the orchestration.

- **Stats (`bind.inference.stats` + `bind-lightcone-stats`):** flat-sky
  `power_spectrum`, `cl_kappa` (tomographic auto/cross + BIND/DMO suppression
  `S(ell)`), `cl_kappa_y` (WL×tSZ cross + `C_ell^yy`), `peak_counts`,
  `nongaussian_stats` (PDF, per-scale variance/skewness/kurtosis, Betti
  `beta0/beta1`), `halo_scaling` (per-halo `Y_500c, f_gas, f_star, T_mw` from
  composite patches). Writes `Cl_kappa.npz`, `Cl_kappa_y.npz`, `peak_counts.npz`,
  `nongaussian_stats.npz`, `halo_scaling.npz` per run.
- **Orchestration:** `assemble_lightcone` gained `manifest_root` (per-run
  composites + shared manifests); `bind-lightcone-maps` gained `--manifest_root`;
  `plan_maps_stats` emits `maps_stats_tasks_<design>.db` (1 task/run, maps→stats
  chained, restart-safe). `bind-science-plan` now emits both generate and
  maps+stats task files (n_real=32 fiducial, 8 for 1P/sobol).
- **Validated** on the demo maps + real composites: skewness 1.36→0.53 with
  smoothing (WL non-Gaussianity), S(ell)→0.91, ℓ(ℓ+1)Cℓ/2π~8e-5 @ ℓ=1000;
  halo_scaling **logY–logM corr 0.971** (Y∝M^5/3), f_gas~0.10, f_star~0.014,
  T_mw~5e6 K, positive κ×y and y×y. New CLIs: `bind-lightcone-stats`.

---

## 2026-06-10 — `lightcone`: tSZ y-map compositing + flat-sky kappa/y lightcone assembly

Built Stage 2c (the missing tSZ piece) and a self-contained kappa+y assembler.

- **Thermo compositing:** generalised `paste_halos_2d` to N channels and extended
  `build_bind_composite` with `thermo_patches=` → composites the 4 gas-thermo
  channels (compton_y, T, entropy, P_e) into full-box maps (`composite_thermo`,
  blended like gas: `alpha*canvas`, no DMO background, no scale_global). Saved in
  each `composite_slab*.npz`; wired through generate + recomposite. So
  `composite_slab` now holds the roadmap's 7-channel `composite_maps` content.
- **Lightcone assembly:** `bind.inference.lightcone_maps.assemble_lightcone` +
  `bind-lightcone-maps` CLI. Flat-sky Born stack of the per-snapshot composites
  onto a **common angular grid** (so kappa & y are pixel-aligned for the 1a
  cross-correlation). kappa = sum W(chi_l,chi_s)*delta_scaled per source z; y =
  sum of compton_y planes (additive, no kernel). Periodic-bilinear FOV tiling;
  random per-snapshot shift gives the N_real realizations. Falls back to
  compositing y from saved `thermo_patches` for older composites.
- **Demo (20-snap lightcone, z_s=1, 5deg, 1024px):** kappa std 0.023 (BIND) <
  0.0234 (DMO) — baryonic suppression visible; y mean ~1e-6, peaks ~2e-4 at
  clusters. Maps + figure at `/ceph/bind_science/demo_maps.{npz,png}`.
- Caveats: Born (not lux raytracing); demo mass used the existing square-taper
  composites while y used circular r200_factor=4 — science runs regenerated with
  thermo compositing + r200_factor=4 will be fully consistent. z>0 thermo
  a-factors still unvalidated (see [[lightcone-kappa-upturn-aliasing]] sibling
  CLAUDE.md note).

---

## 2026-06-09 — `lightcone`: science pipeline for 1a/1b — Stage 0 (design) + Stage 2 orchestration

Started the astro-parameter science pipeline (1a: WL+tSZ cross vs astro params;
1b: non-Gaussian WL emulator). No MAS fix needed — science ratios use same-pipeline
DMO so the aliasing artifact cancels (see entry below).

- **Stage 0 (design):** `bind.inference.design` + `bind-design` CLI. Cosmo pinned
  at TNG300 (indices {0,1,6,7,8}); 30 astro params varied in SB35 space (log/linear
  per LogFlag). Wrote `astro_params_{sobol(256),1P(150=5×30),fiducial}.npy` +
  `design.json` to `/mnt/home/mlee1/ceph/bind_science/design`.
- **Key efficiency:** stage 1 (DMO projection + cutouts) is parameter-independent
  → run once per snapshot, shared across all runs; only stage 2 (generate+composite)
  varies. All 20 shared stage-1 dirs already exist under `bind_lightcone_tng` and
  are reused as-is.
- **Stage 2 orchestration:** `bind.inference.science.plan_generate` + `bind-science-plan`
  emit per-run `params.npy` + restart-safe disBatch task files (one
  `bind-paint-generate --params ...` per run×snapshot, **circular taper r200_factor=4**,
  fm_redshift_thermo). Counts: fiducial 20, 1P 3000, sobol 5120 tasks. Added
  `--params` override to `bind-paint-generate`. Submit via `run_science_generate.sh`.
- Output root `/mnt/home/mlee1/ceph/bind_science`; N_sobol=256 (user choices).

Still to build: 2c tSZ y-map LOS assembly (thermo currently only per-halo patches,
not composited — needed for 1a and to avoid keeping TBs of patches), Stage 3 stats
(Cl auto/cross, peaks, PDF/Betti/moments, halo scaling), Stage 4 emulator set.

---

## 2026-06-09 — `lightcone`: diagnosed the convergence-power "upturn" + added anti-aliasing capability

The BIND lightcone κ power spectrum showed a sharp high-ℓ upturn vs kappaTNG.
Traced it (snap096, TNG300) at the **projected-mass level, no raytracing**: summed
the 4 `composite_slab*.npz` into a full-box map and compared 2-D P(k) against the
truth hydro projection and the lightcone's own DMO. Findings:

- **BIND is correct.** `BIND/DMO_lc` (same-pipeline ratio) matches the true
  `hydro/DMO_truth` baryonic suppression to a few % across all scales.
- The upturn is a **raw-CIC aliasing artifact in the particle→grid projection**
  ([`pixelize_z_projection`](../src/bind/inference/pipeline.py)), present *identically*
  in DMO and BIND (common-mode → cancels in same-pipeline ratios). Not the
  generative model, not compositing, not the 4-vs-2 slab count. Deconvolution
  alone makes it worse (it's additive, not a suppressed window).
- Pixel scale was never the issue: generation (128px/6.25 Mpc/h) and composite
  (4198px/205 Mpc/h) are both 0.0488 Mpc/h; the 4198→4096 lensplane step is a
  crop, not a resample — generative fidelity untouched.

Added an opt-in fix (default off, no behaviour change): `pixelize_z_projection(
mas_correct=True)` does **interlacing + CIC deconvolution** (`interlace_combine_2d`,
`cic_window_2d` in `pipeline.py`; unit-validated to 0.4% out to 0.8·k_Ny on
synthetic data). Wired through stage 1 (`project_and_extract(mas_correct=)` /
`bind-paint-project --mas_correct`) which stores a separate **`dmo_aa`** map —
the raw `dmo` stays the model-condition source. Stage 2 / recomposite auto-use
`dmo_aa` as the composite background when present, so the lensplane inherits it
with no lensplane change. Validation tooling: `examples/lightcone_projection_check.py`
+ a cell in `examples/lightcone_diagnostics.ipynb`. **For the science (S(ℓ),
peaks, non-Gaussian stats): ratio against same-pipeline DMO and no fix is
needed; `--mas_correct` is only for absolute κ-vs-kappaTNG validation.**

---

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
