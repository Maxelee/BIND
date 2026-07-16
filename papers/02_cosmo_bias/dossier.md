# Dossier — Paper II: Two nuisance parameters marginalize baryons for LSST cosmic shear

Mined from: `docs/WORKLOG.md` (2026-06-22 ×7, 06-24 ×3, 07-03), worktree
`wt/wl-cosmo-bias/examples/*.py` (docstrings + read in full), on-disk artifacts in
main-tree `examples/figures_lightcone/` and `examples/rescale_*` (`.png`/`.npz`,
opened and/or loaded with `numpy.load` to cross-check WORKLOG numbers against the
actual saved output — not re-run), and `docs/cosmo_rescaling_plan.md` (worktree
`wt/cosmo-rescale`). No notebooks were named by the brief for this paper, so no
`extract_nb_figures.py` pass was needed.

**Provenance-verification method used throughout:** where a WORKLOG entry states a
number, I cross-checked it directly against the corresponding saved `.npz` in
`examples/figures_lightcone/` by loading the arrays (`python3 -c "import numpy..."`,
read-only, no engine re-run). Where verified this way I give both the WORKLOG line
and the exact recomputed value; they agree everywhere checked. This is stronger
provenance than the WORKLOG text alone and is noted per-entry as "(verified against
`<file>.npz`)".

---

## Intro (background material only — no new numbers to mine; sourced by the writer from the seed-citation list)

No engine/WORKLOG content directly supports the Intro's framing claims (S8 tension,
current BCM/PCA/scale-cut practice) — those are literature background for the
citer/writer to source. The one number from mining that *anchors* the intro's "how
many nuisance numbers" framing is the headline itself (see Results §3).

## Methods

- **Suite recap.** 256-node Sobol design over the 30 CAMELS-TNG astrophysical
  parameters, fixed TNG fiducial cosmology; **256-node design, 253 usable runs**
  is the canonical final count (WORKLOG 2026-06-29 "§6 fix" entry: "253 runs + fid,
  ~34k halos/run over 20 shells z=0.034–2.444"; WORKLOG 2026-06-26 kSZ-field entry:
  "253 nodes" in `emulator_dataset.npz`). Individual analyses in this paper were run
  at intermediate suite sizes as the array completed over the campaign — **96/256**
  (Step 1 first cut), **99/256** (Step 1–4 "refresh", incl. `cosmo_bias.png` and
  `bias_tomography.png` figures on disk), **216 runs** (capstone, `cosmo_bias_capstone.py`
  — verified: `cosmo_bias_capstone.npz` was built from a 216-run `emulator_dataset.npz`
  per its docstring), **237 runs** (`lightcone_beyond2pt.py`/`lightcone_shear_forecast.py`
  docstrings: "237 SB35 Sobol runs"). *Caveat for the writer:* state the canonical
  253/256 figure in the abstract/methods and note in a footnote that specific figures
  were computed at the array-completion count current at the time (96/99/216/237);
  do not present these as different results.

- **S(ℓ,z_s) measurement w/ paired DMO trace.** `S_i(ℓ,z_s) = C_ℓ^κκ(θ_astro^i) /
  C_ℓ^κκ(DMO)` per tomographic source plane (z_s = 0.5, 1.0, 1.5, 2.0, 2.44). DMO
  reference = paired lux trace at `bind_science/runs/dmo/run_0000`, same RT_SEED=1992
  + transforms as every BIND run. Cosmic-variance cancellation verified
  realization-by-realization: **map correlation r≈0.983–0.986** (WORKLOG 06-22 Step 1;
  matches brief's "r≈0.98"; also stated in `lightcone_transfer.py` docstring as
  "r ~ 0.98"). ℓ capped at 1.5e4 (`ELL_MAX_TRUST`) — above this CIC aliasing dominates
  run-to-run scatter (ratio-of-means cancels most of it but numbers are conservative
  above this cut). Engine: `examples/lightcone_transfer.py`.
  Provenance: WORKLOG:897-916 (Step 1 entry).

- **Effective-bias pipeline.** Linearised Fisher bias, Schneider et al. (2020)
  formalism: `Δθ = F⁻¹B`, baryonic residual `ΔC^ii_ℓ = C^ii,fid_ℓ (S_i(ℓ)−1)`, full
  5×5 Gaussian tomographic covariance (auto data vector + cross terms + shape
  noise). Cosmology derivatives ∂C/∂(Ωm,σ8) from **pyccl** (Eisenstein-Hu transfer +
  halofit; installed `pyccl` 3.3.4 via `pip install --only-binary`, WORKLOG 06-22),
  **validated 2–5% vs the sim DMO C_ℓ** across ℓ=200–4000, all 5 source planes.
  LSST-Y10 survey spec used for significance: f_sky=0.4, n_gal=27/arcmin² (split
  over 5 planes), σ_e=0.26 (`lightcone_cosmo_bias.py` constants `F_SKY`,
  `N_GAL_TOT`, `SIGMA_E`). Bias Δθ itself is f_sky-independent; only its
  significance scales as √f_sky. Engine: `examples/lightcone_cosmo_bias.py`.
  Provenance: WORKLOG:873-896 (Step 2).

- **Template construction.** N baryon-template nuisance parameters = the leading
  SVD (unnormalized/uncentered) modes of the residual vectors `dC_r = C_fid·(S_r−1)`
  across all runs — an empirical basis for the WL feedback manifold, the WL analogue
  of Lin+2026's 2D latent. Sweeping N tests necessity (N=1 insufficient), sufficiency
  (N=2 nulls bias), and cost (σ_S8 inflation grows monotonically with N).
  Engine: `examples/cosmo_bias_capstone.py`. Provenance: WORKLOG:571-589
  (2026-06-24 "capstone" entry), independently verified against
  `examples/figures_lightcone/cosmo_bias_capstone.npz` (see Results §3).

- **Held-out validation protocol.** Templates built from even-indexed runs, tested
  for bias removal on odd-indexed runs (WORKLOG 06-24 capstone entry: "templates
  from even runs kill the bias on odd runs, ≤0.08σ"). **Gap:** no separate saved
  `.npz`/figure documenting this even/odd split was found on disk under
  `examples/figures_lightcone/` or in the worktree — the ≤0.08σ number is sourced
  from the WORKLOG text only, not independently re-verified from a saved array.
  The in-sample (all-216) analogue, N=2 max residual = 0.073σ at ℓmax=5000, *was*
  verified directly (see Results §3) and is consistent with (slightly better than)
  the held-out figure, as expected.

- **Robertson+2026-style full likelihood.** `examples/lightcone_shear_forecast.py`:
  full cosmology {Ωm, σ8, h, ns} + N=2 template amplitudes {a1,a2}, 5 BIND-native
  source planes, LSST-Y1 survey spec **(Robertson Tbl 1: 12,300 deg² → f_sky=0.298,
  n_eff=9.78/arcmin², σ_e=0.26)**, flat-ΛCDM priors from **Robertson Tbl 2**
  (Ωm∈[0.20,0.46], σ8∈[0.39,1.01], h∈[0.65,0.78], ns∈[0.95,0.99]), sampled with
  **emcee** (`EnsembleSampler`, parallel `multiprocessing.Pool`, BLAS pinned to 1
  thread per worker to avoid oversubscription). The "truth" mock is a real BIND
  run's suppression S(ℓ,z_s) (Asimov, no noise) applied to the CCL fiducial theory
  — i.e. this tests model misspecification (is the run's suppression captured by
  2 templates?), not just a linear Fisher bias. Companion script
  `examples/lightcone_shear_sweep.py` (a "both-compare" arm: N=2 templates vs. the
  full 30-param feedback emulator via a GP surrogate of the suppression manifold,
  PCA+GP, `k=8` components) extends this to multiple truths — file present but not
  deeply mined beyond its docstring (Sobol Step-2 successor; ran on a 12-run
  representative weak→strong local subset, full 237-run sweep deferred to sbatch
  per its own docstring — **not run here, per hard rules**).
  Provenance: docstrings of `lightcone_shear_forecast.py` / `lightcone_shear_sweep.py`;
  numbers verified against `examples/figures_lightcone/shear_forecast_y1.npz` and
  `shear_sweep{,_linear}.npz` (see Results §4). **No matching WORKLOG entry exists**
  for this step despite the brief's instruction to search WORKLOG for "Robertson" /
  "beyond2pt" / "shear_forecast" — flagged as a gap (see Gaps list); the numbers
  below come directly from the on-disk output artifacts instead.

- **Beyond-2pt manifold-geometry method.** `examples/lightcone_beyond2pt.py`: for
  10 lensing/SZ statistics (κ Cℓ, peaks, minima, PDF, Minkowski V0/V1/V2, moments,
  tSZ-y peaks, κ×y cross), regress the standardized observable on the 30 unit-cube
  astro params (`Beta = d(stat)/d(theta)`, denoises against shot noise uncorrelated
  with θ); left singular vectors of `Beta` = active feedback directions
  (linearised active subspace, **Constantine 2015**); dimensionality = cumulative
  singular-value spectrum to 90%/99% variance; **complementarity** between
  statistics = principal-angle subspace overlap vs. the κ Cℓ active subspace (1 =
  same directions, <1 = complementary → enables self-calibration).
  Provenance: `lightcone_beyond2pt.py` docstring + code; numbers verified against
  the rendered `beyond2pt_manifold.png` (see Figures).

## Results

### (1) Transfer-function ensemble: brackets suppression AND enhancement
- Ensemble spans **~0.80→1.24** in S(ℓ) over 2×10³<ℓ<10⁴ at z_s=0.5.
- **59% of runs enhance WL power at ℓ~500–2000** (cooling bump) — brief's "59% of
  runs enhance at ℓ~1e3" traced to this exact WORKLOG phrase.
- Clean redshift dilution: median band-suppression 0.938→0.982 from z_s=0.5→2.44;
  run-to-run σ[S(ℓ)] grows with ℓ, largest at low z_s.
- Group f_gas orders the curves only moderately (Spearman ρ=+0.27→+0.34 across ℓ)
  — a preview of the Step-5 "1-param models don't span BIND" result.
- At 96/256 runs at first cut. Refreshed to 99 runs: results stable (Step 2
  unchanged; Step 3 top-2 knobs swap rank within noise; Step 4 rank-1 at 99.7%,
  later superseded — see (6)).
- Provenance: WORKLOG:897-916 (Step 1), WORKLOG:843-846 (99-run refresh).
- Figures: `transfer_function_ensemble.png`, `transfer_function_tomography.png`.

### (2) Bias budget: up to 6σ if ignored, ℓmax-dependent
- Linearised Fisher bias on LSST-Y10: bias is a **tight 1-D locus** in (ΔΩm,ΔS8),
  slope dS8/dΩm≈0.4–0.67, rotating with ℓmax.
- Group f_gas orders ΔS8 far better than it ordered S(ℓ): Spearman
  ρ(f_gas,ΔS8)≈+0.51–0.58 (vs +0.27 for S itself) — projecting to cosmology
  amplifies the gas signal; gas-poor/strong-feedback → ΔS8<0.
- Fraction of runs with |ΔS8|>2σ: **1%/9%/22%** at ℓmax=2000/3000/5000; **max 6σ**
  (at ℓmax=5000, LSST-Y10); ΔS8 95% CI ±0.01–0.014.
- Verified directly against `cosmo_bias_capstone.npz` (216-run capstone dataset,
  the N=0 "no baryon" column): max |ΔS8|/σ ratio = **2.003 / 3.590 / 6.028** at
  ℓmax = 2000 / 3000 / 5000 — matches the brief's "2.0/3.6/6.0σ" exactly (these
  numbers appear in the WORKLOG 06-24 capstone entry, not the 06-22 Step-2 entry,
  which used the smaller 96-run dataset and reported "sub-σ at ℓmax=3000" — i.e.
  the *fiducial/median* bias, not the *max*; the two entries describe different
  quantiles of the same distribution, not a contradiction).
- Caveat (WORKLOG-stated): statistical-only/2-param significance (no h, ns, IA,
  photo-z marginalisation) → optimistic; the ΔS8 magnitudes are the robust
  headline, not necessarily the exact σ-significance.
- Provenance: WORKLOG:873-896 (Step 2, 96-run), WORKLOG:571-589 (capstone, 216-run,
  headline 6σ number), verified `cosmo_bias_capstone.npz`.
- Figures: `cosmo_bias.png` (99-run version — Fig 1 scatter + Fig 2 significance
  histogram), `cosmo_bias_capstone.png` left panel top curve (N=0, 216-run version).

### (3) N-template ladder: N=1 insufficient, N=2 sufficient, cost ~3×
**Headline result of the paper.** Schneider+2020 Fisher bias on LSST-Y10 (5
tomographic κ autos, CCL halofit derivs, Gaussian cov+shape noise) using BIND
ΔC=C_fid·(S−1) for all 216 runs, N baryon-template nuisance params (leading SVD
modes of the 216 ΔC) swept N=0…10.

Verified directly against `examples/figures_lightcone/cosmo_bias_capstone.npz`
(exact values, all three ℓmax cuts; `sigX/sig0` = cost multiplier vs. no-baryon
floor):

| ℓmax | N=0 max\|ΔS8\|/σ | N=1 max | N=2 max | N=2 95th | PC1/PC2/PC3 (evr) | N=2 cost | N=6 cost | N=10 cost |
|---|---|---|---|---|---|---|---|---|
| 2000 | 2.003 | 0.154 | 0.023 | 0.016 | 92.5/5.2/1.0% | 2.95× | 3.86× | 5.20× |
| 3000 | 3.590 | 0.331 | 0.027 | 0.019 | 91.1/7.1/0.7% | 3.21× | 4.32× | 5.77× |
| 5000 | 6.028 | 0.674 | 0.073 | 0.050 | 88.6/10.0/0.5% | 3.39× | 4.79× | 5.59× |

- **N=1 leaves up to 0.67σ residual** (max ratio 0.674 at ℓmax=5000) — matches
  brief exactly.
- **N=2 nulls it to ≤0.07σ in-sample for all 216 runs** (max ratio 0.073 at
  ℓmax=5000) — matches brief's "N=2 → <0.08σ" almost exactly (0.073 < 0.08); the
  brief's "≤0.08σ, held-out validated" figure is the (unverified-on-disk)
  cross-validated number from WORKLOG text — see Methods gap above.
- **Cost ~3×**: N=2 inflates σ_S8 by 2.95–3.39× across ℓmax=2000–5000 (WORKLOG
  states "~3×"). N=6 → 3.86–4.79×; N=10 → 5.20–5.77× — "pure cost" since residual
  bias barely improves past N=2 while σ_S8 keeps degrading (matches "BCM N≥6 pure
  cost").
- **Manifold is 2D**: PC1 88.6–92.5%, PC2 5.2–10.0%, PC3 0.5–1.0% of the ΔC
  residual-variance spectrum, across all three ℓmax cuts — matches brief's
  "PC1 89-92%, PC2 5-10%, PC3 <1%" (edge values at ℓmax=5000 slightly outside the
  89% floor / 1% ceiling quoted in the brief; note this nuance for the writer).
- Provenance: WORKLOG:571-589; **independently verified** by loading
  `cosmo_bias_capstone.npz` directly (script: `examples/cosmo_bias_capstone.py`).
- Figure: `cosmo_bias_capstone.png` (2-panel: residual bias vs N [left, log-y,
  3 ℓmax curves, median+95th], σ_S8 inflation vs N [right]).

### (4) Full-likelihood confirmation: Y1 N=2 → S8 bias −0.37σ at 1.11× cost
Verified directly by loading `examples/figures_lightcone/shear_forecast_y1.npz`
(6000-sample emcee chain, {Ωm,σ8,h,ns,a1,a2}, `truth` = run index 16, the
strongest-feedback SB35 run used as the Asimov mock):
- input S8 = 0.827914; recovered S8 = 0.824004 ± 0.010631 (median ± std of the
  marginal chain).
- **bias = −0.368σ** (matches brief's "−0.37σ" to 3 sig figs).
- no-baryon-floor σ(S8) (2-param cosmology-only Fisher at fiducial) = 0.009601.
- **marginalisation cost = σ_marg/σ_floor = 1.107×** (matches brief's "1.11×").
- The `lightcone_shear_forecast.py` script itself prints a comparison line: "(Robertson
  HMCODE Y1: ~1.8×; capstone Fisher N=2: ~3×)" — i.e. the BIND N=2 data-driven
  template basis achieves a **lower** marginalisation cost than both the reference
  Robertson HMcode-AGN nuisance (~1.8×) and BIND's own earlier linearised-Fisher
  N=2 estimate (~3×, see Results §3) — worth flagging explicitly for Discussion as
  the "prior-informed sampled posterior beats the flat-prior linear Fisher" nuance.
- Companion sweep `shear_sweep.npz` (12-run local "weak→strong" subset, `cosmo`
  vs `templates` models): for the `templates` model across the 8 runs with both
  entries, recovered σ ranges 0.01021–0.01070 vs the sweep's own no-baryon floor
  0.009300 → cost range ≈1.10–1.15×, consistent with the Y1 headline. The `cosmo`
  (no-baryon) model's recovered S8 in this subset ranges 0.8186–0.8267 vs input
  0.827914 (bias present but this sweep does not compute σ per-run for the
  no-baryon case in a form directly comparable to the σ-significance headline —
  not separately quoted as a distinct number in the dossier to avoid
  over-interpreting an intermediate artifact).
- Provenance: `examples/lightcone_shear_forecast.py` docstring + code (method);
  `shear_forecast_y1.npz`, `shear_sweep.npz` (numbers, loaded directly — **no
  WORKLOG entry documents this run**, see Methods gap note).
- Figure: **none rendered on disk** — `lightcone_shear_forecast.py` saves only
  the raw `.npz` chain, no corner/triangle plot script exists in the worktree.
  Flagged as a gap (see Gaps list) — the writer needs either a `\todo{}` or to
  generate one respecting the "no analysis engine" rule (i.e., NOT permitted here;
  must be flagged to the human).

### (5) Beyond-2pt: well-measured statistics live on a ≤2–3D manifold
Verified against `beyond2pt_manifold.png` (237-run dataset per script docstring):
- **Left panel** — cumulative feedback-manifold variance vs. # templates for 10
  statistics (κ Cℓ, peaks, minima, PDF, Minkowski V0/V1/V2, moments, tSZ-y peaks,
  κ×y cross): all reach ≥0.99 cumulative variance by N=2–3 templates *except* κ
  peaks and κ minima, which are flagged "noise-limited" (dashed, feedback SNR<1
  per-realization) and only reach ~0.80–0.88 by N=10 — i.e. shot-noise-limited,
  not genuinely higher-dimensional. This matches the brief's "all well-measured
  statistics live on a ≤2–3D manifold" (the *unmeasured-well* exception being
  explicitly noise, not signal).
- **Right panel** — feedback-direction overlap (principal-angle subspace
  alignment) matrix, 10×10: κ Cℓ vs peaks/minima/PDF ≈0.85–0.98 (same directions);
  κ Cℓ vs Minkowski V0/V1/V2/moments ≈0.65–0.82 (partially complementary — MFs
  probe a rotated combination); κ Cℓ vs tSZ-y peaks = 0.90, κ Cℓ vs κ×y cross =
  0.96 (both nearly aligned with the 2pt manifold, consistent with the Results-(4)
  script's remark that tSZ mainly sharpens rather than expands the accessible
  directions — cf. WORKLOG 06-24 "WL feedback latents" entry: "orthogonal tSZ
  probes don't expand the dimensionality much... but sharpen and rotate").
- **MF/PDF complementary to κκ** (brief's phrase): Minkowski V0/V1/V2 and moments
  show the lowest alignment with κ Cℓ (0.58–0.82, off-diagonal minima at
  V0↔peaks=0.58, PDF↔V1=0.59, PDF↔V2=0.61) — the most genuinely complementary
  directions in the matrix, i.e. these are the statistics most likely to help
  self-calibrate the residual (non-2pt) feedback shape.
- Also computed in the same script (`kappa_cost_anchor()`, printed not plotted):
  a reprise of the κ Cℓ σ(S8) cost ladder at ℓ<3000 using the 237-run dataset —
  a consistency check against Results (3), not separately re-verified here.
- Provenance: `examples/lightcone_beyond2pt.py` (method + code); figure read
  directly.
- Figure: `beyond2pt_manifold.png`.

### (6) Which knobs drive the bias; tomographic self-calibration (CORRECTED)
- **Top drivers (99-run Step 3, `bias_sensitivity.png`):** IMFslope (S_T≈0.36) and
  BlackHoleRadiativeEfficiency (S_T≈0.32) — roughly 2/3 of the total-order variance
  between them — then RadioFeedbackReorientationFactor (~0.17 total), WindEnergyIn1e51erg
  (~0.09 total). Surrogate GB model CV R²=0.38 (96-run version: 0.34). Strong
  interaction component (S_Ti≈2×S_i) → the bias lives in parameter *combinations*
  (the f_gas-controlling AGN+IMF sector), consistent with the ~2 effective
  dimensions found elsewhere in the suite (Sobol feedback-latent memory note).
  Three estimators agree on ranking: binned main-effect S_i, GB-surrogate
  Saltelli/Jansen S_i/S_Ti, permutation importance.
  Provenance: WORKLOG:848-871 (Step 3, both 96- and 99-run versions);
  `examples/lightcone_bias_sensitivity.py`. Figure: `bias_sensitivity.png`
  (matches: bar chart shows IMFslope > BlackHoleRadiativeEfficiency >
  RadioFeedbackReorientationFactor > WindEnergyIn1e51erg, exactly this order, both
  total-S_T and first-order-S_i bars).

- **⚠ Self-calibration — the brief's mandatory CORRECTION.** The *original* Step-4
  finding ("z-evolution is rank-1: PC1=99.7% of variance ⇒ baryons act as
  essentially one nuisance amplitude ⇒ WL tomography alone barely self-calibrates
  baryons") was flagged **WRONG** by the human (WORKLOG 2026-06-22, "⚠ CORRECTION
  to Step 4 self-calibration claim" entry). Flaw: it (a) collapsed each source
  plane to a single amplitude, discarding the in-bin ℓ-shape that actually drives
  self-calibration, and (b) used 5 nested δ-sources of *one* lightcone (near-rank-1
  by construction); rank-1 of the run-to-run variation answers "do feedback models
  bias in the same z-direction?", not "can tomography separate baryons from
  cosmology?".
  **Corrected result** (`examples/lightcone_selfcal.py`, joint Fisher on
  (S8,Ωm,A_bary) using the full-ℓ tomographic auto-spectra, A_bary scaling the
  measured suppression template): **tomography DOES self-calibrate.** The baryon
  bias z-signature *declines* 83% from z_s=0.5→2.44 (a true S8 shift would be
  flat) → separable. With 1 bin: σ(S8) degrades ×2.37 when marginalising A_bary,
  r(S8,A)=−0.91 (strongly degenerate). With **≥2 well-separated bins: degradation
  ×1.01, r→+0.12** (degeneracy broken, marginalisation ~free). Honest nuance: the
  feedback *shape* variation beyond the mean template is itself ~rank-1 (leading
  residual mode = 98% of residual variance); 5-bin tomography absorbs the leading
  amplitude for free, but marginalising the 2nd (shape) mode costs **×3.4** in
  σ(S8) — i.e. tomography nails the dominant mode but the residual feedback shape
  (what the N=1 template misses; cf. Results 3) stays partly degenerate even with
  full tomography.
  Provenance: WORKLOG:779-800 (CORRECTION entry). Figure: `selfcal.png` (left:
  per-plane bias declining with z_s vs. the flat pure-S8-shift line; right: σ(S8)
  degradation bar chart vs. # tomographic bins, dropping to ~1 by 2 bins).
  **The superseded `bias_tomography.png`** (pre-correction Step 4, "rank-1 PC1
  99.7%" claim) is on disk and matches the WORKLOG-described pre-correction
  numbers exactly (verified visually: panel 3 shows PC1≈1.0/PC2≈3×10⁻³ on a
  log-y axis) — retained in the dossier ONLY as a methodological cautionary
  figure/footnote (if used at all), never as the paper's self-calibration result.

- **Step 5 — do analytic baryon models span BIND?** (Discussion-adjacent, also
  informs "why 2/richer BCM adds nothing"): the **1-param van Daalen/SP(k)
  relation spans only ~22–54%** of the BIND ensemble; multi-param **BCM/BCemu
  cover more (60–96%) but systematically miss the enhancement corner** — BIND
  produces intermediate-scale power *enhancement* up to S≈1.10–1.21 (cooling/
  condensation, gas-rich runs) where the analytic models top out at S≈1.01–1.05.
  Caveats: BCemu spanned along log10Mc only (lower bound on its coverage);
  **HMcode-AGN not included** (needs CAMB/Fortran toolchain); the enhancement is
  TNG-specific baryon physics. Provenance: WORKLOG:802-826 (Step 5).
  Figure: `model_comparison.png` (verified: BIND median/95% envelope crosses
  below both BCM and BCemu curves at ℓ≳2000 for z_s=0.5,1.0; BCM/BCemu upper
  bounds only reach S≈1.03–1.04, visibly below BIND's own upper envelope which
  approaches/exceeds 1.05–1.10 at ℓ~10³).

## Discussion (mining notes, not full prose)

- **Why 2 — link to Lin+2026 (Paper III).** WORKLOG 2026-06-24 "WL feedback
  latents" entry: built the WL analogue of Lin+2026 ("One latent to fit them all",
  arXiv:2509.01881) on 216 SB35 runs via a β-TCVAE (Chen+2019 TC decomposition):
  **Latent 0 = BH axis** (BHRadEff −0.43, QuasarThreshold, BH accr/Edd), acts
  ~uniformly across z_s; **Latent 1 = SN/wind axis** (VarWindVel −0.46,
  WindEnergy, IMFslope), evolves with z_s — mirrors their scale/time split. The 2D
  feedback manifold is **universal across WL summaries** (every statistic shares
  the first κ-Cℓ latent, canon corr ≈1.0; the second is shared by peaks/PDF
  (~0.7–0.8) but rotated for MFs/moments (0.2–0.4)). Orthogonal tSZ probes **don't
  expand the dimensionality** (κ×y/yy/κ×τ align 0.9–0.98 with κ's 2D) but **do
  sharpen/rotate axes**: IMFslope 0.33→0.56→0.64 across κ-only→+WL-stats→+tSZ;
  tSZ specifically unlocks the SN/thermal-energy axis (WindEnergy 0.13→0.33).
  This is the mechanistic "why 2" the Discussion should cite as independent
  corroboration (different method — VAE latent vs. SVD template — same
  dimensionality). Provenance: WORKLOG:590-623.
- **TNG-only amplitude vs. general dimensionality caveat** — mandatory per brief:
  the specific bias magnitudes (6σ, 0.67σ, etc.) are TNG-IllustrisTNG-feedback-model
  amplitudes; the *dimensionality* claim (≈2 templates suffice) is argued to be
  the robust/general part because it's independently reproduced by Lin+2026's
  latent analysis on different summary statistics. This is explicitly the
  brief's framing (see brief §Discussion and §Mandatory caveats) — no additional
  sourcing needed beyond flagging it prominently.
- **Scale cuts vs. templates** — no dedicated engine found comparing scale-cut
  mitigation directly against the template approach; this appears to be an
  intended Discussion *argument* (comparing the paper's approach to standard
  practice) rather than a mined result. Flag as `\todo{}` if the writer wants a
  quantitative comparison — none exists in the sources.

## Appendix A — towards varying cosmology (rescaling)

Sources: `docs/cosmo_rescaling_plan.md` (worktree `wt/cosmo-rescale`), WORKLOG
2026-07-03 entry, `examples/cosmo_rescale.py`, `examples/rescale_validation.py`,
`examples/rescale_sobol_scan.py` (docstrings + plan doc; not re-run).

- **Method.** Angulo & White (2010) [arXiv:0912.4277] rescaling of TNG300-Dark:
  an N-body output at z\* in the original cosmology, lengths × s (Mpc/h), masses
  × s³·Ωm'/Ωm, approximates a target-cosmology output at z'. `(s,z*)` chosen to
  minimize rms mismatch of linear σ(R) over Lagrangian radii 10^12.5–10^15.5
  Msun/h. Extends the 30-astro-param Sobol suite to the full 35-dim SB35 space
  (adds Ωm, σ8, Ωb, h, ns) *without new N-body runs*. Velocities never rescaled
  (BIND's kSZ observable is velocity-free tau). Simplification: rescale maps and
  halo catalogs only, never particles (D1, taken) — the projection commutes with
  relabeling.
- **Solver validation (`cosmo_rescale.py`, linear theory via pyccl+CAMB).**
  Identity target → s=1.0000, z\*=z', rms=0 exactly. Representative SB35 draws:
  **rms(σ) 0.03–2%** (matches brief exactly); z\*=0-pinned cases are the ~2% ones.
  Joint one-s-per-lightcone fit (8 epochs to z'=2.4) costs almost nothing vs.
  per-epoch fits. **Extreme corner Ωm=0.10 ∧ σ8=1.00**: s pins at the bound s=5.0,
  mass ×40, rms≈7% — the one region where rescaling degrades (matches brief's
  "open corner Ωm=0.1∧σ8=1.0" exactly; decision D2 left open in the plan doc).
- **Error scan (`rescale_sobol_scan.py`, 128-draw Sobol over the SB35 cosmology
  box).** Solver feasibility: s∈[0.68,5] at z'=0 ([0.44,5] at z'=1); 6/128 draws
  pin the s=5 bound (all Ωm≲0.13). σ(R) rms<3% for 86/128 draws at z'=0 and
  120/128 at z'=1. D(k) error decomposition (halofit-level, 3 bands: linear k<0.1,
  BAO 0.1–0.35, halo-band 0.35–5) — linear/BAO bands are deterministic and
  removable by map-level Fourier reweighting (promoted to v1 per decision D5); the
  halo band (concentration/formation-history mismatch) is the genuine residual.
- **Real-data validation (`rescale_validation.py`, TNG300-3-Dark 625³, same
  box/cosmology as production TNG300-Dark).** Rescaled to 3 SB35 targets, P(k)
  (CIC 512³) + M200m HMF vs. halofit/Tinker08 at target, unrescaled-vs-theory as
  control band:
  - **mild target** (Ωm=0.35,σ8=0.75,Ωb=0.045,h=0.70,ns=0.99), s=0.888, z\*=0.000:
    **max P(k) error k<1 = 3.4%** (matches brief's "≤3.4% validation" exactly),
    P(k) err k∈[0.3,1] ≤1.2%, HMF vs control 0.96/0.97 at 1e13/1e14 Msun.
  - **low-Ωm target** (Ωm=0.20,σ8=0.95), s=1.868, z\*=0.461: max P(k) err k<1=26%
    (BAO wiggle spike), ≤12% (±1% for k>0.32); HMF 0.92/0.92.
  - **corner-hi target** (Ωm=0.10,σ8=1.00,ns=1.16), s=5.0 (pinned), z\*=1.414:
    P(k) breakdown (>50% error), HMF 1.08/1.02 (0.51 at high mass) — the pinned
    bound's real-data consequence.
  - Error anatomy for the low-Ωm case: oscillatory ±8–14% band at k≈0.09–0.3 (BAO
    misalignment — settles to ±1% by k≈0.35) + smooth −7% (k=0.7) to −19% (k=1.5)
    deficit (halo-concentration/formation-history mismatch, the term BACCO
    corrects).
  - **Halofit-level D(k) predicts the measured particle-level error nearly
    point-by-point**: k=0.11: +25.5% predicted vs +26.3% measured; k=1.5: −19.1%
    vs −19.2% measured. So the theory scan is a trustworthy per-run quality flag
    with no simulation in the loop (matches brief's "halofit-level D(k) predicts
    the per-run error point-by-point (quality flag)" exactly).
- **Open items (decisions D2/D3 left open in the plan doc; not yet built:
  production pipeline, G1/G2/G2b/G3 validation gates).** This appendix is
  explicitly a *feasibility* result, not a delivered 35-dim suite — the plan doc
  states "pipeline work not started."
- Figure: `examples/rescale_validation.png` (main tree, verified — 3×2 grid, rows
  = mild/low-Ωm/corner-hi targets, columns = P(k)/halofit ratio and
  N(>M)/Tinker08 ratio, control (blue) vs rescaled (red) curves).

---

## FIGURES

| # | path | shows | serves section | provenance |
|---|---|---|---|---|
| 1 | `examples/figures_lightcone/transfer_function_ensemble.png` | 5-panel (one per z_s) spaghetti plot of S(ℓ) for 99 SB35 Sobol runs, colored by group f_gas, with median + 95% envelope | Results (1) transfer-function ensemble | `lightcone_transfer.py`; WORKLOG:897-916 |
| 2 | `examples/figures_lightcone/transfer_function_tomography.png` | 2-panel: median S(ℓ) per z_s bin with 16–84% band (left); run-to-run σ[S(ℓ)] vs ℓ per z_s (right) | Results (1) redshift dilution | `lightcone_transfer.py`; WORKLOG:897-916 |
| 3 | `examples/figures_lightcone/cosmo_bias.png` | 2-panel: (ΔΩm,ΔS8) bias cloud colored by f_gas with 1σ LSST-Y10 error cross (left); histogram of ΔS8/σ at 3 ℓmax cuts (right); 99 runs | Results (2) bias budget | `lightcone_cosmo_bias.py`; WORKLOG:873-896 |
| 4 | `examples/figures_lightcone/cosmo_bias_capstone.png` | 2-panel: residual \|ΔS8\|/σ vs N templates (median+95th, 3 ℓmax, log-y, N=2 vertical line) (left); σ_S8 inflation vs N (right); 216 runs | Results (3) N-template ladder — **headline figure of the paper** | `cosmo_bias_capstone.py`; WORKLOG:571-589; verified `cosmo_bias_capstone.npz` |
| 5 | `examples/figures_lightcone/beyond2pt_manifold.png` | 2-panel: cumulative feedback-manifold variance vs # templates for 10 WL/SZ statistics (left); 10×10 subspace-alignment matrix vs κ Cℓ (right); 237 runs | Results (5) manifold dimensionality / Discussion "why 2" | `lightcone_beyond2pt.py`; no WORKLOG entry (gap) |
| 6 | `examples/figures_lightcone/bias_sensitivity.png` | 2-panel: ranked Sobol total/first-order sensitivity indices of ΔS8 over 12 top astro params (left); scatter of ΔS8 vs top driver (IMFslope), colored by f_gas (right); 99 runs, ℓmax=3000 | Results (6) which knobs drive the bias | `lightcone_bias_sensitivity.py`; WORKLOG:848-871 |
| 7 | `examples/figures_lightcone/selfcal.png` | 2-panel: per-plane bias declining vs z_s (self-cal signal) vs flat pure-S8-shift line (left); σ(S8) degradation vs # tomographic bins, →1 by 2 bins (right) | Results (6) **corrected** self-calibration result | `lightcone_selfcal.py`; WORKLOG:779-800 (CORRECTION entry) |
| 8 | `examples/figures_lightcone/model_comparison.png` | 2-panel (z_s=0.5, 1.0): BIND S(ℓ) 95% envelope + median vs. BCM(Schneider15)/van Daalen 2019/BCemu analytic model spans, all CCL-projected | Results (6)/Discussion — do analytic models span BIND | `lightcone_model_comparison.py`; WORKLOG:802-826 |
| 9 | `examples/rescale_validation.png` | 3×2 grid: P(k)/halofit and N(>M)/Tinker08 ratios for mild/low-Ωm/corner-hi AW10-rescaled TNG300-3-Dark targets, control vs rescaled | Appendix A | `rescale_validation.py`; `docs/cosmo_rescaling_plan.md` §4; WORKLOG 07-03 |
| 10 | `examples/figures_lightcone/bias_tomography.png` | 3-panel: per-plane ΔS8(z_s) spaghetti (left); z-shape universality (mid); SVD rank bar chart, PC1≈99.7% (right); **pre-correction, superseded** | Methods/Discussion cautionary footnote ONLY — do not present as the self-cal result | `lightcone_bias_tomography.py`; WORKLOG:827-846, superseded by WORKLOG:779-800 |

**Not recommended / mismatched candidates found on disk** (brief named the parent
directory as a figure source; investigated and found NOT to serve this paper):
- `examples/figures_lightcone/{A_binned216,B_allmodes_pca,C_allmodes_nopca,
  D_rawcl,E_fulltomo}/cl_sbi_corner.png` — these are 30×30 posterior corner plots
  from a *different* analysis (`lightcone_cl_sbi.py`, WORKLOG 2026-06-24 "auto-Cl
  NPE on 216 runs"): P(30 astro params | fiducial κ Cℓ), NOT the (S8,Ωm,a1,a2)
  cosmology posterior this paper needs. Content confirmed by opening
  `B_allmodes_pca/cl_sbi_corner.png` directly (title: "P(theta | C_l^kk kappa
  auto, fiducial) — 216 SB35 runs, 3-PCA summary, 5-flow ensemble"). This result
  (info is ~3-dimensional, PC1=86.3%+PC2=12.6%=98.9%) is thematically related
  (corroborates the low-dimensionality story) but belongs to Paper III per
  `PLAN.md`'s branch mapping (`analysis/sobol-sb35`). Usable only as a
  Discussion-footnote cross-reference ("see Paper III, in prep."), never as this
  paper's own posterior-corner figure.

**Gap — no Y1/Y10 (S8,Ωm) posterior corner figure exists on disk.**
`lightcone_shear_forecast.py` (the actual engine for the brief's "posterior
corner (Y1/Y10)" figure request) saves only a raw emcee chain (`shear_forecast_y1.npz`,
6000×6 samples over {Ωm,σ8,h,ns,a1,a2}) with no plotting code — no corner/triangle
PNG was ever rendered. Per the hard rules (figure work = copying/extracting
existing renders only, no new analysis/plotting), this figure cannot be produced
by the miner or the drafter. **Flag to the human**: either accept a `\todo{corner
plot from shear_forecast_y1.npz}` in the draft, or have a human run the plotting
step out-of-band and drop the PNG into `figs_raw/`.

---

## CAVEATS

Brief's mandatory list (verbatim, sourced):
1. **Fixed cosmology (TNG fiducial)** — the entire 30-param Sobol suite fixes
   Ωm/σ8/etc. at TNG's own cosmology; hence Appendix A exists to address it.
2. **TNG-only amplitude** — the numeric bias/cost values are specific to
   IllustrisTNG's feedback implementation; the *dimensionality* claim (≈2
   templates) is the argued-robust part, corroborated by Lin+2026's independent
   method (different statistics, same suite) — see Discussion notes above.
3. **ℓ range / map resolution limits** — ℓ capped at 1.5×10⁴ in the transfer-
   function ensemble (`ELL_MAX_TRUST`, CIC aliasing above this dominates
   run-to-run scatter — WORKLOG references the "lightcone κ upturn = CIC
   aliasing" fix elsewhere in the project); capstone/cosmo_bias/beyond2pt Fisher
   analyses use ℓmax∈{2000,3000,5000} (well within the trusted band) but the
   shear_forecast full-likelihood uses ℓ up to 5000 on a 28-point log grid.
4. **Step-4 self-calibration CORRECTION** — use the corrected `lightcone_selfcal.py`
   result (Results §6), never the original rank-1 `bias_tomography.png` claim.

Additional caveats found in the sources (not in the brief's mandatory list but
material):
5. **Statistical-only/2-param significance** (WORKLOG Step 2): the σ-significance
   numbers (2.0/3.6/6.0σ etc.) marginalize only (Ωm,S8) — no h, ns, intrinsic
   alignment, or photo-z systematics — so they are optimistic; only the ΔS8
   *magnitudes* are argued to be robust.
6. **Shear-auto only, Gaussian/linear covariance, P_hyd=P_dmo·T² separability**
   (WORKLOG capstone entry, explicit caveat list) — the capstone Fisher result
   does not include cross-correlations with galaxy clustering, non-Gaussian
   covariance, or a non-separable baryon response.
7. **BCemu span caveat** (Step 5): BCemu's coverage was computed varying only its
   dominant log10Mc parameter (other params fixed at fiducial) — a lower bound on
   its true achievable coverage.
8. **HMcode-AGN not included** in the Step-5 analytic-model comparison (needs a
   CAMB/Fortran toolchain not installed at the time).
9. **Held-out validation (≤0.08σ) not independently re-verifiable from a saved
   artifact** — sourced from WORKLOG text only (see Methods gap note).
10. **No WORKLOG entry for the full-likelihood (Robertson-style) shear forecast**
    despite the brief's explicit instruction to search for "Robertson" /
    "beyond2pt" / "shear_forecast" in WORKLOG — numbers instead recovered by
    directly loading the on-disk `.npz` outputs (verified, see Results §4/§5).
11. **Suite size in flux across the campaign** (96→99→216→237→253 runs) — different
    figures/results in this dossier were computed at different points as the
    Sobol array completed; the canonical final count is 253/256 (see Methods).
12. **`lightcone_shear_sweep.py`'s full 237-run sweep + 34-d emulator chains were
    never run** (its own docstring defers them to `sbatch run_shear_sweep.sh`,
    which per the hard rules this mining pass must not execute) — only the local
    12-run validation subset's outputs exist on disk.
13. **Rescaling appendix is feasibility-only** — "pipeline work not started" per
    `docs/cosmo_rescaling_plan.md` line 10; validation gates G1/G2/G2b/G3 are
    listed as future work, not completed.

---

## CITATIONS

**From the brief's seed list, confirmed present/echoed in the sources:**
- Schneider et al. 2020 — the Fisher-bias formalism explicitly named and used
  (`lightcone_cosmo_bias.py`, `cosmo_bias_capstone.py` docstrings: "standard
  linearised Fisher bias (Schneider et al. 2020)").
- Angulo & White (2010) [arXiv:0912.4277] — explicitly named as the rescaling
  method (`docs/cosmo_rescaling_plan.md` §1, WORKLOG 07-03).
- Robertson et al. (2026) — explicitly named as the forecast structure being
  reproduced/extended (`lightcone_shear_forecast.py`, `lightcone_beyond2pt.py`
  docstrings; "Robertson Tbl 1"/"Tbl 2" cited for the LSST-Y1 survey spec and
  priors). **Gap:** exact title/journal/arXiv ID not resolvable from sources —
  only "Robertson et al. (2026)" appears; citer agent must search for the real
  reference.
- Lin+2026, "One latent to fit them all", arXiv:2509.01881 — explicitly named
  (WORKLOG 2026-06-24 "WL feedback latents" entry) with the exact title/arXiv ID
  already given.
- Schneider & Teyssier 2015 (BCM) — used via CCL's `BaryonsSchneider15` (3-param
  BCM, `lightcone_model_comparison.py`).
- van Daalen 2019 — used via CCL's `BaryonsvanDaalen19` (`lightcone_model_comparison.py`);
  a 2020 van Daalen reference is separately cited elsewhere in the project
  (bridge paper, WORKLOG:1073) — the writer should confirm which year's paper is
  the correct SP(k)/f_bar citation (2019 vs 2020 both appear in this project's
  WORKLOG; the CCL API function used here is explicitly `BaryonsvanDaalen19`).
- CCL (pyccl) — used throughout (`pyccl` 3.3.4, installed via
  `pip install --only-binary`, WORKLOG 06-22); the seed citation Chisari+2019 is
  the standard CCL reference but that author name does not itself appear in the
  sources — safe to cite as the standard CCL paper, citer agent should verify.
- emcee (Foreman-Mackey+2013) — used directly (`import emcee`,
  `emcee.EnsembleSampler`, `lightcone_shear_forecast.py`); standard citation, not
  named explicitly in-source but the package is unambiguous.

**Additional references that appear in the sources but were NOT in the brief's
seed list** (worth adding, per the brief's instruction that "engine comments
often name the observational papers precisely"):
- **BCemu** (2.0.5, WORKLOG 06-22) — the baryonification emulator used as a
  third analytic-model family in Step 5 (`import BCemu`, `BCemu.BCemu2025()`).
- **Constantine (2015)** — "linearised active subspace" method, explicitly named
  in `lightcone_beyond2pt.py`'s `manifold()` docstring.
- **Chen et al. 2019** (β-TCVAE / total-correlation decomposition) — explicitly
  named in WORKLOG:596 as the method behind `wl_feedback_vae.py` (the Lin+2026
  WL-analogue latent), relevant to the Discussion "why 2" section.
- **Mead & Peacock 2014a/b** [arXiv:1308.5183, 1408.1047] — cited in the
  rescaling plan doc for halo-catalogue/concentration extensions to AW10.
- **BACCO project** [arXiv:2004.06245] — cited in the rescaling plan doc as
  precedent for rescaling-based emulator training (1–3% accuracy).
- **Learning-the-Universe** [arXiv:2606.10024] — cited in the rescaling plan doc
  as a second precedent (rescaling for ML training-set augmentation).
- **Tinker et al. 2008** (halo mass function) — used as the target-cosmology HMF
  comparison in `rescale_validation.py` (implicit standard citation, "Tinker08"
  named directly in the plan doc's validation table).
- Public rescaling reference codes named in the plan doc (not citations but
  useful for a footnote): `github.com/amjsmith/rescale-cosmology`,
  `github.com/alexander-mead/particle-rescaling`.

**Not found in sources** (brief's seed citations that the mining pass could not
corroborate with in-repo evidence — still plausible/correct for the writer to
use, just unconfirmed by this pass): Eifler+2015, Huang+2019 (baryon PCA
approaches — these are Intro/background framing, not method citations the
engines would name), Amon & Efstathiou 2022, Preston+2023, DES Y3 (Amon+2022/
Secco+2022), KiDS-1000 (Asgari+2021), HSC Y3, LSST DESC SRD (2018).

---

## Gaps (for the orchestrator)

1. No WORKLOG entry for the beyond-2pt / full-likelihood shear-forecast work,
   despite the brief's explicit search instructions — numbers recovered instead
   by directly loading the saved `.npz` artifacts (high-confidence, exact
   verification, just a different provenance channel than instructed).
2. No rendered posterior-corner figure exists for the (S8,Ωm,a1,a2) LSST-Y1
   full-likelihood result — only the raw emcee chain `.npz`. The brief's
   "posterior corner (Y1/Y10)" figure slot cannot be filled without either (a)
   a `\todo{}` placeholder, or (b) a human generating the plot out-of-band (not
   permitted for this mining/drafting pipeline per the no-new-analysis rule).
3. Held-out (even/odd) N=2 validation number (≤0.08σ) has no saved artifact —
   sourced from WORKLOG text only, not independently re-verified like the other
   headline numbers in this dossier.
4. Robertson et al. 2026's exact title/venue is not resolvable from any source
   in this repo — only "Robertson et al. (2026)" + table numbers (Tbl 1, Tbl 2)
   appear. The citer agent must find the real reference via external search.
5. The brief's seed citations Eifler+2015, Huang+2019, Amon & Efstathiou 2022,
   Preston+2023, DES Y3, KiDS-1000, HSC Y3, LSST DESC SRD have zero corroborating
   evidence in the mined sources (expected — they are Intro background, not
   analysis-engine citations) — the writer/citer must source these independently
   and should not claim in-repo provenance for them.
6. "Scale cuts vs. templates" (Discussion outline bullet) has no dedicated mined
   result comparing the two approaches quantitatively — appears to be an
   intended discursive argument, not a result to mine.
7. `lightcone_shear_sweep.py`'s full 237-run sweep and the 34-d full-emulator
   nuisance arm were never run (deferred to `sbatch` in its own docstring) — only
   the local 12-run validation subset exists on disk (`shear_sweep.npz`,
   `shear_sweep_linear.npz`). If the paper wants the full-sweep "both-compare"
   result, it does not exist yet.
