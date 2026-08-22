# Family-basis model: exact construction (2026-08-11)

Scripts: `family_basis_model.py` (clk exploration, verified), `family_basis_staged.py`
(3-stage arc), `family_basis_ship.py` (clk/pdf ship era, superseded),
**`family_basis_all.py` — THE shipped pipeline: all 7 WL statistics (S(ℓ), PDF, N_pk,
N_min, V0, V1, V2), GP-free.** Product: `figs_preview/family_model_bundle.npz` + the
numpy-only loader `family_model.py` (`FamilyModel().predict(lam, stat)`).

**The shipped fitting function is the λ ROUTE** (the GP θ→a route was dropped — the
linear map from four MEASURED group-bin halo latents beats it for every statistic):

    a_k(λ) = M_k · (λ − λ_ref) + a_ref,k
    λ = (f̃_bar, f̃⋆, c_gas, log T̃)   [§4 conventions; λ_ref ≈ (0.84, 0.09, 0.69, 6.83)]

5-fold-CV end-to-end R² (λ route | free-amplitude ceiling): clk 0.917|0.9997,
pdf 0.887|0.985, pk 0.770|0.857, mn 0.814|0.908, v0 0.912|0.9994, v1 0.928|0.986,
v2 0.941|0.985. Because λ is *measured*, the model evaluates for any simulation or
for observational gas/stellar-fraction constraints without CAMELS parameters. The map
is linear — λ far outside the Sobol design extrapolates unphysically. Count statistics
gate on PAIRED-realization whole-curve χ² (shared RT seeds); a span safeguard adds
twobound-backed-only families (flagged in the bundle) when the fig-05-backed subset
falls >0.05 below the full-family span.

## 1. Member response curves

For each of the 30 twobound parameters `p` with low/high bound runs `(lo, hi)`
(resolved from `twobound_params.npy`: rows 2i, 2i+1 differ in exactly one SB35 column):

**S(ℓ) statistic** — band-average the paired response ratio into the 24 latent-model
bands (edges `E` from `latent_model_coeffs.npz`, the same bands the kernels use), at
z_s = 1 (index ZI=1):

    S̄_r(b)  = mean_{ℓ ∈ [E_b, E_{b+1})}  clk_resp_r(ℓ)          b = 1..24
    ΔS_p(b) = S̄_hi(b) − S̄_lo(b)

**κ-PDF statistic** — the twobound `nongaussian_stats` PDFs are histograms of RAW κ,
and **each run's grid is different** (set from that run's own map σ; they drift by up
to ~1.5 bins between a bound pair — verifier finding 2026-08-11: index-aligned raw
differencing corrupts the 1–2%-level responses at up to the 100% level). The nu05
Sobol dataset PDFs are histograms of the per-realization standardized map
ν = (κ − ⟨κ⟩)/σ_κ (`nu_grid.py:119`), so each twobound PDF is mapped onto the
canonical 22-bin ν grid (Δν = 0.5, −2.75..7.75) using the moments implied by its own
raw histogram ON ITS OWN GRID (the raw grid spans ±9.6σ, so truncation is negligible):

    μ_r  = ∫ κ P_r(κ) dκ ,   σ_r² = ∫ (κ−μ_r)² P_r(κ) dκ     (this run's grid)
    P̃_r(ν) = σ_r · P_r(μ_r + σ_r ν)        (linear interp; 0 outside the raw grid)
    Δ P̃_p(ν) = P̃_hi(ν) − P̃_lo(ν)          (differenced on the COMMON ν grid)

The same per-run standardization now also feeds the S3b pdf clustering
(`paper_s3b_clusters.py`), whose ν = κ/σ_κ axis is thereby genuine.

## 2. Family basis vectors

Families come from the S3b clustering (`paper_s3b_clusters.py` machinery, verbatim:
S/N ≥ 3 gate, average linkage on 1−r of unit-peak curves, t = 0.15): clk → C1(12),
C2(7), C3(5), C4(5), C5(1 = SNII); pdf (ν-standardized, post grid fix) → C1(16),
C2(4), and four singletons C3 = VarWindSpecMom, C4 = WindFreeTravelDens,
C5 = ThermalWindFrac, C6 = SNII. Each singleton is its own family (averaging
mutually-uncorrelated singleton shapes would make a meaningless basis vector).

For family k with members m ∈ F_k, with d_m the member curve of §1:

    b*_m = argmax_b |d_m(b)|          (peak bin)
    p_m  = d_m(b*_m)                  (SIGNED peak value)
    ŝ_m  = d_m / p_m                  (unit-peak curve; peak = +1 → sign-aligned)

    B̃_k  = Σ_m |p_m| ŝ_m  /  Σ_m |p_m|     (amplitude-weighted mean:
                                             "how much the statistic cares")
    B_k   = B̃_k / B̃_k(argmax_b |B̃_k|)      (renormalized to unit peak)

Weighting robustness: S/N or flat weights change the shapes by r ≤ 0.003.

## 3. Amplitudes

Sobol test set (out-of-design for the basis — built from twobound only): y_r = the
run's statistic on the same grid (S(ℓ) band means / the nu05 PDF row), deviations

    D_r = y_r − ⟨y⟩          (mean over the 256 finite runs)

Free per-run amplitudes (2026-08-12: plain OLS everywhere — see the dated addendum
below for the ridge branch this superseded):

    â_r = argmin_a ‖D_r − Σ_k a_k B_k‖²   (minimum-norm solution via pinv)

Shipped models = ALL noise-backed families (≥1 member above that statistic's fig-05
permutation null, the un-crossed cells of `fig05_param_response`):

| stat | shipped basis | cond | amplitudes | span R² | model R² | e2e (λ→a→stat) |
|---|---|---|---|---|---|---|
| S(ℓ)  | {C1,C2,C3,C4} (C5 sub-noise) | 520 | OLS (was ridge pre-2026-08-12) | 0.9997 | 0.9997 | 0.9166 |
| κ-PDF | {C1,C2,C3,C4} (C5,C6 sub-noise) | 91 | OLS | 0.9852 | 0.9852 | 0.8866 |

(model R² = span R² for every statistic now that OLS reaches the free-amplitude
ceiling exactly; under the old ridge branch S(ℓ)'s model R² was 0.9984, 0.0013 below
its own span ceiling — the shrinkage cost. The `e2e (λ→a→stat)` column here is the
pre-extended-lambda, 4-latent number; see the 2026-08-12 EXTENDED-LAMBDA addendum for
the current 8-latent numbers, and the ESTIMATOR addendum further below for how e2e
changed, if at all, when ridge was removed.)

NB the compression table's per-k "adopted" subsets use a FALLBACK: within 5×10⁻⁴ of
the best R² at that k, prefer fully noise-backed subsets, else fall back to the best
regardless of backing — so sub-noise families can appear in small-k table rows even
though shipping excludes them. The shipped marker in the compression figures carries
its own subset label whenever it differs from the scree point.

## 4. Identification scatters (`pfig_family_model_amps[_pdf]`)

Halo latents λ = (f̃_bar, f̃_⋆, c_gas, log T̃) per Sobol run, measured from the atlas
cube (`atlas_cube_snap096.npz`, `predict_from_latents.py` convention): over halos with
13.3 ≤ log₁₀ M_500c < 13.6 (≥5 halos, else NaN),

    f̃_bar = median[(M_gas,500c + M_⋆,500c)/M_tot,500c] / (Ω_b/Ω_m)
    f̃_⋆   = median[M_⋆,500c/M_tot,500c] / (Ω_b/Ω_m)          (Ω_b/Ω_m = 0.0486/0.3089)
    c_gas = median[M_gas,500c/M_gas,200c]   (halos with M_gas,200c > 0)
    log T̃ = log₁₀ median[T_mw,500c]

Each panel: x = the latent with max |Pearson r| against that amplitude, y = â_k
(shipped amplitudes of §3), one point per run. Stamps: that Pearson r, plus the
5-fold-CV R² of an OLS fit of â_k on ALL FOUR latents + intercept (fold rng seed 1;
CV protocol identical for every number quoted).

End-to-end (λ → â(CV) → statistic): S(ℓ) total R² 0.9166; κ-PDF 0.8730.

## 5. Shipped-amplitude identifications

Marginal single-latent correlations degenerate (every amplitude's best marginal
correlate is stellar fraction — one dominant design axis + non-orthogonal basis), so
the per-amplitude PHYSICAL identity comes from `family_amp_physicals.py`: a ~75-scalar
candidate library (quantities × fixed FoF-mass bins + mass-trends + relation scatters
+ self-similar-scaled thermodynamics), scored |marginal r| + 0.3 |partial r| (partial
= controlling the other amplitudes) with a Hungarian uniqueness constraint at the
quantity-family level. Assigned distinct physicals:

S(ℓ), {C1,C2,C3,C4} (ridge amps — pre-2026-08-12; NOT recomputed under the current
OLS amplitudes, see the ESTIMATOR addendum below — treat these r/partial-r numbers as
provisional until refreshed):
  a_C1 = group stellar fraction f⋆[13.0–13.3] (+0.84)
  a_C2 = cluster Compton-Y at fixed mass, Y/M^{5/3}[>14.0] (−0.85)
  a_C3 = group baryon budget f_bar[13.0–13.3] (+0.73)
  a_C4 = baryon-vs-DM concentration excess c_gas/c_dm[13.3–13.6] (+0.64, partial
         +0.46; the relation is QUADRATIC — CV 0.40 linear → 0.59 quadratic)
  unique C1↔C2 separator: Y/M^{5/3} at fixed mass, partial +0.49 (C1) vs −0.51 (C2).

κ-PDF, {C1,C2,C3,C4} (OLS amps): a_C1 = σ_gas/σ_tot[13.6–14.0] (−0.76, partial
−0.53); a_C2 = c_dm[13.0–13.3] (−0.71); a_C3 = f⋆[13.3–13.6] (+0.74); a_C4 = mass-
trend of f⋆ (+0.64). Amplitude-cloud PCA: PC1 99.84% (the feedback-strength axis),
PC2 0.12% = a real THERMAL mode (logT[13.6–14.0], r=−0.62) — the pdf amplitudes are
one physical scalar plus a small thermal correction; interpret PCs, not individual
a_k, when quoting independent information content.

## Caveats

- The 5(4)-family bases are over-complete for a rank-2–3 response space; individual
  amplitudes at high cond (clk, 520) carry a large component along the basis's
  near-null direction and are therefore convention/estimator-dependent — compare them
  only through well-conditioned directions, or re-project with an explicit ridge for
  display (2026-08-12: the estimator itself is plain OLS everywhere now; see the
  ESTIMATOR addendum below for why this is safe for the *model*, not the raw a_k).
- The nu05 dataset carries 256 runs incl. 0114/0115/0117 (the 253-node canon excludes
  them); checked immaterial for S(ℓ) at <10⁻⁴.
- The twobound PDF grid is raw κ — the old κ/σ_κ x-label on `pfig_s3b_pdf_clusters`
  was wrong and is fixed as of 2026-08-11.

## Addendum (2026-08-12): EXTENDED-LAMBDA promotion — 8 measured latents

**Why.** A verified bake-off (`scratchpad/bakeoff_extended_lambda.py`, run
2026-08-12, reference spec for the definitions below) tested extending the shipped
4 group-bin latents λ = (f̃_bar, f̃⋆, c_gas, log T̃) with more measured halo numbers,
scored on a *separate* selection-CV split (`KFold(5, shuffle, random_state=123)`, so
the reported final numbers never see feature-selection leakage) then evaluated on the
final held-out split (`KFold(5, shuffle, random_state=0)`, same folds for every
statistic). Greedy forward selection targeting clk chose 4 additions, in order:
`logPe[grp]`, `f_gas200[grp]`, `f_gas200[cl]`, `logPe[cl]` — the winning 8-latent set
**beats the 4-latent map on ALL 7 statistics**, with only a +0.003 overfit gap
(in-sample OLS R² vs held-out CV R²).

**Promoted latent set** (order fixed; shipped 4 unchanged and come first — downstream
code, e.g. the tutorial's van-Daalen section, indexes `LAT[:, 0]` for f̃_bar):

    λ = (f̃_bar, f̃⋆, c_gas, log T̃,                              [group bin, unchanged]
         log P̃_e,grp, f̃_gas,200,grp,                            [NEW, group bin]
         f̃_gas,200,cl, log P̃_e,cl)                              [NEW, cluster bin]

Group bin: 13.3 ≤ log₁₀ M_500c,bg < 13.6 (unchanged). Cluster bin: 13.6 ≤
log₁₀ M_500c,bg < 14.5 — widened from the `family_amp_physicals.py` 13.6–14.0 bin
because per-run occupancy there is thinner (min 439 halos/run in 13.6–14.5 vs 725 for
the group bin; the narrower 13.6–14.0 bin risks <5-halo runs). All 4 new latents use
the SAME binning field as the shipped 4 (log₁₀ M_500c,bg — never M_200c) even where
the reduced quantity itself is a 200c-aperture ratio. Definitions (≥5 halos/bin,
else NaN; Ω_b/Ω_m = 0.0486/0.3089):

    log P̃_e,{grp,cl} = log₁₀ median[P_e,mw,500c]                 (P_e > 0 only)
    f̃_gas,200,{grp,cl} = median[M_gas,200c,bg / M_tot,200c,bg] / (Ω_b/Ω_m)

**Verified numbers** (pipeline-native e2e CV, `family_basis_all.py`'s own
`np.random.default_rng(1).permutation` + `array_split(5)` folds — NOT identical folds
to the bake-off's sklearn `KFold(5,shuffle,rs=0)`, so exact digits differ from the
bake-off printout but both are valid measurements of the same map):

| stat | old (4-latent) e2e CV R² | new (8-latent) e2e CV R² | Δ |
|---|---|---|---|
| clk | 0.9166 | 0.9585 | +0.042 |
| pdf | 0.8866 | 0.9330 | +0.046 |
| pk  | 0.7704 | 0.7793 | +0.009 |
| mn  | 0.8144 | 0.8232 | +0.009 |
| v0  | 0.9118 | 0.9541 | +0.042 |
| v1  | 0.9275 | 0.9425 | +0.015 |
| v2  | 0.9409 | 0.9565 | +0.016 |

No statistic regressed (the bake-off's finding held under the pipeline's own CV
protocol too). Fiducial clk trough closure (band 19, ℓ≈12651, measured 0.881, TNG300
hydro truth 0.876): old model predicted 0.920 (recovers ~2/3 of the 0.119
suppression); new model predicts **0.911** (recovers ~3/4), and the model's own
predictive σ across the ℓ>10⁴ trough region shrinks by ~30% (band 19, ℓ≈12651:
0.0283→0.0195 S-units; the ratio is consistently 0.68–0.74 across bands 15–23) —
cluster-bin numbers pin the small-scale response substantially better, though not
perfectly (the fiducial sits 1.5σ from the new prediction at the trough, vs 1.4σ
against the wider old band). At the van Daalen band (ℓ≈1533) σ_pred barely moves
(0.00468→0.00407, ~13% smaller) — the shipped 4 already pinned that scale.

**Where this lives.** `family_basis_all.py`'s LAT block (the λ-route section) computes
all 8 columns; `lam_fit_amps`/CV/`sig_pred` machinery is untouched (dimension-agnostic
lstsq on centered latents). `family_model.py`'s `FamilyModel` reads `lat_names`/
`lat_convention` from the bundle with no hardcoded dimensionality — `amplitudes()`/
`predict()`/`predictive_sigma()` all flow through however many latents the bundle
declares. The tutorial (`_build_family_tutorial_nb.py` → `family_model_tutorial.ipynb`)
`measure_latents` helper mirrors the bake-off's `_bin_med` reducer exactly (group AND
cluster bins, positivity guard + log10 for the pressure latents, plain median/(Ω_b/Ω_m)
for f̃_gas,200) and its sweep figure splits into two 4-row panels (shipped-4, then
extended-4) for readability. The prior 4-latent bundle is preserved, NOT overwritten,
at `figs_preview/family_model_bundle_4lat.npz` for comparison/rollback. The 4-kernel
SR latent model (`latent_model_coeffs.npz`, `sr_kernels*`) is a completely separate
model and was not touched by this promotion.

*(Note added 2026-08-12, ESTIMATOR addendum below: the clk numbers above — e2e CV
0.9585, trough prediction 0.911 — were measured under the ridge amplitude estimator
that this dataset shipped with at the time. After the ridge→OLS switch the pipeline-
native e2e CV is 0.9593 and the trough prediction is 0.908, both against the same
0.881 measured/0.019 predictive-σ; see below for why this is expected to be a wash.)*

## Addendum (2026-08-12): ESTIMATOR — ridge removed, plain OLS everywhere

**What changed.** `family_basis_all.py` previously switched between two amplitude
estimators depending on the shipped basis's conditioning: plain OLS (minimum-norm
pinv) when cond(B) ≤ 100, ridge (α = 10⁻² tr(BBᵀ)/K) when cond(B) > 100. Only clk
(cond 520) ever crossed that threshold; the other six statistics (pdf 91, pk 10,
mn 7.3, v0 49, v1 5.7, v2 1) were always OLS. Author decision: remove the switch
entirely — one estimator, plain OLS via pinv, for every statistic. `ridge_amps()` is
kept defined in `family_basis_all.py` (documented as unused) for reference / off-line
re-projection; `{stat}__used_ridge` is still written to the bundle (always `False`
now) for loader/tooling schema compatibility, though nothing currently reads it.

**Why it is safe — the filter-factor / invisible-direction argument.** Ridge
regression is exactly OLS regression with each direction of the basis's singular-value
decomposition rescaled by a *filter factor* fᵢ = σᵢ²/(σᵢ² + α), σᵢ the basis Gram's
singular values: fᵢ ≈ 1 for well-constrained (large-σᵢ) directions, fᵢ → 0 for
near-null ones. Removing ridge (α → 0) sets every fᵢ → 1, i.e. it stops damping
exactly the directions the data barely constrains — by construction those are the
directions where the reconstructed curve Σ a_k B_k is nearly insensitive to a_k (a
near-null direction of B *is* a linear combination of family shapes that ~cancels).
For clk's shipped 4-family basis {C1,C2,C3,C4} (cond 520), the near-null direction is

    u4 = (−0.76, +0.59, +0.27, +0.09)      (unit vector in the C1..C4 amplitude space)

whose curve image ‖u4ᵀB‖ is ≤ 0.005 in S(ℓ) units — below the model's own predictive σ
everywhere on the grid, i.e. observationally invisible. Un-shrinking that direction
inflates individual amplitudes (max |a_k| grows from ≈0.25 under ridge to ≈1.9 under
OLS, almost entirely along u4) without moving the reconstructed curve or the model's
accuracy, because the curve is what every downstream consumer (`family_model.py`,
the tutorial, the paper figures) actually evaluates — never the raw a_k in isolation.

**Measured confirmation** (pipeline-native e2e CV, `rng(1)`-seeded 5-fold, 8-latent
extended-lambda; `family_basis_all.py` rebuilt end-to-end):

| quantity | ridge (old) | OLS (new) |
|---|---|---|
| clk e2e CV R² | 0.9585 | 0.9593 |
| clk trough prediction (band 19, ℓ≈12651) | 0.9112 | 0.908 |
| clk trough predictive σ | ~0.019 | ~0.019 |
| clk free-amplitude ceiling (model R² = span R²) | 0.9997 (0.0013 below span) | 0.9997 (exact) |

Both trough predictions sit within ~1.5σ of the measured 0.881 — the switch is e2e
equivalent, not a regression, exactly as the filter-factor argument predicts. The
other six statistics were never on the ridge branch, so their bundle arrays (basis,
mean, amplitudes, `lat_M`, `sig_pred`, `r2`) are **bit-identical** before and after
this change — verified by direct `np.array_equal` against the pre-switch bundle
(backed up at `figs_preview/family_model_bundle_8lat_ridge.npz`).

**Amplitude-cloud check.** Before this change, `figs_preview/amplitude_sets.npz`
plotted the Sobol amplitude cloud (`clk__a_sobol`) with ridge amplitudes but the
fiducial star (`clk__a_fid`) with a plain-lstsq fit — a convention mismatch that
placed the star at Mahalanobis distance 427 from the cloud despite a <2×10⁻³ curve-
level agreement (a measurement artifact of comparing two different estimators, not a
physical anomaly). With both now OLS (same estimator, same basis), the fiducial star
sits at Mahalanobis distance ≈0.98 from the 256-run OLS cloud (first two amplitudes
at −0.45σ, +0.25σ marginally) — solidly inside the cloud, not an outlier, and its
offset projected along u4 is only ≈0.37 of the cloud's own (u4-inflated) spread. The
lesson stands and is now enforced by construction: amplitude plots must always use
one consistent estimator for cloud and star together (`figs_preview/amplitude_sets_clk.png`,
regenerated 2026-08-12, states this in its caption).

**Amplitude-interpretation caveat.** The §5 physical identifications for S(ℓ)
(`a_C1..a_C4 = ...`) were derived against the *ridge* amplitudes and have not been
recomputed against the new OLS amplitudes; because the ridge→OLS change moves
individual a_k almost entirely along the curve-invisible u4 direction, the marginal/
partial-r identifications built from *combinations* of a_k that are NOT aligned with
u4 should be robust, but this has not been explicitly re-verified. Treat §5's S(ℓ)
numbers as provisional pending a `family_amp_physicals.py` re-run on the OLS bundle.
Bottom line for anyone consuming individual clk amplitudes going forward: compare them
only through directions of the {C1,C2,C3,C4} basis that are well-conditioned (i.e. not
aligned with u4), or explicitly re-project with `ridge_amps()` (kept in
`family_basis_all.py` for exactly this off-line display use) if a single-family
reading is wanted.

## Addendum (2026-08-12): OBSERVABLE AGNOSTIC-LAMBDA promotion — the shipped latent set

**Author rulings.** (i) The latent set must come from a fully agnostic search —
no a-priori anchor latents, no fixed two-bin restriction, no cap on the latent
count. (ii) Only quantities measurable or inferable from actual observations
are admissible (the dark-matter concentration `c_dm` and the ratio
`c_gas/c_dm`, which require the DM-only profile, are excluded).

**Search** (`agnostic_lambda_search.py`, run with `OBS_ONLY=1`; log
`audits/agnostic_lambda_search_obs_2026-08-12b.log`; results
`figs_preview/agnostic_lambda_results_obs.npz`): a 90-candidate library
(10 median quantities + 2 halo-to-halo 16–84% widths × 7 mass bins
13.0–13.2–13.4–13.6–13.8–14.0–14.3–15.0 in log10 M_500c,bg, every bin ≥27
halos in every run, + 6 cross-bin mass-trend slopes + 10 run-shuffled DECOY
columns as a selection-noise tripwire), searched by sequential floating
forward selection from the EMPTY set, scored on the POOLED mean of the 7
per-stat end-to-end curve-space CV R² (selection folds sklearn
KFold(5,shuffle,rs=123) ≠ the rng(1) reporting folds), acceptance threshold
2e-3 ≈ 3× the score's fold-reseeding scatter (6e-4 over 20 reseeds), no cap.
The fiducial is never touched by selection.

**Result.** The search stops at k=8 on its own (first reject +1.3e-3); no
decoy is ever selected. First pick = f_bar[13.0–13.2] (pooled 0.72 alone;
first in 20/20 bootstrap re-selections) — the van Daalen variable — later
EVICTED by the floating step once logY+logT[13.0–13.2] enter (Y ≈ gas
mass × T implies the budget). Final set, in selection order (= LAT columns):

    f_star[13.2-13.4], logT[13.0-13.2], logY[13.0-13.2], logPe[14.0-14.3],
    c_gas[14.0-14.3], logY_ss[13.4-13.6], c_gas[13.2-13.4], logPe[13.4-13.6]

Pipeline-native e2e CV (rng(1) folds; `family_basis_all.py` rebuilt, exact
match to the search's reporting-fold numbers): clk 0.9632, pdf 0.9386,
pk 0.7872, mn 0.8379, v0 0.9599, v1 0.9586, v2 0.9643 (pooled 0.9157 vs
0.9068 for the superseded two-bin 8). Fiducial closure (never used in
selection): clk median |dev| 1.09% vs the measured BIND fid (1.53% vs
full-hydro truth, max 2.34%); trough (band 19) pred 0.895 vs measured 0.881
= 0.7σ of σ_pred 0.019, 12% of the effect (two-bin 8: 0.908/1.4σ;
hand-picked 4: 0.920). σ_pred: 0.019 at the trough, 0.003 at the vD band.

**Dominance/robustness checks** (all on reporting folds): adding f_bar back
to the final 8 gains |Δ|<1e-3; replacing (Y,T)→f_bar costs 0.016
(r(f_bar,logT)=0.21 — the thermal axis is nearly orthogonal to the budget);
swapping logY for its r=0.98 neighbor f_gas200[13.0–13.2] costs 0.006 (10×
fold noise — block representatives are measurable preferences, not pure
convention). Relaxing observability (the unrestricted 111-candidate search)
follows the identical path and swaps the two c_gas cells for c_gas/c_dm in
nearly the same bins at no accuracy change (pooled 0.9154) — no information
in the halo response requires an unobservable quantity.

**What shipped.** `family_basis_all.py` LAT block = the 8 observable latents
(reducer mirrors the search script bit-exactly; lat_ref matches the search
npz to 1e-6); bundle `figs_preview/family_model_bundle.npz` (prior bundles
preserved: `family_model_bundle_2bin8.npz`, `family_model_bundle_4lat.npz`);
`family_model.py` docstring updated (loader itself unchanged — latent-count
agnostic). Tutorial rebuilt: `measure_latents` = the 8 observable latents
(now takes Y_500 and no m_tot_200), and the van-Daalen section measures
f_bar as an AUXILIARY variable — the conditional latent sweep |f_bar rides
the measured S–f_bar cloud at RMS 0.0038 (vD scale ℓ≈2726, r=+0.97) with the
budget NOT among the model inputs. Basis/families/amplitudes machinery
untouched (§§1–3 above still exact); the §5 amplitude physical
identifications remain the provisional ridge-era ones. Paper-section figures:
`family_model_section_figs.py` → `figs_v2/pfig_fm_{basis,freeamp,`
`latent_extraction,model_curves,fiducial}`.

**Post-promotion library cleanup (same day).** `c_gas_raw` (the raw-aperture
convention duplicate of `c_gas`, r>=0.99 in every bin) was dropped from the
candidate library as presentationally confusing; the search path, chosen set,
and every headline number are provably unchanged (a never-selected candidate
cannot alter a greedy path); only bootstrap-frequency bookkeeping shifted.
The library is 90 candidates (was 97). The section figure's stability grid is
now BLOCK-CREDITED: a bootstrap credits a cell if it selected the cell or any
|r|>0.95 near-duplicate (per-boot selections stored as `boot_sets` in the
results npz), so the chosen boxes sit on the hottest cells.

**Floating-vs-plain greedy (2026-08-12, author question "why do evictions lower
R²?").** The floating criterion is best-of-size, not no-cost: an eviction is
accepted when the reduced set beats the best same-size set yet seen, so the
score may dip at the move (f_bar eviction: 0.9038→0.9028; the logPe out-and-
back dips 0.026 and recovers — a noise-level firing, net zero). Measured
justification: plain greedy WITHOUT floating retains f_bar, stops at k=7
(next gain +0.0019 < 2e-3), and finishes at reporting-fold pooled 0.9128 vs
0.9157 for the floating search (fiducial trough 0.8917 vs 0.8950, both within
sigma_pred of the measured 0.8812) — the eviction's −0.0010 buys +0.0029 at
the endpoint by freeing a redundant slot for the final picks. The section
figure pfig_fm_search_path is now the single-panel trajectory (per-step
competition panel removed by author preference).

## Addendum (2026-08-12): twobound MFs remeasured on the full canonical grid

**Author ruling.** The MF cap at nu<=3.75 (the old nongaussian_stats
-3..4 grid) was unacceptable against the paper's other figures (nu to ~8);
the kappa maps are on disk (kappa_maps.npz, (50,5,1024,1024) per run), so the
MFs were remeasured — `remeasure_twobound_nu05.py`, which runs the EXACT
Sobol-reduction engine (`nu_grid.compute` -> bind.inference.stats) over all
60 twobound runs + the BIND fiducial, writing `<run>/nu05_stats.npz` (all six
nu statistics on the canonical 22-bin grid; originals untouched). Validation
on run_0000: V0 reproduces the old grid EXACTLY at all 14 shared thresholds
(same 1' smoothing, per-realization standardization, estimator); V1/V2 differ
only by the delta-binning width (dnu 0.25 -> 0.5; <=1.4%/3.5% of peak) — and
since the Sobol side always used dnu=0.5, the remeasured values REMOVE a
small convention mismatch the old interpolation carried. Run serially in the
session (1-CPU/16GB Slurm allocation; a 12-worker attempt was cgroup-OOMed).
Logs: audits/remeasure_twobound_nu05_2026-08-12.log.

**Pipeline rebuild** (family_basis_all.py: NMF 14->22, `mf_canon` reads
nu05_stats natively, grid assert vs the dataset nu grid; prior bundle at
figs_preview/family_model_bundle_mf14.npz, prior amplitude sets at
figs_preview/amplitude_sets_mf14.npz, refreshed by
refresh_amplitude_sets_mf22.py):
- clk/pdf/pk/mn: bundle arrays BIT-IDENTICAL (verified np.array_equal).
- v0: families 3/5, span 0.9993, e2e CV 0.9597 (was 3/4, 0.9994, 0.9599).
- v1: families 1/2, span 0.9704, e2e CV 0.9458 (was 2/3, 0.9864, 0.9586) —
  the 22-bin re-clustering leaves one noise-backed family; the tail variance
  the dropped family absorbed lowers the ceiling; span safeguard (0.05
  margin) not triggered.
- v2: families 1/2, span 0.9849, e2e CV 0.9638 (was 0.9854, 0.9643).

**Search stability.** The OBS agnostic search re-run with the 22-bin MFs in
the pooled objective is IDENTICAL: same 8 latents, same path incl. both
float events, same first-reject (+0.0013), decoys clean. Pooled reporting
score 0.9137 (was 0.9157; the delta is v1's ceiling, not the latents).
Fiducial trough unchanged (0.895 vs measured 0.881). Re-derived comparisons:
no-float greedy 0.9110 vs floating 0.9137; unrestricted-library search
0.9135 vs observable 0.9137. Fold-reseed scatter unchanged (6.1e-4).
pfig_fm_model_curves' MF panels now run the full nu range (the nu<=3.75
caption clause is retired).
