# Spectrum-head target-engineering experiment

Controlled experiment quantifying three candidate target-representation changes for the six
ell-domain spectrum heads that currently hold out worse (10-21% median |frac err|) than the
WL-field heads (2-3%). Script: `spectrum_head_experiment.py` (same directory). Full run log:
`spectrum_head_experiment_run.log`; machine-readable results: `spectrum_head_results.json`.
Runtime: **46s** wall-clock on this node's single shared CPU core (well under the 30 min budget).

## Setup (recap)

- Data: `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.npz`, `z_s` index 1 (`z_s=1.0`)
  slice/diagonal for every tomographic head.
- Split: `rng = np.random.default_rng(0); test = np.sort(rng.choice(256, 50, replace=False))` →
  **206 train / 50 test**, reproduced exactly.
- Pipeline (identical across variants): standardize the 30 unit-cube params (train mean/std) →
  transform target (per variant) → per-bin center/scale (train stats) → PCA via SVD, 12
  components → one independent `sklearn.gaussian_process.GaussianProcessRegressor` per
  component → invert PCA → invert per-bin scale → invert transform → physical-unit prediction.

**Kernel deviation from the literal spec (documented, applied uniformly):** the brief's literal
30-dim ARD kernel (`RBF(length_scale=np.ones(30))`) was tested first and rejected for two
compounding reasons, verified directly rather than assumed:

1. **Speed.** One ARD component fit costs ~4-6s on this 1-core node (~19 head/variant
   pipelines × 12 components ≈ 15-20 min just for GP fits — tight against the 30 min budget).
2. **Correctness.** With `n_restarts_optimizer=0` (as specified) and the literal
   `length_scale=1` initial value, the single unrestarted optimization run **never moves**:
   confirmed by inspecting `gp.kernel_.theta` after fit (length scale identically `1` for all
   30 dims) and by reproducing sklearn's exact optimizer call manually
   (`scipy.optimize.minimize(..., method="L-BFGS-B")` from the same initial point converges to
   the same degenerate point, gradient ≈ 1e-5). In 30 standardized dimensions, pairwise squared
   distances are already ~2×30=60 at `length_scale=1`, so the RBF kernel is fully saturated
   ("no correlation anywhere") right at the initial point and the log-marginal-likelihood
   gradient w.r.t. length scale is flat there — a known pitfall of unrestarted ARD-GP
   optimization in higher dimensions. The fit collapses to a constant+white-noise split that
   ignores the parameters entirely (verified: `R2_pooled ≈ 0`, `R2_perbin ≈ 0` for every
   component fit this way).

**Fix used:** isotropic RBF (one length scale, not 30) initialized at
`length_scale = sqrt(30) ≈ 5.48` (the natural covariate scale for 30 independent unit-variance
dimensions) instead of the literal default `1.0`. Kernel *form*
(`ConstantKernel()*RBF(...) + WhiteKernel()`), `alpha=1e-8`, `normalize_y=True`, and
`n_restarts_optimizer=0` are exactly as specified — only the kernel dimensionality and initial
length scale changed, **uniformly across every head/variant**, so the (a)/(b)/(c)/(d)
comparisons below are apples-to-apples. Absolute numbers should not be expected to match the
production `gpytorch`-on-GPU backend (see the suppression anchor, §2) — this is a comparison
against itself across variants, not against the release.

**Two "response R²" conventions are reported** because they diverge substantially here:

- **R²_pooled** — the brief's literal formula, `1 - var(pred-truth)/var(truth-train_mean)`, a
  *single* variance computed over the concatenated (test-run × bin) array. Dominated by
  whichever bins have the largest absolute variance (typically low ℓ).
- **R²_perbin** — per-bin `r2_j = 1 - Σ(pred_j-truth_j)²/Σ(truth_j-train_mean_j)²` (over the
  run axis, bins with zero train variance excluded), aggregated by the **median** over bins.
  This is `papers/01_pipeline/_build_figures_nb.py`'s (fig 9) own "response R²" convention and
  is what the reference numbers in the brief (1.00/0.84/…) actually are — the one comparable to
  the production anchor.

## 1. Results table

| head | variant | n_bins | median \|frac err\| | R²_pooled | R²_perbin | cross yardstick |
|---|---|---:|---:|---:|---:|---:|
| cl_yy | (a) baseline | 724 | 12.16% | -0.003 | 0.791 | – |
| cl_yy | (b) masked | 207 | **6.55%** | **0.077** | 0.590 | – |
| cl_yy | (c) masked+scaled | 207 | 6.55% | 0.077 | 0.590 | – |
| cl_tt | (a) baseline | 724 | 14.32% | 0.642 | **0.780** | – |
| cl_tt | (b) masked | 207 | **7.50%** | **0.674** | 0.654 | – |
| cl_tt | (c) masked+scaled | 207 | 7.50% | 0.674 | 0.654 | – |
| cl_kappa | (a) baseline | 724 | 3.23% | 0.719 | **0.801** | – |
| cl_kappa | (b) masked | 207 | **1.98%** | 0.702 | 0.781 | – |
| cl_kappa | (c) masked+scaled | 207 | 1.98% | 0.702 | 0.781 | – |
| cl_kappa | (d) composed (S×C_dmo) | 724 | **3.39%** | 0.720 | 0.768 | – |
| cl_kappa_y | (a) baseline | 724 | 1.1×10⁷ % | 0.000 | 0.000 | 2.5×10⁵ |
| cl_kappa_y | (b) masked | 207 | 1.1×10⁷ % | 0.000 | 0.000 | 4.9×10⁵ |
| cl_kappa_y | (c) masked+scaled | 207 | **14.09%** | -0.013 | -0.016 | **0.662** |
| cl_kappa_tau | (a) baseline | 724 | 1.4×10⁷ % | 0.000 | 0.000 | 3.4×10⁵ |
| cl_kappa_tau | (b) masked | 207 | 1.1×10⁷ % | 0.000 | 0.000 | 3.1×10⁵ |
| cl_kappa_tau | (c) masked+scaled | 207 | **20.23%** | -0.014 | -0.018 | **0.706** |
| cl_yt | (a) baseline | 724 | 1.2×10⁷ % | 0.000 | 0.000 | 2.2×10⁵ |
| cl_yt | (b) masked | 207 | 1.0×10⁷ % | 0.000 | 0.000 | 3.4×10⁵ |
| cl_yt | (c) masked+scaled | 207 | **17.77%** | -0.014 | -0.016 | **0.715** |

(bold = the best/decisive number per head for the recommendation below; exact values in
`spectrum_head_results.json`.)

## 2. Suppression sanity anchor

Fit variant (a) (raw, z_s idx 1, full 724-bin grid — the production convention for
suppression) with the exact same hand-rolled pipeline:

| | median \|frac err\| | R² |
|---|---:|---:|
| hand-rolled (this experiment) | 3.52% | R²_pooled 0.769 / R²_perbin 0.768 |
| production (gpytorch, released) | 2.70% | 0.72 |

**Same ballpark** (1.3× worse frac err, R² actually a hair better) — not the "wildly different"
outcome the brief flags as a possible failure mode. This is the calibration check the rest of
the experiment leans on: the sklearn/isotropic pipeline is a reasonable, if not
production-grade, stand-in *for well-behaved, O(1), bounded targets*. Suppression is exactly
that (a ratio of two positive power spectra that structurally cancels the CIC-aliasing artifact
common to both — consistent with the prior repo finding that "science ratios cancel it"), which
is precisely the property the recommendations below try to extend to the other heads.

**Caveat on cl_kappa's own baseline number.** Variant (a) for cl_kappa here (3.23% / R²_perbin
0.80) looks *much better* than the brief's reference for the released cl_kappa head (21.2%,
R²=-0.02) — the opposite direction from what "our weaker pipeline" would predict. This is very
likely a **scope difference, not a pipeline difference**: this experiment fits only the
`z_s=1` diagonal `C_ℓ^κκ` slice (724 bins), whereas `bind.emulator`'s released cl_kappa head
(`STAT_SPECS["cl_kappa"]`, `src_axis=None`) fits the **full 5×5×724 = 18,100-dim tomographic
cube** (all auto + cross `z_s` pairs) as one PCA-compressed block — a much higher-dimensional,
harder-conditioned target. Treat the cl_kappa(a) number here as an upper bound on what a
single-slice fit could achieve, not as a refutation of the production number; it reinforces
recommendation 3 below rather than undercutting it.

## 3. cl_kappa ↔ suppression proportionality check

By construction, `suppression = cl_kappa_diag / cl_dmo`, so `cl_kappa_diag = suppression × cl_dmo`
should be an *exact* identity (both z_s idx 1), not merely a correlation:

```
ratio = cl_kappa_diag(train) / suppression(train)          # (206, 724)
max relative deviation across the 206 train runs, per bin: median = 2.6e-15, worst-bin = 5.9e-15
np.allclose(ratio, ratio.mean(axis=0), rtol=1e-2): True
```

Confirmed to floating-point precision — this is an identity, not an approximation.
`C_dmo = ratio.mean(axis=0)` is therefore exact and run-independent.

**Composed variant (d):** `cl_kappa_pred = suppression_pred(variant a) × C_dmo`.

| | median \|frac err\| | R²_pooled | R²_perbin |
|---|---:|---:|---:|
| suppression anchor (a) | 3.52% | 0.769 | 0.768 |
| cl_kappa composed (d) | 3.39% | 0.720 | **0.768** |

Matches almost exactly, as expected: multiplying by the positive, run-independent `C_dmo(ℓ)` is
a per-bin rescaling that leaves fractional error and **per-bin** R² invariant (frac err and
R²_perbin agree to within noise). R²_pooled differs slightly (0.720 vs 0.769) because the pooled
metric reweights bins by their now-different absolute variance — `C_dmo(ℓ)` is a steeply
declining function of ℓ, so it changes which bins dominate the pooled sum.

## 4. Key findings

**(i) Masking to ℓ ≤ 1.5×10⁴ is a clear, real win for the three positive spectra.** cl_yy:
12.16%→6.55% (frac err), R²_pooled -0.003→0.077. cl_tt: 14.32%→7.50%, R²_pooled 0.642→0.674.
cl_kappa: 3.23%→1.98%, R²_pooled ~flat. Variant (c)'s extra per-bin standardize is a **no-op**
for positive spectra relative to (b) — the numbers are bit-identical — because log10 + per-bin
standardize was already the baseline convention (`bind.emulator.transforms.StatCompressor`
always center/scales per feature); masking alone drives the entire improvement.

Counter-intuitively, **R²_perbin gets slightly *worse* after masking** (cl_yy: 0.791→0.590;
cl_tt: 0.780→0.654; cl_kappa: 0.801→0.781), even though frac err and R²_pooled improve. Traced
this down directly: splitting the full-grid cl_yy fit's per-bin R² by domain gives median 0.81
on the *untrusted* tail (ℓ>1.5e4) vs median 0.59 on the *trusted* range (ℓ≤1.5e4) — the aliased
tail is not "hard to fit," it can look artificially *easy* because its shape is dominated by a
deterministic CIC-geometry artifact shared by every run (so a GP nails it), while the real
astrophysical response lives in the trusted low-ℓ range, where genuine per-run cosmic-variance
scatter makes the regression intrinsically harder. Masking makes frac err and pooled R² honest
(not propped up by an artifact); the per-bin-median dip is the price of that honesty, not a
regression.

**(ii) For the signed crosses, the raw-target baseline is not just worse — it is numerically
broken under this recipe.** Variants (a)/(b) give median fractional errors of ~10⁷ %, and
R²_pooled/R²_perbin collapse to 0.000. Traced this directly (not a bug): even after masking to
the top-95%-by-|truth| bins, `|truth|` for e.g. cl_kappa_y still spans ~2 orders of magnitude
within the surviving population (median 1.5e-24, floor right at the 6.5e-26 threshold) because
raw, non-log-compressed cross-spectra have enormous dynamic range across ℓ. A single global
percentile threshold cannot protect a fractional-error metric from bins that are genuinely,
physically tiny — and the weak (single-restart, isotropic) GP optimizer, faced with that
dynamic range, further collapses toward a near-zero/no-response prediction (hence R²≈0 too).

**Variant (c)'s `asinh(x/s_bin)` transform (s_bin = 1.4826×MAD_bin over train) fixes this
outright**, not incrementally: frac err drops to a sane 14.1%/20.2%/17.8% (cl_kappa_y/
cl_kappa_tau/cl_yt) and the **cross yardstick** (median-over-bins of
`median_test|pred-truth| / per-bin-(P84-P16)/2-of-train`) lands at 0.66/0.71/0.72 — meaning the
typical residual is *smaller than* the physical Sobol-driven response range at that bin, the
regime where an emulator head is actually useful. R² stays mildly negative (-0.01 to -0.02),
i.e. this lightweight recipe still isn't confidently tracking the parameter response for the
crosses even once the metric is well-posed — masking+scaling makes evaluation possible and the
error small *relative to the response*, it doesn't by itself guarantee the GP has learned that
response; a better-optimized backend (the release's gpytorch/GPU GP, more restarts, or ARD once
affordable) is still needed to close R² fully.

**(iii) The masking alone does not fix the crosses** — variant (b) is just as broken as (a)
(~10⁷ %). The transform is the load-bearing change for signed heads; the ℓ-mask matters too
(it's applied together with the transform in (c) and both plausibly contribute to the smaller,
now-tractable dynamic range) but by itself is not sufficient.

## 5. Recommendation

Adopt, for the release `build_emulator.py` / `bind.emulator` (`dataset.py` `STAT_SPECS`,
`transforms.py`):

1. **Mask all six spectrum heads to ℓ ≤ 1.5×10⁴** (drop the CIC-aliased tail) before PCA/fit.
   Real, ~2× improvement in frac err for the positive spectra (cl_yy, cl_tt, cl_kappa) with flat
   or improved R²_pooled; necessary (if not sufficient alone) for the crosses. Low risk — this
   just removes bins already known to be nonphysical (per repo memory
   "Lightcone κ upturn = CIC aliasing").
2. **Replace the `raw` transform with `asinh(x/s_bin)` + per-bin standardize for the three
   signed cross heads** (cl_kappa_y, cl_kappa_tau, cl_yt), `s_bin` = 1.4826×MAD over train. This
   is the highest-leverage change found here: it turns an evaluation-breaking (and, per the
   R²≈0 collapse, plausibly fit-breaking) representation into a well-posed one with
   error-to-response ratios ~0.7. Expect the release's better-optimized backend to do
   meaningfully better than the 14-20% seen here in absolute terms, but the qualitative
   conclusion — raw signed cross-spectra are numerically fragile and asinh compression is the
   fix — should generalize; verify with an actual retrain before quoting a number.
3. **For cl_kappa specifically, prefer the composed form (`suppression × C_dmo`) over direct
   fitting**, exactly as `Emulator.predict` already does when a `suppression` head is present
   (`core.py` lines ~180-197) — this experiment's §3 confirms the identity holds to
   floating-point precision, so composing is *free accuracy*, not an approximation: it inherits
   suppression's already-good, well-calibrated behavior (matches the anchor almost exactly) and
   sidesteps the scope/conditioning problem of fitting the full 5×5×724-dim tomographic cube
   directly (§2 caveat) that plausibly explains the released head's poor 21.2%/-0.02. If the
   full tomographic cube (not just the z_s=1 diagonal) is needed downstream, consider composing
   the diagonal from suppression per z_s and only directly fitting the (smaller, still-signed)
   off-diagonal cross-`z_s` blocks — out of scope here but a natural next step.

**Caveats before quoting these numbers elsewhere:** (a) this pipeline uses a deliberately
weaker CPU-only sklearn GP (isotropic, single-restart) than the release's gpytorch/GPU backend
— see the kernel-choice discussion above — so absolute frac-err/R² values here should not be
pasted into the paper as-is; they establish *direction and rough magnitude* of each change.
(b) the suppression anchor (§2) confirms the pipeline is in the right ballpark for well-behaved,
bounded targets, which is reassuring for findings (i) and (iii) above but doesn't fully
validate the crosses' absolute 14-20% (the release backend should do better). (c) the cl_kappa
domain caveat (§2) means don't directly compare this experiment's cl_kappa(a) number to the
brief's 21.2% reference — they are different-dimensionality problems. The recommended next step
is a real `build_emulator.py` retrain with masking (1) + asinh crosses (2) + composed cl_kappa
(3) applied, to get release-quality numbers to decide against the paper's error bar.
