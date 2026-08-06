# The model, as a recipe — the math of the full story

Pedagogical companion to `ANALYTIC_LATENT_MODEL.md` (which holds conventions,
provenance, and all live-printed numbers). This file follows the narrative
order: gradients → families → kernels → **the recipe for S(ℓ) at any point in
parameter space**. Teaching figures: `figs_preview/tutorial_family_kernels.png`
(panels a–f) and `figs_preview/tutorial_span_vs_basis.png`.

## 0. The objects

Two simulation sets, same painted machinery:

- **Twobound** (the "1P-style" set): for each parameter $j = 1..30$, a pair of
  simulations at that parameter's prior bounds, $\theta^{(j,\rm lo)}$ and
  $\theta^{(j,\rm hi)}$, everything else fixed.
- **Sobol**: $N = 253$ simulations scattered over the full 30-dimensional
  prior (a space-filling cloud, not a grid).

From every simulation $q$ we measure:

- the WL suppression $S_q(\ell) = C_\ell^{\kappa\kappa}[q]/C_\ell^{\kappa\kappa}[\rm DMO]$
  (seed-paired ratio; band-averaged onto 24 log bands, $z_s = 1$), and
- **four halo-population numbers** ("latents"), medians over groups with
  $\log_{10} m^{\rm bg}_{\rm tot,500c} \in [13.3, 13.6)$ at $z \simeq 0.03$:

$$\lambda_q = \big(\tilde f_{\rm bar},\ \tilde f_\star,\ c_{\rm gas},\ \log\tilde T\big)_q$$

(budget, stellar partition, gas concentration, temperature — exact definitions
in `ANALYTIC_LATENT_MODEL.md` §0.)

## 1. Part I — parameter gradients and the five families (twobound)

The effect of parameter $j$ alone is the bound-to-bound difference — a
finite-difference gradient along that parameter axis:

$$g_j(\ell)\;=\;S\big(\theta^{(j,\rm hi)}\big)(\ell)-S\big(\theta^{(j,\rm lo)}\big)(\ell)\;\approx\;\Delta\theta_j\,\frac{\partial S(\ell)}{\partial\theta_j}$$

Normalize away the amplitude to isolate the **shape**:

$$\hat g_j(\ell) = \frac{g_j(\ell)}{g_j(\ell_{\rm peak})},\qquad \ell_{\rm peak} = \arg\max_\ell |g_j(\ell)|$$

Cluster the 30 shapes: distance $d_{jk} = 1 - \mathrm{corr}(\hat g_j, \hat g_k)$,
average-linkage hierarchical clustering, tree cut at $d = 0.15$ (i.e. members
correlate at $r \gtrsim 0.85$), after discarding parameters whose peak
signal-to-noise is $< 3$. Result: **four families + one singleton**
($\bar r = 0.95$–$0.98$ within families). At this stage we only *observe* the
grouping; Part III explains it.

## 2. Part II — latents and kernels (Sobol)

**The model.** Suppression depends on $\theta$ only through the four halo
numbers, linearly:

$$\boxed{\;S_q(\ell) \;=\; c_0(\ell) \;+\; \sum_{i=1}^{4} c_i(\ell)\,\lambda_{q,i}\;+\;\varepsilon_q(\ell)\;}$$

**The fit — one correction to the intuitive picture.** At each band $\ell_b$,
the five numbers $(c_0, c_1, c_2, c_3, c_4)(\ell_b)$ come from **one joint
least-squares fit** across the Sobol cloud,

$$\beta(\ell_b) = (A^\top A)^{-1} A^\top S(:,\ell_b),\qquad A = \big[\lambda_{q,1},\lambda_{q,2},\lambda_{q,3},\lambda_{q,4},1\big]_{q=1..N}$$

— *not* four separate one-variable line fits. The distinction matters because
the latents are correlated across the cloud (e.g.
$r(c_{\rm gas},\tilde f_{\rm bar}) = +0.94$): a one-variable slope would credit
$\tilde f_{\rm bar}$ with variance that belongs to $c_{\rm gas}$. Each $c_i$ is
therefore a **partial** slope — the response to latent $i$ with the other three
held fixed. Stringing the per-band solutions across $\ell$ gives the four
**kernel functions** $c_i(\ell)$ plus the intercept function $c_0(\ell)$.
(Tutorial panels b–c: kernel = slope of a scatter plot, repeated at every
$\ell$.)

## 3. THE RECIPE — S(ℓ) at any point in the 30-dimensional space

There are two routes, depending on what you hold in your hand. The kernels and
the intercept are the same fixed table in both
(`latent_model_coeffs.npz`, 5 curves × 24 bands × 5 source planes).

### Route A — you have the universe (a simulation, or observations)

1. Measure the four numbers from its group halos:
   $\lambda = (\tilde f_{\rm bar}, \tilde f_\star, c_{\rm gas}, \log\tilde T)$.
2. For each band, one dot product **including the intercept**:

$$\hat S(\ell) \;=\; c_0(\ell) + c_1(\ell)\,\tilde f_{\rm bar} + c_2(\ell)\,\tilde f_\star + c_3(\ell)\,c_{\rm gas} + c_4(\ell)\log\tilde T$$

3. Attach the per-band model error $\sigma_{\rm model}(\ell)$ (the CV residual,
   stored in the same table). Done. Accuracy: CV-$R^2 = 0.91$–$0.96$; the
   out-of-design fiducial reproduces at RMS 0.026 (0.015 without the thermal
   term). Implemented as
   `predict_from_latents.py --fbar --fstar --cgas --logt`.

**Worked example** (the fiducial, band $\ell \approx 4847$, $z_s=1$; kernels
$(c_1..c_4, c_0) = (+0.2022, +0.2769, +0.4518, -0.6789, +5.1129)$; latents
$(0.801, 0.097, 0.645, 6.833)$):

$$\hat S = 5.1129 + 0.2022(0.801) + 0.2769(0.097) + 0.4518(0.645) - 0.6789(6.833) = 0.954$$

versus 0.951 measured. Note $c_0$ is large and the $\log\tilde T$ term large
and negative — they cancel to order unity. **Dropping the intercept is the
most common way to get nonsense from this model.**

### Route B — you only have θ on paper (no simulation)

The map $\theta \to \lambda$ is *not* analytic — it is halo astrophysics. Two
options:

- **B1 (exact):** run the simulation/painting for $\theta$, measure $\lambda$,
  use Route A. This is what the simulations are for.
- **B2 (emulated):** predict $\hat\lambda(\theta)$ with the fitted 30→4 GP
  regression (fig 20h), then apply Route A's formula. Accuracy is capped by
  the GP leg (CV-$R^2 = 0.66$–$0.83$ per latent), and part of that cap is
  **irreducible**: each node is a single painted realization, so
  $\lambda(\theta)$ carries paint stochasticity and halo-sample scatter no
  regressor can recover.

The factorization is the point: $\theta \to \lambda$ is the noisy, nonlinear
leg; $\lambda \to S$ is the linear, tight leg. All 30-dimensional complexity
lives in the first arrow.

### Route Δ — differences (this is where the "amplitudes" belong)

For a *change* (a parameter gradient, a family), the intercept cancels and the
mixture picture applies:

$$\Delta S(\ell) \;=\; \sum_i c_i(\ell)\,\Delta\lambda_i$$

Here — and **only** here — the "amplitude factors" of the narration appear:
$\Delta\lambda_i$ is the parameter's **fingerprint**, i.e. how much turning
that knob moves each halo number, measured from the twobound halo catalogs
(not fitted). For absolute $S$ you must use Route A with the intercept; the
amplitude-mixture language applies to differences.

## 4. Part III — why the families (closing Part I's open question)

Chain rule through the model:

$$g_j(\ell) \;\approx\; \Delta\theta_j\frac{\partial S}{\partial\theta_j}(\ell) \;=\; \sum_i c_i(\ell)\,\underbrace{\Delta\lambda_i^{(j)}}_{\rm fingerprint}$$

Every parameter's gradient shape is a fixed mixture of the same four kernels,
weighted by its fingerprint. **A family is a set of parameters sharing a
fingerprint direction**; its common shape is that mixture. Validated with zero
fitted degrees of freedom (kernels from Sobol; fingerprints and shapes from
twobound): family means reproduce at $r = 0.98$–$1.00$ (singleton 0.86 — the
honest edge of the four-latent description). Figure:
`pfig_family_kernel_bridge`.

Two counting facts that resolve the natural follow-up confusions:

- **Families ≠ kernels.** No family fingerprint is axis-aligned (max
  $|\cos| = 0.77$): physics offers no knob that moves one halo property alone.
  The clustering finds *directions* (mixtures); only the halo measurements can
  de-mix them into the kernel basis.
- **Family count ≠ kernel count.** The number of families counts the distinct
  fingerprint directions among TNG's 30 knobs (a fact about the
  parametrization); any number of directions fits in a 4-D space. Indeed the
  shape curves alone are ~2-dimensional (SVD: 77.5% + 21.9% = 99.4% in two
  modes) — from shapes you could not even count four kernels; the third and
  fourth dials are revealed by the halo data and the other statistics (T–M,
  small scales).

## 5. Pitfalls checklist

1. Joint fit, not marginal slopes (latent collinearity, $r$ up to 0.94).
2. Absolute $S$ needs $c_0(\ell)$; mixtures-without-intercept is only for
   differences.
3. $\theta \to \lambda$ carries irreducible scatter; don't expect Route B2 to
   match Route A.
4. Everything is TNG-prior-conditional, $\ell \in [300, 3\times10^4]$,
   $z_s \in [0.5, 2.44]$ (per-plane kernel tables).
5. The latent definitions are exact conventions (bin, aperture, background,
   reducer) — see `ANALYTIC_LATENT_MODEL.md` §2.2–2.3 before reproducing.
