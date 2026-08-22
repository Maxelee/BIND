# R5 — does $\lambda$ absorb generative noise that $\theta$ cannot?

**Referee comment 5.** *"A subtle confound in the parameters-vs-latents
comparison: lambda is measured from the same generated maps whose statistics
it predicts, so latents can absorb realization-level generative noise that
theta cannot, inflating the latent advantage. Shared seeds across nodes make
most of this common-mode, but not the node-specific draws. The clean rebuttal
experiment: re-measure lambda from an independent seed set and show CV R^2 is
stable... you want the answer in the paper, because 0.96-vs-0.6 is your
flagship number."*

**Answer.** The confound is real and now measured. For $S(\ell)$ it is worth
$+2.8\times10^{-4}$ in CV $R^2$ — **0.07% of the 0.381 gap** — with a hard
ceiling of $1.0\times10^{-3}$ (0.26%). The same noise simultaneously *costs*
$3.1\times10^{-3}$ of CV $R^2$ through ordinary regression dilution, so a
noise-free $\lambda$ would make the latent advantage **larger**, not smaller.

**One premise of the comment is wrong and must not be repeated in the reply.**
There are no shared seeds. BIND's sampler is unseeded, every node was painted
in a separate process, and the draws are fully independent across the 256
nodes. There is therefore *no* common-mode cancellation — the numbers below
carry the whole argument, and they are measured under exactly the
worst-case (fully independent-draw) condition the referee worried about.

---

## 0. Provenance

Everything here was produced from the shipped artefacts and campaign caches,
read-only. Scripts: `papers/01_pipeline/referee/work/r5_{common,repro,
bootstrap,replicas,inflation,truth,figs}.py`. Numeric products:
`/mnt/home/mlee1/ceph/referee_work/r5/r5_{repro,bootstrap,replicas,inflation,truth}.npz`.
Figures: `papers/01_pipeline/referee/figs/r5_fig{1,2,3}_*.png`.

Cross-references: the "no seeds / 30% per-patch spread" finding is shared with
`R3b_texture_results.md` (I re-derived it independently from `src/bind/model.py`
and from the twobound trio); the fiducial-paint cosmology-conditioning caveat
is R3b §5.4.

---

## 1. Baseline reproduction (task 1) — PASSED

`r5_repro.py` recomputes the eight latents from the per-halo atlas cube with
the shipped reducer (`family_basis_all.py` L408–457, re-implemented verbatim
in `r5_common.measure_lambda`), then runs the shipped end-to-end CV
(`lam_fit_amps`, `rng(1)` folds, shipped basis and amplitudes).

| statistic | recomputed | shipped | $\|\Delta\|$ |
|---|---|---|---|
| $S(\ell)$ | 0.9632 | 0.9632 | 2.9e-5 |
| PDF | 0.9386 | 0.9386 | 1.2e-5 |
| peaks | 0.7872 | 0.7872 | 9.5e-6 |
| minima | 0.8379 | 0.8379 | 1.2e-5 |
| $V_0$ | 0.9599 | 0.9599 | 7.4e-6 |
| $V_1$ | 0.9586 | 0.9586 | 3.8e-5 |
| $V_2$ | 0.9643 | 0.9643 | 4.4e-5 |
| **pooled** | **0.9157** | **0.9157** | — |

The full-sample $\lambda\to a$ map also reproduces the bundle's `lat_M` to
1.3e-8 relative. (The shipped script never casts the `float32` atlas cube, so
its latents carry `float32` round-off; the `float64` path used below differs
by $\le1.3\times10^{-6}$ in `lat_ref` and $\le1.6\times10^{-5}$ in every CV
$R^2$.) **The real code path is being exercised.**

**The $\theta$ baseline, same folds, same basis, same amplitudes** — only the
design matrix changes to the canonical `X_unit` (30 astro parameters on the
SB35 unit cube):

| route | $S(\ell)$ | pooled |
|---|---|---|
| $\lambda$, linear | **0.9632** | 0.9157 |
| $\theta$, linear (unit cube) | **0.5822** | 0.4904 |
| $\theta$, linear (raw physical units) | 0.4583 | 0.3859 |
| $\theta$, gradient-boosted trees | 0.1226 | 0.4037 |
| $\theta$, GP (RBF + white) | 0.5443 | 0.4967 |

So the paper's "$\simeq0.6$" is the unit-cube linear route at 0.582, and no
flexible regressor beats it — **the flagship gap for $S(\ell)$ is 0.381**.
All fractions below are quoted against that.

---

## 2. The $\lambda$ noise floor (task 2)

Halos per latent mass bin, over the 256 nodes (min / median / max):
`13.0–13.2` 858/938/1009 · `13.2–13.4` 514/607/685 ·
`13.4–13.6` 336/375/429 · `14.0–14.3` 82/92/98. Each latent is a median over
that many halos, which is what crushes per-halo draw noise.

Three independent estimates of $\sigma(\lambda_i)$, expressed as the
noise-to-signal **variance** ratio $q_i=\sigma_{\rm noise}^2/\sigma_{\rm design}^2$
(fig. `r5_fig1_noise_floor.png`):

| latent | $\sigma_{\rm design}$ | $q$ bootstrap (indep.) | $q$ bootstrap (node-spec.) | $q$ 3 paints (measured) |
|---|---|---|---|---|
| $\tilde f_\star$[13.2–13.4] | 0.0842 | 0.0001 | 0.0001 | 0.0005 |
| $\log\tilde T$[13.0–13.2] | 0.0278 | 0.0141 | 0.0085 | 0.0060 |
| $\log\tilde Y$[13.0–13.2] | 0.1356 | 0.0022 | 0.0013 | 0.0034 |
| $\log\tilde P_e$[14.0–14.3] | 0.1888 | 0.0352 | 0.0174 | 0.0104 |
| $c_{\rm gas}$[14.0–14.3] | 0.0225 | 0.0530 | 0.0225 | **0.0161** |
| $\log\tilde Y_{\rm ss}$[13.4–13.6] | 0.0725 | 0.0045 | 0.0028 | 0.0005 |
| $c_{\rm gas}$[13.2–13.4] | 0.0556 | 0.0019 | 0.0011 | 0.0005 |
| $\log\tilde P_e$[13.4–13.6] | 0.2636 | 0.0024 | 0.0017 | 0.0007 |
| **mean** | | **0.0142** | **0.0069** | **0.0047** |

* **Bootstrap (independent), `NB=1000`.** Resample halos with replacement
  within each latent's mass bin, re-take the median. Upper bound: it contains
  the halo *population*-sampling variance as well as the draw noise. Because
  all 256 nodes paint the **same** 2933 halos, that population term is
  common-mode across the design and cannot be absorbed by a regression.
* **Bootstrap (node-specific), `NBC=400`.** One resampled multiset applied to
  all 256 nodes; the residual $\lambda_b(q)-\langle\lambda_b\rangle_q$ removes
  the shared part. Halves the mean $q$.
* **Three independent paints (the measurement).** See §3.

**Worst latent: $c_{\rm gas}$ in the `14.0–14.3` cluster bin** — only ~92
halos, $q\simeq0.016$–0.053. Everything else is at or below the 1% level.

The bootstrap is conservative for six of the eight latents
($\sigma_{\rm rep}/\sigma_{\rm boot} = 0.33$–0.65) but **understates**
$\tilde f_\star$ by 3.0× and $\log\tilde Y$ by 1.2×: for those the generative
noise is partly coherent across halos, which an i.i.d. halo bootstrap cannot
see. **Report the replica numbers as primary; the bootstrap as a
cross-check.** This is precisely why leg 1 matters.

**Independent scale check** — per-halo BIND/TRUTH ratio at the fiducial
(same 2933 halos, hydro patches vs generated patches), median and 16–84
half-width: $f_\star$ 0.959 ± 0.245 · $T$ 0.987 ± 0.158 · $Y$ 1.054 ± 0.269 ·
$P_e$ 1.097 ± 0.543 · $c_{\rm gas}$ 1.007 ± 0.058 · $Y_{\rm ss}$ 1.039 ± 0.243.
Per-halo errors of 16–54% collapse to sub-1%-of-design medians.

---

## 3. The direct measurement: three independent paints of the same simulation

**Twobound runs 18, 49 and 53 have bit-identical 35-parameter vectors**
(verified: `np.array_equal` on all three pairs of `twobound_params.npy` rows;
three of the 30 scanned parameters have a bound that coincides with the
fiducial value). They share the halo catalogue and the ray-tracing seed
(`RT_SEED=1992`, so cosmic variance cancels exactly), and the sampler is
unseeded — **so they are the same simulation painted three independent
times.** Their per-halo atlas rows differ (median 12.6% in $Y_{500c}$ between
two of them, vs 26.8% between two *different*-parameter runs), confirming
genuinely independent draws.

That is the referee's experiment, already executed at one parameter point, at
zero GPU cost. Decomposing the CV score as
$R^2 = R^2_{\rm honest} - f_{\rm noise} - f_{\rm hat} + 2c$ (definitions in
`R5_leg1_runbook.md` §1):

| statistic | gap ($\lambda-\theta$) | $f_{\rm noise}$ | $f_{\rm hat}$ | **$2c$ (absorption)** | $2c$/gap | ceiling at corr=1 | ceiling/gap | net noise effect |
|---|---|---|---|---|---|---|---|---|
| **$S(\ell)$** | **0.381** | 7.7e-5 | 3.3e-3 | **+2.8e-4** | **+0.074%** | 1.0e-3 | 0.26% | **−3.1e-3** |
| PDF | 0.371 | 2.0e-5 | 6.2e-3 | −2.5e-4 | −0.068% | 7.1e-4 | 0.19% | −6.5e-3 |
| peaks | 0.401 | 9.0e-2 | 4.1e-3 | +2.6e-3 | +0.64% | 3.8e-2 | 9.6% | −9.1e-2 |
| minima | 0.428 | 6.6e-2 | 4.7e-3 | +6.7e-3 | +1.57% | 3.5e-2 | 8.2% | −6.4e-2 |
| $V_0$ | 0.447 | 1.8e-3 | 1.6e-3 | −2.7e-3 | −0.60% | 3.4e-3 | 0.75% | −6.1e-3 |
| $V_1$ | 0.476 | 6.9e-4 | 4.7e-3 | +7.9e-4 | +0.17% | 3.6e-3 | 0.75% | −4.6e-3 |
| $V_2$ | 0.473 | 1.5e-3 | 2.3e-3 | −8.2e-4 | −0.17% | 3.7e-3 | 0.78% | −4.6e-3 |

Reading it:

* $f_{\rm noise}$ is the fraction of the statistic's design variance that is
  node-specific realization noise. For $S(\ell)$ the whole realization noise
  is `rms = 6.0e-4` in $S$ units — 0.06%, i.e. **3% of the model's own
  predictive scatter** at the trough ($\sigma_{\rm pred}=0.019$). Even if
  $\lambda$'s noise reproduced *all* of it perfectly, the absorbable
  explained variance would be $7.7\times10^{-5}$.
* $2c$ is the mechanism itself, measured: the correlation between the
  statistic's noise wobble and the model's noise-driven prediction wobble is
  $r=0.28$ for $S(\ell)$, giving $+2.8\times10^{-4}$ of CV $R^2$.
* $f_{\rm hat}$ is the *price*: $\lambda$'s noise propagates into the
  prediction and wobbles it by $6.6\times$ more than the statistic actually
  wobbles. For every statistic the net effect of the noise is **negative**.
  With a noiseless $\lambda$, CV $R^2(S(\ell))$ would be $0.9663$, not
  $0.9632$.
* Signs are not systematically positive: PDF, $V_0$ and $V_2$ come out
  negative. With $r=0.28\pm{\sim}0.7$ on two degrees of freedom, $S(\ell)$'s
  absorption is **consistent with zero**.

**Units caveat.** The twobound `clk_resp` cache stores responses *as ratios to
a shared reference* (`paper_s3b_clusters.py` L295), so replica **differences**
are reference-free to first order — exactly the quantity
`family_basis_all.py::member_curve` already uses to build the shipped basis.
They exceed the corresponding $\Delta S$ by $1/S_{\rm ref}\le1.14$, so
$f_{\rm noise}$ and $2c$ are if anything overstated by up to 14%. The MF leg
uses the original `nongaussian_stats` grid interpolated onto the canonical
centres — the convention the shipped 14-bin $V_i$ bases were built with, not
the newer 22-bin `nu05` remeasurement.

**Honest limitation.** These are 3-sample (2-dof) variance estimates at one
parameter point; the replica-deviation matrix has $n_{\rm eff}=1.77$
independent bin-modes for $S(\ell)$ (adjacent-band correlation 0.48), so the
extra $\ell$ bins buy almost no degrees of freedom. Read them as an
order-of-magnitude, corroborated by §4 and §5, and let leg 1 replace them
with a 256-node number. A deliberately extreme reading (inflating **both**
variances to their one-sided 95% 2-dof ceilings, ×19.5) puts the $S(\ell)$
absorption ceiling at 0.020 = **5% of the gap**; that is the most pessimistic
statement the data support, against a measured 0.07%.

**Peaks and minima are the exception worth stating out loud**: their
realization noise is 7–9% of their design variance (they are shot-noise
statistics), and their measured absorption reaches +1.6% of their gap with a
pessimistic ceiling near 10%. Their CV $R^2$ (0.79, 0.84) is also the lowest
in the table, and the paper already attributes that to the family basis
rather than the latents. The flagship number is unaffected.

---

## 4. Why the shared channel is structurally small: the lensing kernel

$\lambda$ is measured from `snap_096` alone — one of the 20 snapshots stacked
into the lightcone, filling lens planes 1–4 of 80. Parsing the actual lux
plane geometry (`bind_lightcone_tng/lensplanes/config.dat`: 80 planes,
$\Delta\chi=51.25\,{\rm Mpc}/h$, plane scale factors stored in the file) gives
those planes at $\chi = 25.6, 76.9, 128.1, 179.4\,{\rm Mpc}/h$, i.e.
$z<0.061$, against $\chi(z_s{=}1)=2300\,{\rm Mpc}/h$ (45 planes in front of
the source).

Their Limber contribution to $C_\ell^{\kappa\kappa}$ at $z_s=1$ (halofit,
TNG cosmology) is **2.83% at $\ell=330$, falling monotonically to 0.42% above
$\ell=1.5\times10^4$; band mean 1.06%** (fig. `r5_fig2_absorption.png`b).
Two independent suppressions therefore act on the shared-draw channel:
$\ge97\%$ of $S(\ell)$'s signal comes from planes whose draws $\lambda$ never
sees, and within those planes the draw noise is already a $10^{-4}$-level
effect. This is a *structural* argument that does not depend on the 2-dof
replica statistics, and it agrees with them.

Note the seven statistics in the model are all $\kappa$ statistics, so this
kernel is the right one throughout. **For completeness, the $\tau$/$y$ case
is different and worth recording**: an electron-column kernel
$W_\tau\propto(1+z)^2$ integrated over the whole 80-plane lightcone has no
$\chi(\chi_s-\chi)/\chi_s$ geometric suppression at low $z$, so the
`snap_096` planes carry a **band-mean 3.5%** of $C_\ell^{\tau\tau}$, rising to
**8.8% at $\ell=330$** — 3.3× the lensing share (computed with the matter
power as a proxy for the electron power;
`/mnt/home/mlee1/ceph/referee_work/r5/r5_kernel_tau.npz`). Still small, but if
the latent model is ever extended to SZ or kSZ observables the shared-draw
channel there deserves its own measurement rather than an appeal to this one.

---

## 5. Monte-Carlo: how much does $\lambda$ precision matter at all?

`r5_inflation.py` jitters $\lambda$ with **fresh** noise (uncorrelated with
the statistics by construction) drawn from the per-run 8×8 bootstrap
covariance, and re-runs the exact shipped CV, 300 times. Mean degradation
$\Delta R^2 = R^2(\lambda)-R^2(\lambda+{\rm noise})$:

| jitter | $S(\ell)$ | PDF | peaks | minima | $V_0$ | $V_1$ | $V_2$ |
|---|---|---|---|---|---|---|---|
| bootstrap $\sigma$ | +0.0091 | +0.0105 | +0.0071 | +0.0103 | +0.0083 | +0.0111 | +0.0054 |
| replica $\sigma$ | +0.0059 | +0.0067 | +0.0053 | +0.0073 | +0.0033 | +0.0085 | +0.0042 |
| 2× bootstrap $\sigma$ | +0.0258 | +0.0280 | +0.0200 | +0.0271 | +0.0230 | +0.0327 | +0.0169 |

**What this bounds:** the *attenuation* leg. Doubling the latent noise
variance costs ~0.006 of CV $R^2$ for $S(\ell)$, consistent with the
$f_{\rm hat}=0.0033$ measured from the replicas. Latent precision is worth
$\lesssim1\%$ of $R^2$, and every bit of it works **against** the latent
route.

**What this does not bound:** the absorption leg — by construction, since the
injected noise is independent of the statistics. That is what §3 measures and
what leg 1 will measure properly. The two experiments are complementary and
must not be quoted as if either did the other's job.

---

## 6. Cross-provenance null: latents from zero-noise hydro patches (task 4)

`truth_snap096.npz` holds the same 2933 halos with the **full-hydro TNG300
patches** pasted instead of BIND's generated ones — the same reduction, zero
generative noise. Measuring $\lambda$ there and pushing it through the
shipped model (fig. `r5_fig3_truth_provenance.png`):

| latent | BIND | TRUTH | $\Delta/\sigma_{\rm design}$ |
|---|---|---|---|
| $\tilde f_\star$[13.2–13.4] | 0.09824 | 0.10051 | −0.027 |
| $\log\tilde T$[13.0–13.2] | 6.60332 | 6.60744 | −0.149 |
| $\log\tilde Y$[13.0–13.2] | −7.36031 | −7.39002 | +0.219 |
| $\log\tilde P_e$[14.0–14.3] | −12.48609 | −12.48487 | −0.006 |
| $c_{\rm gas}$[14.0–14.3] | 0.74916 | 0.74972 | −0.025 |
| $\log\tilde Y_{\rm ss}$[13.4–13.6] | −29.05502 | −29.06113 | +0.084 |
| $c_{\rm gas}$[13.2–13.4] | 0.63028 | 0.62170 | +0.154 |
| $\log\tilde P_e$[13.4–13.6] | −13.24779 | −13.28045 | +0.124 |

rms 0.121, max 0.219 $\sigma_{\rm design}$ — i.e. **BIND's fiducial latents
sit within a quarter of the design spread of the hydro truth's**, and much of
that residual is the known BIND bias plus the R3b cosmology-conditioning
mismatch of the released fiducial paint, not draw noise (these offsets are
1–7$\times$ $\sigma_{\rm rep}$, so they are systematic, not stochastic).

Predictions move by a median 0.17–1.14 $\sigma_{\rm pred}$ (0.03–0.79% in
relative terms) across the seven statistics. At the $S(\ell)$ trough:

| | value |
|---|---|
| measured BIND fiducial paint | 0.8812 |
| measured TNG300 full hydro | 0.8765 |
| model $\leftarrow\lambda$(BIND patches) | 0.8950 (0.74 $\sigma_{\rm pred}$) |
| model $\leftarrow\lambda$(HYDRO patches) | 0.8734 (0.41 $\sigma_{\rm pred}$) |
| $\sigma_{\rm pred}$ at the trough | 0.0188 |

Swapping provenance moves the trough by 0.0216 = 1.15 $\sigma_{\rm pred}$ —
**and it moves toward the measurement, not away**. Median $|{\rm model} -
{\rm measured}|$ for $S(\ell)$ improves from 1.09% to 0.31% against the BIND
paint and from 1.53% to 0.85% against the hydro truth.

That is the point: if the $\lambda\to$ statistic map were riding on
generative-noise structure, feeding it latents from a completely different,
noise-free provenance would break it. Instead the closure gets *better*. It
is a one-point test, and it is a test of bias rather than of noise
correlation — but it is the only fully independent provenance available
without new painting, and it comes out on the model's side.

---

## 7. Bottom line

| question | answer |
|---|---|
| How much of the 0.381 $S(\ell)$ gap could noise absorption explain? | **measured +0.07%** ($2c=+2.8\times10^{-4}$); ceiling at perfect correlation 0.26%; most pessimistic 2-dof reading 5% |
| Is the effect statistically significant? | No — $r = 0.28$ on 2 dof, consistent with zero; and it is negative for 3 of the 7 statistics |
| Does removing $\lambda$'s noise raise or lower the score? | **Raises it**: noiseless $\lambda$ gives $R^2 \simeq 0.9663$ vs 0.9632 |
| Which statistics are exposed? | peaks and minima ($f_{\rm noise}\simeq7$–9%, absorption up to +1.6% of their gap) — the two the paper already flags as basis-limited |
| Do shared seeds help, as the referee assumed? | **No — there are none.** The bound above is measured in the fully-independent-draw regime |
| Leg 1 cost | 750,848 halo generations ≈ **89 A100-GPU-hours**, ~6 h wall at `%16`, 18 GB transient with reduce-and-delete. Half-leg (64 nodes): 22 GPU-hr, ~1.5 h, ~3σ on the effect |

---

## 8. Drafted paragraph for `main.tex` §"Halo properties, not parameters" (`\label{sec:why_latents}`)

*Not applied — `main.tex` untouched. Suggested placement: after the sentence
ending "...while the eight measured halo numbers reach $0.96$ under a purely
linear map." Numbers as measured; the bracketed leg-1 clause is a
placeholder for the author.*

> One confound deserves a direct answer, because the comparison is between a
> predictor measured from the simulation's own output and a predictor that is
> exact by construction. The latents are measured from the same generated
> maps whose statistics they predict, so in principle the fit could absorb
> realization-level generative noise that the parameters cannot, inflating
> the latent score. The sampler draws fresh noise for every halo of every
> simulation, so this noise is node-specific and does not cancel across the
> design; we therefore measured it rather than argued it away. Three of the
> bound-to-bound runs turn out to carry identical parameter vectors, so they
> are the same simulation painted three independent times with the same halos
> and the same ray-tracing seed, and their spread isolates the generative
> realization noise exactly. That noise is $6\times10^{-4}$ in $S$ units, or
> $8\times10^{-5}$ of the suppression's variance across the design — three
> per cent of the model's own predictive scatter at the trough — and only the
> $z<0.06$ lens planes, which carry $1.1\%$ of $C_\ell^{\kappa\kappa}$ at
> $z_s=1$, can share draws with $\bm\lambda$ at all. Propagating the measured
> latent noise through the fitted map and correlating it with the measured
> statistic noise puts the resulting inflation of ${\rm CV}\,R^2$ at
> $+3\times10^{-4}$ for $S(\ell)$, with a ceiling of $1\times10^{-3}$ if the
> two noises were perfectly aligned: at most a quarter of a per cent of the
> $0.38$ gap, and formally consistent with zero. The same noise costs an
> order of magnitude more than it gains --- ordinary regression dilution
> removes $3\times10^{-3}$ of ${\rm CV}\,R^2$, and injecting extra latent
> noise degrades the score monotonically --- so a noiselessly measured
> $\bm\lambda$ would widen the gap rather than close it. Measuring the eight
> latents instead from the full-hydro patches of the same halos, which carry
> no generative noise at all, moves the predicted curves by about one
> predictive sigma and moves them *towards* the measurement: the median
> $S(\ell)$ closure at the fiducial improves from $1.1\%$ to $0.3\%$.
> [A direct re-measurement of $\bm\lambda$ from an independent re-paint of
> all 256 nodes is in progress and will be quoted here; the expected shift is
> $3\times10^{-4}$.]

**Hedges deliberately kept in:** "formally consistent with zero" (2 dof),
"at most a quarter of a per cent" (the corr$=1$ ceiling, not the point
estimate), "in progress" for leg 1. If the author prefers not to promise leg
1, the last sentence can be dropped without weakening anything above it.

**Do not write** anything of the form "shared seeds make the draws
common-mode across nodes" — it is false, and `main.tex` already contains such
a claim near line 430 which is being corrected separately.

---

## 9. Files

| file | what |
|---|---|
| `papers/01_pipeline/referee/R5_latent_noise_results.md` | this memo |
| `papers/01_pipeline/referee/R5_leg1_runbook.md` | leg-1 specification (prep only) |
| `papers/01_pipeline/referee/work/r5_common.py` | shipped latent reducer + CV harness, re-implemented verbatim |
| `papers/01_pipeline/referee/work/r5_repro.py` | task 1: baseline + $\theta$ routes |
| `papers/01_pipeline/referee/work/r5_bootstrap.py` | task 2: bootstrap noise floor + BIND/truth per-halo scatter |
| `papers/01_pipeline/referee/work/r5_replicas.py` | task 3: the 3-independent-paint measurement |
| `papers/01_pipeline/referee/work/r5_inflation.py` | task 3: lensing kernel + Monte-Carlo jitter |
| `papers/01_pipeline/referee/work/r5_truth.py` | task 4: hydro-provenance cross-check |
| `papers/01_pipeline/referee/work/r5_figs.py` | the three figures |
| `papers/01_pipeline/referee/figs/r5_fig1_noise_floor.png` | latent noise budget, three estimators |
| `papers/01_pipeline/referee/figs/r5_fig2_absorption.png` | absorption vs the gap; $z_s=1$ kernel share |
| `papers/01_pipeline/referee/figs/r5_fig3_truth_provenance.png` | $S(\ell)$ closure under a $\lambda$-provenance swap |
| `/mnt/home/mlee1/ceph/referee_work/r5/*.npz` | all numeric products |
