# R3b — Multi-sample-averaging test of the "map-scale texture systematic"

**Status: experiment EXECUTED** (with a substituted, cheaper design — see §2).
Agent R3b-texture, 2026-08-12. Repo `/mnt/home/mlee1/BIND`, branch `papers`.

---

## 0. Verdict up front

The referee asked whether the 5–10% gas-spectrum residual is single-sample
stochasticity that would shrink as `1/N`, and proposed that it "looks like
single-sample over-smoothing, which should worsen as feedback makes gas more
diffuse."

**Measured answer: the over-smoothing is real, but it is not a single-sample
effect.** Decomposing the BIND gas and Compton-`y` planes into a deterministic
part and a sampler-stochastic part, using **three genuinely independent flow
draws at identical parameters on identical halos**:

- The stochastic (draw-to-draw) component is **small**: it carries a *sub-percent
  to few-percent* share of the map power, rising with `k` — far below the paper's
  quoted 5–10% residual over most of the trusted range.
- The residual against hydro-pasted truth is **deterministic**. Averaging over
  draws does not remove it; it makes it very slightly *worse*, because the
  stochastic power was partially compensating a power *deficit*.
- So the paper's residual is **not** single-sample noise. For Compton-`y` it is a
  genuine, reproducible model bias — the maps are **under-structured** relative to
  truth (mean level right to a few percent, fluctuation power ~20% low).
- **The referee's proposed mechanism is falsified in sign.** The apparent growth of
  texture with gas diffuseness is **Simpson's paradox**: diffuseness and halo mass
  are anti-correlated at `rho = −0.85`, and mass is the real driver
  (`rho = −0.53`). Controlling for mass, the partial correlation with diffuseness
  is **negative** (−0.37 gas, −0.10 `y`). Pushing the measured diffuseness shifts
  at the extremes of the twobound prior through the calibrated relation changes the
  stochastic amplitude by **at most 6.5% of itself** — from 0.13% to 0.14% of map
  power. It cannot produce a parameter-dependent systematic that matters.

**Separately, and more consequentially: a large part of the paper's `tau` residual
is a parameter-file error, not a systematic at all.** The released fiducial paint
was conditioned on the **CAMELS** cosmology while its DMO input and hydro truth are
**TNG300** — `Omega_b/Omega_m` is 3.8% high, predicting `1.03814^2 = +7.77%` excess
gas/`tau` power against a measured **+7.69%** (§5.4). At matched cosmology the
deterministic gas-column bias is **−0.09%**. This needs an author decision (§9).

Numbers in §5. The net effect on the paper's "immunity" paragraph is that its
conclusion survives but its *reason* must change — and the honest rewrite (§7) is
**stronger** than the original: a deterministic, parameter-shared bias cancels in
rank and ratio statistics far more cleanly than per-realization noise ever could.

---

## 1. What was assigned vs what was possible

The brief assumed a local Tesla V100S and a deterministic per-halo seed ladder.
**Both premises are false on this node**, and I verified each before pivoting.

### 1.1 There is no GPU

```
nvidia-smi          -> "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"
/dev/nvidia*        -> does not exist
torch.cuda.is_available() -> False ; device_count 0
hostname            -> pcn-1-67      nproc -> 1
```

The session is on a **1-core, GPU-less** node. (The stored memory note
"workstation HAS a V100S" refers to a different host.) Slurm was **not** used —
per the guardrails I ran no `sbatch`/`srun` and no repeated Slurm polling.

Measured CPU sampler cost: loading the checkpoint takes **31.6 s**; generating
**4 halos** (`n_steps=50`, `batch_size=4`, 1 thread) had **not finished after
~9 minutes** before I killed it, i.e. **> 130 s/halo**. The assigned run
(2 933 halos × N=8 ≈ 23 k generations) projects to **> 800 CPU-hours**. Even the
reduced 666-halo single-slab, N=3 version is **> 24 h**. Direct re-generation was
therefore impossible within the 4 h budget, at any N.

Contention note: another referee agent's job (`r1_stats.py`, pid 2348211) was
running on the same single core throughout, so all wall-times below are shared-core
numbers.

### 1.2 There is no seed convention to offset

The brief asked me to find the per-halo seed derivation and choose a clean offset.
**There is none.** A repo-wide search of the inference package and the sampler
turns up no seeding of the generation path at all:

```
src/bind/model.py:454:   x = torch.randn(B, self.out_channels, ...)   # global RNG, unseeded
```

`bind.inference.paint.Model.generate` never sets a seed, `paint_stages.py` contains
no occurrence of the string `seed`, and neither the stage-1 manifest nor
`summary.json` records one. The only seeds in the codebase are for the Sobol design,
the lightcone transverse-shift realizations, `lightcone_transforms.from_seed`, and
shape-noise — **none of them touch the flow sampler**.

Two consequences:
1. The canonical paint is **not bit-reproducible** — the requested Stage-1
   provenance smoke test (reproduce stored patches with canonical seeds) is
   **not defined**, so it could not be run and was not faked.
2. Every independent invocation of the painter already draws **fresh** sampler
   noise. That is what made the substitute design below possible — and it is a
   finding worth recording in the paper's reproducibility statement.

---

## 2. The substituted design: a free N=3 ensemble

Because separate paint invocations draw fresh noise, **any two paints of the same
halos at the same parameters differ only by sampler noise**. I searched the
existing campaign trees for such a repeat and found one.

**The twobound campaign contains a duplicate-parameter group.** Of its 60 runs
only 58 parameter vectors are unique: runs **0018, 0049, 0053** have
**byte-identical** 35-dim vectors (`np.array_equal` on `twobound_params.npy` and on
each run's own `params.npy`).

Verified they are genuinely independent draws, not copies:

| check | result |
|---|---|
| `halo_centers`, `halo_masses` identical across the three | **True** (same halos) |
| `generated_patches` identical | **False** |
| per-halo patch relative RMS difference (mass channels) | **0.3055** |
| `thermo_patches` identical | **False** |
| per-halo patch relative RMS difference (thermo) | **0.2394** |

So: three independent flow-matching realizations, same halos, same checkpoint,
same parameters — a **30%-per-patch** stochastic spread. This is exactly the
ensemble the referee asked for, at **zero GPU cost**.

**Where the ensemble sits in parameter space.** Its vector differs from the
canonical lightcone `stage1/params.npy` in exactly 5 indices, and they are all
*cosmological*:

| idx | parameter | ensemble | canonical stage-1 |
|----|---|---|---|
| 0 | `Omega0` | 0.3089 | 0.3 |
| 1 | `sigma8` | 0.8159 | 0.8 |
| 6 | `OmegaBaryon` | 0.0486 | 0.049 |
| 7 | `HubbleParam` | 0.6774 | 0.6711 |
| 8 | `n_s` | 0.9667 | 0.9624 |

All **30 astrophysical/feedback parameters are exactly at fiducial**. The ensemble
carries the **TNG300 cosmology** (the canonical lightcone stage-1 file carries the
CAMELS SB35 fiducial cosmology instead), so the ensemble is, if anything, *better*
matched to the hydro truth than the canonical paint is. Conditioning is the only
route by which cosmology enters — the DMO input maps are the same TNG300-Dark
data in both cases.

---

## 3. Provenance actually used

| item | value |
|---|---|
| snapshot | `snap_096`, z = 0.0337244, a = 0.9673759 |
| halos | 2 933 (`M200c > 1e13`), 4 slabs: 666 / 723 / 743 / 801 |
| stage 1 (shared) | `/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096/stage1` |
| checkpoint | `weights/fm_redshift_thermo/last.ckpt` (+ `norm_stats.npz`) — the **only** weights dir present, and the redshift-conditioned thermo model the lightcone was painted with |
| compositing | `taper_frac=0.15`, `r200_factor=4.0`, `patch_mass_match=True` — read verbatim from the canonical `summary.json` |
| ensemble | twobound `run_0018 / 0049 / 0053`, `snap_096/composite_slab{00..03}.npz` |
| truth | `bind_science/runs/truth/run_0000/snap_096` (patches only, composited by me with the identical code path) |
| canonical | `bind_lightcone_tng/snap_096` (single draw, CAMELS-fiducial cosmology) |

All campaign trees were opened **read-only**; every write went to
`/mnt/home/mlee1/ceph/referee_work/texture/` or this `referee/` directory.

### Which comparison isolates texture

- **Hydro-pasted truth (used).** Same halos, same paste apertures, same taper,
  same compositing code — the *only* difference is patch content (BIND-generated
  vs TNG-measured). This isolates texture.
- **Full-hydro TNG slabs (not used).** These additionally contain all the diffuse
  gas *outside* the paste apertures (~80–85% of the sky per the diffuse-gas
  validation). That is a different, much larger systematic and would swamp the
  texture signal.

Note also that in this pipeline **`tau` is a constant multiple of the composited
gas plane** (`lightcone_maps._tau_per_gas_pixel(...) * gas`, no separate field),
so the gas-column result *is* the `tau` result — the fractional decomposition is
identical by construction. That is why only `gas` and `y` are carried below.

---

## 4. Method

Draw *i* gives a map `m_i = s + n_i`, with `s` the deterministic prediction
(parameters, halos, weights) and `n_i` zero-mean sampler noise independent across
draws. In Fourier space, binned in `|k|`:

```
auto    A(k) = <|F_i|^2>_i            = |S|^2 + P_n
cross   X(k) = <Re F_i F_j*>_{i<j}    = |S|^2          (unbiased: noise cancels)
noise   P_n  = A - X
mean-of-N     P_N = |S|^2 + P_n / N
```

Planes are built with `bind.inference.pipeline`'s own `paste_halos_2d` /
`square_taper_weight` / `circular_taper_weight` and the production mass-match
(one scalar per halo per draw, `m_dmo / m_pred` over all three mass channels;
thermo channels are *not* mass-matched — matching `build_bind_composite`), then
**center-cropped to 4096²**, which is both the paper's own lensplane convention
(`paint_yplane --lp_grid 4096`) and necessary for a fast FFT (4198 = 2 × 2099,
and 2099 is prime).

Slabs are summed: independent slabs add incoherently in *both* signal and noise,
so the fractional decomposition carries over to the line-of-sight stack.

> **Honesty note on "does it scale as 1/N".** Once `|S|^2` is estimated by the
> draw-cross-spectrum, `P_N = |S|^2 + P_n/N` is an **algebraic identity** — my
> measured ratios come out as `e1/e2 = 2.0000000000` and `e1/e3 = 3.0000000000`,
> which is arithmetic, *not* evidence. I therefore do **not** claim the `1/N` law
> as a measurement. The empirical content is (a) the **amplitude** of `P_n`,
> (b) the **consistency of the three pairwise cross-spectra** (which bounds any
> draw-to-draw correlation the decomposition would otherwise miss), and (c) the
> **residual against truth that survives averaging**. Those are what §5 reports.

Scripts (all in this directory, runnable):

| file | role |
|---|---|
| `r3b_texture_analysis.py` | builds planes, FFTs, auto/cross/mean-of-N spectra, per-halo diagnostics |
| `r3b_texture_figs.py` | figures + headline numbers (`r3b_summary.json`) |
| `r3b_feedback_leg.py` | calibrated feedback extrapolation (§6) |

---

## 5. Results

All numbers below are **4 slabs of snap_096 summed**, 2 933 halos, N = 3 draws.
`k` in `h`/Mpc; `ell ~ k*chi` with `chi = 100.3` Mpc/h for this plane.
`f_stoch = P_n / P_single` is the stochastic share of a single map's power.
"floor" is the `N -> inf` limit, i.e. the residual that averaging can never remove.

### 5.1 Gas surface density (identically the `tau` result — see §3)

| band | `f_stoch` | single draw − truth | mean-of-3 − truth | **floor (N→∞)** | canonical paint − truth |
|---|---|---|---|---|---|
| `k < 0.5` | 0.012% | −0.21% | −0.21% | **−0.22%** | +7.19% |
| `0.5 < k < 3` | 0.13% | +0.04% | −0.05% | **−0.09%** | +7.73% |
| `3 < k < 15` | 4.46% | +3.39% | +0.26% | **−1.31%** | +9.70% |
| `k < 15` | 1.38% | +0.95% | −0.02% | **−0.51%** | +8.10% |

### 5.2 Compton-`y`

| band | `f_stoch` | single draw − truth | mean-of-3 − truth | **floor (N→∞)** | canonical paint − truth |
|---|---|---|---|---|---|
| `k < 0.5` | 0.19% | −26.66% | −26.75% | **−26.80%** | −26.58% |
| `0.5 < k < 3` | 0.45% | −19.16% | −19.40% | **−19.52%** | −20.26% |
| `3 < k < 15` | 2.91% | −9.45% | −11.21% | **−12.09%** | −17.60% |
| `k < 15` | 1.08% | −19.25% | −19.88% | **−20.20%** | −21.99% |

Stability of the decomposition: the three pairwise draw-cross spectra agree to a
median **0.16%** (gas) and **1.7%** (`y`), which bounds any draw-to-draw
correlation the split into signal + noise would otherwise have missed.

### 5.3 Reading

1. **The stochastic texture is small and strongly scale-dependent.** It is
   0.1–0.4% of map power through the trusted range (`k < 3`, `ell < 300` on this
   plane), and only reaches the paper's quoted 5–10% band at `k ≈ 5–10` (gas) and
   `k ≈ 15–20` (`y`) — at and beyond the pixelization regime the paper already
   excludes with its `0.8 ell_Ny` cut.
2. **Averaging does not remove the residual — for `y` it slightly worsens it.**
   Going N = 1 → 3 moves the `y` residual from −19.16% to −19.40%, converging on a
   floor of −19.52%. The stochastic power was *masking* a deficit, so removing it
   exposes slightly more of it. This is the direct refutation of "the 5–10% is
   single-sample noise": **at mid-`k` only 0.37 of the 19.16 percentage points
   (1.9%) is stochastic**; ~98% is a reproducible model bias.
3. **`y` maps are genuinely under-structured** — the referee's "over-smoothing"
   intuition is right about the *phenomenon* and wrong about the *cause*. The mean
   `y` level is only ~2–4% low (`y.mean` 1.252–1.275e−8 across draws vs 1.30e−8 for
   truth) while the *power* is ~19% low, i.e. the fluctuation amplitude is low by
   ~8% beyond the mean offset. That deficit is deterministic and survives averaging.
4. **For gas/`tau` at matched cosmology the deterministic bias is essentially
   zero**: −0.09% at mid-`k`, −0.22% at low `k`. BIND's gas column is an excellent
   match to the hydro-pasted truth wherever the paper trusts it.

### 5.4 Incidental finding — the released fiducial paint carries the wrong cosmology

The "canonical paint" column above sits at **+7.2 to +7.7%** for gas across the
whole trusted range while the matched-cosmology ensemble sits at ~0. That gap is
**not texture**. It is a conditioning-provenance error:

| | `Omega0` | `sigma8` | `OmegaBaryon` | `HubbleParam` | `n_s` |
|---|---|---|---|---|---|
| `bind_lightcone_tng/snap_096/stage1/params.npy` (the released paint) | 0.3 | 0.8 | 0.049 | 0.6711 | 0.9624 |
| `bind_science/runs/fiducial/run_0000/params.npy` **and every twobound/Sobol run** | 0.3089 | 0.8159 | 0.0486 | 0.6774 | 0.9667 |
| TNG300 / TNG300-Dark (the DMO input **and** the hydro truth) | 0.3089 | 0.8159 | 0.0486 | 0.6774 | 0.9667 |

The released fiducial lightcone was conditioned on the **CAMELS SB35 fiducial
cosmology** rather than TNG300's own. Because `patch_mass_match` pins the *total*
patch mass to the DMO cutout, the conditioned baryon fraction sets the gas split:

```
(Omega_b/Omega_m) ratio  = 0.163333 / 0.157332 = 1.03814
predicted power excess   = 1.03814^2            = +7.77%
measured (canonical − ensemble, mid-k)          = 7.725 − 0.037 = +7.69%
measured mean-gas ratio (slab 01)               = 6.491e8 / 6.250e8 = 1.0386
```

Agreement to **0.1 percentage points in power and 0.05% in the mean**. So a large
part — plausibly all — of the paper's quoted **5–10% `tau` map-level excess is a
one-line parameter-file error in the fiducial paint, not a model systematic.**
`bind_science/runs/fiducial/run_0000/params.npy` already holds the right vector.

**This is the single most actionable item in this memo.** It needs an author
decision (§9), and it should be checked before the `tau` residual is quoted.

---

## 6. Feedback leg — does the texture grow with feedback?

The direct test (repaint at a strong-feedback node) needs a GPU. What existing
data supports is a **calibrated extrapolation**: the twobound campaign painted
every parameter extreme on the *same halos*, and one draw is enough to measure
gas diffuseness, so the fiducial-calibrated `f(mass, diffuseness)` relation can be
pushed through the measured diffuseness shift.

### 6.1 The mechanism is a mass effect, not a diffuseness effect

| quantity | gas | `y` |
|---|---|---|
| median per-halo `sigma_draw/rms` | 0.196 | 0.192 |
| Spearman `rho` vs `log M200c` | **−0.531** | **−0.396** |
| Spearman `rho` vs diffuseness (RAW) | +0.288 | +0.287 |
| `rho`(diffuseness, `log M`) | −0.851 | −0.851 |
| **partial `rho` vs diffuseness, controlling `log M`** | **−0.369** | **−0.105** |

The raw positive correlation that would appear to support the referee is
**Simpson's paradox**: diffuseness and halo mass are strongly anti-correlated
(`rho = −0.85`), and the real driver is mass — low-mass halos have both more
diffuse gas and larger stochastic scatter. **Controlling for mass, the partial
correlation flips sign** and the within-mass-bin curves are non-monotonic,
turning over at the highest diffuseness (fig. 2, right panel). There is **no
monotonic "more diffuse → more texture" relation** once the confound is removed.

### 6.2 Extrapolated to the parameter extremes

| node | twobound run | mean diffuseness | shift vs fiducial | predicted `f_gas` | predicted `f_y` |
|---|---|---|---|---|---|
| fiducial (ensemble) | 18/49/53 | 0.8931 | — | ×1.000 | ×1.000 |
| `VariableWindVelFactor` max (14.8) | 0005 | 0.8823 | −0.0108 | ×1.029 | ×1.014 |
| `VariableWindVelFactor` min (3.7) | 0004 | 0.9068 | +0.0137 | ×0.969 | ×0.986 |
| `WindEnergyIn1e51erg` max (14.4) | 0001 | 0.8706 | −0.0225 | **×1.065** | ×1.034 |
| `WindEnergyIn1e51erg` min (0.9) | 0000 | 0.9058 | +0.0127 | ×0.970 | ×0.986 |
| `RadioFeedbackFactor` max (4.0) | 0003 | 0.8974 | +0.0043 | ×0.989 | ×0.995 |
| `RadioFeedbackFactor` min (0.25) | 0002 | 0.8885 | −0.0046 | ×1.012 | ×1.006 |

**Largest predicted change anywhere on the twobound envelope: ×1.065 (gas), ×1.034
(`y`).** Applied to the mid-`k` stochastic share that means 0.13% → 0.14% (gas) and
0.45% → 0.46% (`y`). Even at the extremes of the prior, the stochastic texture
cannot generate a parameter-dependent systematic at a level that matters.

Note also the sign: stronger wind feedback slightly *reduces* this patch-based
diffuseness proxy (gas is stripped from the patch outskirts faster than from
inside `R200`), which is the opposite of the intuition in the referee's comment.

---

## 7. Drafted replacement for the "immunity" paragraph (main.tex ~line 592)

`main.tex` was **not edited** — this is a drop-in candidate. It keeps the
paragraph's conclusion but replaces every assertion with a measurement, and it is
*stronger* than the original: a bias that is deterministic and shared across
parameter locations cancels in rank and ratio statistics far more cleanly than a
random one would.

> Given these biases, it is fair to ask what the $\tau$ and Compton-$y$ maps are
> useful for. The answer follows from how they are used in the remainder of this
> work, and we can now demonstrate rather than assert it. Because the flow-matching
> sampler draws fresh noise per halo, two paints of the same halos at the same
> parameters differ only by that noise; we exploit a set of three independent
> paints at identical parameters on the identical $z=0.034$ halo population to
> separate the map residual into a stochastic part and a deterministic part. The
> stochastic part is measured from the auto-minus-cross power of the three draws,
> and it is small: it carries $0.13\%$ of the gas-column map power and $0.45\%$ of
> the Compton-$y$ map power at $k\simeq0.5$--$3\,h\,{\rm Mpc}^{-1}$, rising to
> $4.5\%$ and $2.9\%$ only at $k>3\,h\,{\rm Mpc}^{-1}$, i.e. within the
> pixelization regime our $0.8\,\ell_{\rm Ny}$ cut already excludes. Averaging the
> three draws therefore does not remove the residual: the Compton-$y$ deficit moves
> from $-19.2\%$ to $-19.4\%$ and converges to a floor of $-19.5\%$, so
> $\gtrsim\!98\%$ of it is a reproducible property of the model rather than a
> single-sample fluctuation. This is what licenses the response science of
> \S\S~\ref{sec:astro}--\ref{sec:emulator}: the offset is deterministic given the
> halos and the weights, so the rank correlations of \S~\ref{sec:correlations} are
> invariant to it and ratio statistics such as $S(\ell)$ divide it out, which would
> \emph{not} follow if the residual were per-realization noise. We also verified
> that the stochastic amplitude is set by halo mass (Spearman $\rho=-0.53$) rather
> than by gas diffuseness --- the apparent diffuseness trend is a mass confound,
> and the partial correlation at fixed mass is $-0.37$ --- so the concern that
> stronger feedback would inflate this term is not borne out: propagating the
> measured diffuseness shifts at the extremes of the prior changes the stochastic
> amplitude by at most $6.5\%$ of itself. The honest limit remains absolute
> amplitude: the Compton-$y$ maps are under-structured relative to the hydro-pasted
> truth at the $\sim\!20\%$ level in power (a $\sim\!10\%$ amplitude deficit, with
> the mean $y$ low by only a few percent), and predictions of the absolute
> $C_\ell^{yy}$ or the mean Compton-$y$ inherit that offset, as well as the missing
> diffuse gas discussed in \S~\ref{sec:caveats}. Those are not the intended use of
> this release.

**Two edits the authors must resolve before using this text:**

1. The `tau` sentence. At *matched* cosmology the gas-column bias is $-0.09\%$ at
   mid-$k$ — i.e. essentially perfect. The $\sim\!8\%$ excess in the released paint
   is the cosmology-conditioning error of §5.4. Either fix the paint and quote
   $\lesssim\!1\%$, or quote the excess and attribute it correctly. **Do not
   describe it as a texture systematic.**
2. The bracketed `[5-10\%]` and the `[QUANTIFY ...]` placeholder earlier in
   \S~\ref{sec:val_halo} must be filled from the corrected figure.

---

## 8. Caveats — what this does and does not establish

- **N = 3, one snapshot, plane level.** Three draws give three pairwise crosses;
  their 0.16%/1.7% agreement bounds the estimator noise, but a larger N would
  tighten `P_n` at high `k`. The test is at `z = 0.0337` only and is plane-level,
  not ray-traced. The fractional decomposition should carry to the lightcone
  because independent slabs add incoherently in *both* signal and noise, but that
  is an argument, not a measurement.
- **"Shrinks as 1/N" is not a measurement.** As set out in §4, `P_N = |S|^2 + P_n/N`
  is algebraically exact once `|S|^2` is the draw-cross-spectrum. The measured
  content is the amplitude of `P_n` and the floor, not the exponent.
- **The ensemble is not at the released paint's parameters.** It sits at the TNG
  cosmology with all 30 feedback parameters at fiducial; the released paint sits at
  the CAMELS cosmology (§5.4). This is why both columns are reported.
- **The feedback leg is an extrapolation, not a measurement.** It assumes the
  `f(mass, diffuseness)` relation is itself parameter-independent, and uses one
  patch-based diffuseness proxy (6.25 Mpc/h patch, `M_gas(<R200)/M_gas(patch)`).
  The direct test is one command on a GPU box — `r3b_gpu_repaint.py`, §9.
- **Truth is the hydro-pasted composite**, so everything here is blind by
  construction to the diffuse gas outside the paste apertures. That systematic is
  separate and larger; see the existing diffuse-gas validation.
- **Mass-matching does not drive any of this.** `patch_mass_match` is applied
  identically to BIND and truth, and the scales are all ~1: median 1.0003 (truth),
  0.9984 (draw 18), 0.9987 (canonical), 16–84% spread <0.5%. The truth-vs-BIND
  differential is **0.2%** — two orders below the `y` deficit and far below the
  cosmology effect. And the Compton-`y` channel is **not** mass-matched at all
  (matching `build_bind_composite`), so the −20% `y` result is independent of this
  choice entirely.

---

## 9. To run the direct test on a GPU box, and open items

```bash
# 1. N=8 fresh, SEEDED draws at the fiducial location  (~2933 halos x 8)
python papers/01_pipeline/referee/r3b_gpu_repaint.py \
    --params /mnt/home/mlee1/ceph/bind_science/runs/fiducial/run_0000/params.npy \
    --n_samples 8 --device cuda \
    --out /mnt/home/mlee1/ceph/referee_work/texture/gpu_fid

# 2. the same at the strongest-feedback node (max WindEnergy = twobound run_0001;
#    max VariableWindVelFactor = run_0005)
python papers/01_pipeline/referee/r3b_gpu_repaint.py \
    --params /mnt/home/mlee1/ceph/bind_science/runs/twobound/run_0001/params.npy \
    --n_samples 8 --device cuda \
    --out /mnt/home/mlee1/ceph/referee_work/texture/gpu_wind_max

# 3. analyse either ensemble with the same code path used for this memo
python papers/01_pipeline/referee/r3b_texture_analysis.py \
    --ensemble_root /mnt/home/mlee1/ceph/referee_work/texture/gpu_fid \
    --out /mnt/home/mlee1/ceph/referee_work/texture/r3b_spectra_gpu_fid.npz
python papers/01_pipeline/referee/r3b_texture_figs.py \
    --spec /mnt/home/mlee1/ceph/referee_work/texture/r3b_spectra_gpu_fid.npz --tag _gpu_fid
```

`r3b_gpu_repaint.py` also **adds the seed control the production path lacks**
(`torch.manual_seed(BASE_SEED + s*10^6)`, recorded in a manifest), which is worth
upstreaming into `bind.inference.paint.Model.generate` regardless of this test —
right now no BIND paint anywhere in the campaign is reproducible.

### Open items for the authors

1. **Decide on the cosmology-conditioning error in the released fiducial paint**
   (§5.4). It explains the `tau` excess almost exactly. Re-painting snap_096 (and
   the other 19 snapshots) with `runs/fiducial/run_0000/params.npy` is the fix.
2. **The Compton-`y` power deficit (~20%) is real and unexplained here.** It is
   deterministic, largest at low `k` (−27%) and smallest at high `k` (−12%), and it
   is *not* a normalization error (the mean is only a few percent low). Worth its
   own investigation; the `Y`–`M` rescaling test already reported in the paper is
   consistent with it not being an amplitude offset.
3. **Upstream seeding** into the sampler for reproducibility.
4. Fill the `[QUANTIFY ...]` and `[5-10\%]` placeholders in `main.tex`.

---

## 10. Files

**Deliverables — `papers/01_pipeline/referee/`**

| file | what |
|---|---|
| `R3b_texture_results.md` | this memo |
| `r3b_texture_analysis.py` | plane building, spectra, per-halo diagnostics (`--ensemble_root` consumes GPU ensembles) |
| `r3b_texture_figs.py` | figures + `r3b_summary.json` |
| `r3b_feedback_leg.py` | calibrated feedback extrapolation |
| `r3b_gpu_repaint.py` | runnable GPU driver for the direct test, with seed control |
| `figs/fig_r3b_1_oneoverN.png` | decomposition, stochastic share vs scale, residual vs truth incl. the canonical-paint offset |
| `figs/fig_r3b_2_mechanism.png` | per-halo stochastic fraction vs mass, vs diffuseness, and mass-controlled |

**Data — `/mnt/home/mlee1/ceph/referee_work/texture/`** (224 KB total)

`r3b_spectra.npz` (all binned spectra, 4 slabs) · `r3b_perhalo.npz` ·
`r3b_summary.json` · `r3b_feedback_leg.{npz,json}` · `analysis_all.log` ·
`feedback_leg.log`

**Compute.** No GPU (none present). No Slurm. Single CPU core, shared with another
agent's job throughout. Plane analysis 136–191 s per slab (5 sources x 2 fields:
composite + FFT), ~11.7 min for 4 slabs; whole pipeline under 30 min of CPU.
Sampler throughput on that core: **>130 s/halo**, i.e. >800 CPU-hours for the
originally-assigned 23k generations — the reason for the substituted design.
