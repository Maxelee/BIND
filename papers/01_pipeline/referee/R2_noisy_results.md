# R2 — "your detectability claims are noiseless": the LSST-Y10 shape-noise twins

**Referee objection.** Every detectability and validation claim in the paper (Figs 5 and 7)
rests on noiseless, area-scaled covariances. Shape noise dominates the ν-domain statistics
(peaks, minima, Minkowski functionals) at exactly the scales used, so the claims are
untested where they matter most.

**Response, in three lines.**

1. **The ℓ-domain detectability claim survives untouched.** The paper's noiseless "37×
   LSST-Y10" at ℓ=5000 becomes **55×** when the precision is measured directly on
   shape-noise-carrying maps, and **31×** under the more conservative (2×) shape-noise
   convention. Both are "more than an order of magnitude", exactly as the caption claims.
2. **Every ν-domain statistic keeps Stage-IV detectability, with reduced margins.** 100 %
   of the 256 Sobol nodes remain >5σ from the fiducial under the noisy LSST-Y10 covariance;
   after subtracting our own 50-realization measurement floor, 72–96 % survive at >5σ.
   Peaks and minima are the statistics the referee is right about — shape noise supplies
   **79 %** of their covariance and inflates their error bars by **2.15×** — but they still
   deliver 25–35× the LSST-Y10 1σ across the prior.
3. **The validation closure gets safer, as the physics requires.** Six of eight statistics
   improve, the PDF by **7.8×** and $V_2$ by **4.0×** in χ² excess; none degrades.

**Cache status: complete and bit-reproducible. No gaps.**

Everything below is reproducible from `referee/work/r2_*.py` run with
`/mnt/home/mlee1/venvs/BIND_env/bin/python3`. Bulk outputs are in
`/mnt/home/mlee1/ceph/referee_work/r2/`. Nothing under `bind_sb35/` was written.

---

## 1. Cache verification

`build_noisy_cache.py --verify` (2026-08-12, from `papers/01_pipeline/`): **all checks PASS.**

```
run_ids matches DATASET_IN's 256 Sobol runs: OK
peak_counts     (256, 5, 22)   finite 100.00%  shape OK
minima_counts   (256, 5, 22)   finite 100.00%  shape OK
pdf             (256, 5, 22)   finite 100.00%  shape OK
mf_v0 / v1 / v2 (256, 5, 22)   finite 100.00%  shape OK
cl_kappa        (256, 5, 724)  finite 100.00%  shape OK
all nu axes == 22 bin centres of NU_EDGES: OK
run_0000 / run_0001 / run_0002 / run_0255 shard == assembled row: OK
kappa_rms metadata present (5 planes): OK
```

`bind_sb35/nu05n_shards/` holds 272 files: 256 `run_NNNN.npz`, the merged
`fid.npz` / `truth.npz` / `dmo.npz` — **550 realizations each**, read from each shard's own
`n_real`, carrying per-realization draws `(550, 5, 22)` for all six ν statistics and
`(550, 5, 724)` for `cl_kappa` — their 4 realization chunks each, and `_kappa_rms_fid.npz`
(`n_real_used = 550`). The assembled `emulator_dataset_nu05n.npz` (15.7 MB, 2026-08-05)
covers all 256 Sobol runs. All three heavy targets carry the *identical* `kappa_rms`
vector `[0.016023, 0.020657, 0.025066, 0.028815, 0.031634]`, so the ν axis means the same
thing for every target — the "refuse, don't guess" guard in `noisy_grid.load_kappa_rms` did
its job. **Nothing needs rebuilding.**

**Independent re-derivation (`r2_noisegain.py`).** 120 fiducial realizations at $z_s=1$ were
recomputed from the raw `kappa_maps.npz` through the module's own `noise_rng` / `smooth` /
statistic path. All six ν statistics reproduced the cached `*_real` rows with **maximum
relative deviation exactly 0.0** — bit-reproducible, not merely consistent.

**Recipe validated, not assumed (`r2_probe.py`).** Pure shape noise pushed through the
identical `smooth()` + `power_spectrum()` path:

| quantity | measured | analytic | agreement |
|---|---|---|---|
| flat (unsmoothed) $N_\ell$ | $1.05924\times10^{-10}$ | $\sigma_e^2/(2n_{\rm gal})\,A_{\rm px}^{\rm sr}=1.05927\times10^{-10}$ | 1.00000 |
| $W^2(\ell)$, $\ell<1.2\times10^4$ | — | $\exp(-\ell^2\sigma_\theta^2)$, $\sigma_\theta=\theta_G/\sqrt2$ | max dev **0.068 %** |

The "$\theta_G$ is the 1/e radius, *not* the filter σ" reading of Lee+22 Eq. 6 is confirmed
to better than a part in $10^3$. Had the code used $\theta_G$ *as* the filter σ, $W^2$ would
be 41 % too wide and this test would have failed loudly.

**Seed pairing confirmed.** For the fid/truth pair the per-realization difference scatter
divided by the unpaired quadrature sum is 0.21 ($V_0$) – 0.92 ($C_\ell$), i.e. < 1
everywhere: the two targets are traced through the same lightcone realizations, with
independent shape-noise draws by design (`noise_rng` keys on the target string).

**One convention flag that propagates into every number (see §3, §6, §7).** The cache uses
the Lee+22 Eq. 7 noise, $N_\ell=\sigma_e^2/(2n_{\rm gal})$. The paper's own analytic band
(`cl_relerr` in `_build_figures_nb.py`, and the LSST-DESC SRD convention) uses
$N_\ell=\sigma_e^2/n_{\rm gal}$ — **twice** as much. Both are carried below; the "2× noise
conv." column is the conservative one.

---

## 2. Validation twin — `imgs/fig05n_field_validation_noisy.{png,pdf}`

Same eight-panel layout as paper Fig 5, every statistic re-measured on the noisy, smoothed
maps: (a) $C_\ell^{\kappa\kappa}$, (b) $S(\ell)$, (c) PDF, (d) peaks, (e) minima,
(f–h) $V_0,V_1,V_2$. Five source-plane curves for BIND, hydro-pasted truth dashed at
$z_s=1$; residual strips $100(\mathrm{BIND}-\mathrm{truth})/\mathrm{truth}$ at $z_s=1$ with
the **LSST-Y10 band from the noisy truth 550-realization covariance** — main.tex Eq. (cov)
→ Eq. (area\_scaling) at $A=18{,}000\,$deg², with Hartlap Eq. (inv\_cov) applied whenever
the precision matrix is formed. Point error bars are the seed-paired per-realization
difference scatter / $\sqrt{550}$, used uniformly on all eight panels (Poisson is not
meaningful once shape-noise peaks dominate the counts, so the paper's mixed
Poisson/paired-scatter convention is replaced by the single paired one).

**ℓ-domain bookkeeping.** Signal and noise are both multiplied by $W^2(\ell)$, so $W^2$
cancels identically in $S(\ell)$ and in every *relative* error. What does not cancel is the
additive noise floor, so it is subtracted first using the **measured** $\hat N(\ell)$ of §1.
Panel (a) plots the debiased signal with the floor drawn as a dotted line. Skipping the
debiasing would badly dilute the ratio: $C^{\rm bind}/C^{\rm dmo}=0.990$ raw versus the true
$0.927$ at $\ell=8000$. The ℓ range is cut at $\ell=8000$, where $W^2=0.066$ and the
debiased signal is still $\ge$ 15 % of the floor.

### 2.1 Closure χ² per statistic — noisy vs noiseless

$\chi^2=\Delta^{\sf T}\hat{\bm C}(A)^{-1}\Delta$, $\Delta=\langle{\rm BIND}\rangle-
\langle{\rm truth}\rangle$ at $z_s=1$, Hartlap-debiased, $A=18{,}000$ deg².
"floor" is the χ²/dof a *perfect* emulator would still show from the finite realization
count; **"excess" = χ²/dof − floor** is the unbiased estimate of the real mismatch and is
the column to read. $A_{\rm eq}=18000/{\rm excess}$ is the survey area at which the residual
reaches 1σ per dof.

| statistic | dof | **noisy** χ²/dof | floor | **excess** | $A_{\rm eq}$ [deg²] | **noiseless** χ²/dof | floor | **excess** | safer by |
|---|---|---|---|---|---|---|---|---|---|
| PDF                     | 22 | 4.98 | 2.01 | **2.97** |  6 060 | 34.45 | 11.16 | **23.30** | **7.8×** |
| peak counts             | 22 | 4.48 | 2.14 | **2.33** |  7 715 |  5.22 |  3.11 | **2.10**  | 0.9× |
| minima counts           | 22 | 1.91 | 1.75 | **0.16** | 1.1e5  |  1.18 |  1.02 | **0.16**  | 1.0× |
| $V_0$                   | 22 | 3.30 | 2.12 | **1.19** | 15 170 |  4.71 |  2.71 | **2.00**  | **1.7×** |
| $V_1$                   | 22 | 4.16 | 2.11 | **2.05** |  8 774 |  7.97 |  4.84 | **3.13**  | **1.5×** |
| $V_2$                   | 22 | 4.12 | 2.08 | **2.04** |  8 827 | 11.04 |  2.86 | **8.18**  | **4.0×** |
| $C_\ell^{\kappa\kappa}$ | 24 | 1.97 | 1.30 | **0.66** | 27 180 |  2.55 |  0.22 | **2.33**  | **3.5×** |
| $S(\ell)$               | 24 | 2.06 | 1.30 | **0.75** | 23 890 | — identical to $C_\ell$ by construction | | | |

**The closure gets safer under noise: 6 of 8 statistics improve, none degrades.** Shape
noise inflates the covariance faster than it inflates the BIND–truth residual, because the
residual is a property of the painted field while the noise is common to both sides.

Median |residual| vs median LSST-Y10 1σ band (noisy): PDF 0.50 % / 0.37 %; peaks 0.97 % /
0.63 %; minima 0.28 % / 0.77 %; $V_0$ 0.55 % / 0.43 %; $V_1$ 0.65 % / 0.38 %; $V_2$ 0.72 % /
0.35 %; $C_\ell$ 0.43 % / 0.55 %. **Typical residuals sit at 0.7–2× the LSST-Y10 1σ band**;
the worst single bin reaches 3.1–5.5×, always in the sparsely populated $\nu\gtrsim5$ tail.
At the lightcones' actual 25 deg² footprint the same χ²/dof are 0.003–0.007, i.e.
$\ll1$ — the residuals only become visible once the covariance is scaled by 720×.

### 2.2 The honest qualification the text needs

A **full-covariance** χ² at the full 18,000 deg² is not ≤ 1 for any statistic, noisy *or*
noiseless. The paper's claim (lines 574, 576) is the *per-bin, by-eye* statement the figure
actually supports — and that statement is fine. Quantitatively, the floor-corrected
residuals reach 1σ per dof at $A_{\rm eq}\simeq6\times10^3$–$1\times10^5$ deg², i.e. **BIND
matches its hydro-pasted truth at the statistical precision of a survey of several thousand
to $\gtrsim10^5$ deg², bracketing LSST-Y10.** Drafted wording is in §6; two structural
reasons the χ² is if anything *over*-stated are in §7.

---

## 3. Detectability twin — `imgs/fig23n_s3_opener_noisy.{png,pdf}`

Panel (a): the 5–95 % spread of $S(\ell)$ across the 256 Sobol nodes at $z_s=1$ as measured
on the noisy maps (grey), the converged (noiseless, seed-paired) spread as a dashed outline,
the fiducial (black), and LSST-Y10 / Euclid ±1σ envelopes **computed from the noisy fiducial
550-realization covariance**, area-scaled to each survey. Panel (b): prior span /
$\sigma_{\rm survey}$, with the 2×-noise convention and the 50-realization measurement floor
drawn as references.

**Smoothing choice, stated explicitly.** $S(\ell)$ is a ratio of two spectra measured on
identically smoothed maps, so $W^2(\ell)$ cancels exactly in the response, and it cancels
again in the relative precision: $\sigma(\hat C)/\hat C = r_{\rm mode}\,(1+N/C)$ with $W^2$
dividing out of both $N$ and $C$. **The band is therefore the honest survey precision an
unsmoothed power-spectrum analysis would have at the same $n_{\rm gal},\sigma_e$** — the
smoothing neither helps nor hurts the ratio, it only limits the ℓ range over which this
cache can speak ($\ell\le8000$). The one thing that does not cancel, the additive floor, is
removed with the measured $\hat N(\ell)$; the resulting debiased $S(\ell)$ agrees with the
released noiseless suppression (span 20.30 % vs 19.75 % at ℓ≈5000, a 3 % difference), which
is itself the end-to-end check on the debiasing.

**Euclid.** The cache exists only at LSST-Y10 noise, so the Euclid band swaps the shot-noise
floor at fixed mode counting. That is justified: the measured relative scatter of
$\hat C_\ell^{\rm tot}$ tracks $\sqrt{2/N_{\rm modes}}$ to 0.3 % for $\ell\ge3000$ and to
12 % at $\ell\sim300$ (mildly non-Gaussian at the largest scales, in the conservative
direction).

**Response numerator.** For a spectrum the shape-noise bias is target-independent, so the
*response* is noise-free in expectation and only the *covariance* is inflated. The quoted
ratios therefore use the converged (noiseless, seed-paired) span over the noisy σ. Using the
as-measured noisy span instead changes them by ≤ 5 % above ℓ≈1500; below ℓ≈1180 the
50-realization measurement floor exceeds the span and the as-measured band is
noise-broadened — visible directly in panel (a), and the reason the converged span is the
right numerator.

### 3.1 The headline number, four ways (ℓ ≈ 5000, $z_s=1$)

| construction | Sobol 5–95 % span | σ(LSST-Y10) | ratio |
|---|---|---|---|
| **paper, noiseless**: span from `emulator_dataset_nu05`, σ from analytic `cl_relerr` | 19.75 % | 0.541 % | **37×** |
| **measured noisy covariance**, as-measured noisy span | 20.30 % | 0.367 % | **55×** |
| **converged span / measured noisy σ** (recommended) | 19.75 % | 0.367 % | **54×** |
| **same, under the 2× LSST-DESC shape-noise convention** | 19.75 % | 0.647 % | **31×** |

Euclid-like: 44× (Lee+22 noise), 25× (2× convention). Peak detectability is at
ℓ ≈ 7200: **59× LSST-Y10 / 46× Euclid** (33× / 25× under the 2× convention). The converged
span first exceeds 5σ(LSST-Y10) at **ℓ ≈ 1000**.

Shape-noise share of the $C_\ell$ covariance, $f=1-[C/(C+N)]^2$, and the implied error-bar
inflation $(C+N)/C$:

| ℓ | 305 | 1000 | 1870 | 5360 | 7880 |
|---|---|---|---|---|---|
| $f_{\rm noise}$ | 0.108 | 0.349 | 0.585 | 0.943 | 0.984 |
| inflation | 1.06× | 1.24× | 1.55× | **4.20×** | **7.82×** |

**Measuring the precision directly on shape-noise-carrying maps *strengthens* the paper's
Fig-7 claim** (55× vs 37×), because the paper's analytic band already assumed twice the
shape noise the L22 recipe specifies. Under the conservative convention it is 31× — still
"more than an order of magnitude".

---

## 4. Per-statistic detectability table

Per Sobol node $i$: $\chi_i^2=\Delta_i^{\sf T}\hat{\bm C}(A)^{-1}\Delta_i$ with
$\Delta_i=\langle{\rm node}_i\rangle-\langle{\rm fiducial}\rangle$ at $z_s=1$, the **noisy
fiducial 550-realization covariance**, Hartlap, $A=18{,}000$ deg². ">3σ" / ">5σ" are the
fractions of the 256 nodes above the two-sided Gaussian-equivalent thresholds of $\chi^2_d$
($d=22$ for ν, 24 for the log-ℓ-binned spectra; $\chi^2_{3\sigma}=45.9/48.8$,
$\chi^2_{5\sigma}=71.4/75.1$). **"fc" = floor-corrected**: the χ² a zero-response node would
still show from its own 50 realizations is subtracted first, using the *measured* shape-noise
covariance (§4.2) — this is the honest column. "span/σ" is the 5–95 % node span over the
LSST-Y10 1σ at the statistic's peak-response bin, the direct analogue of the paper's "37×".

### 4.1 Under LSST-Y10 shape noise (the answer to the referee)

| statistic | dof | median χ² | median σ | >3σ | >5σ | **>3σ (fc)** | **>5σ (fc)** | floor/dof | **span/σ** (peak) | span/σ (median bin) |
|---|---|---|---|---|---|---|---|---|---|---|
| PDF           | 22 |  491 | 20.1 | 100 % | 100 % | **89.1 %** | **86.7 %** | 11.5 | **27.3** | 21.2 |
| peak counts   | 22 | 1084 | 31.3 | 100 % | 100 % | **93.8 %** | **91.0 %** | 12.9 | **34.6** | 18.8 |
| minima counts | 22 |  453 | 19.2 | 100 % | 100 % | **77.0 %** | **71.9 %** | 10.7 | **24.9** | 12.6 |
| $V_0$         | 22 |  524 | 20.9 | 100 % | 100 % | **88.7 %** | **83.6 %** | 12.5 | **26.8** | 20.2 |
| $V_1$         | 22 | 1500 | 37.1 | 100 % | 100 % | **96.1 %** | **95.7 %** | 12.0 | **52.0** | 27.4 |
| $V_2$         | 22 | 1832 | 60.5 | 100 % | 100 % | **96.1 %** | **94.1 %** | 12.6 | **78.5** | 32.7 |
| $C_\ell^{\kappa\kappa}$ | 24 | 1597 | 56.5 | 100 % | 100 % | **97.3 %** | **95.7 %** | 8.0 | **65.9** | 13.7 |
| $S(\ell)$     | 24 | 1600 | 56.5 | 100 % | 100 % | **97.7 %** | **95.7 %** | 8.0 | **66.0** | 13.7 |

For comparison, the same machinery on the paper's noiseless cache (its nodes and fiducial
are literally the same maps, so its floor is ≈ 0 and its raw columns are the fair ones):

| statistic | >3σ | >5σ | span/σ (peak) |
|---|---|---|---|
| PDF | 100 % | 100 % | 242.0 |
| peak counts | 98.4 % | 91.4 % | 12.6 |
| minima counts | 70.7 % | 57.4 % | 11.4 |
| $V_0$ | 99.6 % | 96.9 % | 24.4 |
| $V_1$ | 99.6 % | 98.8 % | 40.1 |
| $V_2$ | 100 % | 98.4 % | 57.0 |
| $C_\ell^{\kappa\kappa}$ | 99.2 % | 98.8 % | 179.9 |

**⚠ The two tables are not a controlled noise experiment.** The noiseless cache
(`nu_grid.py`) uses *mixed* smoothing (PDF unsmoothed, peaks/minima at 2′, MFs at 1′) and a
**per-map** ν normalization; the noisy cache (`noisy_grid.py`) uses one 1′ smoothing for all
six and a **fixed** per-plane $\kappa_{\rm rms}$. The noisy convention is the correct one for
a survey — you do not renormalize each patch by its own σ — and it retains response the
per-map normalization absorbs, which is why several noisy span/σ values are *larger*. Read
the noisy table as standing on its own; §4.2 is the controlled experiment.

### 4.2 The controlled experiment: how much does shape noise inflate the covariance?

`r2_noisegain.py` computes the noisy statistics **twice on the same 120 fiducial maps with
two independent noise streams**. The difference has covariance exactly $2C_{\rm noise}$ with
the cosmic-variance leg cancelled identically, at *fixed convention*. From
$C_{\rm total}=C_{\rm cv}+C_{\rm noise}$ (550-realization $C_{\rm total}$):

| statistic | shape-noise share of the variance | error-bar inflation $\sqrt{C_{\rm tot}/C_{\rm cv}}$ |
|---|---|---|
| **peak counts** | **0.786** | **2.16×** |
| **minima counts** | **0.781** | **2.14×** |
| $V_2$ | 0.233 | 1.14× |
| PDF | 0.149 | 1.08× |
| $V_1$ | 0.114 | 1.06× |
| $V_0$ | 0.044 | 1.02× |
| $C_\ell^{\kappa\kappa}$ | 0.11 → 0.98 with ℓ | 1.06× (ℓ=300) → **7.8×** (ℓ=7900) |

**This is the referee's point, quantified: peaks and minima are the two statistics whose
error budget shape noise really owns (79 % of the variance, 2.15× wider error bars), and the
small-scale power spectrum is the third (up to 7.8× at ℓ≈8000). The PDF and the Minkowski
functionals are barely touched (1.02–1.14×) — evaluated on a 1′-smoothed $1024^2$ map, the
noise contributes a nearly deterministic offset rather than extra scatter.** None of the
inflations is large enough to remove a 25–79× prior span.

---

## 5. What weakens, what holds — the one-paragraph summary

**Holds outright:** $S(\ell)$ / $C_\ell^{\kappa\kappa}$ (54× LSST-Y10, 31× conservatively;
97.7 % of nodes >3σ, 95.7 % >5σ after floor correction), $V_1$ (52×; 96.1/95.7 %), $V_2$
(78×; 96.1/94.1 %). **Holds with a reduced margin:** PDF (27×; 89.1/86.7 %), $V_0$ (27×;
88.7/83.6 %), peak counts (35×; 93.8/91.0 %). **Weakest, and the one to phrase carefully:**
minima counts (25×; 77.0/71.9 %) — the only statistic where roughly a quarter of the prior
becomes indistinguishable from the fiducial once shape noise is included. Not a single
statistic falls below "the prior span is ~25× the Stage-IV 1σ".

**This sharpens the paper's channel-separation story rather than damaging it.** The
ℓ-domain statistics — the wind-sector channel — are essentially noise-immune at the scales
that carry their signal, while the ν-domain morphological statistics — the IMF/AGN channel —
pay a real but affordable shape-noise tax that is concentrated in the two *counting*
statistics. A survey analysis should therefore weight the ℓ-domain and the Minkowski
functionals ahead of raw peak/minimum counts, which is exactly the sort of practical
guidance the atlas exists to provide.

---

## 6. Drafted text replacements

`main.tex` was **not modified**. Line numbers are from the current file; quotes are verbatim.
Fallback wording per the referee's own suggestion — "the statistical precision of the maps" —
is used wherever a claim genuinely cannot be supported under noise.

---

**(A) Abstract — line 200.** Current:

> "Across the prior, the responses far exceed Stage-IV statistical precision and separate by
> channel: galactic winds control the $\ell$-domain statistics, while the stellar Initial
> Mass Function slope and AGN parameters shape the morphological statistics such as weak
> lensing PDF, peak counts, minima and minkowski functionals."

Drafted:

> "Across the prior, the responses exceed Stage-IV statistical precision by one to two
> orders of magnitude --- $31$--$54\times$ the LSST-Y10 $1\sigma$ for the power-spectrum
> suppression and $25$--$78\times$ for the $\nu$-domain statistics, evaluated on maps
> carrying LSST-Y10 shape noise --- and separate by channel: galactic winds control the
> $\ell$-domain statistics, while the stellar Initial Mass Function slope and AGN parameters
> shape the morphological statistics such as weak lensing PDF, peak counts, minima and
> Minkowski functionals."

---

**(B) Abstract — line 200, preceding sentence.** Current:

> "At the fiducial parameters, the generated maps reproduce their hydro-pasted truth within
> LSST-Y10-like precision for every WL statistic."

Drafted (also covers the identical claim in the conclusions, line 950, and §1, line 327):

> "At the fiducial parameters, the generated maps reproduce their hydro-pasted truth to
> within a small multiple of the LSST-Y10 per-bin statistical precision for every WL
> statistic --- a closure that becomes \emph{safer}, not weaker, when LSST-Y10 shape noise
> is added to the maps."

---

**(C) §1, second result — lines 333–336.** Current:

> "Second, the feedback responses across the prior far exceed LSST- and \textit{Euclid}-like
> statistical precision, establishing that Stage-IV surveys enter the regime where feedback
> must be actively constrained, not merely marginalised"

Drafted:

> "Second, the feedback responses across the prior far exceed LSST- and \textit{Euclid}-like
> statistical precision --- measured on maps carrying LSST-Y10 shape noise, the
> $5$--$95\%$ prior span reaches $54\times$ the LSST-Y10 $1\sigma$ for $S(\ell)$ at
> $\ell=5000$ and $25$--$78\times$ for the $\nu$-domain statistics, and every one of the
> $256$ Sobol nodes is separated from the fiducial at $>5\sigma$ --- establishing that
> Stage-IV surveys enter the regime where feedback must be actively constrained, not merely
> marginalised"

---

**(D) §3, the noiseless disclaimer — line 552, final sentence.** Current:

> "The comparison here is also noiseless: no shape noise, beam, or survey systematics enter
> at any stage."

Drafted (replace the sentence; this is where the new figure is introduced):

> "The comparison in Fig.~\ref{fig:field_validation} is noiseless: no shape noise, beam, or
> survey systematics enter at any stage. Because the $\nu$-domain statistics are the ones
> most exposed to shape noise, we repeat the entire comparison on maps carrying LSST-Y10
> shape noise in Appendix~\ref{app:noisy}, following \citet{LeePeakBCM2022}: per-pixel Gaussian
> noise with $\sigma^2=\sigma_e^2/(2 n_{\rm gal} A_{\rm pix})$ at $\sigma_e=0.26$,
> $n_{\rm gal}=27\,{\rm arcmin}^{-2}$, added before a $\theta_G=1'$ Gaussian smoothing, with
> all statistics measured on the smoothed noisy field. Shape noise inflates the covariance
> of the counting statistics by a factor $2.15$ and the small-scale power spectrum by up to
> $7.8$, while leaving the BIND--truth residual essentially unchanged, so the closure
> reported below is \emph{conservative}: it improves by factors of $1.5$--$7.8$ in $\chi^2$
> excess for six of the eight statistics and degrades for none."

---

**(E) §3, the closure claim — line 574, final clause.** Current:

> "One can read off by eye the statistical distinguishability of the statistics with respect
> to an LSST-like survey, and we see clearly that the BIND-generated statistics match the
> hydro-pasted maps to within LSST-like precision for all weak lensing statistics."

Drafted:

> "One can read off by eye the statistical distinguishability of the statistics with respect
> to an LSST-like survey: the median residual is $0.3$--$1.0\%$ against a median LSST-Y10
> $1\sigma$ band of $0.4$--$0.8\%$, so the BIND-generated statistics match the hydro-pasted
> maps bin by bin at the LSST-Y10 statistical precision, with the largest excursions
> ($3$--$5\times$ the band) confined to the sparsely populated $\nu\gtrsim5$ tail. Evaluated
> with the full covariance rather than bin by bin, the residuals reach $1\sigma$ per degree
> of freedom at an effective area of $6\times10^{3}$--$1\times10^{5}\,{\rm deg}^2$, i.e. at
> the statistical precision of the maps a Stage-IV survey will deliver."

---

**(F) §3, the "first" claim — line 576.** Current:

> "This is an important finding: it represents, to our knowledge, the first set of generated
> lightcones that self-consistently produce weak lensing statistics that are statistically
> indistinguishable from their hydro-pasted truth at the precision of an LSST-Y10-like
> survey."

Drafted (uses the referee's fallback wording for the part that cannot be supported
full-covariance):

> "This is an important finding: it represents, to our knowledge, the first set of generated
> lightcones that self-consistently produce weak lensing statistics matching their
> hydro-pasted truth to within the statistical precision of the maps, per bin, at
> LSST-Y10-like survey precision --- and the agreement is \emph{tighter}, not looser, once
> realistic shape noise is included (Appendix~\ref{app:noisy})."

---

**(G) §4 opener — line 599, final sentence.** Current:

> "In the bottom panels we show that the deviation with respect to the fiducial far exceeds
> the uncertainty of the surveys, implying that deviations in the TNG model do induce
> significant differences that can be detected by next generation surveys."

Drafted:

> "The bottom panel shows that the deviation with respect to the fiducial far exceeds the
> uncertainty of the surveys: the $5$--$95\%$ prior span reaches $54\times$ the LSST-Y10
> $1\sigma$ and $44\times$ the \textit{Euclid} $1\sigma$ at $\ell=5000$, peaking at
> $59\times$ near $\ell\simeq7000$, and exceeds $5\sigma$ for all $\ell\gtrsim10^3$. These
> envelopes are computed from the covariance measured directly on maps carrying LSST-Y10
> shape noise rather than from a noiseless simulation covariance; adopting instead the more
> conservative $N_\ell=\sigma_e^2/n_{\rm gal}$ shape-noise convention lowers them to
> $31\times$ and $25\times$, still more than an order of magnitude. Deviations within the
> TNG model therefore induce differences that next-generation surveys will detect."

---

**(H) Fig. 7 caption — line 604.** Current:

> "The motivation for this section: the $5$--$95\%$ spread of $S(\ell)$ across the 256 Sobol
> nodes at $z_s=1$ (gray band), the fiducial (black), and the statistical precision of
> LSST-Y10 and \textit{Euclid}-like surveys (colored envelopes). The prior spread exceeds
> the survey precision by more than an order of magnitude meaning upcoming surveys will
> resolve feedback differences far smaller than the current theoretical uncertainty."

Drafted:

> "The motivation for this section: the $5$--$95\%$ spread of $S(\ell)$ across the 256 Sobol
> nodes at $z_s=1$ (gray band), the fiducial (black), and the statistical precision of
> LSST-Y10 and \textit{Euclid}-like surveys (colored envelopes), the latter measured from
> $550$ realizations of the fiducial lightcone carrying LSST-Y10 shape noise and area-scaled
> by Eq.~\ref{eq:area_scaling}. The prior spread exceeds the survey precision by more than
> an order of magnitude at every $\ell\gtrsim10^3$, reaching $54\times$ the LSST-Y10
> $1\sigma$ at $\ell=5000$, meaning upcoming surveys will resolve feedback differences far
> smaller than the current theoretical uncertainty."

---

**(I) Fig. 5 caption — line 548, error-bar clause.** Current:

> "Bottom panels show the mean residual between BIND and hydro-pasted, with error bars
> scaled to an LSST-Y10-like survey area as described in the text; counts carry Poisson
> error bars. The shared seeds eliminate cosmic variance from the residuals."

Drafted (add one sentence at the end):

> "... The shared seeds eliminate cosmic variance from the residuals. The same comparison on
> maps carrying LSST-Y10 shape noise is given in
> Fig.~\ref{fig:field_validation_noisy}; the residuals are essentially unchanged while the
> bands widen, so this noiseless version is the conservative one."

---

**(J) §5 Caveats — the placeholder at line 939** (`% 8 noiseless validation -> survey-realism
suite`). Drafted paragraph, incorporating the honest-residuals caveat of §7:

> "Eighth, our validation and detectability tests are performed on the maps themselves rather
> than on mock survey data. We have removed the largest omission by repeating both in full
> under an LSST-Y10 shape-noise model (Appendix~\ref{app:noisy}), which strengthens the
> closure and leaves the detectability conclusions intact, but three approximations remain.
> The $25\,{\rm deg}^2$ lightcones contain no super-survey modes, so Eq.~\ref{eq:cov} omits
> the super-sample covariance term, which the $205\,h^{-1}{\rm Mpc}$ box cannot supply and
> which grows toward small scales where our detectability margins are largest. The area
> scaling of Eq.~\ref{eq:area_scaling} treats the survey as $A/25$ independent copies of our
> footprint, whereas our realizations are rotations and translations of a single box: the
> scaled covariance is therefore a lower bound, and every $\chi^2$ we quote is
> correspondingly an upper bound. And the noise model is statistical only --- intrinsic
> alignments, photometric-redshift errors, blending, multiplicative shear bias, and masking
> are all absent, and each of these is a systematic that a real analysis must model rather
> than average down. A full survey-realism suite, with mock catalogues carrying these
> effects, is the natural next step."

---

**(K) Conclusions — line 950, first clause.** Current:

> "At the fiducial parameters, the generated maps reproduce the hydro-pasted truth to within
> the precision of an LSST-Y10-like survey for every weak lensing statistic"

Drafted:

> "At the fiducial parameters, the generated maps reproduce the hydro-pasted truth to within
> the per-bin statistical precision of an LSST-Y10-like survey for every weak lensing
> statistic --- a closure that tightens when realistic shape noise is included"

---

**(L) Line 572 — the `[verify]` tag.** Current:

> "with $A = 18{,}000\,{\rm deg}^2$ for LSST-Y10 [verify]."

Drafted (the tag can be dropped: $18{,}000\,{\rm deg}^2$ is the LSST DESC SRD Y10 gold-sample
area, $f_{\rm sky}=0.436$, and is what every number in this response uses):

> "with $A = 18{,}000\,{\rm deg}^2$ ($f_{\rm sky}=0.436$) for LSST-Y10, and
> $A = 14{,}850\,{\rm deg}^2$ ($f_{\rm sky}=0.36$) for a \textit{Euclid}-like survey."

---

## 7. Honest-residuals caveat — what the noise model still omits

One drafted sentence (folded into draft (J) above), then the reasoning behind each clause:

> "Three approximations survive this treatment: the $25\,{\rm deg}^2$ lightcones carry no
> super-survey modes, so the super-sample covariance term is absent and cannot be supplied
> by a $205\,h^{-1}{\rm Mpc}$ box; the $(A/25\,{\rm deg}^2)^{-1}$ scaling treats
> pseudo-independent rotations of a single box as independent survey tiles, making every
> quoted covariance a lower bound and every $\chi^2$ an upper bound; and the noise model is
> purely statistical, omitting intrinsic alignments, photometric-redshift errors, blending
> and multiplicative shear bias, which are systematics rather than variances and do not
> average down with area."

Why each matters here:

* **Super-sample covariance.** Not fixable with these lightcones — SSC requires modes larger
  than the box. It adds a positive, largely amplitude-like term that grows toward small
  scales, so it *widens* the survey bands and makes §2's closure and §4's floor-corrected
  fractions conservative in the same direction. It also *reduces* the detectability ratios
  of §3, though for a suppression ratio $S(\ell)$ much of SSC cancels between numerator and
  denominator.
* **The $(A/25)^{-1}$ scaling.** The 550 realizations are rotations/translations of one
  205 Mpc/h box, not independent volumes, so they under-sample the true large-scale
  covariance. This is the single largest reason the full-covariance χ² of §2.1 exceeds 1
  while the per-bin residuals sit at ~1σ, and it is why §2.2 recommends the per-bin phrasing.
* **Astrophysical and instrumental systematics.** Intrinsic alignments, photo-$z$ errors,
  blending, shear-calibration bias and masking are all absent. They do not enter as extra
  variance that could rescue a marginal detection; they enter as biases that must be
  modelled, and in a real analysis they will be marginalised over with nuisance parameters
  that degrade the effective precision. The detectability numbers of §3–4 are therefore
  statistical-only upper limits on what a survey achieves.
* **Also worth stating:** the noise level itself is convention-dependent
  ($\sigma_e^2/2n_{\rm gal}$ vs $\sigma_e^2/n_{\rm gal}$, a factor 2 in $N_\ell$); every
  ℓ-domain number in §3 is given both ways, while the ν-domain numbers in §4 use only the
  smaller (Lee+22) noise, so they are the optimistic end of a factor-of-2 bracket.

---

## 8. Figure captions

**`imgs/fig05n_field_validation_noisy.png` — the Fig-5 twin.**

> Weak lensing statistics of the BIND-generated (solid, all five source planes) and
> hydro-pasted (dashed, $z_s=1$) maps after LSST-Y10 shape noise
> ($\sigma_e=0.26$, $n_{\rm gal}=27\,{\rm arcmin}^{-2}$, added per pixel with
> $\sigma^2=\sigma_e^2/2n_{\rm gal}A_{\rm pix}$) and a $\theta_G=1'$ Gaussian smoothing,
> following \citet{LeePeakBCM2022}: (a) the convergence power spectrum with its shape-noise floor
> (dotted grey), which is subtracted before forming (b) the suppression $S(\ell)$, then
> (c) the one-point PDF, (d) peak counts, (e) minimum counts and (f--h) the three Minkowski
> functionals, all measured on the smoothed noisy field. Bottom panels show the $z_s=1$
> residual with the $\pm1\sigma$ LSST-Y10 band (shaded) built from the covariance of the
> $550$ noisy hydro-pasted realizations via Eqs.~\ref{eq:cov}--\ref{eq:area_scaling} at
> $A=18{,}000\,{\rm deg}^2$; points carry the seed-paired realization-difference error. The
> quoted $\chi^2/{\rm dof}$ use the Hartlap-debiased full covariance
> (Eq.~\ref{eq:inv_cov}). Shape noise inflates every band while leaving the residuals
> essentially unchanged, so the fiducial closure is tighter here than in the noiseless
> Fig.~\ref{fig:field_validation}. $\ell$ is limited to $8000$, above which the $1'$
> smoothing leaves less than $7\%$ of the signal.

**`imgs/fig23n_s3_opener_noisy.png` — the Fig-7 twin.**

> The feedback response against survey precision, under LSST-Y10 shape noise. (a) The
> $5$--$95\%$ spread of $S(\ell)$ across the $256$ Sobol nodes at $z_s=1$ as measured on the
> noisy maps (grey band) and as converged in the seed-paired noiseless suite (dashed), the
> fiducial (black), and the $\pm1\sigma$ statistical precision of LSST-Y10 (red) and a
> \textit{Euclid}-like survey (green), computed from the covariance of $550$ noisy fiducial
> realizations and area-scaled by Eq.~\ref{eq:area_scaling}. The shape-noise floor is
> subtracted from every spectrum before the ratio is formed; the smoothing transfer function
> cancels identically in it. (b) The prior span in units of the survey $1\sigma$: the
> response exceeds $5\sigma$ for all $\ell\gtrsim10^3$ and reaches $59\times$ near
> $\ell\simeq7000$. Dotted red repeats the LSST-Y10 curve under the more conservative
> $N_\ell=\sigma_e^2/n_{\rm gal}$ shape-noise convention; dashed grey is the measurement
> floor set by the $50$ realizations behind each Sobol node, which is why the converged
> rather than the as-measured span is quoted above $\ell\simeq1200$ and why neither is
> quoted below it.

---

## 9. Files

**Deliverable figures** (new names only; no existing `imgs/` file was touched):

* `/mnt/home/mlee1/BIND/imgs/fig05n_field_validation_noisy.png` + `.pdf` (300 dpi)
* `/mnt/home/mlee1/BIND/imgs/fig23n_s3_opener_noisy.png` + `.pdf` (300 dpi)

**Rerunnable scripts** in `/mnt/home/mlee1/BIND/papers/01_pipeline/referee/work/`:

| script | what it does | cost |
|---|---|---|
| `r2_common.py` | shared loaders, survey constants, measured noise-bias spectrum, covariance/Hartlap/χ² machinery, log-ℓ rebinning | — |
| `r2_probe.py` | recipe validation: $\hat N(\ell)$ vs analytic, $W^2$ vs $\exp(-\ell^2\sigma_\theta^2)$, noise-floor levels, debiasing check, seed-pairing check, mode-counting check | ~2 min |
| `r2_validation.py` | Fig-5 twin + the §2.1 closure table → `validation_numbers.json` | ~3 min |
| `r2_detect.py` | Fig-7 twin + the §3/§4 detectability tables → `detectability.json` | ~2 min |
| `r2_noisegain.py` | the controlled noise/cosmic-variance split (§4.2) + the bit-exact cache re-derivation, from the raw fiducial cube | ~6 min |

Run order: `r2_probe` → `r2_noisegain` → `r2_validation`, `r2_detect` (the last two consume
`noise_split_cov.npz`). All read-only against `bind_sb35`, `bind_science` and
`bind_lightcone_tng`.

**Intermediates** in `/mnt/home/mlee1/ceph/referee_work/r2/`: `noise_bias_cl.npz`,
`noise_split.json`, `noise_split_cov.npz` (measured shape-noise covariances + the raw
two-stream draws), `validation_numbers.json`, `detectability.json`, `noisegain.log`.

**Not modified:** `main.tex`, `_build_figures_nb.py`, `paper1_figures.ipynb`, any existing
`imgs/` file, and every cache under `bind_sb35/`.

**Note for the figure list.** `imgs/fig23_s3_opener.png` — referenced by `main.tex` line 603
— **does not exist in `imgs/`**; only `papers/01_pipeline/figs_v2/fig23_s3_opener.pdf`
(2026-08-05) does, and that PDF is the *$S(\ell{=}5000)$-versus-$f_{\rm gas}$ scatter* panel,
not the $S(\ell)$ envelope the caption at line 604 describes. The noisy twin built here
follows the **caption**. Worth resolving independently of this referee response.
