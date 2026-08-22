# Fig 5 / fig 5a significance methodology — the Spearman response grid and its null ladder

**What this documents.** The complete calculation behind `figs_v2/fig05_param_response`
(the 12-statistic × 30-parameter signed-peak response map — each cell painted with the
signed ρ at its peak-|ρ| bin; paper `fig:corr_matrix`) and
`figs_v2/fig05a_sl_response` (the bin-resolved signed Spearman matrix for $S(\ell)$; paper
`fig:srow`): how the correlation grid is measured, how it is collapsed to one number per
(statistic, parameter) cell, and how the four significance thresholds — the analytic
single-bin null, the **per-statistic permutation null**, the **per-parameter
look-elsewhere null**, and the **whole-grid look-elsewhere null** — are constructed.
Every equation matches the implementation line-for-line.

**Provenance.** Code: the fig-5a cell (`_build_figures_nb.py`, the cell saving
`figs_v2/fig05a_sl_response`) defines the statistic table, the Nyquist cut, and the
Spearman grids; the fig-5 cell (saving `figs_v2/fig05_param_response`) builds the
permutation nulls and the map. The fig-5a cell is the single source of truth: fig 5's
$S(\ell)$ row is *asserted* (not assumed) to equal the column-max of fig 5a's matrix.
Data: `bind_sb35/emulator_dataset_nu05.npz`. Permutation seed:
`np.random.default_rng(4)`, 200 iterations. All numbers quoted below are from the
2026-08-10 render of that dataset (**N = 256 runs**, ℓ-domain scan Nyquist-cut per the
same-day author ruling) and are reproduced deterministically by re-running the notebook.

A plain-language primer opens the document; the numbered sections carry the exact math.

---

## The idea in words — a plain-language primer

*Each step below is "what could fool us here, and what we do about it." The formal
version of every claim lives in the numbered sections; a step → section map closes the
primer.*

**The question.** 256 simulations, each with 30 knobs set to scattered positions (the
Sobol design), each producing 12 measured curves (spectra over $\ell$, histograms over
$\nu$). We want one honest map: **which knobs actually move which observables** — where
"honest" means a cell lights up only if chance alone wouldn't have produced it.

**Step 1 — Correlate ranks, not values.** For one knob and one bin we have 256 pairs
(knob setting, measured value). Replace each side by its *rank* (1st…256th in sorted
order) and correlate the ranks — Spearman's $\rho$. Three immunities come free: (i)
*units and transformations can't matter* — ranks are unchanged by any monotone map, so
log-vs-linear priors and normalization conventions are irrelevant; (ii) *outliers can't
dominate* — a pathological run can be "1st place" but never 10× first place; (iii) *it
tests the right question* — not linearity, just whether turning the knob up
systematically pushes the observable up or down.

**Step 2 — Curate which bins are allowed to compete.** Because of the max in Step 3,
one principle matters: **a maximum gravitates to the worst bins if you let them in** —
whatever bins fluctuate hardest or are least trustworthy are disproportionately likely
to win. So before correlating anything: (i) the **Nyquist cut** — bins above
$\ell_{\rm Ny}=36864$ are the FFT's corner zone (diagonal modes only, no full ring to
average over); we never plot them, so we don't let them win the max either — the scan
and the display now use the same limit; (ii) **pre-averaging ×16** — a single raw
$\ell$ bin is shot-noisy, and noise reshuffles rankings, so we average 16 neighbours
until each bin's run-to-run ordering reflects signal (the $\nu$ histograms skip this:
their $\Delta\nu=0.5$ grid was *designed* well-populated); (iii) a **stability check** —
the notebook re-runs all rankings with the averaging halved and doubled and prints that
the top levers don't move, so the "16" isn't doing the science.

**Step 3 — The collapse, and the trap it sets.** Each (knob, statistic) pair is a whole
curve of $\rho$ over bins; the map keeps the correlation at its **peak-$|\rho|$ bin**:
"at its best scale, how hard does this knob move this observable?" — painted *with its
sign* (red: the knob raises the observable, blue: lowers it), so the direction of the
effect survives the collapse; significance is always judged on the magnitude. The right
summary — but a best-of-many is *biased high*. A knob that does nothing still gets 31 lottery tickets when you scan 31
bins, and you report its luckiest one. That is the **look-elsewhere effect**; everything
after this point is about raising the bar enough to un-bias it.

**Step 4 — The baseline bar.** For a single bin chosen *in advance*, a do-nothing knob
gives $\rho$ scattered around zero with spread $\approx 1/\sqrt{N}$ (the correlation
averages 256 products of independent noise; averaging beats noise down by $\sqrt N$).
The 2σ "could easily be nothing" level is $2/\sqrt{256}=0.125$ — valid for fig 5a's
individual bins, never for a fig-5 cell, which is already a max.

**Step 5 — Why the textbook correction fails.** Bonferroni's fix for "I looked in $k$
places" assumes the $k$ looks are independent. Ours aren't, and unevenly so: scanning a
smooth spectrum's 31 bins is like asking the same friend the same question 31 times
(~3 effectively independent looks), while the peak-count histogram's 22 bins are more
like 14 different friends. Correcting by the nominal counts sets the bar far too high
(crosses out real physics); not correcting sets it too low (promotes noise); and the
right correction differs per statistic and is unknown a priori. So we don't derive the
bar — we **measure** it.

**Step 6 — Manufacture the "nothing is real" world.** Shuffle which parameter vector is
paired with which simulation output — re-deal the deck. If knobs truly don't matter,
every pairing is equally plausible, so the shuffled datasets are statistically
indistinguishable from the real one *except* that any true knob→observable connection is
severed. The shuffle **destroys** only the pairing being tested; it **preserves** every
knob's exact set of values (identical ranks), all correlations *between bins*, and all
correlations *between statistics* — precisely the structure that broke Bonferroni — and
the fake worlds run through the **identical pipeline** (same cut, same averaging, same
masking, same max). 200 seeded shuffles give 200 snapshots of what our exact analysis
produces from pure chance. One deliberate subtlety: within each shuffle, **all 12
statistics see the same re-deal**, because the family-level bars below take maxima
across statistics, and the statistics are strongly correlated (peaks, minima, MFs come
from the same maps) — independent shuffles would fake 12 independent chances that don't
exist and over-correct.

**Step 7 — Three bars for three sizes of claim.** The bar depends on how much searching
preceded the number:

1. *"I pre-chose this knob and this statistic, then scanned its bins"* → that
   statistic's own null (95th percentile of its fake-world maxima): 0.149–0.182 —
   barely above 0.125 for spectra (~3 effective bins), higher for peak counts (~14).
   Cells below it get the **×**.
2. *"Does this knob do anything, anywhere?"* (scanned all 12 statistics) → max across
   the 12 rows in each fake world: 0.200. The **dashed divider**: 11 of 30 knobs clear
   it, the other 19 are statistically inert everywhere. Only modestly above level 1 —
   12 correlated statistics are nowhere near 12 independent chances.
3. *"I scanned the entire grid and this cell popped"* → the single biggest value
   anywhere in each fake world: 0.253. Chance produces a cell that big in fewer than
   1 in 20 *complete searches*. Those cells get the **printed numerals**.

Sanity check for free: bigger searches always have luckier bests, so the bars must come
out ordered $0.125 \le 0.149$–$0.182 \le 0.200 \le 0.253$ — and they do.

**The noise checklist** (threat → countermeasure):

| Threat | Countermeasure |
|---|---|
| Units / log-prior choices influencing results | ranks (invariant) |
| One extreme run dominating | ranks (bounded influence) |
| Bin-level measurement noise scrambling orderings | ×16 pre-averaging; well-populated ν grid; stability check |
| Untrustworthy corner-mode bins winning the max | Nyquist cut (scan = display) |
| Max-over-bins biased high | per-statistic permutation null |
| Scanning 12 statistics | per-parameter null (max over rows, same shuffle) |
| Scanning the whole grid | whole-grid null |
| Correlated bins/statistics breaking analytic corrections | permutation preserves the correlations; the bars adapt automatically |
| Missing bins manufacturing rank structure | pairwise-finite masking, no zero-fill, identical in the null |
| The null itself being noisy (200 shuffles) | pooled draws (6000 for levels 1–2), seeded, precision ~third decimal |

**The spoken version.**

> "We rank-correlate each parameter with each statistic bin by bin — ranks so that
> units, priors, and outliers can't matter — and summarize each pair by its best
> correlation across bins, keeping its sign, so the map's red/blue is the direction of
> the effect. A best-of-many is biased high, so we calibrate it by
> shuffling which parameter vector belongs to which simulation: that severs any real
> connection while preserving every correlation among bins and statistics, giving us
> the exact distribution of 'best correlations' our own pipeline produces from pure
> chance. From those shuffles we read off three bars for three sizes of search — one
> statistic's bins, all twelve statistics for one parameter, and the entire grid — and
> every cell in the figure is marked by which bar it clears. Eleven of thirty
> parameters clear the middle bar; the strongest cells clear the whole-grid bar,
> meaning a full search of nothing-but-noise would produce them less than five percent
> of the time."

**Two questions you will be asked.** (i) *"Why is the whole-grid bar (0.253) so close
to the single-bin bar (0.125) when you scanned ~9500 numbers?"* Because the search is
massively redundant — there are only a few dozen effectively independent looks in it,
and the permutation measures that automatically. (ii) *"Why max over bins and not the
mean?"* The mean dilutes a real response localized at particular scales; and the
notebook prints the check that ranking by mean instead reorders almost nothing (rank
agreement 0.906 with the plotted order).

**Map to the formal sections.** Step 1 → §2; Step 2 → §1 (§1a Nyquist cut, §1b
pre-averaging; note the pipeline applies Step 2 before Step 1 chronologically); Step 3 →
§3; Step 4 → §4.0; Step 5 → §4.1; Step 6 → §4.2; Step 7 → §4.3–4.6; the marks key → §5;
the fine print → §6; code-name ↔ math map → §7.

---

## 0. Inputs and notation

| symbol | meaning | value here |
|---|---|---|
| $N$ | completed Sobol runs (rows of the design) | 256 |
| $P$ | astrophysical parameters (columns) | 30 |
| $S$ | canonical statistics (rows of fig 5) | 12 |
| $\theta_j^{(i)}$ | unit-cube value of parameter $j$ in run $i$ | `X_unit`, shape (256, 30) |
| $y_{s,b}^{(i)}$ | statistic $s$, pre-averaged bin $b$, run $i$ | `Ys[k]` |
| $B_s$ | number of scanned bins of statistic $s$ | 31 (ℓ-domain) or 22 (ν-domain) |
| $T$ | permutation iterations | 200 |

**The design matrix.** `X_unit` maps each parameter's SB35 prior to $[0,1]$, taking
$\log_{10}$ first where the prior is logarithmic (`LogFlag`). Because everything below is
computed on **ranks**, and rank correlation is invariant under any strictly monotone map of
either variable, neither the unit-cube convention nor the log mapping affects a single
number in this pipeline — the choice is cosmetic.

**The 12 canonical statistics** (row order = fig 4 / fig 4b order; $z_s=1$ plane, index
`ZI=1` of the source planes $z_s \in \{0.5, 1.0, 1.5, 2.0, 2.44\}$, wherever the statistic
is tomographic; $y$ and $\tau$ maps are full-lightcone integrated columns and carry no
source-plane axis):

| row | statistic | domain | native bins | after Nyquist cut (§1a) | pre-average `rb` (§1b) | scanned bins $B_s$ |
|---|---|---|---|---|---|---|
| 1 | $S(\ell)$ suppression | $\ell$ | 724 ($\ell = 87$–$52134$) | 511 ($\ell \le 36864$) | 16 | 31 |
| 2 | PDF$(\nu)$ | $\nu$ | 22 ($\Delta\nu = 0.5$, centers $-2.75$…$+7.75$) | — | 1 | 22 |
| 3 | peak counts $N_{\rm pk}(\nu)$ | $\nu$ | 22 | — | 1 | 22 |
| 4 | minima counts $N_{\rm min}(\nu)$ | $\nu$ | 22 | — | 1 | 22 |
| 5–7 | Minkowski $V_0, V_1, V_2$ | $\nu$ | 22 each | — | 1 | 22 |
| 8 | $C_\ell^{yy}$ | $\ell$ | 724 | 511 | 16 | 31 |
| 9 | $C_\ell^{\tau\tau}$ | $\ell$ | 724 | 511 | 16 | 31 |
| 10 | $C_\ell^{\kappa y}$ | $\ell$ | 724 | 511 | 16 | 31 |
| 11 | $C_\ell^{\kappa\tau}$ | $\ell$ | 724 | 511 | 16 | 31 |
| 12 | $C_\ell^{y\tau}$ | $\ell$ | 724 | 511 | 16 | 31 |

$C_\ell^{\kappa\kappa}$ is deliberately **not** a row: at fixed cosmology it equals
$S(\ell)$ times a single run-independent DMO trace, so it would duplicate row 1.

The total number of scanned (parameter, statistic, bin) triples is
$P \times \sum_s B_s = 30 \times (6\cdot31 + 6\cdot22) = 30 \times 318 = 9\,540$ —
the size of the search that the look-elsewhere ladder of §4 exists to calibrate.

---

## 1. Step 1 — the scanned bin grid

### 1a. The Nyquist cut (ℓ-domain rows only; author ruling 2026-08-10)

The maps are $1024^2$ pixels over a $5\times5\,$deg footprint, so the largest multipole
with full azimuthal support is the axis-Nyquist mode

$$
\ell_{\rm Ny} \;=\; \frac{\pi}{\theta_{\rm pix}}
\;=\; \frac{180^\circ \cdot N_{\rm pix}}{\theta_{\rm map}}
\;=\; \frac{180 \times 1024}{5} \;=\; 36\,864 .
$$

The measured grid extends beyond it, to the FFT corner mode
$\sqrt{2}\,\ell_{\rm Ny} = 52\,134$ — but the $\ell \in (36864,\, 52134]$ zone contains
only direction-sparse diagonal modes with no azimuthal-averaging support. Every $C_\ell$
panel in the notebook already *displays* only to $\ell_{\rm Ny}$ (`ELL_MAX_PLOT`); the
2026-08-10 ruling makes the *ranking* consistent with the display:

$$
\text{scan set} \;=\; \{\,\ell_m : \ell_m \le \ell_{\rm Ny}\,\}
\qquad\text{(511 of the 724 native bins; last kept center } \ell = 36\,826\text{)} .
$$

In code this is the boolean mask `LNYQ = ELL <= ELL_MAX_PLOT`, applied by `_ranked(s)`
wherever the ranking machinery consumes an ℓ-domain array (the `Ys` construction and the
rebin-stability check). It is deliberately **not** applied to `STATS[..]["A"]` itself:
fig 7 plots those raw curves under its own, *tighter* display mask
(`ELL_TRUST` $= 3\times10^4$, the measured CIC/pixel-upturn onset — see §6.6). Before
this ruling the scan ran the full grid, and several columns' peak-|ρ| bins landed beyond
Nyquist (the resolved validity flag in `main.tex`).

The ν-domain rows have no analogue of this cut (their 22-bin $\Delta\nu=0.5$ grid is the
canonical measurement grid in full).

### 1b. Bin pre-averaging

Raw single-$\ell$ bandpowers are shot-noisy, and a rank correlation computed on a noisy bin
is unstable. Each Nyquist-cut ℓ-domain statistic is therefore block-averaged by a factor
$k=16$ before anything else happens. With kept bins $y^{\rm raw}_1, \dots, y^{\rm raw}_n$
(here $n = 511$) and block $\mathcal{B}_b = \{(b-1)k+1, \dots, bk\}$:

$$
y_{b} \;=\; \frac{1}{\lvert \mathcal{B}_b \cap \mathcal{F} \rvert}
\sum_{m \,\in\, \mathcal{B}_b \cap \mathcal{F}} y^{\rm raw}_m,
\qquad b = 1, \dots, \lfloor n/k \rfloor ,
$$

where $\mathcal{F}$ is the set of finite (non-NaN) bins (`rebin_nan`: a pre-averaged bin
is NaN only if its *entire* block is NaN; the remainder $n - k\lfloor n/k\rfloor$ is
trimmed). For the ℓ rows: $511 \to 31$ scanned bins (15 native bins at the top,
$\ell \approx 35\,270$–$36\,826$, trimmed as an incomplete block), pre-averaged centers
$\ell = 637$–$35\,206$. The same is done to the axis:
$\bar\ell_b = \tfrac{1}{k}\sum_{m\in\mathcal{B}_b} \ell_m$.

The six ν-domain rows use $k=1$ (identity): the canonical $\Delta\nu = 0.5$ grid
(`nu_grid.py`, 22 bins) was built so that every bin already holds a well-populated,
realization-summed integer count — the same property fig 4 uses for its Poisson error
bars — so no further averaging is needed. Stability of the final ranking under halving or
doubling every `rb` factor is *verified by a printed check* in the notebook (which applies
the same Nyquist cut), not assumed.

---

## 2. Step 2 — the signed Spearman matrix

For every statistic $s$, parameter $j$, and scanned bin $b$, the pipeline computes the
Spearman rank correlation across runs. Let
$r^{\theta}_i = \operatorname{rank}\!\big(\theta_j^{(i)}\big)$ and
$r^{y}_i = \operatorname{rank}\!\big(y_{s,b}^{(i)}\big)$ be the ranks of the $N$ values
(`scipy.stats.rankdata`, average ranks on ties — ties never occur in the Sobol design and
have measure zero in the continuous statistics). Then, exactly as implemented
(standardize the two rank vectors with population moments and take the mean product):

$$
\rho_{s,j,b} \;=\; \frac{1}{N}\sum_{i=1}^{N}
\left(\frac{r^{\theta}_i - \bar r^{\theta}}{\sigma_{r^{\theta}}}\right)
\left(\frac{r^{y}_i - \bar r^{y}}{\sigma_{r^{y}}}\right),
\qquad
\bar r = \frac{1}{N}\sum_i r_i,\quad
\sigma_r^2 = \frac{1}{N}\sum_i (r_i - \bar r)^2 .
$$

This is the Pearson correlation of the rank vectors; with no ties it is algebraically
identical to the textbook form

$$
\rho_{s,j,b} \;=\; 1 - \frac{6\sum_{i=1}^N d_i^2}{N(N^2-1)},
\qquad d_i = r^{\theta}_i - r^{y}_i .
$$

($\sigma_{r^y}$ carries a $10^{-12}$ floor so a constant bin returns $\rho=0$ rather than
dividing by zero.)

**Pairwise-finite masking** (`spearman_masked`). For each bin $b$, only runs with a finite
value enter: $V_{s,b} = \{i : y_{s,b}^{(i)}\ \text{finite}\}$; ranks are recomputed *within*
$V_{s,b}$, and a bin with $\lvert V_{s,b}\rvert < 100$ returns NaN (excluded from every
max below). There is **no zero-filling** before ranking — zero-filling would manufacture
spurious rank structure. In the current dataset all 12 rows are 100 % finite (each row's
finite fraction is printed by the cell), so the mask is a hygiene guarantee rather than an
active correction; the same masked code path is used for the observed grid *and* for every
permutation draw, so the null construction is consistent by construction.

Why Spearman rather than Pearson: (i) invariance under monotone transforms — the results
are identical whether one correlates against $\theta$, $\log\theta$, or the unit-cube
value, and identical for any monotone re-scaling of the statistic; (ii) bounded influence —
a single extreme run can move a Pearson coefficient arbitrarily but shifts a rank by at
most one position; (iii) it tests for *monotone* response, which is the question being
asked of a one-at-a-time importance map.

The signed matrix for $S(\ell)$ — $\rho_{1,j,b}$, shape $(30, 31)$ — is what fig 5a
displays (red: raising $\theta_j$ raises $S(\ell)$; blue: deepens suppression).

---

## 3. Step 3 — collapse to the importance map

Fig 5's cell value is the **peak absolute correlation over that statistic's scanned
bins**:

$$
I_{s,j} \;=\; \max_{b\,\le\,B_s}\; \bigl|\rho_{s,j,b}\bigr| ,
$$

(`IMPg`, computed with `nanmax` so masked bins never contribute). Since the 2026-08-10
signed-display update, the cell is *painted* with the signed value at that bin,

$$
\rho^{\rm sgn}_{s,j} \;=\; \rho_{s,j,b^\ast_{s,j}},
\qquad b^\ast_{s,j} = \arg\max_{b}\,\bigl|\rho_{s,j,b}\bigr| ,
$$

(`SGN`, drawn on fig 5a's RdBu convention: red = raising the parameter raises the
statistic, blue = lowers it), while **every significance decision remains on the
magnitude** $I_{s,j} = \lvert\rho^{\rm sgn}_{s,j}\rvert$ — the identity
`np.abs(SGN) == IMPg` is asserted in the cell. A single signed cell still cannot show a
sign *change* across bins (the SN-wind energy's $S(\ell)$ response flips with scale);
that bin-resolved structure is fig 5a's job. Columns are ordered by
overall peak importance $\text{peak}_j = \max_s I_{s,j}$, descending; this ordering is
shared with fig 5a so the two figures have one $x$ axis. The bridging identity

$$
\max_b \lvert \rho_{1,j,b}\rvert \;\stackrel{!}{=}\; I_{1,j}
\quad\text{for all } j
$$

is enforced by an `assert` in the fig-5a cell (`column-max == IMPg[suppression]`), so
fig 5's $S(\ell)$ row is by construction the collapse of the matrix fig 5a draws. In
fig 5a a circle marks $b^\ast_j = \arg\max_b\lvert\rho_{1,j,b}\rvert$ for each column — the
bin whose value is exported into fig 5; after the Nyquist cut every circle sits at
$\ell \le 35\,206$ (the last pre-averaged center).

**The statistical price of the max.** $I_{s,j}$ is the maximum of $B_s$ correlated test
statistics. Even if every bin were pure noise, $I_{s,j}$ would systematically exceed the
single-bin noise level, because the max gets $B_s$ chances to fluctuate high. This is the
**look-elsewhere effect** (in multiple-testing language: the need for family-wise error
control under a composite search), and it is why fig 5 carries a ladder of nulls rather
than a single threshold. Each level of the ladder answers "how large would this quantity be
under pure chance, given how many places I looked?" for a progressively larger family:

$$
\text{one bin} \;\subset\; \text{one statistic's bins} \;\subset\;
\text{all 12 statistics (one parameter)} \;\subset\; \text{the whole grid.}
$$

---

## 4. The null ladder

### 4.0 Level 0 — the analytic single-bin null (fig 5a's colorbar dashes)

Under the null hypothesis $H_0$ that a parameter and a bin value are independent, the
permutation distribution of the Spearman coefficient has exactly

$$
\mathbb{E}[\rho] = 0, \qquad \operatorname{Var}[\rho] = \frac{1}{N-1},
$$

and is asymptotically normal, so a two-sided $2\sigma$ threshold for **one pre-specified**
(statistic, parameter, bin) triple is

$$
\rho_{2\sigma} \;=\; \frac{2}{\sqrt{N}} \;=\; \frac{2}{\sqrt{256}} \;=\; 0.125 .
$$

(The code uses $2/\sqrt{N}$; the exact-variance value $2/\sqrt{N-1} = 0.1253$ differs by
0.2 %, far below any decision boundary here. A $2\sigma$ two-sided exceedance corresponds
to $p \simeq 0.046$.) This level is drawn as the dashed pair $\pm\,2/\sqrt{N}$ on
**fig 5a's** colorbar, where it is the appropriate reference because fig 5a shows
individual bins. It is *not* an appropriate threshold for any cell of fig 5, because every
fig-5 cell is already a max over bins.

### 4.1 Why no analytic correction is used for the max

If the $B_s$ bins were independent, the max would obey the Šidák relation

$$
\Pr\!\Big(\max_b \lvert\rho_b\rvert > t \,\Big|\, H_0\Big)
= 1 - \bigl(1 - p_1(t)\bigr)^{B_s},
\qquad
p_1(t) = \Pr\bigl(\lvert\rho\rvert > t \,\big|\, H_0\bigr),
$$

(or the Bonferroni bound $\le B_s\, p_1(t)$), and one could invert for the threshold. But
the bins are strongly and *unevenly* correlated — neighbouring pre-averaged $\ell$ bins of
a smooth spectrum are nearly redundant, while ν-histogram bins are closer to independent —
so the effective number of independent bins is unknown, differs per statistic, and any
analytic correction would be wrong in a statistic-dependent way. The pipeline therefore
calibrates every max **empirically, by permutation**, which adapts to the true correlation
structure automatically.

### 4.2 The permutation ensemble (Westfall–Young max-statistic resampling)

One shared ensemble underlies levels 1–3. With `rng = np.random.default_rng(4)` and
$T = 200$ iterations:

> **for** $t = 1, \dots, 200$:
> &nbsp;&nbsp;&nbsp;&nbsp;draw a uniform random permutation $\pi_t$ of the run indices $\{1,\dots,N\}$
> &nbsp;&nbsp;&nbsp;&nbsp;**for** each statistic $s = 1, \dots, 12$:
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\displaystyle
> \mathcal{N}_{t,s,j} \;=\; \max_{b \le B_s}\,
> \Bigl|\rho\bigl(\theta_j^{(\pi_t(i))},\, y_{s,b}^{(i)}\bigr)\Bigr|
> \quad\text{for all } j = 1,\dots,30,$
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;computed by the *same* `spearman_masked` + `nanmax` code, on the *same* Nyquist-cut, pre-averaged `Ys`, as the observed $I_{s,j}$.

The result is the null cube $\mathcal{N} \in \mathbb{R}^{200 \times 12 \times 30}$
(`null[it, r, j]`): 200 draws of the *entire importance map* as it would look if no
parameter influenced any statistic.

**Why permuting run labels is the right null.** Under the sharp null hypothesis that the
statistics are independent of the parameters, the runs are exchangeable:
$\bigl(\theta^{(\pi(i))}, y^{(i)}\bigr) \overset{d}{=} \bigl(\theta^{(i)}, y^{(i)}\bigr)$
for any permutation $\pi$. Re-pairing the parameter rows against the statistic rows
therefore samples the null *exactly* (conditionally on the observed sets of values — no
distributional assumption about either side is needed; in particular it does not matter
that the Sobol design is a deterministic space-filling set rather than an i.i.d. sample).
The permutation:

- **destroys** the only thing being tested — the parameter ↔ statistic pairing;
- **preserves** the marginal distribution of every parameter exactly (same 256 values,
  re-ordered — the ranks are literally the same set);
- **preserves** the full cross-bin and cross-statistic covariance of the data (the $y$
  side is never touched), which is exactly the correlation structure the max must be
  calibrated against;
- **preserves** the inter-parameter structure of the design (whole rows of `X_unit` are
  permuted at once);
- **inherits** the identical Nyquist cut, NaN masking, pre-averaging, and rank code path.

**Why one shared permutation stream.** The same $\pi_t$ serves all 12 statistics and all
30 parameters in iteration $t$. Two reasons. (i) *Comparability:* differences among the 12
per-statistic nulls then reflect only the statistics themselves (bin count, bin-to-bin
correlation), not independent permutation noise. (ii) *Correctness of the joint levels:*
levels 2 and 3 take maxima **across** statistics; for those maxima to inherit the true
dependence *among* the 12 rows (e.g. peaks, minima and Minkowski functionals are computed
from the same maps and are strongly mutually correlated), all rows must be evaluated under
the same re-pairing within each iteration. Maximizing across rows drawn from *different*
permutations would treat the rows as independent, overstating the look-elsewhere
correction.

### 4.3 Level 1 — the per-statistic null (the × marks; the colorbar band)

$$
\text{NULL95}_s \;=\; Q_{0.95}\Bigl(\bigl\{\, \mathcal{N}_{t,s,j} \;:\;
t = 1,\dots,200;\; j = 1,\dots,30 \,\bigr\}\Bigr),
$$

the 95th percentile of the pooled $200 \times 30 = 6000$ null draws for row $s$
(`np.percentile(null[:, r], 95)`, linear interpolation between order statistics).

*Interpretation.* For **one pre-chosen parameter** tested against **one pre-chosen
statistic**, scanning that statistic's bins: under $H_0$,
$\Pr(I_{s,j} > \text{NULL95}_s) = 0.05$. It corrects the look-elsewhere over *bins* only —
not over parameters, not over statistics.

*Pooling over $j$.* Under $H_0$ the 30 columns are identically distributed, so pooling
them is a valid (consistent) way to tighten the quantile estimate. Within one iteration
the 30 draws share $\pi_t$ and are therefore not mutually independent, so the effective
sample size lies between 200 and 6000 — in practice close to the upper end, because the
Sobol parameter columns are nearly orthogonal by design.

*Usage in the figure.* A cell is **crossed out** ($\times$) iff $I_{s,j} < \text{NULL95}_s$.
The shaded band on fig 5's colorbar spans
$[\min_s \text{NULL95}_s,\ \max_s \text{NULL95}_s] = [0.149,\ 0.182]$, and the current
per-row values are:

| statistic | NULL95 | | statistic | NULL95 |
|---|---|---|---|---|
| $S(\ell)$ | 0.149 | | $C_\ell^{yy}$ | 0.157 |
| PDF$(\nu)$ | 0.159 | | $C_\ell^{\tau\tau}$ | 0.155 |
| $N_{\rm pk}$ | 0.182 | | $C_\ell^{\kappa y}$ | 0.149 |
| $N_{\rm min}$ | 0.176 | | $C_\ell^{\kappa\tau}$ | 0.154 |
| $V_0$ | 0.156 | | $C_\ell^{y\tau}$ | 0.154 |
| $V_1$ | 0.151 | | | |
| $V_2$ | 0.152 | | | |

*Why the rows differ — an effective-bins diagnostic (interpretive only, not used in the
pipeline).* Inverting the Šidák relation with the Gaussian single-bin tail
$p_1(t) = 2\bigl[1 - \Phi(t\sqrt{N-1})\bigr]$ gives an effective number of independent
bins $B^{\rm eff}_s = \ln(0.95)/\ln\!\bigl(1 - p_1(\text{NULL95}_s)\bigr)$. The spectra
($\text{NULL95} \approx 0.149$–$0.157$) come out at $B^{\rm eff} \approx 3$–5 despite 31
scanned bins — smooth spectra are massively redundant across $\ell$ — while the peak
counts ($0.182$) give $B^{\rm eff} \approx 14$ of 22 nominal bins: ν-histogram bins carry
far more independent information per bin. This is exactly the statistic-dependent
structure that a fixed analytic correction would have missed, and the reason the nulls are
measured rather than derived. (It is also why the Nyquist cut barely moved the spectra's
nulls — dropping 14 of 45 heavily-redundant pre-averaged bins removes almost no
independent chance to fluctuate.)

### 4.4 Level 2 — the per-parameter look-elsewhere null (the dashed vertical divider)

$$
\text{NULL}_{\rm col} \;=\; Q_{0.95}\Bigl(\bigl\{\, \max_{s\,\le\,12}\,
\mathcal{N}_{t,s,j} \;:\; t = 1,\dots,200;\; j = 1,\dots,30 \,\bigr\}\Bigr)
\;=\; 0.200 ,
$$

(`np.percentile(null.max(axis=1), 95)`) — the 95th percentile of the pooled 6000 draws of
the **12-row maximum**.

*Interpretation.* For **one pre-chosen parameter** scanned over **all 12 statistics and
all of their bins** (the full 318-bin column of the search): under $H_0$,
$\Pr\bigl(\text{peak}_j > \text{NULL}_{\rm col}\bigr) = 0.05$ where
$\text{peak}_j = \max_s I_{s,j}$. This is the right question for "does parameter $j$ do
anything, anywhere?", and it is the level at which a *parameter* (a column) is declared
active. Because the max is taken within a shared permutation, it inherits the true
correlations among the 12 statistics; the step up from the per-statistic level
(0.149–0.182 → 0.200) is modest precisely because the 12 rows are far from independent.

*Usage in the figure.* Columns are sorted by $\text{peak}_j$; the dashed vertical line
sits after the last column with $\text{peak}_j \ge \text{NULL}_{\rm col}$. Currently
**11 / 30 parameters** clear it. This is deliberately a *different* count from "parameters
with at least one cell above that statistic's own null" (level 1), of which there are
currently **16 / 30** — the level-1 count asks a weaker, per-row question and is
correspondingly more permissive. The two counts are printed side by side by the cell so
the text can never conflate them.

### 4.5 Level 3 — the whole-grid null (the printed numerals)

$$
\text{NULL}_{\rm grid} \;=\; Q_{0.95}\Bigl(\bigl\{\, \max_{s\,\le\,12}\ \max_{j\,\le\,30}\,
\mathcal{N}_{t,s,j} \;:\; t = 1,\dots,200 \,\bigr\}\Bigr)
\;=\; 0.253 ,
$$

(`np.percentile(null.max(axis=(1, 2)), 95)`) — the 95th percentile of the 200 draws of the
**grand maximum over the entire grid**.

*Interpretation.* This is single-step **Westfall–Young max-statistic family-wise error
control** over the whole search: under the global null (no parameter affects any
statistic),

$$
\Pr\Bigl(\ \exists\, (s,j):\ I_{s,j} \ge \text{NULL}_{\rm grid} \ \Big|\ H_0 \Bigr) = 0.05 .
$$

A cell at or above 0.253 would arise by chance anywhere in the $12 \times 30$ map (each
cell already maxed over its bins — i.e. anywhere among all 9 540 scanned correlations) in
fewer than 5 % of null experiments. It is the threshold for calling a **single cell**
discovered with *no* qualification about where one looked. Unlike a Bonferroni correction
over 9 540 comparisons (which would demand
$p_1 < 0.05/9540 \Rightarrow \lvert\rho\rvert \gtrsim 0.28$), the permutation max
adapts to the heavy redundancy among bins, statistics, and the shared design, landing at
0.253.

*Usage in the figure.* Every cell with $I_{s,j} \ge \text{NULL}_{\rm grid}$ prints its
value as a numeral (leading zero stripped, 2 decimals). Currently **71 / 360 cells**
qualify; the grid's grand maximum is $I = 0.675$ ($C_\ell^{\tau\tau}$ ×
`VariableWindVelFactor`).

*A note on pooling.* Unlike levels 1–2, the parameter index sits *inside* the max here, so
no pooling over $j$ is possible: this quantile rests on 200 draws only and carries the
largest Monte-Carlo error of the ladder (§6.3).

### 4.6 The ladder is monotone by construction

The four families are nested, and the maximum over a superset dominates the maximum over a
subset pointwise, so their 95 % quantiles are ordered:

$$
\underbrace{0.125}_{\text{single bin}}
\;\le\;
\underbrace{0.149\text{–}0.182}_{\text{per statistic}}
\;\le\;
\underbrace{0.200}_{\text{per parameter}}
\;\le\;
\underbrace{0.253}_{\text{whole grid}} .
$$

(The first link mixes conventions — analytic $2\sigma \approx 95.4\,\%$ two-sided against
empirical 95th percentiles — but the ordering holds with margin.) Each threshold answers a
progressively harder question; which one applies to a claim depends only on how the claim
was *found*: a pre-registered pair → level 0/1; "this parameter matters" → level 2; "this
cell, found by scanning everything" → level 3.

---

## 5. The marks on the figure (key)

The on-figure text key was removed in the 2026-08-10 de-texting (only axis, tick, and
colorbar labels remain); this table and the paper caption now carry it.

| mark | meaning | threshold |
|---|---|---|
| cell color (RdBu diverging) | sign of $\rho$ at the peak-$\lvert\rho\rvert$ bin — red: raising the parameter raises the statistic; blue: lowers it (fig 5a's convention) | — |
| $\times$ on a cell | $I_{s,j} < \text{NULL95}_s$ — consistent with chance even for a single pre-chosen parameter | level 1, 0.149–0.182 |
| numeral on a cell | $I_{s,j} \ge \text{NULL}_{\rm grid}$ — survives the full whole-grid look-elsewhere; prints the *signed* value | level 3, 0.253 |
| dashed vertical line | columns left of it have $\text{peak}_j \ge \text{NULL}_{\rm col}$ | level 2, 0.200 |
| colorbar shaded band (mirrored ±) | the range of the 12 per-statistic nulls — thresholds on $\lvert\rho\rvert$ apply to both signs | ±[0.149, 0.182] |
| colorbar black dashed pair | the whole-grid null, mirrored about zero | ±0.253 |
| fig 5a colorbar black dashes | analytic single-bin $\pm 2/\sqrt{N}$ | ±0.125 |
| fig 5a circles | per-column $\arg\max_b \lvert\rho\rvert$ — the bin exported into fig 5's $S(\ell)$ row (all at $\ell \le 35206$ after the Nyquist cut) | — |

Cells with neither an $\times$ nor a numeral sit between levels 1 and 3: above their
statistic's own null but not safe against the whole-grid search.

---

## 6. Statistical fine print

1. **Two-sidedness.** Every max is over $\lvert\rho\rvert$, so all levels are two-sided
   tests; the signs painted on fig 5 (at the peak bin) and on fig 5a (bin by bin) are
   display, never part of any test.
2. **Exactness on a deterministic design.** Permutation inference does not require the
   runs to be a random sample — only exchangeability under $H_0$, which holds conditional
   on the observed design and maps. The Sobol nodes being deterministic is therefore not a
   caveat.
3. **Monte-Carlo precision of the quantiles.** With $T$ draws, the 95th percentile is read
   off near order statistic $0.95\,T$; its sampling fluctuation corresponds to
   $\pm\sqrt{T\,p(1-p)} \approx \pm 3$ ranks at $T=200$ — i.e. the grid null is determined
   to a few units in the third decimal. Levels 1–2 pool 6000 draws and are tighter.
   Percentiles use NumPy's default linear interpolation between order statistics.
4. **Resolution of the null.** 200 permutations resolve tail probabilities to 1/200 =
   0.5 %; adequate for 95th percentiles (10 draws sit above each threshold), and the seed
   (`rng(4)`) makes every quoted digit reproducible.
5. **Consistency of pipelines.** The null cube is built from the *Nyquist-cut,
   pre-averaged* `Ys` with the *same* masked-Spearman and `nanmax` code as the observed
   map — the null and the observation differ **only** in the permutation. Any
   pre-processing asymmetry would bias the ladder.
6. **Two ℓ limits, two jobs.** The ranking's cut is the axis-Nyquist mode
   $\ell_{\rm Ny} = 36\,864$ (§1a; author ruling 2026-08-10). A second, tighter constant
   exists in the notebook: `ELL_TRUST` $= 3\times10^4$ ($0.8\,\ell_{\rm Ny}$, the
   *measured* CIC/pixelization upturn onset), which governs the **display** of raw
   (unratioed) spectra in fig 7 and the raw-spectrum markers elsewhere. The two are
   deliberately distinct: the Nyquist cut removes the corner-mode zone with no azimuthal
   support from the *scan*, while `ELL_TRUST` flags where the raw spectra's CIC upturn
   *begins* — $S(\ell)$, as a BIND/DMO ratio, cancels that artifact and remains valid
   through it. Bins in $(3\times10^4, \ell_{\rm Ny}]$ of the raw auto/cross rows do enter
   the scan under the current ruling.
7. **What changes if the run count changes.** All four levels scale roughly as
   $1/\sqrt{N}$ (and depend on the bin grids); nothing is hardcoded — every threshold is
   recomputed from the loaded dataset on each notebook run and printed alongside the
   figure. The numbers in this document are the $N=256$, seed-4, 200-iteration,
   Nyquist-cut values of `emulator_dataset_nu05.npz` (2026-08-10).

---

## 7. Implementation map (name ↔ math)

All in `papers/01_pipeline/_build_figures_nb.py` (mirrored into
`paper1_figures.ipynb`); the fig-5a cell binds everything fig 5 and fig 7 consume.

| code | object | equation |
|---|---|---|
| `LNYQ`, `_ranked(s)` | Nyquist mask $\ell \le \ell_{\rm Ny}$ on the ranking's ℓ rows | §1a |
| `rebin_nan(A, k)` | NaN-aware block mean | §1b |
| `spearman(P, Y)` | Pearson-on-ranks kernel, all rows finite | §2 |
| `spearman_masked(P, Y, min_n=100)` | pairwise-finite Spearman, per-bin masks | §2 |
| `Ys[k]` | Nyquist-cut, pre-averaged statistic matrices $(N, B_s)$ | §1 |
| `Rs[k]` | $\rho_{s,j,b}$, shape $(30, B_s)$ | §2 |
| `IMPg` | $I_{s,j} = \max_b\lvert\rho\rvert$, shape $(12, 30)$ | §3 |
| `SGN` (`_signed_peak`) | $\rho^{\rm sgn}_{s,j} = \rho_{s,j,b^\ast}$, the painted signed value; $\lvert$`SGN`$\rvert$ = `IMPg` (asserted) | §3 |
| `RHO_2SIG` | $2/\sqrt{N} = 0.125$ | §4.0 |
| `null` | $\mathcal{N}_{t,s,j}$, shape $(200, 12, 30)$, seed `rng(4)` | §4.2 |
| `NULL95[k]` | $Q_{0.95}$ of `null[:, r]` pooled $(200 \times 30)$ | §4.3 |
| `NULL_COL` | $Q_{0.95}$ of `null.max(axis=1)` | §4.4 |
| `NULL_GRID` | $Q_{0.95}$ of `null.max(axis=(1, 2))` | §4.5 |
| `peak_col`, `col_order` | $\text{peak}_j$, column sort | §3 |
| `n_col_ok` | $\#\{j : \text{peak}_j \ge \text{NULL}_{\rm col}\} = 11$ | §4.4 |
| `assert … == IMPg[0]` | fig-5a ↔ fig-5 bridging identity | §3 |

Current headline numbers (printed by the fig-5 cell, 2026-08-10 Nyquist-cut, signed
render): single-bin $2\sigma = 0.125$; per-statistic nulls 0.149–0.182; per-parameter
look-elsewhere 0.200 (11/30 clear; 16/30 clear ≥1 row-level null); whole-grid 0.253
(71/360 cells clear); grand max 0.675, signed $-0.68$ ($C_\ell^{\tau\tau}$ ×
`VariableWindVelFactor`: faster winds *lower* the $\tau$ auto-spectrum at its peak bin
while raising the $y$-side spectra — opposite-sign responses now visible on the map).
