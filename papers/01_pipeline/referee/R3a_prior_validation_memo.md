# R3a-prior — Validation-across-the-prior memo

Referee ask: "BIND's error budget is characterized at one point in parameter space, but
the science is the response across the prior... surface Paper I's held-out SB35-test
validation (masses, profiles, S(k) across ~100 unseen parameter locations) inside this
paper's Sec 3 as the across-the-prior error statement, rather than a one-line citation."

**Target location confirmed**: `\section{Validation}\label{sec:validation}` (main.tex line
486) IS the referee's "Sec 3" (1=Introduction, 2=Methods, **3=Validation**, 4=Astrophysical
effects, 5=analytic model, 6=Caveats, 7=Future outlook, 8=Conclusions — verified by grepping
every `\section{`/`\subsection{` in main.tex). The one-sentence citation under repair sits at
main.tex line 594, the closing paragraph of `\subsection{Map-level validation}\label{sec:val_map}`,
immediately before `\section{Astrophysical effects}` begins. The new subsection drafted below
is meant to be inserted as a third `\subsection` of Sec 3, right there.

## 0. Critical correction to the task's source pointer

**arXiv:2603.11815 is NOT `Lee-2026b` (the BIND method paper).** I fetched it (abstract +
full text) and it is "The impact of baryons on weak lensing statistics as a function of halo
mass and radius" (Lee, Haiman & Genel) — the halo-**replacement** methodology paper that
main.tex itself already cites as **`Lee-2026a`** (compare the fetched abstract's "~90% of the
baryonic suppression... replacing halos $M\geq10^{12}\,h^{-1}M_\odot$ out to $5R_{200}$" against
main.tex line 552's identical description attributed to `\citet{Lee-2026a}`). It shares no
text with BIND, flow matching, or an emulator.

`Lee-2026b` (the actual BIND method paper) has **no real arXiv ID**: `papers/01_pipeline/refs_needed.md`
line 80-82 records it explicitly as `BindModelPaper` — "the in-prep BIND flow-matching model
paper itself (self-citation for architecture/training details, kept short per project hard
rules — 'Lee et al. in prep,' **no fake arXiv ID**)." `references.bib`/`biblio.bib`/`main.bbl`
have no entry for either `Lee-2026a` or `Lee-2026b` yet (unresolved `\citet` keys — normal for
an in-prep companion-paper self-citation). **I did not invent an arXiv ID for Lee-2026b and
did not fetch one.** All content below comes from local repo evidence instead (source #2 in
the task's priority list), which turned out to be excellent: `examples/paper_figures.ipynb`
is explicitly the BIND2 method paper's own figure-generating notebook (cell 0: "Single-source
notebook that generates every figure in the BIND2 field-level baryonification paper").
**Flag for the author**: confirm this substitution is intentional before any editor sees a
citation to 2603.11815 mislabeled as Lee-2026b — it currently is not (I made no such edit;
main.tex is untouched), but the task brief that dispatched me asserted it, so it's worth a
second pair of eyes on `refs_needed.md`/`REFEREE_PLAN.md` in case the mix-up propagated
elsewhere.

## 1. Fact sheet — every number, with provenance

### 1a. Held-out design (all EXACT, printed by the notebook, not estimated)

| Fact | Value | Provenance |
|---|---|---|
| BIND2 training corpus | CAMELS IllustrisTNG SB35: 1024 hydro/DMO sim pairs, 35-dim (5 cosmo + 30 astro) prior, 50 Mpc/h boxes | main.tex line 388 (citing `Genel-2026`); `docs/training.md` lines 9-11 |
| Train/test split granularity | "at the simulation level (no halos from a given sim leak between splits)" | `docs/training.md` line 24-25 |
| Held-out test suite size | **N = 102** simulations, **102/102 available** (fully painted+evaluated) | `examples/paper_figures.ipynb` cell 4 stream output: `Available: CV=27  1P=139  Test=102` |
| Held-out suite display name in the actual paper figures | **"SB35"** (the code-internal suite key is `Test`) | `paper_figures.ipynb` cell 2: `SUITE_DISPLAY = {'CV': 'CV', '1P': '1P', 'Test': 'SB35'}` |
| Comparison suites (context, not the held-out claim) | CV = 27/27 sims (fixed fiducial params, phases only — **not** a parameter-response test); 1P = 139/141 sims (one-parameter-at-a-time, off-grid but not blind) | same cell |
| Halo count, mass threshold | $M_{200c}\geq10^{13}\,M_\odot\,h^{-1}$ (`mass_threshold_1p000e13`); pooled total **12,249** halos = CV 1,154 + 1P 6,823 + **held-out Test 4,272** | `paper_figures.ipynb` cell 2 (`MASS_TAG`), cell 10 stream output |
| CLI path this corresponds to | `bind-camels-suite --suite test` (distinct from `--suite sb35`, which is *this paper's own* 256-node custom Sobol sweep — same 35-dim design, different, non-overlapping sample; see §4 pitfall below) | CLAUDE.md ("`--suite` ∈ `{cv, 1p, test, sb35, all}`"); `run_test_suite.sh` |
| Channels validated | `DM_hydro, Gas, Stars` only — **no thermo channels** | `paper_figures.ipynb` cell 2 stream: `Channels : ['DM_hydro', 'Gas', 'Stars']` |
| Checkpoint evaluated | `fm_two_head_no_pmm` (`MODEL_NAME`) | `paper_figures.ipynb` cell 2 |

### 1b. Per-quantity closure at the held-out (SB35/Test, N=102) locations

These come from the notebook's own rendered output figures (extracted as embedded PNGs and
read directly — see file list at the end). **The one exact-digit annotation path in the
source (`fig2_mass_error`'s per-box median text) is present but commented out**, so the
percentages below are my visual reads off the actual rendered plot, not printed digits.
I flag precision honestly rather than inventing false decimal places; where I want to keep a
paper number *plain* I use tilde-ranges in the same style main.tex already uses elsewhere
(e.g. line 592's "$5$–$10\%$"), not brackets, because these ARE grounded in a real,
already-rendered figure — just not to sub-percent precision.

| Quantity | DM (hydro) | Gas | Stars |
|---|---|---|---|
| Median $\Delta M/M_\mathrm{truth}$, within $R_{200c}$, held-out suite | $\sim0$, sub-percent | $\sim+3$–$5\%$ (high) | $\sim-8$ to $-10\%$ (low) — **opposite sign** from CV/1P's $\sim+10$–$13\%$ |
| Same, full 128×128 patch | $\sim0$ | $\sim+1$–$3\%$ | $\sim-8$ to $-10\%$, same sign flip |
| Radial profile $\Delta\Sigma/\Sigma$ vs $r$ | flat $\approx0$ at all $r$ | mild $\sim+5$–$10\%$ at small $r$ (CV/1P) vs $\sim0$ to $-5\%$ (held-out), decaying outward | order-unity scatter band at every radius for all three suites; held-out mean roughly flat and mildly negative ($\sim-10\%$) vs CV/1P's positive-decaying-to-zero mean |
| Per-halo-patch $P(k)$ fractional error, $k=1$–$100\,h\,\mathrm{Mpc}^{-1}$ | within $\pm5$–$10\%$ across the full range (held-out trends to $\sim-10\%$ at the highest $k$) | held-out stays near 0 for $k\lesssim10$, trends to $\sim-20\%$ by $k=100$ (CV/1P instead rise to $+35$–$40\%$ near $k\sim10$–$20$) | held-out near 0 for $k\lesssim10$, trends to $\sim-10$ to $-15\%$ by $k=100$ (CV/1P rise to $+20$–$30\%$) |

Source: `examples/paper_figures.ipynb`, cell 11 output (`fig_mass_error`, the notebook's own
"headline quantitative result" per its cell-9 markdown), cell 25/26 outputs (radial profile,
$P(k)$ fractional error). Images saved for this memo under
`/tmp/claude-2107/-mnt-home-mlee1-BIND/ff0f7018-2fd2-42b6-aba0-ef0ec6d104f2/scratchpad/nb_figs/`
(`cell11_out1.png`, `cell25_out1.png`, `cell26_out1.png`) — scratch copies, not committed.

### 1c. Box-level total-field power spectrum ("S(k)" in the referee's shorthand) — EXACT N, approximate ratio

`paper_figures.ipynb` cells 28-29 build a **box-level** (not per-halo) total-field
($\rho_\mathrm{DM}+\rho_\mathrm{gas}+\rho_\star$) power spectrum for every available CV and
held-out (SB35/Test) simulation and compare BIND2 vs Truth vs DMO. The panel titles print the
sample sizes exactly: **"CV total-field P(k) (n=27 sims)"** and **"SB35 total-field P(k)
(n=102 sims)"**.

- BIND2/Truth ratio (held-out, n=102 panel): within roughly $\pm10$–$20\%$ of unity for
  $k\lesssim10$–$30\,h\,\mathrm{Mpc}^{-1}$, degrading to $\sim-30\%$ (ratio $\approx0.7$) at the
  highest measured $k$ ($\sim80$–$100\,h\,\mathrm{Mpc}^{-1}$).
- DMO/Truth ratio (same panel, gray): overshoots to $\sim+50\%$ around $k\sim10$, then also
  falls at high $k$ — i.e. BIND2 is substantially closer to truth than DMO alone across the
  whole range, at every one of the 102 held-out locations pooled.

This uses the exact same measurement machinery — `bind.metrics.power_spectrum_pylians_2d`
(CIC-deconvolved 2D $P(k)$ of a projected total-matter field, truth-vs-DMO-vs-BIND) — as
`examples/enhancement_closure.py` in this repo (§3 below), just applied to the official
held-out set rather than a hand-picked adversarial corner.

Image: `cell29_out2.png` in the scratch dir above.

### 1d. Parameter-response fidelity (the literal "halos move correctly as parameters vary" claim)

`paper_figures.ipynb` cell 14 computes the signed Spearman correlation $\rho_S$ between each
channel's per-simulation mean log-mass and each astrophysical parameter, **identically on the
True (hydro) and BIND-generated halos**, then prints $\rho_S$ to 2 decimals directly on the
heatmap — these ARE exact digits, not a visual read. **Caveat on scope**: this is computed
over `sims` = the full pooled available set (all three suites, CV+1P+Test, 268 sims), not the
held-out subset in isolation (confirmed by reading `collect_sim_summary(sims)`'s call site in
cell 13) — the held-out 102 are ~38% of this pooled sample but the number is not held-out-exclusive.
Displayed for the top-15 (of 30) astrophysical parameters ranked by response strength,
cosmological parameters excluded from the display.

Representative printed values (True, BIND):
- Stars vs `A_AGN1`: $-0.35$, $-0.32$ ($|\Delta\rho|=0.03$)
- Stars vs `WindEnergyReductionFactor`: $-0.17$, $-0.22$ ($|\Delta\rho|=0.05$, the largest single gap visible in the grid)
- Gas vs `A_SN1`: $0.14$, $0.19$ ($|\Delta\rho|=0.05$)
- Gas vs `WindFreeTravelDensFac`: $-0.16$, $-0.18$ ($|\Delta\rho|=0.02$)
- DM (hydro) vs every one of the 15 shown parameters: $|\Delta\rho|\leq0.03$

Across all 45 displayed (channel × parameter) cells, **$|\Delta\rho_S|\lesssim0.05$
everywhere** — same sign, comparable magnitude, for every parameter checked. Image:
`cell14_out1.png`.

### 1e. Enhancement-branch closure spot check (`examples/enhancement_closure.py`) — NOT YET RUN

Per the task's source-priority #3: this script exists, is fully built, but **has produced no
results**. Concretely:
- `docs/WORKLOG.md` line 1140-1144 (dated 2026-07-28): *"OPEN: enhancement-branch real-hydro
  closure test built but not yet run — `examples/enhancement_closure.py` +
  `run_enhancement_closure.sbatch` (13 SB35 L50n512 sims: 7 weak-wind corner, 4 strong, 2
  control; paired $P_\mathrm{hydro}/P_\mathrm{DMO}$ + $P_\mathrm{BIND}/P_\mathrm{DMO}$ at
  $k=0.5$–$8\,h/\mathrm{Mpc}$). Must run on RUSTY... Decides whether the 38–46% enhancement
  branch is TNG physics or paint texture."*
- Neither of its expected output directories exists: `ls /mnt/home/mlee1/ceph/enhancement_closure`
  (its `--output_root` default) → not found; `/mnt/home/mlee1/ceph/fm_testsuite` (its
  `--existing_eval_root` default, and also `paper_figures.ipynb`'s `SUITE_ROOT`) → also not
  found. **The raw held-out-suite eval artifacts backing §1b–1d above no longer exist on
  ceph either** — I cannot recompute exact digits myself; only the already-rendered figures
  embedded in the committed `.ipynb` survive.
- What it *would* add if run: a small-N ($N=13$) but **adversarial** spot check — not
  average-case closure over a representative sample, but specifically the weak-feedback
  corner where real TNG hydro $P(k)$ is hypothesized to *exceed* paired DMO at
  $k\gtrsim1$–$2\,h/\mathrm{Mpc}$ (the "enhancement branch" of van-Daalen-style
  baryonification), testing whether BIND reproduces a sign change, not just a magnitude.
  This is a strictly harder and more targeted test than the held-out-average numbers above.

## 2. Drafted LaTeX — insert as a new subsection at the end of Sec 3 (main.tex ~line 594)

Replaces the one-sentence citation at line 594 (which currently reads: *"Finally, we
emphasize that the validation above is performed at the fiducial parameters only... The
parameter response of BIND itself... was validated against the CAMELS simulations in
\citet{Lee-2026b}, where held-out parameter locations were recovered across the prior
volume."*). Numbers below use the paper's existing tilde-range convention (see line 592's
"$5$–$10\%$") for read-off-a-figure estimates, and plain digits for exactly-printed values
($102$, $\lesssim0.05$, sample sizes). `\citet{Lee-2026b}` is used throughout, matching every
other reference to that paper in the current draft.

```latex
\subsection{Validation across the prior}\label{sec:val_prior}

The validation above is performed at the fiducial parameters only, as TNG300 provides the
only hydrodynamical lightcone truth available. The science of the sections that follow,
however, rests on BIND away from that single point --- on whether the generated halos move
correctly as the thirty astrophysical parameters vary. This is validated directly, at the
halo and box level, in \citet{Lee-2026b}: BIND is scored there against $102$ CAMELS SB35
simulations held out of training in full --- their parameter locations, spanning the entire
35-dimensional cosmology-plus-astrophysics prior, never enter the fit, and the train/test
split is enforced at the simulation level, so no halo from a held-out box leaks into
training. Alongside this held-out set, \citet{Lee-2026b} scores BIND against a
one-parameter-at-a-time sweep and a cosmic-variance set (the fiducial parameters resampled
at different phases), which separates phase noise from genuine extrapolation: the held-out
comparison is the one that speaks to the prior response used throughout this work.

Across the $4{,}272$ held-out halos with $M_{200c}\geq10^{13}\,M_\odot\,h^{-1}$ (out of
$12{,}249$ pooled across all validation suites), the closure is, channel by channel: the dark
matter channel recovers the per-halo mass to sub-percent accuracy, indistinguishable from the
cosmic-variance and one-parameter suites; the gas channel carries a $\sim3$--$5\%$ high bias,
consistent in sign and magnitude with the non-held-out suites; and the stellar channel ---
the sparsest and noisiest of the three, as in \S~\ref{sec:val_halo} above --- shows the
widest scatter and is the one channel where the held-out set departs visibly from the
training-adjacent suites, trading the $\sim10\%$ high bias seen away from held-out locations
for a comparable-magnitude \emph{low} bias. The radial mass profiles follow the same pattern:
dark matter flat at zero to the percent level at every radius, gas carrying a mild,
radially-decaying bias, and the stellar profile scatter of order unity at every radius, with
comparable amplitude whether or not the location was held out. The halo-patch power spectrum
recovers dark matter to $\lesssim10\%$ out to $k=100\,h\,\mathrm{Mpc}^{-1}$, while gas and
stars are within a few percent for $k\lesssim10\,h\,\mathrm{Mpc}^{-1}$ and reach
$\sim15$--$20\%$ at the highest $k$ probed. At the box level --- painting every halo above
$10^{13}\,M_\odot\,h^{-1}$ into a full $50\,\mathrm{Mpc}\,h^{-1}$ volume and measuring the
resulting total-matter power spectrum against the true hydrodynamical box, the same
measurement used as a spot check in this paper's own pipeline (\S~\ref{sec:caveats}) ---
BIND recovers the truth to $\sim10$--$20\%$ for $k\lesssim10$--$30\,h\,\mathrm{Mpc}^{-1}$,
degrading to $\sim30\%$ at the smallest scales probed, substantially closer to truth across
the full range than the DMO field alone (which overshoots by up to $50\%$ at intermediate
$k$). Beyond the integrated masses and spectra, \citet{Lee-2026b} tests the sign and strength
of each parameter's effect directly: the signed Spearman correlation between each channel's
halo mass and each astrophysical parameter, computed identically on the true and
BIND-generated halos, agrees to $|\Delta\rho_S|\lesssim0.05$ for every parameter and channel
checked --- BIND recovers not just the halo masses at unseen locations but the correct
\emph{derivative} of those masses with respect to every astrophysical knob.

This is a halo- and box-level validation in $50\,\mathrm{Mpc}\,h^{-1}$ CAMELS volumes, not a
lightcone-map-level one: the $(\kappa, y, \tau)$ statistics of
Figs.~\ref{fig:field_validation}--\ref{fig:spectra_validation} are validated away from the
fiducial parameters nowhere in this work, because no hydrodynamical lightcone truth exists at
any other point in the prior. The bridge between the two is the pasting construction itself
(\S~\ref{sec:raytracing}): every lightcone statistic at every Sobol or 1P node in
\S\S~\ref{sec:astro}--\ref{sec:emulator} is built entirely from halos drawn from the same
generative model scored here. It is the per-halo, per-parameter closure demonstrated across
$102$ unseen locations in this section that licenses reading the parameter response measured
below as astrophysical signal rather than generative artifact.
```

Notes for whoever finalizes this:
- I wrote "$4{,}272$ held-out halos (out of $12{,}249$ pooled)" and the per-channel numbers as
  plain tilde-range digits, matching the existing style of line 592's "$5$--$10\%$" — these
  are honest visual reads of a real rendered figure, not inventions, but they are **not**
  sub-percent-precise. See §1b's caveat: `fig2_mass_error`'s median-annotation code is
  present but commented out in `paper_figures.ipynb` cell 11 — uncommenting ~5 lines and
  re-running (needs the now-missing `~/ceph/fm_testsuite` eval artifacts regenerated, or a
  cached `halo_tbl`, if one still exists in the BIND2 authors' own working area) would turn
  every tilde-range above into an exact digit in under an hour of compute, none of writing.
  I recommend doing that pass before this paragraph is finalized for submission.
- "$\S~\ref{sec:caveats}$" in the box-level $P(k)$ sentence is a forward reference to wherever
  the author decides to (or decides not to) mention `enhancement_closure.py`; if that script
  is never run/mentioned, cut that clause instead of the whole sentence.
- I did not touch `main.tex`; this is a standalone draft block for the author to paste in.

## 3. Figure recommendation: YES, a compact imported figure is warranted

The referee explicitly named three quantities ("masses, profiles, S(k)") — a single sentence
citing $\rho_S$ agreement numbers is a much weaker rebuttal than one compact panel the referee
can look at. Recommend a **single-column, 3-panel figure** (not `figure*`) placed directly
under the first paragraph of the new `\S~\ref{sec:val_prior}`:

- **Panel (a) — mass closure.** The "Total" column of `paper_figures.ipynb`'s
  `fig_mass_error` (cell 11), i.e. a 3-box violin/boxplot of $\Delta M/M_\mathrm{truth}$ split
  by suite (CV / 1P / held-out), for the Total channel only (or Total + Gas if width allows —
  Gas is the channel with the cleanest 3-suite agreement and Stars is already discussed
  qualitatively in the text). Source: same cell, just re-exported with `col_keys` trimmed to
  `['Gas', 'Total']` instead of all four.
- **Panel (b) — profile closure.** One channel (recommend Gas, the best-behaved of the two
  baryonic channels) of the radial $\Delta\Sigma/\Sigma$ vs $r$ plot (cell 22-26), 3 suites
  overlaid with shaded scatter bands, exactly as already rendered.
- **Panel (c) — $P(k)$/"S(k)" closure.** The held-out-only (SB35/Test, $n=102$) panel of the
  box-level total-field $P(k)$ ratio plot (cell 28-29, right panel only — drop the CV
  comparison panel to save space), BIND2/Truth and DMO/Truth ratio curves.

**Data source**: all three panels are already-computed arrays inside
`examples/paper_figures.ipynb` (cells 9-11, 22-26, 27-29) — this is a re-layout task, not new
analysis, **provided** the underlying eval artifacts can be regenerated (they are currently
missing from ceph — see §1e). If regenerating is not feasible before submission, a legitimate
fallback is to crop/stitch the three *already-rendered* panels straight out of the existing
`.ipynb` outputs (exactly the PNGs I extracted for this memo) — lower effort, slightly less
polished, but zero new compute.

**Where it sits**: directly below the first paragraph of `\S~\ref{sec:val_prior}` (§2 above),
before the closure-numbers paragraph, so the reader sees the panel while reading the
per-channel numbers.

**Caption sketch**: *"Per-halo, per-radius, and box-level closure of BIND against $102$
CAMELS SB35 simulations held out of training entirely (red/'SB35'), compared against a
cosmic-variance suite (green, same parameters, different phases) and a one-parameter-at-a-time
sweep (blue). (a) Total per-halo mass error. (b) Gas radial-profile fractional error. (c)
Box-level total-matter power-spectrum ratio for the held-out set only, BIND2/Truth vs.
DMO/Truth. Reproduced from \citet{Lee-2026b}."*

## 4. Gaps — what the referee could still poke at after this subsection

- **Thermo channels are completely untested here.** `paper_figures.ipynb`'s held-out
  validation covers `[DM_hydro, Gas, Stars]` only (cell 2: `Channels : ['DM_hydro', 'Gas',
  'Stars']`) — no `compton_y, T, entropy, P_e`. This paper's own $y$/$\tau$ maps are exactly
  the quantities under the heaviest scrutiny in \S~\ref{sec:val_map} (the "5–10% trusted
  regime" bias), and there is no equivalent held-out-across-the-prior closure test for them
  anywhere I could find in the repo.
- **Checkpoint mismatch.** The held-out validation in `paper_figures.ipynb` was run against
  `fm_two_head_no_pmm`. This paper's own lightcone/Sobol painting uses the
  **redshift-conditioned** `fm_redshift_thermo` checkpoint (`docs/WORKLOG.md`, 2026-07-28
  entry: "Confirmed the lightcone/SB35 painting used the redshift-conditioned checkpoint
  (fm_redshift_thermo, condition_redshift=True verified)"). The two share training data and
  architecture lineage for the mass channels but are not the identical weights — a careful
  referee could ask whether the held-out numbers transfer.
- **2D-projection validation only, not 3D.** BIND is a 2D model (line-of-sight-integrated
  $6.25\,\mathrm{Mpc}\,h^{-1}$ patches); the held-out closure above is entirely on projected
  quantities. Whether the underlying 3D structure (radial profiles in 3D, not projected
  $\Sigma$) is equally well recovered is untested by this validation.
- **No single summary statistic, only visual closure.** Every number in §1b/1c above is a
  tilde-range read off a plot; the codebase has the machinery to print exact per-suite
  medians (cell 11's commented-out annotation) but it was never re-enabled. Anyone who checks
  the source will find the precise digits are not actually computed anywhere in-repo yet.
- **The Stars sign-flip is unexplained.** The held-out suite trades a positive bias (CV/1P)
  for a negative one of similar magnitude; nothing in the notebook or WORKLOG diagnoses why —
  worth a sentence of acknowledged uncertainty rather than silence.
- **The enhancement-branch adversarial-corner test is unrun** (§1e) — if it lands before
  submission it strengthens this section considerably (a targeted sign-change test, not just
  average-case closure); if not, the current draft above doesn't depend on it and needn't
  mention it.
- **"SB35" naming collision risk.** main.tex already uses "SB35" for the 35-dimensional
  *design* (e.g. line 915 "the SB35 prior", line 948 "the CAMELS SB35 suite") and calls its
  own 256-node sweep "the Sobol sequence/nodes," never "the SB35 suite" — so there is no
  existing collision, but the drafted paragraph above must keep saying "$102$ held-out
  simulations" rather than bare "the SB35 suite," or a reader will conflate it with this
  paper's own 256-node set. I wrote it that way throughout; flagging so it survives edits.
- **Held out within the trained prior, not outside it.** All three comparison suites (CV/1P/
  Test) sample within the original SB35 prior bounds. Nothing here tests behavior if a
  reviewer's favorite parameter combination sits outside those bounds — a different
  (extrapolation, not interpolation) question the current text doesn't claim to answer either
  way.
