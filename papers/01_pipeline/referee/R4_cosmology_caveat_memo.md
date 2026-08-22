# R4 — Fixed cosmology vs. Stage-IV framing: verified reference, bound, and drafted caveat

Agent: R4-cosmo. Scope: `\section{Caveats}` item 3 (main.tex ~l.930, currently
`% 3 fixed cosmology -> varied-cosmology DMO painting`) and its Sec. 6 outlook
pairing. **main.tex was not edited** (read-only per instructions); everything
below is drafted prose + a bibtex entry for the author to paste in.

**Verdict up front**: the referee's pointer is correct and lands cleanly.
Willem Elbers is first author of arXiv:2403.12967 / MNRAS 537, 2160–2178
(2025), "The FLAMINGO project: the coupling between baryonic feedback and
cosmology in light of the $S_8$ tension" — a genuine, quantified
feedback×cosmology non-factorizability result, with a coupling parameter the
paper itself calls $\xi^2$. I verified title/authorship/journal/DOI/arXiv ID
independently (see confidence notes in §1). Propagating their calibrated
coupling through a Stage-IV-width cosmology posterior onto this paper's own
predicted trough gives an induced error of **|ΔS| ≈ 0.01–0.02** at the
suppression trough — comparable to, or several times larger than, both this
paper's own predictive scatter and the raw LSST-Y10 statistical floor. Not
negligible; worth stating candidly, as the referee asks.

---

## 1. The verified reference

### 1.1 Identity (independently cross-checked, 6 fetches, 2 hosts)

| Field | Value |
|---|---|
| First author | **Willem Elbers** (confirmed first-author position) |
| Full author list | Elbers, Frenk, Jenkins, Li, Helly, Kugel, Schaller, Schaye, Braspenning, Kwan, McCarthy, Salcido, van Daalen, Vandenbroucke, Pascoli |
| Title | "The FLAMINGO project: the coupling between baryonic feedback and cosmology in light of the $S_8$ tension" |
| arXiv | **2403.12967** (v1: 2024-03-19; v2: 2025-01-15, "accepted for publication") |
| Journal | **MNRAS 537(2), 2160–2178 (2025)** |
| DOI | **10.1093/mnras/staf093** |

Confirmed via: (1) `arxiv.org/abs/2403.12967` — author list + abstract; (2)
`arxiv.org/html/2403.12967v1` — two independent fetches (one leading, one
neutral prompt) for the coupling equations/numbers; (3)
`arxiv.org/html/2403.12967v2` — two more independent fetches, confirming the
v1 numbers survived to the (accepted) revised version; (4)
`academic.oup.com/mnras/article/537/2/2160/7958946` — journal
volume/issue/pages/DOI, independently of arXiv. The abstract (fetched
verbatim) states the core claim directly:

> "we argue that astrophysical processes are not independent of cosmology and
> that their coupling naturally leads to stronger baryonic feedback in
> cosmological models with suppressed structure formation... we also show
> that the dependence of baryonic feedback on cosmology can be modelled as a
> function of the ratio $f_{\rm b}/c_{\rm v}^2 \sim f_{\rm b}/(\Omega_{\rm
> m}\sigma_8)^{1/4}$... giving an accurate fitting formula for the baryonic
> suppression of the matter power spectrum."

This is a first-order match to the referee's description on every count: it
is FLAMINGO-adjacent, first-authored by Elbers, and shows non-factorizability
of the baryonic suppression with cosmology, with a quantified coupling the
paper itself names $\xi^2$ — exactly the "Elbers ξ² scaling" the referee
invokes. **`docs/WORKLOG.md:21`** independently confirms no one had verified
this yet: *"R4 (fixed cosmology) is a text-only quantitative caveat pending
Elbers citation verification."* This memo is that verification.

### 1.2 The quantitative content (verbatim quotes, cross-checked across 4 independent fetches)

Definitions (their Eq. 16, Eq. 23):
$$\xi \equiv \frac{V_{\rm g}}{V_{\rm max}} = \frac{\sqrt{f_{\rm b}\,GM/R}}{V_{\rm max}} = \frac{\sqrt{f_{\rm b}}}{c_{\rm v}}, \qquad \xi^2 = \frac{f_{\rm b}}{c_{\rm v}^2}, \qquad c_{\rm v} = 1.24\,(\Omega_{\rm m}\sigma_8)^{1/8}$$

Fitting formula and its calibrated sensitivity (their Eq. 21, 22, 24):
$$F_{\rm b} = 1-\exp(\alpha\,\xi^2+\beta), \qquad \frac{\Delta F_{\rm b}}{1-F_{\rm b}} = -\alpha\,\Delta(\xi^2), \quad \alpha = 13.8\pm0.6$$
$$\frac{\Delta F_{\rm b}}{1-F_{\rm b}} = -\alpha^\prime\,\Delta\!\left[\frac{\Omega_{\rm b}}{\Omega_{\rm b}+\Omega_{\rm c}}(\Omega_{\rm m}\sigma_8)^{-1/4}\right], \quad \alpha^\prime = 0.65\,\alpha = 8.98$$

$F_{\rm b}$ is their notation for the power-spectrum suppression ratio itself
(what this paper calls $S(\ell)$/$S(k)$) — confirmed by the internal
consistency check $0.65\times13.8=8.97\approx8.98$ (matches their stated
$\alpha^\prime$ to rounding) and by the qualitative statement (also quoted,
verbatim) *"baryonic effects are enhanced for lower $\sigma_8$... and lower
$\Omega_{\rm m}$"*, which is the correct sign for $F_{\rm b}=S$ under this
formula.

Stated magnitudes (verbatim, cross-checked twice independently against v1
and twice against v2 — identical wording each time):

> "Varying the cosmological parameters by a few percent produces corrections
> on scales $k>2\,{\rm Mpc}^{-1}$ of up to 4–5%, which is relatively large
> compared to the total baryonic effect of 10–15%."

> "the non-factorizable corrections are modest, with effects of 1%–2% for
> $k>2\,{\rm Mpc}^{-1}$, relative to the case without cosmological
> dependence"

> "the non-factorizable corrections are now 5–10% on non-linear scales,
> essentially doubling the strength of feedback" [for decaying dark matter
> models — a more extreme, non-ΛCDM extension, not our regime]

**Note on referee's phrase "at exactly the suppression level you predict":**
Elbers et al. quote their own effect as fractions of a "total baryonic
effect of 10–15%" — this is essentially identical to this paper's own
measured $S(\ell)$ trough depletion, $1-S=1-0.881=11.9\%$ (main.tex
l.895–896). The referee's phrasing is apt, not loose.

### 1.3 Confidence / verification method — read before citing

These quotes were extracted by fetching the arXiv HTML rendering and asking
an automated summarizer to quote verbatim text — **not** by a human or by
Claude directly reading the typeset PDF/proofs. I deliberately re-fetched
with differently-worded (one leading, one strictly neutral, "say NOT FOUND
rather than guess") prompts across two arXiv versions (v1, v2) and got
**identical** equation numbers, symbols, and numeric values every time,
including a non-trivial internal-consistency arithmetic check ($0.65\times
13.8=8.98$) that a confabulating summarizer would be unlikely to reproduce
by chance. I judge this good, not perfect, verification. **Recommend**: before
this goes in the submitted paper, the author (or a co-author with journal
access) should confirm Eq. 16/21/22/23/24 and the $\alpha=13.8\pm0.6$ /
$\alpha'=8.98$ values directly against the MNRAS proofs or PDF — a 30-second
check given the equation numbers above to search for. A direct PDF-text
fetch was attempted and failed (returned binary/compressed stream, not
parseable by the fetch tool) — this is a tooling limitation, not a red flag
on the paper.

### 1.4 Candidates considered and ruled out / not needed

Given how cleanly 2403.12967 matches (FLAMINGO, Elbers first-author,
explicit non-factorizability, explicit $\xi^2$ coupling, quantified
percentages, "$S_8$ tension" framing matching the referee's own words), I
did not need to fall back to adjacent candidates. For completeness, other
FLAMINGO/cross-cosmology baryon papers surfaced in the search that are
**not** the right citation but may be useful elsewhere in the paper: a
follow-up FLAMINGO "resummation model... precisely predicting matter power
suppression from observed halo baryon fractions" (MNRAS 545(2), search hit
only, not fetched/verified — likely Braspenning et al. or similar, not
Elbers as first author) and the general Schneider/van Daalen baryonification
literature already in `references.bib`/`refs_needed.md`. None of these were
needed as substitutes.

---

## 2. The "Paper-3 planning doc" — NOT FOUND as a standalone proposal

`REFEREE_PLAN.md:161` (this campaign's own planning doc, `papers/01_pipeline/REFEREE_PLAN.md`)
asserts: *"The Paper-3 planning doc already proposes this scaling argument —
lift it."* I searched for that antecedent doc and **could not find it**.
Searched (each grep scoped and timed out at 60s per the guardrails):

```
timeout 60 grep -rn -i "xi\^2|xi2|factoriz|cosmolog" papers/03_latent_sbi/ --include=*.md --include=*.tex
timeout 60 grep -n -i "elbers|xi\^2|xi2|factoriz" papers/03_latent_sbi/{leftovers.md,refs_needed.md}
timeout 60 grep -n -i "elbers|xi\^2|xi2|factoriz" papers/PLAN.md papers/README.md
timeout 60 grep -rln -i "xi\^2|xi2|factoriz|elbers" papers/02_cosmo_bias/ papers/04_ksz_gas/ papers/05_anisotropy/
timeout 60 grep -n -i "elbers|xi\^2|xi2|factoriz" docs/WORKLOG.md docs/ADVISOR_BRIEFING_2026-07-16.md docs/baryonify.md
unzip -l papers/03_latent_sbi_overleaf.zip   # confirms no extra file beyond the live tree
```

Results: `papers/03_latent_sbi/` (brief.md, dossier.md, main.tex) contains
**many** "fixed cosmology" / "TNG-only" caveat mentions (e.g. dossier.md:496–499,
main.tex:1222–1231 `\subsection{TNG-only, fixed-cosmology, and
interpolation-only caveats}`) but **none** propose the Elbers $\xi^2$
scaling or cite Elbers/FLAMINGO by name — they state the fixed-cosmology
limitation qualitatively, not with a quantitative bounding argument. The
01_pipeline hits for "factoriz" (dossier.md, ANALYTIC_LATENT_MODEL.md,
refs_needed.md, MODEL_RECIPE.md, `_build_figures_nb.py`) are all a
**different, unrelated use of the word** — the statistical/kSZ-estimator
"factorization" of $\theta\to\lambda\to$ statistic, nothing to do with
cosmology-baryon coupling. The only genuine, on-topic hits anywhere in the
repo are `docs/WORKLOG.md:21` (quoted above — flags this as *pending*, not
proposed) and `REFEREE_PLAN.md` itself (§R4, which **names** the idea and
gives the ask, but contains no derivation, no $\xi^2$ formula, no numbers —
it is the task specification, not the antecedent planning doc it claims
exists).

**Conclusion: the "Paper-3 planning doc" proposing this scaling argument is
not present anywhere I could search.** Either it lived only in a
conversation/session not captured to a file, it's a forward-looking
paraphrase by whoever wrote REFEREE_PLAN.md (i.e., "this is the kind of
argument Paper 3 should eventually make," not "Paper 3 already worked this
out"), or it exists somewhere outside the scoped search roots I'm permitted
to touch. I am not treating REFEREE_PLAN.md's claim as verified; **§3 below
is derived from scratch**, using the Elbers formula directly and this
paper's own quoted numbers, not lifted from a prior derivation.

---

## 3. The bound — full arithmetic

### 3.1 Setup

**Coupling** (Elbers et al. 2025, §1.2 above): $\Delta F_{\rm b}/(1-F_{\rm
b}) = -\alpha^\prime\,\Delta X$, with $X \equiv \dfrac{\Omega_{\rm
b}}{\Omega_{\rm m}}(\Omega_{\rm m}\sigma_8)^{-1/4} = \Omega_{\rm
b}\,\Omega_{\rm m}^{-5/4}\sigma_8^{-1/4}$ and $\alpha^\prime=8.98$ (using
$\Omega_{\rm b}+\Omega_{\rm c}=\Omega_{\rm m}$, no massive-neutrino
contribution to $\Omega_{\rm m}$ assumed, matching this paper's TNG300
cosmology). Log-derivatives: $\partial\ln X/\partial\Omega_{\rm m} =
-5/(4\Omega_{\rm m})$, $\partial\ln X/\partial\sigma_8 = -1/(4\sigma_8)$ —
**the $\Omega_{\rm m}$ sensitivity is 5$\times$ the $\sigma_8$ sensitivity**
in elasticity terms; this drives everything below.

**Fiducial cosmology** (main.tex l.395, TNG300/TNG300-Dark, Planck-2015):
$\Omega_{\rm m}=0.3089$, $\sigma_8=0.8158$, $\Omega_{\rm b}=0.0486$
$\Rightarrow X_0 = 0.22206$.

**This paper's own numbers used as the comparison scale** (main.tex,
quoted verbatim):
- Fiducial $S(\ell)$ trough: model predicts $S=0.898$, **measured $S=0.881$**
  (l.895–896) $\Rightarrow$ depletion $1-S=0.119$ used below.
- Predictive scatter $\sigma_{\rm pred}$: **0.003** in $S$ units at the
  van-Daalen-like scale $\ell\simeq1.5\times10^3$, **0.019** at the
  $\ell\simeq1.3\times10^4$ suppression trough (l.842–846) — i.e. the
  0.9σ/14% figure at l.896–898 is self-consistent ($|0.898-0.881|/0.9\approx0.019$).
- LSST-Y10 statistical floor on $S$: task-supplied assumption, **~0.005
  (0.5%)**, consistent in order of magnitude with main.tex's own
  area-scaled-covariance construction (Eqs. 8–10, l.558–572) though no
  single number is hardcoded there; I did not re-derive it independently
  (out of scope for this memo) — flagged for author cross-check.

**Stage-IV cosmology posterior widths** — sourced, not the task's
illustrative numbers taken on faith. I could not extract a clean table cell
from the canonical LSST DESC SRD (Mandelbaum et al. 2018, arXiv:1809.01669)
or Euclid Blanchard et al. (2020, arXiv:1910.09273) via automated fetch (both
are long, table-heavy documents; the fetch tool truncates before reaching
the results tables — repeated attempts returned "NOT FOUND" honestly rather
than guessing, which I take as the tool behaving correctly under
uncertainty). I did successfully pull a **real, directly-quoted table** from
a 2025 LSST-era forecast paper instead:

> **Wayland, Alonso & Zennaro (2025)**, "Calibrating baryonic effects in
> cosmic shear with external data in the LSST era," arXiv:2506.11943, Table 2
> (mock Stage-IV/LSST weak-lensing Fisher forecast, Planck-2020 fiducial
> $\Omega_{\rm m}=0.3097$, $\sigma_8=0.8102$):

| scenario | $S_8$ | $\Omega_{\rm m}$ |
|---|---|---|
| WL only, baryons fixed | $0.8181^{+0.0081}_{-0.0090}$ | $0.308^{+0.019}_{-0.018}$ |
| WL only (baryons marginalized) | $0.8220^{+0.0170}_{-0.0140}$ | $0.309^{+0.022}_{-0.021}$ |
| WL + long-term X-ray + long-term kSZ | $0.8196^{+0.0088}_{-0.0096}$ | $0.309^{+0.011}_{-0.012}$ |

I use two bracketing scenarios from this table:
- **"tight"** (best available, multi-probe-calibrated): $\sigma(S_8)=0.0092$, $\sigma(\Omega_{\rm m})=0.0115$
- **"loose"** (WL alone, no external calibration): $\sigma(S_8)=0.0155$, $\sigma(\Omega_{\rm m})=0.0215$

(Publication status of this paper: arXiv abstract page shows an
internally-inconsistent signal — a quoted "Journal reference: MNRAS 543(2),
1518–1534" alongside a DOI field that is still the generic
`10.48550/arXiv.2506.11943` arXiv-minted DOI rather than a
`10.1093/mnras/...` journal DOI. Real published-and-indexed arXiv abstract
pages show the journal DOI once assigned, so this is not self-consistent —
I flag it as **unverified** and cite as an arXiv e-print only; see bibtex
note in §5.)

I approximate $\delta\sigma_8\approx\delta S_8$ throughout (exact when
$\Omega_{\rm m}\approx0.3$, since $S_8\equiv\sigma_8\sqrt{\Omega_{\rm
m}/0.3}$; a small, sub-10% effect here given $\Omega_{\rm m,fid}\approx0.31$).

### 3.2 Propagation — two framings, to bound modeling-choice sensitivity

**Case A — worst-case coherent shift** ($\Omega_{\rm m}$, $\sigma_8$ both
shift the same sign by their respective 1σ; the maximal-adversarial case,
no covariance assumed). **Case B — the actual WL degeneracy direction**
($S_8$ held exactly fixed at its tightly-measured value, only the poorly
constrained orthogonal combination moves, parameterized by $\sigma(\Omega_{\rm
m})$ alone via $\delta\sigma_8=-\tfrac{\sigma_8}{2\Omega_{\rm
m}}\delta\Omega_{\rm m}$) — this is the physically realistic shape of a
single-probe cosmic-shear posterior, and turns out to give **nearly the same
answer as Case A**, because the $\Omega_{\rm m}$ elasticity so dominates the
$\sigma_8$ one that whether they move together or oppositely barely matters.

Full script + output (saved at `papers/01_pipeline/referee/work/r4_elbers_bound.py`,
alongside the other R-agents' work scripts in this campaign; re-run with
`/mnt/home/mlee1/venvs/BIND_env/bin/python3 r4_elbers_bound.py` to audit):

```
X0=0.22206  dlnX/dOm=-4.0466 (per unit Om)  dlnX/ds8=-0.3064 (per unit s8)

CASE A -- coherent worst-case shift
  [tight]  sigma(S8)=0.0092 sigma(Om)=0.0115
    dlnX=-0.0494 -> Delta(1-Fb)/(1-Fb)=-9.84% -> Delta S = +0.0117
    |Delta S| / sigma_pred(trough,0.019) = 0.62x
    |Delta S| / sigma_pred(vD-scale,0.003) = 3.90x
    |Delta S| / LSST-Y10(0.005) = 2.34x
  [loose]  sigma(S8)=0.0155 sigma(Om)=0.0215
    dlnX=-0.0918 -> Delta(1-Fb)/(1-Fb)=-18.30% -> Delta S = +0.0218
    |Delta S| / sigma_pred(trough,0.019) = 1.15x
    |Delta S| / sigma_pred(vD-scale,0.003) = 7.26x
    |Delta S| / LSST-Y10(0.005) = 4.35x

CASE B -- WL degeneracy direction (S8 held fixed, only Om free)
  [tight]  sigma(Om)=0.0115
    dlnX=-0.0419 -> Delta(1-Fb)/(1-Fb)=-8.35% -> Delta S = +0.0099
    ratios: 0.52x sigma_pred(trough), 3.31x sigma_pred(vD), 1.99x LSST floor
  [loose]  sigma(Om)=0.0215
    dlnX=-0.0783 -> Delta(1-Fb)/(1-Fb)=-15.61% -> Delta S = +0.0186
    ratios: 0.98x sigma_pred(trough), 6.19x sigma_pred(vD), 3.72x LSST floor

CASE C -- Om held fixed, only sigma8/S8 free (for reference only)
  [tight]  Delta S = +0.0007   (0.04x sigma_pred trough -- negligible)
  [loose]  Delta S = +0.0011   (0.06x sigma_pred trough -- negligible)

Cross-check against Elbers+2025's own worked example ("a few percent -> 4-5%"):
  coherent 1.0% Om & s8 shift -> Delta(1-Fb)/(1-Fb) = -2.99%
  coherent 1.5% Om & s8 shift -> Delta(1-Fb)/(1-Fb) = -4.49%   <- matches "4-5%" almost exactly
  coherent 2.0% Om & s8 shift -> Delta(1-Fb)/(1-Fb) = -5.98%
```

**Sanity check passed**: plugging in Elbers et al.'s own example magnitude
("a few percent" $\simeq1.5\%$ simultaneous $\Omega_{\rm m}$/$\sigma_8$
shift) into this same formula reproduces their quoted "4–5%" almost exactly
(4.49%), confirming the formula is being applied correctly and not
mis-scaled. (An earlier draft of this arithmetic had a units bug —
conflating $\partial\ln X/\partial\Omega_{\rm m}$ with the dimensionless
elasticity $\partial\ln X/\partial\ln\Omega_{\rm m}$ — that overstated the
effect by a factor of $\sim1/\Omega_{\rm m}\approx3.2$; caught precisely by
this cross-check, which is why it's included here rather than trimmed.)

**Mechanistic note** (Case C, above): the bound is driven almost entirely by
$\Omega_{\rm m}$, not $\sigma_8$/$S_8$ — a survey that pinned $\Omega_{\rm
m}$ perfectly (not achievable by weak lensing alone; that's exactly why
cosmic shear reports $S_8$ rather than $\Omega_{\rm m}$ and $\sigma_8$
separately) would shrink the induced error to $\lesssim0.001$, well under
every floor. **This is why the caveat bites specifically for
WL-focused Stage-IV analyses**: they measure $S_8$ tightly but leave
$\Omega_{\rm m}$ comparatively free, and $\Omega_{\rm m}$ is exactly the
direction the Elbers coupling is most sensitive to.

### 3.3 Headline numbers

$$|\Delta S|_{\rm trough} \approx 0.010\text{--}0.022 \;\; (\text{tight-to-loose Stage-IV bracket, both framings agree to} \lesssim20\%)$$

Rounding to one number for the caveat text: **|ΔS| ≈ 0.01–0.02** at the
suppression trough, i.e.:
- **≈0.5–1.2×** this paper's own predictive scatter there ($\sigma_{\rm pred}=0.019$) — comparable, not dwarfed;
- **≈3–7×** this paper's predictive scatter at well-measured scales ($\sigma_{\rm pred}=0.003$);
- **≈2–4×** the raw LSST-Y10 statistical floor ($\sim0.005$).

This is a real, non-negligible systematic at the level this paper is trying
to control for other effects (e.g. the $[5$–$10\%]$ gas-spectra biases
already flagged, or the $\sigma_{\rm pred}$ the model ships with) — not
something that can be waved off as "surely subdominant." It is also,
reassuringly, **bounded and characterizable**: it is a smooth, monotonic,
first-order-calculable function of $(\Omega_{\rm m},\sigma_8)$ displacement
via a published fitting formula, not an unknown unknown.

---

## 4. Drafted text

### 4.1 Caveat paragraph (§sec:caveats, item 3)

Ready to replace `% 3 fixed cosmology -> varied-cosmology DMO painting`:

> A third limit is cosmological. \bind{} itself is not architecturally
> fixed-cosmology: the underlying flow model is trained on the SB35 suite,
> whose simulations vary five cosmological parameters alongside the thirty
> astrophysical ones, so painting a DMO substrate run at a different
> cosmology from TNG300 is in-architecture --- a real asymmetry with
> fixed-cosmology BCM fits, which must be recalibrated at every cosmology by
> construction. What \emph{is} fixed to one cosmology, and unproven beyond
> it, are the TNG300-Dark substrate itself, the measured halo-property
> atlas, and the eight-latent map $\bm{M}$ of \S~\ref{sec:why_latents}, all
> calibrated at the single \citet{planck-2015} point of the parent suite
> ($\Omega_m=0.3089$, $\sigma_8=0.8158$). \citet{Elbers-2025} show, with the
> FLAMINGO suite, that baryonic feedback and cosmology are not factorizable
> at exactly this suppression level: the coupling, controlled by the ratio
> $\xi^2\equiv f_{\rm b}/c_{\rm v}^2$ of baryon fraction to
> (cosmology-dependent) halo concentration, moves the small-scale
> suppression by several percent of its own depth for few-percent shifts in
> $\Omega_m$ and $\sigma_8$ --- comparable to the $12\%$ depletion we
> measure at the $S(\ell)$ trough. Propagating their calibrated coupling
> ($\alpha^\prime=8.98$; their Eq.~24) through a Stage-IV-width cosmology
> posterior ($\sigma(S_8)\sim0.01$--$0.02$, $\sigma(\Omega_m)\sim0.01$--$0.02$;
> \citealt{Wayland-2025}) bounds the induced error on our trough at
> $|\Delta S|\approx0.01$--$0.02$: comparable to our own model's predictive
> scatter there ($\sigma_{\rm pred}=0.019$), and several times both its
> predictive scatter at better-measured scales ($\sigma_{\rm pred}=0.003$)
> and the raw LSST-Y10 statistical floor ($\sim0.5\%$). We therefore do not
> treat the atlas or $\bm{M}$ as cosmology-independent; unlike for a fixed
> BCM template, the remedy is architectural rather than conceptual, since
> \bind{} already conditions on cosmology.

(~7 sentences, one paragraph, matches the density of the neighboring
"model's range of validity" paragraph just above §Caveats. The
$\bm{M}$/$\sigma_{\rm pred}$/$\bm{\lambda}$ notation matches
§sec:why_latents' own definitions at l.839, 923.)

### 4.2 Outlook mirror sentence (§sec:outlook)

> Extending the painter's cosmology-conditioning --- already present in its
> architecture via the five SB35 cosmological parameters, merely unexercised
> in this application because TNG300-Dark itself is single-cosmology --- to
> a varied-cosmology DMO substrate would let the atlas and $\bm{M}$ be
> measured and validated as an explicit function of cosmology, replacing the
> bound of \S~\ref{sec:caveats} with a direct measurement.

This is consistent with the Conclusions paragraph's existing language
("painting across cosmologies" is already named there, l.955, as one of
"the most direct extensions... this paper's architecture specifies") — the
outlook sentence just gives that phrase its own home in §6 with the
Elbers-bound context attached.

---

## 5. BibTeX

Both entries verified per §1.3/§3.1; paste into whichever `.bib` file
main.tex actually reads (see filename note below — **do not** assume
`references.bib` is it).

```bibtex
@ARTICLE{Elbers-2025,
   author = {{Elbers}, Willem and {Frenk}, Carlos S. and {Jenkins}, Adrian and
             {Li}, Baojiu and {Helly}, John C. and {Kugel}, Roi and
             {Schaller}, Matthieu and {Schaye}, Joop and {Braspenning}, Joey and
             {Kwan}, Juliana and {McCarthy}, Ian G. and {Salcido}, Jaime and
             {van Daalen}, Marcel P. and {Vandenbroucke}, Bert and {Pascoli}, Silvia},
    title = "{The FLAMINGO project: the coupling between baryonic feedback and cosmology in light of the $S_8$ tension}",
  journal = {Monthly Notices of the Royal Astronomical Society},
     year = 2025,
   volume = {537},
   number = {2},
    pages = {2160--2178},
      doi = {10.1093/mnras/staf093},
archivePrefix = {arXiv},
   eprint = {2403.12967},
 primaryClass = {astro-ph.CO},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.537.2160E},
  adsnote = {Bibcode constructed via the standard ADS algorithm from
             independently-verified year/journal/volume/first-page/first-author-initial
             (ADS itself did not render through the automated fetch tool used
             for this verification pass -- confirm on ADS/INSPIRE before submission).
             Month field omitted: MNRAS publication month not independently confirmed.}
}

@ARTICLE{Wayland-2025,
   author = {{Wayland}, Amy and {Alonso}, David and {Zennaro}, Matteo},
    title = "{Calibrating baryonic effects in cosmic shear with external data in the LSST era}",
  journal = {arXiv e-prints},
     year = 2025,
archivePrefix = {arXiv},
   eprint = {2506.11943},
 primaryClass = {astro-ph.CO},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250611943W},
  adsnote = {CAUTION -- publication status unverified: one automated fetch
             reported "Journal reference: MNRAS 543(2), 1518-1534" for this
             paper but the same page's DOI field showed only the generic
             arXiv-minted DOI (10.48550/arXiv.2506.11943), not a journal DOI
             -- an internally inconsistent signal. Cite as an arXiv e-print
             (as above) unless/until independently confirmed via
             ADS/INSPIRE/the journal site. Used here only to source
             Stage-IV LSST-like Fisher-forecast widths on S8/Omega_m
             (its Table 2), not for any claim requiring journal status.}
}
```

**Bibliography filename mismatch** (task item 5, confirmed real): main.tex
l.1039 has `\bibliography{biblio}{}`, i.e. it looks for **`biblio.bib`**, but
the file actually present in `papers/01_pipeline/` is **`references.bib`**
(no `biblio.bib` exists there). Neither entry above currently exists in
`references.bib` (checked, zero hits for "elbers"/"wayland"). **The author
needs to resolve which file main.tex is actually meant to read** (rename
`references.bib`→`biblio.bib`, or fix the `\bibliography{}` argument) before
either of these entries — or anything else in `references.bib` — will
actually compile into the bibliography. This is a pre-existing issue, not
something introduced by this memo.

---

## 6. Summary table for the author

| Item | Status |
|---|---|
| Elbers reference identity | **Verified**: arXiv:2403.12967, MNRAS 537(2):2160–2178 (2025), DOI 10.1093/mnras/staf093, Elbers first author |
| Elbers $\xi^2$ formula + $\alpha'=8.98$ | Verified via 4 independent cross-checked fetches + internal-consistency arithmetic check + reproduces their own "4–5%" example |
| Confidence caveat | Extracted via automated HTML fetch/summarize, not hand-read PDF proofs — recommend a final human spot-check of Eqs. 16/21/22/23/24 |
| Paper-3 planning doc (xi² proposal) | **NOT FOUND** in any searched location; REFEREE_PLAN.md's claim that one exists is itself unverified — treat §3 here as the first derivation, not a "lift" |
| Stage-IV widths source | Wayland, Alonso & Zennaro, arXiv:2506.11943, Table 2 (publication status flagged unverified) |
| Computed bound | **\|ΔS\| ≈ 0.01–0.02** at the $S(\ell)$ trough (0.5–1.2× $\sigma_{\rm pred}$ there; 3–7× $\sigma_{\rm pred}$ at well-measured scales; 2–4× LSST-Y10 floor) |
| Caveat paragraph | Drafted, §4.1 above |
| Outlook sentence | Drafted, §4.2 above |
| BibTeX | Drafted, §5 above, for **both** references |
| `\bibliography{biblio}` vs `references.bib` | Confirmed real mismatch; unresolved, flagged for author |
