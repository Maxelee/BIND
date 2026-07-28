# High-impact paper rubric — SZ stacking × feedback confrontation

Synthesized 2026-07-28 from a three-agent literature study of the highest-cited
papers in this niche (Schaan+21 ~200 cites, Amodeo+21 ~157, Hadzhiyska+24/25,
Bigwood+24 ~131 & Bigwood+25 benchmark, McCarthy+25 FLAMINGO, van Daalen+20
~195, Amon & Efstathiou 22 ~237, Popesso+24 eROSITA, Qu+26 DESI DR2×ACT,
Schaye+23 FLAMINGO ~315, Hand+12 ~290) plus the referee-standards and
figure-convention literature (Moser+23 systematics audit, Amodeo erratum,
covariance-methodology papers). Applies to: the P4c paper
(`examples/paper_p4c_tsz_ksz.ipynb` and its eventual manuscript).

## How to grade

Each criterion scores **0 (absent) / 1 (partial, or present but not to the
niche's standard) / 2 (fully meets the standard set by the cited exemplars)**.
Category score = mean of its criteria / 2 × category weight. Total /100.
Grade the *paper as represented by the notebook + repo docs*
(`examples/_build_p4c_paper_nb.py` is the full source of the notebook;
`docs/p4c_decisions_table.md`, `docs/p4c_referee_hardening_plan.md` count as
part of the paper's supporting material). Every score MUST cite its evidence
(section/figure/line in the notebook source, or "absent"). Manuscript-only
matters (author lists, journal formatting) are out of scope.

Calibration: the exemplar papers score ~85–95. A solid but unexceptional
PRD/MNRAS paper scores ~60–70. Grade against the 2026 standard (Qu+26 /
Bigwood+25 era), not the 2021 standard.

---

## A. Claim & framing (weight 12)

- **A1.** Abstract quotes ≥1 headline number *with uncertainty* (σ-significance,
  asymmetric CI, or constraint ± error) — not qualitative language.
  (Universal among top papers: 6.5σ/10σ Schaan; S8=0.823+0.019−0.020 Bigwood;
  ε=(40±9)e−6 Amodeo.)
- **A2.** Abstract closes on interpretation hooked to a *named* tension/debate
  (S8 tension, missing baryons, feedback-strength ranking of named sims) in the
  same breath as the number.
- **A3.** Title follows the niche convention: "[Instrument/Suite]: description"
  or a hedged claim-first title ("Evidence for…", "…?"). Generic thematic
  titles fail.
- **A4.** A large, quotable scale statement in the abstract (sample size, sky
  area, design size: "160,150 LRGs", "253 nodes", "50 realizations × 25 deg²").
- **A5.** Defensible novelty/superlative framing ("first map-level many-model
  SZ confrontation", "highest S/N X to date") — claimed where true, absent
  where not.

## B. Measurement rigor (weight 22) — table stakes per the referee study

- **B1.** Headline robust across ≥2 independently component-separated maps;
  the deprojection choice defended and the excluded variant shown.
- **B2.** Null suite: random-position AND rotation/shuffle nulls, consistent
  with zero in the fit range, with PTEs or σ stated.
- **B3.** Two independent estimators or covariance methods cross-checked
  (jackknife vs bootstrap; real vs harmonic; second pipeline).
- **B4.** Covariance-inversion bias (Hartlap) corrected for every
  sample-estimated covariance that is inverted.
- **B5.** Satellite fraction and miscentering quantified with explicit bounds
  or forward-modeled with stated f_sat and calibration.
- **B6.** Mass anchor probe-independent of the fitted tracer (lensing-based),
  with its uncertainty propagated as a template/nuisance.
- **B7.** Goodness-of-fit per subsample AND joint (χ²/dof + PTE), so a reader
  can see what drives any rejection.
- **B8.** End-to-end pipeline validation against external references
  (known clusters; a published measurement reproduced).

## C. Model-confrontation credibility (weight 22)

- **C1.** Rejection stated against a pre-defined statistical threshold
  computed from the full covariance (not eyeballed; threshold registered
  before the comparison).
- **C2.** Two-halo term explicitly modeled (fixed or free amplitude) or its
  neglect quantitatively justified for the filter/aperture used.
- **C3.** Emulator/interpolation error validated on held-out data with a
  stated accuracy number AND propagated into the likelihood.
- **C4.** Plausibility checks on nuisance/rescaling freedoms: any parameter
  value required to reconcile model and data is checked against independent
  constraints (lensing mass, HOD) and flagged if implausible.
- **C5.** Model-family scope honesty: multi-suite comparison (independent
  subgrid codes), or the single-family limitation stated prominently
  (abstract/conclusions, not only a caveat section) with the claim scoped
  accordingly ("no IllustrisTNG-family variant", not "no simulation").
- **C6.** Box-size / cosmic-variance convergence of the sim side addressed
  (converged volume cited, or sub-volume scatter propagated into the tension
  significance).
- **C7.** Look-elsewhere / design-dependence of parameter-level claims
  controlled with an explicit null (ESS-matched or equivalent).
- **C8.** Internal consistency across sample splits (z bins, mass cuts,
  footprint halves) with a stated criterion and willingness to down-weight a
  failing split.

## D. Cosmological stakes (weight 14)

- **D1.** A suppression-curve figure — P(k)/P_DMO (or S(ℓ)) with the
  measurement's implied band and ≥3 named sims threading it — translating the
  gas result into the lensing-relevant statement. (The van Daalen axis; the
  niche's canonical closing figure.)
- **D2.** Conclusions quantitatively close the loop to S8 / power suppression
  ("our f_gas implies suppression at k=1 of X%, at the strong end of
  cosmic-shear priors"), echoing the intro stakes.
- **D3.** A multi-probe win/lose ledger: how the preferred feedback region
  fares against each independent probe (kSZ, tSZ, X-ray, WL), itemized rather
  than a single verdict (the McCarthy+25 pattern).
- **D4.** A specific, falsifiable future test named (which survey, which
  observable, what it would show if we're wrong) — not a generic "future
  surveys will improve this".

## E. Figures & presentation (weight 15)

- **E1.** Money figure overlays the measured data vector (with errors) on ≥3
  named model curves/variants with distinct styles, fit range marked on-figure.
- **E2.** Profile/spectrum comparison panels carry residual or ratio
  sub-panels (data/model or model/fiducial) — the niche's most universal
  convention.
- **E3.** On-figure physical-scale markers (R500/R200/θ200 vertical lines)
  and σ/S-N annotations placed on panels, legible standalone.
- **E4.** Posterior/contour figures include the reference model (TNG fiducial
  / Planck / cosmic value) as an explicit marker in-panel.
- **E5.** A compact main-text significance/systematics summary table (per
  effect), with the granular budget allowed in an appendix — all significance
  numbers buried in prose is a fail.
- **E6.** Methods separate "what was measured" from "what was assumed/modeled"
  by explicit structure (the Schaan/Amodeo companion split, internalized as
  section structure).

## F. Reusability & products (weight 15) — the citation engines

- **F1.** The measured data vector + covariance released in a directly
  reusable form (machine-readable table with documented conventions).
- **F2.** The analysis pipeline public and reusable by others (ThumbStack
  precedent: tool reuse compounds citations independently of the result).
- **F3.** A drop-in product for other analyses: a closed-form relation or a
  ready-to-use constraint (e.g., the gas-plane posterior as a prior for
  cosmic-shear baryon marginalization — the van Daalen fitting-function
  pattern).
- **F4.** Chains/posterior samples released with column documentation.
- **F5.** A reproducibility map: figure → product → code → verdict, complete
  enough for an outsider to regenerate every figure.

---

## Scoring sheet template

| Criterion | Score (0/1/2) | Evidence |
|---|---|---|
| A1..F5 | | |

Category subtotal = mean/2 × weight; total /100. Report the three lowest
categories and the five cheapest single-criterion gains.
