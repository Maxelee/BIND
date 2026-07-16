# References needed — Paper II (02_cosmo_bias)

Every `\citep`/`\citet`/`\citealt` key used in `main.tex`, for a later citer
agent to resolve to real BibTeX entries in `references.bib`. Do NOT invent
DOIs/years/venues here — flag uncertainty explicitly where the dossier
flagged it.

## Confirmed / named explicitly in the mined sources

- **schneider2020fisher** — Schneider et al. 2020, the linearized Fisher-bias
  formalism (`Δθ = F⁻¹B`) explicitly named in `lightcone_cosmo_bias.py` and
  `cosmo_bias_capstone.py` docstrings as "standard linearised Fisher bias
  (Schneider et al. 2020)". Distinct from `schneiderTeyssier2015` (BCM
  model) and `schneider2019bcm` (brief's seed "Schneider+2019 (BCM)") below —
  verify these are not the same paper cited three ways.
- **anguloWhite2010** — Angulo & White (2010), arXiv:0912.4277, the AW10
  cosmology-rescaling method used in Appendix A. Explicitly named in
  `docs/cosmo_rescaling_plan.md` §1 and the WORKLOG 2026-07-03 entry.
- **robertson2026** — Robertson et al. (2026), the LSST baryon-marginalization
  forecast setup being reproduced/extended in Section 2.6/3.4 ("Robertson
  Table 1"/"Table 2" cited for the LSST-Y1 survey spec and priors). **GAP:
  exact title/journal/arXiv ID not resolvable from any in-repo source** —
  only "Robertson et al. (2026)" + table numbers appear in engine docstrings.
  Citer agent MUST find the real reference via external search before this
  paper can be submitted.
- **lin2026** — Lin et al. 2026, "One latent to fit them all", arXiv:2509.01881.
  Exact title/arXiv ID given directly in WORKLOG 2026-06-24 "WL feedback
  latents" entry — verify exact author list.
- **schneiderTeyssier2015** — Schneider & Teyssier (2015), the BCM baryon-correction
  model, used via CCL's `BaryonsSchneider15` in `lightcone_model_comparison.py`.
- **vanDaalen2019** — van Daalen et al. 2019, used via CCL's `BaryonsvanDaalen19`
  in `lightcone_model_comparison.py`. Note: a *2020* van Daalen reference also
  appears elsewhere in this project (bridge paper, WORKLOG:1073, different
  figure) — confirm 2019 is correct for the CCL `BaryonsvanDaalen19` API
  function used specifically here, not the 2020 paper.
- **chisari2019** — Chisari et al. 2019, the standard CCL/pyccl paper. Used
  throughout (pyccl v3.3.4, WORKLOG 06-22); this is the brief's seed citation
  and a safe default, but the author name itself does not appear in-source —
  citer should verify.
- **foremanMackey2013** — Foreman-Mackey, Hogg, Lang & Goodman 2013, `emcee`.
  Used directly (`import emcee`, `EnsembleSampler`) in
  `lightcone_shear_forecast.py`. Standard citation, package usage unambiguous.
- **constantine2015** — Constantine (2015), the linearized "active subspace"
  method explicitly named in `lightcone_beyond2pt.py`'s `manifold()`
  docstring. Not in the brief's seed list — verify exact title (likely
  "Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies", SIAM 2015).
- **bcemu2025** — BCemu v2.0.5, the baryonification emulator used as a third
  analytic-model family in the Step-5 / Section 3.6 comparison (`import
  BCemu`, `BCemu.BCemu2025()`). Not in the brief's seed list; verify the
  correct citation for the BCemu package/paper (likely Giri & Schneider).
- **chen2019** — Chen et al. 2019, the β-TCVAE / total-correlation
  decomposition, explicitly named in WORKLOG:596 as the method behind the WL
  feedback latent analysis referenced in Discussion §4.1 (Paper III, in
  prep.). Not in the brief's seed list; verify exact title (likely "Isolating
  Sources of Disentanglement in Variational Autoencoders", NeurIPS 2018/2019).
- **meadPeacock2014a**, **meadPeacock2014b** — Mead & Peacock (2014a, 2014b),
  arXiv:1308.5183 and arXiv:1408.1047, cited in the rescaling plan doc for
  halo-catalogue/concentration extensions to AW10 (Appendix A). Not in the
  brief's seed list (which instead lists "Mead+2015/2021" for HMcode — see
  `mead2015`/`mead2021` below, a DIFFERENT pair of Mead references).
- **bacco2021** — the BACCO project, arXiv:2004.06245, cited in the rescaling
  plan doc as precedent for rescaling-based emulator training (1-3%
  accuracy). Verify exact author list/year (Angulo et al. 2021?).
- **learningTheUniverse2026** — "Learning the Universe", arXiv:2606.10024,
  cited in the rescaling plan doc as a second rescaling-for-ML-training-set
  precedent. Verify exact author list/year.
- **tinker2008** — Tinker et al. 2008, the halo mass function used as the
  target-cosmology HMF comparison in `rescale_validation.py` (named directly
  as "Tinker08" in the plan doc's validation table).

## Brief's seed citations NOT corroborated in the mined sources

These are Intro/background-literature citations with zero in-repo
corroborating evidence (expected — they are standard-practice background,
not analysis-engine citations). The writer used them in Section 1 based on
the brief's seed list alone; the citer agent must source real
BibTeX entries independently and should not claim in-repo provenance.

- **mead2015**, **mead2021** — HMcode (Mead et al. 2015; Mead et al. 2021).
  NOTE: distinct from `meadPeacock2014a/b` above (2014 papers about AW10
  halo-catalogue extensions) — do not conflate the two Mead reference pairs.
- **eifler2015** — Eifler et al. 2015 (baryon PCA nuisance approach).
- **huang2019** — Huang et al. 2019 (baryon PCA nuisance approach).
- **amonEfstathiou2022** — Amon & Efstathiou 2022.
- **preston2023** — Preston et al. 2023.
- **desY3Amon2022** — DES Y3, Amon et al. 2022.
- **desY3Secco2022** — DES Y3, Secco et al. 2022.
- **kids1000Asgari2021** — KiDS-1000, Asgari et al. 2021.
- **hscY3** — HSC Y3 cosmic shear (exact reference/year not specified in the
  brief; citer must identify the correct HSC Y3 cosmic-shear paper).
- **lsstDescSrd2018** — LSST DESC Science Requirements Document (2018).

## Ambiguous / needs disambiguation before BibTeX entry is written

- **schneider2019bcm** — used in the Introduction (Section 1) alongside
  `schneiderTeyssier2015` to represent "Schneider+2019 (BCM)" per the
  brief's seed list. This may be the same underlying baryonification
  framework as `schneiderTeyssier2015` (2015) cited under a different year,
  or a genuinely distinct 2019 paper (e.g., Schneider, Teyssier, Potter et
  al. 2019 extending the BCM to more free parameters). The citer agent
  should verify whether these are two distinct citable works or whether one
  of the two keys in `main.tex` should be merged/removed.

## Suite cross-references (in-prep, this paper series)

Used in `main.tex` as `\citep[Paper~N]{bindN}` per the suite convention.
Both need a `references.bib` stub entry (e.g. `@misc`/`@unpublished` with
`note = {in preparation}`) once the author list/target venue is finalized —
not a real external reference to search for.

- **bindI** — Paper I (in prep.), the BIND lightcone pipeline / emulator
  paper this paper builds on (Section 2.1, "The BIND lightcone Sobol suite",
  and the Introduction). Cited via `\citep[Paper~I]{bindI}`.
- **bindIII** — Paper III (in prep.), the WL feedback β-TCVAE latent-space
  paper referenced in Discussion §4.1 ("Why two?") and the
  Abstract/Conclusions as corroborating evidence for the "why two" claim
  (branch `analysis/sobol-sb35` per the dossier's PLAN.md mapping). Cited via
  `\citep[Paper~III]{bindIII}`.
