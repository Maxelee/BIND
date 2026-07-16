# References needed — Paper III (2-D latent + SBI)

Every `\citep`/`\citet`/`\citealp` key used in `main.tex`, for the Cite agent
to verify against arXiv/ADS and add to `references.bib`. Status notes carry
forward what the dossier/mining pass already established; "verify
independently" means the mining pass did not find the exact citation
spelled out anywhere in the repo (WORKLOG, docstrings, notebooks) and it
must be checked before use.

## Suite cross-references (in prep)

- `bindI` — Paper I, in prep. Working title (per `PLAN.md`): "Baryon-painted
  weak-lensing and SZ lightcones from dark-matter-only simulations."
- `bindII` — Paper II, in prep. Working title: "Two nuisance parameters
  suffice to marginalize baryonic feedback in LSST-era cosmic shear."
- `bindIV` — Paper IV, in prep. Working title: "Gas thermodynamics across a
  30-parameter feedback space: BIND vs DESI×ACT kSZ and eROSITA."

## Brief's seed citations

- `Lin2026latent` — Lin, Li, Genel, Villaescusa-Navarro+ 2026, "One latent
  to fit them all," arXiv:2509.01881. CONFIRMED verbatim in
  `wl_feedback_vae.py` docstring; this is the paper's central literature
  anchor (2-D latent for $T^2(k,a)=P_{\rm hyd}/P_{\rm dmo}$ across CAMELS
  suites), and this paper's WL result is presented as its direct analogue.
- `Cranmer2020SBI` — Cranmer, Brehmer & Louppe 2020, SBI review (PNAS). Not
  found by name in mined sources; standard SBI-review citation, verify
  independently.
- `PapamakariosMurray2016` — Papamakarios & Murray 2016, "Fast
  $\epsilon$-free Inference of Simulation Models with Bayesian Conditional
  Density Estimation" (SNPE). Not found by name in mined sources; the `sbi`
  package's NPE method is used throughout, so this is methodologically
  appropriate but unconfirmed as an explicit in-repo citation. Verify exact
  venue/year.
- `Greenberg2019APT` — Greenberg, Nonnenmacher & Macke 2019, "Automatic
  Posterior Transformation for Likelihood-Free Inference" (SNPE-C / APT).
  Same status as above — not found by name in-repo, methodologically
  standard, verify independently.
- `TejeroCantero2020sbi` — Tejero-Cantero et al. 2020, "sbi: A toolkit for
  simulation-based inference," JOSS. Methodologically CONFIRMED: the `sbi`
  Python package is used directly and by name across `lightcone_cl_sbi.py`,
  `wl_latent_sbi.py`, `sobol_perhalo_sbi.py`. The package-paper citation
  itself was not spelled out in any docstring; verify exact JOSS
  volume/year.
- `Talts2018SBC` — Talts, Betancourt, Simpson, Vehtari, Gelman 2018,
  "Validating Bayesian Inference Algorithms with Simulation-Based
  Calibration." Methodologically CONFIRMED: `sbc_ranks` / rank-histogram
  calibration used in both `wl_latent_sbi.py` §3 and `sobol_sbi.ipynb`. Not
  spelled out by name in-repo; standard citation, verify arXiv id.
- `Cheng2020WST` — Cheng & Ménard (or Cheng+) 2020, wavelet scattering
  transform for cosmology. Methodologically CONFIRMED: `bind.inference.stats`
  gained a `wst` statistic via `kymatio` (WORKLOG 2026-06-23). Exact
  author list/year not spelled out in-repo; verify (there are several
  Cheng WST papers — confirm which one is the intended reference, likely
  Cheng, Ménard, Melchior 2020/2021 or Cheng & Ménard 2021).
- `Hotelling1936CCA` — Hotelling 1936, "Relations between two sets of
  variates" (foundational canonical-correlation-analysis reference).
  Methodologically CONFIRMED: `sklearn.cross_decomposition.CCA` used
  directly in `wl_stat_latents.py` for exactly the canonical-correlation
  measurements reported in Results (2). The specific citation key is a
  standard placeholder for "the CCA method"; a more recent
  methods/textbook reference may be preferred by the Cite agent instead.
- `VillaescusaNavarro2021CAMELS` — Villaescusa-Navarro et al. 2021, the
  CAMELS project paper. Not spelled out by name in the Paper-III-specific
  mined sources (BIND/SB35 is CAMELS-adjacent, built on IllustrisTNG); this
  is the standard suite citation, likely also cited in Papers I/II. Verify.
- `Gatti2021DESACT` / `Pandey2022DESACT` — Gatti+ and Pandey+ (DES Y3 × ACT
  shear × Compton-$y$ cross-correlation). Author names CONFIRMED in
  `desact_sheary_realfit.py` docstring: "FIRST REAL-DATA confrontation:
  BIND shear x y vs DES Y3 x ACT (Pandey/Gatti+)"; data product
  `shivampcosmo/ACTxDESY3` `DES_ACT.fits`. Exact publication year(s) NOT
  confirmed by the mining pass — verify against the actual DES/ACT
  cross-correlation paper(s) that produced this specific data release
  (there may be one Gatti et al. paper and one Pandey et al. paper, or
  these may be the same paper with co-authorship — check).
- `Troster2021KiDSPlanck` — Tröster et al. 2021, KiDS-1000 × Planck tSZ.
  Not found in mined sources; verify independently (standard background
  citation for shear × tSZ cross-correlations, used in Introduction only,
  no attached number).
- `HillSpergel2014` — Hill & Spergel 2014. Not found in mined sources;
  verify independently (standard tSZ/kSZ×CMB-lensing background citation,
  Introduction only, no attached number).
- `Battaglia2012pressure` — Battaglia, Bond, Pfrommer, Sievers 2012,
  pressure-profile paper. Not found in the Paper-III-scoped mined sources
  (a *different* GNFW pressure profile, Hadzhiyska+26, is referenced in
  Paper-IV-adjacent files per WORKLOG); verify independently before use,
  and confirm it is the intended profile reference rather than a
  Paper-IV-specific one.
- `Amodeo2021kSZ` — Amodeo et al. 2021, ACT DR5 kSZ gas-profile paper.
  Found in Paper-IV-adjacent `profiles.py` docstring ("ACTxDESI /
  Schaan+21 / Amodeo+21"); used here only as general kSZ/gas-extent
  background in the Introduction, not for a specific number — verify this
  is an appropriate general citation or whether it belongs more properly
  to Paper IV only.

## Additional citations added by the Writer (literature texture; not
## explicitly seeded by the brief, harvested from `wl-tsz-bridge/examples/draft.tex`
## per the dossier's "additional citation keys found in-repo" list — NONE
## of these were cross-checked against arXiv/ADS by the mining pass)

- `vanDaalen2020` — van Daalen et al. 2020, baryonic suppression of the
  matter power spectrum from cosmic shear. Used in Introduction to
  motivate the baryon-suppression problem generally; no number attached.
  Verify.
- `Schneider2022clusters` — Schneider et al. 2022, review of baryonic
  effects on cosmological probes (clusters/baryonification context). Used
  in Introduction/Discussion; no number attached. Verify.
- `Arico2024` — Aricò et al. 2024, baryonification model. Used in
  Introduction to contrast baryonification vs.\ simulation-emulation
  strategies; no number attached. Verify.
- `Schaye2023FLAMINGO` — Schaye et al. 2023, the FLAMINGO simulation suite.
  Used in Introduction and Discussion (TNG-only caveat, alternate-suite
  comparison); no number attached. Verify.
- `Amonetal2023` — Amon et al. 2023, WL baryon-feedback constraints. Used
  in Introduction as general literature context; no number attached.
  Verify.
- `Martinet2021peaks` — Martinet et al. 2021, WL peak statistics. Used in
  Introduction to cite peak counts as a higher-order WL statistic; no
  number attached. Verify.
- `Zurcher2022MFs` — Zürcher et al. 2022, Minkowski functionals for
  cosmology. Used in Introduction; no number attached. Verify.
- `Gatti2022HOScos` — Gatti et al. 2022, higher-order WL statistics for
  cosmology. Used in Introduction; no number attached. Verify.
- `Gatti2023maplevel` — Gatti et al. 2023, map-level WL inference. Used in
  Introduction; no number attached. Verify.
- `DESACTTSZWL2025` — a DES × ACT joint tSZ+WL analysis (2025, exact
  authors/title unconfirmed). Used in Introduction as background for the
  real-data confrontation section; no number attached from this citation
  itself (the actual numbers in \S6 come from the dossier-sourced WORKLOG
  entries, cited separately). Verify this reference exists and get the
  correct author list — flagged in the dossier as harvested from an
  adjacent worktree's draft.tex, unverified.
- `BigwoodAmon2024` — Bigwood, Amon et al. 2024, joint WL+kSZ feedback
  constraints. Used in Introduction; no number attached. Verify.

## Notes for the Cite agent

- All numbers in the text are sourced to WORKLOG entries or on-disk
  figures per the dossier, NOT to any of the citations above — the
  citations above are literature framing/background only (Introduction,
  Discussion) and carry no `% src:` numeric claims themselves. Do not let
  verifying/fixing a citation change any `% src:`-tagged number.
- The dossier explicitly did NOT confirm `Troster2021KiDSPlanck`,
  `HillSpergel2014`, `Battaglia2012pressure`, `Cranmer2020SBI`,
  `PapamakariosMurray2016`, `Greenberg2019APT` against any mined
  repository source — they are included here as standard, expected
  literature citations for an SBI/tSZ paper, but must be verified as real,
  correctly-attributed papers before the reference list is finalized.
- A larger, unfiltered list of additional bib keys was harvested by the
  mining pass from `wl-tsz-bridge/examples/draft.tex` (an adjacent,
  earlier LaTeX draft in the same worktree) but NOT used in `main.tex`
  because they were not clearly relevant to specific claims made here:
  `Gatti2023maplevel` (used, see above), `FLAMINGOscattering2025`,
  `Siegel2025`/`Siegeletal2025mnras` (eROSITA missing-baryons tension —
  relevant to the "robust feedback signal remains inner-CGM" discussion in
  \S6/\S7 and to Paper IV; consider adding if the Discussion is expanded),
  `Pandey2022joint`, `MartinezConcepcion2024baryons`, `Fong2021TNGbaryons`,
  `LeeAmon2026` (possible self-citation — check if this is the author's
  own prior work before citing), `BindPaper1` (likely redundant with
  `bindI`). None of these were cross-checked against arXiv/ADS; add only
  if a specific claim needs them.
