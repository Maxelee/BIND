# References needed — Paper I (BIND lightcone pipeline)

Every `\citep`/`\citet` key used in `main.tex`, with what the dossier/brief says about
it. **None of these have been verified as real, correctly-attributed references** — that
is the Cite step's job. Do not create `references.bib` from this list without
verification; several entries below are flagged as uncertain matches.

## High confidence (explicit arXiv IDs available)

None of the arXiv-ID-bearing entries from `docs/wl_tsz_plan.md` "Key references" ended up
directly cited by number in this draft (they were considered for the broader
feedback-crisis paragraph but Paper I's narrower methods+validation+bridge scope did not
need all of them — see `leftovers.md`). If the Cite step wants to swap any authoryear key
below for its arXiv-ID counterpart, the candidate mapping (unconfirmed 1:1, per dossier) is:
- `Siegel2025` — plausibly arXiv:2509.10455 ("X-ray+kSZ+WL gas expulsion") or a companion
  paper; draft.tex describes it as "FLAMINGO >8σ discrepant with kSZ profiles, eROSITA
  X-ray gas fractions ~2x lower than FLAMINGO predicts."
- `DESACTTSZWL2025` — plausibly arXiv:2506.07432 (DES Y3×ACT DR6 shear×y, 21σ), matched by
  the "21σ" description in both `draft.tex` and `wl_tsz_plan.md`.
- Other explicit IDs from `wl_tsz_plan.md` (2512.02954, 2410.19905, 2309.07959,
  2503.19441/2, 2602.10065, 2602.12238, 2505.07949, 2406.01672, 2407.20448, 2312.08450,
  2406.08540, 2201.08320, 2109.04458, 2602.11279, 2201.12591) were not cited in this draft
  — see `leftovers.md` for why (mostly Paper II's feedback-systematics territory).

## Cited in main.tex

- **vanDaalen2011** — van Daalen et al. 2011, original demonstration that baryonic
  feedback suppresses the nonlinear matter power spectrum ~tens of percent at
  k>1 h/Mpc. Brief's seed list ("van Daalen+2011"). arXiv ID not found in-repo; standard
  literature ID is arXiv:1104.1174 — NOT confirmed in-repo, verify independently.
- **vanDaalen2020** — van Daalen, McCarthy & Schaye 2020. Used quantitatively: the
  functional-form fit `−exp(−5.990·f̃−0.5107)` for ΔP/P vs. renormalized baryon fraction
  f̃_bar, explicitly reproduced in `fig_bridge_vandaalen.pdf` (WORKLOG 2026-06-18 item 5).
  High confidence, directly used in Results §3 (van Daalen analog) and the bridge framing
  throughout. Brief's seed list.
- **Siegel2025** — cited for the observational feedback-crisis motivation (eROSITA X-ray +
  kSZ gas-expulsion tension with FLAMINGO). Sourced from `examples/draft.tex` Introduction
  §, worktree `wl-tsz-bridge` (not yet in any references.bib). arXiv ID not confirmed
  in-repo — see high-confidence candidate mapping above.
- **DESACTTSZWL2025** — DES Y3 shear × ACT tSZ cross-correlation, 21σ detection, evidence
  for reduced thermal pressure below 1e14 Msun/h. Sourced from `draft.tex` Intro. Candidate
  arXiv:2506.07432 (unconfirmed 1:1).
- **BigwoodAmon2024** — WL + kSZ joint constraint on gas fractions systematically lower
  than simulations. Sourced from `draft.tex` Intro (bibkey only, no arXiv ID in-repo).
- **SchneiderTeyssier2015** — the original baryon correction model (BCM). Sourced from
  `src/bind/inference/spherical.py` context / WORKLOG 2026-06-17 ("a real
  Schneider–Teyssier BCM reproduces..."), also brief's seed list ("Schneider & Teyssier
  2015"). High confidence.
- **Chisari2019** — review of modelling baryonic physics for weak lensing surveys; used in
  Methods discussion of BCM accuracy/limitations. Brief's seed list ("Chisari+2019"); not
  independently confirmed by an in-repo text search beyond the seed list — treat as
  brief-seeded, not dossier-confirmed.
- **Mead2021** — HMcode-2020, halo-model-based baryon correction to the power spectrum.
  Brief's seed list ("Mead+2021 (HMcode-2020)"); not independently confirmed by in-repo
  search — brief-seeded only.
- **Martinet2021peaks**, **Gatti2022HOScos**, **Zurcher2022MFs** — Stage-IV non-Gaussian
  WL statistics (peak counts, higher-order statistics, Minkowski functionals) motivating
  the intro's "beyond the power spectrum" paragraph. Sourced from `draft.tex` Intro
  (bibkeys only, no arXiv IDs in-repo).
- **LeeAmon2026** — the group's own prior BCM-vs-hydro peaks paper. Per
  `wl_tsz_plan.md` §1: "Our own BCM-vs-hydro peaks paper (MNRAS 519, 573; 2201.08320)
  showed BCMs hold only to ν≲4." The dossier cross-checks this against `draft.tex`'s
  in-text description of `LeeAmon2026` ("biased cosmological inference even when the power
  spectrum is correctly reproduced") and finds it a plausible match, but this is NOT
  independently confirmed 1:1 — verify MNRAS 519, 573 / arXiv:2201.08320 is indeed the
  paper referred to as `LeeAmon2026` before finalizing.
- **VN2021fieldemulator** — Villaescusa-Navarro field-level baryonic emulator work
  (CAMELS-trained). Brief's seed list ("Villaescusa-Navarro+2021/2023"); matches
  `draft.tex` bibkey `VN2021fieldemulator`.
- **Arico2021** — Aricò baryonification/field-level emulator, cited alongside
  VN2021fieldemulator for the "simulation-based field-level emulators" modelling
  category. NOTE: the dossier flags a discrepancy — `draft.tex` uses bibkey `Arico2024`,
  while the brief's seed list says "Aricò+2020/21." We used `Arico2021` as a compromise
  key; **the actual year (2020, 2021, or 2024) needs to be pinned by the Cite step** —
  these may refer to different papers in an Aricò author sequence (BCM paper vs. later
  field-emulator paper) and should not be assumed identical.
- **Gatti2023maplevel** — map-level joint BCM baryonification for WL + tSZ full-sky
  painting. Sourced from `draft.tex` bibkey list, matches brief's modelling-categories
  paragraph.
- **BindModelPaper** — the in-prep BIND flow-matching model paper itself (self-citation
  for architecture/training details, kept short per project hard rules — "Lee et al. in
  prep," no fake arXiv ID). Provenance: `CLAUDE.md`, `PLAN.md` hard rules per dossier.
- **VillaescusaNavarro2023** — CAMELS + SB35 (35-dim subgrid parameter suite) design
  paper. Brief's seed list ("Villaescusa-Navarro+2021/2023"); used for the SB35
  conditioning citation in Methods §2.1. Verify whether 2021 or 2023 is the correct year
  for the SB35-specific design (vs. the original CAMELS paper) — dossier does not
  disambiguate.
- **LipmanEtAl** — conditional flow matching (the generative-modelling technique
  underlying BIND). Brief's seed list gives "Lipman+2022"; the well-known flow-matching
  paper (Lipman, Chen, Ben-Hamu, Nickel & Le, "Flow Matching for Generative Modeling") is
  actually arXiv:2210.02747, first appeared 2022, published ICLR 2023 — **verify correct
  year/venue before finalizing**, as the brief's "2022" may refer to the arXiv preprint
  date while a "2023" publication year is also defensible.
- **Schaan2021**, **Hadzhiyska2024**, **RiedGuachalla2025** — kSZ tier-1 literature
  justifying the velocity-free τ/optical-depth estimator design (the observable kSZ
  estimator factorizes out the velocity dependence). Named directly, no arXiv IDs given,
  in `docs/wl_tsz_plan.md` §2. Used in Methods §2.2 and the Discussion τ-caveat.
- **Arnaud2010** — Arnaud et al. 2010 universal pressure profile, used quantitatively as
  the comparison curve for the pressure-profile validation figure. Sourced from
  `figs_raw/paper_lightcone_figs/cell005_out1.png` legend and `draft.tex` §Pressure
  Profiles. High confidence, directly used.
- **Popesso2024** — Popesso et al. 2024, source of the eROSITA-low f_gas value (0.026)
  used in the corrected eROSITA-tension Discussion figure. Sourced from `fig_siegel.pdf`
  legend ("eROSITA-low (Popesso+24)") and WORKLOG 2026-06-15pm. High confidence, directly
  used quantitatively.

## Forward self-citations (suite cross-references, in prep)

- **bindII**, **bindIII**, **bindIV**, **bindV** — the four companion papers in this
  series (Paper II: feedback-response/thermodynamic-decomposition flagship; Paper III:
  survey-data application and/or cosmology-varying lightcone grid; Paper IV:
  velocity-resolved kSZ extension — explicitly named in the brief's own Discussion outline
  ("τ is velocity-free electron column ... — Paper IV"); Paper V: field-level anisotropy
  of the baryon correction, a letter). Cited as `\citep[Paper~N]{bindN}` /
  `\citet[Paper~N]{bindN}` throughout, per the brief's instruction ("Suite
  cross-references: \citep[Paper N]{bindI..bindV}, listed in refs_needed.md as
  in-prep"). Paper II's content assignment is grounded in `docs/wl_tsz_plan.md` §7
  ("Papers" plan); Paper III's split (data vs. cosmology grid) is explicitly left open
  there ("choose after II"); Paper V's anisotropy assignment is inferred from the
  `wl-anisotropy` memory topic and is not pinned by an explicit "Paper V" label anywhere
  in the sources read — **flag for the human/editor to confirm the final Paper II–V
  content assignment** before this cross-referencing language is treated as final.

## Considered but NOT cited (left out of the narrower Paper I scope)

See `leftovers.md` for the full list of brief-seeded citations not used in this draft
(mostly the explicit arXiv-ID list from `wl_tsz_plan.md` and several `draft.tex` intro
bibkeys that belong to Paper II's systematic feedback-response study rather than this
paper's methods+validation+bridge scope) — e.g. `Salcido2025kSZ`,
`FLAMINGOkSZlensing2024`, `Marques2024HSC`, `FLAMINGOscattering2025`, `Fong2021TNGbaryons`,
`MartinezConcepcion2024baryons`, `HarnoisDeraps2021peaks`, `Weiss2019MFs`,
`FLAMINGOmps2024`, `Schaye2023FLAMINGO`, `Schneider2019BCM`, `Schneider2025baryonification`,
`Schneider2022clusters`, `Pandey2022joint`, `PrestonRogers2025`, `Siegeletal2025mnras`.

## Not cited but present in dossier, deliberately omitted (see leftovers.md for reasoning)

- **Pillepich2018, Springel2018, Nelson2019** (TNG300 simulation papers) — the brief's seed
  list names these for the TNG300 box description; we describe TNG300-Dark by name and
  box size (205 Mpc/h) but did not add explicit simulation-paper citations in this draft.
  **Recommend the Cite step add these** at the first mention of TNG300-Dark in
  Section~2.2 — this is a plausible oversight to fix, not a deliberate exclusion.
- **Duffy2008** (NFW concentration–mass relation) — used in `halo_atlas.py` per WORKLOG
  2026-06-15pm ("NFW+Duffy") but not directly quoted with a number in this paper's final
  text (mass-concentration is used internally by the halo-atlas machinery, not stated as
  a headline result here). Candidate for a Methods-section citation if a future revision
  adds the concentration-mass relation explicitly.
- **Eckert2016, Eckert2019, Lovisari2015** (mainstream X-ray f_gas comparisons) — named
  in the dossier's eROSITA-tension writeup ("BIND fid f_gas matches mainstream X-ray
  determinations (Eckert et al.) to ~9%"), used descriptively in the Discussion but the
  specific citep was not added to the sentence that names "Eckert et al." — **the Cite
  step should add `\citep{Eckert2016,Eckert2019}` (and optionally `Lovisari2015`) at that
  sentence** in the Discussion §"the eROSITA gas-fraction tension" paragraph.
- **Macquart2020** (FRB dispersion measure / "Macquart term") — named in the brief's seed
  list and in `stats.py` docstring per dossier, relevant to the τ=DM discussion in
  Methods §2.2, but no explicit `\citep` was added — **recommend adding** at the sentence
  introducing DM = τ/TAU_PER_DM.
- **Planck2015XXII** (or a later Planck y-map release) — the brief's seed citation for the
  y-map normalization validation ("validated vs Planck", mean-y comparison). The dossier
  explicitly flags this as unpinned: "Exact Planck y-map paper/release ... not pinned
  in-repo beyond a MEMORY note." We used a `\todo{}` in the Discussion (tSZ auto-spectrum
  deficit paragraph) instead of guessing the citation — **do not add `Planck2015XXII` to
  references.bib until the Cite step confirms which Planck y-map release is meant.**
