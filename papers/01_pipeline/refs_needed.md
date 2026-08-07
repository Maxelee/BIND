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

## New — fig20/"New Image 3" van Daalen comparison, flags (ii)/(iii) (`ni3-flags` audit, 2026-08-03)

Not yet cited in `main.tex` (fig20's markdown/caption text lives only in
`_build_figures_nb.py` today) but needed once New Image 3's caption is written. All four
arXiv IDs below were fetched and read directly (arXiv abstract + full text), not guessed.

- **Flag (ii) verdict: CONFIRMED, with one caveat already correctly implemented but not yet
  in a citable caption sentence.** Read van Daalen, McCarthy & Schaye 2020 (arXiv:1906.00968,
  §3.9 "The baryon fraction as a predictor of power suppression", eq. 4, Fig. 16) directly:
  their exponential fit $-\exp(-5.990\,\tilde f_{\rm bar,500c}-0.5107)$ — exactly the
  coefficients hardcoded in the builder — is presented AT $k=0.5\,h\,{\rm Mpc}^{-1}$
  specifically (their Fig. 16), for the renormalized baryon fraction (gas+stars, within
  $R_{500c}$, normalized by $\Omega_b/\Omega_m$) of haloes with $M_{500c}\in[6\times10^{13},
  2\times10^{14}]\,M_\odot$ (their eq. 4) — both the $k$ and the mass bin used by the builder
  match exactly. The one real mismatch: van Daalen+20's $\tilde f_{\rm bar,500c}$ is a true
  3-D spherical quantity (halo-finder $R_{500c}$ sphere); the builder's `sobol_m_*_500c_bg`
  fields (used identically in figs 3/19/20) are a *projected-cylinder*, annulus-subtracted
  proxy at the same nominal aperture. Panel (c) already corrects for this via a $\times1.16$
  truth calibration (own TNG300-hydro $\tilde f^{\rm cyl}=0.93$ vs. `Nelson2024TNGCluster`
  3-D $\tilde f\approx0.81$); panels (a)/(b) do not overlay the literature curve so carry no
  apples-to-oranges risk, but use the same cylinder quantity. See
  `audits/report_ni3-flags.md` for the full verdict and recommended caption sentence.
- **`Nelson2024TNGCluster`** — Nelson et al. 2024, "Introducing the TNG-Cluster Simulation:
  overview and physical properties of the gaseous intracluster medium," A&A 686, A157
  (arXiv:2311.06338). Already named in-text in the fig20 markdown cell (the source of the
  3-D $\tilde f\approx0.81$ used for the $\times1.16$ calibration) but missing from
  `references.bib`. High confidence, verified directly (arXiv abstract).
- **Flag (iii) — the uncited "obs. groups" band (`axvspan(0.55, 0.76, ...)`).** This band
  reproduces van Daalen et al. (2020)'s own green band (their Figs. 15/16), which is itself a
  12-paper compilation in their footnote 10 (Vikhlinin+06, Maughan+08, Sun+09, Pratt+09,
  Rasmussen & Ponman+09, Lin+12, Sanderson+13, Gonzalez+13, Budzynski+14, Lovisari+15,
  Pearson+17, Kravtsov+18 — confirmed by reading van Daalen+20's own reference list). Rather
  than add all 12 (most otherwise unused in this paper), we recommend citing `vanDaalen2020`
  itself (already in `references.bib`) as the direct source of the reproduced band, plus four
  representative primary papers matching the task's own "usual suspects" list — all four
  *are* confirmed in van Daalen+20's footnote 10; `Akino2022` (the fifth suspect) is NOT,
  since it postdates van Daalen+20 by two years and cannot be part of that specific
  compilation — do not cite it for this band.
  - **`Vikhlinin2006`** — Vikhlinin et al. 2006, "Chandra Sample of Nearby Relaxed Galaxy
    Clusters: Mass, Gas Fraction, and Mass-Temperature Relation," ApJ 640, 691
    (arXiv:astro-ph/0507092). Verified directly.
  - **`Sun2009`** — Sun et al. 2009, "Chandra Studies of the X-Ray Gas Properties of Galaxy
    Groups," ApJ 693, 1142 (arXiv:0805.2320). Verified directly.
  - **`Gonzalez2013`** — Gonzalez et al. 2013, "Galaxy Cluster Baryon Fractions Revisited,"
    ApJ 778, 14 (arXiv:1309.3565). Verified directly.
  - **`Lovisari2015`** — Lovisari, Reiprich & Schellenberger 2015, "Scaling properties of a
    complete X-ray selected galaxy group sample," A&A 573, A118 (arXiv:1409.3845). Verified
    directly. NOTE: this key was already flagged above (the Eckert/Lovisari X-ray f_gas
    entry) as a candidate for the eROSITA-tension paragraph — it is now independently needed
    here too; one `references.bib` entry covers both uses.
  Full bibtex for all five (`Nelson2024TNGCluster` + the four X-ray papers) is in
  `audits/report_ni3-flags.md`.

## Positioning refs for the analytic latent model (2026-08-07 literature check, verified via arXiv)

Budget→suppression lineage (our f̃_bar kernel is this thread's WL projection):
- van Daalen+2011 (the original P(k) challenge); Semboloni+2011 (WL tomography + f_gas-informed correction)
- van Daalen, McCarthy & Schaye 2020 (ΔP/P at k=0.5 ↔ f̃_bar,500c universality)
- Salcido+2023 SP(k) (arXiv:2305.09710): mean f_b(M) → P(k) suppression, ~percent level
- FLAMINGO "resummation" model (arXiv:2509.04552, 2025): observed f_b at R500c+R200m (+stellar
  fractions for k≤25) → P(k) suppression ≲1% to k≤10, ZERO free params. State of the art of the
  thread. NB: they independently need STELLAR fractions to reach small scales = convergent with
  our partition latent (we discovered it via residual PCA).

One-model-many-statistics competitor:
- Zhou, Gatti, Anbajagane+2025 (arXiv:2505.07949, JCAP): map-level baryonification (3-param BCM,
  FLAMINGO-calibrated) unifying WL 2pt + HOS within 2% at ℓ<2000. FORWARD model (modify DMO sims);
  ours is a RESPONSE model on measured halo-population properties, no profile assumption, ℓ→3e4,
  + scalings & SZ/τ statistics.

Latent/analytic descriptions:
- Lin+2025 "One latent" (arXiv:2509.01881): 2 abstract PCA latents for feedback on matter
  distribution — we name them with measured halo observables + add structure/thermal.
- Schaller & Schaye 2025 (arXiv:2504.15633): analytic z-independent 1-param sigmoid (A_mod
  family) for P(k) suppression, k<3, z≤1 — parallels our "z_s lives in the coefficients" at P(k) level.
- syren-baryon (arXiv:2506.08783): symbolic-regression analytic emulators, params→P(k).
- BACCO baryonification NN emulator (arXiv:2011.15018).

Observational-inference side (what our corner formalizes at latent level):
- arXiv:2512.02954 (2025): suppression from X-ray f_gas + kSZ profiles + GGL.
- arXiv:2511.10975: DMO counterpart of the observed Universe from WL + baryon censuses.
- Bigwood+2024 (arXiv:2404.06098): WL + kSZ joint feedback constraints.

Apparently novel in our work (no equivalent found in searches): the family↔kernel chain-rule
closure (parameter-response families explained as kernel mixtures via measured fingerprints);
the multi-statistic latent scorecard with a single measured latent set; the SH-scores analytic
latent posteriors with LOO coverage verification; the decomposed error budget.
