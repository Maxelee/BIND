# Leftovers — Paper I (BIND lightcone pipeline)

Results, figures, and citations from the dossier that were deliberately left out of this
draft, one line each with why.

## Figures not used

- `fig_fgas.pdf` (f_gas(M,z) six-redshift evolution + Δf_gas envelope) — Paper II /
  twobound-flagship material per `draft.tex` §3.1 and the dossier's own figure notes; not
  in Paper I's methods+validation+bridge outline.
- `fig_peaks_minima.pdf`, `fig_mf.pdf`, `fig_hierarchy.pdf`, `fig_orthogonality.pdf`,
  `fig_decomp.pdf`, `fig_pressure.pdf`, `fig_budget.pdf`, `fig_morphology*.pdf` — twobound
  1P/Sobol response-library figures (per-parameter sensitivity rankings, gas morphology
  vs. feedback, baryon budget closure by mass bin); belongs to the systematic
  feedback-response study (Paper II/III territory), not this paper's fixed-fiducial
  validation + bridge scope.
- `fig_lightcone_population.pdf` (§6 z=0.034-vs-lightcone-halo-population fix) — belongs
  to the WL-latent story (Paper III per the dossier's mapping).
- `bcm_warp_comparison_fid.png` (anisotropy / f_aniso) — explicitly Paper V territory
  (field-level anisotropy of the baryon correction), reserved for that letter.
- `sb35_sobol_figs/scaling_relations_z0.png` — per-halo scaling relations at Sobol scale;
  not inspected in detail by the miner, judged low priority / likely Paper III material.
- `examples/figures_ksz/f6b_lowmass.pdf` — superficially related to the low-mass
  completeness result (Results §4) but is actually a kSZ/DESI×ACT-specific figure
  (BGS/ELG f_gas(M) with a "patch reuse" shaded region); carries content outside this
  paper's scope (Paper IV's domain) and was explicitly flagged by the dossier as
  not-reusable-as-is. We used prose only for the completeness result and left a
  `\todo{completeness schematic figure}` rather than reuse this figure or fabricate a new
  one.
- No standalone `R(nu)` vs. feedback-parameter curve figure exists on disk (the source
  notebook `examples/lightcone_comparison.ipynb` was not found in the working tree). We
  used the related `bind_bridge_hero.png` (ejection-heating decomposition) as the closest
  available figure and left a `\todo{}` flagging the gap explicitly in
  Section~\ref{subsec:res2}, rather than mislabel `bind_bridge_hero.png` as if it were the
  R(nu) curve itself.

## Numbers/results not used

- **σ(log Y|M) two different ranges** (0.23–0.44 dex map-level peak-stacking vs.
  0.12–0.22 dex per-halo R500c atlas) — both are reported in the text with their distinct
  provenance rather than merged or averaged, per the dossier's explicit flag that they
  come from different analyses/apertures.
- **The retracted "f_gas saturation ~41% cosmic" number** — reported in the Discussion
  only as an explicitly superseded/retracted figure, alongside the corrected
  background-subtracted framing (0.076–0.086 fiducial, best single knob closes ~58% of
  the gap to eROSITA-low, recomputed directly from the sourced 0.076/0.047/0.026 values —
  see verification log at the end of main.tex; an earlier draft pass had this at "~40%",
  which inverted the closed/remaining fraction). We did not present "~41%" as a standalone
  headline result anywhere, per the dossier's explicit recommendation.
- **"n_halos=666"** — not used anywhere in this draft; the dossier flags this as an
  ambiguous per-snapshot verification count that should not be conflated with the
  lightcone's total halo count (contrast the 2,933 M200c>1e13 halos at snap_096 alone).
  Omitted entirely to avoid misuse.
- **BIND acronym expansion** ("Baryon-Incorporated N-body with Diffusion") — not spelled
  out anywhere in this draft's body text (we refer to "BIND" as a name only), since the
  expansion is corroborated only by `examples/draft.tex` and not by `CLAUDE.md` or any
  other source. If a future revision wants the expansion, it should carry an explicit
  caveat per the dossier's flag.
- **GPU-hour cost comparison** (BIND lightcone vs. full TNG300 hydro run) — left as a
  `\todo{}` in the Data and code availability section; no number exists anywhere in the
  sources searched (draft.tex itself has an unfilled X/Y placeholder for this).

## Citations not used

See the "Considered but NOT cited" section of `refs_needed.md` for the full list of
brief-seeded citations (explicit arXiv IDs from `wl_tsz_plan.md`, and several `draft.tex`
intro bibkeys) that were not incorporated into this narrower Paper I draft — most of them
support the systematic feedback-parameter-sensitivity narrative (dominant knobs per
statistic, detectability against Stage-IV noise floors, morphology/offset feedback
sensitivity) that belongs to Paper II, not this methods+validation+bridge paper.
Specifically not cited: `Salcido2025kSZ`, `FLAMINGOkSZlensing2024`, `Marques2024HSC`,
`FLAMINGOscattering2025`, `Fong2021TNGbaryons`, `MartinezConcepcion2024baryons`,
`HarnoisDeraps2021peaks`, `Weiss2019MFs`, `FLAMINGOmps2024`, `Schaye2023FLAMINGO`,
`Schneider2019BCM`, `Schneider2025baryonification`, `Schneider2022clusters`,
`Pandey2022joint`, `PrestonRogers2025`, `Siegeletal2025mnras`, and the full explicit
arXiv-ID list from `wl_tsz_plan.md` beyond the two (`Siegel2025`/`DESACTTSZWL2025`)
candidate-mapped in `refs_needed.md`.

## Content structurally out of scope (per dossier's explicit note on draft.tex)

`examples/draft.tex`'s own scope is actually the twobound 1P feedback-sensitivity paper
(closer to Paper II/the flagship), not Paper I. We harvested its Introduction framing and
its Methods §2.1–2.5 subsection *structure* (BIND methodology / halo catalog / lightcone
construction / validation / feedback suite) as instructed, but did not carry over its
§3–§6 halo-statistics and WL-response-as-function-of-feedback content (gas fractions vs.
six redshifts, pressure-profile AGN core deficit, baryon budget closure by mass bin, gas
morphology and gas–DM offsets, WL statistics vs. feedback parameter with dominant-knob
attributions) — all of that is Paper II material per the dossier's explicit note, and
reproducing it here would blur the intended Paper I/II split.
