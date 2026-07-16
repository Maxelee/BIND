# References needed — Paper V (anisotropy of the baryonic WL suppression)

Every `\citep`/`\citet`/`\citealt` key used in `main.tex`, for the Cite stage to
resolve to real bibliographic entries (venue, volume, DOI/arXiv) and add to
`references.bib`. No `references.bib` was created by the Writer per instructions.

## Found directly in the dossier's sources (high confidence — real, published work; only full bibliographic details are missing)

- `schneider_teyssier2015` — Schneider & Teyssier 2015. Baryonic correction model
  (BCM) origin paper. Cited for the general BCM radial-displacement framework.
  Source: `wl_anisotropy_paper.ipynb` cell 0; `src/bind/inference/spherical.py`
  module docstring. Names/year only — venue/arXiv needed from Cite stage.
- `arico2020` — Aricò et al. 2020. BCM. Source: `wl_anisotropy_paper.ipynb`
  cell 0. Names/year only — venue/arXiv needed.
- `mead2021` — Mead et al. 2021. BCM / HMcode-style baryon correction. Source:
  `wl_anisotropy_paper.ipynb` cell 0. Names/year only — venue/arXiv needed.
- `lee2022` — Lee et al. 2022, MNRAS 519, 573; arXiv:2201.08320. Real,
  already-published paper (not "in prep"), same author group as this Letter;
  cited for the result that BCMs reproduce hydrodynamic peak counts only to
  nu <~ 4. Source: `wl_anisotropy_paper.ipynb` cell 39/40. Full bibliographic
  details (journal/volume/arXiv) ARE given directly in the dossier — should be
  low-effort to resolve/confirm.
- `planck_intermediate_v_2013` — Planck Intermediate V (2013), Fig. 15:
  azimuthal scatter ~20% in the stacked cluster pressure profile at r500.
  Source: `ejection_anisotropy.ipynb` cell 13. **NOT currently cited in
  `main.tex`** (left out per leftovers.md — not core to the Letter's WL-focused
  argument) but flagged here in case the Discussion/detectability section is
  expanded. Venue/DOI needed if used.
- `lau2011` — Lau et al. (2011). X-ray morphology quadrupoles in simulated
  clusters, ~15-25% at r500. Source: `ejection_anisotropy.ipynb` cell 13.
  **NOT currently cited in `main.tex`** — same status as Planck Int. V above.
- `biffi2016` — Biffi et al. (2016). Same context as Lau+2011 (X-ray morphology
  quadrupoles). Source: `ejection_anisotropy.ipynb` cell 13. **NOT currently
  cited** — same status as above.

## Brief's seed citations — NOT found verbatim in the dossier's searched sources; used in `main.tex` for general literature framing only (never attached to a sourced number), Cite stage MUST independently verify these are real papers with the claimed content, or the citation should be dropped/replaced

- `dai_feng_seljak2018` — Dai, Feng & Seljak 2018, potential-gradient-descent-style
  gas displacement modeling (per brief's own framing, "potential-gradient
  descent?"). Used in Methods (\S2.2) as a parenthetical "in the spirit of"
  citation for radial-push BCM displacement laws generally — NOT attached to
  any specific number. Dossier note: "no occurrence found" in the searched
  notebooks/scripts. Verify real paper + relevance, or drop the parenthetical.
- `osato_triaxiality` — Osato+ (halo triaxiality and WL bias). Used in Methods
  (\S2.2) as a parenthetical citation for halo-triaxiality effects on WL,
  motivating why the mono control's triaxiality-blindness matters. Dossier
  note: "no occurrence found." Verify real paper (there are several Osato et
  al. papers on halo triaxiality / weak lensing; need the specific one the
  brief intended) or drop.
- `van_daalen2020` — van Daalen et al. 2020. Used once in the Introduction for
  the generic (non-quantitative) claim that baryonic suppression/anisotropy of
  gas maps in hydrodynamic simulations is well documented. Dossier note: only
  a generic, year-less "van Daalen" mention found in `ejection_anisotropy.ipynb`
  (in the context of the f_gas-suppression correlation); no explicit "2020"
  citation with number found in the Paper-V source material itself. (The
  memory file `fgas-bridge-anatomy.md` does cite a van Daalen 2020 WL paper
  elsewhere in the repo, but that is a different source, not part of this
  Letter's dossier — confirm it's the intended paper before finalizing.)

## Cross-paper self-citations (BIND Lightcone Suite, in-prep convention)

- `bindI` — Paper I of the BIND Lightcone Suite (introduces the BIND
  painting/composite method used throughout). Cited once in the Introduction.
  Status: in prep, per repo `PLAN.md` convention noted in the dossier.
- `bindII` — Paper II of the BIND Lightcone Suite (the N=2 data-driven
  baryon-template cosmic-shear bias-removal capstone; corresponds to
  `examples/cosmo_bias_capstone.py` / the `baryon-cosmo-bias-2template` memory
  file). Cited once in the Discussion to connect the anisotropy result to why
  data-driven templates outperform analytic BCMs. Status: in prep. Dossier
  note: "argued in the Discussion above but not literally cross-cited in the
  source notebooks (they predate the paper-suite split)" — the Paper II
  content itself (N=2 templates) IS independently sourced in the maintainer's
  memory file `baryon-cosmo-bias-2template.md`, but the connection is this
  Letter's own argument, not lifted from a dossier citation.

## Not used, flagged in dossier as explicitly NOT recommended

- No entry — `examples/figures_lightcone/fig_anisotropy_stats.pdf` was
  deliberately not cited or figured (see `leftovers.md` and dossier Gaps #3).
