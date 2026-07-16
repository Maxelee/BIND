# Leftovers — Paper V (anisotropy of the baryonic WL suppression)

Results/figures the dossier surfaced that were deliberately left out of
`main.tex`, one line each, with why.

- **Field 4 / Pillar 3 parameter-response figures** (`cell037_out0.png`,
  Field-4 heatmap; per-halo Pillar-3 bar chart) — not used as a figure. The
  underlying numbers are summarized qualitatively in \S3.3 with an explicit
  caveat that they come from the superseded mono control, but a dedicated
  figure risked visually overselling a "corrected feedback-dependence" result
  that does not yet exist (dossier Gap #1).
- **Non-Gaussian statistics (Field 3: peaks/minima/PDF, `cell032`/`cell033`)**
  — left out entirely. The dossier itself flags this panel as noisy (peaks
  chi^2 p=1.0, minima chi^2 p=1.0, low-significance high-nu shifts) and a
  letter-length paper has no room for a null/marginal result; kept the
  qualitative Lee et al. 2022 peak-count tie-in instead, which is the sourced,
  higher-confidence version of the same argument.
- **Pillar 1 per-halo `f_quad(k)` figure** (`cell008_out0.png`) — not used.
  The dossier flags an internal inconsistency between the notebook's
  Conclusions text (f_quad ~ 30-55%) and the directly plotted curve (~5-20%
  peak); rather than adjudicate that inconsistency in a letter-length paper,
  the per-halo multipole *methodology* is described (\S2.1's Pillars concept
  is implicit in the field-level construction) without citing the specific
  f_quad numeric result.
- **Toy-model validation figure** (`cell006_out0.png`) — not used. Appendix-
  grade validation (injected quadrupole recovered exactly, null test passes,
  alignment-dilution curve validated) that supports \S2.5's error-model
  numbers but isn't itself a result; omitted for space in a 3-4-figure letter.
- **`examples/figures_lightcone/fig_anisotropy_stats.pdf`** — not used and not
  cited. Dossier found no WORKLOG entry, notebook, or script referencing this
  file's provenance, and its claim ("non-radial adds nothing" to a
  feedback-response regression R^2) answers a different question (SBI
  parameter-inference feature importance) that could be misread as
  contradicting this Letter's thesis if included without sourcing (dossier
  Gap #3).
- **Memory-file-only peak-count numbers** (436->418->424 halos, "~43%
  anisotropic") — not used. Dossier could not locate the originating notebook
  cell for these numbers; flagged as lower-confidence in the dossier itself
  (Gap #7) and dropped rather than guessed.
- **Pillar-2 Conclusions' "S/N ~ 3-5" claim** — not used; the directly
  computed/printed forecast (cited in \S4.2 of `main.tex`) only reaches
  S/N ~ 1.5-2.3 at N_cl=10^4 (cluster bin), so the Conclusions-text number in
  the source notebook is not supported by its own printed output (dossier
  caveat 8b). Used the directly-printed forecast numbers instead and did not
  flag the internal discrepancy explicitly in the Letter text (kept the letter
  focused on the paper's own thesis rather than auditing the source
  notebook's internal consistency in the main text).
- **Ejection-anisotropy Figure 4** (`cell019_out0.png`, bar chart of which
  feedback knobs change ejection *orientation*, starred AGN-jet-reorientation
  factor) — not used as a figure; the qualitative bipolar/anti-aligned result
  (cos2(Delta theta) = -0.773) is stated in \S3.4 using the numeric result
  from cell 11 instead, which was judged sufficient without a second bar-chart
  figure in a 4-figure letter.
- **Planck Intermediate V (2013), Lau et al. (2011), Biffi et al. (2016)** —
  citations sourced in the dossier (via `ejection_anisotropy.ipynb` cell 13,
  azimuthal pressure-profile scatter in stacked clusters, ~15-25% at r500) but
  not actually cited in `main.tex`: the Letter's Discussion ties the ejection
  anisotropy to the WL/tSZ measurements already in hand rather than to the
  literature's X-ray/tSZ morphology-quadrupole numbers, to keep the letter's
  citation list tight. Listed in `refs_needed.md` in case a later revision
  wants the pressure-profile literature context restored.
- **60-vs-57 OAT-run discrepancy** — included only as a footnote in \S2.4
  rather than a full discussion point; the dossier itself calls this
  unreconciled bookkeeping, not a physical result worth main-text space in a
  letter.
