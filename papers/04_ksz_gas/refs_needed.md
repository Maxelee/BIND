# References needed — Paper IV (04_ksz_gas)

Every `\citep`/`\citet`/`\citealp` key used in `main.tex`, with what the dossier
says about it. **Do not fabricate a `references.bib` from this list** — a later
verification agent must confirm each real reference (arXiv ID, journal,
correct author list) before entries are added to the bibliography.

## Confirmed in sources (still verify exact venue/arXiv before finalizing)

- `schaan21` — Schaan et al. (2021). ACT kSZ stacking, cited in the notebook's
  closing citation list as "earlier stacks Schaan et al. (2021), Amodeo et al.
  (2021)." No arXiv ID given in sources.
- `amodeo21` — Amodeo et al. (2021). Companion kSZ-stacking reference, same
  provenance as schaan21.
- `popesso24` — Popesso et al. (2024). eROSITA strong-feedback hot-gas-fraction
  edge; formula `f_gas = 2.23e-7 * M^0.39` confirmed directly in
  `examples/ksz_fgas_confront.py`. No arXiv ID found in sources.
- `siegel25` — Siegel et al. (2025), arXiv:2509.10455. Missing-baryon
  consensus reference, confirmed directly in notebook closing citation and
  `docs/ksz_desi_act_plan.md`.
- `kovac25` — Kovač et al. (2025), arXiv:2507.07991. Grouped with the
  missing-baryon consensus; confirmed in sources.
- `eckert19` — Eckert et al. (2019). Mainstream X-ray f_gas,500 relation
  (weak-feedback edge); formula `f_gas = 0.131*(M/2e14)^0.21` confirmed in
  `examples/ksz_fgas_confront.py`.
- `desicollab24` — DESI Collaboration (2024) — used in text implicitly for
  "DESI" but not currently cited with an explicit key in main.tex; add if a
  DESI-survey-level citation is wanted in Methods.
- `hahn23` — Hahn et al. (2023), DESI BGS target selection — confirmed in
  notebook closing citation list; not currently cited by key in main.tex, add
  if BGS selection details are expanded.
- `zhou23` — Zhou et al. (2023), DESI LRG target selection — confirmed in
  notebook closing citation list; not currently cited by key in main.tex.
- `raichoor23` — Raichoor et al. (2023), DESI ELG target selection — confirmed
  in notebook closing citation list; not currently cited by key in main.tex.
- `vandaalen20` — van Daalen et al. (2020). Baryonic suppression S(l) =
  Cl_hydro/Cl_DMO definition; confirmed in field-companion notebook closing
  citation list.
- `sailer24` — Sailer et al. (2024). DESI-LRG host mass from ACT CMB lensing,
  logM200 = 13.18 Msun/h; confirmed in `examples/_reduce_ycap_lrg.py` and
  WORKLOG 2026-06-26.
- `gatti22` — Gatti et al. (2022). DES Y3 x ACT shear x y real-data
  confrontation; confirmed in field-companion notebook closing citation list.
- `pandey22` — Pandey et al. (2022). Companion DES Y3 x ACT reference, same
  provenance as gatti22.
- `bindI` — BIND Lightcone Suite Paper I (in prep.), suite/engine recap
  cross-reference. In-prep sibling paper, not an external reference.

## AMBIGUOUS — must be resolved before citing (flagged prominently in text with \todo)

- `kszpartI` — "Ried Guachalla et al. (2025)" per WORKLOG 2026-06-25 §6c,
  arXiv:2604.19744 ("Part I", LRG kSZ-amplitude rescaling table). Used in
  \S\ref{sec:intro} for the DESI x ACT kSZ measurement citation.
- `kszpartII` — the SAME two arXiv IDs (2604.19744 / 2604.19745) are
  attributed inconsistently to BOTH "Hadzhiyska et al. (2024/2026)" (earlier
  code comments, `ksz_tau_gnfw.py`, `ksz_cap_compare.py`,
  `ksz_fgas_profile.py`, `ksz_posterior*.py`) and "Ried Guachalla et al.
  (2025)" (later code/WORKLOG, `_reduce_fgas_cap.py`, quoting §III.2.1
  verbatim; also the notebook's own closing citation list, which lists BOTH
  names as if separate references). arXiv:2604.19745 is "Part II" (BGS/ELG,
  GNFW Table II, §III.2.1 CAP-ratio definition) and is used throughout
  Methods (\S\ref{sec:cap}, \S\ref{sec:gnfw}) and Results
  (\S\ref{sec:results2}) for the GNFW form and the CAP-ratio observable
  definition. The arXiv prefix 2604 implies an April-2026 submission,
  inconsistent with both quoted years (2024/2025). **THE CITE STEP MUST
  independently verify the actual author name(s) and year for both
  arXiv:2604.19744 and arXiv:2604.19745 before either is added to
  references.bib** — do not print both names as if citing two different
  papers, and do not trust either year without checking.
- Bigwood et al. — mentioned in \S\ref{sec:intro} discussion of the
  missing-baryon consensus but **NOT given a citation key** in main.tex
  because no independently confirmed arXiv ID exists in any source read
  (brief says 2024; notebook closing cell says 2025; `ksz_desi_act_plan.md`
  groups it under the same ID as Siegel, 2509.10455, which is ambiguous/likely
  wrong for Bigwood specifically). A `\todo{}` flags this explicitly in the
  text. If a real Bigwood reference is found, add it as `bigwood24` (or
  `bigwood25`, matching whichever year is confirmed).

## Author name unresolved (not currently cited by key — needs a name before use)

- tSZ y-CAP data paper, arXiv:2502.08850, Zenodo 14706729 (ACT x DESI
  photometric-LRG Compton-y CAP data, used in \S\ref{sec:results5}/Fig 7).
  Code comments cite only the arXiv ID, never an author name. **Not cited by
  key in main.tex** — the corresponding sentences in the draft describe the
  data without a `\citep`. Add a key (e.g. `ycapdata25`) once the author name
  is confirmed.

## Explicitly NOT cited (checked against sources, found absent — do not add without independent verification)

- **Bahar et al.** (brief's alternative eROSITA guess alongside Popesso) —
  never mentioned in any ksz-desi-act source. Do not cite.
- **Battaglia et al. (2012)** GNFW — never mentioned in any ksz-desi-act
  source; the GNFW form actually used is explicitly the
  Hadzhiyska/Ried-Guachalla paper's own Eq. 26-27 fit, not Battaglia's. Do not
  cite Battaglia for the GNFW form in this paper without independently
  verifying the form actually traces back to it.
- **Ferraro/Schaan CAP-filter-origin citation** — the brief suggests this as
  the CAP filter's origin, but no dedicated CAP-filter-origin citation was
  found in any source; the code attributes the CAP-ratio definition only to
  Ried Guachalla+25 §III.2.1. If an independent CAP-filter-origin citation is
  needed, it must be verified externally, not assumed to be Ferraro/Schaan.

## Cross-suite reference

- `bindI` — Paper I of the BIND Lightcone Suite (in prep.), cited via
  `\citep[in prep.]{bindI}` for the suite/emulator engine recap in
  \S\ref{sec:suite}. Sibling in-prep papers `bindII`–`bindV` are not currently
  cited in this draft but may be added if cross-references to companion
  analyses are wanted (e.g. Paper III for the low-dimensional-latent
  motivation, currently cited instead via `lin25`).

## Unverified leads only (from project memory, NOT tied specifically to the ksz-desi-act notebooks — do NOT cite without independent confirmation)

Not used anywhere in main.tex; listed here only so the CITE step knows they
were considered and rejected as insufficiently sourced:
- possible eROSITA arXiv 2411.16555 (maybe = Popesso+24 itself)
- possible Bigwood arXiv 2512.02954
- possible earlier-kSZ arXiv 2407.07152
- 1003.2270, 2507.16816 — out of scope (different WORKLOG topics: gas/DM
  shapes, FRB-DM-mass tension)

## Additional citations mentioned in the dossier but NOT used in this draft (left out deliberately — see leftovers.md)

- CAMELS: Villaescusa-Navarro et al. (2021) — the suite recap in
  \S\ref{sec:suite} refers the reader to Paper I rather than re-citing CAMELS
  directly; add `villaescusanavarro21` if Methods is expanded to restate the
  simulation suite in detail.
- IllustrisTNG: Weinberger et al. (2017), Pillepich et al. (2018), Nelson et
  al. (2019) — same reasoning as CAMELS above.
- Lin et al. (2025, arXiv:2509.01881) — motivates/parallels the 2-D
  (\S\ref{sec:results5}) and 1-D field (\S\ref{sec:field}) latent findings;
  **not currently cited by key** even though the text discusses "a
  low-dimensional feedback latent" — consider adding `lin25` as a citation on
  the relevant sentences in \S\ref{sec:results5}/\S\ref{sec:field} in a
  revision pass.
- Constantine (2015) — active-subspaces method, cited alongside the
  PCA/latent machinery in the sources but not currently in main.tex.
- Lipman et al. (2023, arXiv:2210.02747); Tong et al. (2023,
  arXiv:2302.00482) — BIND's own flow-matching/OT engine; likely already
  cited in Paper I and not re-cited here since \S\ref{sec:suite} only recaps
  the suite at a high level.
