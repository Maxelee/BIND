# Leftovers — Paper II (02_cosmo_bias)

Results/figures the dossier surfaced but that were deliberately left out of
`main.tex`, one line each with the reason.

- **`cl_sbi_corner.png` (all 5 subdirs `A_binned216`…`E_fulltomo`)** — a
  30-parameter posterior-corner plot from a *different* analysis
  (`lightcone_cl_sbi.py`, auto-Cl NPE on 216 runs); dossier explicitly flags
  this as mismatched to this paper's needs (belongs to Paper III per
  `PLAN.md`'s branch mapping) and only usable as a Discussion footnote
  cross-reference. Left out entirely rather than mis-captioned as this
  paper's own (S8,Ωm,a1,a2) posterior.
- **Rendered $(\Omega_{\rm m},\sigma_8,a_1,a_2)$ posterior corner for the
  LSST-Y1 full-likelihood result** — no such figure exists on disk;
  `lightcone_shear_forecast.py` only saves the raw `emcee` chain `.npz`.
  Flagged in-text with a `\todo{}` in Section 3.4 rather than fabricated or
  substituted with an unrelated figure.
- **99-run version of `cosmo_bias.png` bias-scatter figure vs. the 216-run
  capstone equivalent** — both exist; the paper uses `cosmo_bias.png` (99
  runs) for the bias-budget scatter/histogram (Fig. 3, Section 3.2) since it
  is the dedicated figure for that step, and separately quotes the 216-run
  capstone N=0 column numbers in the same section as an independent
  cross-check rather than swapping in a different image — avoids duplicating
  near-identical panels as two figures.
- **`shear_sweep_linear.npz` (companion "linear" comparison arm of the
  12-run shear sweep)** — the dossier notes this file exists alongside
  `shear_sweep.npz` but only the latter's `templates`/`cosmo` model
  comparison numbers were mined/verified; the `_linear` variant's specific
  numbers were not independently checked against a saved array in the
  dossier, so it is not quoted as a separate number in the paper (matches
  the dossier's own caution about not over-interpreting intermediate
  artifacts from this file).
- **Full 237-run `lightcone_shear_sweep.py` sweep + 34-dimensional
  full-emulator nuisance arm** — never run (deferred to `sbatch` per its own
  docstring, and out of scope for this drafting pass per the hard
  no-analysis rule). Mentioned qualitatively in Methods §2.6 and flagged as
  a limitation in Discussion §4.4(vii), but no quantitative results from it
  are quoted since none exist yet.
- **"Scale cuts vs. templates" quantitative comparison** — brief's outline
  bullet, but the dossier found no dedicated engine/result comparing the two
  approaches head-to-head. Written as a qualitative Discussion argument
  (Section 4.3) with an explicit `\todo{}` flagging the missing quantitative
  analysis, rather than inventing a number.
- **`kappa_cost_anchor()` reprise of the σ(S8) cost ladder at ℓ<3000 on the
  237-run beyond2pt dataset** — dossier notes this is computed
  (printed, not plotted) in `lightcone_beyond2pt.py` as an internal
  consistency check against the Results-3 template ladder, not a distinct
  headline number. Omitted from the main text to avoid a third, redundant
  restatement of the same N=2 cost result already given from the 216-run
  capstone and the Y1 full-likelihood analyses.
