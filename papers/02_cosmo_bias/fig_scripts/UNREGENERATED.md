# Unregenerated figures

Two of the ten figures in this paper could not be regenerated from cached
data and remain the extracted-notebook PNG placeholders. Both are flagged
with `\todo{regenerate: ...}` in `main.tex` immediately above their
`\includegraphics`. Full provenance detail for both is in `DATA_MAP.md`
("fig07_selfcal.png" / "fig08_model_comparison.png" sections); summary:

## fig07_selfcal.png (`\label{fig:selfcal}`, Section 3.6)

- Placeholder is byte-identical to `examples/figures_lightcone/selfcal.png`
  (a real, non-fabricated result) but the source script
  (`examples/lightcone_selfcal.py`) computes the joint
  $(S_8,\Omega_{\rm m},A_{\rm bary})$ Fisher forecast **fresh at plot time**
  via `pyccl` (`ccl.Cosmology`, `ccl.WeakLensingTracer`, `ccl.angular_cl`)
  and never calls `np.savez` — the final plotted numbers exist nowhere on
  disk as an array.
- Regenerating this figure faithfully would require re-running a `pyccl`
  cosmology-derivative + Fisher computation, which is "computing science,"
  not loading a precomputed array, and is explicitly forbidden by this
  task's hard rules (no re-running analysis engines).
- **Action taken**: kept the placeholder PNG, added
  `\todo{regenerate: ...}` in `main.tex`, left `\label{fig:selfcal}` and
  caption untouched.

## fig08_model_comparison.png (`\label{fig:model_comparison}`, Section 3.7)

- Placeholder is byte-identical to
  `examples/figures_lightcone/model_comparison.png`. Only the BIND envelope
  half of the figure is cached (`transfer_Sell.npz`, shared with
  fig01/fig02); the three analytic-model spans it is compared against —
  BCM (Schneider15), van Daalen 2019, BCemu — are all computed fresh at
  plot time by `examples/lightcone_model_comparison.py`
  (`bcm_span()`/`vandaalen_span()` via `pyccl.baryons`, `bcemu_span()` via
  the external `BCemu` package), with no `np.savez` anywhere in the script.
- The BIND-envelope-only half is insufficient to reproduce the placeholder's
  actual content (a three-way analytic-model comparison), so a from-cache
  regeneration would silently drop the figure's scientific point rather
  than faithfully reproduce it.
- **Action taken**: kept the placeholder PNG, added
  `\todo{regenerate: ...}` in `main.tex`, left `\label{fig:model_comparison}`
  and caption untouched.

## Everything else

The remaining 8/10 figures (fig01, fig02, fig03, fig04, fig05, fig06, fig09,
fig10) were regenerated from cached `.npz` arrays into house-style PDFs;
see `fig_scripts/notes_fig*.md` for per-figure content/caption reconciliation
notes and `fig_scripts/DATA_MAP.md` for full data provenance.
