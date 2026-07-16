# notes: fig10_bias_tomography_superseded

## SUPERSEDED framing -- keep in caption
This figure documents a since-corrected, flawed methodological step (main
text Sec. 3.6, "Corrected tomographic self-calibration" paragraph). The
existing `main.tex` caption already states this explicitly ("retained here
only as a methodological cautionary figure ... **This figure is superseded
by Figure~\ref{fig:selfcal} and is not the paper's self-calibration
result.**") -- that framing must be kept verbatim by whoever integrates
this regenerated image; nothing in this regeneration changes the
superseded status or its content.

## Content mapping (for the caption pass)
- Panel (a) [was left, "Per-plane bias"]: Delta_S8(z_s) spaghetti, 99 runs
  + black mean line, unchanged data (`dS8z`, `zs` read directly from
  `bias_tomography.npz`).
- Panel (b) [was middle, "z-shape universality"]: same runs normalized to
  each run's z_s=1 value, median (red = `COLORS["highlight"]`) + 16-84%
  band (`BAND_ALPHA`), unchanged data/transform (same `ref`/`good` mask
  logic as the source script).
- Panel (c) [was right, "Rank of Delta_S8(z_s)"]: SVD variance-fraction bar
  chart of the RAW (not centered) matrix, `var_frac_raw` read directly from
  the cache -- PC1 = 99.7040% (verified numerically: `var_frac_raw[0] =
  0.99704`), matching the caption's "PC1 approx 99.7%" and the placeholder
  bar heights exactly.

## Panels kept/dropped
All 3 panels kept.

## DEVIATION FROM PLACEHOLDER (documented, not a content change to the
## plotted arrays)
The placeholder's panel (a) additionally color-codes each of the 99 lines
by group f_gas (a `viridis` colormap + colorbar), which required loading
`cosmo_bias.npz["fgas"]` -- a file **not** listed in this figure's
DATA_MAP.md data-file entry (only `bias_tomography.npz` is listed) and
**not** referenced by the existing figure caption (the caption describes
only the spaghetti/mean, normalized shape, and SVD rank -- never the
f_gas color axis). Per the task's data-provenance scoping, this
regeneration loads only the single mapped file and draws panel (a) as
plain grey ensemble lines + a black mean line (matching panel (b)'s
existing grey-ensemble convention) rather than adding an unmapped file.
**No plotted array/statistic changed** -- only a coloring convenience was
dropped. If an integrator wants the f_gas coloring restored, `cosmo_bias.npz`
(`fgas`, keyed by the same `run_idx` present in `bias_tomography.npz`) is
available and was already used by figs 01/03/06 in this paper.

## Style changes from placeholder
- No titles; `panel_label` "(a)"/"(b)"/"(c)" replaces the three per-panel
  titles ("Per-plane bias...", "z-shape universality...", "Rank of
  Delta_S8(z_s)..."). The R^2/percentage figures in those titles (PC1
  99.7%) are already stated in the `main.tex` caption text, so no
  information is lost.
- Figure size: `TWO_COL` width x 2.6in (was 16.5x4.8in at dpi 130).
- Colorbar (f_gas) removed along with the dropped coloring (see deviation
  note above).

## Caption reconciliation needed
None for content; the "SUPERSEDED" framing must simply be preserved by the
integrating agent, as instructed.
