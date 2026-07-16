# notes: fig05_beyond2pt_manifold

## Content mapping (for the caption pass)
- Panel (a) [was left, untitled originally "Dimensionality of the feedback
  response"]: unchanged content -- cumulative feedback-manifold variance vs
  N baryon-template modes, 10 statistics, `turbo` qualitative colormap
  (same as original) sampled across the 10 curves in the same STATS order
  as the source script. Solid+marker = signal-to-noise >= 1 (well-measured);
  dashed, lower alpha = noise-limited (kappa peaks, kappa minima). Red
  vertical line at N=2 (`COLORS["highlight"]`) marks the capstone N=2
  result. Horizontal dotted line at 0.90 unchanged.
- Panel (b) [was right, untitled originally "Feedback-direction overlap..."]:
  unchanged content -- 10x10 principal-angle subspace-alignment matrix,
  cell values identical to the placeholder (verified numerically, e.g. row
  1 = 1.00/0.98/0.98/0.87/0.65/0.78/0.82/0.73/0.90/0.96, matches placeholder
  exactly). Colormap changed from `magma` to `cividis` (house-style default
  for non-diverging map colormaps); values/labels unchanged.
- Both panels' legend/tick labels shortened to short math labels (e.g.
  "$\kappa\,C_\ell$ (2pt)" instead of "kappa C_ell (2pt)") -- same
  statistics, same order, no content change.

## Panels kept/dropped
All content kept; nothing dropped. (`kappa_cost_anchor()`'s printed
console cross-check was never part of the figure and is not reproduced,
per DATA_MAP.)

## Style changes from placeholder
- No title (was two-line titles above each panel) -- replaced by
  `panel_label` "(a)"/"(b)" tags; the title text ("well-measured statistics
  are all <=2-3D", "1 = same directions, <1 = complementary ->
  self-calibration") is exactly what the existing main.tex caption already
  states in prose, so no information is lost.
- Legend for panel (a) moved to a single column (was 2-col) at smaller
  font to fit `ONE_COL`-derived `TWO_COL` sizing; "(noise-limited)" suffix
  shortened to "(noise)" to keep legend entries short.
- Figure size: `TWO_COL` width (7.2in) x 3.15in (was 13.5x5.2in at dpi 130)
  -- matches the paper's other 2-panel `\textwidth` figures in this suite.

## Caption reconciliation needed
None -- the existing `main.tex` caption for `fig:beyond2pt` (lines ~561-567)
describes exactly this content and does not name specific colors/styles
that changed (it refers to "solid curves"/"dashed" which is preserved).
