# notes: fig06_bias_sensitivity

## Content mapping (for the caption pass)
- Panel (a) [was left]: unchanged content -- ranked bars for the 12 most
  important of 30 astro params, `S_total` (total-order, red =
  `COLORS["highlight"]`) and `S_first_bin` (binned first-order, blue =
  `COLORS["bind"]`), values read directly from `bias_sensitivity.npz`,
  verified numerically identical to placeholder (IMFslope S_T=0.359,
  S_i=0.180; BlackHoleRadiativeEfficiency S_T=0.323, S_i=0.139; etc.).
  Bar order and parameter names unchanged.
- Panel (b) [was right]: unchanged content -- Delta_S8 (ell_max=3000) vs
  IMFslope (top driver, normalized [0,1] via the same `load_design()`
  log/min-max transform as the source script), colored by group f_gas.
  Colormap changed from `viridis` to `cividis` (house-style default);
  point styling (edgecolor black, lw 0.3) unchanged.

## Panels kept/dropped
Both panels kept. `S_first_surr` (surrogate-Sobol first order) and `perm`
(permutation importance) are cached in `bias_sensitivity.npz` but were
never plotted in the original figure either -- not drawn here either.

## Style changes from placeholder
- No title. The placeholder's panel-(a) title stated the surrogate
  cross-validated R^2 = 0.38 in text; this number is **not** carried into
  the regenerated figure (no titles allowed) and is **not** currently in
  the `main.tex` figure caption for `fig:bias_sensitivity` (it appears
  instead in the body paragraph immediately above the figure, "A
  gradient-boosted surrogate model of Delta_S8 achieves ... R^2=0.38").
  No caption change needed since the caption never quoted the R^2 value.
- Legend for panel (a) shortened ("first-order $S_i$" from "first-order
  $S_i$ (binned)"); no content change, still 2 entries.
- Figure size: `TWO_COL` width x 3.2in (was 13.5x5.6in at dpi 130), same
  1.5:1 width-ratio split between panels.

## Caption reconciliation needed
None -- existing caption for `fig:bias_sensitivity` (main.tex ~line 596)
matches this content; it already describes "IMFslope and
BlackHoleRadiativeEfficiency dominate" and "colored by group f_gas",
both preserved verbatim.
