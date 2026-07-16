# notes: fig06_money_forecast

Content-faithful re-plot of `cell028_out0.png` (forecast corner plot). Pure
re-plot of pre-computed MCMC chains, no re-sampling.

## Color / style mapping (for the caption pass)
- kSZ only (density) -> `COLORS["bind"]` blue -- matches existing caption
  text ("blue") exactly.
- tSZ only (pressure) -> `COLORS["secondary"]` green -- matches existing
  caption text ("green") exactly.
- kSZ+tSZ joint -> `COLORS["highlight"]` red -- matches existing caption
  text ("red") exactly.
  No caption color-word changes needed.
- Legend entries shortened to 2-3 words each (dropped the "(density)" /
  "(pressure)" parentheticals from the notebook version's legend text,
  which are already carried by the caption).

## Sizing note
Used a custom 4.6x4.6in square instead of `ONE_COL_SQ` (3.5x3.4in): corner's
default 2-parameter triangular layout (its own default is 5.5x5.5in) needs
more room than `ONE_COL_SQ` to keep the two-line per-axis loading labels
("dir1 (nu=0.10) [prior sigma] / +VariableW -BlackHole +WindEnerg") from
overlapping at the bottom -- confirmed by testing `ONE_COL_SQ` first (labels
collided) then widening per FIGURE_STYLE rule 7 ("resize the figure instead"
of shrinking fonts). This is a 3-populated-panel (hist/joint/hist)
triangular figure, consistent with the rule's "map galleries and >=3-panel
rows" sizing exception, just kept as a custom square rather than the
non-square `TWO_COL`/`TWO_COL_TALL` presets since corner plots are inherently
square.

## Dropped from the notebook cell (not part of the figure)
The notebook cell's tail also prints diagnostic text (fraction of the
forecast direction 1 captured by the Sec-4 (fig08) response-latent plane,
loaded from a `/tmp/bind_latent.npz` scratch file written by an earlier
notebook cell, plus per-param prior-ratio counts). These numbers appear in
the paper's prose (Sec. "Results 5" paragraph, "0.81 inside the 2-D latent
plane... 0.26 for a random direction") with their own `% src:` provenance
and are not part of the figure itself -- omitted here since (a) they render
no plot element and (b) the scratch `/tmp` file is not a legitimate cached
data artifact.

## Data / numbers verified against placeholder
- Direction labels (nu=0.10 / nu=0.16) and top loadings
  (+VariableW -BlackHole +WindEnerg / +SNIa_Rate +WindFreeT +RadioFeed)
  match the placeholder exactly (deterministic from the cached chain
  covariance).
- Contour shapes/extents (blue widest, green similar width but different
  orientation, red tightest, centered near origin) match.
