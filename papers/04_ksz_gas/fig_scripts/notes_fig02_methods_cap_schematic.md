# notes: fig02_methods_cap_schematic

**Panels kept**: all 3, unchanged from placeholder (a) fiducial tau stack
by mass bin, (b) SHMR 2-D histogram + DESI M* cuts, (c) CAP filter
schematic (synthetic).

**Content mapping** (for caption pass):
- (a): 4 viridis-colormap lines (purple->green->yellow-green), BGS bin
  (index 1, logM=[13.4,13.8]) drawn thicker/higher-zorder, matching
  placeholder. Grey shaded clean band 0.3<x<1.5.
- (b): blue 2-D histogram (log10 counts) of BIND SHMR at the BGS
  snapshot; orange star/dashed/dotted = M*>11.0 cut -> logM200=13.56;
  red star/dashed/dotted = M*>11.25 cut -> logM200=13.81
  (`COLORS["highlight"]` for the red cut, was `tab:red` — same hue).
- (c): black curve = illustrative analytic tau(theta) profile (NOT real
  data, matches placeholder exactly — `(1+(theta/1.5)^2)^-1`); blue fill
  = CAP disk (+1, theta<theta_d); red fill = CAP ring (-1, equal area,
  theta_d<theta<sqrt(2)*theta_d).

**Style changes from placeholder**: no panel titles — replaced with
in-axes `panel_label` (a)/(b)/(c). Panel (c) legend moved from
"upper right" to "lower right" to avoid overlapping the theta_d /
sqrt(2)*theta_d annotation text (both were competing for the same
upper-right corner at this figure's narrower width; content/wording
unchanged, only position). `imshow`/`pcolormesh` in panel (b) now has
`rasterized=True` per FIGURE_STYLE rule 8.

**No data changes.** Same 2 npz files + same parquet columns as the
notebook; panel (c) remains fully synthetic (no cache file), as noted in
DATA_MAP.
