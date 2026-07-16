# notes: fig01_lightcone_design_hmf

**Panels kept**: all 3, unchanged from placeholder (a) lightcone coverage,
(b) Sobol design in physical units, (c) halo mass function.

**Content mapping** (for caption pass):
- (a): blue circles+line = halo counts/snapshot (`COLORS["bind"]`, was
  `tab:blue` — same hex family, no caption change needed). Open red
  circle = BGS snap (z=0.18) using `COLORS["highlight"]`. Open green
  circle = ELG snap (z=1.16) using `COLORS["secondary"]`.
- (b): purple dots = 256-node Sobol design (unchanged `tab:purple`), black
  star = fiducial TNG.
- (c): solid grey step = BGS-slice HMF, dashed grey step = ELG-slice HMF
  (unchanged `0.35` grey), navy dotted verticals = BGS stacking-bin edges,
  blue shaded band = BGS science bin (`COLORS["bind"]` at alpha 0.20, was
  `tab:blue`), red/purple/green horizontal bars = DESI BGS/LRG/ELG host
  mass ranges (`COLORS["highlight"]`/`tab:purple`/`COLORS["secondary"]`).

**Style changes from placeholder**: no panel titles (dropped
"(a) lightcone coverage" etc. title text) — replaced with in-axes bold
`panel_label` tags (a)/(b)/(c); panel (a) label moved to lower-right (empty
region) to avoid colliding with the BGS-snap circle marker at upper-left.
Figure resized to the suite's `TWO_COL` width (3-panel row, per
FIGURE_STYLE rule 7). Legend fontsize trimmed to fit suite conventions;
content/entries unchanged.

**No data changes.** Same parquet columns, same bin edges, same DESI
host-mass literature dict as the notebook.
