# notes: fig01_map_gallery

**Output**: `figs/fig01_map_gallery.pdf` (+ `figs_preview/fig01_map_gallery.png`)
**Script**: `fig_scripts/fig01_map_gallery.py`
**Data**: `/mnt/home/mlee1/ceph/bind_science/demo_maps.npz` (unchanged from placeholder)

## Panel mapping (unchanged from placeholder, restyled)

- Panel 1 (left): DMO $\kappa$ — cividis colormap, shared color scale with panel 2
  (vmin/vmax = 1st/99th percentile of BIND $\kappa$).
- Panel 2 (middle): BIND $\kappa$ (z_s = 1) — cividis, shares the colorbar with panel 1
  (attached to panel 2, labeled $\kappa$).
- Panel 3 (right): BIND Compton-$y$ — inferno colormap, `LogNorm` (vmin = 50th
  percentile of positive $y$, vmax = 99.8th percentile), own colorbar labeled $y$.
- 5x5 deg field of view, axes in degrees ($\theta_x$, $\theta_y$), ticks at 0/2.5/5.

## Caption-relevant changes vs. placeholder

- **Colormap**: placeholder (the raw `demo_maps.png` render) used `viridis` for
  $\kappa$ and per-panel individual colorbars/titles. This regeneration follows
  FIGURE_STYLE.md's default map colormap (`cividis`) and the properly-styled
  source cell (`paper_lightcone_figs.ipynb` cell 3), which uses a *shared*
  color scale across the two $\kappa$ panels and drops per-panel titles in
  favor of in-axes labels ("DMO $\kappa$", "BIND $\kappa$", "BIND $y$",
  white text, upper-left of each panel) — same three underlying arrays,
  no data change.
- Adds a light (1-px) cosmetic Gaussian smoothing to the display arrays only,
  matching the source cell exactly (comment there labels it "cosmetic
  smoothing" — does not affect any statistic computed elsewhere in the paper,
  purely a rendering choice for the sky maps).
- If the caption or main text references specific colors/titles ("titled DMO
  kappa" etc.) those should be updated to reference the in-axes labels
  instead — no titles remain per FIGURE_STYLE.md.
