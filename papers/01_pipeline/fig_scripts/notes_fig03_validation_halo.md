# notes: fig03_validation_halo

**Output**: `figs/fig03_validation_halo.pdf` (+ `figs_preview/fig03_validation_halo.png`)
**Script**: `fig_scripts/fig03_validation_halo.py`
**Data**: `halo_atlas/{fid,truth}_snap096.npz`, `tau_profiles/tau_profiles_snap096.npz`,
`profiles/{fid,truth}_snap096.npz` — all unchanged from placeholder, snap 96 (z~0.03).

## Panel mapping (unchanged data/curves from placeholder)

- (a) $\log_{10} Y_{500c}$ vs $\log_{10} M_{200c}$ — BIND (solid blue, `COLORS["bind"]`)
  with 16-84th pct scatter band (shaded, same color, alpha 0.18) vs. TNG300
  hydro truth (dashed, `COLORS["truth"]`).
- (b) stacked electron-column ratio $\tau_{\rm BIND}/\tau_{\rm hydro}$ vs
  $r/r_{500c}$, 3 mass bins ($10^{13.0}$-$10^{13.5}$, $10^{13.5}$-$10^{14.0}$,
  $10^{14.0}$-$10^{14.5}\,M_\odot/h$), grey band = +/-5% agreement, all bins
  with count >= 20 (matches source cell's cut).
- (c) projected Compton-$y$ profile $y(r)/y(r_{500c})$ vs $r/r_{500c}$, 4
  mass bins ($10^{13.0}$, $10^{13.4}$, $10^{13.8}$, $10^{14.2}\,M_\odot/h$
  bin centers), compared to the Arnaud+2010 universal GNFW pressure profile
  (dotted black), same hardcoded fit parameters as the source cell
  (P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905).

## Caption-relevant changes vs. placeholder

- **Mass-bin colors (panels b and c)**: both now use the *same* mass-ordered
  sequential `viridis` colormap. The original notebook cell already used
  viridis for panel (c) but the unstyled matplotlib default color cycle
  (blue/green/orange) for panel (b); this regeneration makes both
  mass-binned panels share one consistent color scale (same underlying
  curves, only the color assignment for panel (b) changed — same 3 mass
  bins, same line data). If the caption names panel-(b) colors explicitly
  ("blue", "orange", "green"), update to reference viridis ordering
  (dark-purple = lowest mass bin -> light-green = highest).
- Added `panel_label` tags (a)/(b)/(c). Corner placement: (a) upper-left
  (matches legend, no collision since legend sits just below), (b)
  lower-right (its default upper-left collided with the $M_{200c}$ legend
  title), (c) upper-right (its default upper-left collided with the
  bundle of profile curves, all of which start high at small $r$).
- No data, curve, or fit changed from the source notebook cell.
