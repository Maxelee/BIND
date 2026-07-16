# notes: fig04_cross_spectra

**Output**: `figs/fig04_cross_spectra.pdf` (+ `figs_preview/fig04_cross_spectra.png`)
**Script**: `fig_scripts/fig04_cross_spectra.py`
**Data**: `runs/bind/run_0000/Cl_kappa_y.npz` (cheaper cached equivalent of the
notebook's on-the-fly `S.cl_kappa_y(...)` call — same statistic, same
fiducial realization set, no recomputation).

## Panel mapping (unchanged data/curves from placeholder)

- (a) $\ell(\ell+1)|C_\ell^{\kappa y}|/2\pi$ vs $\ell$, one curve per source
  plane $z_s \in \{0.5, 1.0, 1.5, 2.0, 2.44\}$ (all 5 lux planes, same as
  placeholder).
- (b) $\ell(\ell+1)C_\ell^{yy}/2\pi$ vs $\ell$ — tSZ auto-spectrum, single
  curve.
- Both panels log-log, $\ell \in [10^2, 2\times10^4]$.

## Caption-relevant changes vs. placeholder

- **Panel (a) line colors**: placeholder used matplotlib's default
  qualitative `tab10` cycle (blue/orange/green/red/purple, unordered).
  Regenerated with a `viridis` sequential colormap ordered by $z_s$ (dark
  purple = $z_s{=}0.5$ -> yellow = $z_s{=}2.44$), since $z_s$ is a
  monotonic physical quantity — same 5 curves/data, more informative color
  choice per the suite semantic-color convention. If the caption/legend
  text names specific colors, update to "purple->yellow with increasing
  $z_s$" rather than a categorical list.
- **Panel (b) color**: switched from `firebrick` to `COLORS["highlight"]`
  (`#FF2C00`, the suite's fixed semantic color for "the single curve being
  called out"). Same underlying `cl_yy` curve.
- Removed the placeholder's in-axes titles ("WL$\times$tSZ cross
  $C_\ell^{\kappa y}$", "tSZ auto $C_\ell^{yy}$") per the no-titles rule;
  panel identity is now carried by `panel_label` tags (a)/(b) plus the
  distinct y-axis labels. Caption should introduce panels (a)/(b) by name.
- No data, statistic, or realization count changed from the source notebook
  cell (which itself just calls `bind.inference.stats.cl_kappa_y` on the
  same cached 100-realization fiducial lux run that produced this cache
  file).
