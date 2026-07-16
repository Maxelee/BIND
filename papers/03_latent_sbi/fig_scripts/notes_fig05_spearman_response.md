# notes: fig05_spearman_response

**Script**: `fig_scripts/fig05_spearman_response.py` -> `figs/fig05_spearman_response.pdf`

## Layout change from the placeholder (content-preserving simplification)
The placeholder (`f1_corr_heatmaps.png`) is a 2x4 grid (7 heatmaps + 1 blank),
each panel repeating its own copy of all 30 rotated parameter-name x-tick
labels at ~5.5pt inside a ~20x9 inch canvas. Reproduced at the suite's
TWO_COL width (7.2 in) that layout put 30 labels into <2 inches per panel
four times over -- illegible at print size (violates the FIGURE_STYLE
quality gate).

**Fix**: same 7 statistics, same per-bin Spearman-r values, same shared
column order (by peak |r| with the C_l suppression) -- but stacked as ONE
column of 7 panels (`sharex=True`) so the 30 parameter names are drawn ONCE,
at the bottom, at full 7.2-in width and a legible 6.5pt. This is a pure
rearrangement (7 separate axes either way; here they share one x-axis
instead of being tiled 4-across) -- no data, no per-panel value, and no
statistic dropped or added.

Per-panel colorbars (8, one mostly-blank) also replaced by ONE shared
colorbar for the whole column: the 7 panels' individual 99th-percentile
|r| range 0.38-0.52 (computed in-script), close enough that a single shared
`vmin=-vmax,vmax=vmax` does not wash out any panel relative to the
placeholder's per-panel normalization, and this satisfies the FIGURE_STYLE
"one shared colorbar per row/panel-group when panels share units" rule more
cleanly than 7 separate ones.

## Content mapping
- Panel (a) = C_l suppression S(ell), y-axis = ell (log-spaced ticks, same
  100 < ell < 2e4 WL-band cut as the placeholder).
- Panels (b)-(g) = peak counts, minima counts, kappa PDF, Minkowski V0/V1/V2,
  y-axis = the statistic's native bin (nu or kappa), rebinned exactly as the
  source notebook (`REBIN = {peak:4, minima:4, pdf:3, mf_v0/v1/v2:2}`) to
  suppress per-run shot noise.
- Colormap: `RdBu_r`, diverging, centered at 0 (matches FIGURE_STYLE rule 6
  for a diverging quantity).
- `panel_label` used for both the panel letter AND a short statistic name
  (e.g. "(a) C_l suppression S(l)") in place of the placeholder's
  `ax.set_title(...)`, since titles are forbidden by house style; this text
  is placed inside the axes, upper-right.
- The placeholder's `fig.suptitle(...)` ("Model-free per-bin parameter
  response (Spearman r...)") is dropped (no titles/suptitles allowed); the
  `n=253, z_s=1` annotation is kept as small in-figure text (top-right,
  above panel a) since it is data provenance, not a title.

## Numbers reproduced (printed by the script; match main.tex sec:res-spearman)
```
IMFslope                           |r|=0.49   (main.tex: 0.49)
VariableWindVelFactor              |r|=0.48   (main.tex: 0.48)
BlackHoleRadiativeEfficiency       |r|=0.43   (main.tex: 0.43)
WindEnergyIn1e51erg                |r|=0.34   (main.tex: 0.34)
```
Exact match to all reported digits.

## No caption changes needed
`main.tex`'s caption (\label{fig:spearman}) describes the content
qualitatively ("response concentrated in a handful of wind/SN and BH
parameters... left columns, dark bands") and does not reference panel
count/layout or specific colors, so no caption edit is required despite the
1-column-of-7 vs 2x4 rearrangement.

## Data provenance
`/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` only --
`t__suppression__value/valid` + its axis, `t__peak_counts__value/valid` +
`a__peak_counts__nu`, `t__minima_counts__value/valid` + `a__minima_counts__nu`,
`t__pdf__value/valid` + `a__pdf__pdf_bins`, `t__mf_v0/v1/v2__value/valid` +
`a__mf_v0/v1/v2__mf_nu`, `X_unit`, `param_names`, `source_redshifts`.
`bin_param_correlation`/`param_importance`/`stat_axis` ported verbatim from
`sobol-sb35/examples/wl_latent_sbi.py` (Spearman rank correlation via
`scipy.stats.rankdata`, closed-form, no fitting/emulator).
