# notes: fig01_maps.png

## Panels (unchanged order/content vs placeholder)
- (a) DMO
- (b) spherical BCM -- this is `fid/mono/kappa_maps.npz` (crude circular-monopole
  control), **not** the current `fid/sph/` (which is now the faithful
  `bcm_warp` control). Same substitution the original notebook cell made at
  render time, before `sph/` was redefined -- see `fig_scripts/DATA_MAP.md`.
  The existing caption in `main.tex` (lines 210-222) already states this
  explicitly ("uses the earlier, superseded `mono` control... no rendered
  `bcm_warp` map exists in our source material") -- **no caption change
  needed**, this script's data matches what the caption already describes.
- (c) BIND (full)
- (d) BIND $-$ mono (labelled "anisotropy" in the original notebook prose;
  caption calls it $\kappa_{\rm bind}-\kappa_{\rm mono}$)

## Style changes from the placeholder
- Dropped `suptitle` and per-panel `set_title`; replaced with in-axes
  `(a)/(b)/(c)/(d)` + short name tags (white-boxed text, upper-left) per
  FIGURE_STYLE rule 3.
- Colormap: `cividis` for (a)-(c) (shared vmin=0, vmax=99th-pct of BIND
  panel, same as source cell), `RdBu_r` diverging (symmetric, 99th-pct of
  |residual|) for (d) -- unchanged from source.
  values from run: vmax_cividis=0.0851, va_RdBu=0.01433.
  print diagnostics: rms dmo=2.34e-02 mono=2.32e-02 bind=2.29e-02, residual
  rms = 16.0% of bind rms (matches placeholder's printed stdout exactly).
- Each panel gets its own colorbar (fraction=0.046) rather than one shared
  colorbar for (a)-(c) -- a shared colorbar via `fig.colorbar(im, ax=[...])`
  produced label/tick collisions with neighboring panels in `GridSpec`; four
  individual colorbars (all sharing the same numeric range for a-c) convey
  the same comparison without the layout fragility, and matches how the
  original notebook cell drew it (one colorbar per panel).
- Added physical-scale axes: extent set to `[0, 5, 0, 5]` deg (fov_deg=5.0).
  **Update (verifier pass):** every panel now carries its own
  `$\theta_x$ [deg]` x-axis label (previously only panel (a) did, which
  left panels (b)-(d) with bare numeric ticks and no explicit unit --
  flagged as a medium finding against FIGURE_STYLE rule 5, "axis labels
  always, with units, on every panel"). Only panel (a) keeps the
  `$\theta_y$ [deg]` y-axis label since it is the only panel with y-tick
  labels shown (panels (b)-(d) share the same extent, `set_yticklabels([])`,
  and would duplicate the row's y-axis otherwise). The original notebook
  cell had bare `set_xticks([]); set_yticks([])`.
- `rasterized=True` on all `imshow` calls (vector PDF + raster maps).

## Caption
No caption edits needed -- `main.tex` lines 209-224 already correctly
describe this as the mono/pure-spherical control, not bcm_warp. Colorbar
label text now reads "$\kappa$" (a-c) and "$\Delta\kappa$" (or
$\kappa_{\rm bind}-\kappa_{\rm mono}$, both appear as axis/colorbar labels
in the script) -- semantically identical to the caption's own notation, no
symbol clash.
