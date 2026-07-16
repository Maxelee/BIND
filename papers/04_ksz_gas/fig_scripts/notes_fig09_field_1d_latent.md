# notes: fig09_field_1d_latent

Script: `fig_scripts/fig09_field_1d_latent.py`. Data:
`/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` only. Reproduces
`examples/paper_ksz_field.ipynb` cell 10 verbatim (plus the handful of
constants-cell lines it depends on: `stat()`, `F_B`, `fgas`).

Recomputed numbers (match the placeholder to the shown precision):
- `lam[:5] = [0.948, 0.043, 0.003, 0.002, 0.001]`, `lambda1 = 0.95`,
  cumulative-at-2 = 0.991.
- `r_fgas = +0.49`.
- Same 9 leading parameters in panel (c), same sign pattern, same
  approximate magnitudes (IMFslope / VariableWindVelFac negative,
  BlackHoleRadiative / WindEnergyIn1e51erg positive, etc.).

## Content mapping (placeholder -> regenerated)

- Panel (a) scree plot: bars = `tab:blue` -> `COLORS["bind"]` (semantic BIND
  color, since the response being decomposed is BIND's emulated field
  output). Cumulative marker line: black, unchanged (`COLORS["truth"]`,
  visually identical to original "ko-"). 95%-threshold dotted line:
  `tab:red` -> `COLORS["highlight"]` (same red family, now the fixed
  suite-wide "called out" color).
- Panel (b) latent scatter: colored by `f_gas` (cosmic units). Colormap
  changed from the original `RdYlBu_r` to the suite default `cividis`
  (rcParams `image.cmap`, applied automatically since no `cmap=` was passed)
  per FIGURE_STYLE's semantic-color guidance — same values, same ordering
  (low f_gas = dark, high f_gas = light), just a different (colorblind-safe,
  perceptually uniform) palette. Content (point positions, `r_fgas=+0.49`,
  arrow annotation) unchanged.
- Panel (c) loadings: `tab:orange`/`tab:purple` -> `COLORS["highlight"]`
  (red, e1/gas axis) / `COLORS["secondary"]` (green, e2) — same two bars,
  same 9 params, same order, only the color pair changed to the fixed
  suite palette.
- All three `ax.set_title(...)` calls (which literally spelled out "(a)
  dimensionality — ~1-d" etc.) were dropped and replaced with bare
  `panel_label(ax, "(a)")` / `(b)` / `(c)` tags per FIGURE_STYLE rule 3 (no
  titles). The descriptive parts of those titles are already covered by the
  existing main.tex caption for `fig:field1d`, so no caption change is
  needed.
- The "(~4%)" in panel (b)'s y-axis label was hardcoded text in the source
  notebook; here it is computed live as `lam[1]*100:.0f` (currently renders
  "~4%", i.e. identical output) so it can't silently go stale.
- Figure size: original was a bespoke `9.8x3.0in` 3-panel row with width
  ratios `[1, 1.15, 1.4]`; regenerated at `TWO_COL = 7.2x3.1in` with the
  same width ratios (FIGURE_STYLE mandates `TWO_COL` for >=3-panel rows).

## Caption

No changes needed — `main.tex`'s existing caption for `fig:field1d`
(lines ~730-738) describes panel content generically ("scree plot",
"dominant gas-amplitude latent versus... second latent, colored by
f_gas", "parameter loadings") and does not name specific colors, so it
remains accurate for the regenerated figure.

## Format

Saved as `figs/fig09_field_1d_latent.pdf` (+ PNG preview in
`figs_preview/`). Integrating agent should update the `\includegraphics`
extension from `.png` to `.pdf` in `main.tex`.
