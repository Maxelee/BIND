# notes: fig10_field_desY3_act

Script: `fig_scripts/fig10_field_desY3_act.py`. Data:
`/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` +
`/mnt/home/mlee1/ceph/bind_science/ksz_confront/desact_data.npz`.
Reproduces `examples/paper_ksz_field.ipynb` cell 12 verbatim (Hankel
transform `xi_gy`, physical `2.4'` ACT beam, robust `8'-40'` window).

Recomputed robust-window ratios (match main.tex body text exactly):
- bin 3 (DES bin index 2, $\bar z\approx0.74$): BIND/data $\approx 2.47\times$
  (quoted in the paper as $2.5\times$).
- bin 4 (DES bin index 3, $\bar z\approx0.94$): BIND/data $\approx 2.29\times$
  (quoted in the paper as $2.3\times$).

## Content mapping (placeholder -> regenerated)

- Curve colors: BIND band/solid/dotted were `tab:red` in the source
  notebook -> now `COLORS["bind"]` (blue), per FIGURE_STYLE's fixed
  semantic palette ("BIND=blue"). DES Y3 x ACT data points: `k` (black)
  unchanged, now referenced as `COLORS["truth"]` (real measured data =
  the suite's "truth" semantic slot). This is a styling-only change — the
  underlying values, curves, and points are numerically identical to the
  placeholder.
- Grey shaded "beam/res" / field-limit bands: unchanged (neutral `0.85`
  grey, not one of the suite's semantic colors since it denotes an
  excluded-range annotation, not a data series).
- Panel labels: the small in-axes `ax.text(.05,.93, "DES bin N (...)")`
  title-like label was folded into `panel_label(ax, "(a) DES bin 3
  ($\\bar z\\approx0.74$)")` / `(b) DES bin 4 (...)` — still an in-axes tag,
  not a matplotlib `set_title`/`suptitle`, so it complies with FIGURE_STYLE
  rule 3.
- **`fig.suptitle(...)` was DROPPED entirely.** The original notebook figure
  had `fig.suptitle("BIND shear x y vs REAL DES Y3 x ACT — ~1.5-2x high
  (too gas-bound) + too steep (grey = aperture-limited)", y=1.04)`.
  FIGURE_STYLE rule 3 forbids titles/suptitles outright, so it is removed
  here. This is a fortunate coincidence with a pre-existing content note in
  `main.tex`'s own caption for `fig:field_desy3` (lines ~770-772), which
  explicitly calls out that this same super-title's "~1.5-2x" rounding is
  STALE relative to the precise 2.5x/2.3x values used in the body text and
  in-panel annotations. **Caption action needed**: the caption's
  parenthetical "(The panel's own super-title rounds this to a stale
  '~1.5-2x' estimate ... )" no longer applies since the regenerated figure
  has no super-title at all — the integrating agent should either delete
  that parenthetical sentence or reword it to note the correction was
  already applied during regeneration, rather than leaving a live pointer
  to a title that no longer exists on the figure.
- Legend order fixed to match the placeholder's visual order (DES Y3xACT,
  BIND Sobol 16-84%, BIND 2.4' beam, BIND no-beam) — matplotlib's default
  handle ordering did not match despite matching draw-call order, so
  handles are passed to `ax.legend()` explicitly.
- Figure size: original `8.4x3.4in` 2-panel row -> `TWO_COL = 7.2x3.1in`
  per FIGURE_STYLE.

## Layout justification (2026-07-16 verification pass)

FIGURE_STYLE rule 7 literally reserves `TWO_COL`/`TWO_COL_TALL` for "map
galleries and >=3-panel rows", and fig07 (also exactly 2 panels) already
established the suite's precedent of stacking a 2-panel figure vertically
at `ONE_COL` width instead. fig10 deliberately keeps the `TWO_COL`
side-by-side layout rather than following that precedent, because the two
panels here are a direct paired comparison across DES source-redshift bins
(bin 3 vs bin 4) at the SAME theta-axis and the SAME y-scale -- the science
point (the two bins have essentially the same shape/amplitude excess,
2.5x/2.3x) is read by eye-tracking horizontally between the two panels at
fixed theta, which only works with a side-by-side row (stacking vertically
would still work but makes that horizontal comparison awkward across a much
taller page span). All in-panel text (panel labels, legend, "beam/res",
robust-ratio annotation) was re-floored to >=7pt in this pass and remains
fully legible at the `TWO_COL` size (7.2x3.1in, each panel effectively
ONE_COL-width) -- confirmed by inspecting figs_preview/fig10_field_desY3_act.png.
Kept as `TWO_COL` by exception rather than restacked; flagged here per rule
7's own escape hatch (cf. fig06_money_forecast.py's custom-size
justification) rather than silently deviating.

## Caption

`main.tex`'s caption for `fig:field_desy3` (lines ~761-773) is otherwise
accurate (band/solid/dotted line-style description, robust-window ratios,
grey-region explanation) but its parenthetical about the stale super-title
should be revisited per the note above — flagged for the caption-integration
pass, not fixed here (this agent does not edit main.tex).

## Format

Saved as `figs/fig10_field_desY3_act.pdf` (+ PNG preview in
`figs_preview/`). Integrating agent should update the `\includegraphics`
extension from `.png` to `.pdf` in `main.tex`.
