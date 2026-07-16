# notes: fig13_bridge_residual

## Status
Regenerated from cached data. Content matches the placeholder exactly
(same three panels, same curve shapes/ranges, same AGN/SN/wind/other
counts n=18/31/8 reproduced by the console printout).

## Panel mapping (unchanged from placeholder, restyled to suite conventions)

- **(a)** "what fills f_gas's blind spot" — partial correlation
  `r[S(ell), X | f_gas]` vs `ell`.
  - Structural (solid lines, green shades): `c_dm` = `#1b7837` (medium-dark
    green), `c_gas` = `#5aae61` (light green), `f_star` = `#00441b`
    (darkest green — this is the curve that rises highest, to ~0.93 at
    ell~1e4).
  - Thermal (dashed lines): `log Y` = `#c1272d` (red), `log K` = `#e08214`
    (orange).
  - In-axes text labels "structural" (green) / "thermal" (red) group the
    two families; repositioned vs. the source notebook (moved to (230,
    0.93) and (2200, -0.72) in data coords) purely to avoid overlapping
    the now-differently-sized legend — no data change.
  - Grey vertical band ell in [1000, 2000] = the WL-informative reference
    range used elsewhere in the bridge figures (fig06/fig07/fig12).
- **(b)** "a 2nd variable restores it" — black = `f_gas` only (`|r|`),
  green (`#1b7837`, same hue as panel (a)'s `c_dm`) = 2-variable multi-R
  of `{f_gas, c_dm}`. Black here is a neutral baseline color, not the
  suite's "hydro truth" semantic (no hydro-truth curve appears in this
  figure).
- **(c)** "the residual is feedback-specific" — mean off-bridge residual
  `S - bridge(f_gas)` by family. Recolored to the suite's semantic
  `COLORS` dict since the hues already matched closely: AGN ->
  `COLORS["highlight"]` (red, was `#c1272d`), SN/wind -> `COLORS["bind"]`
  (blue, was `#0b53c1`), other -> `COLORS["dmo"]` (grey, was `#9a9a9a`).
  Legend labels keep the `(n=NN)` counts from the source.

## Feedback-family classifier (unchanged, caption-relevant)

Uses this script's own keyword-only `_family(name)` (checks only the
varied parameter *name*, not its description) — reproduces the n=18
AGN / n=31 SN-wind / n=8 other split quoted in the existing caption. This
is a DIFFERENT classifier from the one behind fig06/fig07 (20/25/12
split, inspects name+description) and from `bind_bridge.py`'s version.
The paper's own `\todo{}` note on this discrepancy should be left as-is —
regenerating did not resolve it, only reconfirmed it (same 18/31/8 as the
placeholder).

## Style deltas from placeholder (styling only, no data changes)

- 3-panel figure now `TWO_COL` (7.2, 3.1) per FIGURE_STYLE (was ad hoc
  9.8 x 3.2 in the notebook).
- Per-axes `set_title` -> `panel_label(ax, "(a)"/"(b)"/"(c)")` in the
  upper-left corner; no figure/axes titles anywhere.
- Legend fontsize left at the suite default (`legend.fontsize=7` from
  `paper_style.setup()`) rather than the notebook's smaller ad hoc sizes
  (6.0-7.5) — still fits cleanly at TWO_COL width.
- Panel (c) colors standardized to the suite's semantic `COLORS` dict
  (see above) instead of the notebook's bespoke hex values — visually
  near-identical hues, now consistent with the rest of the suite's
  AGN/SN-wind/other palette usage.

## Caption

No caption text changes needed — the existing caption's content (panel
descriptions, the 18/31/8 counts, the `\todo{}` about the classifier
discrepancy) all still hold. Only the included file changes from PNG to
`figs/fig13_bridge_residual.pdf`.
