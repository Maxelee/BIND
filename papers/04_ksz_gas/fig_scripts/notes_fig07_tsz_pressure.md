# notes: fig07_tsz_pressure

Content-faithful re-plot of `cell022_out0.png` (tSZ pressure-leg 2-panel
consistency check). Both panels of the placeholder kept.

## Layout note
Placeholder was a 1x2 side-by-side row (figsize 8.8x3.6in). Per
FIGURE_STYLE rule 7 ("ONE_COL by default; TWO_COL only for map galleries and
>=3-panel rows"), this 2-panel figure was **stacked vertically** at
`ONE_COL`-family width (3.5in) instead of using `TWO_COL` width for a
2-panel row -- height extended to 5.1in to fit both panels legibly, per rule
7's "resize the figure, don't shrink fonts." Panel order (a) amplitude,
(b) shape preserved top-to-bottom instead of left-to-right.

## Color / style mapping (for the caption pass)
- `COLORS["bind"]` blue, alpha 0.10 = full 256-node SB35 ensemble -- matches
  the existing caption's "(light blue)" callout in spirit (same hue family,
  low alpha reads as "light").
- `tab:orange` = the 49 kSZ-CAP-consistent nodes -- matches the caption's
  "(orange)" callout exactly, no caption change needed.
- `tab:purple` = BIND fiducial curve + mass-systematic band. Not named in
  the caption (caption only calls out light-blue/orange), kept as a third,
  visually distinct hue from the placeholder rather than forced into the
  suite `COLORS` dict, since it is a locally-scoped "the one highlighted
  fiducial run" role that doesn't map cleanly onto BIND/truth/DMO/highlight
  here (both the ensemble AND the fiducial are BIND-painted).
- Black open squares = ACT x DESI LRG data (real observational data,
  `COLORS["truth"]`-equivalent black, kept as plain "k"/white-face markers
  matching the placeholder).
- Panel (a) legend moved from the notebook's free-floating in-panel text
  box (unchanged) plus title "(a) amplitude -- mass-selection-dominated" to
  a `panel_label(A, "(a)", loc="upper right")` tag (style rule 3: no
  titles); the descriptive "mass-selection-dominated" / "don't compare"
  framing from the original titles now belongs in the caption only (already
  present in the current main.tex caption).
- Panel (b) annotation shortened ("data ROLLOVER (compact) / BIND plateaus
  (extended)" -> "data compact, / BIND extended") to fit the narrower
  stacked-panel width; same meaning.

## Data / numbers verified against placeholder
- yr (BIND/data at 3.5'), rlo/rhi (mass-syst band factor), rho (kSZ/tSZ
  chi2 rank correlation) all recomputed identically from the same cached
  arrays and print the same in-panel annotation text as the original
  (BIND ~1.5x data, mass syst 1.3-2.1x, rho=0.80).
- Panel (b) peak locations (BIND ~4.75', data ~3.5') match the placeholder
  and the main.tex caption's stated numbers.
