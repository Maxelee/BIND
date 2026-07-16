# notes: fig05_cap_ratio_confront

Content-faithful re-plot of `cell014_out1.png` (CAP-ratio kSZ confrontation).
Single panel, no simplification (all elements of the placeholder kept: 2
DESI M* cuts x {fiducial curve, 16-84% SB35 band, best-fit node, data points
+ errorbars}, plus the theta(r200) dotted verticals and the y=1 "cosmic"
reference line).

## Color / style mapping (for the caption pass)
- Green = `M*>11.0` cut (BIND fiducial + SB35 band + data), matches the
  existing caption text ("green, orange") verbatim -- no caption change
  needed for color words.
- Orange = `M*>11.25` cut, ditto.
- Thin colored line (no separate legend swatch per cut) = the single
  chi2-best-fit ("coherent") node, alpha 0.85, same hue as its cut.
- Black dotted horizontal at f~gas=1 = cosmic baryon fraction; "cosmic" label
  kept as a small in-axes annotation (not a title).
- Vertical dotted lines (colored per cut) = theta(r200) for that cut's
  logM200.
- Legend shortened to <=4-word entries per FIGURE_STYLE (dropped the
  descriptive top-of-axes text "CAP-ratio (paper's filter) sigma_v-free
  M*-matched" that appeared in the notebook version -- that context now
  belongs in the caption only, per FIGURE_STYLE rule 3).

## Data / numbers verified against placeholder
- theta range 0-10.5 arcmin, f~gas range 0-1.45, both match.
- SB35 16-84% band shape/amplitude matches (green band sits slightly above
  orange at large theta, tracking placeholder).
- Best-fit node curves lie close to the fiducial curves, consistent with
  chi2~0.1 quoted in the text.

## Nothing dropped
All panels/series from the placeholder are present; this is a 1:1
re-plot from `KS/fgas_cap_mstar_snap085.npz` +
`KS/desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm{11.00,11.25}.npz`, no new
computation beyond what cell 14 already did (theta(r200) conversion + chi2
best-node selection, both ported verbatim).
