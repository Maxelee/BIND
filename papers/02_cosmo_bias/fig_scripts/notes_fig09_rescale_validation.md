# notes: fig09_rescale_validation

## Content mapping (for the caption pass)
- 3 rows (mild / low-$\Omega_{\rm m}$ / extreme corner) x 2 columns (P(k)
  ratio / cumulative HMF ratio), same as placeholder. Row order unchanged
  (mild, low-Om, corner-hi -- matches `main.tex` text "top row" / "middle
  row" / "bottom row" references).
- Blue = `COLORS["bind"]` = control (unrescaled sim vs its own original
  cosmology's halofit/Tinker08 theory at the snapshot redshift z*). Red =
  `COLORS["highlight"]` = rescaled (box relabeled by AW10 s, mass_factor)
  vs the TARGET cosmology's theory at z'=0. Same curves/values as
  placeholder (verified: mild s=0.888, z*=0.00; low-Om s=1.868, z*=0.46;
  corner-hi s=5.000, z*=1.41 -- matches placeholder legend text exactly).
- y-limits unchanged: P(k) column (0.75, 1.25), HMF column (0.5, 1.5) for
  all 3 rows (reproduces the placeholder's clipped BAO spikes / HMF
  breakdown at high mass in rows 2-3 exactly).
- Per-row (s, z*) values, previously in each panel's legend text, moved to
  a small in-axes annotation (bottom-right of the left-column panel) since
  house style limits legend entries to short generic labels
  ("control"/"rescaled", shown once in row 1 only, colors constant across
  all 6 panels).

## Panels kept/dropped
All 6 panels kept; nothing dropped. `n_ctrl`/`n_resc`/`rms`/`snap` fields
in the npz are cached but not plotted in either version.

## Style changes from placeholder
- No `suptitle` ("AW10 rescaling of TNG300-3-Dark to SB35 target
  cosmologies"); replaced by `panel_label` "(a) mild" / "(b)
  low-$\Omega_{\rm m}$" / "(c) extreme corner" on the left-column panel of
  each row (upper right, to avoid the row-1 legend at upper left).
  Row identification previously carried in the y-axis label prefix
  ("mild\nP(k) / halofit" etc.) is now in the panel label instead; y-axis
  labels are now uniform "$P(k)\,/\,{\rm halofit}$" /
  "$N({>}M)\,/\,{\rm Tinker08}$" across rows.
- Per-panel legends (6 legends showing s/z* in the label text in the
  placeholder) collapsed to one 2-entry legend (row 1 only) + small
  s/z* text annotation per row; same information, less visual repetition.
- Figure size: `TWO_COL` width x 6.6in (was 11x9.6in at dpi 130).

## Caption reconciliation needed
None expected -- existing caption for `fig:rescale_validation` (main.tex
~line 998) already describes "rows: mild, low-Omega_m, extreme corner" and
"Blue is the unrescaled ... control; red is the rescaled simulation", which
matches this regeneration exactly.
