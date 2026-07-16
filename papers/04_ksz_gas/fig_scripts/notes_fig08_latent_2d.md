# notes: fig08_latent_2d

Content-faithful re-plot of `cell010_out0.png` (2-D feedback latent). All 4
placeholder panels kept.

## In-notebook state dependency (per DATA_MAP)
Cell 10 in the source notebook depends on `x, tau, y, nodes, fg` computed in
cell 8. This script ports those ~6 lines verbatim (loading
`KS/bind_tauy_xprof_snap085.npz` + the `fg` parquet query filtered to
`snap==85, log10(M_tot_500)>13.3`) before running cell 10's SVD/rotation
body unchanged. `PARAMS` (the 30-param list, schema-filtered against the
parquet) is also ported from cell 2 since it is needed for the panel-(b)
y-tick labels.

## Color / style mapping (for the caption pass)
- Panel (a): bars `COLORS["bind"]` blue (was `tab:blue`, same hue family),
  95%-line `COLORS["highlight"]` red (was `tab:red`) -- no meaningful
  change.
- Panel (b): `e1` (inner gas) loadings -> `COLORS["bind"]` blue (was
  `tab:orange`); `e2` (outer gas) loadings -> `COLORS["secondary"]` green
  (was `tab:purple`). The existing main.tex caption does not name colors
  for panel (b) (only lists the top 4 driving params), so no caption
  color-word conflict.
- Panels (c)/(d): scatter colormap switched from `RdYlBu_r` / `viridis` to
  `cividis` (FIGURE_STYLE default sequential map; both quantities are
  non-diverging physical gas fractions, so `cividis` is the correct choice
  per rule 6, not a diverging map).
- Panel titles ("(a) dimensionality", "(b) what drives each axis", "(c)
  colored by INNER gas f_gas(<R500) (r=+0.95)", "(d) SAME plane colored by
  OUTER gas ...") replaced with plain `panel_label` letter tags per
  FIGURE_STYLE rule 3 (no titles); the descriptive half of each ((a)
  dimensionality / (b) what drives each axis) belongs in the caption only
  -- already present in the current main.tex caption. The `r=+0.95`
  correlation values were folded into the existing in-axes arrow
  annotations ("inner f_gas^ (r=+0.95)", "outer f_gas^ (r=+0.95)") instead
  of a separate text box, both to avoid overlapping the panel-letter tag
  and because that is exactly the number the caption already quotes.

## Data / numbers verified against placeholder
- Scree: lambda_1=0.51, lambda_2=0.46, cumulative at 2 = 97% -- matches
  placeholder and main.tex caption ("two components explain 97%").
- Panel (b) top loadings, order and sign, match placeholder: VariableWindVelFac
  (positive, largest, both axes), IMFslope (e1 strongly negative, e2
  positive), WindFreeTravelDens (both negative), WindEnergyIn1e51erg (e1
  positive, e2 ~0), BlackHoleRadiative (e1 slightly positive, e2 negative),
  RadioFeedbackReior (e1 negative, e2 slightly positive), etc. -- matches
  main.tex's "loads positively on VariableWindVelFactor... IMFslope,
  WindFreeTravelDensFac" text (that text is actually describing fig06's
  forecast directions, not this figure's loadings, but the same underlying
  parameter list appears consistently).
- r_in_e1 = +0.95, r_out_e2 = +0.95, inner/outer corr = 0.24,
  R^2(e1)=0.90, R^2(e2)=0.96 -- all match the placeholder's printed values
  and the main.tex caption/prose exactly.
