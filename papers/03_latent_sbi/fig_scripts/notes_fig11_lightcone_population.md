# fig11_lightcone_population -- caption/content notes

## Content verification
All numbers reproduce the caption exactly:
- z=0.034 shell carries 2.9% (~"3%") of the total z_s=1 kernel weight;
  kernel W(z) peaks near z=0.42 (caption says z~0.42).
- Single-epoch proxy fidelity: amplitude r=0.98 (caption ">0.95"),
  redistribution r=0.47 (caption "~0.47").
- Per-halo Born Delta-kappa vs. f_gas(<r500c): r=+0.62 (caption "+0.62").

## Panel/color mapping (for caption reconciliation)
- Panel (a): grey bars (`COLORS["dmo"]`) = kernel weight W(z)*M; red line+
  markers (`COLORS["highlight"]`) = measured per-halo |Delta-kappa|; black
  dotted vline = the old z=0.034 proxy epoch. The original notebook used a
  local 2-color "feedback family" scheme (`FAMC`, AGN-red/SN-blue) for the
  vline/marker colors -- replaced here with the suite's grey/highlight
  roles, which map naturally (grey=baseline weighting, highlight=the
  featured measured signal).
- Panel (b): red bars (`COLORS["highlight"]`) = "amplitude" features (f_gas,
  f_star, Y/M, T_mw, K_mw), green bars (`COLORS["secondary"]`) =
  "redistribution" features (gas concentration/ejection). Same 2-way split
  as the source notebook, colors substituted for the suite's highlight/
  secondary roles (was AGN-red/SN-blue there too).
- Panel (c): `cividis` hexbin (bins='log', mincnt=1, rasterized=True per
  house style for imshow/hexbin-like density plots), black in-axes
  correlation-coefficient annotation.
- Repositioned the "old §6 epoch" annotation text in panel (a) from
  overlapping the legend (its placeholder position, upper-left near the
  legend, collided with "kernel weight W(z)M") to the empty lower-right
  region (z>0.8, low y-values) -- purely a layout fix, same text/number.
- Panel (c)'s r-value/panel-label text color changed from the source
  notebook's white (which assumed a dense/dark hexbin in the top-left
  corner) to black, since with `mincnt=1` the top-left corner
  (f_gas~-0.2, high Delta-kappa) is empty/white in this rendering -- white
  text there was invisible. This is a legibility fix only, not a data
  change.

## Panels kept
All 3 original panels kept unchanged in content (single np.load, no
recomputation, per DATA_MAP.md).
