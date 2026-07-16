# fig08_sbc_crosschecks -- caption/content notes

## What changed vs. the placeholder
- **Panel (a) now genuinely shows the emcee overlay.** The placeholder PNG
  (extracted from the notebook run) rendered only the 3 solid NPE curves;
  main.tex's current caption explicitly flags this ("the panel contains only
  the three solid NPE curves -- no dashed emcee curves are actually
  rendered") with a `\todo`. The `emcee_cl.npz` cache (`chain`, shape
  (115200, 30)) is fully populated and well-mixed (checked: covers the full
  [0,1] range, std~0.26-0.27 for all 3 plotted params, not degenerate), so
  the regenerated panel (a) **does** show the dashed emcee marginals overlaid
  on the solid NPE marginals, as the source cell always intended.
  - **Caption action needed**: remove the sentence "the in-panel title
    anticipates ... no dashed emcee curves are actually rendered" and the
    associated `\todo`; the emcee cross-check is now visually substantiated.
    The surrounding text in section (just above Fig. `fig:sbc`) that says
    "we therefore do not cite that panel as visual evidence of NPE/emcee
    agreement" should also be revisited/removed since the overlay is now
    present and can be cited.
  - **Debugging note for future scripts**: `ax.hist(..., histtype="step",
    ls="--")` silently renders as SOLID in this matplotlib/style
    combination (matplotlib 3.10.1) -- the `ls` alias is dropped for step
    histograms. Must use the full `linestyle="--"` kwarg. Verified this is
    the actual (and only) reason the placeholder's dashed curves are
    missing -- almost certainly the same bug in the original notebook cell
    (`ax[0].hist(mc[:,i], ..., ls="--", ...)` uses the same broken alias).

## Panel/color mapping (for caption reconciliation)
- (a): 3 best-constrained params by NPE shrink = IMFslope (blue = `COLORS["bind"]`),
  WindEnergyIn1e51erg (red = `COLORS["highlight"]`), VariableWindVelFactor
  (green = `COLORS["secondary"]`). Solid = NPE, dashed = emcee (in-axes text
  annotation replaces the old title). Black dotted vline at 0.5 = prior
  centre / fiducial truth.
- (b): SBC rank histogram, blue bars (`COLORS["bind"]`), red dashed
  reference line at the uniform-count expectation (`sbc.size/20`).
- (c): per-parameter 68% coverage, green bars (`COLORS["secondary"]`), red
  dashed line at the nominal 0.68, black dotted line at the empirical mean
  (0.70) with an in-axes text label (title text moved off the axes per
  house style -- no titles allowed).

## Numbers (for caption cross-check)
- 3 best-constrained params + shrink: IMFslope 0.63, WindEnergyIn1e51erg
  0.70, VariableWindVelFactor 0.72 (matches the `pn`-derived order already
  used for fig06/fig07's corner/latent-posterior figures).
- mean 68% coverage = 0.70, matching the caption's stated number exactly.

## Panels kept
All 3 original panels kept unchanged in content; only the previously-missing
emcee dashed overlay was restored (a genuine content fix, not a
simplification) and the informal in-panel titles were moved to in-axes
annotations / removed per house style (no titles).
