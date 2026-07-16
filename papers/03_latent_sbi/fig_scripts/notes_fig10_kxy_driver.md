# fig10_kxy_driver -- caption/content notes

## Content verification
Regenerated D values: y*tau=0.179, tau*tau=0.162, yy=0.135, kappa*y=0.098,
kappa*kappa=0.032 -- reproduces the caption's stated ranking exactly
(y*tau > tau*tau > yy > kappa*y >> kappa*kappa) and the visually-read bar
lengths in the placeholder. Panel (b) curve shapes/ordering (kappa*kappa flat
near 0 out to ell~1e4; the others rise from ell~5e2) match the placeholder.

## Panel/color mapping (for caption reconciliation)
- Panel (a): all bars green (`COLORS["secondary"]`), matching the
  placeholder's uniform `tab:green` -- this panel does not need per-probe
  color identity since probes are already labeled on the y-axis.
- Panel (b): kappa*kappa = blue (`COLORS["bind"]`), kappa*y = green
  (`COLORS["secondary"]`), yy = red (`COLORS["highlight"]`), tau*tau =
  `tab:purple`, y*tau = `tab:brown`. The last two reuse plain matplotlib
  qualitative colors (no suite-semantic role exists for a 4th/5th
  categorical probe beyond bind/secondary/highlight) -- same assignment the
  original notebook cell used (`tab:blue/green/red/purple/brown` in probe
  insertion order), just with the suite's own blue/green/red hex values
  substituted for the first three. This is a straight color-for-color swap,
  not a re-ranking.
- No titles retained (placeholder's in-panel titles "(a) which probe
  carries feedback" / "(b) cross & tSZ light up; WL auto dark" moved out
  per house style -- caption should carry that framing text, which
  main.tex's Fig. 10 caption already does).

## Provenance
Cross-paper reuse confirmed: `t__cl_kappa__value` etc. and the `stat()`
accessor are ported verbatim from `paper_ksz_field.ipynb` cell 2 (preamble)
+ cell 14 (this figure), same `emulator_dataset.npz` file already used by
the sibling kSZ paper. `zi = len(ZS)-1` (index 4, z_s=2.44, the last/highest
tomographic source-redshift bin in the 5-plane dataset) is used verbatim
from the source cell for the kappa*kappa/kappa*y tomographic selection;
yy/tt/yt have no z_s axis in this dataset and are used as-is.

## Panels kept
Both original panels kept unchanged.
