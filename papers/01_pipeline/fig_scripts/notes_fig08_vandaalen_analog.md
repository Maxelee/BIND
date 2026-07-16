# notes: fig08_vandaalen_analog

Regenerated `figs/fig08_vandaalen_analog.pdf` in place (placeholder was
already `.pdf`; no extension change needed in main.tex).

## Content check vs. placeholder
Reproduces exactly: 57/57 1P runs carry WL statistics; TNG-CAMELS
`f_bar-tilde` range `0.812-0.980` (median 0.918); panel (b) fit
`r=0.96 (l=1000), 0.94 (l=3000), 0.87 (l=6000)` — matches the placeholder's
legend values verbatim. van Daalen fit hardcoded exactly as in the source:
`vd(f) = -exp(-5.990*f - 0.5107)`.

## Provenance subtlety (documented, not resolved)
This figure's source (`_build_fgas_bridge_explore_nb.py` section 5) uses its
own, simpler `_family(name)` classifier (name only, no param Description) —
**different** from the `_family(name, desc)` classifier in `bind_bridge.py`
used for fig06/fig07. Both exist in the source tree; DATA_MAP.md already
flags this exact discrepancy for fig13. I ported the build-script's own
classifier here (not bind_bridge.py's) to match this figure's original
source exactly; family assignment is only used for the panel (a) rug tick
colors here (not for a printed count), so the discrepancy is cosmetic for
this figure specifically.

## What changed (style only)
- Panel (a): "van Daalen+20" line color -> `COLORS["truth"]` (near-black
  dashed, was `k--`, same); shaded ±1% band -> `COLORS["dmo"]` grey (was
  `0.6` grey, same); "obs. group" span -> `COLORS["secondary"]` green (was
  `tab:green`, same hue family); "TNG-CAMELS 1P range" span ->
  `COLORS["bind"]` blue (was `tab:blue`, same). Rug tick colors use the
  shared `FAMC` (AGN=highlight red, SN/wind=bind blue, other=dmo grey)
  instead of the build script's bespoke hex triple (`#c1272d`/`#0b53c1`/
  `#9a9a9a`) — visually near-identical, now on the suite palette.
  Legend entries shortened ("van Daalen+20", "obs. group", "TNG-CAMELS") —
  dropped the tilde-f_bar subscript text since the axis label already states
  it.
- Panel (b): the 3 ell-curves (1000/3000/6000) recolored from the original
  ad hoc `#4575b4/#fc8d59/#b30000` triple to 3 samples of the suite's default
  `cividis` colormap (low/mid/high = navy/olive/yellow) — chosen because
  these 3 curves are a *continuous progression* in ell (not a BIND/truth/DMO
  role), so a perceptual sequential colormap sample was used rather than
  forcing them into the 4 fixed semantic roles; this is the one deliberate
  departure from a literal color port in this batch. **Caption note**: if
  the caption names these lines by color ("blue"/"orange"/"red" for
  l=1000/3000/6000), it needs updating to the new navy/olive/yellow triple,
  or better, refer to them by ell value only (the legend already spells out
  ell and r per line, so color-naming in the caption is avoidable).
- Dropped the printed `plt.show()`-adjacent second console line
  ("WL relation tightness...") — not needed for the figure itself, values
  already on the legend.

## Caption pass TODO
- If the caption calls out panel (b) curve colors by name, update per the
  cividis note above.
