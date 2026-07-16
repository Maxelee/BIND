# notes: fig06_bridge_fgas

Regenerated from cache: `figs/fig06_bridge_fgas.pdf` (was `.png`; main.tex
needs `\includegraphics{figs/fig06_bridge_fgas.pdf}`).

## Content check vs. placeholder
Numbers reproduce exactly: fiducial group `f_gas=0.086`, `S(l~1500)=0.992`,
van Daalen-style bridge fit `r=0.91`, family split `AGN(20)/SN-wind(25)/
other(12)` — same 57-run 1P suite, same `_fgas_run`/`_family` definitions
ported verbatim from `examples/bind_bridge.py`.

## What changed (style only)
- Dropped `fig.suptitle` ("BIND bridge II ..."), replaced per-axes titles
  with `(a)`/`(b)` panel tags.
- Panel (a) line colors remapped to the fixed suite palette instead of the
  original tab10 colors: fiducial (baseline BIND curve) -> `COLORS["bind"]`
  (blue, was black `k-`); lowest-f_gas run -> `COLORS["highlight"]` (red,
  same role/hue as original `tab:red`); highest-f_gas run ->
  `COLORS["secondary"]` (green, was `tab:blue`). **Caption note**: if the
  caption names "black = fiducial" it must be updated to "blue = fiducial";
  the lowest/highest-f_gas color roles (red/green) also changed from
  red/blue to red/green.
- Panel (a) legend gained a title ("group $f_{\rm gas}$") holding the units
  that used to be spelled out per-entry, to keep entries short.
- Panel (b) family colors use the same `FAMC` mapping as fig10/fig07
  (`AGN->highlight` red, `SN/wind->bind` blue, `other->dmo` grey) instead of
  the original bespoke `tab:red`/`tab:blue`/`0.6` — same hues, now shared
  suite-wide with fig07/fig10 for cross-figure consistency. Fiducial star
  marker kept as a non-palette gold accent (`#f2c14e`), matching the
  placeholder's gold star.
- Fit line in panel (b) is `COLORS["truth"]` dashed (was plain black dashed
  `k--`) — same appearance.
- x-axis label now states explicit units (`$h^{-1}M_\odot$`) per house
  style; the placeholder's axis label omitted units.

## Caption pass TODO
- Update any "black solid = fiducial" line in the caption to "blue solid".
- Confirm "highest f_gas" color reference (blue -> green) if the caption
  calls out colors by name.
