# notes: fig07_bridge_response

Regenerated from cache: `figs/fig07_bridge_response.pdf` (was `.png`;
main.tex needs `\includegraphics{figs/fig07_bridge_response.pdf}`).

## Content check vs. placeholder
All 7 correlation values reproduce exactly: `S(l)=0.92, R(nu)=0.92,
N_min=0.90, C_ky=0.87, V1=0.83, V2=0.82, N_pk=0.28` (same ordering, same
green/orange >0.7 threshold split). Panel (b) example scatter
(field Delta-R(nu) vs. halo Delta-ln f_gas) reproduces `r=0.92`.

## What changed (style only)
- Dropped `fig.suptitle` ("BIND bridge III ...") and both per-axes titles.
- Panel (a) bar colors: `>0.7` bars -> `COLORS["secondary"]` (green, was
  `tab:green`); `<=0.7` bar (N_pk only) -> `#e08214` (orange-brown, matches
  the extra bar-chart accent color already used in fig10's panel c, instead
  of the original `tab:orange`).
- Panel (b) family colors: same shared `FAMC` mapping as fig06/fig10
  (`AGN->highlight` red, `SN/wind->bind` blue, `other->dmo` grey).
- The `r=0.92` callout for panel (b), previously a bare `label=f"r={...}"` in
  the legend, is now the fit-line's own legend label ("fit, r=0.92") to avoid
  a free-floating text annotation colliding with the `(b)` panel tag; legend
  moved to lower-right (data-empty corner) for the same reason.
- Panel (a) value labels keep 2 decimals in-axes as in the placeholder.

## Caption pass TODO
- None expected — colors/values/ordering unchanged from the placeholder's
  described content; only the title text ("every field observable is driven
  by...", "example: R(nu) tracks...") needs to live in the caption instead of
  on-figure, if not already there.
