# notes: fig03_crossauto.png

## Data
Uses `fid/{bind,dmo,mono}/kappa_maps.npz`, with **`mono` substituted for
the notebook cell's `sph` read** -- same gotcha as fig01 (the `sph/`
directory was redefined to the faithful bcm_warp control after this
notebook cell was rendered; `mono` is what reproduces the placeholder's
printed band numbers exactly: cross/dP=+1.39, auto/dP=-0.39,
closure=1.000, re-verified by this script's own printed output). The
existing `main.tex` caption (lines 267-276) already states "this
measurement uses the earlier, superseded circular/monopole control" --
**no caption change needed**, matches what this script computes.

## Color mapping (new -- differs from the original PNG's black/red/blue)
Chosen to be consistent with fig02's mapping (see notes_fig02_decomposition.md):
- $\Delta P = P_{\rm bind}-P_{\rm sph}$ -> `COLORS['bind']` (blue). This is
  literally the same BIND$-$sph(mono) difference field that fig02 colors
  blue (the "BCM misses" / $f_{\rm aniso}$ term) -- was **black** in the
  original PNG.
- **cross (1st order)** -> `COLORS['highlight']` (red) -- unchanged color
  from the original (`tab:red`), and doubles as the suite's "the single
  curve being called out" semantic, appropriate since the cross term
  dominating is the paper's headline mechanism.
- **auto (2nd order)** -> `COLORS['secondary']` (green) -- was **blue**
  (`tab:blue`) in the original PNG; swapped to green so blue stays reserved
  for "BIND-vs-sph" and matches fig02's use of green for the
  sph-related/subdominant term.
**If the caption or body text names specific line colors for this figure,
it must use: blue=$\Delta P$, red=cross, green=auto** (checked
`main.tex` lines 266-278: no colors are currently named, so no caption
edit is required).

## Panels (unchanged from placeholder, no panels dropped)
- (a) raw $\Delta C_\ell$ contributions: $\Delta P$, cross, auto vs $\ell$
  (linear y, log x), reference line at 0.
- (b) fraction of $\Delta P$: cross$/\Delta P$, auto$/\Delta P$ vs $\ell$
  (log x), reference lines at 0 and 1.

## Style changes from placeholder
- Dropped `suptitle`; added `(a)/(b)` panel_label tags.
- Legend moved to `lower right` in panel (b) (the original notebook's
  `upper left` collided with the `panel_label` tag under the new style --
  panel (a)'s `lower right` legend placement was kept as-is, no collision).
- `TWO_COL` figsize `(7.2, 3.1)` vs source's `(13, 4.8)`.

## Numeric check
Script prints `band ell in [3000,20000]: cross/dP = +1.39 | auto/dP = -0.39
| closure (cross+auto)/dP = 1.000` on every run -- exact match to the
notebook cell 24's original stdout.
