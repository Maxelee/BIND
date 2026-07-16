# notes: fig02_decomposition.png (headline figure)

## Data / computation
Uses the **current, faithful** `fid/{bind,sph,mono,dmo}/kappa_maps.npz` --
no gotcha/substitution here (unlike fig01/fig03); `sph` = radial-BCM-warp,
`mono` = pure-spherical, matching the caption exactly. Computation is a
line-for-line port of `examples/bcm_warp_comparison.py`'s `compare()`
(worktree copy under `wl-anisotropy`), using
`bind.inference.stats.power_spectrum` (Pylians flat-sky FFT) per
realization, averaged over the 8 matched-seed realizations at z_s=1.0.

## Color mapping (new -- differs from the original PNG, chosen to reuse the
suite's fixed `COLORS` consistently across fig02+fig03; see also
notes_fig03_crossauto.md)
- **BIND** -> `COLORS['bind']` (blue) -- panels (a) suppression curve and,
  by the same logic, the "BCM misses" / $f_{\rm aniso}$ curves in (b)/(c)
  (both are BIND$-$sph differences).
- **radial BCM (sph)** -> `COLORS['secondary']` (green) -- panel (a) curve,
  and the "triaxiality" / $f_{\rm tri}$ curves in (b)/(c) (both are
  sph$-$mono differences, i.e. what a faithful BCM captures that the crude
  mono control misses).
- **pure spherical (mono)** -> `COLORS['highlight']` (red, dashed where it's
  the "flagged/crude" curve) -- panel (a), and the "pure-sph error" /
  $f_{\rm mono}$ curves in (b)/(c) (BIND$-$mono differences).
- **total baryon (b$-$d) / S=1 reference** -> `COLORS['dmo']` (grey) --
  the DMO-anchored reference quantity in all three panels.
This is a deliberate re-mapping from the original placeholder (which used
plain black/`C0`/`C3`/`C2` matplotlib defaults) so that "blue always means
the BIND-vs-sph aniso signal", "green always means the sph-vs-mono
triaxiality signal", and "red always means the mono/pure-spherical control"
consistently across fig02 AND fig03. **If the caption text names specific
colors it must use this mapping** (currently `main.tex` lines 364-379 do
not name colors, so no caption edit is required).

## Panels kept (matches placeholder 1:1, no panels dropped)
- (a) suppression $S(\ell)=C_\ell/C_\ell^{\rm dmo}$: BIND (blue solid),
  radial BCM/sph (green solid), pure spherical/mono (red dashed), $S=1$
  reference (grey).
- (b) residual-field power (log-log): total baryon $b-d$ (grey), pure-sph
  error $b-m$ (red), BCM misses $b-s$ (blue), triaxiality $s-m$ (green).
- (c) decomposition fractions vs $\ell$: $f_{\rm aniso}$ (blue),
  $f_{\rm tri}$ (green), $f_{\rm mono}=f_{\rm aniso}+f_{\rm tri}$ (red
  dashed), reference lines at 0 and 1 (grey).

## Style changes from placeholder
- Dropped `suptitle` + per-panel titles; added `(a)/(b)/(c)` panel_label
  tags per FIGURE_STYLE rule 3.
- 3-panel row at `figsize=(7.2, 2.5)` (close to `TWO_COL` width, slightly
  shorter than the source's `(15,4.4)` aspect to fit the suite's compact
  font sizes).
- Legends shortened to <=3 words/symbol per entry, `frameon=False` (style
  default).
- **Update (verifier pass, legend/label collisions on the headline
  figure):** (b)'s legend was `loc="lower right"`, which put the "total
  baryon"/"pure-sph error" text directly under the rising triaxiality
  curve; moved to a 2-column legend below the panel via
  `bbox_to_anchor=(0.5, -0.28)` so no curve crosses it. (c)'s legend was
  `loc="upper left"`, the same corner as `panel_label`'s default, so the
  bold "(c)" tag sat on top of the $f_{\rm aniso}$ legend swatch; the
  legend stayed `upper left` (the curves don't reach that corner) and
  `panel_label(ax[2], "(c)", loc="lower left")` moved instead, into the
  one corner none of the three curves populate. Re-rendered and visually
  re-verified (PIL crops) no overlap in either panel.

## Numeric check
Re-running the script reproduces the placeholder's curve shapes/ranges
(suppression bottoming near S~0.90 for BIND at ell~1e4; f_aniso falling
from ~0.4 to ~0 and going negative past ell~2e4 while f_tri rises past it,
matching the placeholder) -- content-faithful, styling-only changes.
