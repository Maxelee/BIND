# notes: fig04_ejection_profiles.png

## Data
`bind_science/ejection/{fid,truth}_snap096.npz` -- already-reduced,
cached per-mass-bin arrays (no raw slab access needed). Re-loading and
printing reproduces the notebook cell 3's stdout exactly: `counts=[2512,
416]`, `q2k_ann=[0.0431,0.0349]`, `q2y_ann=[0.1312,0.1634]`.

## Color / style mapping
- **group** ($10^{13}$-$5\times10^{13}\,M_\odot/h$) -> `COLORS['bind']`
  (blue) -- was `tab:blue` in the original, essentially unchanged.
- **cluster** ($5\times10^{13}$-$5\times10^{14}\,M_\odot/h$) ->
  `COLORS['highlight']` (red) -- was `tab:red` in the original, essentially
  unchanged.
- **line style** encodes the source, not color: solid = BIND fid, dashed
  (alpha=0.7) = TNG truth -- matches the original and matches the existing
  `main.tex` caption text (lines 456-465), which already states "solid" /
  "dashed" explicitly.
- Note: this is a deliberate departure from the strict "blue=BIND" /
  "black=hydro truth" semantic used elsewhere in the suite, because this
  figure needs a 2-way mass-bin split (group/cluster) as its primary color
  channel, with BIND-vs-truth conveyed by linestyle instead (as the caption
  already documents). If a suite-wide relegend pass is done later, this is
  the one figure where color != bind/truth.
- **Update (verifier pass):** although no caption color-word directly
  contradicted the plot, a reader carrying the blue=BIND/black=truth
  convention from Figures 2-3 could momentarily misread which line is
  truth here. Rather than recolor truth to black (which would destroy the
  mass-bin color coding for the truth curves, since both bins would become
  indistinguishable black dashed lines -- a net loss of information), added
  one clause to the `main.tex` caption stating explicitly "color encodes
  mass bin ... only linestyle distinguishes BIND from TNG truth in this
  figure." No script/data change; caption-only fix per the verifier's own
  suggested resolution.

## Panels (unchanged from placeholder, no panels dropped)
- (a) $q_2(\kappa-\kappa_{\rm dmo})=|c_2|/c_0^{\rm dmo}$ [%] vs
  $r/r_{200c}$ -- WL convergence residual anisotropy.
- (b) $q_2(y)=|c_2|/c_0$ [%] vs $r/r_{200c}$ -- tSZ pressure anisotropy.
- Shaded band = ejection annulus [0.5, 2.0] $r_{200c}$ in both panels
  (unchanged).

## Style changes from placeholder
- Dropped per-panel titles ("WL convergence residual is mildly anisotropic
  (3-5%)", "tSZ pressure is strongly anisotropic in the outskirts") and the
  `suptitle`; this qualitative content is already carried by the existing
  `main.tex` caption (lines 456-465: "mildly anisotropic (3-5%)", "grows
  strongly in the outskirts (~35-43% at r~2.9 r200c)") -- **no caption
  change needed**, added `(a)/(b)` panel_label tags instead.
- Legend (group/cluster) shown once, on panel (a) only (color meaning is
  shared across both panels, avoids redundant legends per FIGURE_STYLE
  rule 4's "only when needed" + "concise" guidance).
- `TWO_COL` figsize `(7.2, 3.1)` vs source's `(12, 4.6)`.
- Markers removed (`-o` -> `-`) for a cleaner line-only look at print size;
  all 15 radial points are still resolvable via the polyline.
