# notes: fig05_bridge_hero

Regenerated from cache: `figs/fig05_bridge_hero.pdf` (was `.png`; main.tex needs
`\includegraphics{figs/fig05_bridge_hero.pdf}`).

## Content check vs. placeholder
Numbers reproduce exactly: per-run fit `slope=0.080`, `r=0.50` (panel a), same
57 runs x 42 peak-halos. Both panels keep the same data/quantities as the
placeholder — no panels dropped, no data changed.

## What changed (style only)
- Dropped the `fig.suptitle` ("BIND bridge I ...") — panel content only, per
  house style (no titles). Caption should carry that framing sentence.
- Panel (a): the two numeric callouts (slope, r) moved from the legend label
  into an in-axes text box (bottom-left) so the legend stays to ≤2 short
  entries ("per peak", "per run (mean)").
- Panel (a) colors: per-peak cloud now `COLORS["dmo"]` (grey, was `0.8`
  grey — same idea); per-run points `COLORS["bind"]` (blue, was
  `tab:blue` — same hue family); fit line `COLORS["truth"]` (near-black,
  was `k-`).
- Panel (b): colormap changed from `coolwarm` to `RdBu_r` (semantically
  equivalent diverging map, centered at 0, per FIGURE_STYLE's diverging-map
  rule) — same diagonal color pattern (blue=cooling/ejection-dominated,
  red=heating-dominated) as the placeholder.
- Panel labels are now `(a)`/`(b)` tags via `panel_label()` instead of
  in-axes titles ("WL peak dims..." / "tSZ probes..."); the descriptive
  clause of each former title should move into the caption if not already
  covered by the caption text.
- Colorbar label shortened to `$\Delta\ln Y$ (tSZ)`.

## Caption pass TODO
- If the caption currently says "top" / "left/right" panel language matching
  the old suptitle wording, it should still parse fine (panel order/content
  unchanged), but any reference to the dropped subplot titles ("WL peak dims
  \propto its gas content", "tSZ probes the gas-temperature plane; \kappa only
  the x-axis") should be folded into the caption text since they no longer
  appear on the figure itself.
