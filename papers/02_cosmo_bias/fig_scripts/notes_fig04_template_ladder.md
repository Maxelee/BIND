# notes: fig04_template_ladder

**Panels**: unchanged — (a) residual `|dS8|/sigma_S8` vs N templates
(median solid + 95th pct dashed, one color per `ell_max`, log-y, N=2
vertical marker); (b) `sigma_S8` inflation vs N (same color/`ell_max`
coding), N=2 marker. Tagged `(a)`/`(b)` via `panel_label`; the placeholder's
two-line `ax.set_title(...)` banners ("Necessity & sufficiency: residual
bias collapses at N=2 templates" / "Cost: sigma(S8) degradation grows with
N (over-modelling = pure cost)") are dropped per FIGURE_STYLE rule 3 — this
is the headline interpretive sentence and should move into the caption
essentially verbatim, since it states the paper's central claim.

**Color mapping (identical data)**: 3 lines per panel, one per
`ell_max in {2000, 3000, 5000}`, `viridis` sampled at
`linspace(0.15, 0.85, 3)` — identical to the placeholder.

**Legend simplification**: the placeholder's panel (a) legend had 6 entries
(median + 95th pct per `ell_max`, e.g. "ell_max=2000 (median)",
"ell_max=2000 (95th)", ...). The regenerated legend keeps only the 3
`ell_max` color entries; the solid/dashed = median/95th-pct convention is
stated once via a small in-axes text annotation ("solid: median / dashed:
95th pct") instead of repeating it per color. All 6 curves (3 `ell_max` x
{median, 95th}) are still plotted in both versions — only the legend
entries were consolidated (FIGURE_STYLE rule 4: concise, ≤4-word entries).

**N=2 marker**: vertical line at `N=2` in both panels, drawn in
`COLORS["highlight"]` (was plain `color="C3"`/red in the placeholder — same
semantic role, "the value being called out").

**Table 1**: this npz (`cosmo_bias_capstone.npz`) also backs every number in
Table 1 (median/95th `|dS8|/sigma_S8` and `sigma_S8` cost ratio per
`(ell_max, N)`, plus `evr_{ellmax}` PCA explained-variance fractions quoted
in text) — unchanged by this figure regeneration, no new computation.
