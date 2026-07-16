# notes: fig01_transfer_ensemble

**Panels**: unchanged from placeholder — 5 panels, one per source-plane
redshift `z_s in {0.50, 1.00, 1.50, 2.00, 2.44}` (left to right, same order).

**Content mapping (identical data, restyled)**:
- Thin colored lines: one per SB35 Sobol run (99 total), color = group-scale
  (1e13-1e14.5 Msun) mass-weighted gas fraction `f_gas`, colormap `viridis`
  (was also viridis in the placeholder — unchanged).
- Solid near-black line: ensemble median `S(ell)` per panel (`paper_style`
  `COLORS["truth"]`, #111111 — was pure black `k-` in the placeholder;
  visually indistinguishable).
- Dashed near-black lines: 2.5/97.5 percentile envelope (was black dashed
  `k--`, same convention).
- Dotted grey horizontal line at `S=1`: no-baryon reference (unchanged).
- Colorbar: group `f_gas` range, same normalization (2nd-98th percentile
  clip) as the placeholder.

**Dropped**: the `fig.suptitle(...)` banner ("Baryonic WL transfer
functions — 99 SB35 Sobol realizations, fixed DMO skeleton") — per
FIGURE_STYLE rule 3 (no titles/suptitles); this sentence should live in the
caption instead. Per-panel `ax.set_title(f"z_s=...")` replaced by an
in-axes `panel_label`-style tag (`z_s=0.50` etc.) in the upper-left corner
of each panel (same information, no title).

**Caption must state**: 99 realizations; x-axis capped at
`ell_max_trust = 1.5e4` (above this, CIC aliasing dominates run-to-run
scatter — this cap was already applied in the placeholder, not new).

**Style**: `TWO_COL`-width figure (5 panels + colorbar), `rasterized=True`
on the ~495 thin per-run lines to keep the vector PDF size reasonable
(purely a rendering choice, no data change).
