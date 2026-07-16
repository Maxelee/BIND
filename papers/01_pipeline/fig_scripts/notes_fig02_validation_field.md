# notes: fig02_validation_field

**Output**: `figs/fig02_validation_field.pdf` (+ `figs_preview/fig02_validation_field.png`)
**Script**: `fig_scripts/fig02_validation_field.py`
**Data**: `runs/{bind,truth}/run_0000/{Cl_kappa,peak_counts,nongaussian_stats}.npz`,
z_s = 1.0 (ZIDX=1 of the 5 lux source planes), unchanged from placeholder.

## Panel mapping (unchanged data/curves from placeholder)

Top row = statistic, bottom row = residual, 6 columns:

- (a) $C_\ell^{\kappa\kappa}$ vs $\ell$ — log-log, $\ell \in [90, 10^4]$.
- (b) $N_{\rm peak}(\nu)$ — linear-$x$/log-$y$, $\nu \in [-1, 6]$.
- (c) $N_{\rm min}(\nu)$ — linear-$x$/log-$y$, $\nu \in [-6, 1]$.
- (d) $V_0(\nu)$ — linear/linear, $\nu \in [-3, 3]$.
- (e) $V_1(\nu)$ — linear/linear, $\nu \in [-3, 3]$.
- (f) $V_2(\nu)$ — linear/linear, $\nu \in [-3, 3]$.

**Colors**: BIND = solid `COLORS["bind"]` (suite blue, `#0C5DA5`), TNG300 hydro
= dashed `COLORS["truth"]` (near-black, `#111111`). The original notebook
cell used a feedback-family blue (`#0b53c1`, visually indistinguishable from
suite blue) and `0.25` grey for hydro — no visible color change, just now
pinned to the suite-wide semantic palette.

**Residuals**: columns (a)-(c) are `100*(B/H-1)` (ratio residual, since these
statistics are strictly positive); columns (d)-(f) are
`100*(B-H)/max|H|` (normalized-amplitude residual, since the Minkowski
functionals V0/V1/V2 cross zero and a ratio residual is undefined there).
Grey band = +/-5%.

## Caption-relevant changes vs. placeholder

- Added `panel_label` tags (a)-(f) to each top-row panel (not present in the
  placeholder, which distinguished panels only by y-axis label). If the
  caption enumerates panels, it can now reference (a)-(f) directly.
- Panel-label corner placement was chosen per-panel to avoid colliding with
  the curve or the top-of-axis tick label: (a) and (b) and (d) use
  "upper right" (their curves/plateaus are high on the left edge); (c),
  (e), (f) use the default "upper left" (empty there). This is a purely
  cosmetic placement choice — no data or curve changed.
- No other content changes; same 6 quantities, same z_s=1 plane, same
  residual definitions as the source notebook cell.
