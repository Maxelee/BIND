# notes: fig04_multistat_align

**Script**: `fig_scripts/fig04_multistat_align.py` -> `figs/fig04_multistat_align.pdf`

## Content mapping (for the caption pass)
- Single panel, grouped bar chart. x-axis = 8 WL/tSZ summary statistics
  (peaks, minima, PDF, Minkowski V0/V1/V2, C_l^{kappa y}, C_l^{yy}).
- Blue bars ("latent 1") = alignment (|canonical correlation|) of that
  statistic's own top-2 PCA latent's FIRST component to the kappa-C_l 2-D
  latent's first component. Semantic color = `COLORS["bind"]` (was
  matplotlib "C0" in the original notebook).
- Red/orange bars ("latent 2") = same for the second CCA component.
  Semantic color = `COLORS["highlight"]` (was "C1"/orange in the original).
- Dotted horizontal line at y=1.0 = perfect alignment reference (unchanged
  from placeholder).
- Legend moved from inside-plot ("lower left", overlapping bars in the
  original placeholder) to a frameless 2-column legend above the axes
  (`bbox_to_anchor=(0,1)`) for legibility -- no content change.
- No title/suptitle (placeholder had a suptitle "the 1st WL latent is
  universal; the 2nd rotates for some probes" -- dropped per house style;
  this sentence belongs in the caption, and IS already the caption's
  message in `main.tex` \label{fig:multistat}).

## Numbers reproduced (printed by the script; match main.tex sec:res-latent)
```
peak_counts      1.00  0.76      (main.tex: 0.99, 0.69-0.76 band)
minima_counts    0.99  0.73
pdf              1.00  0.69
mf_v0            0.98  0.20      (main.tex: Minkowski V0 0.98, 0.20)
mf_v1            1.00  0.43      (main.tex: V1 1.00, 0.20-0.43 band)
mf_v2            1.00  0.41      (main.tex: V2 1.00)
cl_kappa_y       0.98  0.91      (main.tex: C_l^{kappa y} 0.97, 0.90)
cl_yy            0.93  0.89      (main.tex: C_l^{yy} 0.93, 0.89)
```
All values match `main.tex`'s prose numbers to within the last reported
digit (main.tex text was hand-transcribed from the same notebook cell's
printed/plotted output, e.g. "0.99" vs the script's "1.00" for peak_counts
first-axis alignment -- a rounding artifact, not a discrepancy in the
underlying computation, since both come from the identical cached
`emulator_dataset.npz` and identical PCA+CCA math).

## No caption changes needed
Colors are now the suite-semantic blue/red instead of matplotlib C0/C1, but
the caption in `main.tex` (\label{fig:multistat}) does not name colors
explicitly, so no caption edit is required. The dropped suptitle text is
already captured by the caption's "Takeaway" sentence.

## Data provenance
`/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` only -- `t__cl_kappa__value/valid`,
`t__{peak_counts,minima_counts,pdf,mf_v0,mf_v1,mf_v2,cl_kappa_y,cl_yy}__value/valid`,
`a__cl_kappa_y__ell`, `a__cl_yy__ell`. `load_stat`/`reduce_stat` ported verbatim
from `sobol-sb35/examples/wl_stat_latents.py` (PCA via sklearn, closed-form).
CCA via `sklearn.cross_decomposition.CCA` (closed-form, not trained/fit in
the ML sense -- a linear canonical-correlation solve).
