# notes: fig04_response_fan

**Panels kept**: all 4 (2x2: tau/y columns x profile/spread rows),
unchanged from placeholder.

**Content mapping** (for caption pass):
- Top row: RdYlBu_r `LineCollection`, one curve per Sobol node (256),
  colored by that node's f_gas,500/(Omega_b/Omega_m) (red=high/weak
  feedback, blue=low/strong feedback) — colormap and normalization
  (5th-95th percentile of node f_gas) unchanged from the notebook. Heavy
  black line = fiducial TNG reference (`COLORS["truth"]` is equivalent
  black; used plain "k" to match the LineCollection legend proxy exactly
  as in the source). Grey band = resolution-clean window 0.3<x<1.5.
- Bottom row: purple line = fractional node spread (p84-p16)/2/fiducial;
  vertical dotted grey line at x=1 (r200c).
- Shared colorbar on the right: f_gas,500 (node), RdYlBu_r.
- Left column = tau(R) [kSZ, electron column]; right column = y(R) [tSZ,
  pressure] — same BGS mass bin (logM200 in [13.4,13.8]), same snapshot
  85 (z=0.18).

**Style changes from placeholder**: dropped `fig.suptitle` ("256-node
Sobol feedback sweep · BGS bin logM200∈[13.4,13.8] · z=0.18") per the
no-titles rule — **this context must move into the caption text**, e.g.
"...256-node Sobol feedback sweep, BGS-mass bin (logM200∈[13.4,13.8]),
z=0.18...". No other panel titles existed. The small in-axes annotation
"feedback fans the core" (top-left panel) is present in the script
(ported verbatim from the source, repositioned to `xytext=(0.40,0.14)`
to avoid the legend) but **does not actually render** in either the
original placeholder or the regenerated figure — matplotlib's default
`annotation_clip` drops it because its `xy` anchor `(0.18, 0.5)` sits far
outside the data axis limits (tau/y range from ~1e-10 to ~1e-3); this is
a latent bug in the original notebook cell, reproduced faithfully here
(verified against `figs/fig04_response_fan.png`, which also lacks the
annotation). Not treated as a content change.

**No data changes.** Same 2 npz caches + same parquet columns/z-filter
for the per-node f_gas color axis as the notebook.
