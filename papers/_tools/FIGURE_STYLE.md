# Figure standards — BIND Lightcone Suite (binding for every figure agent)

Every figure in `papers/*/figs/` must be REGENERATED as a standalone script —
extracted notebook PNGs are placeholders, not deliverables.

## The non-negotiables

1. **One script per figure**: `papers/<id>/fig_scripts/figNN_<slug>.py`,
   runnable as `/mnt/home/mlee1/venvs/BIND_env/bin/python fig_scripts/figNN_<slug>.py`
   from the paper directory. Header docstring states: what the figure shows,
   the exact data files loaded, and the original source (notebook cell /
   engine) it reproduces.
2. **Style**: `from paper_style import setup, save, panel_label, COLORS, ...`
   (add `sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")`). Call
   `setup()` before any plotting. Save ONLY via `save(fig, "figs/figNN_slug")`
   → vector PDF + PNG preview.
3. **No titles.** Not `set_title`, not `suptitle`. Panel distinction =
   `panel_label(ax, "(a)")` or a short in-axes text; shared context belongs in
   the caption.
4. **Legends**: `frameon=False` (default), ≤4 words per entry, only when >1
   line needs distinguishing. Prefer direct annotation for ≤2 curves. Never
   repeat what the axis label already says.
5. **Axis labels always, with units**: e.g. `$\ell$`, `$C_\ell^{\kappa\kappa}$`,
   `$S(\ell)=C_\ell/C_\ell^{\rm DMO}$`, `$M_{200c}\,[h^{-1}M_\odot]$`,
   `$f_{\rm gas}/(\Omega_b/\Omega_m)$`. Log axes where the source used them.
6. **Semantic colors fixed suite-wide** via `COLORS`: BIND=blue, hydro
   truth=black, DMO=grey, highlight=red. Ensemble bands:
   `alpha=BAND_ALPHA`. Colormap for maps: default `cividis` unless the
   quantity demands diverging (`RdBu_r`, centered).
7. **Sizes**: `ONE_COL` (3.5 in) by default; `TWO_COL`/`TWO_COL_TALL` only for
   map galleries and ≥3-panel rows. Fonts come from the style — never
   shrink/enlarge fonts manually to fit; resize the figure instead.
8. **Maps/images**: `imshow(..., rasterized=True)` inside the vector PDF; add
   a physical scale (axis in deg/Mpc or a scale bar); one shared colorbar per
   panel row when panels share units.
9. **Data provenance**: scripts load ONLY existing cached arrays
   (`.npz`/`.npy`/`.parquet`/`.csv` already on disk — e.g.
   `examples/figures_lightcone/*.npz`, `examples/wl_latent_sbi_figs/*.npz`,
   per-notebook caches, or explicit known paths on ceph). NEVER re-run
   engines, training, MPI, or anything that *computes* science. If no cached
   data exists for a figure: keep the extracted placeholder, add
   `\todo{regenerate fig NN — no cached data; needs <engine> re-run}` in the
   tex, and record it in `fig_scripts/UNREGENERATED.md`.
10. **Truthfulness**: the regenerated figure must show the SAME data as the
    placeholder it replaces (same curves/points/ranges up to styling). If the
    cached arrays disagree with the placeholder figure, STOP and flag it in
    the report — do not silently pick one.

## Caption/tex contract

After regenerating, update `main.tex`: `\includegraphics{figs/figNN_slug.pdf}`
(PDF now, was PNG), and reconcile the caption with the final styling (line
colors/styles named in the caption must match; "blue"=BIND etc.). Keep the
`% src:` provenance comments.

## Quality gate (what the verifier checks)

Open the PNG preview at figs_preview/: fonts legible at print size, nothing
clipped, no titles, legend concise, units present, colors semantic, caption
matches. A figure failing any check goes back to its script.
