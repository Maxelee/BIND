# notes: fig13_shmr_scatter_budget.png (deliverable is now .pdf; see below)

## Content mapping (for the caption pass)

- Single panel, `ONE_COL` size (was a slightly larger ad hoc 5.2x4in figure
  in the notebook; content unchanged, just resized/restyled to the suite
  standard). No panel label needed (single panel).
- Two bars, same quantities as the placeholder, same values:
  - "feedback (Sobol prior)": 0.149 dex$^2$ (97.2% of total variance) — now
    `COLORS["highlight"]` (red, `#FF2C00`) — was `tab:orange` in the
    notebook. Chosen because this is the figure's single called-out
    finding (feedback dominates the SHMR scatter budget).
  - "halo-to-halo": 0.0042 dex$^2$ (2.8%) — now `COLORS["secondary"]`
    (green, `#00B945`) — was `tab:blue` in the notebook.
  - Dashed reference line (twobound one-at-a-time marginal spread, 0.026
    dex$^2$): now `COLORS["truth"]` (black) — was `tab:red` in the
    notebook. Recolored to black because it is an independent/reference
    benchmark value (a different experiment's marginal spread, analogous in
    role to a "ground truth" comparator), and red was reassigned to the
    "feedback" bar above to avoid two unrelated things sharing the
    highlight color.
- **Dropped**: the placeholder's title `"SHMR scatter budget (SB35
  Sobol)"`. FIGURE_STYLE.md forbids titles — this should become (or be
  folded into) the caption's opening clause instead, e.g. *"SHMR
  residual-variance budget on the SB35 Sobol suite: ..."*
- Legend: single entry ("twobound one-at-a-time (marginal)") for the dashed
  line only, matching the placeholder's one-line legend (frameless, per
  style).
- Y-axis label unchanged: "variance of $\log_{10}M_\star$ at fixed $M$
  [dex$^2$]".

## Data provenance recap

- `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz`
  only — single file, index-aligned Sobol (256 runs) + twobound (60 runs)
  stellar masses on the same 2933 snap-096 halos.
- Computation: quadratic `np.polyfit(log10 M200c, Sobol-mean log10 M_star)`
  mass-trend removal -> residual `r` -> balanced variance decomposition
  (`r.var(axis=0).mean()` = feedback, `r.mean(axis=0).var()` = halo-to-halo)
  -- closed-form, no accretion-history file needed for this figure (that
  dependency only gates the notebook's later S6-S8 sections).

## Verification

Printed values from the regenerated script match the placeholder to the
digits shown in its bar heights:
  sigma_tot = 0.3915 dex
  feedback  = 0.3860 dex (0.1490 dex^2, 97.2%)   [placeholder: ~0.149]
  halo-to-halo = 0.0651 dex (0.0042 dex^2, 2.8%) [placeholder: ~0.005]
  twobound one-at-a-time (marginal) = 0.1607 dex (0.0258 dex^2) [placeholder: ~0.026]

## Naming note

The assigned figure name has a `.png` extension (matching the placeholder),
but per FIGURE_STYLE.md's save() contract every regenerated figure is saved
as vector PDF + a gitignored PNG preview. The script writes
`figs/fig13_shmr_scatter_budget.pdf` (+ `figs_preview/...png`); the tex
integration pass should point `\includegraphics` at the `.pdf`, same as the
other regenerated figures ("PDF now, was PNG" per the style doc's
caption/tex contract).
