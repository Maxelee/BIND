# notes: fig01_latent_axes_zones

## Panels (unchanged from placeholder)
- (a) PCA scree of the WL suppression latent (bar = per-component variance
  fraction, black line = cumulative). Annotation "2 comp: 99%" in-axes
  (replaces the placeholder's `ax.set_title`).
- (b) Horizontal bar chart of the 8 largest-|loading| params on
  $\hat e_1$/$\hat e_2$. $\hat e_1$ = `COLORS["bind"]` (blue, was `C1`
  orange in the placeholder), $\hat e_2$ = `COLORS["secondary"]` (green, was
  `C4` purple). Legend entries unchanged ("$\hat e_1$"/"$\hat e_2$").
- (c) Latent plane coloured by inner $f_{\rm gas}(<R_{500c})$, colormap
  changed **RdYlBu_r -> cividis** (FIGURE_STYLE §6: sequential quantity, not
  diverging). Pearson $r$ annotated in-axes (was in a title).
- (d) Latent plane coloured by outer $f_{\rm gas}(R_{500c}\to R_{200c})$,
  colormap changed **viridis -> cividis** (same rule; keeps (c)/(d)
  colour-consistent with each other and with fig02/fig02b/fig03).

All 4 panels now carry a bold `(a)`-`(d)` `panel_label` tag (upper-left for
a/b/c, unchanged) instead of the placeholder's `ax.set_title` strings — no
figure content lost, titles just moved off the axes per house style. Figure
size is `TWO_COL_TALL` (2x2 grid; `paper_style.py` categorizes this as a
"multi-panel" case allowed to exceed `ONE_COL`).

## Numbers reproduced (match the placeholder / current caption exactly)
- 2-comp variance: 98.7%
- $r_{\rm in}$ (panel c) = +0.59, $r_{\rm out}$ (panel d) = +0.71
- Top loadings: $\hat e_1$ IMFslope=-0.389, BlackHoleRadiativeEfficiency=+0.383
  (caption: "nearly tied, |loading|~0.38-0.39" — matches); $\hat e_2$
  VariableWindVelFactor=+0.478, BlackHoleRadiativeEfficiency=-0.424 (caption:
  "~0.48 and ~0.42" — matches).
- N valid Sobol runs used = 253.

## Caption/main.tex mismatch to flag (pre-existing, not introduced by this pass)
main.tex (lines ~477-478) currently reads: *"from the $\beta$-TC-VAE
decomposition on the $N=216$ Sobol subsample."* The actual computation (both
in the source notebook `wl_latent_sbi.ipynb` cell 8 and in this regenerated
script) is a **plain linear SVD/PCA** on the **z-scored $\log_{10}S(\ell,z_s)$
stack over all N=253 valid Sobol runs** — no VAE, no 216-run subsample
anywhere in the data-provenance chain (`t__cl_kappa__valid` sums to 253).
Every quantitative number the caption cites (loadings, variance fractions)
matches this SVD/PCA computation, not a VAE. The caption's method/N
description should be corrected to "linear PCA" / "N=253" — flagging for the
caption-integration pass since edits to main.tex are out of scope here.

## Data provenance
- `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz`
- `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet`

Script: `fig_scripts/fig01_latent_axes_zones.py`.
