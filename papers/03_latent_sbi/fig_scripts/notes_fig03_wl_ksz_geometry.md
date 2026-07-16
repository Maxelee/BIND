# notes: fig03_wl_ksz_geometry

## Panels (unchanged content from placeholder)
- (a) WL suppression latent, coloured by inner-halo $f_{\rm gas}$.
- (b) kSZ $\tau$/$y$ latent, coloured by the same quantity, same colour scale
  (shared `vmin`/`vmax` across both panels + one shared colorbar, same as
  the placeholder's single shared colorbar).

## Styling changes vs. placeholder
- Colormap **"RdYlBu_r" -> "cividis"** (FIGURE_STYLE §6: sequential
  quantity, not diverging; also now visually consistent with fig01(c)/(d),
  fig02, fig02b which all colour by the same kind of gas-fraction quantity).
- Per-panel title ("WL latent (colour = inner $f_{\rm gas}$)" / "kSZ tau/y
  latent...") replaced by a `panel_label` "(a)"/"(b)" tag plus a short
  in-axes text ("WL latent" / "kSZ $\tau/y$ latent") at lower-left — the
  "colour = inner $f_{\rm gas}$" clause moved to the shared colorbar label
  instead of being repeated per panel.
- The placeholder's `fig.suptitle` banner ("WL<->kSZ latent alignment:
  canonical corr = [...], principal angles = [...]") is **dropped from the
  figure** (FIGURE_STYLE rule 3: no suptitle; shared context belongs in the
  caption) — these numbers are already stated in the main.tex caption
  ("canonical correlations [0.98, 0.79] and principal angles [12°, 38°]")
  and are printed to stdout by the script for the caption pass to verify.

## Numbers reproduced (match the placeholder / current caption exactly)
- canonical corr = [0.98, 0.79]
- principal angles = [12°, 38°]
- n_common (shared runs between the WL Sobol sample and the kSZ profile
  sample) = 253

No caption change needed for this figure — all cited numbers reproduce
exactly.

## Data provenance
- `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` (WL side; same
  file/keys as fig01)
- `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet`
  (gas-zone labels, both WL and kSZ sides)
- `/mnt/home/mlee1/ceph/bind_science/ksz_confront/bind_tauy_xprof_snap085.npz`
  (kSZ side: radial $\tau$/$y$ profiles, mass-bin index 1 = [13.4,13.8],
  matching the WL gas-zone mass bin)

Script: `fig_scripts/fig03_wl_ksz_geometry.py`.
