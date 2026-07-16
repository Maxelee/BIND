# notes: fig02b_latent_plane

## Panels (unchanged content from placeholder)
- (a) Scree bar + cumulative line of the $z_s=1$ suppression latent.
  Bar colour **grey ($COLORS["dmo"]$) replaces the placeholder's "0.4" grey**
  (same shade family, now the shared suite token); line/marker colour
  **`COLORS["highlight"]` (red) replaces the placeholder's `FAMC["AGN"]`
  red** — same hex-family choice, now sourced from the shared palette.
- (b) The 253 Sobol nodes on the latent plane, coloured by group-cluster
  $f_{\rm gas}$ (colormap **viridis -> cividis**, FIGURE_STYLE §6 sequential
  default), fiducial (black star, unchanged), and the 1P AGN/SN-wind spokes.

## 1P spoke colour mapping
The placeholder used ad hoc `FAMC = {"AGN": "#c1272d", "SN / wind":
"#0b53c1", "other": "#9a9a9a"}`. These are re-mapped onto the suite's fixed
semantic palette with (by luck) the same visual identity:
- AGN -> `COLORS["highlight"]` (red, `#FF2C00` vs placeholder `#c1272d`)
- SN / wind -> `COLORS["bind"]` (blue, `#0C5DA5` vs placeholder `#0b53c1`)
- other -> `COLORS["dmo"]` (grey, `#949494` vs placeholder `#9a9a9a`)
No "other"-family 1P runs appear in the legend (matches the placeholder,
which only labelled AGN/SN-wind).

## Panel labelling
Added `panel_label` "(a)"/"(b)" tags (placeholder had none — two-panel
figures still need panel distinction per FIGURE_STYLE rule 3). Panel (b)'s
tag is placed **lower-right** (not the default upper-left) to avoid
colliding with the in-axes legend ("Sobol runs"/"fiducial"/"1P: AGN"/"1P: SN
/ wind") which occupies the upper-left corner.

## Numbers reproduced (match the placeholder / current caption exactly)
- PC1 = 88.9%, PC2 = 10.3%, cumulative 2-comp = 99.2% (caption: "PC1~89%,
  PC2~11%... cumulative 99.2%" — matches).
- $r[{\rm latent1}, f_\star]=+0.57$, $r[{\rm latent2}, f_{\rm
  gas}@13.9]=+0.96$ (sign-convention check only; not quoted in the caption).
- This 99.2% is independently consistent with fig01's 98.7% headline (same
  physical latent, computed two different ways — z_s=1-only 724-ell stack
  here vs. the 5-source-plane x 724-ell stack in fig01/fig03) — the caption
  already notes this cross-check ("consistent with the 98.7% headline number
  of Figure~\ref{fig:latent-axes}a").

## Data provenance
- `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz`
- `/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json`
- `/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz`
- `/mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz`
- `/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv` (packaged repo
  asset, used only for per-param family keyword classification — not a
  cached science array)

Script: `fig_scripts/fig02b_latent_plane.py`.
