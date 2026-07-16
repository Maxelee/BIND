# notes: fig02_latent_physical

## Panels (unchanged content from placeholder)
11 panels (4x3 grid, last cell blank), same (latent-1, latent-2) plane as
fig02b, each coloured by a different measured physical quantity: $f_{\rm
gas}$, $f_\star$, $T$, $Y$, gas conc., DM conc., $\star$ conc., $P_e$, $K$,
$y$-map, $\tau$-map. Colormap unchanged (`RdBu_r`, diverging, since the
per-panel colour is a standardized (z-scored) value centred on 0 — this is
the one case in this figure set where a diverging map is correct per
FIGURE_STYLE §6).

## Styling changes vs. placeholder
- Per-panel quantity name (e.g. "$f_{\rm gas}$") moved from a real
  `ax.set_title` to an in-axes text at top-centre (style rule: no titles).
- $r_1$/$r_2$ annotation moved from upper-left to **lower-right** in-axes
  (placeholder had it upper-left, colliding with nothing there, but I found
  lower-right cleaner once the quantity name text also lives near the top);
  added a space between the mathtext subscript and the number
  (`"$r_1$ {r1:+.2f}"` vs. the original `"$r_1${r1:+.2f}"`) — purely
  cosmetic, same numbers.
- No panel_label (a)/(b)/... lettering added: each panel's own quantity name
  IS its label (11 different physical quantities), consistent with
  FIGURE_STYLE's "or a short in-axes text" allowance for panel distinction.

## Numbers reproduced (match the placeholder / current caption exactly)
$f_{\rm gas}$: r1=+0.26, r2=+0.94 | $f_\star$: r1=+0.56, r2=-0.69 | $T$:
r1=-0.33, r2=+0.05 | $Y$: r1=+0.27, r2=+0.93 | gas conc.: r1=+0.58, r2=+0.78 |
DM conc.: r1=+0.74, r2=+0.61 | $\star$ conc.: r1=+0.42, r2=+0.42 | $P_e$:
r1=+0.65, r2=+0.69 | $K$: r1=-0.41, r2=-0.89 | $y$-map: r1=-0.01, r2=+0.90 |
$\tau$-map: r1=+0.21, r2=+0.84.

All match the caption's cited values (DM concentration r1=+0.74, $P_e$
r1=+0.65, $f_{\rm gas}$ r2=+0.94, $Y$ r2=+0.93, $y$-map r2=+0.90, $\tau$-map
r2=+0.84, $K$ r1=-0.41/r2=-0.89) exactly — no caption change needed for this
figure.

## Data provenance
- `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz`
- `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/halo_atlas/run_{rid:04d}_snap096.npz`
  (253 of 5120 files, looped per valid Sobol run)

Script: `fig_scripts/fig02_latent_physical.py`.
