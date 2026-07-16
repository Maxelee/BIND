# notes_fig12_bridge_mass

Regenerated from `examples/fgas_bridge_explore.ipynb` / `_build_fgas_bridge_explore_nb.py`
(worktree `analysis/wl-tsz-bridge`), section 2, reading only cached `.npz`/`.json`.

## Panel mapping

- Panel labels `(a)`, `(b)`, `(c)` added (placeholder used in-axes titles "bridge per
  mass bin" / "tightness vs. lever arm" / "correlation vs. ell" — dropped per the
  no-titles rule; caption already spells out each panel).
- Mass-bin colors (`group`=`#1b7837` green, `intermediate`=`#762a83` purple,
  `cluster`=`#b35806` orange) are **kept exactly as in the original** — these are a third,
  genuinely different categorical dimension (halo mass bin, not BIND/truth/DMO/highlight),
  so there is no natural mapping onto the 5 fixed suite colors; introducing 3 *new* hex
  values here (rather than reusing `COLORS`) is consistent with "Semantic colors fixed
  suite-wide" governing the BIND/truth/DMO/highlight axis specifically, not every possible
  categorical grouping in every figure.
- (a) legend moved from `loc="upper left"` to `loc="lower right"` (it collided with the
  new `panel_label` tag at upper-left; the lower-right region of the scatter is
  comparatively sparse). Content/values unchanged.
- (b) twin-axis Pearson-r / slope-alpha plot: unchanged styling (already legend-based, no
  collision with panel label at upper-left).
- (c) r[f_gas,S] vs ell per mass bin: unchanged styling.

## Panels kept/dropped

All three panels kept — all three are explicitly described in the caption
(`fig:bridge_mass`, panels a/b/c) and the body text around line 553-563 of `main.tex`.

## Numbers reproduced (cross-check against main.tex / caption)

- group: r=**0.913**, alpha=**0.587** — caption/text: r=0.91, alpha=0.59. Match.
- intermediate: r=**0.841**, alpha=**0.857** — caption/text: r=0.84, alpha=0.86. Match.
- cluster: r=**0.593**, alpha=**1.119** — caption/text: r=0.59, alpha=1.12. Match.

## Caption / main.tex — nothing needs to change

All three (mass bin, r, alpha) triples match the existing caption/body-text numbers to
the quoted 2-decimal precision. No color names appear in the caption.

## Data provenance (all pre-existing caches, no engine re-run)

- `/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz` (per-run `{run}_clk`)
- `/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json`
- `/mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz`
  (`M_fof`, `m_gas_500c_bg`, `m_tot_500c_bg`)
- `/mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz` (fiducial S(ell))
