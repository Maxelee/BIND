# notes_fig10_bridge_compton_y

Regenerated from `examples/fgas_bridge_explore.ipynb` / `_build_fgas_bridge_explore_nb.py`
(worktree `analysis/wl-tsz-bridge`), section 3, reading only cached `.npz`/`.json`.

## Panel mapping

- Panel labels `(a)`, `(b)`, `(c)` added (placeholder used in-axes titles "the Compton-Y
  bridge" / "Y vs f_gas (r=0.98)" / "raw vs. partial (//) corr." — dropped per the
  no-titles rule; the caption already spells out what each panel is, so no information
  is lost).
- (a) the Compton-Y bridge: scatter colored by family (`AGN`→`COLORS["highlight"]`,
  `SN / wind`→`COLORS["bind"]`, `other`→`COLORS["dmo"]`, cosmetic snap to the suite
  palette, same as fig09/fig11/fig12), black fit line → `COLORS["truth"]`.
  `panel_label` placed upper-left; legend (fit r + family swatches) moved to lower-right
  in the same call as the original notebook (no change needed, no collision).
- (b) Y vs f_gas degeneracy: original had the r=0.98 value in an axes *title*; moved to
  an in-axes text annotation at lower-right (originally would have collided with the
  panel-label tag at upper-left, so this is the one non-cosmetic layout choice — data
  and value unchanged, r=0.98 exactly as in the caption).
  scatter recolored to the same family palette as (a).
- (c) raw vs. partial correlation bars: bar colors mapped to
  `f_gas`→`COLORS["bind"]`, `logY`→`COLORS["highlight"]`, `logT`→ kept as `#e08214`
  (orange; no direct suite-palette equivalent for a third categorical class — matches
  the original hex exactly), partial-corr bars reuse `highlight`/`bind` with hatching as
  in the original. `panel_label` moved to upper-right (upper-left collided with the
  "0.91" bar-top value label of the first bar).

## Panels kept/dropped

All three panels kept — all three are explicitly described in the caption
(`fig:bridge_compton_y`, panels a/b/c) and in the body text discussion around line
505-520 of `main.tex`.

## Numbers reproduced (cross-check against main.tex / caption)

- r(f_gas, S) = **0.914** — caption/text: 0.91. Match.
- r(logY, S) = **0.935** — caption/text: 0.93. Match.
- r(logT, S) = **0.846** — caption: 0.85. Match.
- r(f_gas, logY) = **0.982** — caption panel (b): r=0.98. Match.
- partial r(Y, S | f_gas) = **0.485** — caption/text: 0.48 (main.tex explicitly notes this
  is the reader-facing, bar-label-derived value, superseding an earlier "≈0.49" WORKLOG
  rounding — our script reproduces 0.485, consistent with the 0.48 bar label to the
  displayed precision).
- partial r(f_gas, S | Y) = **-0.052** — caption/text: "≈0". Match (small negative,
  consistent with "subsumes essentially all" phrasing).

## Caption / main.tex — nothing needs to change

All panel descriptions and numbers in the existing caption match the regenerated figure
exactly to the quoted precision. No color names appear in the caption.

## Data provenance (all pre-existing caches, no engine re-run)

- `/mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz`
  (`M_fof`, `m_gas_500c_bg`, `m_tot_500c_bg`, `Y_500c`, `T_mw_500c`)
- `/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz` (per-run `{run}_clk`)
- `/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json`
- `/mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz` (fiducial S(ell))
