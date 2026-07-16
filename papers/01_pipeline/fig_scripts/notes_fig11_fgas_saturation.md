# notes_fig11_fgas_saturation

Regenerated from `examples/paper_decomp_figs.ipynb` (worktree `analysis/wl-tsz-bridge`)
cell 19, section 6.1 "Calibration to the Siegel et al. regime", reading only cached
`.npz`/`.json`.

## Panel mapping

Single panel, no panel label needed (only one panel in the figure). Bars: sorted group
f_gas per 1P run (M200c 1-2e13), colored by feedback family
(`AGN`→`COLORS["highlight"]`, `SN / wind`→`COLORS["bind"]`, `other`→`COLORS["dmo"]`,
cosmetic snap to the suite palette used identically in fig09/fig10/fig12). Reference
lines: BIND fiducial → `COLORS["truth"]` solid black; X-ray-standard band →
`COLORS["secondary"]` (green) shaded, same role as the original `tab:green`; eROSITA-low
→ `COLORS["highlight"]` red dashed (was `FAMC["AGN"]`, same color family).

Legend text shortened from "eROSITA-low (Popesso+24)" to "eROSITA-low" (≤4-word rule);
the citation is already carried by the caption (`\citep[$0.026$;][]{Popesso2024}`), so no
information is lost.

## Panels kept/dropped

Single panel, kept in full — matches the caption (`fig:fgas_saturation`) exactly.

## Numbers reproduced (cross-check against main.tex / caption)

- BIND fiducial group f_gas = **0.081** — caption/text: 0.081. Exact match.
- Minimum single-knob f_gas = **0.051** (parameter `WindFreeTravelDensFac`) — caption/text:
  0.051. Exact match.
- X-ray-standard band (0.06-0.10) and eROSITA-low (0.026) — hardcoded reference values,
  unchanged from the original script, matches caption exactly.

## Caption / main.tex — nothing needs to change

Caption already states "the minimum single-knob excursion (0.051) closes roughly half
the gap to the X-ray band but does not reach eROSITA-low" and flags this figure as
distinct provenance from the retracted "41% of cosmic" number — that flag stays valid,
this script does not touch or reproduce the 41% claim at all (per the DATA_MAP note, that
number has a separate, unverified provenance and was correctly left out of this figure's
reproduction).

## Data provenance (all pre-existing caches, no engine re-run)

- `/mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz`
  (`M_fof`, `m_gas_500c_bg`, `m_tot_500c_bg`) including `fid_snap096.npz` for the fiducial line
- `/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json`
