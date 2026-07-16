# notes_fig09_cl_envelope

Regenerated from `examples/paper_lightcone_figs2.ipynb` (worktree `analysis/sobol-sb35`)
cell 12, reading only cached `.npz`/`.json` (no engine/notebook re-run).

## Panel mapping (unchanged from placeholder content, restyled)

- Panel labels added: `(a)` left, `(b)` right (placeholder had none — panel identity was
  implicit from position only; caption text can now say "panel (a)"/"panel (b)" if useful,
  though it currently already says "Left"/"Right" which is still accurate).
- Left panel — S(ell) envelope:
  - Tan/khaki Sobol 5-95% band (original `#d8d2c4`) → recolored to suite `COLORS["dmo"]`
    grey at alpha 0.30 (same semantic role: background prior envelope, not a specific
    line). This is the one deliberate color substitution — content/shape unchanged.
  - Black `fiducial` line → `COLORS["truth"]` (`#111111`), same as before.
  - 1P per-run curves: family color, `AGN` → `COLORS["highlight"]` (`#FF2C00`, was
    `#c1272d` — nearly identical red), `SN / wind` → `COLORS["bind"]` (`#0C5DA5`, was
    `#0b53c1` — nearly identical blue), `other` → `COLORS["dmo"]` (`#949494`, was
    `#9a9a9a`). Purely cosmetic snap to the fixed suite palette.
  - LSST-Y10 shaded band and Euclid dotted lines: unchanged (grey/dotted, matches
    original).
- Right panel — r[f_gas, S(ell)] vs ell:
  - Grey dashed = 1P (n=57) → `COLORS["dmo"]`, was `0.6` grey (same).
  - Black solid = Sobol (n=253) → `COLORS["truth"]`, unchanged.
  - Peak-r marker dot recolored `COLORS["highlight"]` (was `FAMC["AGN"]`, same net color).
  - Inset scatter: Sobol points → `COLORS["secondary"]` (`#00B945` green, was `#2a6f6f`
    teal — cosmetic only); 1P triangles → `COLORS["highlight"]` (was `FAMC["AGN"]`, same).
  - Inset fit line → `COLORS["truth"]` black dashed (unchanged).

## Panels kept/dropped

All content kept — no panels dropped. Both panels (envelope + r-vs-ell with inset) are
directly referenced in the caption (`fig:cl_envelope`) and in the body text discussion of
"1P corners to the full Sobol cloud" (Section around line 634-644 of `main.tex`).

## Numbers reproduced (cross-check against main.tex / caption)

- 1P peak r in the 1000-2000 band: **0.917** (script printout) — caption says
  "$r\approx0.86$-$0.91$", consistent (1P peak is the top of that range).
- Sobol peak r: **0.856** at **ell=1037** — caption inset text says "ell=1037, r=0.86";
  matches to the quoted precision.
- n=57 (1P), n=253 (Sobol) — matches caption exactly.

## Caption / main.tex — nothing needs to change

The existing caption text ("Left:... Right:... inset shows the Sobol scatter at
ell=1037 (r=0.86)") already matches the regenerated figure's content, numbers, and
panel order. No color names are mentioned in the caption, so the palette substitution
above requires no caption edit.

## Data provenance (all pre-existing caches, no engine re-run)

- `/mnt/home/mlee1/ceph/bind_science/dashboard_cache/{g1_design.json,g1_stats.npz}`
- `/mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz`
- `/mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz` (1P atlas, incl. `fid_snap096.npz`)
- `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` (Sobol S(ell) + design; confirmed
  same ell grid as the 1P cache, bit-for-bit)
- `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/halo_atlas/run_%04d_snap096.npz` (Sobol atlas)
