# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

---

## 2026-05-27 — `feature/thermo`: emulate the 4 gas thermo fields

New branch `feature/thermo`. The training files (no-lowmass) now carry four
extra gas-derived maps — `compton_y`, `temperature`, `entropy`, `pressure`
(added by `make_train_data/add_gas_thermo_maps.py`). Extended the emulator to
generate them jointly with the mass fields via a single flow-matching model
(extended output channels), gated behind `--predict_thermo`. Default
3-channel and `--stars_two_head` paths are untouched.

- **Normalization** (`data.py`): thermo fields are strictly positive with ~4–9
  dex of range, so `log10(1+x)` collapses sub-unity fields (compton_y/pressure)
  to ~0. Use per-channel `log10(max(x, floor))` standardization instead
  (`thermo_forward`/`thermo_inverse`; floor = 0.1th pct, zero-safe). New
  back-compatible `NormStats` fields (`predict_thermo`, `thermo_mean/std/floor`);
  `AstroDataset` appends `N_THERMO` channels after the mass target →
  `[DM, Gas, Stars(/occ,dens), Y, T, S, P]`.
- **model.py**: decoupled the stars-loss weighting from `out_ch==4` (now keys
  off an explicit `stars_two_head` flag) so appended thermo channels (weight 1)
  don't break two-head stars. UNet/FM are otherwise channel-agnostic.
- **train.py**: `--predict_thermo` (requires `fm` + large-scale path; cube data
  has no thermo). `out_ch = (4 if two_head else 3) + 4`.
- **Evaluation** (`test_suite/`): inference denormalizes + returns thermo in the
  trailing channels; thermo is evaluated **per-halo, not composited** (compton_y
  is extensive; T/P/S are intensive mass-weighted means — mass conservation
  N/A). Ported the exact gas-thermo recipe into `pipeline.py`
  (`project_thermo_fullbox`, axis-aligned to match the suite cutout frame) to
  reconstruct per-halo truth thermo patches; `runner.py` reports per-channel
  log10 bias/scatter (`thermo_metrics`).
- **Data gap**: ~0.1% of no-lowmass files lack thermo maps (the thermo job
  skipped some sims). `--predict_thermo` skips them — `compute_norm_stats`
  ignores them and `AstroDataset` resamples a random index on a miss (misses
  cluster by sim). Without this the first `fm_thermo` job died instantly on a
  `KeyError: compton_y`.
- **Validated**: norm-stats save/load + back-compat; dataset emits 7 ch;
  forward/inverse round-trip <1e-4; FM loss/grad for single- and two-head+thermo;
  truth-thermo port reproduces stored `sim_0_halo_0_rot_0` maps to float32
  round-off (median rel err ~1e-6).

Launch: `python train.py --predict_thermo --run_name fm_thermo` (add
`--stars_two_head` to keep the stellar split).

---

## 2026-05-27 — Repo hygiene, branch reorganization, and agent instructions

**Repo cleanup.** The repo had no `.gitignore`, so ~304 untracked items
(2 GB of caches/outputs/figures, committed `.pyc`) were noise. Added a
`.gitignore` (caches, `outputs/`, figures, `*.npz`/`*.npy`/`*.log`, pycache,
notebook checkpoints, machine-local `.claude/settings.local.json`), untracked
the committed `.pyc` files, and refreshed the tracked paper figures. Untracked
count: 304 → 0.

**Branch reorganization.** Decision: keep `main` a clean trunk and park distinct
analyses on topic branches instead of dumping everything on `main`.
- `main` — core engine (`data/model/train/metrics`, `test_suite/`) + the
  ~890-line engine evolution since the last working-model commit + refreshed
  `paper_figures.ipynb`.
- `feature/3d-cube` — 3D / cube-projection extension.
- `analysis/2d` — scatter package, observables, `project1-7`, CV derivatives.
- `wip` — scratch notebooks, parameter-injection experiments, planning notes.

Notebooks are committed with outputs (per preference). No git remote — local-only.

**Agent instructions.** Added `CLAUDE.md` (architecture + commands + conventions
+ data caveats), this `docs/WORKLOG.md`, and `.github/copilot-instructions.md`
mirroring the project context for GitHub Copilot. Then merged `main` into each
topic branch so they all carry the shared docs, and appended a tailored
`## This branch: …` section to `CLAUDE.md` + the Copilot file on each
(`feature/3d-cube`, `analysis/2d`, `wip`) describing that branch's projects.
`main`'s copy stays generic.
