# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

BIND2 is a conditional **flow-matching emulator that paints baryonic fields onto dark-matter-only (DMO) maps** for the CAMELS IllustrisTNG suite. Given a DMO density projection and a 35-dim cosmological+astrophysical parameter vector, it generates the corresponding hydro fields — `[DM_hydro, Gas, Stars]` — as 128×128 maps. With `--predict_thermo` it additionally emits 4 gas-thermodynamic fields (`compton_y, T, entropy, P_e`). The downstream science (in the topic branches, see below) uses the trained emulator to study how baryonic physics responds to feedback parameters.

## Environment & commands

The package is installed editable into a Python venv (Python ≥3.10, PyTorch ≥2.0); it imports as `bind` from the `src/bind/` layout. The *physics evaluation pipeline* lives in `bind.inference` (driven by the `bind-camels-suite` CLI over CAMELS suites); "test suite" here means that pipeline, not unit tests. CI runs `ruff check src` plus an import smoke test.

```bash
pip install -e .            # imports as `bind`
```

**Train** (multi-GPU via Lightning DDP, bf16, EMA):
```bash
python -m bind.train --data_root /path/to/train_data_rotated2_128_cpu \
    --run_name fm_two_head --stars_two_head --interpolant fm --max_epochs 200
# Or, equivalently, set $BIND_DATA_ROOT once and omit --data_root.
sbatch run_train.sh            # SLURM, 8× H100 (mass model); THERMO=1 sbatch run_train.sh for +thermo
```
Key flags that change the architecture/data path: `--stars_two_head` (out_ch 3→4), `--predict_thermo` (appends 4 gas-thermo channels; requires the large-scale data path — rejected with `--no_large_scale`), `--interpolant {fm,si}`, `--no_large_scale` (cube data, in_ch −3), `--exclude_cosmo_params` (35→31 params, drops indices 0,1,7,8 but keeps Ω_b). `--output_dir` defaults to `./runs`.

**Generate / evaluate** (DMO→hydro over a CAMELS simulation suite — the `bind-camels-suite` CLI):
```bash
bind-camels-suite --suite cv --run_dir weights/fm_two_head \
    --checkpoint_path weights/fm_two_head/last.ckpt \
    --model_name fm_two_head --output_root /path/to/eval_outputs \
    --cv_param_file ... --cv_nbody_root ... --cv_hydro_root ... --cv_fof_root ...
sbatch --array=0-9 run_test_suite.sh   # SLURM array
```
`--suite` ∈ `{cv, 1p, test, sb35, all}`. All CAMELS data roots are required flags — no hardcoded defaults. `run_test_suite.sh` is model-agnostic (mass vs mass+thermo is selected by `RUN_DIR`/`MODEL_NAME`/`CHECKPOINT_PATH` env overrides) and builds the SB35 manifest in chunk 0, gating the others on a lock file.

**Paint onto an arbitrary N-body sim** (the general, deploy-facing path — `bind.paint()` / `bind-paint`):
```bash
bind-paint --snapshot snap_090.hdf5 --group_catalog fof_subhalo_tab_090.hdf5 \
    --params my_params.npy --run_dir weights/fm_two_head --output_dir bind_output/run1
```
This reads any Gadget/Arepo HDF5 DMO snapshot via `bind.inference.io_gadget`, tiles the box, and composites per-halo patches. Released weights come from HF `Maxelee/BIND2` via `bind-download-weights {fm_two_head,fm_thermo}`.

**Filesystem layout** (large data lives on ceph, never in git):
- Training data: `<DATA_ROOT>/{train,test}/` (file lists are cached in `file_list_cache*.txt`). The pipeline that *produces* `<DATA_ROOT>` from the CAMELS SB35 suite lives in `data_generation/` (the exact MPI scripts used for the released data: `run_mpi_cpu.sh`→mass maps, `run_thermo_maps.sh`→gas-thermo channels, `run_mpi_cpu_lowmass.sh`→optional 1e12–1e13 halos; documented in `docs/data_generation.md`). They are Flatiron-rusty-specific (SLURM/MPI, hardcoded CAMELS paths as argparse defaults).
- Run outputs: `<output_dir>/<run_name>/` → `checkpoints/`, `norm_stats.npz`.
- Released weights: `weights/<run>/{last.ckpt,norm_stats.npz}` (gitignored, populated by `bind-download-weights`).

## Architecture (the big picture)

The trainable engine lives on `main`. Understanding it requires reading `src/bind/model.py` + `src/bind/data.py` + `src/bind/train.py` together:

- **`model.py`** — `UNet` predicts a flow-matching velocity. Conditioning is injected two ways: the 35 params go through `ParamEncoder` and the diffusion time through a sinusoidal embedding; their **sum** drives `AdaGroupNorm` (adaptive scale/shift) inside every `ResBlock`. The UNet input is a channel concat `[noisy_state, DMO condition, large_scale]`. Two formulations share the model:
  - `FlowMatching` — OT flow matching, **noise → hydro** (`x_t = (1-t)·noise + t·x1`), the production path.
  - `StochasticInterpolant` — a **DMO → hydro** bridge; present but not used in current analyses (and not wired for two-head).
- **`data.py`** — `NormStats` is the contract between training and inference: per-channel `log10(1+x)` standardization, plus param min/max bounds read from the **SB35 CSV** with per-param `LogFlag` (so normalization is well-defined for any sim, not just the training subset). It is **versioned/back-compatible**: old `norm_stats.npz` files load with new fields defaulting safely. Two dataset classes: `AstroDataset` (2D maps *with* `large_scale`) and `CubeAstroDataset` (6.25 Mpc/h cube projections, *no* `large_scale`, params looked up from the SB35 table by `sim_NNNN` in the path).
- **Stars two-head mode** (`--stars_two_head`) is the subtle part that threads through all three files. The Stars channel is split into **(occupancy mask, conditional log-density)** so the model emits 4 channels; `compute_norm_stats` computes occupancy/conditional stats over *occupied pixels only* (avoids zero-pixel domination); inference in `bind.inference.pipeline._denormalize_to_physical` **recombines them via a hard 0.5 occupancy gate × density** back to the standard 3-channel artifact.
- **Thermo mode** (`--predict_thermo`) appends `N_THERMO` gas-thermodynamic channels (`bind.data.THERMO_KEYS`: `compton_y, T, entropy, P_e`) to the output. Unlike the mass channels these use a plain `log10` (not `log10(1+x)`) normalization, and `norm_stats.npz` records whether it was computed with thermo stats (`NormStats.predict_thermo`) — training asserts the flag matches the stats file. Not wired into the `StochasticInterpolant` branch or the cube dataset.
- **`train.py`** — `FlowMatchingLit` (Lightning) + `AstroDataModule`. Computes/loads `norm_stats.npz` up front, derives `star_zero_norm` from it, then builds the model. AdamW + linear-warmup→cosine LR, gradient clipping, EMA weights saved into the checkpoint.
- **`bind.inference/`** — orchestration for evaluation + the general paint engine, intentionally mirroring the original analysis notebooks ("notebook-equivalent"):
  - `paint.py` is the deploy-facing API: `bind.paint(sim, model, params, output_dir, ...)` plus the `bind.Simulation` / `bind.Model` / `bind.PaintResult` classes re-exported at top level (see `src/bind/__init__.py`).
  - `io_gadget.py` reads arbitrary Gadget/Arepo HDF5 DMO snapshots + FoF/Subfind catalogs (so painting isn't limited to the CAMELS file layout).
  - `runner.py` (`run_suite`) loads a `FlowMatchingLit` checkpoint and fans CAMELS simulations out over a thread pool.
  - `pipeline.py` holds the physics primitives: particle→grid projection (`MAS_library` CIC, optional dependency), halo-cutout extraction, truth-map projection, and the "BIND composite" that pastes generated halo patches back into a full-box map (**circular `r200_factor` paste is now the standard, default `4.0`**; `r200_factor=0` is the legacy square taper — see `docs/circular_aperture.md`). `_denormalize_to_physical` lives here.
  - `config.py` builds per-suite `SimulationSpec`s; `schemas.py` defines `RunConfig`/`SimulationSpec`; `artifacts.py` handles save/load + JSON serialization (`to_jsonable`).
  - `bind.cli.camels_suite` (`bind-camels-suite`) is the CAMELS-suite CLI; `bind.cli.paint` (`bind-paint`) the single-snapshot CLI (both support `--help`; `camels_suite` supports `--n_chunks/--chunk_id` for SLURM arrays).
- **`params.py`** — parameter helpers exported at top level: `bind.fiducial_params()`, `random_params()`, `vary_param()`/`vary_params()`, `param_dataframe()`, backed by the bundled SB35 metadata in `assets/`.

## Working conventions in this repo

- **Branch organization** — `main` is the clean trunk: the core engine (`bind.data`/`bind.model`/`bind.train`/`bind.metrics`, `bind.inference/`) plus `examples/paper_figures.ipynb`. Distinct projects/analyses are **parked on topic branches**, not accumulated on `main`:
  - `feature/3d-cube` — 3D / cube-projection extension (`*_3d.py`, cube notebooks).
  - `analysis/2d` — matured 2D analyses (`scatter/` package, observables, `project1-7`, CV derivatives).
  - `analysis/tsz-icm` — tSZ / ICM thermo science: Y–M mass bias, WL calibration, entropy/pressure, Sobol assembly (`scatter/assembly_*`, `*_sobol` notebooks). Notebooks/scripts still on flat (`from data`) imports — fix per-file before reuse.
  - `analysis/ksz_project` — kSZ science analyses (renamed from `ksz_project`).
  - `feature/thermo` — **archival**: original thermo dev history + `stale/` graveyard. Its engine support is on `main`, the model notebooks were promoted to `examples/`, and the science notebooks moved to `analysis/tsz-icm`. Kept for history; don't add new work here.
  - `wip` — scratch notebooks, parameter-injection experiments, planning notes.
  - `3D` — legacy, superseded by `feature/3d-cube`.
  The two thermo model notebooks live on `main` at `examples/{paper_figures_thermo,analysis_thermo}.ipynb` (imports already rewritten to `bind.*`). Training and eval each use a single unified SLURM script — `run_train.sh` (`THERMO=1` toggles `--predict_thermo`) and `run_test_suite.sh` (model selected by env overrides) — there are no longer separate `*_two_head`/`*_thermo`/`*_parallel` variants. `feature/thermo` was deleted; its `stale/` graveyard + a full-tree tarball are archived at `/mnt/ceph/users/mlee1/bind_archive/feature_thermo/`.
  When starting new analysis, put it on the appropriate topic branch (or a new one) rather than on `main`. The remote is **`origin` → https://github.com/Maxelee/BIND.git**; topic branches are pushed there too.
- **`main` is both the trunk and the release.** It is the installable `bind` package (`src/bind/` layout) used for training (`bind.train`), evaluation (`bind.inference`), and the `bind.paint()` inference API — there is no separate flat "training" layout. Releases are cut as **git tags + GitHub Releases** (e.g. `v0.1.0`), not long-lived `release/*` branches, so the released package is always identical to validated `main`.
- **Generated artifacts are not versioned.** `.gitignore` excludes caches, `outputs/`, figures (`*.pdf/*.png/*.gif`, `figures/`, `paper_figures/`), `*.npz`/`*.npy`, `*.log`, `weights/`, and `__pycache__`. The bundled demo input (`examples/data/dmo_sample.npz`) and packaged assets (`src/bind/assets/`) are explicit allow-list exceptions.

## Known data caveats (cost real time if forgotten)

- **CAMELS `p14` bug**: the CV simulations were actually run with parameter index 14 = 0, even though the CV parameter files list 2000. Override it before normalization when working with CV.
- **1P truth maps**: the hydro `truth_maps` *do* vary with astrophysical parameters (only the DMO input is shared across a 1P set). An "all identical" check on 1P truth is a false alarm, not a data bug.

## Project memory & work log

Two complementary records, both worth consulting at the start of a task and updating as you work:

- **`docs/WORKLOG.md`** (in-repo, shared with the human and Copilot) — a reverse-chronological log of notable sessions: what changed, why, and decisions. **At the end of a session that made a meaningful change** (a reorg, a new analysis branch, a non-trivial fix, an abandoned approach worth recording), prepend a short dated entry. Keep it terse; link files/commits rather than restating diffs. Don't log trivial edits.
- **Claude Code file-based memory** at `/mnt/home/mlee1/.claude/projects/-mnt-home-mlee1-vdm-bind2/memory/` (indexed by `MEMORY.md`) — durable, non-obvious facts not derivable from code/git (project overview, the data caveats above, the branch convention, analysis findings). Add to it when you learn something that should persist across sessions but doesn't belong in the repo.
