# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

BIND2 is a conditional **flow-matching emulator that paints baryonic fields onto dark-matter-only (DMO) maps** for the CAMELS IllustrisTNG suite. Given a DMO density projection and a 35-dim cosmological+astrophysical parameter vector, it generates the corresponding hydro fields — `[DM_hydro, Gas, Stars]` — as 128×128 maps. The downstream science (in the topic branches, see below) uses the trained emulator to study how baryonic physics responds to feedback parameters.

## Environment & commands

Everything runs in a Python venv on the Flatiron Rusty cluster; there is **no build system, no linter, and no unit-test framework** in this repo. "Test suite" here means the *physics evaluation pipeline* (`test_suite/`), not unit tests.

```bash
source /mnt/home/mlee1/venvs/torch3/bin/activate    # required for all commands below
```

**Train** (multi-GPU via Lightning DDP, bf16, EMA):
```bash
# Local/interactive single run:
python train.py --data_root /mnt/home/mlee1/ceph/train_data_rotated2_128_cpu \
    --run_name fm_two_head --stars_two_head --interpolant fm --max_epochs 200
# On SLURM (8×H100): edit + sbatch
sbatch run_train_two_head.sh
```
Key `train.py` flags that change the architecture/data path (read these before assuming defaults):
`--stars_two_head` (out_ch 3→4), `--interpolant {fm,si}`, `--no_large_scale` (cube data, in_ch −3), `--exclude_cosmo_params` (35→31 params, drops indices 0,1,7,8 but keeps Ω_b).

**Generate / evaluate** (DMO→hydro over a simulation suite):
```bash
# Single suite locally:
python run_test_suite.py --suite cv --run_dir /mnt/home/mlee1/ceph/fm_runs/fm_two_head \
    --model_name fm_two_head --output_root /mnt/home/mlee1/ceph/fm_testsuite
# All suites in parallel on SLURM (array job; N_CHUNKS must equal array size):
sbatch --array=0-9 run_test_suite_parallel.sh
```
`--suite` ∈ `{cv, 1p, test, sb35, all}`. The parallel script builds the SB35 manifest in chunk 0 and gates the others on a lock file.

**Filesystem layout** (large data lives on ceph, never in git):
- Training data: `/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/{train,test}/` (file lists are cached in `file_list_cache*.txt`).
- Run outputs: `/mnt/home/mlee1/ceph/fm_runs/<run_name>/` → `checkpoints/`, `norm_stats.npz`.
- Eval outputs: `/mnt/home/mlee1/ceph/fm_testsuite/`.
- Source sims: CAMELS DM/hydro/FOF under `/mnt/home/mlee1/Sims/...` and `/mnt/ceph/users/camels/...`.

## Architecture (the big picture)

The trainable engine lives on `main`. Understanding it requires reading `model.py` + `data.py` + `train.py` together:

- **`model.py`** — `UNet` predicts a flow-matching velocity. Conditioning is injected two ways: the 35 params go through `ParamEncoder` and the diffusion time through a sinusoidal embedding; their **sum** drives `AdaGroupNorm` (adaptive scale/shift) inside every `ResBlock`. The UNet input is a channel concat `[noisy_state, DMO condition, large_scale]`. Two formulations share the model:
  - `FlowMatching` — OT flow matching, **noise → hydro** (`x_t = (1-t)·noise + t·x1`), the production path.
  - `StochasticInterpolant` — a **DMO → hydro** bridge; present but not used in current analyses (and not wired for two-head).
- **`data.py`** — `NormStats` is the contract between training and inference: per-channel `log10(1+x)` standardization, plus param min/max bounds read from the **SB35 CSV** with per-param `LogFlag` (so normalization is well-defined for any sim, not just the training subset). It is **versioned/back-compatible**: old `norm_stats.npz` files load with new fields defaulting safely. Two dataset classes: `AstroDataset` (2D maps *with* `large_scale`) and `CubeAstroDataset` (6.25 Mpc/h cube projections, *no* `large_scale`, params looked up from the SB35 table by `sim_NNNN` in the path).
- **Stars two-head mode** (`--stars_two_head`) is the subtle part that threads through all three files. The Stars channel is split into **(occupancy mask, conditional log-density)** so the model emits 4 channels; `compute_norm_stats` computes occupancy/conditional stats over *occupied pixels only* (avoids zero-pixel domination); inference in `test_suite/pipeline.py` **recombines them via a soft multiplier** back to the standard 3-channel artifact. This was the fix behind the "stellar bias" commit.
- **`train.py`** — `FlowMatchingLit` (Lightning) + `AstroDataModule`. Computes/loads `norm_stats.npz` up front, derives `star_zero_norm` from it, then builds the model. AdamW + linear-warmup→cosine LR, gradient clipping, EMA weights saved into the checkpoint.
- **`test_suite/`** — orchestration for evaluation, intentionally mirroring the original analysis notebooks ("notebook-equivalent"):
  - `runner.py` (`run_suite`) loads a `FlowMatchingLit` checkpoint and fans simulations out over a thread pool.
  - `pipeline.py` holds the physics primitives: particle→grid projection (`MAS_library` CIC), halo-cutout extraction, truth-map projection, and the "BIND composite" that pastes generated halo patches back into a full-box map (square taper or `r200_factor` circular paste).
  - `config.py` builds per-suite `SimulationSpec`s; `schemas.py` defines `RunConfig`/`SimulationSpec`; `artifacts.py` handles save/load + JSON serialization (`to_jsonable`).
  - `run_test_suite.py` is the CLI that wires these together (supports `--n_chunks/--chunk_id` for SLURM arrays).

## Working conventions in this repo

- **Branch organization** — `main` is the clean trunk: the core engine (`data/model/train/metrics`, `test_suite/`) plus `paper_figures.ipynb`. Distinct projects/analyses are **parked on topic branches**, not accumulated on `main`:
  - `feature/3d-cube` — 3D / cube-projection extension (`*_3d.py`, cube notebooks).
  - `analysis/2d` — matured 2D analyses (`scatter/` package, observables, `project1-7`, CV derivatives).
  - `wip` — scratch notebooks, parameter-injection experiments, planning notes.
  When starting new analysis, put it on the appropriate topic branch (or a new one) rather than on `main`. There is **no git remote** — this is a local-only repo.
- **Generated artifacts are not versioned.** `.gitignore` excludes caches, `outputs/`, figures (`*.pdf/*.png/*.gif`, `figures/`, `paper_figures/`), `*.npz`/`*.npy`, `*.log`, and `__pycache__`. Existing committed figures were kept, but do not add new generated figures/data. Notebooks are committed *with* their outputs.

## Known data caveats (cost real time if forgotten)

- **CAMELS `p14` bug**: the CV simulations were actually run with parameter index 14 = 0, even though the CV parameter files list 2000. Override it before normalization when working with CV.
- **1P truth maps**: the hydro `truth_maps` *do* vary with astrophysical parameters (only the DMO input is shared across a 1P set). An "all identical" check on 1P truth is a false alarm, not a data bug.

## Project memory & work log

Two complementary records, both worth consulting at the start of a task and updating as you work:

- **`docs/WORKLOG.md`** (in-repo, shared with the human and Copilot) — a reverse-chronological log of notable sessions: what changed, why, and decisions. **At the end of a session that made a meaningful change** (a reorg, a new analysis branch, a non-trivial fix, an abandoned approach worth recording), prepend a short dated entry. Keep it terse; link files/commits rather than restating diffs. Don't log trivial edits.
- **Claude Code file-based memory** at `/mnt/home/mlee1/.claude/projects/-mnt-home-mlee1-vdm-bind2/memory/` (indexed by `MEMORY.md`) — durable, non-obvious facts not derivable from code/git (project overview, the data caveats above, the branch convention, analysis findings). Add to it when you learn something that should persist across sessions but doesn't belong in the repo.

---

## This branch: `feature/3d-cube`

Extends the 2D engine (documented above) to **3D halo volumes** and the **2D "cube"-projection** dataset. Everything above still holds; the parallel `*_3d` modules mirror their 2D counterparts but operate on `(C, D, H, W)` volumes. Keep the 3D code parallel to the 2D code — same conditioning scheme, same two-head Stars option.

- **`model_3d.py`** — `UNet3d` / `FlowMatching3d`: 3D-conv versions of the UNet and OT flow matching. `in_ch=4` (state 3 + DMO condition 1); **there is no `large_scale` conditioning in 3D**. Same AdaGroupNorm time+param conditioning.
- **`data_3d.py`** — `NormStats3d` + `AstroDataset3d` work on pre-extracted halo volumes. Two 3D-specific points: (1) `fill_zeros_smooth` does **mask-aware Gaussian filling of empty voxels** before normalization, and the fill settings are stored in the stats so inference reproduces training; (2) the param vector's 36th column is a constant `50000.0` placeholder and is dropped — only the first 35 are used.
- **`train_3d.py`** — `FlowMatching3dLit`, like the 2D trainer but: **uses `torch.compile`** (compiled on first train start, with `_orig_mod.` checkpoint-prefix handling so compiled/uncompiled checkpoints interchange), migrates `torch_ema` shadow params onto the correct device, and loads checkpoints with `map_location='cpu'` to avoid DDP OOM. `batch_size` defaults to 1 (volumes are large).
- **`run_test_suite_3d.py`** — generates directly on **pre-extracted halo volumes** (`sim_N/halo_M.npz`); unlike the 2D `test_suite/`, there is no full-box projection or compositing step. Supports `--n_chunks/--chunk_id` for SLURM arrays.
- **2D cube path** — reuses `main`'s `CubeAstroDataset` (6.25 Mpc/h projections, `--no_large_scale`); `analysis_2d_cube.ipynb` / `comparison_2d_vs_cube.ipynb` compare it against the original 2D projections.

Commands: `sbatch run_train_3d_two_head.sh` (3D train) · `sbatch run_train_cube_two_head.sh` (2D cube train) · `python run_test_suite_3d.py …` or `sbatch run_test_suite_parallel_3d.sh` (3D eval).
