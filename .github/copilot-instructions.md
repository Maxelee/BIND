# GitHub Copilot instructions for BIND2

BIND2 is a conditional **flow-matching emulator that paints baryonic fields onto
dark-matter-only (DMO) maps** for the CAMELS IllustrisTNG suite. Input: a DMO
density projection + a 35-dim cosmological/astrophysical parameter vector.
Output: hydro fields `[DM_hydro, Gas, Stars]` as 128×128 maps.

## Stack & layout
- Python, PyTorch + **PyTorch Lightning** (DDP, bf16, EMA), `torch_ema`, `h5py`,
  `MAS_library` (CAMELS particle→grid projection), NumPy/pandas. Runs in the
  venv at `/mnt/home/mlee1/venvs/torch3` on the Flatiron Rusty SLURM cluster.
- Core engine (on `main`): `model.py` (UNet + `FlowMatching` / `StochasticInterpolant`),
  `data.py` (`NormStats`, `AstroDataset`, `CubeAstroDataset`), `train.py`
  (`FlowMatchingLit`, `AstroDataModule`), `metrics.py`.
- `test_suite/` is the *physics evaluation pipeline* (not unit tests):
  `runner.py` orchestrates, `pipeline.py` has projection/halo/composite
  primitives, `config.py`/`schemas.py`/`artifacts.py` support it,
  `run_test_suite.py` is the CLI.

## Conventions to follow when suggesting code
- **Normalization is the train/inference contract.** Fields use `log10(1+x)`
  per-channel standardization; the 35 params are min/max scaled using bounds
  from the SB35 CSV with per-param `LogFlag` (log10 for flagged params). Always
  round-trip through `NormStats`; keep `norm_stats.npz` loading backward-compatible.
- **Conditioning** flows through `AdaGroupNorm` (scale/shift from time-embedding +
  param-embedding sum). UNet input is a channel concat `[state, DMO condition,
  large_scale]`; `--no_large_scale` drops the 3 large-scale channels.
- **Stars two-head mode** (`--stars_two_head`): Stars channel → (occupancy,
  conditional log-density), so `out_ch=4`; occupancy/conditional stats are
  computed over occupied pixels only, and inference recombines via a soft
  multiplier back to 3 channels. Keep both single-head and two-head paths working.
- Match the existing terse, docstring-first style; avoid adding frameworks or
  config systems. Prefer extending the argparse flags in `train.py` /
  `run_test_suite.py` over new entrypoints.

## Repo workflow (important)
- `main` is a clean trunk. New analyses go on **topic branches**
  (`feature/3d-cube`, `analysis/2d`, `wip`), not on `main`.
- **Do not commit generated artifacts**: caches, `outputs/`, figures
  (`*.pdf/*.png/*.gif`), `*.npz`/`*.npy`, `*.log` are gitignored. Large data
  lives on `/mnt/.../ceph`, never in git.
- See `CLAUDE.md` for the fuller architecture/commands and `docs/WORKLOG.md` for
  recent decisions.

## Data caveats
- **CAMELS `p14` bug**: CV sims were run with parameter index 14 = 0 though the
  CV param files list 2000 — override before normalizing CV data.
- **1P truth maps** vary with astro params (only the DMO input is shared); an
  "all identical" check on 1P truth is a false alarm, not a bug.

## This branch: feature/3d-cube
Adds a 3D variant of the engine: `model_3d.py` (`UNet3d`/`FlowMatching3d`, 3D
convs, `in_ch=4`, no large_scale), `data_3d.py` (`NormStats3d`, mask-aware
zero-voxel fill via `fill_zeros_smooth`, drops the 36th constant param column),
`train_3d.py` (uses `torch.compile`; handles `_orig_mod.` checkpoint prefixes;
`map_location='cpu'` for DDP). `run_test_suite_3d.py` runs on pre-extracted halo
volumes (no projection/compositing). Keep `*_3d` modules parallel to their 2D
counterparts. Also covers the 2D "cube" projection path (`--no_large_scale`).
