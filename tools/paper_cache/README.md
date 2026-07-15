# Paper-figure cache pipeline

Splits the slow BIND2 paper-figure workflow into **parallel cache builders** (heavy
compute, resumable, array/pool) and **load-only notebooks** (`examples/paper_figures2.ipynb`
+ companions) that just read the cache and plot. One model everywhere: mass +
thermodynamics as a function of redshift and parameters, validated **by halo-mass bin**.

## Layout

| file | what |
|---|---|
| `paper_config.py`    | single source of truth (paths, constants, discovery, loaders). Shared by builders **and** notebooks — flip dev→spine here (env vars). |
| `build_metric.py`    | CPU per-sim cache builders + `--reduce` collation. Metrics: `mass profiles profiles_r200 pk shapes thermo field1p`. |
| `build_gpu_insets.py`| GPU `Model.generate` caches for Figs 8-11: `covering redshift vdm obs`. |
| `../../run_paper_cache.sh` | launcher: local pool **or** SLURM array + dependent reduce. |
| `../../examples/_build_paper_nbs.py` | regenerates the 4 load-only notebooks. |

Cache lands in `$PAPER_CACHE_DIR` (default `/mnt/home/mlee1/ceph/paper_cache/<model_tag>`),
as compact artifacts (`mass_table.pkl`, `profiles.pkl`, `pk.npz`, …) + `partials/` (resumable intermediates).

## Build the cache

```bash
source /mnt/home/mlee1/venvs/torch3/bin/activate     # libstdc++ fix lives in activate

# CPU metrics — LOCAL (one node, 16-way pool, build+field1p+reduce):
POOL=16 bash run_paper_cache.sh
#   ... or SLURM: build array then dependent reduce:
#   jid=$(N_CHUNKS=8 sbatch --parse --array=0-7 run_paper_cache.sh)
#   REDUCE=1 sbatch --dependency=afterok:$jid run_paper_cache.sh

# GPU insets (needs a GPU; ~5 min on an A100):
python tools/paper_cache/build_gpu_insets.py --which all
```

Individual metric (resumable; skips finished sims):
```bash
python build_metric.py --metric mass --pool 8      # or --suite CV / --sim_ids CV/sim_0,...
python build_metric.py --metric mass --reduce
```

## Make the figures

```bash
python examples/_build_paper_nbs.py                 # (re)write the 4 load-only notebooks
# then run any notebook top-to-bottom (torch3 kernel) — every cell just loads + plots.
```

## dev → spine flip (Phase III)

The paper's spine is `fm_redshift` @ z=0. Once its suite eval exists, point the cache at it:

```bash
export PAPER_SUITE_ROOT=/mnt/home/mlee1/ceph/fm_redshift_suite
export PAPER_MODEL_SUBDIR=fm_redshift_ema
export PAPER_MODEL_TAG=fm_redshift
POOL=16 bash run_paper_cache.sh && python tools/paper_cache/build_gpu_insets.py --which all
```

The suite eval itself (GPU, user submits) is just a parameterization of `run_lowmass_suite.sh`:
```bash
RUN_DIR=/mnt/home/mlee1/ceph/fm_runs/fm_redshift MODEL_NAME=fm_redshift_ema \
CHECKPOINT_PATH=/mnt/home/mlee1/ceph/fm_runs/fm_redshift/checkpoints/last.ckpt \
OUTPUT_ROOT=/mnt/home/mlee1/ceph/fm_redshift_suite \
SUITE=cv N_CHUNKS=4 sbatch --array=0-3 run_lowmass_suite.sh   # then 1p, test
```
⚠ Smoke 1 sim first (`--sim_ids …`) to confirm `generated` is `(N,7,128,128)` at z=0.
For the P(k) hydro-replace control keep truth-thermo projection **on** for Test (no `SKIP_TRUTH`).

## Notes
- CV has fixed parameters — the parameter-response figures (3a/3b, thermo/shape response)
  populate only once 1P + Test (which vary params) are in the cache.
- Parameter-response is computed **per mass window** (`C.PARAM_WINDOWS`: `trained`=≥1e13,
  `lowmass`=1e12–1e13). Main figs read `['trained']`; `paper_fig_lowmass.ipynb` reads
  `['lowmass']`. Pooling the per-sim mean over the full ≥1e12 range inflates the True−BIND
  residual — always window it. By-mass-bin figures (2b, 4) are full-range.
- `dev` = `fm_thermo` / `fm_lowmass` (mass+thermo, z=0, on disk) — used to develop and
  smoke-test the pipeline; identical layout to the spine eval.
