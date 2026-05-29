# BIND2

**B**aryon **I**maging via **N**eural **D**iffusion — a conditional flow-matching
emulator that paints baryonic fields (`DM_hydro`, `Gas`, `Stars`, optionally
thermo channels) onto dark-matter-only (DMO) projections for the CAMELS
IllustrisTNG suite, conditioned on a 35-dim cosmology + astrophysics parameter
vector.

> Internal architecture, training conventions, and data caveats are documented
> in [CLAUDE.md](CLAUDE.md). This README focuses on **using a released
> checkpoint to generate hydro maps from DMO inputs.**

## Install

```bash
git clone https://github.com/Maxelee/BIND.git
cd BIND
pip install -r requirements.txt          # if/when you add one
# core deps: torch, lightning, torch_ema, h5py, numpy, MAS_library
```

## Pretrained checkpoints

Two checkpoints are released on the Hugging Face Hub at
[`Maxelee/BIND2`](https://huggingface.co/Maxelee/BIND2):

| Run | Branch required | Output channels | Download size |
|-----|-----------------|-----------------|---------------|
| `fm_two_head` | `main` | `[DM_hydro, Gas, Stars]` (Stars via two-head occupancy + density) | ~950 MB |
| `fm_thermo`   | `feature/thermo` | adds thermodynamic fields | ~1.9 GB |

Each release contains a slimmed Lightning checkpoint (optimizer/scheduler
state stripped) and the matching `norm_stats.npz` required for normalization.

### Download

```bash
pip install huggingface_hub
python -m tools.download_weights                # both runs
python -m tools.download_weights fm_two_head    # one run
# files land under weights/<run>/{last.ckpt,norm_stats.npz}
```

The `weights/` directory is gitignored.

## Generate hydro maps from a DMO simulation

```bash
# fm_two_head (3-channel hydro) on the CAMELS CV suite:
python run_test_suite.py \
    --suite cv \
    --run_dir weights/fm_two_head \
    --checkpoint_path weights/fm_two_head/last.ckpt \
    --model_name fm_two_head \
    --output_root /path/to/eval_outputs

# fm_thermo: switch to the feature/thermo branch first
git checkout feature/thermo
python run_test_suite.py \
    --suite cv \
    --run_dir weights/fm_thermo \
    --checkpoint_path weights/fm_thermo/last.ckpt \
    --model_name fm_thermo \
    --output_root /path/to/eval_outputs
```

`run_test_suite.py` expects CAMELS DMO/hydro/FOF data on disk (see paths in its
`--cv_*`, `--onep_*`, `--test_*` flags). To run on data outside the CAMELS
layout, use `test_suite.runner.load_model_bundle` directly to instantiate the
model and call `model.fm.sample(...)` on your own DMO + parameter inputs.

## Publishing new checkpoints

To prepare a checkpoint for release, slim it first to drop the optimizer and
scheduler state (cuts size by ~3×):

```bash
python -m tools.slim_checkpoint \
    /path/to/run_dir/checkpoints/last.ckpt \
    /path/to/release/<run>/last.ckpt
cp /path/to/run_dir/norm_stats.npz /path/to/release/<run>/

# Upload to Hugging Face:
huggingface-cli login
huggingface-cli upload Maxelee/BIND2 /path/to/release/<run> <run>
```

## Repo layout

- `model.py`, `data.py`, `train.py`, `metrics.py` — core engine
- `test_suite/` — physics evaluation pipeline (notebook-equivalent)
- `tools/` — release helpers (`slim_checkpoint`, `download_weights`)
- `paper_figures.ipynb` — main figure notebook
- Topic branches host distinct analyses (`analysis/2d`, `feature/3d-cube`,
  `feature/thermo`, `ksz_project`, `wip`, `3D`)

See [CLAUDE.md](CLAUDE.md) for the full architecture and training docs.
