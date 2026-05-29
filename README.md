# BIND2

**B**aryon **I**maging via **N**eural **D**iffusion — a conditional flow-matching
emulator that paints baryonic fields (`DM_hydro`, `Gas`, `Stars`; optionally
thermodynamic channels) onto dark-matter-only (DMO) projections for the CAMELS
IllustrisTNG suite, conditioned on a 35-dim cosmology + astrophysics parameter
vector.

> Internal architecture, training conventions, and data caveats are documented
> in [CLAUDE.md](CLAUDE.md). This README focuses on **installing the package
> and using a released checkpoint to generate hydro maps from DMO inputs.**

## Install

```bash
git clone https://github.com/Maxelee/BIND.git
cd BIND
pip install -e .
```

The package installs as `bind` and exposes three console scripts:

- `bind-download-weights` — fetch released checkpoints from Hugging Face Hub
- `bind-test-suite` — run DMO→hydro generation on a CAMELS suite
- `bind-slim-checkpoint` — strip optimizer/scheduler state for release

Python ≥ 3.10 and PyTorch ≥ 2.0 are required.

## Pretrained checkpoints

Two checkpoints are released on the Hugging Face Hub at
[`Maxelee/BIND2`](https://huggingface.co/Maxelee/BIND2):

| Run            | Branch required   | Output channels                                                   | Size    |
|----------------|-------------------|-------------------------------------------------------------------|---------|
| `fm_two_head`  | `main`            | `[DM_hydro, Gas, Stars]` (Stars via two-head occupancy + density) | ~950 MB |
| `fm_thermo`    | `feature/thermo`  | adds thermodynamic fields                                         | ~1.9 GB |

Each release contains a slimmed Lightning checkpoint and the matching
`norm_stats.npz` required for normalization.

### Download

```bash
bind-download-weights                 # both runs
bind-download-weights --run fm_two_head
# files land under weights/<run>/{last.ckpt, norm_stats.npz}
```

The `weights/` directory is gitignored.

## Quick start: self-contained inference demo

A minimal example that requires only the bundled DMO sample and the
`fm_two_head` checkpoint:

```bash
bind-download-weights --run fm_two_head
python examples/generate_demo.py
# writes examples/demo_output.png
```

The demo normalizes the input, runs the flow-matching sampler, and
denormalizes the predicted [DM_hydro, Gas, Stars] fields. It is the recommended
entry point for understanding the inference contract — read
[examples/generate_demo.py](examples/generate_demo.py).

## Generate hydro maps over a CAMELS suite

```bash
# fm_two_head on the CAMELS CV suite:
bind-test-suite \
    --suite cv \
    --run_dir weights/fm_two_head \
    --checkpoint_path weights/fm_two_head/last.ckpt \
    --model_name fm_two_head \
    --output_root /path/to/eval_outputs \
    --cv_param_file /path/to/CosmoAstroSeed_IllustrisTNG_L50n512_CV.txt \
    --cv_nbody_root /path/to/IllustrisTNG_DM/L50n512/CV \
    --cv_hydro_root /path/to/IllustrisTNG/L50n512/CV \
    --cv_fof_root   /path/to/FOF_Subfind/IllustrisTNG_DM/L50n512/CV

# fm_thermo: switch to the feature/thermo branch first
git checkout feature/thermo
pip install -e .
bind-test-suite --suite cv --run_dir weights/fm_thermo \
    --checkpoint_path weights/fm_thermo/last.ckpt \
    --model_name fm_thermo --output_root /path/to/eval_outputs ...
```

All CAMELS data root flags (`--cv_*`, `--onep_*`, `--sb35_*`, `--test_*`) are
required when their suite is selected — there are no hardcoded defaults.

To run on data outside the CAMELS layout, instantiate the model directly:

```python
from bind.train import FlowMatchingLit
from bind.data import NormStats

model = FlowMatchingLit.load_from_checkpoint("weights/fm_two_head/last.ckpt", map_location="cuda")
ns = NormStats.load("weights/fm_two_head/norm_stats.npz")
# ... build (cond, large_scale, params) tensors per examples/generate_demo.py
gen = model.fm.sample(cond, large_scale, params, n_steps=50)
```

## Publishing new checkpoints

```bash
bind-slim-checkpoint /path/to/run_dir/checkpoints/last.ckpt \
                     /path/to/release/<run>/last.ckpt
cp /path/to/run_dir/norm_stats.npz /path/to/release/<run>/
huggingface-cli login
huggingface-cli upload Maxelee/BIND2 /path/to/release/<run> <run>
```

## Repo layout

```
src/bind/
  __init__.py
  model.py            # UNet + FlowMatching / StochasticInterpolant
  data.py             # NormStats, AstroDataset, CubeAstroDataset
  train.py            # FlowMatchingLit (Lightning)
  metrics.py
  assets/             # bundled SB35 parameter tables
  test_suite/         # CAMELS DMO->hydro evaluation pipeline
  tools/              # release helpers (slim, download)
  cli/                # console-script entrypoints
examples/
  generate_demo.py    # self-contained inference demo
  data/dmo_sample.npz # bundled 128x128 DMO + 35-dim params + truth
  paper_figures.ipynb
  analysis_2d.ipynb
```

Topic branches host distinct analyses (`analysis/2d`, `feature/3d-cube`,
`feature/thermo`, `ksz_project`, `wip`, `3D`).

See [CLAUDE.md](CLAUDE.md) for architecture, training, and data caveats.

## License

MIT — see [LICENSE](LICENSE).
