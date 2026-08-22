# How the model was trained

BIND is a **conditional Optimal-Transport flow-matching** model.  This page
documents the training data, network, conditioning, loss, and the SLURM
recipe used to produce the released `fm_two_head` checkpoint.

## Training data

- **CAMELS IllustrisTNG SB35** — 1024 hydrodynamic simulations spanning a
  35-parameter space (cosmology + astrophysics), plus the CV / 1P sets used for
  evaluation. 50 Mpc/h boxes, $512^3$ particles, $z = 0$ (snapshot 090).
- For each (DMO, hydro) pair the particles are projected onto $1024^2$ pixel
  maps at a 50/1024 Mpc/h ≈ 50 kpc/h pixel size with
  [Pylians](https://github.com/franciscovillaescusa/Pylians3) `MAS_library.MA`
  CIC, mass-weighted so the maps are exactly mass-conserving. Periodic
  particles wrap correctly at the box edges.
- Per simulation, halo-centered $128 \times 128$ patches are extracted at every
  halo with $M_{200c} > 10^{13}\,M_\odot/h$
  (`data_generation/process_simulations2_cpu.py`, `load_halos(..., mass_threshold=1e13)`),
  with **10 random 3-D rotations per halo** for augmentation.
- Each sample carries four nested DM scales, all resampled to $128^2$:
  the **6.25 Mpc/h** cutout is the `condition`, and **12.5 / 25 / 50 Mpc/h**
  (i.e. 2×, 4×, 8× the patch) are stacked into the 3-channel `large_scale`
  context. The 50 Mpc/h scale is the whole box, so patches are projections
  through the full box depth.
- Field channels per patch: `[DM_hydro, Gas, Stars]`. **Stars** is split
  into `(occupancy, conditional log-density)` — see *two-head Stars* below.

Train/test is split at the simulation level (`--test_frac 0.1 --seed 1993`), so
no halo from a given sim leaks between splits.

The full generation pipeline is documented in {doc}`data_generation`.

## Normalization

Mass channels are standardized with $y \mapsto \log_{10}(1 + y)$ followed by
per-channel mean/std. Thermo channels use a zero-safe plain
$\log_{10}$ instead (see {doc}`thermo`). The 35-dim parameter vector is min-max
scaled using the SB35 prior bounds, with `LogFlag == 1` parameters first
transformed by $\log_{10}$. The full normalization state lives in
`norm_stats.npz` and is loaded by `bind.Model`; it is the **train/inference
contract** and is versioned for backward compatibility with older checkpoints.

## Two-head Stars

The stellar field is dominated by zero-mass pixels (galaxies are sparse). A
naïve regression to a single log-density channel collapses around zero. We
split Stars into

1. an **occupancy** channel $o(x) \in \{0, 1\}$ (1 if the pixel has any stars
   in the training simulation), standardized by the occupied fraction $p$ and
   $\sqrt{p(1-p)}$, and
2. a **conditional log-density** $\rho(x) = \log_{10}(1 + \mathrm{Stars})$
   whose mean/std are computed over **occupied pixels only** (so the zero
   pixels cannot dominate the statistics).

The model is trained with `out_ch = 4` (DM, Gas, occupancy, conditional
density). At inference the two heads are recombined with a **hard** 0.5
threshold, not a sigmoid:

$$
\mathrm{Stars}(x) = \mathbb{1}\big[o(x) > 0.5\big]\ \big(10^{\rho(x)} - 1\big)
$$

The occupancy prediction is near-bimodal; a soft gate lets the density head leak
onto empty pixels and inflates the occupied fraction by about 55 percentage
points, whereas the hard threshold keeps that error below 0.5 pp. See
`bind.inference.pipeline._denormalize_to_physical`.

## Network

```
input  : [noisy_state, DMO_condition, large_scale_3]
         out_ch (state) + 1 (condition) + 3 (large_scale)
output : velocity field v_θ, out_ch channels
```

With two-head Stars, `out_ch = 4` and `in_ch = 4 + 1 + 3 = 8`. Adding
`--predict_thermo` appends 4 channels to both (`out_ch = 8`, `in_ch = 12`);
`--no_large_scale` drops 3 input channels.

| | released `fm_two_head` value |
|---|---|
| `base_ch` | 128 |
| `ch_mult` | (1, 2, 4, 8) — four levels |
| `n_blocks` | 2 residual blocks per level |
| `emb_dim` | 512 |
| `attn_resolutions` | (32, 16) — self-attention at the two coarsest levels |
| `dropout` | 0.1 |
| `cfg_dropout` | 0.1 |
| `n_params` | 35 |
| UNet tensors | 370 |
| UNet parameters | **248,860,548** (≈249 M) |

Each `ResBlock` uses **`AdaGroupNorm`**: scale and shift are produced by an MLP
from the *sum* of a sinusoidal time embedding of $t$, a `ParamEncoder` MLP of
the parameter vector, and — for a redshift-conditioned model — a sinusoidal→MLP
embedding of the scale factor $a = 1/(1+z)$ ({doc}`redshift`).

## Loss and sampler

We use OT-coupled flow matching:

$$
x_t = (1 - t)\,x_0 + t\,x_1, \qquad x_0 \sim \mathcal{N}(0, I),\ x_1 = \text{hydro patch}
$$

with $t \sim \mathcal{U}[0, 1]$. The loss is

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, x_1, c}\,\big\| v_\theta(x_t, t \mid c) - (x_1 - x_0) \big\|_2^2,
$$

where $c = (\text{DMO}, \theta_\mathrm{cosmo+astro}, \text{large\_scale})$.
`cfg_dropout = 0.1` randomly zeroes the parameter vector during training so the
model also learns an unconditional velocity.

At inference we integrate the learned ODE with $n_\text{steps}$ forward-Euler
substeps:

$$
x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t \mid c).
$$

`n_steps = 50` is the default in every inference entry point.

```{image} _static/flow_matching_evolution_3panel.png
:alt: Flow-matching trajectory: noise -> intermediate -> data
:width: 95%
:align: center
```

(Top: DM_hydro. Middle: Gas. Bottom: Stars in two-head mode showing the gated
recombination at inference time.)

## Optimization

| | |
|---|---|
| Hardware           | 8× H100 (Flatiron *rusty*), one node |
| Strategy           | Lightning DDP, `precision='bf16-mixed'` |
| Optimizer          | AdamW, `weight_decay = 1e-4` |
| Learning rate      | **1e-4**, 1000-step linear warmup → cosine decay |
| EMA                | decay 0.9999 (`torch_ema`), updated every step |
| Gradient clip      | 1.0 |
| Batch              | `--batch_size 64` per DataLoader (i.e. per GPU under DDP) × 8 GPUs |
| `--max_epochs`     | 200 |
| Checkpointing      | `save_top_k=3` on `val/loss`, plus `last.ckpt` |

```{warning}
**The released `fm_two_head` checkpoint stores raw weights, not EMA weights.**
Inspecting it directly: `epoch = 80`, `global_step = 64638`, top-level keys
`{epoch, global_step, pytorch-lightning_version, state_dict, hparams_name,
hyper_parameters}` — there is **no** `ema_state_dict`. Nothing in
`bind.inference` applies an EMA shadow at load time, and `bind-slim-checkpoint`
does not swap raw for EMA either (it only drops optimizer/scheduler/loop/callback
state; see {doc}`cli`). So the served weights are exactly the epoch-80 raw
`state_dict`. Training does maintain an EMA and writes `ema_state_dict` into
checkpoints from newer runs — it simply is not what this release ships.
```

```{note}
Flow-matching `val/loss` tracks sample quality only weakly, so `save_top_k` on
`val/loss` is not a reliable model-selection criterion. Keep an epoch ladder and
select on physics metrics instead.
```

## SLURM recipe

```{code-block} bash
:caption: run_train.sh
sbatch run_train.sh             # mass model (fm_two_head)
THERMO=1 sbatch run_train.sh    # + 4 gas-thermo fields (fm_thermo)
REDSHIFT=1 sbatch run_train.sh  # multi-redshift, mass + thermo (fm_redshift)
OBS=1 sbatch run_train.sh       # observable conditioning (fm_observables)
```

The mass-model case is equivalent to

```{code-block} bash
python -m bind.train \
    --data_root /path/to/train_data_rotated2_128_cpu \
    --batch_size 64 --num_workers 8 \
    --base_ch 128 --n_blocks 2 --emb_dim 512 \
    --dropout 0.1 --cfg_dropout 0.1 \
    --interpolant fm --lr 1e-4 --max_epochs 200 \
    --stars_two_head \
    --output_dir /path/to/runs --run_name fm_two_head
```

`$BIND_DATA_ROOT` can replace `--data_root`. Env overrides recognised by
`run_train.sh`: `RUN_NAME`, `DATA_ROOT`, `OUTPUT_DIR`, `MAX_EPOCHS`, the mode
toggles above, and `MASK` (with `OBS=1` it adds `--mask_observables` and names
the run `fm_observables_masked`); any extra arguments are passed straight through to
`bind.train`.

See [`run_train.sh`](https://github.com/Maxelee/BIND/blob/main/run_train.sh) for
the full SLURM submission script.
