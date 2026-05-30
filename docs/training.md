# How the model was trained

BIND is a **conditional Optimal-Transport flow-matching** model.  This page
documents the training data, network, conditioning, loss, and the SLURM
recipe used to produce the released `fm_two_head` checkpoint.

## Training data

- **CAMELS IllustrisTNG SB35** — 1024 hydrodynamic simulations spanning a
  35-parameter space (cosmology + astrophysics), plus the 27-element 1P / CV
  sets. 50 Mpc/h boxes, $512^3$ particles, $z=0$.
- For each pair (DMO, hydro) we project the particles onto $1024^2$ pixel
  maps with a 50 kpc/h pixel size using
  [Pylians](https://github.com/franciscovillaescusa/Pylians3) `MAS_library.MA`
  CIC, with mass weights so the maps are exactly mass-conserving. Periodic
  particles wrap correctly at the box edges.
- Per simulation we extract halo-centered $128 \times 128$ patches at every
  $M_{200c} > 10^{12.5} M_\odot/h$ halo, plus 3 large-scale context patches
  at $\{4\times, 8\times, 16\times\}$ the patch size, downsampled to $128^2$.
  Halos within `taper_frac` of the box edge wrap periodically.
- Field channels per patch: `[DM_hydro, Gas, Stars]`. **Stars** is split
  into `(occupancy, conditional log-density)` — see *two-head Stars* below.

Training and validation splits are at the simulation level (no halos from a
given sim leak between splits).

## Normalization

All channels are standardized with $y \mapsto \log_{10}(1 + y)$ followed by
per-channel mean/std. The 35-dim parameter vector is min-max scaled using the
SB35 prior bounds, with `LogFlag == 1` parameters first transformed by
$\log_{10}$. The full normalization state lives in `norm_stats.npz` and is
loaded by `bind.Model`; it is the **train/inference contract** and is
versioned for backward compatibility with older checkpoints.

## Two-head Stars

The stellar field is dominated by zero-mass pixels (galaxies are
sparse). A naïve regression to a single log-density channel collapses around
zero. We split Stars into

1. an **occupancy** channel $o(x) \in \{0, 1\}$ (1 if the pixel has any stars
   in the training simulation), and
2. a **conditional log-density** $\rho(x)$ defined and normalized only on
   occupied pixels.

The model is trained with `out_ch = 4` (DM, Gas, occupancy, conditional
density). At inference we recombine via a soft gate

$$
\mathrm{Stars}(x) = \sigma\big( k\,(o(x) - 0.5) \big)\,\rho(x)
$$

with $k \approx 8$, recovering the standard 3-channel artifact.

## Network

```
input  : [noisy_state, DMO_condition, large_scale_3]   # (3+1+3 channels)
         (in two-head Stars mode the noisy_state has 4 channels: 7 total)
output : velocity field v_θ                            # 3 or 4 channels
```

- UNet with four encoder/decoder levels and ~56 M parameters.
- Each `ResBlock` uses **`AdaGroupNorm`**: scale and shift are produced by an
  MLP from the *sum* of (a sinusoidal time embedding of $t$) + (a
  `ParamEncoder` MLP of the 35-dim parameter vector).
- Standard self-attention at the two coarsest levels.

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

At inference we integrate the learned ODE with $n_\text{steps}$ Euler
substeps:

$$
x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t \mid c).
$$

`n_steps = 50` is the trained sweet spot; smaller values trade fidelity for
speed and larger values give negligible gains.

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
| Hardware           | 8× H100 (Flatiron Rusty) |
| Strategy           | Lightning DDP, bf16 mixed-precision |
| Optimizer          | AdamW |
| Learning-rate      | 2e-4, linear-warmup → cosine decay |
| EMA                | decay 0.9999 (`torch_ema`) |
| Gradient clip      | 1.0 |
| Effective batch    | 8 GPUs × 64 patches |
| Total epochs       | 200 |

The released checkpoint stores **EMA weights** (the raw weights and optimizer
state are stripped by `bind-slim-checkpoint`).

## SLURM recipe

```{code-block} bash
:caption: run_train_two_head.sh
sbatch run_train_two_head.sh
```

equivalent to

```{code-block} bash
python -m bind.train \
    --data_root /path/to/train_data_rotated2_128_cpu \
    --run_name fm_two_head \
    --stars_two_head \
    --interpolant fm \
    --max_epochs 200
```

See [`run_train_two_head.sh`](https://github.com/Maxelee/BIND/blob/release/v0.1/run_train_two_head.sh) for the full SLURM submission script.
