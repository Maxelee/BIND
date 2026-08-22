# Redshift conditioning

By default BIND is a $z = 0$ emulator. A model trained with
`--condition_redshift` additionally conditions on the **scale factor**
$a = 1/(1+z)$, so one set of weights paints baryons across a range of
redshifts.

## The convention: $a$, not $z$

The network conditions on $a = 1/(1+z)$, never on $z$ directly. `bind.data`
provides the two conversions:

```python
from bind.data import z_to_a, a_to_z
z_to_a(1.0452)   # -> 0.48895...
a_to_z(0.5)      # -> 1.0
```

Conditioning on $a$ keeps the variable bounded in $(0, 1]$ over the training
range and makes the embedding's resolution roughly uniform in growth rather
than piling up at high $z$.

Inside the UNet the scale factor gets its own sinusoidal → MLP embedding
(`redshift_emb`), which is **summed** with the diffusion-time embedding and the
`ParamEncoder` output before driving `AdaGroupNorm` in every `ResBlock`
(`bind.model.UNet.forward(x, t, params, scale_factor=None)`). A
redshift-conditioned model that is handed no scale factor defaults to
$a = 1$ (i.e. $z = 0$).

## Training data

The multi-redshift dataset spans **8 CAMELS IllustrisTNG snapshots**
(`bind.data.SNAPSHOT_REDSHIFTS`):

| snapshot | 90 | 82 | 74 | 60 | 52 | 44 | 32 | 24 |
|---|---|---|---|---|---|---|---|---|
| $z$ | 0.0000 | 0.2087 | 0.4679 | 1.0452 | 1.4837 | 2.0020 | 3.0081 | 4.0079 |

It is produced by `data_generation/process_simulations_multiz.py` in one MPI
pass that writes both the mass maps and the four gas-thermo maps, with **one
rotation per halo** (the $z=0$ dataset uses ten). The layout gains a snapshot
level:

```
train_data_multiz_128_cpu/
    train/sim_<i>/snap_<NNN>/sim_<i>_halo_<k>_rot_0.npz
    test/  ...
```

Each `.npz` carries a scalar `redshift` and `scale_factor` alongside the usual
keys. `AstroDataset` reads `scale_factor` if present, else `redshift`, else
falls back to the `snap_<NNN>` level in the path via `SNAPSHOT_REDSHIFTS` —
and raises if the snapshot number is unknown. The train/test split is by
simulation, so a sim is train-or-test across **all** its snapshots (no redshift
leakage).

## Training

```bash
REDSHIFT=1 sbatch run_train.sh          # run_name fm_redshift
```

`REDSHIFT=1` adds `--condition_redshift --predict_thermo` and points
`DATA_ROOT` at the multi-redshift tree — the multi-z dataset always carries
thermo channels, so the two flags travel together. Equivalently:

```bash
python -m bind.train --data_root /path/to/train_data_multiz_128_cpu \
    --run_name fm_redshift --stars_two_head \
    --condition_redshift --predict_thermo \
    --interpolant fm --lr 1e-4 --max_epochs 200
```

`--condition_redshift` requires the large-scale data path and
`--interpolant fm`.

### Leave-one-redshift-out

`--exclude_snaps 60` drops a whole snapshot from **both** the train and
validation splits. The excluded snapshot's held-out patches then test whether
the model genuinely interpolates in $a$ rather than memorizing the eight
discrete training redshifts. Filtering the validation split too keeps
checkpoint selection clean.

## Inference

Every entry point takes either a redshift or a scale factor, never both:

```python
result = bind.paint(sim, model, params=p, output_dir="out", redshift=1.0452)
# or, equivalently
result = bind.paint(sim, model, params=p, output_dir="out", scale_factor=0.48895)

patches = model.generate(cutouts, p, n_steps=50, redshift=0.5)
```

```bash
bind-paint ... --redshift 1.0452
bind-paint ... --scale_factor 0.48895
```

Behaviour in the edge cases (`bind.inference.paint.Model._resolve_scale_factor`):

| model | argument | what happens |
|---|---|---|
| redshift-conditioned | `redshift=z` | uses $a = 1/(1+z)$ |
| redshift-conditioned | `scale_factor=a` | uses $a$ as given |
| redshift-conditioned | neither | **warns**, defaults to $z = 0$ ($a = 1$) |
| not redshift-conditioned | either | **warns**, ignores the argument |

Off-grid redshifts are allowed — that is the point of conditioning on a
continuous $a$ rather than a snapshot index.

### CAMELS suite evaluation

`bind-camels-suite` does *not* take a redshift flag. It detects a
redshift-conditioned checkpoint from its hyperparameters and derives the scale
factor from `--snapshot` via `SNAPSHOT_REDSHIFTS`, erroring out if that
snapshot has no known redshift. The resolved `scale_factor` and `redshift` are
written into each `summary.json`, so a suite run at snapshot 60 is
unambiguously a $z = 1.0452$ evaluation.

## Released weights

`bind-download-weights fm_redshift_thermo` fetches the redshift-conditioned
checkpoint (mass + thermo). Its `norm_stats.npz` is computed over the
multi-redshift dataset, so it is not interchangeable with the $z=0$ stats.

## Caveats

```{warning}
**Thermo physics at $z > 0$ is not validated.** The multi-redshift data
generator carries the textbook comoving→physical scale-factor powers (physical
density $\propto a^{-3}$, so pressure and entropy pick up $a$-factors; physical
pixel area $\propto a^{2}$ for Compton-$y$; temperature is $a$-independent), but
these have never been checked against an independent $z > 0$ reference. The
three **mass** channels are $a$-factor-free and unaffected. Treat
`compton_y`, `temperature`, `entropy` and `pressure` from a redshift model at
$z > 0$ as provisional.
```

```{warning}
**`bind.data.m200c_to_r200c` is a $z = 0$ relation.** It inverts
$M_{200c} = \tfrac{4}{3}\pi R_{200c}^3 \cdot 200\,\rho_{\rm crit,0}$ with the
*present-day* critical density, which is exactly why $h$ cancels and it needs no
cosmology. At $z > 0$ the correct relation carries $E(z)^2$ and the per-simulation
$\Omega_m$; using the $z=0$ form at higher redshift produces a spurious apparent
evolution in any aperture quantity (a fake $f_b(z)$ trend, for instance). Read
`Group_R_Crit200` from the catalog, or compute $R_{200c}(z)$ yourself.
```

- A redshift model is still a **single-redshift-per-call** emulator: one
  `paint()` call paints one $a$ for the whole box. Lightcones need one call per
  shell.
- Silently omitting the scale factor is the most common failure mode: the model
  falls back to $a = 1$, so a $z > 0$ evaluation quietly becomes a $z = 0$ one.
  The warning is there for a reason — do not filter it out.
