# Training data generation

BIND is trained on halo-centered projection maps extracted from the
[CAMELS](https://camels.readthedocs.io) IllustrisTNG **SB35** suite (1024
simulations spanning a 35-dim cosmology + astrophysics space, 50 Mpc/h boxes,
$512^3$ particles, $z=0$). This page documents how those maps are produced so
the full path — *generate data → train → paint* — is reproducible.

The scripts live in [`data_generation/`](https://github.com/Maxelee/BIND/tree/main/data_generation)
and are the exact ones used for the released weights.

```{admonition} Environment
:class: note
These are MPI batch scripts for the Flatiron *rusty* cluster (SLURM
`cca`/`rome`, `module load openmpi python python-mpi hdf5`, a `gen_train_data`
venv with `mpi4py` and Pylians `MAS_library`). The CAMELS paths are argparse
defaults (`--hydro_base`, `--nbody_base`, `--fof_nbody_base`, `--param_file`);
override them for your own data copy. Submit each script from the
`data_generation/` directory.
```

## The pipeline

```text
CAMELS SB35 (DMO + hydro snapshots + FOF catalogs)
        │
        │  [1] run_mpi_cpu.sh → process_simulations2_cpu.py
        │      project + rotate → condition, target, large_scale, params
        ▼
train_data_rotated2_128_cpu/{train,test}/  sim_<id>/halo_<h>_rot_<r>.npz
        │
        │  [2] run_thermo_maps.sh → add_gas_thermo_maps.py
        │      append compton_y, temperature, entropy, pressure (in place)
        ▼
   (same files, now with thermo channels)   → used as <DATA_ROOT> for training

[3] run_mpi_cpu_lowmass.sh (optional): 1e12–1e13 M⊙ halos, same output layout.
```

### Step 1 — mass maps (`run_mpi_cpu.sh`)

Drives `process_simulations2_cpu.py` (128 MPI ranks over 8 nodes). For every
halo with $M_{200c} > 10^{13}\,M_\odot$ it:

- loads the DMO and hydro particle snapshots and the FOF catalog,
- applies a random 3-D rotation (10 per halo, for augmentation), observer along
  $z$,
- projects particles to **128×128** maps with Pylians CIC over the full 50 Mpc/h
  periodic depth,
- extracts a 6.25 Mpc/h halo-centered cutout.

Defaults: `--resolution 128 --total_sims 1024 --test_frac 0.1 --seed 1993
--num_rotations 10`. Output goes to `train_data_rotated2_128_cpu/{train,test}/`,
one `.npz` per halo per rotation: `sim_<id>/halo_<h>_rot_<r>.npz`.

### Step 2 — gas-thermo maps (`run_thermo_maps.sh`)

Drives `add_gas_thermo_maps.py`, which **appends** 4 mass-weighted gas
thermodynamic channels to the step-1 `.npz` files (in place):

| key | quantity | units |
|-----|----------|-------|
| `compton_y` | Compton-$y$ | dimensionless |
| `temperature` | gas temperature | K |
| `entropy` | gas entropy $k_BT/n_e^{2/3}$ | keV cm² |
| `pressure` | thermal pressure | Pa |

Only needed for the `fm_thermo` model (`--predict_thermo`). It re-uses
`--total_sims 1024 --test_frac 0.1 --seed 1993` so the train/test split matches
step 1. Smoke-test a couple of sims with `--only_sims 0,1`.

### Step 3 — low-mass halos (`run_mpi_cpu_lowmass.sh`, optional)

Drives `process_simulations2_cpu_lowmass.py` for $10^{12} < M \le 10^{13}\,M_\odot$
halos as a SLURM array (10 sims/task). **No released model trains on these yet**
— it's kept for a future low-mass extension.

## Output `.npz` format

Each file is one (halo, rotation) sample:

| key | shape | description |
|-----|-------|-------------|
| `condition` | (128, 128) | DMO projection, 6.25 Mpc/h cutout — the model's DMO input |
| `target` | (3, 128, 128) | hydro maps: `[DM_hydro, Gas, Stars]` (the prediction targets) |
| `large_scale` | (3, 128, 128) | wider DM context at 12.5 / 25 / 50 Mpc/h, same center/rotation |
| `params` | (35,) | CAMELS cosmology + astrophysics vector for the sim |
| `halo_mass` | scalar | $M_{200c}$ [$M_\odot$] |
| `halo_center` | (3,) | halo position in the box [Mpc/h] |
| `compton_y`, `temperature`, `entropy`, `pressure` | (128, 128) each | gas-thermo channels (present only after step 2) |

```{admonition} large_scale vs condition
:class: tip
The projection produces four nested DM scales (6.25 / 12.5 / 25 / 50 Mpc/h).
The innermost 6.25 Mpc/h scale is stored as `condition`; the three wider scales
are stored as `large_scale`. Training with `--no_large_scale` drops those three
context channels (and reduces the UNet `in_ch` by 3).
```

The directory this produces is what you pass to training as `<DATA_ROOT>` (see
[How the model was trained](training.md) and `run_train.sh`).
