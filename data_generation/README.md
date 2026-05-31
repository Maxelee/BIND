# Training-data generation

These are the **exact scripts used to build the BIND training set** from the
CAMELS IllustrisTNG SB35 suite. They are included for transparency and
reproducibility: the released weights were trained on the output of this
pipeline. Full prose walkthrough is in the docs:
[bind.readthedocs.io → Training data generation](https://bind.readthedocs.io/en/latest/data_generation.html).

> **Environment note.** These are MPI batch scripts written for the Flatiron
> Institute *rusty* cluster (SLURM `cca`/`rome`, `module load openmpi python
> python-mpi hdf5`, and a `gen_train_data` venv with `mpi4py` + Pylians
> `MAS_library`). Paths to the CAMELS sims are hardcoded as argparse defaults
> (`--hydro_base`, `--nbody_base`, `--fof_nbody_base`, `--param_file`) — point
> them at your own CAMELS copy. Submit each script **from this directory** so
> the `python3 <script>.py` calls resolve.

## Pipeline (run in order)

| step | script (SLURM) | driver | what it does |
|------|----------------|--------|--------------|
| 1 | `run_mpi_cpu.sh` | `process_simulations2_cpu.py` | Project DMO + hydro particles to 128² halo-centered maps (`condition`, `target` = DM_hydro/Gas/Stars, `large_scale`, `params`, `halo_mass`, `halo_center`). Halos > 1e13 M⊙, 10 random rotations each. **Produces `train_data_rotated2_128_cpu/{train,test}/`.** |
| 2 | `run_thermo_maps.sh` | `add_gas_thermo_maps.py` | Add 4 gas-thermodynamic channels (`compton_y`, `temperature`, `entropy`, `pressure`) **in place** to the step-1 `.npz` files. Needed only for the `fm_thermo` model. |
| 3 *(optional)* | `run_mpi_cpu_lowmass.sh` | `process_simulations2_cpu_lowmass.py` | Same projection for 1e12 < M ≤ 1e13 halos (SLURM array, 10 sims/task). **No released model uses these yet** — kept for future low-mass extension. |

Steps 1 and 2 share `--total_sims 1024 --test_frac 0.1 --seed 1993` so the
train/test split stays consistent when thermo channels are appended.

## Output format

Each halo → `sim_<id>/halo_<h>_rot_<r>.npz`. See the docs page for the full
key/shape table; in brief: `condition` (128,128) DMO 6.25 Mpc/h cutout,
`target` (3,128,128) hydro, `large_scale` (3,128,128) wider 12.5/25/50 Mpc/h
DM context, plus `params`, `halo_mass`, `halo_center`, and (after step 2) the 4
thermo channels.

This directory is set to `<DATA_ROOT>` for training — see
[`run_train.sh`](../run_train.sh) and `bind.train`.
