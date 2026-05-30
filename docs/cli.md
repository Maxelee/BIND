# Command-line tools

BIND ships four console scripts. Run any of them with `--help` for the full
flag list.

## `bind-paint`

Paint baryons onto a single Gadget/Arepo HDF5 snapshot.

```{code-block} bash
bind-paint \
    --snapshot       /path/to/snap_090.hdf5 \
    --group_catalog  /path/to/fof_subhalo_tab_090.hdf5 \
    --params         my_params.npy \
    --run_dir        weights/fm_two_head \
    --output_dir     bind_output/run1
```

Common flags:

| flag | description |
|---|---|
| `--snapshot`         | DMO HDF5 snapshot (required) |
| `--group_catalog`    | FoF/Subfind HDF5 catalog (required) |
| `--snapshot_index`   | snapshot number, used in output filenames |
| `--params`           | path to a `(35,)` `.npy` / `.txt` parameter vector |
| `--run_dir`          | directory containing `last.ckpt` + `norm_stats.npz` |
| `--checkpoint`       | explicit checkpoint path (overrides `--run_dir`) |
| `--norm_stats`       | explicit norm-stats path (overrides `--run_dir`) |
| `--output_dir`       | where the per-slab `.npz`s go |
| `--halo_mass_min`    | M200c cut, $M_\odot/h$ (default `1e13`) |
| `--pixel_size`       | override the native 50 kpc/h pixel scale |
| `--slab_depth`       | override the native 50 Mpc/h slab depth |
| `--n_steps`          | flow-matching ODE steps (default `50`) |
| `--batch_size`       | per-batch halos in the sampler (default `16`) |
| `--device`           | `cuda` / `cpu` / `auto` |
| `--no_amp`           | disable bf16 / fp16 mixed-precision sampling |
| `--no_patch_mass_match` | skip per-patch DM mass matching |
| `--taper_frac`       | cosine taper edge fraction (default `0.15`) |
| `--r200_factor`      | circular paste radius factor (default `0.0` = square taper) |
| `--no_save_patches`  | drop the per-halo `halo_patches` array from the `.npz`s |

## `bind-camels-suite`

Batch-generate over a CAMELS suite (CV / 1P / SB35 / test). This is the path
used to produce all CAMELS validation outputs.

```{code-block} bash
bind-camels-suite \
    --suite cv \
    --run_dir weights/fm_two_head \
    --checkpoint_path weights/fm_two_head/last.ckpt \
    --model_name fm_two_head \
    --output_root /path/to/eval_outputs \
    --cv_param_file /path/to/CosmoAstroSeed_IllustrisTNG_L50n512_CV.txt \
    --cv_nbody_root /path/to/IllustrisTNG_DM/L50n512/CV \
    --cv_hydro_root /path/to/IllustrisTNG/L50n512/CV \
    --cv_fof_root   /path/to/FOF_Subfind/IllustrisTNG_DM/L50n512/CV
```

`--suite` ∈ `{cv, 1p, test, sb35, all}`. `--n_chunks/--chunk_id` allow SLURM
array parallelism (see `run_test_suite_parallel.sh`).

## `bind-download-weights`

Download released checkpoints from the Hugging Face Hub.

```{code-block} bash
bind-download-weights              # both releases
bind-download-weights fm_two_head  # just the standard 3-channel model
```

Files land at `weights/<run>/{last.ckpt, norm_stats.npz}`. The `weights/`
directory is gitignored.

## `bind-slim-checkpoint`

Strip optimizer/scheduler state and convert from raw to EMA weights, in
preparation for releasing a checkpoint.

```{code-block} bash
bind-slim-checkpoint /path/to/run_dir/checkpoints/last.ckpt \
                     /path/to/release/<run>/last.ckpt
```
