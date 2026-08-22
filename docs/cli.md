# Command-line tools

BIND installs eight console scripts (`[project.scripts]` in `pyproject.toml`).
Run any of them with `--help` for the full flag list.

| script | what it does |
|---|---|
| `bind-paint` | paint one Gadget/Arepo DMO snapshot end-to-end |
| `bind-paint-project` | stage 1 of the split pipeline: project + extract cutouts (CPU/MPI) |
| `bind-paint-generate` | stage 2: run the sampler on stage-1 cutouts (GPU) |
| `bind-paint-recomposite` | stage 3: re-paste saved patches with different compositing flags |
| `bind-camels-suite` | batch evaluation over CAMELS CV / 1P / SB35 suites |
| `bind-download-weights` | fetch released checkpoints from the Hugging Face Hub |
| `bind-slim-checkpoint` | strip training state from a Lightning checkpoint |
| `bind-wlemu` | the weak-lensing statistics emulator (`bind.wlemu`) |

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
| `--snapshot`         | DMO HDF5 snapshot file or directory (required) |
| `--group_catalog`    | FoF/Subfind HDF5 catalog file or directory (required) |
| `--snapshot_index`   | snapshot number; required when the two paths above are directories |
| `--params`           | path to a `(35,)` `.npy` / `.npz` / `.txt` parameter vector (required) |
| `--run_dir`          | directory containing `last.ckpt` + `norm_stats.npz` (default `weights/fm_two_head`) |
| `--checkpoint`       | explicit checkpoint path; must be given together with `--norm_stats` |
| `--norm_stats`       | explicit norm-stats path (used with `--checkpoint`) |
| `--output_dir`       | where the per-slab `.npz`s go (default `bind_output`) |
| `--halo_mass_min`    | M200c cut, $M_\odot/h$ (default `1e13`) |
| `--pixel_size`       | override the native 0.0488 Mpc/h (50 kpc/h) pixel scale |
| `--slab_depth`       | override the native 50 Mpc/h slab depth |
| `--n_steps`          | flow-matching Euler steps (default `50`) |
| `--batch_size`       | per-batch halos in the sampler (default `16`) |
| `--device`           | `cuda` / `cpu` / `auto` (default `auto`) |
| `--no_amp`           | disable mixed-precision sampling |
| `--redshift` / `--scale_factor` | target $z$ or $a=1/(1+z)$ for a redshift-conditioned model (mutually exclusive; see {doc}`redshift`) |
| `--no_patch_mass_match` | skip per-patch DM mass matching |
| `--taper_frac`       | cosine taper edge fraction (default `0.15`) |
| `--r200_factor`      | circular paste radius in units of R200c (default `4.0`, the standard; `0` = legacy square taper) |
| `--paste_mode`       | `shared` (default, standard) or `average` (legacy) — see [Compositing](#compositing-defaults) |
| `--seed`             | seed the sampler's initial noise so the run is reproducible; recorded in the provenance block of `summary.json` and every output `.npz` (default: unseeded) |
| `--no_save_patches`  | drop the per-halo `generated_patches` array from the `.npz`s |

## Compositing defaults

Two flags control how generated halo patches are pasted back into the full-box
canvas. Both `bind-paint`, the three staged CLIs, `bind-camels-suite`,
`bind.paint()` and `bind.inference.pipeline.build_bind_composite` share the same
defaults.

| flag | standard (default) | legacy |
|---|---|---|
| `--r200_factor` | `4.0` — circular Hann-tapered aperture of radius $4\,R_{200c}$ | `0` — square Hann taper over the whole $128^2$ patch |
| `--paste_mode`  | `shared` — overlapping halos adopt one shared realization before blending | `average` — every halo pastes its own realization, blended by weighted average |

Why these are the standard:

- **Circular 4×R200c** confines each (slightly over-smooth) generated patch to a
  physically motivated aperture, which restores the small-scale total-matter
  power the square taper smears away: −10.6% → −0.8% at $k = 40$–$70\,h/$Mpc
  over 26 CV sims. If a halo has no valid R200c the code falls back to the
  square taper for that halo alone.
- **Shared content** fixes a second, independent artifact. Averaging $N$
  *independent* flow-matching realizations of the same region preserves their
  conditional mean but divides the stochastic small-scale variance by $\sim N$,
  so overlapping apertures lose exactly the sampled high-$k$ power (−10% CV to
  −12% SB35 at $k \approx 40$–$70$ for a $\geq 10^{12}\,M_\odot/h$ halo
  population). A greedy set-cover in descending halo mass makes overlapping
  halos share one realization, so the average is lossless. In `shared` mode
  `patch_mass_match` operates *aperture-locally*.

Both artifacts and their measurements are documented in
{doc}`circular_aperture`. Use `--paste_mode average --r200_factor 0` only to
reproduce pre-2026 composites.

## `bind-paint-project` / `bind-paint-generate` / `bind-paint-recomposite`

The same workflow as `bind-paint`, split into three stages so that the
CPU-bound projection, the GPU-bound sampling, and the (cheap) compositing can
run on different resources — and so compositing flags can be re-swept without
re-generating.

```{code-block} bash
# stage 1 (CPU; launch under srun with the `mpi` extra for large boxes)
bind-paint-project --snapshot snap_090.hdf5 --group_catalog fof_subhalo_tab_090.hdf5 \
    --params my_params.npy --output_dir stage1/

# stage 2 (GPU)
bind-paint-generate --stage1_dir stage1/ --run_dir weights/fm_two_head \
    --output_dir bind_output/run1

# stage 3 — re-paste the SAME generated patches under different flags
bind-paint-recomposite --stage1_dir stage1/ --generated_dir bind_output/run1 \
    --output_dir bind_output/run1_square --r200_factor 0 --paste_mode average
```

`bind-paint-generate` and `bind-paint-recomposite` both default to
`--r200_factor 4.0 --paste_mode shared --taper_frac 0.15`.

## `bind-camels-suite`

Batch-generate over a CAMELS suite (CV / 1P / SB35 / test-manifest). This is the
path used to produce all CAMELS validation outputs.

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

`--suite` ∈ `{cv, 1p, test, sb35, all}` (`test` reads a JSON manifest given by
`--test_manifest`). `--n_chunks` / `--chunk_id` allow SLURM array parallelism
(see `run_test_suite.sh`). Defaults worth knowing: `--snapshot 90`,
`--box_size 50.0`, `--npix 1024`, `--patch_pix 128`, `--halo_mass_min 1e13`,
`--n_steps 50`, `--batch_size 16`, `--device cuda`, `--r200_factor 4.0`,
`--paste_mode shared`. For a redshift-conditioned checkpoint the scale factor is
derived automatically from `--snapshot` (see {doc}`redshift`).

`--repaste` rebuilds composites from already-saved per-halo patches, so
compositing sweeps do not re-run the sampler.

## `bind-download-weights`

Download released checkpoints from the Hugging Face Hub (default repo
`mel2260/BIND`, override with `--hf_repo`).

```{code-block} bash
bind-download-weights                    # all known runs
bind-download-weights fm_two_head        # just the standard 3-channel model
```

Known runs: `fm_two_head`, `fm_thermo`, `fm_redshift_thermo`. Files land at
`weights/<run>/{last.ckpt, norm_stats.npz}` (`--dest` to change the root). The
`weights/` directory is gitignored.

## `bind-slim-checkpoint`

Shrink a Lightning training checkpoint for inference-only release.

```{code-block} bash
bind-slim-checkpoint /path/to/run_dir/checkpoints/last.ckpt \
                     /path/to/release/<run>/last.ckpt
```

It **keeps** only these top-level keys, when present: `epoch`, `global_step`,
`pytorch-lightning_version`, `state_dict`, `hparams_name`, `hyper_parameters`,
and `ema_state_dict`. Everything else — `optimizer_states`, `lr_schedulers`,
`loops`, `callbacks` — is dropped; that is where the size goes.

```{warning}
`bind-slim-checkpoint` does **not** swap raw weights for EMA weights. It copies
`state_dict` verbatim and, if the run recorded one, carries `ema_state_dict`
alongside it as a separate entry. Nothing in `bind.inference` applies the EMA
shadow at load time either, so a slimmed checkpoint is served exactly as it was
trained. The released `fm_two_head` checkpoint contains no `ema_state_dict` at
all — it is the raw epoch-80 `state_dict` (370 UNet tensors, 248,860,548
parameters).
```

## `bind-wlemu`

CLI front-end for the packaged weak-lensing statistics emulator. See
{doc}`wl_emulator`.
