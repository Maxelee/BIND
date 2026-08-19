# Baryonify a 205 Mpc/h N-body simulation

This page is the recipe for the most common BIND use case: **you ran an N-body
simulation, you have a snapshot and a halo catalog, and you'd like hydro
fields**. The example assumes a 205 Mpc/h box at TNG-like resolution, but the
same recipe applies to any DMO simulation in the rough range
$L \in [50, 500]\,\mathrm{Mpc}/h$ at $\gtrsim$ TNG resolution.

## What you need

1. A **DMO snapshot** in Gadget/Arepo HDF5 format.  BIND reads:
   - `Header/BoxSize` (in ckpc/h, divided internally by 1000),
   - `Header/MassTable[1]` (DM particle mass × $10^{10}\,M_\odot/h$),
   - `PartType1/Coordinates` (positions in ckpc/h).
2. A **FoF / Subfind catalog** in the same format. BIND reads:
   - `Group/GroupPos`,
   - `Group/Group_M_Crit200` (the field name is configurable),
   - optionally `Group/Group_R_Crit200`.
3. The **pretrained checkpoint** ([Hugging Face](https://huggingface.co/mel2260/BIND)):

   ```bash
   bind-download-weights fm_two_head
   ```

   For tSZ / kSZ / X-ray applications, swap or also fetch the thermo variant:

   ```bash
   bind-download-weights fm_thermo   # adds compton_y, temperature, entropy, pressure
   ```

   See [Gas thermodynamics](thermo.md) for the field definitions and
   normalization. Both checkpoints share the same `Simulation` interface
   and parameter vector.

4. A **35-dim parameter vector** that describes the cosmology and astrophysics
   you'd like to paint. For a generic external sim you can simply use the
   CAMELS-IllustrisTNG fiducial; see {doc}`parameters` for full details.

## The recipe

```python
import bind, numpy as np

# 1. Wrap the simulation -----------------------------------------------------
sim = bind.Simulation.from_paths(
    snapshot      = "/path/to/snap_090.hdf5",
    group_catalog = "/path/to/fof_subhalo_tab_090.hdf5",
    halo_mass_min = 1e13,           # M200c cut [Msun/h]; 1e13 is the trained regime
)
print(sim)
# Simulation(box=205.00 Mpc/h, n_part=1073741824, n_halos=..., M_halo_min=1.00e+13 Msun/h)

# 2. Load the model ----------------------------------------------------------
model = bind.Model.from_local("weights/fm_two_head")

# 3. Choose a parameter vector ----------------------------------------------
params = bind.fiducial_params()                            # CAMELS-TNG fiducial
# alternatives:
# params = bind.random_params(rng=0)                       # one draw from the prior
# params = bind.vary_param("RadioFeedbackFactor", fraction=1.0)  # max-AGN
# params = np.load("my_params.npy")                        # from disk

# 4. Paint -------------------------------------------------------------------
result = bind.paint(
    sim, model,
    params     = params,
    output_dir = "bind_output/L205_run1",
    n_steps    = 50,           # Euler steps per cutout (~50 is the trained sweet spot)
    batch_size = 16,
)
print(result)
```

### How the box is divided

`paint()` derives both numbers by **rounding**, not by taking a ceiling:

```python
npix    = int(round(box_size / pixel_size))     # bind.inference.paint._round_npix
n_slabs = max(1, int(round(box_size / slab_depth)))  # _round_n_slabs
```

At the native scale (`pixel_size = 50/1024 = 0.048828125` Mpc/h,
`slab_depth = 50` Mpc/h) a 205 Mpc/h box therefore gives

- `npix = round(205 / 0.048828125) = round(4198.4) = 4198` pixels per side,
- `n_slabs = round(205 / 50) = round(4.1) = 4` z-slabs, each
  $205/4 = 51.25$ Mpc/h deep.

So the run produces **4 slabs of $4198 \times 4198$ pixels**, not 5 slabs of
4096. Rounding is deliberate: it keeps the pixel scale and slab depth as close
as possible to the trained values instead of leaving a thin, under-populated
final slab. Note the *effective* depth is `box_size / n_slabs` = 51.25 Mpc/h,
2.5% deeper than the 50 Mpc/h the model was trained on. `paint()`'s off-native
warning only inspects the `slab_depth` and `pixel_size` you passed (5% relative
tolerance), so it stays silent here — the residual mismatch is yours to judge.

Every halo above `halo_mass_min` is assigned to a slab by its $z$ coordinate,
sampled independently, and pasted back into that slab's canvas.

## Outputs

`output_dir` will contain:

```
bind_output/L205_run1/
    composite_slab00.npz
    composite_slab01.npz
    composite_slab02.npz
    composite_slab03.npz
    summary.json
```

Each `.npz` contains (keys exactly as `bind.inference.paint.paint` writes them):

| key | shape | meaning |
|---|---|---|
| `dmo`            | `(npix, npix)` | DMO column density for this slab |
| `composite`      | `(3, npix, npix)` | `[DM_hydro, Gas, Stars]` painted hydro, after the global mass rescale |
| `alpha`          | `(npix, npix)` | paste weight in $[0, 1]$; the DM channel is $(1-\alpha)\,\mathrm{DMO} + \alpha\,\widehat{\mathrm{DM}}$ |
| `patch_scales`   | `(N_h,)` | per-patch mass-match rescale factors (empty if `patch_mass_match=False`) |
| `scale_global`   | scalar | the single global factor applied to enforce total-mass conservation |
| `coverage_pct`   | scalar | percent of slab pixels with $\alpha > 0.01$ |
| `n_halos`        | scalar | halos in this slab |
| `slab_idx`       | scalar | which z-slab this is |
| `n_slabs`        | scalar | total number of slabs |
| `box_size`       | scalar | Mpc/h |
| `halo_centers`   | `(N_h, 2)` | halo $(x, y)$ in **Mpc/h** (not pixels) |
| `halo_masses`    | `(N_h,)` | M200c, $M_\odot/h$ |
| `halo_r200`      | `(N_h,)` | R200c, Mpc/h (zeros if the catalog had none) |
| `generated_patches` | `(N_h, 3, 128, 128)` | per-halo hydro patches before pasting; present unless `save_per_halo_patches=False` |
| `thermo_patches` | `(N_h, 4, 128, 128)` | thermo channels in `THERMO_KEYS` order; only for a `--predict_thermo` model |
| `provenance`     | 0-d string | JSON provenance block (see below); read with `bind.inference.artifacts.read_provenance` |

A slab with no halos is written with only `dmo`, `composite` (DM = DMO, Gas =
Stars = 0), `n_halos`, `slab_idx`, `n_slabs`, `box_size`, and `provenance`.

`summary.json` records `box_size`, `pixel_size`, `slab_depth`, `npix`,
`n_slabs`, `n_halos`, `model` (its repr), `n_steps`, `patch_mass_match`,
`taper_frac`, `r200_factor`, `paste_mode`, `predict_thermo`, `thermo_keys`, a
`per_slab` list, and a `provenance` block.

### Provenance

Every output — `summary.json` and each slab `.npz` — carries the same
`provenance` block, built by `bind.inference.artifacts.build_provenance`:
`bind_version`, `checkpoint_path` and its `checkpoint_sha256`,
`norm_stats_path` and its `norm_stats_sha256`, the *resolved* `n_steps`,
`r200_factor`, `paste_mode` and `seed`, a `created` timestamp, plus the rest of
the run's settings. Inside an `.npz` it travels as a single JSON string under
the `provenance` key; read it back with

```python
from bind.inference.artifacts import read_provenance
read_provenance("bind_output/L205_run1/composite_slab00.npz")
```

Each slab additionally records its own `slab_idx` and derived `slab_seed`, so a
single slab can be regenerated on its own.

Mass is conserved by construction: `scale_global` rescales each slab composite
so that $\sum(\mathrm{DM}+\mathrm{Gas}+\mathrm{Stars}) = \sum \mathrm{DMO}$ for
that slab.

## Tuning knobs

`bind.paint(...)` accepts a handful of physics + bookkeeping flags:

| flag | default | what it does |
|---|---|---|
| `n_steps`           | `50`    | flow-matching Euler steps |
| `batch_size`        | `16`    | per-batch halos in the sampler |
| `pixel_size`        | `50/1024 ≈ 0.0488` Mpc/h | only override if your DMO map should be at a different resolution |
| `slab_depth`        | `50.0` Mpc/h | only override if you want non-native z-slabs |
| `patch_pix`         | `128`   | halo cutout size; do not change unless retraining |
| `use_amp`           | `True`  | mixed-precision sampling |
| `patch_mass_match`  | `True`  | rescale each painted patch so its DM total matches the DMO patch DM total |
| `taper_frac`        | `0.15`  | fraction of the aperture used for the cosine (Hann) taper |
| `r200_factor`       | `4.0`   | **standard**: circular paste aperture of radius $4\,R_{200c}$. `0` selects the legacy square taper |
| `paste_mode`        | `"shared"` | **standard**: overlapping halos share one realization before blending. `"average"` is the legacy independent-patch blend |
| `save_per_halo_patches` | `True` | include `generated_patches` in the output `.npz` |
| `progress`          | `True`  | tqdm progress bar |
| `redshift` / `scale_factor` | `None` | target $z$ or $a=1/(1+z)$ for a redshift-conditioned model ({doc}`redshift`) |

```{important}
`r200_factor = 4.0` with `paste_mode = "shared"` is the standard and the
default everywhere. The legacy combination (`r200_factor = 0`, square taper,
and/or `paste_mode = "average"`) leaves a measurable small-scale power deficit —
−10.6% in total-matter $P(k)$ at $k = 40$–$70\,h/$Mpc for the square taper, and
a further −10% (CV) to −12% (SB35) wherever independent realizations are
averaged in overlapping apertures. Only use the legacy settings to reproduce
older composites. Full numbers in {doc}`circular_aperture`.
```

## Working in slabs by hand (power-user)

Every step is exposed if you'd rather call them individually:

```python
slabs    = sim.project()                           # (n_slabs, npix, npix) DMO
slab_idx = sim.slab_assignment()                   # which halo lives in which slab

for s, dmo_map in enumerate(slabs):
    sel = slab_idx == s
    cutouts = bind.extract_halo_cutouts(
        dmo_map,
        sim.halo_positions[sel, :2],
        box_size = sim.box_size,
    )
    hydro_patches = model.generate(cutouts, params, n_steps=50, batch_size=16)
    # ... do whatever you want with hydro_patches ...
```

Use `bind.inference.pipeline.build_bind_composite(...)` to reproduce the
patch-back-into-canvas step that `paint()` performs internally — remember to
pass `r200_factor=4.0, paste_mode="shared"` (its own defaults) to match.

For very large boxes, prefer the three staged CLIs (`bind-paint-project`,
`bind-paint-generate`, `bind-paint-recomposite`) so the CPU projection and the
GPU sampling can run separately; see {doc}`cli`.

## Reproducibility

`bind.paint(..., seed=N)` (or `bind-paint --seed N`) seeds the sampler's initial
noise using a *local* generator, so the global RNG is untouched and the result
is reproducible for the same cutout order and `batch_size`. Without a seed each
run draws a different realization — which is the point of a generative
emulator, but not what you want when diffing two runs.

## Memory and runtime

- **Pixelization (Pylians CIC)** is the dominant cost for large boxes.
- **Sampling** cost scales as $N_\mathrm{halos} \times n_\mathrm{steps}$ network
  evaluations; batching over halos (`batch_size`) is what keeps a GPU busy.
- Because generation is per halo, lowering `halo_mass_min` raises the cost
  roughly with the halo mass function.

Measured timings for the released model on one A100 are reported in the methods
paper; see {doc}`reproducing_the_paper` for the benchmark script.

## Caveats

- BIND was trained at a fixed pixel scale (50/1024 Mpc/h ≈ 50 kpc/h) on the
  CAMELS IllustrisTNG L50n512 resolution, on halos with
  $M_{200c} > 10^{13}\,M_\odot/h$. Below that mass the model is extrapolating.
- Patches are projections through the **full 50 Mpc/h box depth**, so every
  halo-aperture quantity carries line-of-sight contamination from foreground and
  background material. This matters for aperture-integrated observables.
- The 35-dim conditioning vector is in CAMELS-IllustrisTNG SB35 ordering. For
  external simulations, the cosmology entries should be adjusted (`Omega0`,
  `sigma8`, `OmegaBaryon`, `HubbleParam`, `n_s`); the astrophysics entries
  generally do *not* have direct counterparts and should be left at the
  fiducial values unless you're explicitly exploring posteriors over them.
- `Stars` is modelled with a two-head occupancy + conditional density
  parameterization. The composite output recombines them with a **hard** 0.5
  threshold on the occupancy channel (`occ_raw > 0.5`) multiplied by the
  conditional density; near that threshold the field is intrinsically noisy.
