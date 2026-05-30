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
3. The **pretrained checkpoint** ([Hugging Face](https://huggingface.co/Maxelee/BIND2)):

   ```bash
   bind-download-weights fm_two_head
   ```

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
    halo_mass_min = 1e13,           # M200c cut [Msun/h]; 1e13 is a sensible default
)
print(sim)
# Simulation(box=205.0 Mpc/h, N_part=1024**3, N_halo=4172, particle_mass=4.7e+09)

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
    n_steps    = 50,           # ODE steps per cutout (~50 is the trained sweet spot)
    batch_size = 16,
)
print(result)
```

For a 205 Mpc/h box, this generates **5 z-slabs** (`⌈205 / 50⌉`) of size
$4096 \times 4096$ pixels each, evaluating the model on every halo above
`halo_mass_min` and pasting the per-halo hydro patches back into a global
canvas.

## Outputs

`output_dir` will contain:

```
bind_output/L205_run1/
    composite_slab00.npz
    composite_slab01.npz
    ...
    composite_slab04.npz
    summary.json
```

Each `.npz` contains:

| key | shape | meaning |
|---|---|---|
| `dmo`            | `(npix, npix)` | DMO column density in $M_\odot/h$/pixel |
| `composite`      | `(3, npix, npix)` | `[DM_hydro, Gas, Stars]` painted hydro |
| `halo_centers`   | `(N_h, 2)` | halo $(x, y)$ in pixels (for this slab) |
| `halo_masses`    | `(N_h,)` | M200c, $M_\odot/h$ |
| `halo_patches`   | `(N_h, 3, 128, 128)` | per-halo hydro patches, before pasting |
| `slab_index`     | scalar | which z-slab this is |

Mass is conserved up to a global rescale: $\sum_\mathrm{slabs}(\text{DM}+\text{Gas}+\text{Stars}) = \sum \text{DMO}$.

## Tuning knobs

`bind.paint(...)` accepts a handful of physics + bookkeeping flags:

| flag | default | what it does |
|---|---|---|
| `n_steps`           | `50`    | flow-matching ODE Euler steps |
| `batch_size`        | `16`    | per-batch halos in the sampler |
| `pixel_size`        | `0.0488` Mpc/h | only override if your DMO map should be at a different resolution |
| `slab_depth`        | `50.0` Mpc/h | only override if you want non-native z-slabs |
| `patch_pix`         | `128`   | halo cutout size; do not change unless retraining |
| `patch_mass_match`  | `True`  | rescale each painted patch so its DM total matches the DMO patch DM total |
| `taper_frac`        | `0.15`  | fraction of patch edge for cosine taper when pasting |
| `r200_factor`       | `0.0`   | `0` = square taper; `>0` = circular paste at $r_{200} \times r200\_factor$ |
| `save_per_halo_patches` | `True` | include `halo_patches` in the output `.npz` |
| `progress`          | `True`  | tqdm progress bar |

## Working in slabs by hand (power-user)

Every step is exposed if you'd rather call them individually:

```python
slabs = sim.project()                              # (n_slabs, 4096, 4096) DMO
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
patch-back-into-canvas step that `paint()` performs internally.

## Memory and runtime

- **Pixelization (Pylians CIC)** is the dominant cost for large boxes. A 205
  Mpc/h slab with $1024^3$ particles takes ~1 minute on a single CPU.
- **Sampling** is ~5–10 ms per cutout on an H100 at `n_steps=50`. The total
  cost is therefore $\sim N_\mathrm{halos} \times 10\,\mathrm{ms}$.
- A 205 Mpc/h L1024 box with $N_\mathrm{halos}(M_{200c} > 10^{13}) \sim 10^4$
  paints in ~2 minutes wall clock on a single H100, plus pixelization.

## Caveats

- BIND was trained at a fixed pixel scale (~50 kpc/h) and on the CAMELS
  IllustrisTNG L50n512 resolution. It will work on coarser N-body sims, but
  results below the training scale (e.g. $\sim 25 \mathrm{kpc}/h$ at L25)
  should be treated with caution.
- The 35-dim conditioning vector is in CAMELS-IllustrisTNG SB35 ordering. For
  external simulations, the cosmology entries should be adjusted (`Omega0`,
  `sigma8`, `OmegaBaryon`, `HubbleParam`, `n_s`); the astrophysics entries
  generally do *not* have direct counterparts and should be left at the
  fiducial values unless you're explicitly exploring posteriors over them.
- `Stars` is modelled with a two-head occupancy + conditional density
  parameterization. The composite output recombines them via a soft 0.5
  occupancy gate; near the gate boundary the field is intrinsically noisy.
