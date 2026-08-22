# API reference

The whole user-facing surface is three classes plus one function, all
exported at the top level of `bind`. The staged variants of `paint()` and the
parameter helpers are exported alongside them.

## Top-level

```{eval-rst}
.. autosummary::
   :nosignatures:

   bind.Simulation
   bind.Model
   bind.paint
   bind.PaintResult
   bind.extract_halo_cutouts
   bind.project_and_extract
   bind.generate_from_stage1
   bind.recomposite_slab
   bind.recomposite_from_saved
   bind.fiducial_params
   bind.random_params
   bind.vary_param
   bind.vary_params
   bind.param_dataframe
```

### `bind.Simulation`

```{eval-rst}
.. autoclass:: bind.Simulation
   :members:
   :member-order: bysource
```

### `bind.Model`

```{eval-rst}
.. autoclass:: bind.Model
   :members:
   :member-order: bysource
```

### `bind.paint`

```{eval-rst}
.. autofunction:: bind.paint
```

### `bind.PaintResult`

```{eval-rst}
.. autoclass:: bind.PaintResult
   :members:
   :member-order: bysource
```

### `bind.extract_halo_cutouts`

```{eval-rst}
.. autofunction:: bind.extract_halo_cutouts
```

### Staged painting

The three-stage split behind `bind-paint-project` / `bind-paint-generate` /
`bind-paint-recomposite` (see {doc}`cli`).

```{eval-rst}
.. autofunction:: bind.project_and_extract
.. autofunction:: bind.generate_from_stage1
.. autofunction:: bind.recomposite_slab
.. autofunction:: bind.recomposite_from_saved
```

## Parameter helpers

```{eval-rst}
.. autofunction:: bind.fiducial_params
.. autofunction:: bind.random_params
.. autofunction:: bind.vary_param
.. autofunction:: bind.vary_params
.. autofunction:: bind.param_dataframe
```

## Constants

```{eval-rst}
.. autodata:: bind.NATIVE_PIXEL_SIZE_MPCH
.. autodata:: bind.NATIVE_SLAB_DEPTH_MPCH
.. autodata:: bind.PATCH_PIX
.. autodata:: bind.N_PARAMS
.. autodata:: bind.PARAM_NAMES
.. autodata:: bind.N_THERMO
.. autodata:: bind.THERMO_KEYS
```

## Gas thermodynamics

Helpers powering the optional `fm_thermo` checkpoint. See
[Gas thermodynamics](thermo.md) for the field definitions.

```{eval-rst}
.. autofunction:: bind.data.thermo_forward
.. autofunction:: bind.data.thermo_inverse
```

## Redshift conditioning

See {doc}`redshift`.

```{eval-rst}
.. autofunction:: bind.data.z_to_a
.. autofunction:: bind.data.a_to_z
```

`bind.data.SNAPSHOT_REDSHIFTS` maps CAMELS IllustrisTNG L50n512 snapshot
numbers to redshifts; the table is reproduced in {doc}`redshift`.

## Provenance

Every `paint()` output carries a provenance block; see {doc}`baryonify`.

```{eval-rst}
.. autofunction:: bind.inference.artifacts.build_provenance
.. autofunction:: bind.inference.artifacts.read_provenance
.. autofunction:: bind.inference.artifacts.file_sha256
```

## Lower-level primitives

If you need to script something that the high-level `paint()` doesn't cover,
the underlying building blocks live in `bind.inference.pipeline` and
`bind.inference.io_gadget`:

```{eval-rst}
.. autosummary::

   bind.inference.io_gadget.read_box_size
   bind.inference.io_gadget.read_dmo_particles
   bind.inference.io_gadget.read_hydro_particles
   bind.inference.io_gadget.read_fof_catalog
   bind.inference.pipeline.pixelize_z_projection
   bind.inference.pipeline.extract_multiscale
   bind.inference.pipeline.extract_halo_cutouts
   bind.inference.pipeline.normalize_cutout
   bind.inference.pipeline.build_bind_composite
```
