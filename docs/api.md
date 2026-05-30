# API reference

The whole user-facing surface is three classes plus one function, all
exported at the top level of `bind`.

## Top-level

```{eval-rst}
.. autosummary::
   :nosignatures:

   bind.Simulation
   bind.Model
   bind.paint
   bind.PaintResult
   bind.extract_halo_cutouts
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
