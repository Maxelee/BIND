# Quickstart

## Install

```{code-block} bash
pip install git+https://github.com/Maxelee/BIND.git
# or, for development
git clone https://github.com/Maxelee/BIND.git
cd BIND && pip install -e .
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0, and
[Pylians](https://github.com/franciscovillaescusa/Pylians3) (the `MAS_library`
CIC projector) — pulled in as a hard dependency.

## Fetch pretrained weights

```{code-block} bash
bind-download-weights fm_two_head      # ~950 MB into weights/fm_two_head/
```

This populates `weights/fm_two_head/{last.ckpt, norm_stats.npz}`.

## Paint baryons onto a snapshot

```{code-block} python
import bind

sim    = bind.Simulation.from_paths(
    snapshot      = "snap_090.hdf5",
    group_catalog = "fof_subhalo_tab_090.hdf5",
    halo_mass_min = 1e13,
)
model  = bind.Model.from_local("weights/fm_two_head")

result = bind.paint(
    sim, model,
    params      = bind.fiducial_params(),
    output_dir  = "bind_output/run1",
)
print(result)
```

`result.composite_paths` is a list of `.npz` files — one per z-slab — each
containing the DMO input, the BIND composite `[DM_hydro, Gas, Stars]`, the
per-halo cutouts, and bookkeeping. A `summary.json` records the geometry and
the parameter vector used.

## Same thing from the shell

```{code-block} bash
bind-paint --snapshot snap_090.hdf5 \
           --group_catalog fof_subhalo_tab_090.hdf5 \
           --params my_params.npy \
           --run_dir weights/fm_two_head \
           --output_dir bind_output/run1
```

## What's a "params" file?

A `(35,)` numpy array (`.npy` or `.txt`) in CAMELS SB35 ordering. If you don't
have one, use the bundled CAMELS-IllustrisTNG fiducial:

```{code-block} python
import numpy as np, bind
np.save("my_params.npy", bind.fiducial_params())
```

See [Parameters](parameters.md) for the full list of names and ranges.

## Next steps

- {doc}`baryonify` — full recipe for a 205 Mpc/h N-body box.
- [`examples/paint_walkthrough.ipynb`](https://github.com/Maxelee/BIND/blob/release/v0.1/examples/paint_walkthrough.ipynb) — end-to-end notebook.
- {doc}`api` — the public API reference.
