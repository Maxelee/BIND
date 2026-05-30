# BIND — paint baryons onto your N-body simulation

```{image} _static/fig1_showcase.png
:alt: BIND showcase
:width: 100%
:align: center
```

**B**aryons **I**nduced via **N**eural **D**iffusion is a flow-matching
emulator that takes a dark-matter-only (DMO) snapshot, plus a 35-dim
cosmology-and-astrophysics parameter vector, and returns the corresponding
hydro fields `[DM_hydro, Gas, Stars]` as projected mass maps.

It is *fast* (one forward pass per ~50 Mpc/h slab), *probabilistic* (samples
from the posterior over hydro fields given DMO + parameters), and *scalable*
(applies tile-by-tile to boxes much larger than the 50 Mpc/h training box —
e.g. a 205 Mpc/h N-body simulation is just ~5 z-slabs × 64 tiles per slab).

```{image} _static/flow_matching_sampling.gif
:alt: Flow-matching sampling
:width: 80%
:align: center
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀  Quickstart
:link: quickstart
:link-type: doc

Install the package, fetch the pretrained weights, and paint baryons in three
lines of Python.
:::

:::{grid-item-card} 🌌  Baryonify a 205 Mpc/h N-body sim
:link: baryonify
:link-type: doc

End-to-end recipe for users who already have a Gadget/Arepo snapshot they want
hydro fields for.
:::

:::{grid-item-card} 🎛️  Parameters
:link: parameters
:link-type: doc

The 35-dim CAMELS conditioning vector, prior box, and the helpers for picking
fiducial / random / one-at-a-time variants.
:::

:::{grid-item-card} 🧠  How the model was trained
:link: training
:link-type: doc

Training data, network architecture, conditioning, loss, and the SLURM recipe.
:::

:::{grid-item-card} 🔬  Method details
:link: method
:link-type: doc

Conditional optimal-transport flow matching, two-head Stars, multi-slab
compositing, and mass conservation.
:::

:::{grid-item-card} 📚  API reference
:link: api
:link-type: doc

`Simulation`, `Model`, `paint`, `PaintResult`, parameter helpers, and
`bind.inference.pipeline` primitives.
:::
::::

## Why BIND?

Running a hydrodynamic simulation is expensive. Running an N-body simulation is
~10–100× cheaper but gives you *only the dark-matter field*. BIND closes that
gap: train once on CAMELS-IllustrisTNG, then **paint baryons onto any DMO
simulation** of comparable resolution at near-zero cost.

It is a conditional generative model: given a DMO column-density patch and a
35-dim parameter vector, the network samples a hydro field consistent with the
training simulations. You can therefore

- generate plausible hydro fields for boxes you've only run with N-body,
- explore the posterior by drawing multiple samples,
- forecast how your sim would look under different feedback prescriptions by
  varying the parameter vector while keeping the DMO field fixed.

```{toctree}
:hidden:
:caption: Getting started
quickstart
baryonify
parameters
```

```{toctree}
:hidden:
:caption: Reference
method
training
api
cli
```

```{toctree}
:hidden:
:caption: Project
GitHub <https://github.com/Maxelee/BIND>
Hugging Face weights <https://huggingface.co/Maxelee/BIND2>
```
