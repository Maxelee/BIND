# Method details

## Conditional optimal-transport flow matching

BIND learns a velocity field

$$
v_\theta:\ (\mathbb{R}^{C\times H\times W},\ [0,1],\ \mathcal{C})\;\longrightarrow\; \mathbb{R}^{C\times H\times W}
$$

so that the ODE

$$
\frac{d x_t}{d t} = v_\theta(x_t, t \mid c),\qquad x_0 \sim \mathcal{N}(0, I),
$$

transports a Gaussian sample at $t=0$ to a hydro patch $x_1$ at $t=1$ that is
consistent with the conditioning $c = (\text{DMO patch}, \theta_\mathrm{params}, \text{large-scale})$.
The training pairs $(x_0, x_1)$ are coupled along the **straight-line
optimal-transport interpolation** $x_t = (1-t)\,x_0 + t\,x_1$, giving the
target velocity $\dot x_t = x_1 - x_0$.

This recovers the simulation-free flow-matching objective of
[Lipman et al. 2023](https://arxiv.org/abs/2210.02747) and the
OT-coupled variant of [Tong et al. 2023](https://arxiv.org/abs/2302.00482).

## Sampler

```python
x = torch.randn_like(x1)
for t, dt in linspace(0, 1, n_steps):
    x = x + dt * v_θ(x, t, c)
return x
```

`n_steps=50` is the trained sweet spot. Smaller values trade fidelity for
speed; larger values give negligible improvements.

## Multi-slab compositing

A trained BIND patch is $128 \times 128$ pixels at the native scale (~6.25
Mpc/h on a side, in projection through a 50 Mpc/h slab). For arbitrary
simulation boxes:

1. **Tile the box** into $\lceil L_z / 50 \rceil$ z-slabs of depth 50 Mpc/h.
2. **Project DMO particles** in each slab onto a $1024 \times (L/50)$ pixel
   grid using mass-conserving CIC ([Pylians](https://github.com/franciscovillaescusa/Pylians3) `MAS_library.MA`).
3. **Assign halos** to slabs by their $z$ coordinate and run the model on
   every halo above `halo_mass_min` (default $10^{13} M_\odot/h$).
4. **Paste patches** back into the per-slab canvas with a smooth taper. The
   default is a square cosine taper occupying `taper_frac` of the patch
   edge; setting `r200_factor > 0` switches to a circular paste at
   $r_{200} \times r200\_factor$.
5. **Mass-match per patch** (optional, default on): rescale each painted
   hydro patch so its DM total equals the corresponding DMO patch DM total.
6. **Global rescale**: rescale the final composite so that
   $\sum_\text{slabs}(\text{DM}+\text{Gas}+\text{Stars}) = \sum \text{DMO}$.

## DM channel uses DMO as a fallback

The DM_hydro channel of the composite is

$$
\text{DM}_\text{hydro}(x) = (1 - \alpha(x))\,\text{DMO}(x) + \alpha(x)\,\widehat{\text{DM}}_\text{hydro}(x),
$$

where $\alpha$ is the (tapered) patch mask. Outside the patches there is no
model prediction, but DMO is itself an excellent predictor of the hydro DM
field at large scales (DM evolves nearly identically between DMO and hydro
runs away from baryonic cores), so it serves as a safe fallback. **Gas** and
**Stars** have no large-scale fallback and are zero outside the patches by
construction.

## Two-head Stars at inference

In two-head mode the network outputs four channels: `[DM, Gas, occupancy,
conditional_density]`. The Stars artifact is recombined as

$$
\mathrm{Stars}(x) = \sigma\big(k\,(o(x) - 0.5)\big)\,\rho(x), \qquad k \approx 8.
$$

This soft gate avoids hard 0/1 thresholds (which would make the field
non-differentiable) while still respecting the bimodal occupied/empty
structure of the stellar field.

## What BIND does not do

- It does not ingest 3D N-body cubes — only 2D projected slabs. Cube models
  are an active line of work on the `feature/3d-cube` branch.
- It does not produce thermodynamic fields out-of-the-box. The
  `feature/thermo` branch trains a sibling model that adds gas pressure and
  temperature channels.
- It does not retrain at inference time. To use a different cosmology /
  feedback prescription, change the parameter vector — *not* the weights.
