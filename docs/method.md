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

A redshift-conditioned model adds the scale factor $a = 1/(1+z)$ to the
conditioning; see {doc}`redshift`.

## Sampler

```python
x = torch.randn_like(x1)          # or a seeded local generator
for t, dt in linspace(0, 1, n_steps):
    x = x + dt * v_θ(x, t, c)
return x
```

Plain forward Euler, `n_steps` uniform substeps. `n_steps = 50` is the default
in `bind.paint`, `bind-paint` and `bind-camels-suite`. Smaller values trade
fidelity for speed; larger values give diminishing returns.

## Multi-slab compositing

A trained BIND patch is $128 \times 128$ pixels at the native scale (6.25
Mpc/h on a side, in projection through the full box depth). For arbitrary
simulation boxes:

1. **Tile the box** into `round(L_z / slab_depth)` z-slabs (default
   `slab_depth = 50` Mpc/h). Note `round`, not `ceil` — see
   {doc}`baryonify` for the worked 205 Mpc/h case.
2. **Project DMO particles** in each slab onto a `round(L / pixel_size)` square
   grid using mass-conserving CIC
   ([Pylians](https://github.com/franciscovillaescusa/Pylians3) `MAS_library.MA`).
3. **Assign halos** to slabs by their $z$ coordinate and run the model on
   every halo above `halo_mass_min` (default $10^{13}\,M_\odot/h$ — the mass
   cut the training set itself uses).
4. **Paste patches** back into the per-slab canvas. The standard, and the
   default everywhere, is a **circular Hann-tapered aperture of radius
   $4\,R_{200c}$** (`r200_factor = 4.0`) with **shared-content overlap
   handling** (`paste_mode = "shared"`). See below.
5. **Mass-match per patch** (optional, default on). In `shared` mode this is
   *aperture-local*: each paste's weighted content mass is matched to the
   weighted DMO mass inside its own footprint. In the legacy `average` mode the
   whole patch is matched against its DMO condition cutout.
6. **Global rescale**: rescale the final composite so that
   $\sum(\text{DM}+\text{Gas}+\text{Stars}) = \sum \text{DMO}$ per slab.

### Paste aperture and overlap handling

| control | standard (default) | legacy |
|---|---|---|
| `r200_factor` | `4.0` — circular aperture, radius $4\,R_{200c}$ | `0` — square Hann taper over the whole $128^2$ patch |
| `paste_mode`  | `"shared"` | `"average"` |

Both defaults exist because the legacy behaviour costs measurable small-scale
power:

- The **square taper** leaves the total-matter $P(k)$ 10.6% low at
  $k = 40$–$70\,h/$Mpc over 26 CV sims; the circular aperture reduces that to
  0.8%. The mechanism is geometric — confining a slightly over-smooth generated
  patch to a tight aperture raises its concentration. A halo with no valid
  R200c falls back to the square taper individually, so circular is always safe
  to leave on.
- **Averaging** overlapping pastes blends *independent* flow-matching
  realizations of the same region. That preserves the conditional mean but
  divides the stochastic small-scale variance by roughly the number of covering
  patches, so wherever apertures overlap the composite loses exactly the
  sampled high-$k$ power (−10% CV, −12% SB35 at $k \approx 40$–$70$ for a
  $\geq 10^{12}\,M_\odot/h$ population, where 30–50% of the painted area is
  multi-covered). `"shared"` runs a greedy set-cover in descending halo mass so
  that a halo whose aperture fits inside a more massive halo's footprint adopts
  *that* patch's realization; overlapping contributions are then identical and
  the weighted average is lossless.

A hydro-replaced control (pasting *true* hydro patches through the same
machinery) cannot see the averaging artifact, because overlapping truth patches
are cutouts of the same map and their average is a no-op. A flat control
therefore does not exonerate the paste for generated content. Full study,
numbers and caveats: {doc}`circular_aperture`.

## DM channel uses DMO as a fallback

The DM_hydro channel of the composite is

$$
\text{DM}_\text{hydro}(x) = (1 - \alpha(x))\,\text{DMO}(x) + \alpha(x)\,\widehat{\text{DM}}_\text{hydro}(x),
$$

where $\alpha$ is the (clipped) paste weight. Outside the apertures there is no
model prediction, but DMO is itself an excellent predictor of the hydro DM
field at large scales (DM evolves nearly identically between DMO and hydro
runs away from baryonic cores), so it serves as a safe fallback. **Gas** and
**Stars** have no large-scale fallback and are zero outside the apertures by
construction.

## Two-head Stars at inference

In two-head mode the network outputs four channels: `[DM, Gas, occupancy,
conditional_density]`. `bind.inference.pipeline._denormalize_to_physical`
recombines them with a **hard** threshold:

```python
occ_raw   = gen[:, 2] * norm_stats.stars_occ_std  + norm_stats.stars_occ_mean
occ_gate  = (occ_raw > 0.5).astype(np.float32)          # hard 0/1
dens_log  = gen[:, 3] * norm_stats.stars_cond_std + norm_stats.stars_cond_mean
stars     = occ_gate * (10.0 ** dens_log - 1.0)
```

$$
\mathrm{Stars}(x) = \mathbb{1}\big[o(x) > 0.5\big]\ \big(10^{\rho(x)} - 1\big)
$$

The gate is hard, not a sigmoid. The predicted occupancy is near-bimodal
($\approx 0$ or $\approx 1$), and a soft multiply lets the density head leak
through on "empty" pixels — inflating the occupied fraction by about 55
percentage points. Thresholding at 0.5 reduces that error to below 0.5
percentage points. The cost is that pixels sitting near the threshold flip
discontinuously between realizations.

## What BIND does not do

- It does not ingest 3D N-body cubes — only 2D projected slabs. Cube models are
  an active line of work on the `feature/3d-cube` branch.
- It does not localize along the line of sight. Every map is a projection
  through the full box depth, so aperture quantities include foreground and
  background material.
- It does not retrain at inference time. To use a different cosmology /
  feedback prescription, change the parameter vector — *not* the weights.

Gas thermodynamics **is** supported in the released engine (`--predict_thermo`,
the `fm_thermo` checkpoint); see {doc}`thermo`.
