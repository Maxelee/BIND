# Gas thermodynamics (`fm_thermo` checkpoint)

The default `fm_two_head` checkpoint emits three projected mass maps —
`[DM_hydro, Gas, Stars]`. Training with `--predict_thermo` (the released
`fm_thermo` checkpoint) appends **four gas thermodynamic fields**, painted as
additional 128×128 patches per halo.

The canonical order is `bind.data.THERMO_KEYS`:

```python
>>> from bind import THERMO_KEYS, N_THERMO
>>> THERMO_KEYS
('compton_y', 'temperature', 'entropy', 'pressure')
>>> N_THERMO
4
```

## The four fields

| index in `thermo_patches` | index in the physical `(3 + 4)` array | key | units | definition |
|---|---|---|---|---|
| 0 | 3 | `compton_y`   | dimensionless | Compton-$y$, $\int (k_B T_e / m_e c^2)\, n_e \sigma_T\, dl$, CIC-projected through the full 50 Mpc/h box depth |
| 1 | 4 | `temperature` | K | gas **mass-weighted** temperature in the projected column |
| 2 | 5 | `entropy`     | keV cm² | gas mass-weighted $K = k_B T / n_e^{2/3}$ |
| 3 | 6 | `pressure`    | **Pa** (SI) | gas mass-weighted **total thermal** pressure $P = (\gamma - 1)\,\rho\, u$ |

```{warning}
Two things about `pressure` are easy to get wrong: it is the **total** thermal
pressure of the gas, not the electron pressure $P_e$, and it is in **pascals**,
not keV cm⁻³. It is computed per particle as `P = (GAMMA - 1) * rho_phys *
u_phys` with `rho_phys` in kg/m³ and `u_phys` in (m/s)², then CIC-projected with
mass weights and divided by the projected mass
(`data_generation/add_gas_thermo_maps.py`). Convert yourself if you need
$P_e$ or keV cm⁻³.
```

Further caveats that matter for ICM work:

- **Star-forming gas is excluded** before projection (`StarFormationRate > 0`
  particles are dropped), so these are hot-phase quantities.
- `temperature`, `entropy` and `pressure` are **mass-weighted line-of-sight
  averages** (projected $q\,m$ divided by projected $m$), not emission-weighted
  or spectroscopic-like quantities.
- `compton_y` is an **integral** along the line of sight, not an average, and it
  is already normalized by the physical pixel area at deposit time.
- All four are projected through the **full box depth**, so they carry
  foreground/background contamination just like the mass maps.

## Where the channels appear

Channels 0–2 (mass maps) behave identically to `fm_two_head`. The four thermo
channels are intensive/line-of-sight quantities, **not** mass densities — the
mass-conservation rescale does not apply to them, so they are never composited
into a full-box mosaic. They are stored per halo instead, under the
`thermo_patches` key of each `composite_slab{NN}.npz`:

```python
d = np.load("bind_output/run1/composite_slab00.npz")
d["composite"]        # (3, npix, npix)  mass mosaic
d["thermo_patches"]   # (n_halos, 4, 128, 128) in THERMO_KEYS order
```

`summary.json` records `predict_thermo` and the `thermo_keys` list, so the
channel order travels with the output.

## Choosing a checkpoint

| Use case | Checkpoint | Output channels |
|---|---|---|
| Lensing / clustering / baryonification of mass | `fm_two_head` | `[DM_hydro, Gas, Stars]` |
| tSZ / kSZ / X-ray / ICM analyses | `fm_thermo` | `[DM_hydro, Gas, Stars, compton_y, temperature, entropy, pressure]` |
| Multi-redshift + thermo | `fm_redshift_thermo` | as `fm_thermo`, plus scale-factor conditioning ({doc}`redshift`) |

`bind-download-weights` pulls all three. Pick one at inference:

```python
import bind
model = bind.Model.from_local("weights/fm_thermo")
print(model.predict_thermo, model.norm_stats.predict_thermo)  # True, True
result = bind.paint(sim, model, params=...)
print(result.predict_thermo, result.thermo_keys)
```

## Normalization

Mass channels use `log10(1 + x)` standardization. Thermo channels use a
zero-safe plain `log10` — no `+1`:

```python
t = (np.log10(np.maximum(x, floor)) - mean) / (std + 1e-8)   # bind.data.thermo_forward
x = 10.0 ** (t * std + mean)                                  # bind.data.thermo_inverse
```

Per channel, `floor = 10 ** percentile(log10(x[x > 0]), 0.1)` — the 0.1st
percentile of the strictly positive pixels — and `mean` / `std` are then taken
over **all** pixels after clipping to that floor
(`bind.data._compute_thermo_stats`). This avoids `-inf` on empty pixels while
preserving the dynamic range of populated regions. Because the inverse is a pure
power of ten, reconstructed thermo maps are strictly positive: a pixel that was
empty in truth comes back at roughly the floor, not at zero.

`norm_stats.npz` records `predict_thermo` alongside `thermo_mean`,
`thermo_std`, `thermo_floor`; training asserts the flag matches the stats file.

## Training a thermo model

```bash
THERMO=1 sbatch run_train.sh
```

equivalently

```bash
python -m bind.train --data_root /path/to/train_data_rotated2_128_cpu \
    --run_name fm_thermo --stars_two_head --predict_thermo \
    --interpolant fm --lr 1e-4 --max_epochs 200
```

With `--stars_two_head --predict_thermo` the UNet has `out_ch = 4 + 4 = 8` and
`in_ch = 8 + 1 + 3 = 12`.

Cube datasets (`--no_large_scale`) and the SI bridge (`--interpolant si`) do not
support thermo; the trainer rejects those combinations. The thermo maps must
already be present in the training `.npz` files — see step 2 of
{doc}`data_generation`.

```{warning}
Thermo physics at $z > 0$ has **not** been validated. The comoving→physical
conversions in the multi-redshift data generator carry the textbook scale-factor
powers (physical density $\propto a^{-3}$ for pressure and entropy, physical
pixel area $\propto a^{2}$ for Compton-$y$, temperature $a$-independent), but
they have not been checked against an independent $z>0$ reference. Treat thermo
outputs from a redshift-conditioned model at $z > 0$ as unvalidated.
```
