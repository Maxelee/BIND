# Gas thermodynamics (`fm_thermo` checkpoint)

The default `fm_two_head` checkpoint emits three projected mass maps —
`[DM_hydro, Gas, Stars]`. The `fm_thermo` checkpoint adds **four gas
thermodynamic fields**, painted as additional 128×128 patches per halo:

| Channel | Field          | Units                          | Description                                                  |
| ------- | -------------- | ------------------------------ | ------------------------------------------------------------ |
| 4       | `compton_y`    | dimensionless                  | Line-of-sight integrated $y = \int (k_B T_e / m_e c^2)\, n_e\sigma_T\, dl$ (tSZ). |
| 5       | `temperature`  | $K$                            | Mass-weighted gas temperature in the projected slab.          |
| 6       | `entropy`      | $\mathrm{keV\,cm^2}$           | Mass-weighted ICM entropy $K = k_B T / n_e^{2/3}$.           |
| 7       | `pressure`     | $\mathrm{keV\,cm^{-3}}$        | Mass-weighted electron pressure $P_e = n_e k_B T$.            |

Channels 0–2 (mass maps) behave identically to `fm_two_head`. Channels 3–6
are intensive/extensive physical quantities — they are **not** composited
into a full-box mosaic the way mass densities are, because the
mass-conservation rescaling does not apply. Per-halo patches are saved
under the `thermo_patches` key in each slab `.npz`.

## Choosing a checkpoint

| Use case                                              | Checkpoint     | Output channels                                          |
| ----------------------------------------------------- | -------------- | -------------------------------------------------------- |
| Lensing / clustering / baryonification of mass        | `fm_two_head`  | `[DM_hydro, Gas, Stars]`                                 |
| tSZ / kSZ / X-ray / ICM analyses                      | `fm_thermo`    | `[DM_hydro, Gas, Stars, compton_y, T, K, P_e]`           |

`bind-download-weights` pulls both. Pick one at inference:

```python
import bind
model = bind.Model.from_local("weights/fm_thermo")
print(model.predict_thermo, model.norm_stats.predict_thermo)  # True, True
result = bind.paint(sim, model, params=...)
print(result.predict_thermo, result.thermo_keys)
```

The resulting `slab_*.npz` files contain a `thermo_patches` array of shape
`(n_halos, 4, 128, 128)` in the channel order above.

## Normalization

Mass channels use `log10(1 + x)` standardization. Thermo channels use
`log10(max(x, floor))` per-channel standardization, where `floor` is the
0.1 percentile of the strictly-positive log-pixel distribution from
training. This avoids `-inf` on empty pixels while preserving the dynamic
range of populated regions. Inverse: `x = 10**(t * std + mean)` (no
"+1"). See {py:func}`bind.data.thermo_forward` and
{py:func}`bind.data.thermo_inverse`.

## Training a thermo model

The pre-release run uses `--predict_thermo` on the standard 2D maps:

```bash
python -m bind.train --data_root /path/to/train_data_rotated2_128_cpu \
    --run_name fm_thermo --stars_two_head --predict_thermo \
    --interpolant fm --max_epochs 200
```

Cube datasets (`--no_large_scale`) and the SI bridge (`--interpolant si`)
do not support thermo; the trainer rejects those combinations.
