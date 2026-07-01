# Observable conditioning

A conditioning mode (`feature/observable-conditioning`) that replaces the 35-dim
cosmology+astrophysics parameter vector with **aperture-integrated observables
within R200**, measured in projection from each halo's own maps — the quantities
an SZ / X-ray / optical / lensing survey actually reports.

Everything else is unchanged: the DMO projection is still the image conditioning,
and the outputs are the same `[DM_hydro, Gas, Stars]` mass fields (+ the 4
gas-thermo fields with `--predict_thermo`). Only the conditioning *vector*
changes, so the U-Net and `ParamEncoder` are identical apart from `n_params =
N_OBS`; the observable vector flows through the existing `params` batch slot.

Conceptually this turns BIND into a **reconstruction / forward-modeling** task:
*given the dark matter (the DMO map ≈ lensing) and a set of integrated baryonic
observables, paint the full baryon+thermo field that is consistent with both.*

## The observable vector

Seven strictly-positive scalars, in canonical `OBSERVABLE_KEYS` order, each
computed inside the **projected `r < R200` circular aperture** about the
halo-centered map (`bind.data.compute_observables`):

| key | quantity | how | probe |
|-----|----------|-----|-------|
| `Y_200` | integrated Compton-y | `Σ compton_y · A_pix` in R200 | SZ |
| `Mgas_200` | projected gas mass | `Σ target[Gas]` in R200 | X-ray / SZ |
| `Mstar_200` | projected stellar mass | `Σ target[Stars]` in R200 | optical |
| `Tx_200` | mean temperature [K] | gas-mass-weighted over R200 | X-ray |
| `K_200` | mean entropy [keV cm²] | gas-mass-weighted over R200 | X-ray |
| `P_200` | mean pressure [Pa] | gas-mass-weighted over R200 | SZ / X-ray |
| `M_200` | M200c [M⊙/h] | the saved `halo_mass` | lensing anchor |

`Tx/K/P` weight each pixel by the Gas surface density (on pixels where the field
is `> 0`), giving the hot-gas-weighted aperture value. Like the thermo channels,
each feature spans a huge dynamic range, so it is **log10-transformed with a
per-feature floor, then standardized** (`thermo_forward` with the `obs_*` stats).
The stats are computed by `compute_norm_stats(..., condition_observables=True)`
and stored in `norm_stats.npz` (`condition_observables`, `obs_mean`, `obs_std`,
`obs_floor`; safe defaults keep old files loading).

## R200

R200c is a pure function of the already-saved `halo_mass` (M200c) at z=0:

```
M200c = (4/3) π R200c³ · 200 · ρ_crit,0,   ρ_crit,0 = 2.7754e11 h² M⊙/Mpc³
⇒ R200c[Mpc/h] = ( M200c[M⊙/h] / (4/3 π · 200 · ρ_crit,0) )^(1/3)
```

In h-units the explicit `h`'s cancel, so it is independent of the per-sim
cosmology and matches the FOF `Group_R_Crit200` by definition
(`bind.data.m200c_to_r200c`). `r200_from_sample` prefers a saved `r200` key and
falls back to deriving it — **so no data regeneration is needed to train.** To
persist it anyway (provenance / external use), `data_generation/add_r200.py`
appends an `r200` scalar to each `.npz` in place (optional, idempotent, atomic).

## Training

```bash
# observable conditioning + mass+thermo output (rotated2_128 thermo data path)
python -m bind.train --data_root /path/to/train_data_rotated2_128_cpu \
    --run_name fm_observables --stars_two_head --predict_thermo \
    --condition_observables --interpolant fm
# or via the launcher (implies --predict_thermo, run_name fm_observables):
OBS=1 sbatch run_train.sh
```

`--condition_observables` requires `--interpolant fm` and the large-scale
(rotated2_128) data path (it needs the thermo maps to build the observables); it
is mutually exclusive with `--no_large_scale` and `--exclude_cosmo_params`. A
fresh `norm_stats.npz` is computed on first launch.

### Input dropout (`--mask_observables`)

A real survey (or another sim suite) rarely reports all seven observables, and
predicting f_b is only well-posed if the *baryon-mass* observables
(`Mgas_200`/`Mstar_200`) can be **withheld** from the conditioning. `--mask_observables`
(requires `--condition_observables`; `MASK=1 OBS=1 sbatch run_train.sh`, run name
`fm_observables_masked`) trains the model to tolerate any subset:

- The conditioning vector is packed to **`2·N_OBS` = `[obs·mask, mask]`**
  (`bind.data.pack_observable_conditioning`): dropped observables are zeroed and
  flagged by a 0/1 presence mask, so the model can tell a genuinely mean-valued
  observable (which normalizes to ~0) from a missing one.
- Each training sample draws a random keep-mask (`sample_observable_keep_mask`):
  25% fully-observed (keeps the model sharp on the full-info case), else each
  observable kept independently w.p. 0.5 — covering realistic survey subsets
  (SZ+X-ray+lensing = `Y,Tx,M`) down to a single observable. An all-False draw is
  the unconditional case, consistent with the classifier-free-guidance zero vector.
- `n_params = 2·N_OBS`; `NormStats.mask_observables` records the mode so inference
  auto-detects it. The U-Net/`ParamEncoder` are unchanged apart from the width.
  Because the architecture differs (`ParamEncoder` input `N_OBS`→`2·N_OBS`), a
  masked model is a **fresh run**, not a fine-tune of the unmasked checkpoint.

## Inference

Observable conditioning is wired through the suite runner. `bind-camels-suite`
**auto-detects** an observable model from `norm_stats.condition_observables` (no
extra flag) and, for each halo, measures the `N_OBS` observables from the truth
maps and feeds them as the per-halo conditioning vector
(`bind.inference.pipeline.build_observable_vectors` → `compute_observables`, the
*same* definition as training — verified bit-identical). The mechanics:

- `generate_halo_patches(..., cond_vectors=...)` takes an `(N_halos, N_OBS)`
  matrix of already-normalized per-halo conditioning vectors, bypassing the
  per-sim param path. This is the reusable primitive any caller can drive.
- `extract_truth_mass_patches` pulls per-halo `[DM,Gas,Stars]` 6.25 Mpc/h patches
  from the full-box truth maps (same halo-center convention as the truth thermo
  patches); `build_observable_vectors` turns those + the truth thermo patches +
  the halo catalog (`halo_mass`, FOF `r200`) into the normalized conditioning.
- This is **validation-by-reconstruction**: condition on truth-measured
  observables, compare generated vs truth. It needs `load_truth=True` and the
  truth thermo maps (raises a clear error otherwise). Suite truth patches are
  axis-aligned vs the rotated training cutouts, so aperture observables differ
  slightly from training — fine for validation.

**Deploying on real (non-truth) observables** is the identical path: normalize
your measured `(Y_200, Mgas_200, Mstar_200, Tx_200, K_200, P_200, M_200)` with
`thermo_forward(obs, ns.obs_mean, ns.obs_std, ns.obs_floor)` and pass the result
as the conditioning vector to `fm.sample`. The checkpoint records
`condition_observables=True` so the mode is detectable downstream.

For a `--mask_observables` model, `build_observable_vectors` packs an all-ones mask
(full observation) automatically; to **condition on a subset** (e.g. predict
`Mgas`/f_b from `Y,Tx,M` only), build the `2·N_OBS` vector yourself with
`pack_observable_conditioning(obs_norm, keep_mask)`, setting the mask bits of the
observables you have to 1 and the rest to 0.

## Inspecting a checkpoint

`examples/analysis_observables.ipynb` loads an observable-conditioned checkpoint
(default the `fm_observables` run) and (1) reconstructs held-out halos from their
own observables — generated-vs-truth maps, per-channel dex error, stacked radial
profiles — and (2) runs a **conditioning-response** test: fix the DMO + all-but-
one observable for a halo, sweep one observable over ±dex with fixed noise, and
watch the painted field respond (proof the observables are used). Point it at
another run via `OBS_RUN_DIR` / `OBS_CKPT`.
