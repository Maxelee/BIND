# The BIND weak-lensing statistics emulator (`bind.wlemu`)

Instant, self-consistent weak-lensing convergence summary statistics as a
function of baryonic-feedback physics. Given the 30 CAMELS SB35 astrophysical
parameters and a source redshift, `bind.wlemu.WLEmulator` returns the full set
of κ summary statistics with calibrated uncertainties in a few milliseconds —
the community-facing product of the BIND weak-lensing pipeline.

```python
from bind.wlemu import WLEmulator

emu = WLEmulator.load()                                   # packaged artifact
pred = emu.predict({"WindEnergyIn1e51erg": 7.2}, z_source=1.0)
pred["Cl"], pred["Cl_std"]                                # + pdf, peak, min, V0-V2, scat, moments
cov = emu.covariance(z_source=1.0, blocks=("Cl", "peak"))  # single-field covariance

# custom output grids: interpolated from the native grid (Cl in log-log,
# nu-binned blocks linearly in nu); out-of-range points raise ValueError
pred = emu.predict(params, z_source=1.0, ell=my_ell, nu=my_nu)
pred["grids"]                                             # {"Cl": my_ell, "pdf": my_nu, ...}
cov = emu.covariance(z_source=1.0, blocks=("Cl",), ell=my_ell)  # exact A @ C @ A.T
```

CLI: `bind-wlemu --z 1.0 --set WindEnergyIn1e51erg=7.2 --out pred.npz`
(`--list-params` prints the parameter table). Tutorial + validation:
`examples/wlemu_tutorial.ipynb`.

`predict`/`predict_vector`/`covariance` accept optional custom output grids for
the 7 binned blocks (Cl, pdf, peak, min, V0, V1, V2 — `scat`/`moments` are
discrete coefficients and cannot be regridded): a per-block `grids={...}`
dict, or the shorthands `ell=` (→ `grids["Cl"]`) and `nu=` (applied to all six
nu-binned blocks, each interpolated from its own native grid: `pdf_x`,
`peak_x` for peak/min, `mink_thr` for V0-V2). Values are linearly interpolated
from the native grid (log10-log10 for Cl); requests outside the native
range raise `ValueError`. `predict`'s regridded `_std` uses the same linear
map as the mean (`A @ std`, a conservative neighbor-correlation
approximation); `covariance`'s regrid is exact (`A @ C @ A.T`).

## What it emulates

| | |
|---|---|
| Inputs | 30 SB35 astro parameters (unit cube or physical), source redshift ∈ {0.5, 1.0, 1.5, 2.0, 2.44} |
| Fixed | cosmology at the IllustrisTNG fiducial (Ωm=0.3089, σ8=0.8159, Ωb=0.0486, h=0.6774, ns=0.9667) |
| Field | convergence κ of a 5×5 deg flat-sky patch at 1024², raytraced to z≈2.5 |
| Outputs | `Cl` (18 ℓ-bins), `pdf` (60), `peak` (40), `min` (40), `V0/V1/V2` Minkowski (36 each), `scat` scattering coefficients (113), `moments` (4) — 383 numbers per (θ, z_s), each with a GP σ |
| Extras | single-field statistic covariance per z_s (from 50 map realizations), parameter metadata, exact statistic estimators (`bind.wlemu.stats`) to apply to your own maps |

Statistic conventions (fixed by the training pipeline; `bind.wlemu.stats`
reproduces them bit-faithfully in numpy): power spectrum on the raw map;
PDF/peaks/minima/Minkowski on the map smoothed with a periodic Fourier
Gaussian of σ = 2 arcmin and standardized per map to S/N units; scattering
coefficients on the map box-averaged to 256².

## Training data (provenance)

BIND painted baryons onto IllustrisTNG-DMO halos to z = 2.5 at **253 Sobol
points** of the 30-dim astro-parameter space (`bind_sb35` suite, shared DMO
halos → controlled same-halo experiment). The painted snapshots were tiled
into lightcones and raytraced to κ maps at 5 source planes; **50 map
realizations per parameter point** (the same 50 underlying lightcone fields at
every point, so runs are noise-paired and parameter *responses* are far less
noisy than single-field scatter). The per-map statistics form
`stats_cache.npz` (253 × 50 × 5 × 383); the regression target is the mean
over realizations at each (run, z_s).

## Design (and why)

Per source redshift: standardize the (log-transformed where strictly positive:
`Cl`, `scat`) statistics vector → PCA to 32 coefficients → one **exact GP per
coefficient** (ARD Matérn-5/2, constant mean, batched float64) — the classic
cosmology-emulator construction. Nine designs were compared on held-out runs
(nearest-neighbor, linear/quadratic ridge, RBF, gradient-boosted trees, MLP
ensembles, joint-z GP, per-z GP): **the per-z GP wins on every statistic
block**, and beats the field-level flow-matching map emulator by 5–100× per
statistic. Per-z models beat a joint-z GP because the z-dependence is strong
and nonlinear while the astro response is subtle.

Because each redshift's 383-dim statistics vector is emulated **jointly**
(one PCA basis, statistics all measured from the *same* maps), the predicted
statistics are mutually self-consistent — the joint response across statistics
is the response of the underlying κ maps, not of independently fitted curves.

**Inference is numpy-only.** The artifact (`src/bind/assets/wlemu_gp.npz`,
~4 MB) stores GP hyperparameters, training inputs, PCA bases, normalizations,
covariances, and parameter metadata; `WLEmulator` rebuilds the exact posterior
from them (verified against gpytorch to <1e-6). Fitting (`bind.wlemu.fit`,
`pip install bind[wlemu-fit]`) needs gpytorch + scikit-learn and ~15 s per
source redshift on one GPU.

## Validation (see the tutorial for figures)

- **Protocol**: 13-fold cross-validation over the 253 parameter points — every
  point is predicted by an emulator that never saw it, with the identical
  recipe as the shipped artifact. Held-out predictions ship in
  `examples/data/wlemu_validation.npz`.
- **Accuracy**: per-block fractional errors at the few-percent level or below,
  and — the stricter statement — **median |error| ≤ ~0.3× the realization-noise
  floor** (SEM of the 50-map mean) on every block; the apparent % error on
  peaks/minima is dominated by sparsely populated histogram tails.
- **Calibration**: GP σ validated against held-out z-scores.
- **Pipeline identity**: the shipped `measure_stats` reproduces the training
  statistics on the raw maps to float32 precision, and the standard-split
  protocol reproduces the historical design-comparison table exactly.

## Caveats

- Cosmology is **fixed**; this emulates baryonic-feedback response only.
- Source redshifts are the 5 discrete raytraced planes — no z interpolation.
- The covariance is estimated from 50 (noise-paired) realizations of one
  5×5 deg field: restrict to a data-vector subset well below 50 dims before
  inverting, and apply a Hartlap-style correction.
- Parameters outside the SB35 prior box extrapolate (a warning is raised).
- The training maps inherit BIND's trained regime (halos ≥ 1e13 M⊙/h painted;
  see the paper) and the lightcone construction choices.

## Files

| file | role |
|---|---|
| `src/bind/wlemu/emulator.py` | numpy-only `WLEmulator` (predict, covariance, param conversion) |
| `src/bind/wlemu/stats.py` | the exact statistic estimators (numpy, for your own maps) |
| `src/bind/wlemu/fit.py` | fit/refit + k-fold validation (gpytorch; `python -m bind.wlemu.fit`) |
| `src/bind/assets/wlemu_gp.npz` | the shipped emulator artifact |
| `examples/data/wlemu_validation.npz` | k-fold held-out predictions (the proof) |
| `examples/data/kappa_sample.npz` | 3 sample raytraced κ maps for the stats demo |
| `examples/wlemu_tutorial.ipynb` | tutorial + validation notebook |
