# The BIND weak-lensing statistics emulator (`bind.wlemu_stats`)

Instant, self-consistent weak-lensing convergence summary statistics as a
function of baryonic-feedback physics. Given the 30 CAMELS SB35 astrophysical
parameters and a source redshift, `bind.wlemu_stats.WLEmulator` returns the full set
of κ summary statistics with calibrated uncertainties in a few milliseconds —
the community-facing product of the BIND weak-lensing pipeline.

```python
from bind.wlemu_stats import WLEmulator

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

CLI: `bind-wlemu-stats --z 1.0 --set WindEnergyIn1e51erg=7.2 --out pred.npz`
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
| Inputs | 30 SB35 astro parameters (unit cube or physical), source redshift ∈ [0.5, 2.44] (any value; the raytraced planes {0.5, 1.0, 1.5, 2.0, 2.44} are emulated directly, other values PCHIP-interpolated — see below) |
| Fixed | cosmology at the IllustrisTNG fiducial (Ωm=0.3089, σ8=0.8159, Ωb=0.0486, h=0.6774, ns=0.9667) |
| Field | convergence κ of a 5×5 deg flat-sky patch at 1024², raytraced to z≈2.5 |
| Outputs | `Cl` (18 ℓ-bins), `pdf` (60), `peak` (40), `min` (40), `V0/V1/V2` Minkowski (36 each), `scat` scattering coefficients (113), `moments` (4) — 383 numbers per (θ, z_s), each with a GP σ |
| Extras | single-field statistic covariance per z_s (from 50 map realizations), parameter metadata, exact statistic estimators (`bind.wlemu_stats.stats`) to apply to your own maps |

Statistic conventions (fixed by the training pipeline; `bind.wlemu_stats.stats`
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
from them (verified against gpytorch to <1e-6). Fitting (`bind.wlemu_stats.fit`,
`pip install bind[wlemu-stats-fit]`) needs gpytorch + scikit-learn and ~15 s per
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

## Continuous source redshift

`z_source` accepts any value in `[0.5, 2.44]`, not just the 5 raytraced
planes; values outside that range raise `ValueError`. Exact plane values (or
`z_idx=`) are emulated directly (bit-exact, no interpolation). Other values
are handled by PCHIP-interpolating (monotone cubic Hermite, hand-rolled in
numpy — no scipy dependency) the (up to 4) bracketing planes' predictions in
the *transformed* statistics space (i.e. before the `10**` step for `Cl`,
`scat`), for both the mean and the GP std. `covariance()` instead
**linearly** interpolates between the two bracketing planes' covariance
matrices (a convex combination of two PSD matrices stays PSD; PCHIP could
overshoot and break that).

Because interpolation error is unmodeled at the GP level, `predict`'s
returned std at a non-plane `z_source` is inflated by a per-block empirical
term, added in quadrature: `sqrt(sd**2 + (epsilon_b * w(z) * |mean|)**2)`,
where `epsilon_b` is a fixed per-block fractional error from leave-one-plane-
out (LOO) validation at the 3 interior planes (hardcoded as
`bind.wlemu_stats.emulator._LOO_FRAC_ERR`) and `w(z)` is the normalized distance to
the nearest plane (0 at a plane, 1 at the midpoint between two planes) — so
the inflation vanishes at the planes themselves and peaks at the midpoints.

## Reduced feedback space

`src/bind/wlemu/analysis.py` also provides a small pipeline that collapses
the 30-dim astro-parameter response down to **2 physically-labeled axes**,
built entirely from the emulator (no training-time internals needed):

1. **`active_subspace(emu, z_idx, anchors=None)`** — a global whitener over
   the full 383-dim statistics vector (same zero-variance-mask +
   eigen-truncation recipe as `block_whitener`, shared via `_eigen_whiten`,
   K ≤ 40), then the 30×30 Gram matrix of whitened-response Jacobians
   (central differences, h=0.02 in unit-cube coordinates) averaged over the
   fiducial point plus 16 Sobol-sampled prior anchors. Its eigenvectors are
   the directions the *joint* statistics respond to most strongly. On the
   shipped artifact (z_s=1): **λ2/λ1 ≈ 0.42**, **λ3/λ1 ≈ 0.11** — the
   response is dominated by one direction, with a clear second mode and a
   much weaker third.
2. **`standardized_direction(X, y)`** — a standardized-linear-regression
   direction (params → an integrated halo quantity) from an independent
   dataset: the BIND SB35 256-pt Sobol design's R200-aperture integrated
   quantities (`/mnt/ceph/users/mlee1/bind_sb35/{design,analysis_cache}/`,
   z≈0, summed/mass-weighted over the shared halo sample). Used to build a
   gas-fraction direction `g_fgas` (plus `g_mstar`, `g_Y`, `g_T` — `K`/`P`
   were not available in the cached integrated quantities and are skipped).
   These vectors are bundled (with provenance) at
   `examples/data/wlemu_phys_dirs.npz`.
3. **`rotate_to_physical_axes(eigenvectors, g_fgas, n_top)`** — projects
   `g_fgas` onto `span(top-n_top eigenvectors)`; `a1` is the (unit-norm)
   projection, `a2` its orthogonal complement in that plane. Gated on the
   **capture fraction** `‖P g_fgas‖ / ‖g_fgas‖` ≥ 0.6. On the shipped
   artifact the top-2 plane only captures 0.27 of `g_fgas` (fails the gate);
   the top-3 plane captures **0.675** and is used instead (`a1`/`a2` still
   the 2 axes carried forward). `a2` correlates most with the **Y** and
   **T** directions after removing each one's `g_fgas` component
   (cos ≈ −0.59 and −0.52, vs −0.21 for `g_mstar`) and loads most on
   `BlackHoleRadiativeEfficiency`, `IMFslope`, `QuasarThreshold`,
   `WindEnergyIn1e51erg` — an **AGN-heating/quenching axis** rather than the
   naively-expected pure stellar-wind direction (though `WindEnergyIn1e51erg`
   is still a top-4 loading, so the known f_gas–M⋆ anticorrelation,
   ρ≈−0.54, is present but not dominant).
4. **`reduced_grid_theta`/`gaussian_chi2`** — grid `θ(α) = u_fid + α1·a1 +
   α2·a2` (masking points that leave the unit cube) and score two
   likelihood variants against the noise-free fiducial mock: (i) Cl +
   well-populated peak bins with the single-field covariance + GP σ² on the
   diagonal + Hartlap, as in the tutorial's toy-inference section; (ii) the
   global-whitened K=20 modes (Hartlap, p=20, n=50). Both recover the truth
   at χ²=0 (the noise-free global minimum, trivially inside 68%); variant
   (i) gives **σ(α1) ≈ 0.35** (unit-cube), i.e. **σ(Δf_gas) ≈ 1.7e-3** for
   one 5×5 deg field via the standardized-regression scaling from step 2 —
   the number that ties this section to the paper's f_b↔Cℓ discussion.
   Variant (ii) is markedly weaker in this 2D slice (σ(α1) ≈ 0.54, and its
   68%/95% contours do not close within the unit-cube-safe sweep range) —
   the top global-whitened modes are dominated by directions other than
   `a1`/`a2`, so Cl+peaks is the more informative variant here.

See `examples/wlemu_tutorial.ipynb` §7 for the eigenspectrum, loading-bar,
and posterior-contour figures.

### Follow-ups: systematic axis matching + a K-dim corner plot

Two additions extend the above without replacing it (`rotate_to_physical_axes`'s
hand-picked `a1`/`a2` still work and the tutorial keeps both cells):

- **`identify_axes(eigenvectors, candidates, k)`** — replaces the
  hand-pick-`a1`/post-hoc-check-`a2` pattern with a full `k × 4`
  cosine-similarity match between the top-`k` eigenvectors and *all four*
  candidate directions (`g_fgas, g_mstar, g_Y, g_T`) at once. Each
  eigenvector gets a best-matching candidate (flagged `weak` if
  `|cosine| < 0.3`, i.e. no candidate in the library explains it), and each
  candidate gets its full loading vector across all `k` eigenvectors, since a
  physical direction can spread across several eigenvectors rather than
  aligning with exactly one. `k` is chosen from the mode-to-mode eigenvalue
  ratio in the spectrum plotted by `active_subspace`: the steep decline
  (ratio ≤ 0.85) runs through mode 7, settling onto a ~0.85–0.95 noise-floor
  plateau from mode 8 on (`λ[7]/λ[14] ≈ 2.1`, i.e. still only ~2× the
  15th-mode noise-floor proxy) — **k=8**. On the shipped artifact no
  candidate exceeds `|cosine| ≈ 0.64` for any eigenvector (`eig3` vs `Y`),
  and `eig5/6/8` are weak matches for all four candidates — the top-8 active
  subspace and the four R200-integrated observables are related but not
  one-to-one; every candidate's loading spreads over 2–4 eigenvectors rather
  than concentrating on one.
- **A `K`-dim corner plot in the raw eigenvector coordinates**
  (`theta(alpha) = u_fid + sum_k alpha_k * eigenvectors[:, k]`, no rotation)
  via a **Laplace/Fisher approximation**: since the fiducial mock is
  noise-free (MAP = fiducial exactly), the posterior covariance in `alpha`
  is `Sigma_alpha = inv(Fisher_alpha)`,
  `Fisher_alpha = J_alpha.T @ (hartlap * inv(C_eff)) @ J_alpha`, with
  `J_alpha` the central-finite-difference Jacobian of variant (i)'s
  Cl+peaks data vector w.r.t. `alpha` (`alpha_jacobian`, same `h=0.02`
  convention as `active_subspace`, and the same `C_eff`/Hartlap recipe that
  passed the truth-in-68% check cleanly above — variant (ii) did not).
  `laplace_alpha_covariance` builds `Fisher`/`Sigma`; `confidence_ellipse`
  draws the 68%/95% (Δχ²=2.30/6.17) boundary of a 2×2 covariance block. The
  corner plot is a plain `K×K` matplotlib grid: 1D Gaussian marginals on the
  diagonal, ellipses from the corresponding `Sigma_alpha` sub-block below.
  **Caveat (real, not cosmetic):** `Fisher_alpha`'s condition number is
  `~8.5e5` on the shipped artifact — the top-8 directions, though orthogonal
  in the whitened *full* (383-dim, 9-block) statistics sense that defines
  them, are strongly degenerate under Cl+peaks alone beyond the first 2–3
  modes. A cross-check evaluated `gaussian_chi2` on a real local grid
  (`reduced_grid_theta` generalizes directly to any two eigenvector
  directions) for the most- and least-correlated `Sigma_alpha` pairs found
  on the shipped artifact (eig1/eig3, r≈+0.98; eig1/eig8, r≈−0.01): the
  *conditional* (2-parameter-only) Laplace ellipse tracks the real grid
  contour reasonably well (confirming the Jacobian/Fisher machinery itself
  is correct), but the **marginal** ellipses shown in the corner plot are
  one to two orders of magnitude wider along the same axes than the
  conditional ones — so the corner-plot panels should be read as "these
  directions are individually well measured but not jointly separable from
  Cl+peaks data alone," not as a literal small-volume 8D confidence region.
  A CCA-based alternative to the cosine-similarity match (canonical
  correlation between per-design-point `alpha` coordinates and the raw
  per-design observable arrays, rather than the pre-fit regression
  directions) was prototyped against the BIND SB35 Sobol design's cached
  integrated quantities and gives a consistent picture (leading canonical
  correlation ≈0.80, dominated by `M_star`, spread across eig2–5) but needs
  ceph-only data not shipped with the package, so it is not wired into the
  tutorial.

## Caveats

- Cosmology is **fixed**; this emulates baryonic-feedback response only.
- Source redshifts: interpolated off the 5 raytraced planes as described
  above; accuracy is validated by leave-one-plane-out at the 3 interior
  planes but not independently confirmed for every block/plane combination.
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
| `src/bind/wlemu/fit.py` | fit/refit + k-fold validation (gpytorch; `python -m bind.wlemu_stats.fit`) |
| `src/bind/assets/wlemu_gp.npz` | the shipped emulator artifact |
| `examples/data/wlemu_validation.npz` | k-fold held-out predictions (the proof) |
| `examples/data/kappa_sample.npz` | 3 sample raytraced κ maps for the stats demo |
| `examples/wlemu_tutorial.ipynb` | tutorial + validation notebook |
