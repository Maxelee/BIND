# Observable → f_b PoC — Results (2026-06-03)

*Branch `analysis/observable-fb-map`. Notebook `observable_fb_map.ipynb`,
machine-readable `figures/observable_fb/results.json`. **⚠️ Provisional** — one
subgrid model (CAMELS-TNG), fixed cosmology, scalar integrated quantities, no
hyperparameter tuning. Numbers are hypotheses pending raw-output review.*

Reframing note: σ(f_b | observables) **is** the error bar on "predict f_b given
an observation." A point prediction f̂_b(Y) is only usable if its residual
scatter is small relative to the mass-only spread — that ratio is the deliverable.

## (a) What the numbers say
Held-out-θ test (`GroupKFold(design)`, 5 folds, HistGradientBoosting), within
mass bins. σ = RMS(f_b − f̂_b) over held-out rows; "reduction" = 1 − σ/σ_marginal.

| observed | logM 13–13.5 | 13.5–14 | 14–14.5+ |
|---|---|---|---|
| mass only (σ_marg) | 0.0227 | 0.0131 | 0.0069 |
| + Y (tSZ) | 0.0167 (−26%) | 0.0110 (−16%) | 0.0062 (−9%) |
| + Y, SX (X-ray) | 0.0097 (−57%) | 0.0075 (−43%) | 0.0053 (−22%) |
| + Y, SX, T, S, P | 0.0082 (−64%) | 0.0058 (−56%) | 0.0035 (−49%) |
| kSZ anchor (Y, τ) | 0.0144 (−37%) | 0.0097 (−26%) | 0.0052 (−24%) |

Design-bootstrap 16/84 intervals on σ are ±~0.0005 (tight; the unit is the
design). Phase-1 (Y-only) dex scatter: 0.058 / 0.034 / 0.018.

**Degeneracy diagnostic** — residual std(f_b) at fixed conditioning:

| bin | marginal | at fixed (mass,Y) | at fixed (mass,Y,SX) |
|---|---|---|---|
| 13–13.5 | 0.0227 | 0.0170 | 0.0110 |
| 13.5–14 | 0.0131 | 0.0110 | 0.0081 |
| 14+ | 0.0069 | 0.0064 | 0.0058 |

Adding SX collapses the residual at fixed Y by ~30–35% in the two lower bins.

## (b) What it might mean
- **The observable → f_b map is well-posed**, but as a *vector*, not from Y
  alone. A tSZ + X-ray observation recovers f_b to ~half the mass-only
  uncertainty on feedback settings never seen in training (the parameter-free
  premise holds). This is the PoC **succeeding** — not the null where
  σ(f_b|Y) ≈ σ_marginal.
- **Y alone is a weak predictor** (−9 to −26%) and degenerate: at fixed (mass, Y)
  f_b still scatters by ~0.017 (low mass). tSZ measures pressure ∝ n_e·T, which
  trades gas content against temperature — many (f_b, T) combinations give the
  same Y. **SX ∝ n_e²√T breaks that** because it weights density differently,
  resolving the f_b–T direction.
- **The signal is strongest at low mass and fades toward clusters** (−64% → −49%
  full vector; −26% → −9% for Y). Plausibly the high bin is closer to
  self-similar / less feedback-modulated, and has fewer halos (13k vs 212k rows)
  with an already-small marginal spread, leaving less to predict.
- The **kSZ/τ gas-mass proxy is not the best predictor** (worse than Y+SX at low
  mass), so the result does not ride on the τ↔f_b circularity — independent
  X-ray information genuinely improves the f_b prediction.

## (c) What must be true for the optimistic reading — and what is still untested
- **Untested: off-grid / other subgrid model.** Everything here is CAMELS-TNG.
  "Parameter-free" is only shown *within one model's* feedback family; a SIMBA /
  Astrid / real-data test is the eventual point and is out of scope here.
- **Untested: real observational systematics.** SX, Y here are clean field-level
  integrals in a perfect R200 aperture — no instrument noise, beam, projection
  of uncorrelated structure, mis-centering, or aperture/mass-proxy error. Those
  inflate σ(f_b|obs); the numbers above are a best case.
- **BIND fidelity is assumed.** f_b and the observables are both *emulator*
  outputs; the map's tightness partly reflects internal BIND consistency. The
  honest test maps **BIND-observables → CAMELS-truth f_b** (validate the predictor
  against held-out hydro truth, not against BIND's own f_b). Not yet done.
- **Fixed cosmology.** Marginalising Ω_m, σ8 will add scatter and possibly new
  degeneracies (Y and f_b both move with cosmology). Out of scope (current
  generation is single-cosmology).
- **Scalars only.** Profiles/images could tighten the map further but are a later
  stretch.

## Verdict
The observable vector → f_b map is **well-posed within CAMELS-TNG at fixed
cosmology**: a held-out tSZ+X-ray observation predicts f_b to ~half the mass-only
uncertainty, and the X-ray observation (not the gas-mass proxy) is the
degeneracy-breaker. Y alone is insufficient. No paper-ready generalisation claim
until the off-grid + BIND-vs-truth tests are run.

## Reproduce
```bash
python tools/observable_fb_reduce.py          # -> ceph/sobol_ss_cv/obs_fb_extra.npz (self-checks vs cube)
python tools/build_observable_fb_nb.py         # -> observable_fb_map.ipynb
jupyter nbconvert --execute --inplace \
  --ExecutePreprocessor.kernel_name=torch3 --ExecutePreprocessor.timeout=1800 \
  observable_fb_map.ipynb
```
