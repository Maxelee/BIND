# Observable → f_b PoC — CAP-filtered real-data validation (2026-06-03)

*Branch `analysis/observable-fb-map`. Notebook `profile_fb_cap_validation.ipynb`,
json `figures/observable_fb/cap_results.json`, figs `cap_*.png`. **⚠️ Provisional.***

Re-runs the BIND→CAMELS-truth validation using **Compensated Aperture Photometry
(CAP)**-filtered observables — `CAP(θ)=mean(disk r<θ)−mean(annulus θ<r<√2θ)` — instead
of the raw stacked Y(r), SX(r). CAP nulls any uniform LOS background exactly and is the
estimator a real tSZ/X-ray stack must use (it cannot measure the raw zero-point). CAP is
a linear, radius-only functional, so it is computed exactly from the stored azimuthal
profiles + per-bin pixel counts (no re-reduction). Target f_b(r) unchanged (mass-based,
LOS-robust).

## Result 1 — the validation survives the realistic estimator
Noise-free `RMS|pred−truth|` in f_b (×10⁻³), train on BIND → predict CAMELS truth:

| features | 13.0–13.5 | 13.5–14 | 14+ |
|---|---|---|---|
| raw Y, SX | 4.0 | 2.4 | 5.0 |
| **CAP Y, SX** | **4.7** | **5.0** | **6.7** |

CAP still recovers truth f_b(r) to ~0.005–0.007 (~5–7% of the 0.08–0.18 range) and tracks
the evacuated-core + rise shape (`cap_pred_vs_truth.png`). It is **modestly less accurate
than raw** because CAP discards the absolute zero-point / large-scale modes — information
the raw-profile run had but a **real observation could not measure**. So "CAP ≈ raw to
within ~2×10⁻³" is the meaningful statement: the result does **not** depend on an absolute
Y normalisation an observation can't provide, which makes it *more* credible for real data,
not less. Most of CAP's deficit is at the innermost radius (nulling the background also
removes the central absolute level).

## Result 2 — CAP is NOT more noise-robust (hypothesis overturned)
Median per-measurement `RMS|pred−truth|` (×10⁻³) over 300 log-normal noisy draws
(honest metric — not RMS-of-mean, which understates error at high noise):

| bin | method | 0% | 5% | 10% | 20% | 40% |
|---|---|---|---|---|---|---|
| 13.0–13.5 | raw | 4.0 | 5.6 | 9.6 | 18.2 | 34.1 |
|           | CAP | 4.7 | 6.5 | 10.9 | 17.5 | 37.3 |
| 13.5–14   | raw | 2.4 | 3.8 | 6.2 | 11.8 | 24.3 |
|           | CAP | 5.0 | 5.8 | 6.9 | 13.1 | 30.1 |
| 14+       | raw | 5.0 | 5.0 | 6.5 | 10.8 | 20.8 |
|           | CAP | 6.7 | 6.7 | 8.8 | 19.2 | 37.2 |

Per-measurement error grows **~linearly with noise for both**, and CAP is **slightly
worse throughout** — CAP is *not* the more noise-stable estimator here. Why: CAP buys
robustness against the *background/zero-point*, not against *per-bin measurement noise*;
the mass-based f_b target is already LOS-robust and the whole-profile Ridge fit averages
noise either way. At ≳20% per-bin noise the error approaches σ_marg(r)≈0.02 — the
prediction loses most of its skill (the practical S/N requirement for a real stack).

*(An earlier draft used RMS-of-the-mean-prediction and spuriously suggested CAP was more
noise-robust; that was a regression-to-the-mean artifact and is corrected here.)*

## Verdict
Using the **realistic, background-nulled CAP observables**, BIND's map still predicts
CAMELS-truth f_b(r) to ~0.005–0.007 — the validation **holds under the estimator a real
survey is forced to use**, at a modest accuracy cost vs the (unmeasurable) raw zero-point.
CAP does not add per-bin-noise robustness. Useful at ≲10% stacking noise; degrades toward
σ_marg by ~20%.

## Caveats (unchanged)
Single feedback point (interpolation), shared DMO halos, projected mass-based f_b,
uncalibrated ρ²√T SX, uniform toy noise, and CAP computed from the 50 Mpc/h projection —
a fully realistic Y-LOS test still needs 6.25 Mpc/h thermo cubes.

## Reproduce
```bash
python tools/build_cap_validation_nb.py
jupyter nbconvert --execute --inplace \
  --ExecutePreprocessor.kernel_name=torch3 --ExecutePreprocessor.timeout=1800 \
  profile_fb_cap_validation.ipynb
```
Figures: `figures/observable_fb/cap_{truth_profiles,pred_vs_truth,noise_robustness}.png`.
