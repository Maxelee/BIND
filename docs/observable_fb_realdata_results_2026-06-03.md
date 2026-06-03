# Observable → f_b PoC — **real-data validation** results (2026-06-03)

*Branch `analysis/observable-fb-map`. Notebook `profile_fb_realdata.ipynb`, truth
reduction `tools/stack_profiles_truth.py`, json
`figures/observable_fb/realdata_results.json`. **⚠️ Provisional.***

The first test against **CAMELS hydro truth** (not BIND outputs). The map
`f_b(r) ← (Y,SX,…)(r)` is **trained on the 256 BIND/Sobol feedback designs** and
applied to the **27-sim CV** set (fiducial cosmology+feedback, 1111 halos), stacked
from the *hydro-truth* fields with the identical geometry. Truth observables and
truth f_b were never used to fit anything.

## Setup validation
- Truth mass patches cut from `full_maps.truth_maps` with a crop convention
  validated to reproduce the stored DMO `condition` patch to **corr=1.0000**.
- Truth Y(r), SX(r), f_b(r) all land **100% inside** the BIND design 1–99% cloud
  per radius → this is an **interpolation** to the centre of the design (the
  favourable regime; CV is a single feedback point).

## Result 1 — the relation transfers (domain check)
`real_truth_on_joint.png`: the CAMELS-truth (Y, f_b) point at r=259 kpc/h (mid bin)
sits **on the BIND-learned conditional mean** E[f_b|Y], within the 256-design cloud
and within its small cosmic-variance bars. The relation BIND learned across feedback
space holds for real hydro.

## Result 2 — predicted vs truth f_b(r), noise-free
`RMS|pred − truth|` in f_b (×10⁻³), over radii, per mass bin:

| features | 13.0–13.5 | 13.5–14 | 14+ |
|---|---|---|---|
| Y | 10.8 | 4.3 | 5.7 |
| **Y, SX** | **4.0** | **2.4** | **5.0** |
| Y, SX, T, S, P | 5.2 | 2.8 | 16.2 |

For scale: truth f_b spans ~0.08–0.18 (range ~0.10) and the BIND feedback scatter is
σ_marg(r) ≈ 0.02–0.03. So **Y+SX predicts real hydro f_b(r) to ~0.004 RMS** — ~4% of
the f_b range, ~15–20% of σ_marg. `real_pred_vs_truth.png` shows the prediction
recovers the **truth-specific evacuated core** (f_b dips to 0.08 at ~100 kpc in the
low bin), i.e. it uses the truth observables — it is **not** regressing to the BIND
design-mean (whose core is shallower, ~0.12).

## Result 3 — robustness to observational noise
Injecting multiplicative log-normal noise per radial bin on Y and SX (300-draw MC),
`RMS|pred − truth|` for Y+SX stays essentially flat to **20%** noise
(3.98→3.11 / 2.42→2.51 / 4.97→4.79 ×10⁻³). The 16–84% prediction band widens with
noise but the mean keeps tracking the truth core+rise shape. The map leans on the
overall amplitude/shape of the stacked profile, which averages down per-bin noise.

## Result 4 — more features transfer *worse* out-of-distribution
The full Y,SX,T,S,P vector — which **won in-distribution** (BIND→BIND, §profile
notebook) — transfers **worse** than Y+SX on truth, badly so in the sparse top bin
(16.2 vs 5.0 ×10⁻³). The richer feature set overfits the BIND→truth domain gap
(thermo channels carry more emulator-specific structure; 51 halos/bin is noisy).
**Takeaway: Y+SX is the robust real-data feature set; do not chase the richer model.**

## Verdict
Within its scope, the map **passes the real-data test**: BIND's learned
observable→f_b relation predicts actual CAMELS hydro baryon-fraction profiles from
actual hydro tSZ+X-ray profiles to ~0.004 in f_b, robust to ~20% stacking noise,
recovering the real evacuated-core shape. **Caveats that still bound any claim:**
single feedback point (interpolation, not extrapolation or off-grid); the 1111 test
halos share DMO structure with training; projected (not 3-D) f_b; uncalibrated ρ²√T
SX; uniform-in-radius toy noise. Next: off-grid (SIMBA/Astrid/real instruments) and
held-out-halo tests.

## Reproduce
```bash
python tools/stack_profiles_truth.py   # -> ceph/sobol_ss_cv/truth_stacked_profiles.npz
python tools/build_realdata_nb.py       # -> profile_fb_realdata.ipynb
jupyter nbconvert --execute --inplace \
  --ExecutePreprocessor.kernel_name=torch3 --ExecutePreprocessor.timeout=1800 \
  profile_fb_realdata.ipynb
```
Figures: `figures/observable_fb/real_{truth_on_joint,pred_vs_truth}.png`.
