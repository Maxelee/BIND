# Observable → f_b PoC — **profile edition** results (2026-06-03)

*Branch `analysis/observable-fb-map`. Notebook `profile_fb_map.ipynb` (reviewer-mode
docs), reduction `tools/stack_profiles_reduce.py`, json
`figures/observable_fb/profile_results.json`. **⚠️ Provisional** — emulator-generated
fields (obs *and* f_b are BIND outputs), CAMELS-TNG, fixed cosmology, projected
noise-free stacks. See §0.4 / §6 of the notebook for the full threat list.*

Extends the scalar PoC from per-halo integrated quantities to **stacked radial
profiles**: per (design, mass bin) stack all in-bin halos at the field level → Y(r),
SX(r), thermo(r), and the projected baryon-fraction profile f_b(r)=Σ_b/Σ_tot. Then
predict the f_b **profile** from the observable **profiles**, on held-out feedback
designs (multi-output Ridge, `KFold(5)` over the 256 designs).

## Headline — profile → profile, mean scatter-reduction over radii (held-out θ)
| observable set | logM 13–13.5 | 13.5–14 | 14+ |
|---|---|---|---|
| Y (tSZ) | 69% | 73% | 67% |
| Y, SX | 73% | 77% | 70% |
| Y, SX, T, S, P | 80% | 80% | 75% |

α-robust (mid bin, Y+SX): 79/77/73% for α=1/10/100.

## The profile beats the integrated scalar (target = design-mean f_b)
| input | logM 13–13.5 | 13.5–14 | 14+ |
|---|---|---|---|
| integrated Y (one number) | 54% | 49% | 25% |
| **Y profile** | **79%** | **73%** | **60%** |
| Y + SX profiles | 80% | 74% | 62% |

Shape carries large extra information: going from the single integrated Y to the
Y(r) **profile** lifts the reduction from ~54% to ~79% (low bin) — the central
motivation for the profile extension is borne out.

## What the numbers say
- **At the population/profile level, tSZ Y is already a strong f_b predictor**
  (67–73%), in sharp contrast to the *per-halo integrated* result where Y was weak
  (−9 to −26%). Two compounding reasons: stacking removes per-halo scatter (the
  design-level f_b signal is cleaner), and the **profile shape** disambiguates
  feedback states that a single integrated Y conflates.
- **SX and the full thermo vector still help, but modestly here** (+4–11% over Y),
  because Y is no longer the bottleneck. The full Y,SX,T,S,P reaches 75–80%.
- **Reduction is radius-dependent**, peaking ~85–92% at **r≈100–200 kpc/h**
  (~0.3–0.5 R200, the feedback-active region), and falling to ~35–50% in the
  outskirts (≳1 Mpc/h) where f_b→f_cosmic and the design-to-design variation
  σ_marg(r) is small (little left to predict), and slightly at the very centre.
- **Distinct feedback regimes have distinct f_b(r) shapes** (example figure,
  mid bin): strong-feedback-evacuated cores dip to f_b≈0.087 and rise to ≈0.18,
  weak-evacuation cases stay flat near 0.16 — and the (Y,SX) profile recovers all
  three held-out shapes.
- **The joint distribution view** (§3.5, `prof_joint_Y_fb.png` /
  `prof_joint_vs_r.png`): the 256 designs at fixed (r, mass) are a sample of
  p(f_b(r), Y(r)); the regressor learns its conditional mean E[f_b|Y] and we report
  its conditional width. The Y–f_b joint is **tight at intermediate radii**
  (Spearman ρ=0.94 at 141 kpc/h, mid bin) and loosens in the noisy core (ρ=0.59 at
  57 kpc/h) and the outskirts (ρ=0.28 at 1595 kpc/h, where f_b→f_cosmic and the
  dynamic range collapses) — the joint-correlation-vs-radius *is* the
  reduction-vs-radius curve. A single feedback parameter (SN wind energy) does not
  cleanly order the locus (it is a 30-D feedback space); Y captures the net effect.

## What it might mean
A stacked tSZ (+X-ray) **profile** of a cluster population predicts that
population's baryon-fraction **profile** to ~70–80% scatter reduction on unseen
feedback, with the recovered information concentrated in the core/feedback-active
region. As a forward statement: *the radial shape of the SZ/X-ray signal encodes how
feedback has redistributed baryons, and BIND can read it back out* — modulo the
caveats below.

## Caveats (bounding any claim)
Same as the scalar note plus: (1) obs **and** f_b are emulator outputs →
self-consistency, not hydro truth; the required test is BIND-obs → CAMELS-truth
f_b(r). (2) f_b(r) is a **2-D projected** ratio, not 3-D enclosed. (3) SX is an
uncalibrated ρ²√T proxy. (4) stacks are noise-free/perfectly-centred ⇒ reductions
are an **upper bound**. (5) n=256 designs; fixed-α ridge, α-robustness checked.

## Reproduce
```bash
python tools/stack_profiles_reduce.py     # -> ceph/sobol_ss_cv/stacked_profiles.npz
python tools/build_profile_fb_nb.py        # -> profile_fb_map.ipynb (reviewer-mode docs)
jupyter nbconvert --execute --inplace \
  --ExecutePreprocessor.kernel_name=torch3 --ExecutePreprocessor.timeout=1800 \
  profile_fb_map.ipynb
```
Figures: `figures/observable_fb/prof_{stacks,reduction_vs_r,pred_examples,joint_Y_fb,joint_vs_r}.png`.
