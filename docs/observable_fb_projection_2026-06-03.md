# Observable → f_b PoC — projection / LOS sanity check (2026-06-03)

*Branch `analysis/observable-fb-map`. Notebook `projection_fb_check.ipynb`,
figures `figures/observable_fb/proj_*.png`. **⚠️ Provisional.***

**Question raised:** the stacked f_b(r) exceeds the cosmic value (0.163) at r≳400
kpc/h — impossible for an enclosed baryon fraction. Projection artifact (50 Mpc/h
LOS) or real? Test: the `fm_testsuite_cube` truth (`truth_halos_cube.npz`,
same halos, **6.25 Mpc/h** depth, same 48.83 kpc/h pixel) isolates the LOS effect.

## Findings
1. **The >cosmic differential f_b is real, not projection.** The 6.25 Mpc/h cube and
   the 50 Mpc/h projection give nearly identical differential f_b(r); max LOS effect
   < 0.02 (low bin), < 0.008 (mid), only at r ≳ 600 kpc, and the long LOS pulls the
   profile *toward* cosmic (adds a cosmic-f_b background), never inflates it.
2. **Differential vs enclosed was the confusion.** The *annular* f_b(r) can exceed
   cosmic at intermediate radii — feedback evacuates the core and deposits gas at
   large r. The *enclosed* f_b(<R200) — the missing-baryon metric — stays **below
   cosmic**: 0.11 / 0.15 / 0.16 for the three mass bins (deficit ~33%/9%/2%, largest
   at low mass). No baryon-conservation problem. (`proj_fb_differential_enclosed.png`.)
3. **tSZ Y is LOS-sensitive where mass-f_b is not.** Y∝∫P dl integrates diffuse,
   extended pressure (+2-halo +uncorrelated); the stacked y profile flattens to a
   uniform background (~1.8e-8) at large r, whereas the ρ-concentrated mass maps do
   not. SX∝∫n_e²√T is ρ²-weighted → far less contaminated; mass-f_b least.

## How observers handle tSZ LOS, and our analog
- **Compensated Aperture Photometry (CAP):** disk(θ) − equal-area annulus[θ,√2θ];
  cancels any uniform background/large-scale mode → the canonical stacked tSZ/kSZ
  estimator. Demonstrated in `proj_tsz_cap_demo.png`.
- Matched filters; mean-y + 2-halo modelling; X-ray blank-sky subtraction + Abel
  **deprojection** for 3-D gas mass.
- **Our profile-level analog = mean-column subtraction** (remove Σ_bg, and
  f_cosmic·Σ_bg from baryons) — same idea as CAP. Negligible for mass-f_b (halo
  dominates the column); would matter for Y.

## Implications + next step
- The **target** (mass-based f_b) is LOS-robust → the PoC's predicted f_b(r) is not a
  projection artifact. Report **enclosed f_b(<R200)** for the missing-baryon number.
- The **input observables** Y(r), SX(r) are LOS-integrated; Y especially carries
  2-halo + uncorrelated pressure. A real application should **CAP-filter /
  background-subtract** the observables (SX/mass far less affected). The PoC is
  internally consistent because train and test share the same 50 Mpc/h projection.
- **Cannot yet quantify Y's LOS contamination** — the 6.25 Mpc/h cube products have no
  thermo. **Concrete next data-gen step:** make 6.25 Mpc/h **thermo** cubes (y,T,S,P)
  for the CV halos, then redo the 6.25-vs-50 comparison on Y/SX and rerun the map with
  CAP-filtered observables.

## Reproduce
```bash
python tools/build_projection_nb.py
jupyter nbconvert --execute --inplace \
  --ExecutePreprocessor.kernel_name=torch3 --ExecutePreprocessor.timeout=1800 \
  projection_fb_check.ipynb
```
