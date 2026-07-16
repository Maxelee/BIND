# Paper II brief — two nuisance parameters marginalize baryons for LSST

**Working title:** The BIND Lightcone Suite II: Two nuisance parameters
suffice to marginalize baryonic feedback in LSST-era cosmic shear
**Format:** full paper (~10–15 pp) + Appendix on the 35-dim cosmology
rescaling. The flagship cosmology result of the suite.

## Thesis
Across a 256-node Sobol sweep of all 30 CAMELS-TNG astrophysical parameters,
the baryonic WL suppression S(ℓ, z_s) occupies a low-dimensional manifold.
Ignoring it biases S8 by up to 6σ for LSST-Y10; marginalizing with exactly
two data-driven templates removes the bias (<0.08σ, held-out validated) at
~3× the statistical σ(S8) floor, while richer BCM-style parametrizations
(N≥6) add cost without benefit. Independently corroborated by Lin+2026's
2-D latent; reproduces and extends the Robertson+2026 LSST forecast setup.

## Sources
1. `docs/WORKLOG.md`: 2026-06-22 (×6: transfer ensemble Step 1; (ΔS8,ΔΩm)
   Step 2; knobs+non-Gaussian Step 3; tomography Step 4 + CORRECTION entry;
   analytic-model span Step 5), 06-24 "capstone — 2 baryon templates",
   plus the beyond2pt/shear-forecast entries (search "Robertson",
   "beyond2pt", "shear_forecast"), 07-03 (rescaling, for the appendix).
2. Worktree `wt/wl-cosmo-bias/examples/`: `lightcone_transfer.py`,
   `lightcone_cosmo_bias.py`, `lightcone_bias_sensitivity.py`,
   `lightcone_bias_tomography.py`, `lightcone_selfcal.py`,
   `lightcone_model_comparison.py`, `cosmo_bias_capstone.py`,
   `lightcone_beyond2pt.py`, `lightcone_shear_forecast.py`,
   `lightcone_shear_sweep.py` (docstrings carry the method details).
3. Appendix sources, worktree `wt/cosmo-rescale`: `docs/cosmo_rescaling_plan.md`,
   `examples/{cosmo_rescale,rescale_validation,rescale_sobol_scan}.py`.

## Section outline
- **Intro:** S8 tension + baryons as the limiting WL systematic; current
  practice (HMcode/BCM marginalization, scale cuts, PCA nuisance approaches
  — Eifler+2015, Huang+2019); the open question = how many nuisance numbers
  does the data actually require; our answer: two, derived from a 30-dim
  feedback design.
- **Methods:** suite recap (cite Paper I); S(ℓ,z_s) measurement w/ paired DMO
  trace (cosmic-variance cancellation, correlation r≈0.98); effective-bias
  pipeline (Fisher/likelihood on (S8, Ωm) with LSST Y1/Y10 specs); template
  construction (PCA of suppression ensemble); held-out validation protocol;
  Robertson+2026-style full likelihood (emcee + CCL).
- **Results:** (1) the transfer-function ensemble: brackets suppression AND
  enhancement (59% of runs enhance at ℓ~1e3); clean z_s dilution; (2) the
  bias budget: up to 6σ at ℓmax=5000 Y10 if ignored; (3) N-template ladder:
  N=1 insufficient (0.67σ residual), N=2 → <0.08σ held-out; cost ~3× σ(S8)
  floor; BCM N≥6 pure cost; (4) full-likelihood confirmation: Y1 N=2 → S8
  bias −0.37σ at 1.11× cost; (5) beyond-2pt: all well-measured statistics
  live on a ≤2–3D manifold; MFs/PDF complementary to κκ; (6) which knobs
  drive the bias; tomographic self-calibration limits (use the CORRECTED
  Step 4 result, not the original claim!).
- **Discussion:** why 2 (link to Lin+2026 latent = Paper III); TNG-only
  amplitude vs general dimensionality caveat; scale cuts vs templates.
- **Appendix A: towards varying cosmology.** AW10 rescaling of TNG300-Dark
  adds (Ωm,σ8,Ωb,h,ns): identity exact; typical SB35 draw rms σ(R) 0.03–2%;
  real-data validation ≤3.4% for mild targets; halofit-level D(k) predicts
  the per-run error point-by-point (quality flag); open corner Ωm=0.1∧σ8=1.0.

## Numbers that MUST appear (verify against WORKLOG)
256-node design / 253 usable runs; r≈0.98 paired-trace; 59% enhancement at
ℓ~1e3; 6σ @ ℓmax 5000 Y10; 0.67σ (N=1); <0.08σ (N=2, held out); ~3× cost;
BCM N≥6; Y1: −0.37σ at 1.11×; manifold ≤2–3D; rescaling: 0.03–2% rms,
≤3.4% validation.

## Figures (candidates)
Disk: `examples/figures_lightcone/` incl. `beyond2pt_manifold.png` and
subdirs `A_binned216/…E_fulltomo/`; `examples/rescale_validation.png` if
present (check main tree `examples/`). Engines named their outputs after
themselves — search `examples/figures_lightcone` listing for transfer/bias/
capstone/forecast keywords. Target 8–10: S(ℓ,z_s) ensemble spaghetti+PCA,
bias scatter (ΔS8 per run), template-ladder residual bias vs N, posterior
corner (Y1/Y10), manifold dimensionality, rescaling validation (appendix).

## Seed citations (verify)
Lin+2026 = arXiv:2509.01881 ("One latent to fit them all" — verify exact
authors/title); Robertson+2026 (LSST baryon marginalization — find the real
reference); Angulo & White 2010; Mead+2015/2021; Schneider+2019 (BCM);
Eifler+2015; Huang+2019 (baryon PCA); Amon & Efstathiou 2022; Preston+2023;
DES Y3 (Amon+2022/Secco+2022), KiDS-1000 (Asgari+2021), HSC Y3; LSST DESC
SRD (2018); CCL (Chisari+2019); emcee (Foreman-Mackey+2013).

## Mandatory caveats
Fixed cosmology (TNG fiducial) — hence Appendix A; TNG-only amplitude (the
*dimensionality* claim is the robust part, per Lin+2026 agreement); ℓ range /
map resolution limits; the Step-4 self-calibration claim was CORRECTED
downward in the 06-22 WORKLOG — use the corrected statement.
