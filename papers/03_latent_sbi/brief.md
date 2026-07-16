# Paper III brief — the 2-D feedback latent + what multiprobe data constrain

**Working title:** The BIND Lightcone Suite III: The two-dimensional latent
space of baryonic feedback, and what lensing×SZ statistics can constrain
**Format:** full paper (~12–16 pp). The ML/inference paper of the suite.

## Thesis
Thirty astrophysical parameters collapse onto a two-dimensional latent space
in every observable we measure: WL suppression is 2-D (98.7% of variance;
BH-driven and SN-driven axes), independently reproducing Lin+2026. The WL
latent shares its leading axis with the kSZ/gas latent (12°) but the second
axis is rotated (38°) — κ responds to a blend of the gas zones kSZ separates.
Simulation-based inference shows WL auto-spectra alone pin the 2-D latent
(not the 30 params); the κ×y cross + tSZ carry the feedback information;
per-halo κ stacks are a feedback-blind mass anchor. Population-level SBI on
8.6M halos gives coverage-calibrated parameter posteriors.

## Sources
1. `docs/WORKLOG.md`: 2026-06-22 ("Sobol latent space + SBI", "SHMR scatter"),
   06-23 (×2: NPE auto-Cl; bind.emulator), 06-24 (×2: "auto-Cl NPE 216 runs +
   all-modes/Nyquist", "WL feedback latents (Lin+2026 for WL) + multi-probe"),
   06-29 (×2: wl_latent_sbi notebook — global response/latent/SBI-rethought;
   §6 lightcone halo population fix).
2. Worktree `wt/sobol-sb35/examples/`: `paper_lightcone_figs2.ipynb` (THE
   figure notebook for this paper), `wl_latent_sbi.ipynb` + `wl_latent_sbi.py`,
   `sobol_latent.ipynb`, `sobol_sbi.ipynb`, `sobol_ml.py`, `wl_stat_latents.py`,
   `wl_feedback_vae.py`, `lightcone_cl_sbi.py`, `sobol_perhalo_sbi.py`,
   `train_perhalo_sbi.py`, `lightcone_halo_{catalog,observables,dkappa}.py`,
   `shmr_accretion_scatter_sb35.ipynb` (optional section, see below).
3. Worktree `wt/wl-tsz-bridge/examples/`: `wl_sz_multiprobe.py`,
   `wl_sz_cosmo_anchor.py`, `wl_sheary_fit.py`, `desact_sheary_realfit.py`
   (the κ×y / shear×y multiprobe + FIRST REAL-DATA fit vs DES Y3 × ACT —
   include as the closing "contact with data" section).

## Section outline
- **Intro:** feedback marginalization needs to know the effective
  dimensionality; latent-space view (Lin+2026); SBI for field-level baryon
  constraints; contributions.
- **Methods:** suite recap (cite Papers I–II); statistics vector (Cl, peaks,
  MFs, PDF, WST, κ×y, y auto); latent extraction (PCA + VAE cross-check);
  canonical correlation between WL and kSZ/gas latents; NPE setup (sbi
  package, logit-space, 5-flow ensemble; emulator-as-simulator with
  calibrated noise; SBC calibration); per-halo population SBI (IID product
  Σlog q + emcee, leave-run-out temperature calibration); model-free
  Spearman global-response maps.
- **Results:** (1) 2-D latent: PCA explained variance 0.884/0.103 (98.7%);
  axis physics (BH vs SN); (2) WL↔kSZ latent geometry: canonical corr
  [0.98, 0.79], principal angles [12°, 38°]; WL tracks inner/outer f_gas
  zones r=0.59/0.71 vs kSZ 0.95; (3) global response: top auto-Cl drivers
  |Spearman| IMFslope 0.49, VariableWindVel 0.48, BHRadEff 0.43, WindEnergy
  0.34; (4) SBI: auto-Cl 30-param posterior broad (~5 params constrained;
  info ~3-dim — 510 modes→3 PCA comps), but the 2-D latent posterior is
  tight (~0.17–0.18 of prior cloud); cautionary: stacking emulated HOS/κ×y
  onto Cl degrades the fiducial fit (2.1σ→5.2σ) — emulator HOS fidelity, not
  physics; (5) κ×y + y dominate feedback info (pin 2 directions to var
  0.06/0.11); per-halo κ stack = feedback-blind mass anchor; (6) halo-level
  anchor: per-halo Born Δκ↔f_gas r=+0.62, imprint peaks z≈0.33; lightcone
  halo population: z=0.034 shell = ~3% of z_s=1 kernel weight yet amplitude
  features proxy at r>0.95 (coherent feedback), redistribution features
  don't (r≈0.47); (7) real-data contact: BIND shear×y vs DES Y3 × ACT.
- **Discussion:** what 2-D means for marginalization (ties to Paper II);
  emulator fidelity as the SBI bottleneck; TNG-only caveat.
- **Optional §/Appendix:** SHMR scatter — joint-feedback σ≈0.39 dex vs
  single-knob 0.16 dex (compounding); accretion history ~10% of residual.
  Include if figures extract cleanly, else move to leftovers.md.

## Numbers that MUST appear (verify against WORKLOG)
All numbers listed in the outline above, plus: 216-run NPE refresh → 253-run
final (state clearly which N each analysis used); Nyquist ℓ≈36864 all-modes
summary ≳ 10-bin ℓ<5k; SBC 68% coverage 0.70 (calibrated).

## Figures (candidates)
Disk: `examples/wl_latent_sbi_figs/` (12), `examples/sobol_sbi_figs/` (9),
`examples/sobol_latent_figs/` (5), `examples/figures_lightcone/B_allmodes_pca/`
+ `A_binned216/` etc. Embedded: `paper_lightcone_figs2.ipynb` (primary),
`wl_latent_sbi.ipynb`, `sobol_latent.ipynb`, `sobol_sbi.ipynb`. Target 10–12:
latent plane + explained variance, axis-physics loadings, WL-vs-kSZ principal
angles, Spearman heat map, NPE posterior corner + latent posterior, SBC ranks,
κ×y information figure, Δκ–f_gas per-halo scatter, §6 population figure,
shear×y real-data overlay.

## Seed citations (verify)
Lin+2026 arXiv:2509.01881; Cranmer+2020 (SBI review); Papamakarios &
Murray 2016 / Greenberg+2019 (NPE/APT); Tejero-Cantero+2020 (sbi pkg);
Talts+2018 (SBC); Cheng+2020 (WST); CCA/canonical correlation standard ref;
Villaescusa-Navarro+2021 (CAMELS); Gatti+2021/Pandey+2022 (DES×ACT shear×y);
Tröster+2021 (KiDS×Planck y); Hill & Spergel 2014; Battaglia+2012 (pressure);
Amodeo+2021. Papers I/II/IV = in prep.

## Mandatory caveats
Emulator HOS/cross fidelity limits stacked SBI (the 5.2σ artifact is a
methods warning, present it as such); TNG-only + fixed cosmology; n=253
design points in 30-D (interpolation, not extrapolation); per-halo map link
is fiducial-only (Sobol runs have no lensplanes).
