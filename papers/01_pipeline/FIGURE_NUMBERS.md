# Paper I figure package — verified numbers crib sheet

Extracted from the EXECUTED papers/01_pipeline/paper1_figures.ipynb (Slurm job **2454934**, 2026-07-30,
1107 s wall, **zero cell errors**, 26 code cells all with outputs, 22 figures written to figs_v2/).
(Job 2454889 produced an identical package except for C^yy, which it computed on the z_s=1 column
instead of the released total column — a wiring slip in the field-cache branch, fixed and re-run.
All C^yy numbers below are the total-column values and match the pre-overhaul release.)
**This supersedes the 2454501 (2026-07-28) extraction**, which predates the figure overhaul: fig02 and
fig17 were deleted, fig04b/fig06/fig09b/fig09c were added, fig03/fig04/fig05/fig07/fig08/fig09 were
rebuilt, and the fig-4 Minkowski-functional error band was re-derived. Numbers that MOVED carry a
parenthetical; see `## superseded` at the end for the old values.

Each line cites its source cell. These are the printed values — quote these, not older markdown.
Regenerate by re-running the notebook; all stamps are computed live.

## dataset
- dataset: emulator_dataset_xpkfix.npz, 253 runs x 30 params, z_s planes [0.5, 1.0, 1.5, 2.0, 2.44], ell 87-52134 (cell 2)

## hero (fig00_hero)
- hero bracket is now a ONE-PARAMETER A_SN1 pair, not a Sobol pair: fiducial realization 0; twobound/run_0000
  (A_SN1=0.9) vs run_0001 (A_SN1=14.4), all 34 other params fiducial (cell 4)
  (CHANGED: the 2454501 hero used Sobol nodes run_0153 / run_0160 bracketed in VarWindVelFactor 3.71/14.73)
- hero mean y fid/low/high: 1.045e-06 / 9.778e-07 / 1.124e-06; node/fid = 0.936 (low) / 1.076 (high)
  (CHANGED from 1.045e-06 / 9.622e-07 / 1.085e-06 and 0.921/1.039 — different bracket, see above) (cell 4)
- hero mean tau (fiducial, total column) = 1.803e-03 (cell 4) (this number used to be stamped by the
  now-deleted fig 2; it is unchanged, only relocated)
- hero kappa rms DMO/painted: 0.0187 / 0.0183 (cell 4)
- hero stretches: kappa norm +-0.1379 (99.9th pct |kappa|); y-ratio norm +-0.764 in log2 (x1.70)
  (CHANGED: the kappa stretch is now a 99.9th-percentile norm, not the old 2-sigma +-0.0374; y-ratio
  widened from +-0.714 / x1.64) (cell 4)

## fig1 (fig01_pipeline_diagram)
- fig1 showcase halo: log10M = 13.79, r200c = 0.66 Mpc/h; runs [58, 145, 56] with
  WindEnergyIn1e51erg = 0.91 / 3.59 / 14.37 (cell 6)

## fig3 (fig03_halo_validation) — scaling relations only
- fig3 median per-halo Y ratio BIND/truth: 1.054 +/- 0.007 (offset significance 8 sigma);
  median f_gas ratio 1.074; median M_star,200c ratio 0.943 (cell 10)
  (the M_star leg is NEW in this run; the tau-profile ratio 1.040 that the old fig-3 line carried has
  moved to fig 4b / fig 13 — see those sections)
- fig3 calibration (pivot logM=13.5, weighted fit to binned medians):
  Delta_Y(logM)     = +5.33% - 10.60%/dex x (logM-13.5)  [binned-median range +11.7% -> -3.4%] (cell 10)
- fig3 calibration: Delta_fgas(logM)  = +7.20% - 6.15%/dex x (logM-13.5)  [binned-median range +10.0% -> +1.7%] (cell 10)
- fig3 calibration: Delta_Mstar(logM) = -5.08% + 5.79%/dex x (logM-13.5)  [binned-median range -0.2% -> -7.8%,
  bootstrap SE 0.8-5.5%] (NEW) (cell 10)
- fig3 post-correction residuals: max|Y| 5.4%, max|fgas| 2.4%; apply as divide BIND Y/f_gas by (1 + Delta(logM)/100) (cell 10)

## fig6 (fig06_radial_profiles) — NEW figure
- fig6 setup: stacked profiles, snap 96 (z = 0.03372); r/r200c = 0.659 * r_cen (exact undo of the reduction
  constant); tau = 2.219785e-14 * Sigma_gas at a = 0.96738 (cell 12)
- fig6 plotted window 0.148-1.855 r200c (12 of 18 radial bins); mass bins 13.40-13.78 (N=728) and
  14.18-14.70 (N=59) (cell 12)
- fig6 mean BIND/truth inside r200c, plotted window (all cached bins r<r200c in parentheses) (cell 12):
  - y      13.40-13.78: 1.038 (1.050);  14.18-14.70: 0.991 (0.998)
  - tau    13.40-13.78: 1.068 (1.076);  14.18-14.70: 1.045 (1.044)
  - M_star 13.40-13.78: 0.867 (0.928);  14.18-14.70: 0.923 (0.968)
  - DM (control, not plotted) 13.40-13.78: 0.996 (1.002);  14.18-14.70: 1.006 (1.004)

## fig4 (fig04_field_validation) — 8 WL panels + noise audit
- fig4 provenance guards: paired spectra source = cache; MF guards V0/V1/V2 bind and truth both
  max|live-ref|/max|ref| = 0.00e+00; MFs on nu=-3..8 (45 thresholds, 5 planes, both sides) read from
  /mnt/home/mlee1/ceph/bind_science/mf_cache/mf_nu8_snap096.npz in 0 s (cell 14)
- fig4 note: bind/nongaussian_stats.npz V0/V1/V2 are SUPERSEDED (differ from the live and per-realization
  values by 3.57e-04 / 2.47e-03 / 5.58e-03) and are NOT used in this figure (cell 14)
- fig4 noise audit nulls: split-half chi2/dof kk 0.87 (truth) / 0.82 (bind), ky 0.57/0.56, yy 0.60/0.60;
  realization-realization rho_bar -0.020 / -0.020 / -0.018 -> N_eff ~ 50; verdict null chi2/dof ~1 (0.6-0.9),
  no band rescaling applied (cell 14)
- fig4 cache/live agreement: cached vs live mean Cl_kk at z_s=1.0, median |ratio-1| = 0.0000% (BIND and truth) (cell 14)
- fig4 PDF nu normalization sigma0 (2'-smoothed per-plane rms):
  BIND [0.00562, 0.01052, 0.01427, 0.01728, 0.01952]; truth [0.00557, 0.01045, 0.01420, 0.01721, 0.01942] (cell 14)
- fig4 median |resid| in ell=300-5000: Cl_kk 0.7%, Cl_ky 1.4%, Cl_yy 5.6% (unchanged) (cell 14)
- fig4 diagonal chi2/dof (seed-paired bands; spectra over ell 100-1.5e4; counts over the plotted nu range
  with truth counts > 5; PDF over bins with nonzero paired scatter; MFs over nu<=4):
  kk 19, S(ell) 19, PDF 128, N_pk 1.1, N_min 1.3, **V0 62.2, V1 35.0, V2 24.4**, ky 120, yy 177 (cell 14)
  (CHANGED: **the MFs only** — V0 0.2 / V1 0.1 / V2 0.2 at 2454501, see the band-recipe note
  below. kk, S, PDF, N_pk, N_min, ky and yy are all UNCHANGED.)
- fig4 MF chi2/dof over the FULL extended range nu=-3..8: **V0 49.2, V1 22.7, V2 16.1** (NEW range) (cell 14)
- fig4 **MF band-recipe change — the cause of the MF chi2 jump**: the v3 band is the genuine paired
  per-realization DIFFERENCE scatter on BOTH sides; the earlier sqrt(2) x BIND-only-scatter proxy for the
  truth-side scatter is retired. Because the traces are seed-paired, the difference scatter is much smaller
  than the proxy, so the band shrinks and chi2 grows. This is a band redefinition, NOT new data or a new
  systematic. (cell 14)
- fig4 peaks/minima/PDF bands = paired per-realization ratio/difference scatter / sqrt(50), recomputed live
  from the cached maps (the v2 band-recipe fix; the earlier quadrature recipe double-divided the truth SE
  by sqrt(50)) (cell 14)
- fig4 PDF residual amplitude: median |res| 0.01%, max |res| 0.41% of peak density — sub-percent in
  amplitude but resolved as significant by the tight paired band (chi2/dof 128) (cell 14)
- fig4 V0/V1/V2 residual amplitude (percent of peak, nu<=8): median |res| 0.00 / 0.02 / 0.12%;
  V0/V1/V2 at nu=8 = 4.44e-04 / 5.07e-05 / 9.15e-06 (the extension carries real signal) (cell 14)
- fig4 full-covariance Hartlap chi2/dof: kk 15 (25 bins, mean |ell-bin correlation| 0.25, N_eff ~4/25);
  ky 101 (24 bins, 0.43, N_eff ~2/24); yy 34 (25 bins, 0.48, N_eff ~2/25) (unchanged) (cell 14)
- fig4 unpaired T2 chi2/dof (truth+bind covariance, no seed cancellation — the scale a map user faces vs an
  independent simulation): kk 0.2, ky 6.7, yy 9.6 (unchanged) (cell 14)
- fig4 MF max normalized residual on the cached nu<=4 grid: V0 0.1%, V2 0.6% (cell 14)
- fig4 shape-noise (ngal=10) peaks: median |resid| 0.8% (cell 14)

## fig4b (fig04b_spectra) — NEW figure (3 autos + 2 crosses of the released field vector)
- fig4b identity anchor: Cl_kappa == S(ell) x Cl_dmo over the 253 Sobol nodes, np.allclose = True,
  max |Cl_kk/(S x Cl_dmo) - 1| = 2.2e-16 — any rho = 1.00 between them in figs 5/7/8 is this identity,
  not a measurement (cell 16)
- fig4b cumulative-column guard: per-plane C^yy at the LAST plane vs the total-column C^yy used for the
  published numbers, median |ratio-1| = 0.000% (cell 16)
- fig4b cross guards: C^ky bind and truth at z_s=1 (5-plane vs the fig-4 paired leg) max |ratio-1| = 0.00e+00 (cell 16)
- fig4b median |resid| in ell=300-5000: Cl_kk 0.7%, Cl_ky 1.4%, Cl_yy 5.6% (total column, released),
  Cl_yy 5.7% (z_s=1 column, plotted); chi2/dof yy(total) 177 vs yy(z_s=1) 63 — these are DIFFERENT
  statistics (the y maps are cumulative per plane; z_s=1 is a 0.22x shallower column) (cell 16)
- fig4b chi2/dof (seed-paired, ell 100-1.5e4): kk 19, ky 120, yy(total) 63, yy(z_s=1) 63 (cell 16)
- fig4b CV-only floor on the y/tau panels (no shape-noise term, fsky=0.44), median/max in the trusted
  range: 0.05% / 2.32% (cell 16) (unchanged value; at 2454501 this was stamped under fig 4)
- fig4b BIND-only tau spectra at ell~1000, l(l+1)C/2pi: 1.325e-07 (tau-tau, column to z=1),
  2.338e-07 (tau-tau, total column), 2.504e-06 (kappa-tau, z_s=1). No seed-paired hydro-truth tau map
  exists in this release — the only tau closure available is halo-level (fig 13; painted/truth = 1.040
  inside r500c) (cell 16)
- fig4b Sobol 253-node 16-84% width at ell~1000: Cl_tautau 18.4%, Cl_kappatau(z_s=1) 13.1%;
  suite-median / fiducial-map amplitude over ell=300-5000 = 0.985 (cell 16)

## fig18 (fig18_highell_localize) — Appendix A, mechanism of the high-ell Cl^yy misfit
- fig18 unmasked high-ell (5e3 < ell < 1.5e4, NR18 = 10 realizations, effect size +/- paired SE):
  Cl_yy +7.2% +/- 1.0%, Cl_ky +11.4% +/- 1.6% (cell 18)
- fig18 p=99.9 (top 0.1% of pixels, 5% of total y flux): field Cl_yy +20.5% +/- 1.1% (Cl_ky +19.5%);
  core Cl_yy -10.7% +/- 1.4% (Cl_ky +0.6%) (cell 18)
- fig18 p=99.5 (top 0.5%, 14% of flux): field Cl_yy +24.0% +/- 1.0% (Cl_ky +22.5%);
  core Cl_yy -2.6% +/- 1.1% (Cl_ky +5.3%) (cell 18)
- fig18 p=99.0 (top 1.0%, 20% of flux): field Cl_yy +25.3% +/- 1.0% (Cl_ky +28.3%);
  core Cl_yy -0.6% +/- 1.0% (Cl_ky +6.2%) (cell 18)
- fig18 p=95.0 (top 5.0%, 42% of flux): field Cl_yy +30.2% +/- 1.1% (Cl_ky +112.8%, numerically unstable
  and omitted from the caption); core Cl_yy +3.9% +/- 0.9% (Cl_ky +7.7%) (cell 18)
- fig18 radial texture (radial_texture_results_FULL.npz, n_halo = 2784 matched group-scale halos,
  M_fof 1.00e13-9.99e13 Msun/h): var ratio 1.290 (<0.5 r200c) / 1.043 (0.5-1) / 0.977 (1-2) / 0.599 (2-4);
  mean-y ratio 1.041 / 0.997 / 0.962 / 0.933; corr(resid) 0.841 / 0.887 / 0.948 / 0.948 (cell 18)
- fig18 smoothing mitigation FALSIFIED (painted y smoothed for R > r200c only, cores untouched):
  <0.5 r200c 1.290 -> 1.290 (1px) -> 1.290 (2px); 0.5-1 1.043 -> 1.043 -> 1.043;
  1-2 0.977 -> 0.900 -> 0.770; 2-4 0.599 -> 0.565 -> 0.501 (cell 18)

## fig11 (fig11_zclosure) — validation across redshift
- fig11 z-drift: median Y ratio 1.054 (z=0.03) -> max 1.304 (z=2.00); f_gas ratio range 1.068-1.142;
  y-profile ratio range 1.039-1.263; drift fit Y ratio ~ a^-0.18 (a single missing a-factor, alpha=-1,
  is excluded) (cell 20)
- fig11 debias template Y_500c (quadratic_a is BEST): r(a) = 1.5628 - 1.1351a + 0.6358a^2
  (+/- 0.0296 / 0.0892 / 0.0647; rms 1.59%, max 4.69%, chi2/dof 1.55 — quoted from the full-precision
  DEBIAS_TEMPLATES dict; the cell's human-readable summary line rounds the same value to 1.6;
  AIC linear 127.0 / quadratic 32.4 /
  powerlaw 61.3, so dAIC vs linear ~94.6) (cell 20)
- fig11 debias template f_gas: r(a) = 1.2139 - 0.2998a + 0.1558a^2 (+/- 0.0099 / 0.0316 / 0.0239;
  rms 0.25%, max 0.59%, chi2/dof 0.49; AIC linear 54.9 / quadratic 14.3 / powerlaw 20.4, dAIC ~40.6) (cell 20)
- fig11 debias template y_profile: r(a) = 1.4428 - 0.9484a + 0.5592a^2 (+/- 0.1160 / 0.3533 / 0.2542;
  rms 3.09%, max 8.83%, chi2/dof 0.21; AIC linear 12.4 / quadratic 9.6 / powerlaw 9.7, dAIC < 3 so the
  quadratic preference is marginal) (cell 20)
- fig11 debias templates apply as BIND_debiased = BIND_raw / r(a), a = 1/(1+z) (cell 20)
- fig11 map-level kappa-y closure per z_s (median resid, ell 300-5000): +1.1% / -0.1% / -0.1% / +0.1% / +0.1%
  at z_s = 0.5 / 1.0 / 1.5 / 2.0 / 2.44 (cell 20)

## fig5 (fig05_param_response) — single 12-statistic x 30-parameter heatmap
- fig5 validity: all 12 statistics 100.00% finite (cl_kappa, suppression, pdf, peak_counts, minima_counts,
  mf_v0, mf_v1, mf_v2, cl_yy, cl_tt, cl_kappa_y, cl_kappa_tau) (cell 22)
- fig5 significance ladder: single-cell 2sigma |rho| = **0.126**; per-statistic permutation null 95th pct
  spans **0.147-0.177**; per-parameter look-elsewhere null (12 rows) = **0.195**; whole-grid null = **0.257** (cell 22)
  (CHANGED: the old "7-family row-max null 0.191" is replaced by the 12-row look-elsewhere null 0.195, and
  the whole-grid null 0.257 is new)
- fig5 per-statistic permutation nulls (95th pct): cl_kappa 0.150, suppression 0.150, pdf 0.155,
  peak_counts 0.177, minima_counts 0.163, mf_v0 0.155, mf_v1 0.147, mf_v2 0.148, cl_yy 0.158, cl_tt 0.159,
  cl_kappa_y 0.153, cl_kappa_tau 0.158 (cell 22)
- fig5 grid 12 statistics x 30 params: max peak |rho| = **0.706 (cl_tt x VarWindVelFactor)**;
  **69/360 cells clear the whole-grid null** (cell 22)
- fig5 parameter counts: **11/30 params clear the per-parameter look-elsewhere null; 14/30 have >=1 cell
  above their own statistic's null** — the caption must not confuse the two (cell 22)
  (CHANGED: the old file's "14/30 super-null, 16 hidden" corresponds only to the second, weaker criterion)
- fig5 column order by peak |rho|: VarWindVelFactor 0.71, IMFslope 0.61, WindEnergy1e51erg 0.54,
  BHRadiativeEff 0.48, WindFreeTravelDens 0.45, RadioFdbkReorient 0.35, QuasarThreshold 0.32,
  VarWindSpecMom 0.30 (cell 22)
  (CHANGED: IMFslope rose 0.53 -> 0.61 and moved from rank 3 to rank 2, ahead of WindEnergy1e51erg)
- fig5 top-3 lever per row (cell 22):
  - cl_kappa: VarWindVelFactor 0.58, IMFslope 0.46, BHRadiativeEff 0.43
  - suppression: VarWindVelFactor 0.58, IMFslope 0.47, BHRadiativeEff 0.43
  - pdf: VarWindVelFactor 0.59, BHRadiativeEff 0.42, IMFslope 0.35
  - peak_counts: IMFslope 0.44, BHRadiativeEff 0.41, WindEnergy1e51erg 0.31
  - minima_counts: IMFslope 0.48, BHRadiativeEff 0.42, VarWindVelFactor 0.33
  - mf_v0: IMFslope 0.59, VarWindVelFactor 0.43, BHRadiativeEff 0.42
  - mf_v1: IMFslope 0.53, BHRadiativeEff 0.48, WindEnergy1e51erg 0.30
  - mf_v2: IMFslope 0.61, BHRadiativeEff 0.42, RadioFdbkReorient 0.32
  - cl_yy: VarWindVelFactor 0.56, WindEnergy1e51erg 0.42, IMFslope 0.37
  - cl_tt: VarWindVelFactor 0.71, WindEnergy1e51erg 0.54, WindFreeTravelDens 0.45
  - cl_kappa_y: VarWindVelFactor 0.59, IMFslope 0.47, WindEnergy1e51erg 0.44
  - cl_kappa_tau: VarWindVelFactor 0.65, WindEnergy1e51erg 0.51, WindFreeTravelDens 0.39
- fig5 identity anchor: the cl_kappa and suppression rows agree to max |delta peak |rho|| = 0.017
  (rank corr 0.996) — S = C_kk / a run-independent DMO trace (cell 22)
- fig5 robustness: mean|rho| (instead of max) column order = VarWindVelFactor, IMFslope, BHRadiativeEff,
  WindEnergy1e51erg, WindFreeTravelDens; rank agreement with the plotted max|rho| order = 0.904 (cell 22)

## fig5 rebin-stability check
- rebin halved: top-1 identical **11/12**; top-2 set 11/12 (ordered 10/12); top-3 set 9/12 (ordered 7/12) (cell 24)
- rebin doubled: top-1 identical **11/12**; top-2 set 11/12 (ordered 10/12); top-3 set 8/12 (ordered 8/12) (cell 24)
  (CHANGED: the check now runs over 12 statistics, not 7/10; the old "top-1 identical 9/10, top-2 10/10,
  only an IMFslope<->BHRadiativeEff swap" is superseded)
- rebin halved, rows whose top-3 moved: peak_counts (IMFslope/BHRadiativeEff swap), minima_counts
  (VarWindVelFactor -> WindEnergy1e51erg in slot 3), mf_v0 (VarWindVelFactor/BHRadiativeEff swap),
  mf_v1 (WindEnergy1e51erg -> RadioFdbkReorient), mf_v2 (RadioFdbkReorient -> VarWindVelFactor) (cell 24)
- rebin doubled, rows whose top-3 moved: pdf (IMFslope -> VarWindSpecMom), minima_counts
  (IMFslope/BHRadiativeEff swap + WindEnergy1e51erg in slot 3), mf_v0 (-> WindFreeTravelDens,
  RadioFdbkReorient), mf_v2 (RadioFdbkReorient -> VarWindVelFactor) (cell 24)

## fig7 (fig07_covariation) — 12 panels, one lever
- fig7 lever: all 12 panels coloured by VariableWindVelFactor (param_names[2], unit-cube range 0.002-0.996);
  top-ranked in 7/12 fig-5 rows (cell 26)
- fig7 per-panel peak |rho| for this parameter: cl_kappa 0.58, suppression 0.58, pdf 0.59, peak_counts 0.24,
  minima_counts 0.33, mf_v0 0.43, mf_v1 0.21, mf_v2 0.24, cl_yy 0.56, cl_tt 0.71, cl_kappa_y 0.59,
  cl_kappa_tau 0.65 (cell 26)
- fig7 per-panel run-to-run 16-84 spread of curve/median (max over plotted bins): cl_kappa 20.5%,
  suppression 22.5%, pdf 197.6%, peak_counts 100.0%, minima_counts 50.0%, mf_v0 2.3%, mf_v1 58.1%,
  mf_v2 54.9%, cl_yy 71.4%, cl_tt 83.6%, cl_kappa_y 75.7%, cl_kappa_tau 104.6%. Cl_kappa and V0 are
  envelope-dominated by construction; their panels are absolute and say so on-panel (cell 26)
- fig7 Y(M) run-to-run scatter per mass bin [dex]: [0.172, 0.132, 0.081, 0.051, 0.032, 0.027, 0.030],
  median 0.051 (cell 26) (the old "0.172 -> 0.027 dex" was the min, not the last bin — the full array is above)

## fig20 (fig20_vandaalen_matrix) — which halos set which scales

- **2026-08-05 split**: fig20_vandaalen_matrix → three standalone figures, same cell:
  `fig20a_vandaalen_matrix` (the r matrix), `fig20b_hinge_gas` (group-bin hinge + fig 12b's stacked-τ gas
  diagnostic, intentionally duplicated across the two figures until a keep-one decision), `fig20c_vandaalen_plane`.
  Old combined pdf/png deleted. The numbers below predate the ELL_TRUST=3e4 bands and the 256-node
  re-assembly; the 2026-08-05 subset render prints best cell ell~1194 (k~1.04) r=0.977 (n=256),
  fiducial S(ell~1194)=0.9941 at f~_bar=0.801.
- **2026-08-05 fig20e added** (`fig20e_analytic_model`, same cell): the analytic latent model
  S(ℓ) = c0(ℓ) + c1(ℓ)f̃_bar + c2(ℓ)f̃_star + c3(ℓ)c_τ, per-bin OLS, deterministic index%5 folds.
  CV-R²(S) at ℓ~1e3/5e3/1.9e4: 3-latent 0.95/0.96/0.93, 2-latent 0.95/0.90/0.85, 30-raw-param linear
  0.51/0.52/0.61 (latents linearize the response). Generation demo (leave-one-out): nodes 81/216/245
  RMS(S) 0.0078/0.0043/0.0097; FIDUCIAL (out-of-design, c_τ,fid=2.15 from bind_tauy_fiducial_snap085)
  RMS 0.0128 vs cloud std 0.057 at ℓ~5e3; PDF RMS 0.0005–0.0010. Cross-stat scorecard (CV-R² | shared-latent
  |r|): pdf 0.85|0.99, mf_v1 0.90|0.81, mf_v2 0.94|0.86, Y–M 0.92|0.90, f_gas–M 0.97|0.89, κτ 0.83|0.68,
  cl_yy 0.76|0.05, T–M 0.48|0.17 (thermal sector = own latent, deferred); peak_counts excluded (spread =
  0.5× per-node err, noise-dominated at Δν=0.5) (cell 28).
- **2026-08-05 fig20f added** (`fig20f_redshift_thermal`, same cell): (a) one low-z latent triplet covers all
  five source planes — 3-latent CV-R² median 0.96/0.95/0.93/0.92/0.91 at z_s=0.5/1/1.5/2/2.44; coefficient
  amplitudes dilute with z_s (max|c1| 0.38→0.22, max|c2| 2.10→1.14). (b) latent evolution (plain-500c,
  uniform defn): rank vs z=0.03 — f̃_star 1.00/1.00/0.99/0.97/0.94 at z=0.18/0.5/0.92/1.41/2.0 (partition set
  EARLY), f̃_bar 0.99/0.92/0.79/0.61/0.44 (budget built LATE); z-epoch latents for z_s=2.44: 0.88(z=0.03)
  →0.90(0.5)→0.70(1.41)→0.47(2.0) — z-matched is WORSE; end-state budget = sufficient statistic. (c) thermal
  4th latent logT̃ (group-bin T_mw_500c; r=−0.39 vs f̃_bar): T–M 0.48→0.78, Y–M 0.92→0.94, PDF 0.85→0.87,
  S(ℓ) unchanged, cl_yy 0.76→0.77 (cluster-bin T/Y/Pe ≤0.78 — y-auto residual = profile-level pressure,
  stated boundary). Full latent set: (f̃_bar, f̃_star, c_τ, logT̃) (cell 28).
- **2026-08-05 (later) fig20c DEMOTED** (author ruling): slimmed to the single literature panel (vD fit + obs
  band + calibrated-f̃ rug, over-closure nodes highlighted); the measured-ΔS panel was redundant with figs
  20b/20d and is gone; fig 20d is the section's main figure. σ(ΔS) and slope-ratio numbers stay printed.
- **2026-08-05 fig20c redesign** (five-point audit; panel (b) described here was later dropped — see above):
  two stacked panels in a SINGLE 3-D-equivalent coordinate —
  (a) literature z≈0 relation (vD+20 fit ±1%, obs band 0.55–0.76) with the Sobol cloud as a rug only (no
  ΔP/P measured); (b) measured ΔS_ℓ (band 0, ℓ∈[300,754], k≈0.26–0.66 at χ*) with own y-scale. Calibration
  applied once to the data: CAL = 1.147 (live truth-atlas cylinder f̃=0.929 in the exact vD bin, 233 halos /
  Nelson+24 3-D 0.81); calibrated cloud f̃ 0.66–0.91 (median 0.83), in-obs-band fraction 0.11 (anchor ±0.03 →
  0.19/0.06); WL response slope +0.0907 vs vD tangent +0.0245 at the median (3.7×, why the y-axes are separate);
  per-node σ(ΔS) = 5.0e-4 from the 50 seed-paired fid/DMO realization pairs = cloud spread /9 (scatter is
  signal); 10/256 f̃^cyl>1 over-closure nodes drawn as open markers; fiducial (vD cell) ΔS=+0.0008 at
  f̃^cyl=0.956 → 3-D-equiv 0.834 (cell 28).
- **2026-08-05 fig20d added** (`fig20d_budget_partition`, same cell): the two halo-level latents of S(ℓ).
  R²[S ~ f̃_bar(13.3–13.6)] = 0.83–0.95 at ℓ≲3000 → 0.20 at ℓ~1.9e4; residual-after-budget PCA: PC1 97.9%,
  PC2 1.8% (one extra latent); residual PC1 vs f̃_star r=−0.90, vs c_τ −0.44, vs f̃_gas +0.51 (f̃_star vs
  S(ℓ~1e3) r=+0.08 — invisible to the amplitude; c_τ vs f̃_bar r=+0.69). Nested R² at ℓ~1.9e4:
  f̃_bar 0.20 → +f̃_star 0.85 → +c_τ 0.94. Top levers — budget: IMFslope −0.48, WindEnergy +0.35,
  BHRadEff +0.28; partition (res-PC1): VarWindVelFactor +0.64, WindFreeTravelDens −0.30, BHRadEff −0.28
  = the quick_deltacl_clusters C2 wind-velocity family. Halo-level restatement of the Lin+2026 2-D latent.
- fig20 binning: mass-bin edges log10 M500c,bg = [13.0, 13.3, 13.6, 13.9, 15.5]; halo counts per node
  median [1267, 660, 337, 184], min [1133, 570, 281, 167]; nodes with valid f~ per bin 253/253 in all bins (cell 28)
- fig20 geometry: ell-band edges [300, 656, 1435, 3137, 6860, 15000], fine bins per band [5, 11, 24, 51, 113];
  chi(z_s=1) = 2301 Mpc/h, chi* = 1150 Mpc/h; ell centers [444, 970, 2121, 4639, 10144] -> k =
  [0.39, 0.84, 1.84, 4.03, 8.82] h/Mpc; vD20 k=0.5 h/Mpc <-> ell = 575 (band 0) (cell 28)
- fig20 Pearson r matrix S(ell-band) x f~_bar(mass-bin), rows ell 444/970/2121/4639/10144:
  [[0.92, 0.955, 0.958, 0.915], [0.952, 0.975, 0.967, 0.917], [0.97, 0.971, 0.946, 0.889],
   [0.936, 0.899, 0.85, 0.79], [0.765, 0.685, 0.619, 0.581]] (cell 28)
- fig20 Pearson r vs f~_gas alone (saturates, sign-flips at the smallest scales):
  [[0.84, 0.803, 0.713, 0.498], [0.846, 0.795, 0.692, 0.464], [0.81, 0.734, 0.607, 0.361],
   [0.678, 0.563, 0.405, 0.14], [0.37, 0.216, 0.036, -0.228]] — max r = 0.846, smallest-scale flip -0.228 (cell 28)
- fig20 R^2 of S per ell band, best single mass bin vs best pair (n = 253 nodes):
  ell~444 single bin2 0.918, pair (0,2) 0.925 (+0.007); ell~970 single bin1 0.951, pair (0,2) 0.957 (+0.006);
  ell~2121 single bin1 0.943, pair (0,2) 0.957 (+0.014); ell~4639 single bin0 0.877, pair (0,1) 0.883 (+0.006);
  ell~10144 single bin0 0.585, pair (0,1) 0.678 (+0.093) — the second bin only helps at ell~1e4 (cell 28)
- fig20 partial r (cluster bin | group bin): -0.102 at ell~2121, -0.188 at ell~10144 (cell 28)
- fig20 best cell: ell~970 (k~0.84 h/Mpc) x mass bin 1 [13.3,13.6): r = 0.975 (cell 28)
- fig20 top-3 levers for S(ell~970): IMFslope -0.47, BHRadiativeEff +0.32, WindEnergy1e51erg +0.32 (cell 28)
- fig20 vD cell (band 0, ell~444, k~0.39): counts median 244 (min 220); f~_bar median 0.955, range 0.755-1.041;
  r(f~_vd, S) = 0.943; WL slope +0.0619 vs vD 3-D tangent +0.0118 at the median; measured dS
  -0.0101..+0.0082 (incl. the S>1 branch); vD prediction over the cloud -0.0065..-0.0012 (cell 28)
- fig20 halo-side rank stability z=0.03 vs z=0.18 (per mass bin): 0.989 / 0.990 / 0.991 / 0.979;
  fiducial S(ell~970) = 0.9975, f~_bar[13.3,13.6) = 0.801 (632 halos) (cell 28)
- fig20 panel (b) caption: r(S(ell~970), f~^cyl_bar in [13.3,13.6)) = 0.975, OLS slope +0.0970;
  aperture = projected R500c, +-25.6 Mpc/h slab, annulus-subtracted — NOT the 3-D spherical f~ of vD+20 (NEW) (cell 28)
- fig20 panel (c) caption: 10/253 nodes have f~^cyl > 1 (cylinder-closure regime, no 3-D counterpart);
  truth calibration f~^cyl = 0.93 <-> 3D f~ = 0.81 (Nelson+24) = x1.16, so the top axis is x1/1.16 and the
  cloud maps to 3D f~ 0.65-0.90 (median 0.82) (NEW) (cell 28)

## fig12 (fig12_survey_context)
- fig12 S(5000) range + enhancement fraction: z_s=0.50 [0.797, 1.211] 0.28; z_s=1.00 [0.839, 1.139] 0.38;
  z_s=1.50 [0.865, 1.110] 0.43; z_s=2.00 [0.882, 1.094] 0.45; z_s=2.44 [0.891, 1.086] 0.46 (cell 30)
- fig12 panel (a) LSST-Y10 sigma(C_ell)/C_ell (Gaussian, single-bin, dlnl=0.15): ell=87 4.57%, ell=305 1.41%,
  ell=964 0.59%, ell=5000 0.52%, ell=19978 2.47% — log bins give mode count ~ ell^2 so CV error ~ 1/ell and
  the band is widest at the largest scales (NEW) (cell 30)
- fig12 central (R < 0.3 r200c) tau, enhancement/suppression branch ratio: 1.77x (cell 30)
- fig12 the hardcoded SCHEMATIC kSZ-informed band (S in [0.80, 0.90], $A_{\rm mod}\sim0.8$) has been
  REMOVED from the figure — it was indicative, not derived from anything on disk (cell 29 markdown).
  Do not quote it; it appears in drafts written against the 2454501 stamp. (SUPERSEDED)

## fig13 (fig13_tauy_profiles)
- fig13 convention guard: beam-on fiducial y-CAP recompute matches the released cache to 0.00%
  (patch grid 0.127'/px, 1517 halos) (cell 32)
- fig13 spans: group-bin (10^13.0-13.4 Msun/h) tau amplitude 0.89 dex across the suite (z=0.18);
  y-CAP 0.35 dex at R=2.9' (logM200c = 13.18 Msun/h, z=0.50); halo-DM sigma spans 398-626 pc/cm^3
  across 177 nodes (cell 32)
- fig13 mass-systematic envelope (+-0.1 dex host-mass shift): fiducial y-CAP moves x[0.83, 0.90] (low) /
  x[1.25, 1.48] (high) across R = 1-6'; hi/lo span 1.74x — selection, not feedback, dominates the absolute
  amplitude (cell 32)
- fig13 beam demonstration: the 1.6' beam suppresses the fiducial y-CAP by 62% at R=1' and 11% at R=2.25';
  mean theta_200c = 1.31' (marked on panel c) (cell 32)

## fig8 (fig08_latent_pca) — now 12 families
- fig8 canonical list: 12 unique families over 13 slots (8 WL field + 3 autos + 2 crosses; C_kk fills a WL
  slot and an auto slot). C_kk and S(ell) are the same family at fixed cosmology (np.allclose = True), so
  their rho1/rho2 = 1.00 is an identity kept as a consistency anchor (cell 34)
- fig8 PC1+PC2 explained variance: Cl_kk 99.1%, S(ell) 99.1%, PDF 92.3%, N_pk 27.1%, N_min 45.0%,
  V0 98.0%, V1 98.7%, V2 97.8%, Cl_yy 98.7%, Cl_tautau 98.4%, Cl_ky 98.4%, Cl_kappatau 99.6% (cell 34)
  (CHANGED: the MFs are now resolved individually (98.0/98.7/97.8) instead of the single concatenated
  "MFs 95%", and N_min 45.0% is newly reported)
- fig8 canonical rho1/rho2 with S(ell): Cl_kk 1.00/1.00 [identity], PDF 0.99/0.73, N_pk 0.99/0.28,
  N_min 0.97/0.14, V0 0.98/0.22, V1 1.00/0.37, V2 1.00/0.38, Cl_yy 0.93/0.87, Cl_tautau 0.96/0.74,
  Cl_ky 0.91/0.31, Cl_kappatau 0.99/0.89 (cell 34)
- fig8 families sharing the SECOND direction with S(ell) (rho2 >= 0.5, C_kk identity row excluded):
  PDF, Cl_yy, Cl_tautau, Cl_kappatau — **4 of 10** genuinely independent families; all others align in the
  leading direction only (cell 34)
  (CHANGED: was "only yy/tt/PDF"; Cl_kappatau (0.89) now joins)
- fig8 MF concatenation control: the v1 concatenated V0|V1|V2 block gives rho1/rho2 vs S(ell) = 1.00/0.08,
  but resolved individually rho2 = 0.22 / 0.37 / 0.38 (null 95th pct 0.08) — concatenation diluted a real
  second direction, it did not measure its absence (NEW; explains the retired "MFs 1.00/0.08" line) (cell 34)
- fig8 plane identification: corr(group f_gas, suppression PC1/PC2) = 0.11 / 0.96 (PC orientation is
  arbitrary; the gas axis lives in the 2-PC plane); corr(PC1, high-ell mean log S) = -1.00 (PC2: -0.09) (cell 34)
- fig8 permutation nulls (pooled over all 66 family pairs): rho_1 median 0.11, 95th pct 0.18;
  rho_2 median 0.03, 95th pct 0.08 (cell 34)
- fig8 count noise/signal variance ratio (median over bins) = 2.9 (peaks) / 5.7 (minima) (cell 34)
  (the minima value 5.7 is newly reported alongside the unchanged peaks value 2.9)

## fig9 (fig09_emulator_validation) — lead panel + demo
- fig9 identity check: np.allclose(C_kk[:,ZI,ZI,:], S[:,ZI,:] * C_DMO[ZI]) = True -> the C_kk head is an
  anchor, not a 15th independent measurement (cell 37)
- fig9 backend: gpgpu emulator, 15 stats, <=12 PCA components each, 203 training runs; loaded from the
  persisted bundle in 1 s; batch prediction of 50 held-out runs in 3.54 s (cell 37)
  (CHANGED: at 2454501 the notebook fit from scratch, 27 s, and predicted in 0.09 s; this run loads a
  persisted bundle and times a cold batched GPU predict, so neither timing is comparable to the old pair)
- fig9 per-head median |frac err| / response R^2 / cov |z|<1, |z|<2 (cell 37):
  - cl_kappa 2.58% / 1.00 / 0.69, 0.91 (NEW head — the identity anchor)
  - suppression 2.66% / 0.72 / 0.69, 0.90
  - peak_counts 0.96% / 0.23 / 0.66, 0.94
  - minima_counts 0.62% / 0.33 / 0.65, 0.93  (NB the fig15 section reports 0.66 for the same
    |z|<1 coverage — a 1-in-the-last-digit self-inconsistency between cells 37 and 45, transcribed
    as printed rather than silently reconciled)
  - pdf 1.26% / 0.73 / 0.70, 0.90
  - mf_v0 0.04% / 0.74 / 0.66, 0.89
  - mf_v1 0.41% / 0.71 / 0.63, 0.89
  - mf_v2 0.74% / 0.71 / 0.64, 0.88
  - cl_yy 11.67% / 1.00 / 0.67, 0.88
  - cl_tt 9.62% / 0.84 / 0.70, 0.91
  - cl_kappa_y 11.48% / 1.00 / 0.67, 0.89
  - cl_kappa_tau 13.54% / 0.78 / 0.75, 0.91
  - cl_yt 13.85% / 1.00 / 0.73, 0.90
  - scaling_Y 4.24% / 0.66 / 0.58, 0.89
  - scaling_f_gas 2.45% / 0.86 / 0.61, 0.92
- fig9 lead/inset: typical held-out run (median of per-run median |frac err|) 1.67%; 95th-pct run 10.7%
  (enhancement branch); suppression S<1 branch 3.20%, S>1 branch 2.15% (cell 37)
- fig9 peak-counts caveat: noise/signal variance ratio ~2.9 per fine nu-bin -> realization-noise dominated;
  achievable response R^2 consistent with ~0, so the measured 0.23 is noise-limited, not a GP failure
  (the x4-rebinned counts of fig 5 do retain a detected |rho| ~ 0.44 response) (cell 37)
- fig9 any-z/any-grid demo: em.predict(theta, z_s=0.75, grids=...) -> S(ell) on 15 custom log-spaced ell
  bins [316, 19953], range [1.007, 1.182]; peak_counts on 25 custom nu bins, shape (25,); z_s echoed back
  0.75; axes['ell'] matches request True (cell 37)

## fig9b (fig09b_emulator_heads) — NEW figure (was fig 9 panels b,c)
- fig9b WL field (8 heads): median |frac err| 0.85%, median response R^2 0.71 (cell 39)
- fig9b SZ/cross spectra (5 heads): median |frac err| 11.67%, median response R^2 1.00 (cell 39)
- fig9b scalings (2 heads): median |frac err| 3.34%, median response R^2 0.76 (cell 39)
- fig9b all 15 heads: median |frac err| 2.58%, median response R^2 0.74 (C_kk is the S(ell) identity
  anchor, not an independent head) (cell 39)

## fig9c (fig09c_emulator_curves) — NEW figure (truth vs emulated, one panel per canonical statistic)
- fig9c setup: 12 canonical statistics (8 WL + 3 auto + 2 cross = 13 roles, 12 unique arrays: C_kk fills a
  WL and an auto slot), z_s = 1.00, 8 held-out runs at S(ell)-error ranks [0, 7, 14, 21, 28, 35, 42, 49]
  of 49 (dataset run ids [180, 1, 252, 123, 72, 141, 5, 105]) (cell 41)
- fig9c panel median |frac err| (head-level all-planes/bins value in parentheses): cl_kappa 1.56% (2.58%),
  suppression 1.64% (2.66%), pdf 1.26% (1.26%), peak_counts 1.06% (0.96%), minima_counts 0.64% (0.62%),
  mf_v0 0.05% (0.04%), mf_v1 0.46% (0.41%), mf_v2 0.86% (0.74%), cl_yy 7.03% (11.67%), cl_tt 5.08% (9.62%),
  cl_kappa_y 5.61% (11.48%), cl_kappa_tau 7.73% (13.54%) (cell 41)
- fig9c positivity precondition for the 5 log-y spectrum panels: cl_kappa min 1.48e-12, cl_yy min 4.86e-21,
  cl_tt min 7.27e-16, cl_kappa_y min 1.23e-17, cl_kappa_tau min 4.72e-15 -> all > 0 over ell <= 2e4 at
  z_s=1.00, so the signed cross-spectra need no symlog at this cut (they would if the cut were widened) (cell 41)

## alpha-hat
- alpha-hat GP inflation (suppression, fitted 2-sigma restore): 1.71 (cells 43, 45)

## coverage (§4b″, 50-node posterior coverage test)
- coverage forward model: gpgpu backend pinned to CPU, 203 runs, suppression only, k=12 PCs, fit in 30 s (cell 43)
- coverage run setup: 50/50 held-out nodes completed (no silent caps), parallel pool, 12 workers (n_cpu=16),
  32 walkers x 6000 steps (burn 25%), **275 s** total, mean acceptance 0.24 (cell 43)
  (CHANGED from 271 s — timing only)
- coverage table 68% / 95% (nominal 0.683 / 0.954): BHRadiativeEff 0.34+/-0.07 / 0.54+/-0.07;
  VarWindVelFactor 0.42+/-0.07 / 0.66+/-0.07; WindFreeTravelDens 0.32+/-0.07 / **0.56**+/-0.07;
  IMFslope 0.38+/-0.07 / 0.64+/-0.07; VarWindSpecMom 0.36+/-0.07 / 0.60+/-0.07;
  BHEddingtonFac 0.42+/-0.07 / 0.72+/-0.06; all-30 average 0.40 / 0.73 (cell 43)
  (CHANGED: WindFreeTravelDens 95% was 0.58)

## fig15 (fig15_emulator_calibration)
- fig15 learning curve (suppression, gpgpu, 16 fits in 19 s): N=50 6.09% (subset spread 0.99%) cov 0.44+/-0.06
  over 5 seeded subsets; N=100 3.93% (0.32%) cov 0.57+/-0.04; N=150 3.66% (0.65%) cov 0.61+/-0.05;
  N=203 3.02% (0.00%) cov 0.68+/-0.00 over 1 subset (cell 45)
- fig15 per-head coverage |z|<1 (nominal 0.68) / |z|<2 (nominal 0.95) / fitted inflation alpha (cell 45):
  cl_kappa 0.69/0.91/1.58 (NEW head); suppression 0.69/0.90/1.71; peak_counts 0.66/0.94/1.06;
  minima_counts 0.66/0.93/1.12; pdf 0.70/0.90/1.46; mf_v0 0.66/0.89/1.30; mf_v1 0.63/0.89/1.33;
  mf_v2 0.64/0.88/1.38; cl_yy 0.67/0.88/1.41; cl_tt 0.70/0.91/1.32; cl_kappa_y 0.67/0.89/1.46;
  cl_kappa_tau 0.75/0.91/1.48; cl_yt 0.73/0.90/1.27; scaling_Y 0.58/0.89/1.27; scaling_f_gas 0.61/0.92/1.16.
  Errors on coverage are +/-0.07 (|z|<1) and +/-0.03 (|z|<2) for every head (cell 45)
- fig15 reading: the 1-sigma shortfall is marginal at run-level errors; the robust signal is the 2-sigma
  deficit (heavy tails, dominated by the enhancement corner) — COHERENT across the 15 heads (they share the
  same 50 held-out runs), not 15 independent detections (cell 45)

## fig10 (fig10_astro_corner)
- fig10 GP: suppression-only gpgpu GP on all 253 runs, k=12 PCs, fit in 1 s (device cuda) (cell 47)
- fig10 likelihood: 25 bins from 50 realizations; Hartlap factor 0.47; mean |off-diag correlation| 0.74 (cell 47)
- fig10 Percival/DS parameter-variance factor (n=50, p=25): m1 = 1.87 (n_par=6) to 2.05 (n_par=2);
  quoted 1sigma widths are PRE-correction — multiply by sqrt(m1) = 1.37-1.43 (cell 47)
- fig10 chain: emcee full-cov 128 x 10000 (~53 tau_int) in **126 s**; acc 0.16; tau_int mean/max 167/188;
  ESS ~6312; 25728 samples; timed trial 200 steps x 128 walkers in 2.4 s -> 12 ms/step (cuda) (cell 47)
  (CHANGED from 128 s — timing only)
- fig10 posterior/prior widths, top 10 (* = fiducial on prior edge, excluded from corner):
  VarWindVelFactor 0.43, BHRadiativeEff 0.43, IMFslope 0.47, WindFreeTravelDens 0.50,
  WindEnergy1e51erg 0.59, QuasarThreshold 0.60, VarWindSpecMom* 0.60, BHAccretionFac 0.60,
  SeedBlackHoleMass 0.65, BHEddingtonFac 0.66 (cell 47)

## fig16 (fig16_inference_robustness)
- fig16 variant chains (128 x 10000 each) in **362 s**: diag acc 0.12 ESS 4122; conservative acc 0.11
  ESS 4548; GPx1.71 acc 0.19 ESS 7812 (cell 49)
  (CHANGED from 368 s — timing only)
- fig16 median top-6 width ratios: diag/full 1.10, conservative/full 1.45, GPx1.71(fitted)/full 0.96 (cell 49)
- fig16 bootstrap-over-walkers SE on the top-6 width ratio: diag/full 0.011 (1.0% relative),
  conserv/full 0.014 (1.0%), GPxalpha/full 0.009 (0.9%) (cell 49)
- fig16 posterior-predictive chi2/dof (diag) = 0.05 (<<1 is EXPECTED: 30 params vs 25 correlated bins,
  GP variance in the likelihood but not this denominator — quoted only to show the posterior mean threads
  the data, not as goodness-of-fit) (cell 49)

## fig14 (fig14_completeness)
- fig14 FoF band 1e12-1e13: 25617 halos (catalog read 1 s) (cell 51)
- fig14 capture fractions (halo COUNT): 28.7% at 4xR200 / 51.6% at full patch of the 25,617 band halos;
  the live recompute matches the engine reference exactly (28.7% / 51.6%) (cell 51)
- fig14 captured-halo fidelity: 13,226 captured halos 1e12-1e13; median gas ratio 1.034, y ratio 0.986 (cell 51)

## fig19 (fig19_fgas_erosita, Appendix C)
- fig19: fiducial group f_gas(<R500c) = 0.0807 (inside the X-ray-standard band 0.06-0.10); 57 1P runs span
  0.0515-0.1348; strongest single knob (WindFreeTravelDensFac, SN/wind) closes 53% of the fiducial->eROSITA-low
  gap; NO single knob reaches eROSITA-low (0.026); the 30-dim Sobol design does not reach it either
  (0/256 nodes — companion Paper IV, external) (cell 54)

## other
- release check: emulator_dataset_xpkfix.npz sha256 67d6a8c3e2c6fa7e..., 126 MB, 253 runs, 23 stat targets;
  map coverage 253/253 listed runs have kappa_maps.npz on disk (cell 56)

## superseded
Values from the 2454501 (2026-07-28) extraction that this run REPLACES. If a draft still quotes any of
these, it is out of date.

- **fig4 MF chi2/dof (nu<=4): V0 0.2, V1 0.1, V2 0.2 -> V0 62.2, V1 35.0, V2 24.4.** Cause is a BAND-RECIPE
  change, not new data: the old band used sqrt(2) x BIND-only scatter as a proxy for the truth-side scatter;
  the new band uses the genuine paired per-realization difference scatter on BOTH sides. Because the traces
  are seed-paired, the difference scatter is much smaller than the proxy, so the band shrinks and chi2 grows.
  A new full-range set (nu=-3..8) is also reported: V0 49.2, V1 22.7, V2 16.1. (cell 14)
- fig4 Cl_yy diagonal chi2/dof: 177 -> 63 (cell 14)
- fig4 Cl_yy full-covariance Hartlap chi2/dof: 34 -> 12; mean |ell-bin correlation| 0.48 -> 0.45 (cell 14)
- fig4 Cl_yy unpaired T2 chi2/dof: 9.6 -> 4.3 (cell 14)
- fig4 CV-only floor line (median 0.05% / max 2.32%) is now stamped by fig 4b, not fig 4 (value unchanged) (cell 16)
- fig5 "7-family row-max null 0.191" -> per-parameter look-elsewhere null (12 rows) 0.195, plus a new
  whole-grid null 0.257 and a new count 69/360 cells clearing it (cell 22)
- fig5 "14/30 params super-null, 16 hidden" -> 11/30 clear the per-parameter look-elsewhere null;
  14/30 have >=1 cell above their own statistic's null (two different criteria) (cell 22)
- fig5 IMFslope peak |rho|: 0.53 (rank 3) -> 0.61 (rank 2, ahead of WindEnergy1e51erg) (cell 22)
- fig5 rebin stability: "top-1 identical 9/10, top-2 set 10/10, only an IMFslope<->BHRadiativeEff swap"
  -> 11/12 top-1 and 11/12 top-2 in both directions, with five (halved) / four (doubled) rows changing
  their top-3 (cell 24)
- fig8 "PC1+PC2: MFs 95%" (concatenated block) -> V0 98.0%, V1 98.7%, V2 97.8% resolved individually;
  N_min 45.0% newly reported (cell 34)
- fig8 "rho1/rho2 MFs 1.00/0.08" -> that is now reported as the concatenation CONTROL; the resolved values
  are V0 0.98/0.22, V1 1.00/0.37, V2 1.00/0.38 (cell 34)
- fig8 "rho2 >= 0.5 only yy/tt/PDF" -> PDF, Cl_yy, Cl_tautau, Cl_kappatau (4 of 10) (cell 34)
- fig9 timings "fit 27 s, 50-run batch predict 0.09 s" -> bundle load 1 s, batch predict 3.54 s (different
  code path: persisted bundle + cold batched GPU predict) (cell 37)
- hero: Sobol bracket run_0153 (VWV 3.71) / run_0160 (VWV 14.73) -> one-parameter A_SN1 bracket
  twobound/run_0000 (0.9) / run_0001 (14.4); mean y low/high 9.622e-07/1.085e-06 -> 9.778e-07/1.124e-06;
  node/fid 0.921/1.039 -> 0.936/1.076; kappa stretch +-0.0374 (2 sigma) -> +-0.1379 (99.9th pct);
  y-ratio stretch +-0.714 (x1.64) -> +-0.764 (x1.70) (cell 4)
- coverage: WindFreeTravelDens 95% 0.58 -> 0.56; run wall 271 s -> 275 s (cell 43)
- fig10 chain wall 128 s -> 126 s; fig16 variant-chain wall 368 s -> 362 s (timings only) (cells 47, 49)
- fig7 "Y(M) scatter 0.172 -> 0.027 dex" was the min, not the last bin; the full per-bin array is
  [0.172, 0.132, 0.081, 0.051, 0.032, 0.027, 0.030] (cell 26)
- **fig2 (fig02_map_suite) and fig17 (fig17_feedback_sky) no longer exist.** Their stamped numbers are gone:
  fig2's "kappa rms 0.0225, mean tau 1.803e-3, mean y 1.045e-6" and stretches (kappa +-0.0367, tau log
  2.08e-4 - 6.24e-3), and fig17's "mean-y ratio fid/weak 1.09, fid/strong 0.96; kappa-rms ratio 0.989/1.005".
  The surviving equivalents are the hero stamps (cell 4): mean tau 1.803e-03, mean y 1.045e-06,
  kappa rms DMO/painted 0.0187/0.0183, node/fid mean-y 0.936/1.076.
