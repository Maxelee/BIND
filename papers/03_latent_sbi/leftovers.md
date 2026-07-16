# Leftovers — Paper III (2-D latent + SBI)

Results/figures from the dossier that were deliberately NOT used in
`main.tex`, and why.

- **Per-halo population SBI numeric posterior (8.6M halos).** The brief's
  Thesis line claims "population-level SBI on 8.6M halos gives
  coverage-calibrated parameter posteriors," but per dossier Gap #1 no such
  result exists — the notebook falls back to an explicitly-labeled
  "illustration only, not the science result" demo and no
  `population_posterior.npz` was found on disk. The Methods description
  (amortized NPE, IID-product likelihood, leave-one-run-out temperature
  calibration) is real and is reported in `main.tex` §\ref{sec:methods-perhalo},
  but the quantitative claim itself is dropped and replaced with an
  explicit `\todo{}` in the Abstract, Methods, and Conclusions. Do not
  re-add a numeric population-posterior claim until
  `train_perhalo_sbi.py` has actually been run to completion on GPU.

- **`f2a_effdim.png` as a standalone figure.** Its left panel (linear
  scree, "2 comps = 99% var") duplicates panel (a) of
  `f2b_wl_latent_gaszones.png`, which is used as Figure 1
  (`fig01_latent_axes_zones.png`). Rather than include a near-duplicate
  figure, the nonlinear information-ordered-bottleneck (IOB) cross-check
  from `f2a`'s right panel is described qualitatively in
  \S\ref{sec:methods-latent} ("the nonlinear information curve plateaus by
  the second latent rank") without a dedicated figure, to keep the figure
  count near the brief's 10–12 target while still reporting the
  cross-check.

- **`f1_global_sobol.png` (deprecated Sobol-index global-sensitivity
  figure).** Per dossier Caveat #6 and the `wl_latent_sbi.py` docstring's
  own deprecation note, this metric disagrees with the endorsed Spearman
  ranking (top driver VariableWindVelFactor vs.\ Spearman's IMFslope) and
  is explicitly "kept for reference but no longer drives" the headline
  analysis. The disagreement and its two headline numbers
  ($S_i\approx0.33$, $S_i\approx0.23$) are reported in prose in
  \S\ref{sec:res-spearman} with a caveat, but the figure itself is not
  included, to avoid presenting two competing "the top driver is X"
  figures side by side without a clear resolution.

- **`fig_latent_drivers.pdf` (`examples/figures_lightcone/`).** A second,
  partial-correlation-based axis-loading readout (gas amplitude/ejection/
  gas-DM offset colored latent plane + driver bar chart). The dossier
  notes this figure has no cached output in the source notebook itself
  (only an on-disk PDF from a prior run) and its claims substantially
  overlap with `f2b_wl_latent_gaszones.png` panel (b) and
  `cell026_out1.png`, both of which are already used
  (Figures~1 and~2 in `main.tex`). Dropped to avoid redundancy.

- **`wl_vae_{latent_perturbation,param_corr,latent_params}.png`
  ($\beta$-TC-VAE Lin+2026 Fig.\ 1/3 analogue).** These three figures
  directly visualize the same axis-physics claims already reported via
  `f2b_wl_latent_gaszones.png` (Figure~1) and described in prose in
  \S\ref{sec:methods-latent}/\S\ref{sec:res-latent} (BH axis loading
  $-0.43$, SN/wind axis loading $-0.46$, VAE reconstruction RMSE $1.3\%$).
  Left out of the main figure set to control figure count; a natural
  addition if a referee specifically asks for a side-by-side comparison
  with the Lin+2026 figure layout.

- **`wl_stat_latent_summary.png` / `wl_recoverability_byparam.png`.** The
  underlying numbers (recoverability $\sim5/30$ params stable across
  probe sets; IMFslope $0.33\to0.56\to0.64$; WindEnergy $0.13\to0.33$;
  RadioFeedbackReorient $\to0.16$) are reported in prose at the end of
  \S\ref{sec:res-geometry}, but the two supporting figures themselves are
  not included, again to control total figure count — this material is
  secondary to the CCA geometry result that anchors that subsection.

- **`perhalo_data.png` (per-halo training-data illustration, 60,000 halos,
  $Y_{200}$–$M_{200}$).** Explicitly labeled in the dossier as illustration
  only, not a posterior result. Consistent with dropping the per-halo SBI
  numeric claim (see above), this figure is also left out — including it
  without the corresponding posterior result risked implying the pipeline
  was further along than it is.

- **`sobol_latent_figs/{feedback_dim,active_loadings}.png` (first-generation
  IOB/active-subspace figures, 2026-06-22).** Per dossier Caveat #9, these
  are methodologically superseded by the `wl_latent_sbi.ipynb` generation
  (2026-06-29) used throughout `main.tex`. Not included; the corresponding
  methods (IOB, active-subspace) are described in prose only in
  \S\ref{sec:methods-latent} as cross-checks, without their own figure.

- **`sobol_sbi_figs/{corner_fiducial,calibration,degeneracy_matrices,
  degenerate_directions,joint_degeneracy,loo_recovery,marginals,
  constrained_params}.png` (first-generation MDN/GaussNPE posteriors,
  2026-06-22).** Per dossier Caveat #9, methodologically superseded by the
  later `wl_latent_sbi.ipynb` NPE generation used for all SBI figures in
  \S\ref{sec:res-sbi}. Not included in `main.tex`. The one robust,
  independently-established finding from this generation — an AGN
  BlackHoleRadiativeEfficiency$\leftrightarrow$QuasarThresholdPower
  degeneracy at $+0.27$ — was NOT carried into the text either, since it
  is a secondary cross-check number not called out by the brief and not
  central to any of the seven headline results; flagging here in case a
  future revision wants a "first-generation cross-check" paragraph in
  Methods.

- **SHMR appendix: `cell009_out0.png`, `cell013_out1.png`,
  `cell015_out0.png`/`cell017_out1.png`.** Only `cell011_out1.png` (the
  scatter-budget bar chart carrying the two headline numbers, $\sigma
  \approx 0.386$ vs.\ $0.161$ dex) is used as Figure~13
  (`fig13_shmr_scatter_budget.png`). The relation-and-scatter, accretion-
  regression, and formation-time-colored SHMR figures are left out of the
  appendix to keep it to a single, focused figure; they remain available
  in `figs_raw/shmr_accretion_scatter_sb35/` if a longer appendix is
  wanted later.

- **`g5_desact_confront.pdf` / `g6_kxy_driver.pdf` cross-paper duplication
  with Paper IV.** These two figures are used in `main.tex`
  (Figures~10 and~12) because the brief explicitly assigns the underlying
  engines to this paper, but per dossier Gap #2 they were produced by
  `paper_ksz_field.ipynb`, a notebook conceptually closer to Paper IV.
  This is flagged in-text (\S\ref{sec:discussion}, "A note on shared
  figure provenance with Paper IV") and here for the cross-paper
  coherence pass — Paper IV should not re-publish the identical figures
  without either (a) coordinating which paper is the figure's "home," or
  (b) generating Paper-IV-specific variants.

- **AGN BlackHoleRadiativeEfficiency$\leftrightarrow$QuasarThresholdPower
  degeneracy ($+0.27$).** Mentioned above under the first-generation SBI
  figures; not reported as a standalone result anywhere in `main.tex`
  because it is not part of the brief's seven headline results and its
  primary source generation is superseded. Available for a future
  degeneracy-focused appendix or footnote.
