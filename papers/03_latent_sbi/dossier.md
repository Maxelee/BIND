# Dossier — Paper III: The two-dimensional latent space of baryonic feedback

Mined from `docs/WORKLOG.md` (2026-06-22/23/24/29 entries), worktrees
`wt/sobol-sb35/examples/` and `wt/wl-tsz-bridge/examples/` (engine docstrings +
embedded notebook figures), and on-disk gitignored figure directories in the
main tree (`examples/wl_latent_sbi_figs/`, `examples/sobol_sbi_figs/`,
`examples/sobol_latent_figs/`, `examples/figures_lightcone/`,
`examples/figures_field/`). All numbers below carry a provenance line.

Design-size bookkeeping (brief requires this be explicit): the SB35 Sobol
sweep is a fixed **256-node** design over the 30-dim astro prior, same shared
DMO halos. Different analyses ran at different points as the suite grew:
123 valid runs (`lightcone_cl_sbi.py`, 2026-06-23) → 216 completed runs
(auto-Cl NPE refresh + WL feedback-latent β-VAE, 2026-06-24) → 237-run
`bind.emulator` rebuild (2026-06-24, same session) → **253 usable runs**, the
final state used by `wl_latent_sbi.ipynb`/`paper_lightcone_figs2.ipynb`
(2026-06-29 onward; explicit "`n=253`" label in the Spearman-heatmap figure).
State the N used for every number below.

---

## Intro (framing only — no numbers to source)

- Thesis: feedback marginalization needs the effective dimensionality of the
  response; latent-space view following Lin+2026; SBI for field-level baryon
  constraints. No quantitative claims in this section beyond what Results
  states below.

---

## Methods

- **Suite recap** (cite Papers I/II, in prep): 256-node Sobol design, 30 astro
  params (SB35), fixed TNG cosmology, same shared fiducial halos painted under
  every design point. *Provenance: WORKLOG 2026-06-22 "SB35 Sobol latent space
  + SBI", 2026-06-21 "SB35 Sobol lightcone pipeline (256 runs)".*
- **Statistics vector.** `bind_sb35/emulator_dataset.npz` aggregates per-run,
  per-source-plane (5 planes) tomographic `Cl_κκ`, `Cl_κy`, `Cl_yy`, `Cl_ττ`
  (suppression `S(ℓ)=Cl/Cl_dmo` where applicable, shared-DMO denominator
  cancels cosmic variance), peak/minima counts, κ PDF, Minkowski functionals
  V0/V1/V2, WST (kymatio 2-D scattering, needs an `sph_harm` shim for
  scipy≥1.13), and binned scaling relations (Y–M, f_gas–M, T–M).
  *Provenance: WORKLOG 2026-06-23 "`bind.emulator`"; `wl_latent_sbi.py`
  docstring (sobol-sb35 worktree).* **Caveat:** WST is excluded from the
  `wl_stat_latents.py` cross-statistic latent-alignment test — only 40 valid
  runs at that point. *Provenance: WORKLOG 2026-06-24 "WL feedback latents...".*
- **Latent extraction — two independent methods, consistent result.**
  (a) Linear PCA on `log10 S(ℓ,z_s)` (or other binned statistics), truncated
  at 2 components. (b) A conditional β-TCVAE (Chen+2019 total-correlation
  decomposition, full-batch, KL annealing) → disentangled 2-D latent, RMSE
  1.3% on the reconstructed suppression. *Provenance: `wl_feedback_vae.py`
  docstring + WORKLOG 2026-06-24 "WL feedback latents (Lin+2026 for WL) +
  multi-probe test".* Also cross-checked against a nonlinear
  Information-Ordered Bottleneck (IOB, torch MLP, information curve
  `R²(k)`) and an active-subspace decomposition (`E[JᵀJ]` eigendecomp of a
  θ→x surrogate Jacobian) in `sobol_ml.py`. *Provenance: WORKLOG 2026-06-22
  "SB35 Sobol latent space + SBI".*
- **Canonical correlation (WL ↔ kSZ/gas latents).** `align_planes` in
  `wl_latent_sbi.py`: canonical correlation analysis (`sklearn.cross_decomposition.CCA`)
  between each statistic's top-2 PCA latent and the κ-Cl top-2 latent, plus
  the principal angle between the two 2-D planes. Reuses the kSZ §4 latent
  kernel from `paper_ksz_desi_act.ipynb`. *Provenance: `wl_latent_sbi.py`
  docstring §2; `wl_stat_latents.py` docstring.*
- **NPE setup.** `sbi`-package neural posterior estimation. Two generations:
  (i) `lightcone_cl_sbi.py` — direct NPE on the n=123→216 measured Sobol runs
  (data-starved at these N); bounded `[0,1]^30` prior trained in **logit
  space** (avoids rejection-sampling stall, acceptance ~0.8³⁰≈1e-3 otherwise),
  a **5-flow ensemble** pooled for stability (a single flow's VarWindVel
  shrink varies 0.97–1.12 across draws). (ii) `wl_latent_sbi.py` §3 —
  **`EmulatorSimulator`**: the `bind.emulator` GP surrogate used as an
  unlimited (θ→x) generator with calibrated noise, escaping the n=253
  data-starvation; NPE run + an explicit-likelihood `emcee` cross-check;
  `sbc_ranks` for simulation-based calibration (Talts+2018-style rank
  histograms). Target = the *measured* fiducial (`bind_science/runs/{bind,truth}/run_0000`,
  θ known). *Provenance: WORKLOG 2026-06-23 "NPE on the WL auto-Cl", 2026-06-24
  "auto-Cl NPE on 216 runs", 2026-06-29 "wl_latent_sbi.ipynb"; engine docstrings
  `lightcone_cl_sbi.py`, `wl_latent_sbi.py`.*
- **Per-halo population SBI.** `sobol_perhalo_sbi.py` + GPU trainer
  `train_perhalo_sbi.py`: amortized NPE `q(θ|x_halo)` trained on ~8.6M
  per-halo `(θ, log M200, a, log Y200)` tuples (same painted DMO halos, 256
  design points × 20 redshifts); population posterior via the IID product
  `log p(θ|{x_i}) = Σ_i log q(θ|x_i) + const` (flat prior on the unit box),
  sampled in 30-d with `emcee`; honesty check = **leave-one-run-out
  coverage**, with a likelihood **temperature** `T` (effective `1/N_eff`)
  calibrated so the leave-out 68% band actually covers 68%.
  *Provenance: `sobol_perhalo_sbi.py` + `train_perhalo_sbi.py` docstrings.*
  **⚠ GAP (see Gaps list): this pipeline was never run to completion on
  GPU** — see Caveats/Gaps below.
- **Model-free global-response maps.** `bin_param_correlation` in
  `wl_latent_sbi.py` §1: per-bin **Spearman** rank correlation of each
  measured statistic against each of the 30 params, read directly off the
  253 measured Sobol runs — *deliberately emulator-free* (the emulator is
  only percent-accurate for the suppression amplitude, not trusted to rank
  sensitivity). *Provenance: `wl_latent_sbi.py` docstring §1; WORKLOG
  2026-06-29.*

---

## Results

### (1) The 2-D latent

- **PCA explained variance: 0.884 / 0.103 → 98.7% in 2 components** (WL
  suppression `S(ℓ,z_s)`, β-TCVAE/PCA on the 216-run SB35 suite).
  *Provenance: WORKLOG 2026-06-29 "wl_latent_sbi.ipynb" §2; figure
  `examples/wl_latent_sbi_figs/f2a_effdim.png` and
  `f2b_wl_latent_gaszones.png` panel (a), both titled "2 comps = 99% var"
  (bars read ≈0.89/≈0.10).*
- A closely related **independent** computation on the final 253-run Sobol
  sweep (embedded in `paper_lightcone_figs2.ipynb` §5.1, `fig_latent_plane`)
  gives PC1≈89%, PC2≈11%, cumulative **99.2%** — consistent with, but not
  numerically identical to, the 0.884/0.103 figure above (different exact
  ℓ-cut/binning). *Provenance: `paper_lightcone_figs2.ipynb` cell 22 (§5.1);
  extracted PNG `figs_raw/paper_lightcone_figs2/cell022_out1.png`.* **Use
  the WORKLOG 0.884/0.103 (98.7%) number as the headline** per the brief;
  note the second measurement as a consistency check, not a correction.
- **Axis physics.** β-TCVAE: **Latent 0 = BH axis** (BlackHoleRadiativeEfficiency
  loading −0.43, plus QuasarThreshold, BH accretion/Eddington), acts
  ~uniformly across source redshift; **Latent 1 = SN/wind axis**
  (VariableWindVelFactor loading −0.46, plus WindEnergy, IMFslope), evolves
  with `z_s` — reproduces Lin+2026's Fig. 1 & 3 scale/time split for WL.
  *Provenance: WORKLOG 2026-06-24 "WL feedback latents (Lin+2026 for WL)";
  figures `examples/figures_lightcone/wl_vae_{latent_perturbation,param_corr,
  latent_params}.png`.*
- A second, independent axis-loading readout (`f2b_wl_latent_gaszones.png`
  panel b) ranks driver params by |loading| on ê1/ê2: VariableWindVelFactor,
  BlackHoleRadiativeEfficiency, IMFslope, WindFreeTravelDensFac,
  RadioFeedbackReorient, WindEnergyIn1e51, QuasarThreshold,
  BlackHoleAccretionFactor — same physical sector (wind/SN + BH), consistent
  ordering with the β-TCVAE loadings. *Provenance:
  `examples/wl_latent_sbi_figs/f2b_wl_latent_gaszones.png`.*
- **Physical readout of the latent plane** (§6.1 `fig_latent_colored`,
  253-run Sobol, correlations `r1` = latent-1, `r2` = latent-2 read directly
  off the figure): `f_gas` r1=+0.26/r2=+0.94; `f_star` r1=+0.56/r2=−0.69; `T`
  r1=−0.33/r2=+0.05; `Y` r1=+0.27/r2=+0.93; gas concentration r1=+0.58/r2=+0.78;
  DM concentration r1=+0.74/r2=+0.61; stellar concentration r1=+0.42/r2=+0.42;
  `P_e` r1=+0.65/r2=+0.69; `K` r1=−0.41/r2=−0.89; y-map r1=−0.01/r2=+0.90;
  τ-map r1=+0.21/r2=+0.84. *Provenance:
  `figs_raw/paper_lightcone_figs2/cell026_out1.png` (§6.1), read off figure
  annotations.*

### (2) WL ↔ kSZ latent geometry

- **Canonical correlation [0.98, 0.79]; principal angles [12°, 38°].**
  Dominant axis is *shared* with the kSZ τ/y inner-gas/suppression axis;
  second axis rotated ~38°. WL κ is the projected total-mass perturbation →
  responds to a *blend* of gas zones, not their separation ("same leading
  direction, rotated/blended second"). *Provenance: WORKLOG 2026-06-29
  "wl_latent_sbi.ipynb" §2; figure `examples/wl_latent_sbi_figs/f2c_wl_vs_ksz_latent.png`
  (title states the numbers exactly).*
- **WL tracks inner/outer f_gas zones r=0.59/0.71**, vs kSZ's r≈0.95 (kSZ
  cleanly separates the two zones; WL blends them). *Provenance: WORKLOG
  2026-06-29; figure `f2b_wl_latent_gaszones.png` panels (c)/(d), titled
  "colour = INNER f_gas(<R500) (r=+0.59)" / "colour = OUTER f_gas(R500→R200)
  (r=+0.71)".*
- **The 1st WL latent is universal across statistics; the 2nd rotates.**
  Canonical alignment of every WL summary's own top-2 PCA latent to the
  κ-Cl 2-D latent: latent-1 alignment ≈1.0 (peak_counts 0.99, minima_counts
  0.99, PDF 1.00, `mf_v0` 0.98, `mf_v1` 1.00, `mf_v2` 1.00, `Cl_κy` 0.97,
  `Cl_yy` 0.93); latent-2 alignment is *statistic-dependent*: peak_counts
  0.76, minima_counts 0.73, PDF 0.69, `mf_v0` 0.20, `mf_v1` 0.43, `mf_v2` 0.41,
  `Cl_κy` 0.90, `Cl_yy` 0.89 — Minkowski functionals/moments see a rotated
  2nd combination; the tSZ cross/auto spectra align *more* strongly with
  latent-2 than peaks/PDF do. *Provenance: WORKLOG 2026-06-24 "WL feedback
  latents..." (canon corr ≈1.0 first, 0.7–0.8 for peaks/PDF, 0.2–0.4 for
  MFs/moments — qualitative match); figure
  `examples/wl_latent_sbi_figs/f2d_multistat_align.png`, numeric values read
  off bars.*
- **Orthogonal tSZ probes don't expand dimensionality but sharpen/rotate the
  accessible axes.** # recoverable params (CV-ridge R²>0.1) stays ~5/30 for
  κ-2pt vs all-WL vs WL+tSZ, but IMFslope recoverability 0.33→0.56→0.64 and
  tSZ specifically unlocks the SN/thermal-energy axis: WindEnergy 0.13→0.33,
  RadioFeedbackReorient →0.16. *Provenance: WORKLOG 2026-06-24 "WL feedback
  latents..."; figures `wl_stat_latent_summary.png`,
  `wl_recoverability_byparam.png` (`examples/figures_lightcone/`).*

### (3) Global response (model-free Spearman)

- **Top `S(ℓ)`-suppression drivers, |Spearman r| (n=253, z_s=1):**
  IMFslope 0.49, VariableWindVelFactor 0.48, BlackHoleRadiativeEfficiency
  0.43, WindEnergy 0.34 (the wind/SN/BH feedback sector); peak-function
  sensitivity concentrates in the high-ν tail. *Provenance: WORKLOG
  2026-06-29 §1; figure `examples/wl_latent_sbi_figs/f1_corr_heatmaps.png`
  (7-panel, explicitly labeled `n=253, z_s=1`) — also embedded in
  `paper_lightcone_figs2.ipynb` cell 18 (§4.6a), a closely matching 6-panel
  version.*
- **Caveat — a second, deprecated metric exists in the code**: an
  emulator-pushed Sobol variance-decomposition (`global_sensitivity`/
  `sobol_indices` in `wl_latent_sbi.py`) is "kept for reference but no
  longer drives §1" per the docstring — it ranks VariableWindVelFactor
  (`S_i≈0.33`) and BlackHoleRadiativeEfficiency (`S_i≈0.23`) far above
  everything else and does **not** surface IMFslope as a top driver,
  disagreeing with the Spearman ranking above. The Spearman map is the
  one WORKLOG and the brief number set endorse; the Sobol-index figure
  (`examples/wl_latent_sbi_figs/f1_global_sobol.png`) should be used only
  as an appendix/consistency note, if at all, with this discrepancy
  flagged. *Provenance: `wl_latent_sbi.py` docstring §1; figure comparison.*

### (4) SBI

- **Auto-`Cl` 30-param posterior is broad; only ~5 params constrained.**
  From `Cl` alone (`EmulatorSimulator`-based NPE at n=253-equivalent):
  IMFslope 0.66, WindEnergy 0.72, VariableWindVelFactor 0.75,
  BlackHoleRadiativeEfficiency 0.83 (posterior σ / prior σ; smaller =
  tighter), rest near-prior. *Provenance: WORKLOG 2026-06-29 §3; figure
  `examples/wl_latent_sbi_figs/f3b_corner_cl.png` (6-param corner, physical
  units, TNG fiducial marked in red).*
- **Info is ~3-dimensional.** All 510×5 (ℓ-mode × source-plane) modes
  PCA-compress to **3 components = 99.5% variance (0.863/0.126/0.006)** —
  the 0.126 component is a genuine high-ℓ direction the old ℓ<5000 cut
  missed. *Provenance: WORKLOG 2026-06-24 "auto-Cl NPE on 216 runs...".*
- **The 2-D latent posterior is tight** — even though the 30-param marginals
  stay broad, the projection onto the 2-D suppression latent is **~0.17–0.18
  of the node-cloud width** (`Cl` "pins the suppression-latent hard").
  *Provenance: WORKLOG 2026-06-29 §3; figure
  `examples/wl_latent_sbi_figs/f3c_latent_posterior.png` panel (b) (NPE
  30-d posterior, blue, projected into the 2-D latent, visibly tight inside
  the grey Sobol-node cloud).*
- **Nyquist / all-modes summary beats the old coarse binning.** Three
  configs compared, all coverage≈0.90: (A) old 10-bin/ℓ<5000 → mean shrink
  0.994; **(B) all-modes/ℓ<Nyquist + PCA → 0.986 (best/most informative)**;
  (C) all-modes, no PCA (2550 features, 216 sims) → 1.004 (worst/diluted).
  Map Nyquist **ℓ≈36864** (=π/pixel for the 5°/1024-px κ maps); the
  suppression `S=Cl/Cl_dmo` cancels the CIC-aliasing artifact that
  contaminates raw `Cl` at high ℓ, so the high-ℓ range is usable and the
  baryon signal (run-to-run spread in S) *grows* with ℓ (~0.13 at ℓ~3000 →
  ~0.6 at ℓ~45000). Full tomography (15 unique auto+cross spectra) does
  **not** help further (mean shrink 0.992 vs 0.986, still compresses to
  3 PCA comps, same 0.864/0.125/0.006 split) — the cross-spectra are
  redundant with the autos for the baryon response at fixed cosmology.
  *Provenance: WORKLOG 2026-06-24 "auto-Cl NPE on 216 runs + all-modes/Nyquist
  summary" (full paragraph, both same-session sub-findings).*
- **Model-free proof the auto-Cl inverse problem is rank-2/degenerate.**
  `PC1=86.3%+PC2=12.6%=98.9%` of all-run `Cl` variation (≈ suppression
  amplitude + ℓ-tilt, the van Daalen/BCM 2-param family); several params
  drive the *same* 2 directions (forward |corr| 0.37–0.45 for
  VariableWindVel/BHRadEff/WindFreeTravelDens/IMFslope) → degenerate. The
  40 runs nearest the fiducial in `Cl`-space span ~100% of the prior in θ
  (kNN std/prior ≈1.01); linear inverse R² ≤0.43, mean 0.06.
  *Provenance: WORKLOG 2026-06-24, same entry, "diagnosed why (rank-2
  forward map)" paragraph.*
- **SBC: mean 68% coverage = 0.70 (calibrated).** Rank histogram
  approximately uniform; cross-checked NPE vs an explicit-likelihood `emcee`
  fit. *Provenance: WORKLOG 2026-06-29 §3 (exact number, matches the brief's
  mandatory number); figure
  `examples/wl_latent_sbi_figs/f3e_crosschecks.png` panel (c), title "mean
  68% coverage = 0.70", red dashed reference line ≈0.68.*
- **Cautionary result — stacking degrades the fit (§3d).** Stacking emulated
  peaks/κ×y onto `Cl` **degrades** the fiducial fit: mean shrink
  0.97→1.01→1.03 as `Cl`→`Cl`+peaks→`Cl`+peaks+κ×y, informed-param count
  5→0. Cause: the measured fiducial is in-distribution for `Cl` (max
  |z|=2.1σ) but becomes a **5.2σ outlier** once the emulated HOS/κ×y stats
  are appended — this is an **emulator HOS/cross-spectra fidelity
  limitation**, not evidence the extra statistics lack information (§(3)
  above shows peaks/κy respond strongly in the model-free Spearman test).
  Must be presented explicitly as a methods-fidelity warning, not a physics
  result. *Provenance: WORKLOG 2026-06-29 §3 "Cautionary result (§3d)"
  (exact numbers); figure `examples/wl_latent_sbi_figs/f3d_progressive.png`
  (bars read ≈0.96/≈1.01/≈1.03, "stacking emulated HOS DEGRADES the Cl
  constraint").*

### (5) κ×y + tSZ dominate feedback information

- **κ×y cross-spectrum is the single most feedback-sensitive field-level
  observable in the whole suite** (Sobol spread 0.76, vs peaks/WST ~0.08).
  *Provenance: `wl_sz_multiprobe.py` docstring (wl-tsz-bridge worktree).*
- **WL κ auto-statistics (suppression `S(ℓ)` + peaks) barely constrain
  feedback; the κ×y cross + tSZ y dominate — pins 2 directions to variance
  0.06/0.11** at fixed cosmology. κ×y is amplitude-dominated → degenerate
  with σ8/Ωm (BIND is fixed-cosmology, so an explicit amplitude nuisance A
  is needed); amplitude-marginalized shear×y constraint is modest
  (variance 0.25). κ×y is *redundant* with per-halo y (both trace thermal
  pressure) — no further gain from folding both in. Per-halo κ stack is a
  **feedback-blind mass anchor** (the per-halo signal is dominated by total
  mass, not feedback; the feedback information is field-level). *Provenance:
  WORKLOG 2026-06-24 "P4 → multi-probe WL×SZ..." (Findings 1–3); memory note
  `wl-sz-kappay-feedback.md`.*
- **Cosmology anchor.** κ-auto (∝A²) pins the amplitude: A = 1.00±0.06 →
  0.999±0.008 once fit jointly with κ×y (∝A¹); freeing κ×y this way tightens
  the feedback-direction variance 0.25→0.15 (toward the fixed-cosmology
  floor of 0.06). *Provenance: WORKLOG 2026-06-24 "FIRST REAL-DATA fit
  (D9-D10)"; engine `wl_sz_cosmo_anchor.py` docstring (same mechanism
  description).*
- **Figure candidate (see Figures/Caveats):** `examples/figures_field/g6_kxy_driver.pdf`
  — panel (a) ranks band-integrated "feedback response D" by probe:
  y×τ > τ×τ > y×y > κ×y ≫ κ×κ (bars, read off image, decreasing order);
  panel (b) shows node-spread/fiducial vs ℓ for the same 5 probes — κκ
  (WL auto) stays flat/dark out to ℓ~10⁴, the cross/tSZ probes "light up"
  much earlier. This figure was produced for the sibling field-level kSZ
  paper (`paper_ksz_field.ipynb`, WORKLOG 2026-06-26) using the same
  `emulator_dataset.npz`/κ×y machinery the brief assigns to this paper —
  see the Caveats section for the cross-paper provenance flag.

### (6) Halo-level anchor + lightcone population validation

- **Per-halo Born Δκ tracks f_gas: r=+0.62** (plus T), imprint peaks at
  **z≈0.33**. Measured directly (no ray-trace randomization) from the
  fiducial composite slabs: `dkappa_halo(z;z_s) ~ W(χ;χ_s) · <Σ_BIND−Σ_DMO>_aperture`,
  aperture = 0.659×R200 (≈R500c). *Provenance: WORKLOG 2026-06-29 "§6 of
  paper_lightcone_figs2.ipynb fixed..." (`lightcone_halo_dkappa.py` bullet);
  engine `lightcone_halo_dkappa.py` docstring/code (`R500_FAC = 0.659`).*
  Figure: `examples/figures_lightcone/fig_lightcone_population.pdf`, right
  panel ("Born Δκ (norm.) vs halo f_gas(<r500c)", annotated "r=+0.62").
- **§6 fix: use the actual lightcone halo population, not the z=0.034
  snapshot.** The z=0.034 shell carries only **~3%** of the z_s=1 lensing
  kernel weight (kernel `W(χ)` peaks at z≈0.42, or ≈0.3–0.4 more generally).
  Despite this, run-ranking of **amplitude** features (f_gas, f_star, Y, T,
  K, all at the cluster/group mass bins) at z=0.034 matches the
  lightcone-kernel-weighted readout at **r>0.95** — feedback acts coherently
  across cosmic time. **Redistribution** features (gas concentration
  f500/f200, gas ejection 1−f500/f200) are *not* proxied well: **r≈0.47**
  (figure reads ≈0.46–0.48 for the two redistribution bars). On the
  lightcone-weighted readout, latent-1 (radial/ejection) CV R² improves
  0.94→0.95 versus the z=0.034-only proxy. *Provenance: WORKLOG 2026-06-29
  "§6 of paper_lightcone_figs2.ipynb fixed: the lightcone halo population,
  not the z=0.034 box" (full entry); figure
  `examples/figures_lightcone/fig_lightcone_population.pdf` (3-panel: kernel
  weight vs z with "old §6 epoch (z=0.034): 3%" annotation; amplitude vs
  redistribution driver bars; Born Δκ–f_gas scatter).*
- Engines: `lightcone_halo_catalog.py` (253 runs + fiducial, ~34k halos/run
  over 20 shells z=0.034–2.444, transverse position already in the lightcone
  frame via the stage-1 transform), `lightcone_halo_observables.py`
  (kernel-weighted vs z=0.034-only comparison), `lightcone_halo_dkappa.py`
  (per-halo Born Δκ). **Caveat carried from the engine docstrings: Sobol
  runs have no lensplanes (only randomized ray-traced `kappa_maps.npz`), so
  the per-halo map link (Born Δκ) is fiducial-only** — this is one of the
  brief's mandatory caveats.

### (7) Real-data contact: BIND shear×y vs DES Y3 × ACT

- **First real-data confrontation** (Pandey/Gatti+ DES Y3 × ACT compton×shear;
  `shivampcosmo/ACTxDESY3` `DES_ACT.fits`, **26.5σ** total detection, joint
  covariance, DES n(z)). *Provenance: `desact_sheary_realfit.py` docstring;
  WORKLOG 2026-06-24 "FIRST REAL-DATA fit (D9-D10)".*
- **A real pipeline bug was found and fixed**: Pylians `XPk_plane`
  normalization ≠ `Pk_plane` (constant factor **F≈1.2016e7**) → the stored
  κ×y cross was ~1e7 too low (auto-spectra were unaffected; pixel-space
  `corr(κ,y)=0.52` was fine). Patched in `bind.inference.stats.power_spectrum`.
  *Provenance: WORKLOG 2026-06-24 same entry; constant confirmed in code
  (`F_XPK = 1.2016e7` in `desact_sheary_realfit.py`).*
- **⚠ Correction chain (WORKLOG self-correction; use the LATEST number).**
  (i) An initial "BIND ~2× above the data → feedback too weak" read was
  identified by WORKLOG itself as a **pipeline error**: it omitted the y-map
  beam entirely (BIND's y-map is beam-free; the data have an effective
  beam) and applied no scale cut. *Provenance: WORKLOG 2026-06-24, last
  paragraph of "P4 → multi-probe WL×SZ..." entry — "a first 'BIND ~2× =
  feedback too weak' figure was a PIPELINE ERROR".*
  (ii) With a beam correction (first tried as a 10′ "effective" Gaussian +
  θ>8′ cut) the same WORKLOG entry reports: BIND matches the data's
  intermediate-θ peak but **over-predicts large θ by ~1.5–2×**, entangled
  with cosmology/beam/cuts/κ×y-normalization — **"NO clean feedback claim
  from shear×y yet"** at that point. *Provenance: WORKLOG 2026-06-24, same
  paragraph.*
  (iii) A later, more careful pass (WORKLOG 2026-06-26, "NEW field-level
  companion paper `paper_ksz_field.ipynb`") replaced the 10′ "effective"
  beam with the **physical 2.4′ ACT beam** (a user-caught bug: the 10′ beam
  over-smoothed BIND, shifting its peak to 11′ vs the data's 4.5′) and
  reports the field-level result explicitly: **"BIND ~1.5–2× high" (TNG too
  gas-bound)**, and additionally **"too steep"** (over-bound gas + 5°
  field-of-view cutting low-ℓ power at large θ). This is the number to use
  for the "contact with data" headline, per WORKLOG's own correction
  chain. *Provenance: WORKLOG 2026-06-26 entry, verbatim: "BIND ~1.5–2× high
  (TNG too gas-bound), the field-level missing-baryon tension."*
- **Figure-level numbers** (read directly off `examples/figures_field/g5_desact_confront.pdf`,
  which implements the 2.4′-beam, θ>8′-cut version): DES source bin 3
  (z̄≈0.74): robust 8′–40′ aperture BIND/data ≈ **2.5×**; source bin 4
  (z̄≈0.94): BIND/data ≈ **2.3×**. Figure caption on the plot: "BIND
  shear×y vs REAL DES Y3×ACT — ~1.5–2× high (too gas-bound) + too steep
  (grey = aperture-limited)." *Provenance: figure text/annotations,
  `examples/figures_field/g5_desact_confront.pdf`.* This is consistent
  with, and slightly more precise than, the WORKLOG (ii)/(iii) prose
  above — use the figure's per-bin ratios (2.5×/2.3×) plus the WORKLOG
  "~1.5–2×, too steep" summary framing.
- **Robust feedback signal remains inner-CGM, not shear×y.** WORKLOG is
  explicit that the *un-ambiguous* feedback constraint at this stage still
  comes from the kSZ-CAP / X-ray f_gas legs (Paper IV territory), not from
  shear×y, because the shear×y residual is entangled with several
  systematics. *Provenance: WORKLOG 2026-06-24, "...the robust feedback
  signal stays inner-CGM (kSZ-CAP, X-ray f_gas)."*

---

## Discussion (numbers to carry forward, not new results)

- What "2-D" means for marginalization: ties to Paper II (in prep; that
  paper's capstone found exactly 2 baryon-template nuisance parameters
  necessary & sufficient to remove the TNG baryonic S8 bias from LSST-Y10
  cosmic shear — cite as sibling paper, no new number to source here beyond
  what Paper II's own dossier owns).
- **Emulator fidelity as the SBI bottleneck.** The `bind.emulator` GP is
  accurate to ~2.0% median fractional error on `S(ℓ)` (dip-corr 0.87) on the
  237-run interior held-out split, below the ~6% cosmic-variance/BIND
  generative-noise floor — good enough for the `Cl`-only NPE, but *not* good
  enough for the stacked HOS/κ×y NPE (§3d 5.2σ artifact above).
  *Provenance: WORKLOG 2026-06-23 "bind.emulator" "Full-set rebuild + backend
  verdict" paragraph.* A quantitative Stage-IV-precision check
  (`paper_lightcone_figs2.ipynb` §7, `fig_emulator`) compares a small
  N-physical-feature→statistic emulator's held-out fractional error to the
  LSST-Y10/Euclid measurement-error floor: median `S(ℓ)` error (ℓ>300) is
  **N2=5.3%, N3=4.4%, N4=4.4%, N5=4.4%** across feature-count sweeps — the
  emulator saturates around N≈3–4 physical features, i.e. the shortfall at
  small scales is feature-completeness-limited, not simulation-count-limited.
  *Provenance: figure `figs_raw/paper_lightcone_figs2/cell034_out1.png`
  (§7), annotation text on the left panel.*
- **TNG-only + fixed-cosmology caveat** applies throughout (mandatory,
  see Caveats).

---

## Optional §/Appendix — SHMR scatter (include if figures extract cleanly: they do)

- **Joint Sobol feedback scatter σ≈0.386 dex (variance ≈0.149 dex²) is
  LARGER than the twobound one-at-a-time spread σ≈0.161 dex (variance
  ≈0.026 dex²)** at fixed halo mass — varying all 30 knobs jointly
  *compounds*, so single-knob extremes are **not** an upper bound on the
  joint feedback variance (this corrects the twobound notebook's own
  caveat). Accretion history (`t_form` on the DMO tree) explains **~10%**
  of the halo-to-halo (non-feedback) residual (halo-to-halo variance
  ≈0.005 dex², ρ(t_form, residual)<0, early-forming halos sit above the mean
  SHMR). *Provenance: WORKLOG 2026-06-22 "SHMR scatter (feedback vs
  accretion) on the SB35 Sobol suite" (exact σ/variance numbers); figure
  `figs_raw/shmr_accretion_scatter_sb35/cell011_out1.png` ("SHMR scatter
  budget (SB35 Sobol)": feedback bar ≈0.148–0.149 dex², twobound dashed
  line ≈0.026 dex², halo-to-halo bar ≈0.005 dex² — matches text exactly).*
- Same-halo drop-in: Sobol/twobound/fiducial paint the *same* 2933 snap-96
  DMO halos (matches 2933/2933 in the reuse check); §8's per-knob lo/hi
  panels become per-param low/high **tercile main-effects** (no
  single-knob runs exist in a Sobol design — terciles marginalize the
  other 29 params). *Provenance: WORKLOG 2026-06-22, same entry.*
- Additional figures available and extracted cleanly: `cell009_out0.png`
  (SHMR relation + scatter), `cell013_out1.png` (accretion-history
  regression), `cell015_out0.png` / `cell017_out1.png` (fiducial SHMR /
  early-vs-late-former split colored by formation time).

---

## FIGURES

Primary set (target 10–12 per brief; all copied candidates live under
`papers/03_latent_sbi/figs_raw/<source>/` for the extracted-from-notebook
ones, on-disk gitignored paths otherwise — the Draft agent should copy the
final selections into `papers/03_latent_sbi/figs/` with descriptive names).

| # | Path | What it shows | Section | Provenance |
|---|------|----------------|---------|------------|
| 1 | `examples/wl_latent_sbi_figs/f2a_effdim.png` | Linear PCA scree (2 comps=99% var) + nonlinear IOB information curve (effective dim ~2, R²(k)/R²max flattens by k=1–2) | §5 Results (1) 2-D latent | `wl_latent_sbi.ipynb` §2 cache (WORKLOG 2026-06-29) |
| 1alt | `papers/03_latent_sbi/figs_raw/paper_lightcone_figs2/cell022_out1.png` | Latent-plane scree (PC1≈89%,PC2≈11%,cum 99.2%) + Sobol/1P spokes on the (latent1,latent2) plane, colored by group-cluster f_gas | §5 Results (1) | `paper_lightcone_figs2.ipynb` cell 22, §5.1 `fig_latent_plane` |
| 2 | `examples/wl_latent_sbi_figs/f2b_wl_latent_gaszones.png` | (a) 2-comp scree; (b) which params drive ê1 vs ê2 (BH/SN/wind sector bar chart); (c)/(d) latent plane colored by inner/outer f_gas, r=+0.59/+0.71 | Results (1) axis physics + (2) WL/kSZ zones | `wl_latent_sbi.ipynb` §2 (WORKLOG 2026-06-29) |
| 2alt | `examples/figures_lightcone/fig_latent_drivers.pdf` | Latent plane colored by gas amplitude/ejection(radial)/gas–DM offset(anisotropy); partial-correlation driver bar chart (f_star, f_gas slope, ejection, DM/gas ellipticity, gas–DM offset, T) | Results (1) axis physics | `paper_lightcone_figs2.ipynb` §6.2 `fig_latent_drivers` (image not embedded in the .ipynb itself — cell has no cached output — but the on-disk PDF exists from a prior notebook run) |
| 2alt2 | `examples/figures_lightcone/wl_vae_{latent_perturbation,param_corr,latent_params}.png` | β-TCVAE reproduction of Lin+2026 Fig. 1/3 for WL: latent-perturbation effect on S(ℓ), param↔latent Pearson corr, params on the (Latent0,Latent1) plane | Results (1) axis physics | `wl_feedback_vae.py` (WORKLOG 2026-06-24) |
| 3 | `examples/wl_latent_sbi_figs/f2c_wl_vs_ksz_latent.png` | WL latent vs kSZ τ/y latent, both colored by inner f_gas; title states canon corr [0.98,0.79], principal angles [12°,38°] | Results (2) WL↔kSZ geometry | `wl_latent_sbi.ipynb` §2 (WORKLOG 2026-06-29) |
| 4 | `examples/wl_latent_sbi_figs/f1_corr_heatmaps.png` | Model-free per-bin Spearman r, 7 statistics × 30 params, explicitly labeled n=253, z_s=1 | Results (3) global response | `wl_latent_sbi.py` §1 (WORKLOG 2026-06-29) |
| 4alt | `papers/03_latent_sbi/figs_raw/paper_lightcone_figs2/cell018_out1.png` | Same Spearman-heatmap analysis, 6-panel, embedded in the paper notebook | Results (3) | `paper_lightcone_figs2.ipynb` cell 18, §4.6a |
| 5 | `examples/wl_latent_sbi_figs/f3b_corner_cl.png` | NPE posterior corner, 6 best-constrained params, physical units, TNG fiducial marked red | Results (4) SBI | `wl_latent_sbi.ipynb` §3 (WORKLOG 2026-06-29) |
| 6 | `examples/wl_latent_sbi_figs/f3c_latent_posterior.png` | (a) χ² over the latent plane w/ MAP; (b) 30-d NPE posterior projected into the 2-D latent — visibly tight vs. the full Sobol-node cloud | Results (4) SBI, latent posterior | `wl_latent_sbi.ipynb` §3 |
| 7 | `examples/wl_latent_sbi_figs/f3e_crosschecks.png` | (a) NPE vs emcee marginals; (b) SBC rank histogram; (c) per-param 68% coverage, mean=0.70 | Results (4) SBI, SBC calibration | `wl_latent_sbi.ipynb` §3 |
| 8 | `examples/wl_latent_sbi_figs/f3d_progressive.png` | Cautionary: mean posterior/prior width for Cl / Cl+peaks / Cl+peaks+κ×y (degrades ~0.96→1.01→1.03); per-param breakdown | Results (4) SBI, cautionary §3d | `wl_latent_sbi.ipynb` §3 |
| 9 | `examples/figures_field/g6_kxy_driver.pdf` | (a) band-integrated feedback response D by probe (yτ>ττ>yy>κy≫κκ); (b) node-spread/fiducial vs ℓ per probe | Results (5) κ×y dominance | `paper_ksz_field.ipynb` (WORKLOG 2026-06-26) — **cross-paper provenance, see Caveats/Gaps** |
| 10 | `examples/figures_lightcone/fig_lightcone_population.pdf` | 3-panel: lensing-kernel weight vs z (3% at z=0.034 annotated); amplitude-vs-redistribution driver bars (r>0.95 vs r≈0.47); per-halo Born Δκ vs f_gas scatter (r=+0.62) | Results (6) halo-level anchor + §6 population fix | `paper_lightcone_figs2.ipynb` §6.4 `fig_lightcone_population` (WORKLOG 2026-06-29) |
| 11 | `examples/figures_field/g5_desact_confront.pdf` | ξ_γy(θ) BIND Sobol band vs real DES Y3×ACT data, 2 source bins (z̄=0.74,0.94), 2.4′ beam, robust-aperture BIND/data ≈2.5×/2.3× annotated | Results (7) real-data contact | `paper_ksz_field.ipynb` (WORKLOG 2026-06-26) — **cross-paper provenance, see Caveats/Gaps** |
| 12 (optional) | `papers/03_latent_sbi/figs_raw/shmr_accretion_scatter_sb35/cell011_out1.png` | SHMR scatter-budget bar chart: feedback var ≈0.149 dex² vs twobound-marginal ≈0.026 dex² (dashed) vs halo-to-halo ≈0.005 dex² | Optional appendix (SHMR) | `shmr_accretion_scatter_sb35.ipynb` §5 (WORKLOG 2026-06-22) |

Supplementary / methods-context candidates (not in the primary 10–12, but
useful for Methods/Discussion figures if space allows):
- `papers/03_latent_sbi/figs_raw/paper_lightcone_figs2/cell010_out1.png` —
  §3.1 experimental design: 256-node unit-cube design ranked by |r| with
  S(ℓ), with 1P/twobound axis-aligned spokes overlaid.
- `papers/03_latent_sbi/figs_raw/paper_lightcone_figs2/cell034_out1.png` —
  §7 emulator-vs-Stage-IV-measurement-error comparison (N2=5.3%,
  N3/N4/N5=4.4% median S(ℓ) error), supports the Discussion "emulator
  fidelity as bottleneck" caveat.
- `examples/wl_latent_sbi_figs/f2d_multistat_align.png` — 1st-latent
  universal (~0.99 alignment across 8 statistics) vs 2nd-latent rotated
  (0.20–0.90) bar chart, supports Results (2) numeric claims above.
- `examples/sobol_sbi_figs/perhalo_data.png` — per-halo training-data
  illustration only (60,000 halos, Y200–M200, colored by wind energy); NOT
  a posterior result (see Gaps #1) — usable only as a Methods "what the
  per-halo datum looks like" figure, captioned accordingly.
- `examples/sobol_latent_figs/{feedback_dim,active_loadings}.png` —
  earlier (2026-06-22, MDN-era) IOB/active-subspace figures from the
  first-generation `sobol_latent.ipynb`; superseded by the wl_latent_sbi
  generation above but usable as Methods-history figures if useful.
- Additional SHMR figures: `cell009_out0.png` (relation+scatter),
  `cell013_out1.png` (accretion-history regression),
  `cell015_out0.png`/`cell017_out1.png` (formation-time coloring), all in
  `papers/03_latent_sbi/figs_raw/shmr_accretion_scatter_sb35/`.

---

## CAVEATS

Brief's mandatory list (verbatim, sourced/confirmed):
1. **Emulator HOS/cross fidelity limits stacked SBI** — the 5.2σ artifact
   (§3d, Results (4) above) is a **methods warning**, present it as such,
   not as a physics finding. *Confirmed: WORKLOG 2026-06-29 §3d.*
2. **TNG-only + fixed cosmology** — every result in this dossier is a
   single-simulation-suite (IllustrisTNG/SB35), fixed-cosmology sweep;
   generalization across suites/cosmologies is untested (explicitly flagged
   in `wl_stat_latents.py`'s own docstring caveats: "cross-suite/cosmology
   not testable with our data"). *Confirmed: WORKLOG 2026-06-24 "WL feedback
   latents..." closing parenthetical.*
3. **n=253 design points in 30-D (interpolation, not extrapolation)** — the
   final Sobol suite size; all SBI/latent claims above are interior to this
   256-node design's prior-predictive cloud, not an extrapolation test.
   *Confirmed: `n=253` label directly on `f1_corr_heatmaps.png` and cited
   throughout WORKLOG 2026-06-29.*
4. **Per-halo map link (Born Δκ) is fiducial-only** — "Sobol runs have no
   lensplanes (only randomized ray-traced `kappa_maps.npz`)". *Confirmed:
   WORKLOG 2026-06-29 §6 entry, closing sentence; also stated in
   `lightcone_halo_dkappa.py`'s data-source discussion.*

Additional caveats found during mining:
5. WST is excluded from `wl_stat_latents.py`'s cross-statistic latent
   comparison — only 40 valid runs had WST computed at that point (WORKLOG
   2026-06-24).
6. Two different "global sensitivity" metrics exist in the codebase
   (model-free Spearman vs an emulator-pushed Sobol-index decomposition)
   and **disagree on the top driver** (Spearman: IMFslope 0.49 top;
   Sobol-index: VariableWindVelFactor top, IMFslope absent from the top
   list) — `wl_latent_sbi.py`'s own docstring explicitly deprecates the
   Sobol-index version for §1; the paper should lead with Spearman and, if
   showing the Sobol-index figure at all, flag the discrepancy rather than
   silently picking numbers from either.
7. Per-halo population SBI (8.6M halos) — **no completed result exists**;
   see Gaps #1 below. This directly affects how strongly the brief's Thesis
   sentence "Population-level SBI on 8.6M halos gives coverage-calibrated
   parameter posteriors" can be stated.
8. The desact_sheary_realfit real-data result went through **three
   successive corrections** in one day (WORKLOG 2026-06-24) and a further
   correction two days later (WORKLOG 2026-06-26); the final, most-corrected
   number ("~1.5–2× high, too gas-bound, too steep", figure ratios
   2.5×/2.3× at 8′–40′) should be used, and the correction chain itself is
   worth a one-sentence methods note given how easy the beam/normalization
   bugs were to get wrong.
9. `sobol_sbi_figs/{corner_fiducial,calibration,degeneracy_matrices,
   degenerate_directions,joint_degeneracy,loo_recovery,marginals,
   constrained_params}.png` (dated 2026-06-22) are from the **first-generation**
   diagonal-MDN-in-[0,1] / then reworked GaussNPE posterior on the
   216/256-run suite (WORKLOG 2026-06-22 "SB35 Sobol latent space + SBI"),
   which is methodologically superseded by the `wl_latent_sbi.ipynb`
   generation (2026-06-29, logit-space full-cov NPE ensemble on the
   emulator-as-simulator). Use the later generation for headline SBI
   figures/numbers; the earlier set is Methods-history material only, and
   the robust finding it independently established (AGN
   BlackHoleRadiativeEfficiency↔QuasarThresholdPower degeneracy, +0.27) is
   worth keeping as a cross-check citation.
10. The twobound-suite response-error convention (marginal, not paired
    per-run) means the SHMR-appendix "twobound one-at-a-time" dashed
    reference line in `cell011_out1.png` is a **conservative marginal**
    error, not a paired/matched comparison to the Sobol feedback variance —
    worth a one-line caveat if the two numbers are directly compared.
    *(memory note `twobound-response-error-convention.md`.)*

---

## CITATIONS

**Brief seed citations — verification status against mined sources:**
- **Lin+2026, arXiv:2509.01881** — ✅ CONFIRMED, precisely: `wl_feedback_vae.py`
  docstring names it in full: "Lin, Li, Genel, Villaescusa-Navarro+ 2026
  ('One latent to fit them all', arXiv:2509.01881)", describing exactly the
  method this paper's WL analogue reproduces (2D latent for
  T²(k,a)=P_hyd/P_dmo across CAMELS suites).
- **Cranmer+2020 (SBI review)** — not found in mined sources; standard
  reference, verify independently.
- **Papamakarios & Murray 2016 / Greenberg+2019 (NPE/APT)** — not found by
  name in mined sources; the NPE method used throughout is the `sbi`
  package's standard SNPE, so these are methodologically appropriate but
  unconfirmed as explicit citations in the repo.
- **Tejero-Cantero+2020 (sbi package)** — ✅ methodologically confirmed: the
  `sbi` Python package is used directly and by name across
  `lightcone_cl_sbi.py`, `wl_latent_sbi.py` (`run_npe (sbi, logit-space)`),
  `sobol_perhalo_sbi.py` ("train an amortized neural posterior ... with
  `sbi`"). The package-paper citation itself was not spelled out in any
  docstring, but this is the standard/correct cite for "we use `sbi`."
- **Talts+2018 (SBC)** — ✅ methodologically confirmed: `sbc_ranks` /
  "SBC: ranks ≈ uniform if calibrated" / rank-histogram calibration is
  exactly the Talts+2018 SBC procedure, used in both `wl_latent_sbi.py` §3
  and the earlier `sobol_sbi.ipynb` (leave-run-out rank coverage). Paper
  name not spelled out in-repo; standard citation.
- **Cheng+2020 (WST)** — ✅ methodologically confirmed: `bind.inference.stats`
  gained a `wst` statistic (WORKLOG 2026-06-23, "kymatio 2-D scattering —
  needs the `sph_harm` shim for scipy≥1.13"); Cheng+2020 is the standard WST
  citation, not spelled out by name in-repo.
- **CCA / canonical correlation, standard ref** — ✅ methodologically
  confirmed: `sklearn.cross_decomposition.CCA` used directly in
  `wl_stat_latents.py` for exactly the canonical-correlation alignment
  measurements reported in Results (2).
- **Villaescusa-Navarro+2021 (CAMELS)** — not spelled out by name in the
  mined Paper-III-specific sources (it's the standard CAMELS suite paper;
  BIND is built on IllustrisTNG/SB35, itself a CAMELS-adjacent suite) —
  verify independently; likely also cited in Papers I/II for the suite
  description.
- **Gatti+2021/Pandey+2022 (DES×ACT shear×y)** — ✅ CONFIRMED (author names,
  not exact year): `desact_sheary_realfit.py` docstring: "FIRST REAL-DATA
  confrontation: BIND shear x y vs DES Y3 x ACT (Pandey/Gatti+)"; data file
  `shivampcosmo/ACTxDESY3` `DES_ACT.fits`. The Cite agent should verify the
  exact Pandey/Gatti publication year(s) against the data release used.
- **Tröster+2021 (KiDS×Planck y)** — not found in mined sources; verify
  independently (see Gaps #4).
- **Hill & Spergel 2014** — not found in mined sources; verify independently.
- **Battaglia+2012 (pressure profile)** — not found in the Paper-III-scoped
  sources (a Battaglia-style GNFW pressure profile is referenced
  tangentially in Paper-IV-territory files, e.g. `ksz_desi_act_plan.md`
  mentions per WORKLOG "vs Hadzhiyska+26 GNFW" — a different profile
  paper); verify independently if used here.
- **Amodeo+2021** — found, but in Paper-IV-adjacent territory:
  `profiles.py` (wl-tsz-bridge worktree) docstring: "the kSZ 'gas is more
  extended than the dark matter' result, ACT×DESI / Schaan+21 / Amodeo+21".
  Relevant only if this paper borrows the profile-shape framing; otherwise
  belongs to Paper IV.
- **Papers I/II/IV** — cite as *in prep.*, per PLAN.md convention. Working
  titles (from `PLAN.md`): I = "Baryon-painted weak-lensing and SZ
  lightcones from dark-matter-only simulations"; II = "Two nuisance
  parameters suffice to marginalize baryonic feedback in LSST-era cosmic
  shear"; IV = "Gas thermodynamics across a 30-parameter feedback space:
  BIND vs DESI×ACT kSZ and eROSITA".

**Additional citation keys found in-repo (unverified, for the Cite agent to
triage), from `wl-tsz-bridge/examples/draft.tex`** (an adjacent/earlier
LaTeX draft in the same worktree, not one of the brief's named sources but
worth harvesting): `Martinet2021peaks`, `Gatti2022HOScos`, `Zurcher2022MFs`,
`Marques2024HSC`, `FLAMINGOscattering2025`, `Gatti2023maplevel`,
`BigwoodAmon2024`, `DESACTTSZWL2025`, `Pandey2022joint`, `vanDaalen2020`,
`VN2021fieldemulator`, `Schaye2023FLAMINGO`, `Schneider2022clusters`,
`Siegel2025`/`Siegeletal2025mnras`, `Arico2024`, `Fong2021TNGbaryons`,
`MartinezConcepcion2024baryons`, `Amonetal2023`, `LeeAmon2026`, `BindPaper1`.
None of these were cross-checked against arXiv/ADS by this mining pass —
flagging their existence only; the Cite agent must verify each before use.

---

## Gaps (brief demands not sourced — do NOT guess)

1. **Per-halo population SBI (8.6M halos) has NO completed numeric result
   to report**, despite the brief's Thesis line ("Population-level SBI on
   8.6M halos gives coverage-calibrated parameter posteriors") and Methods
   line. Direct inspection of `sobol_sbi.ipynb`'s executed text outputs
   shows: `"GPU results present: False -> .../perhalo_sbi_results"` and
   `"[DEMO] no GPU results; training a small inline model (illustration
   only)"`; the notebook's own markdown for §4 states the fallback path is
   "for wiring/illustration only, not the science result." No
   `population_posterior.npz` exists on disk in the main tree
   (`examples/__pycache__` shows the two `.py` engines were imported, but no
   cached results directory `perhalo_sbi_results` was found under
   `examples/`). **The Methods description (IID-product + emcee +
   leave-run-out temperature calibration) is real and sourced from the
   engine docstrings, but the specific coverage/posterior-width numbers for
   this pipeline do not exist anywhere in the mined sources.** The writer
   should either drop the quantitative per-halo-population-SBI result from
   Results/Thesis, or mark it `\todo{run train_perhalo_sbi.py on GPU to get
   real numbers}` (which this mining task cannot do — running GPU jobs is
   out of scope for this agent).
2. **g5/g6 figure provenance is cross-paper.** `examples/figures_field/g5_desact_confront.pdf`
   and `g6_kxy_driver.pdf` are outputs of `paper_ksz_field.ipynb`
   (WORKLOG 2026-06-26), which is NOT one of the notebooks the brief names
   for Paper III (it lives conceptually closer to Paper IV,
   `analysis/ksz-desi-act`). The brief does explicitly assign the
   underlying *engines* (`wl_sz_multiprobe.py`, `wl_sz_cosmo_anchor.py`,
   `wl_sheary_fit.py`, `desact_sheary_realfit.py`) to this paper (per
   PLAN.md: "The κ×y multiprobe results ... belong scientifically in Paper
   III"), and these two figures are the most complete/corrected renderings
   of exactly that science available on disk. Flagging for the coherence
   pass: **Paper IV may also want to show these same two figures** (they
   were built for its sibling notebook) — coordinate to avoid duplicate
   near-identical figures across Papers III and IV, or split which paper
   owns the figure vs. which just cites the number.
3. **`desact_sheary_realfit.py`'s own docstring is stale.** It still reads
   "Result: BIND sits ~2x above the data at well-measured scales -> TNG
   feedback too weak" even though the code body already implements the
   corrected 2.4′-beam version (`Y_BEAM_FWHM` default 10.0 in this copy of
   the script, not yet updated to 2.4 — a further inconsistency between the
   docstring/defaults in this exact file and the corrected numbers reported
   in WORKLOG 2026-06-26 and realized in `g5_desact_confront.pdf`). Use the
   WORKLOG-2026-06-26 + figure numbers, not this script's docstring/defaults,
   per the "WORKLOG corrections win" rule.
4. **Tröster+2021 (KiDS×Planck y) and Hill & Spergel 2014 and
   Battaglia+2012** (brief's seed citations) were **not found** in any
   mined engine docstring, notebook markdown, or WORKLOG entry for this
   paper's sources. They may still be legitimate general references for
   the Discussion/Intro (standard tSZ/pressure-profile literature), but the
   Cite agent should treat them as **unconfirmed by primary sources** and
   verify independently (arXiv/ADS), not as "seen in the repo."
5. **Cranmer+2020, Papamakarios & Murray 2016, Greenberg+2019 (APT)** — no
   direct citation of these exact papers found in the mined sources either
   (the `sbi` package and NPE method are used extensively, and Tejero-Cantero+2020
   — the `sbi` package paper — is implicitly the correct citation for
   "we use the `sbi` package," confirmed via `import sbi` / `pip install sbi`
   usage across `lightcone_cl_sbi.py`, `wl_latent_sbi.py`,
   `sobol_perhalo_sbi.py`); Cranmer+2020 / Papamakarios-Murray / Greenberg+2019
   are standard SBI-review/APT citations the Cite agent should verify
   independently.
6. **`docs/ksz_desi_act_plan.md`** (referenced by WORKLOG for D6–D10 of the
   multi-probe program) is **not present in this working tree** (`papers`
   branch) — it is an untracked file that exists only in the `lightcone`
   source branch's working directory, outside the read-only worktrees
   provided for this task. Could not be consulted; all P4/D6-D10 numbers
   above come from WORKLOG prose only.
