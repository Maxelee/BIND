# Dossier — Paper I: BIND lightcones (methods + validation + bridge)

Mined for `papers/01_pipeline/brief.md`. Every number below carries a provenance
line. Where the brief's mandated number differs from what the sources actually
say, both are given and the discrepancy is flagged explicitly (search "**FLAG**").

---

## INTRO — framing material

- The observational "feedback crisis" framing (DESI×ACT kSZ + eROSITA vs
  FLAMINGO tension, DES Y3×ACT tSZ 21σ cross-correlation, S8 status) is
  drafted in `examples/draft.tex` §Introduction (worktree `wl-tsz-bridge`,
  lines 238–382). Candidate citation keys used there: `Siegel2025`,
  `Siegeletal2025mnras`, `DESACTTSZWL2025`, `BigwoodAmon2024`, `Amonetal2023`,
  `Salcido2025kSZ`, `FLAMINGOkSZlensing2024`, `Martinet2021peaks`,
  `Gatti2022HOScos`, `Zurcher2022MFs`, `Marques2024HSC`,
  `FLAMINGOscattering2025`, `Fong2021TNGbaryons`,
  `MartinezConcepcion2024baryons`, `HarnoisDeraps2021peaks`, `Weiss2019MFs`,
  `LeeAmon2026`, `FLAMINGOmps2024`, `Schaye2023FLAMINGO`,
  `SchneiderTeyssier2015`, `Schneider2019BCM`, `Schneider2025baryonification`,
  `Gatti2023maplevel`, `VN2021fieldemulator`, `Arico2024`, `BindPaper1`,
  `vanDaalen2020`, `Schneider2022clusters`, `Pandey2022joint`,
  `PrestonRogers2025`. **Provenance**: `examples/draft.tex` (worktree
  `wl-tsz-bridge`), not yet in `references.bib` — arXiv IDs not inline, need
  Cite-step resolution. NOTE: this draft's title/scope ("BINDing the
  lightcone: A Feedback Atlas...") is actually the twobound 1P
  feedback-sensitivity paper (closer to the flagship/Paper-III science), not
  Paper I's narrower methods+validation+bridge scope — only the Intro
  framing, citation seeds, and the Methods subsection structure
  (§2.1–2.5: BIND methodology / halo catalog / lightcone construction /
  validation / feedback suite) transfer directly; the halo-statistics and
  WL-response sections (§3–§6) are out of Paper I's outline.
- Literature positioning + full decision-gate/systematics plan:
  `docs/wl_tsz_plan.md` (worktree `wl-tsz-bridge`), written 2026-06-11,
  companion to WORKLOG 2026-06-11 entry. Its "Key references" list (§ end of
  file) gives arXiv IDs for the positioning paragraph: 2506.07432 (DES
  Y3×ACT DR6 shear×y, 21σ), 2512.02954 (strong-feedback consensus),
  2509.10455 (X-ray+kSZ+WL gas expulsion), 2410.19905 (FLAMINGO kSZ×GGL),
  2309.07959 (FLAMINGO S8), 2503.19441/2 (KiDS-Legacy S8=0.815±0.02),
  2602.10065 (DES Y6 shear), 2602.12238 (S8 status 2026), 2505.07949
  (map-level baryonification HOS), 2406.01672 (BCM+tSZ, Aricò & Angulo),
  2407.20448 (Stage-IV NG baryon correction), 2312.08450 (FLAMINGO peaks),
  2406.08540 (MTNG Born vs RT), 2201.08320 (BCM peaks hold only to ν≲4 — the
  group's own prior work, motivates BIND), 2109.04458 (KiDS×y), 2602.11279
  (SPT-3G D1 y-maps), 2201.12591 (thermal-energy census → WL). **This
  arXiv-ID list is the most reliable citation source found** (explicit IDs,
  not just author-year keys) — hand to the Cite step directly.
- Gap statement ("no framework does per-halo generative + self-consistent
  multi-probe on a lightcone") is explicit in `docs/wl_tsz_plan.md` §1
  "BIND's unique claims" (4 numbered points) and in `draft.tex` Introduction
  ¶3 ("Modelling approaches fall into three broad categories...").

---

## METHODS

### (a) BIND recap
- Conditional flow-matching emulator; 35-dim CAMELS SB35 conditioning;
  thermo channels (`compton_y, T, entropy, P_e`). **Provenance**: `CLAUDE.md`
  "What this project is" (main tree, identical in `wt/lightcone`).
- Cite as `Lee et al. in prep.` per `PLAN.md` hard rules — no fake arXiv ID.
  Candidate acronym expansion "Baryon-Incorporated N-body with Diffusion"
  appears **only** in `examples/draft.tex` (worktree `wl-tsz-bridge`, line
  331) — not corroborated elsewhere in the repo; use with a `\todo` flag if
  the acronym itself is asserted, since no authoritative source confirms it.

### (b) Lightcone construction
- **TNG300-Dark, 20 snapshots**, low-z first:
  `LIGHTCONE_SNAPSHOTS = (96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46,
  43, 41, 38, 35, 33, 31, 29)`. **Provenance**:
  `src/bind/inference/science.py:34` (worktree `lightcone`).
- **Redshift shells span z=0.034–2.444** (20 shells). **Provenance**: WORKLOG
  2026-06-29 §"§6 of `paper_lightcone_figs2.ipynb` fixed" (line 58): "20
  shells z=0.034–2.444".
- **4×R200c circular-taper paste aperture** is the default/standard:
  `r200_factor: float = 4.0` in `schemas.py:50` ("circular paste at
  r200_factor * R200c (standard); 0 = legacy square taper"),
  `pipeline.py:732-741` ("When r200_factor > 0 (the standard, default 4.0)
  each halo patch is blended with a circular Hann-tapered weight of radius
  r200_factor * R200c ... the square taper (r200_factor == 0) smears away").
  **Provenance**: `src/bind/inference/{schemas,pipeline,paint_stages,science}.py`
  (worktree `lightcone`).
- **Box**: TNG300 (205 Mpc/h) — "box=205" confirmed in WORKLOG 2026-06-21
  "SB35 Sobol lightcone pipeline (256 runs)" (halo-for-halo verification:
  `halo_centers` identical, n_halos=666, box=205 — NB "n_halos=666" appears
  to be a per-snapshot verification count, not the lightcone-wide halo
  total; the low-mass-reuse entry separately reports **2,933 halos**
  M200c>1e13 at snap_096 alone in the twobound catalog — do not conflate the
  two numbers).
- **Sky area**: default field of view **5×5 deg²**
  (`fov_deg: float = 5.0` in `stats.py` `power_spectrum()`; confirmed
  visually in `demo_maps.png` and `fig_skymaps.pdf`, both 0–5 deg axes).
- **Fiducial realizations**: 50 (WORKLOG 2026-06-10 "BIND validated vs
  hydro...": "BIND reproduces TNG300 hydro (50 realizations, identical
  geometry)"). A separate, unrelated 2000-realization hydro-replacement
  covariance exists for a Fisher-forecast study (WORKLOG 2026-06-15pm "Rung
  3 FINALIZED") — that is NOT the fiducial-closure realization count; don't
  conflate.
- **Pipeline stages** (methods pipeline docstrings, worktree `lightcone`):
  1. Stage 1 (per-snapshot, parameter-independent): project TNG300-Dark into
     z-slabs + extract per-halo DMO cutouts (`bind.inference.science`
     module docstring).
  2. Stage 2: `model.generate` (flow-matching) on cutouts → per-halo
     patches; paste via `r200_factor*R200c` circular Hann taper
     (`pipeline.paste_halos_2d`).
  3. Lens planes: `lensplane.py` — `mass_map_to_delta_scaled` →
     `density_to_lensplane` (2D Poisson FFT matching lux's `Fourier_trs()`)
     → `write_lensplane` (`lenspot{PP}.dat`, lux binary format) →
     `write_lux_config`. "Conventions match lux exactly so that lensplane
     files written here are drop-in replacements for those lux would write
     itself."
  4. lux multi-plane ray tracing (external C++ repo `/mnt/home/mlee1/lux/`,
     separate git repo): samples κ at deflected ray positions; y-planes
     ALSO sampled at deflected positions (`raytracing.cpp:272` `calc_y`,
     `:876`) — "the tSZ lightcone IS already ray-traced" (`wl_tsz_plan.md`
     §0.2). τ-plane (kSZ optical depth / FRB DM) cloned from the y
     machinery 2026-06-18 (WORKLOG "lux-RT kSZ/FRB tau plane"): additive
     along the deflected ray, no lensing kernel; `tau{plane}.dat`; validated
     on snap_096 (τ_max≈0.018, slab mean≈1.3e-5).
  5. Truth lightcone: `truth_lightcone.py` (module docstring) — projects the
     actual IllustrisTNG **hydro** fields (DM_hydro, Gas, Stars +
     compton_y/T/entropy/P_e) at the *same* M≥1e13 DMO-FoF halos under the
     *same* lightcone transform, into BIND's `composite_slab{NN}.npz`
     format — drop-in for the standard recomposite→lensplane→lux→stats
     chain. Per-particle thermo + comoving→physical a-factors match
     `data_generation/process_simulations_multiz.py`.
  6. Paired DMO trace: `bind-paint-lensplane --dmo` builds lensplanes
     straight from stage-1 DMO maps (same geometry) — WORKLOG 2026-06-10
     "BIND-vs-hydro validation notebook + DMO ray-traced baseline".
- **τ / electron column**: `tau = sigma_T * x_e * Sigma_gas / m_p`,
  `DM[pc/cm^3] = tau / TAU_PER_DM` with `TAU_PER_DM = SIGMA_T * PC_CM`
  (`lightcone_maps.py` module docstring + constants block). Velocity-free —
  the observable kSZ estimator factorizes out the velocity per
  Schaan+21/Hadzhiyska+24/Ried Guachalla+25 (cited by name, no arXiv ID, in
  `wl_tsz_plan.md` §2 "kSZ, tier 1"). **Provenance**:
  `src/bind/inference/lightcone_maps.py` (worktree `lightcone`).

### (c) Map suite
κ(z_s) [5 tomographic source planes z_s=0.5,1.0,1.5,2.0,2.44 per
`fiducial_lightcone_stats.ipynb`], Compton-y, τ/electron-column (=FRB DM up
to the constant `TAU_PER_DM`). **Provenance**: `lightcone_maps.py` docstring;
`fiducial_lightcone_stats.ipynb` cell markdown/legend (`Z_S` values read off
`cell013_out0.png` legend: z_s=0.5,1.0,1.5,2.0,2.44).

### (d) Statistics pipeline
`bind.inference.stats` module docstring (worktree `lightcone`):
`cl_kappa` (tomographic Cl + BIND/DMO suppression S(ℓ)), `cl_kappa_y` (κ×y
cross + Cl^yy), `peak_counts` (peaks + minima vs S/N ν, 8-neighbour
definition), `nongaussian_stats` (PDF, variance/skew/kurtosis, Minkowski V0
area/V1 perimeter/V2 genus vs threshold), `peak_cross_stats` (R(ν)=⟨y⟩/⟨κ⟩ at
WL peaks — **explicitly documented in the code as "a specific-thermal-energy
proxy: R ∝ Y/M ∝ f_gas·T_mw — NOT a pure gas fraction"**), `halo_scaling`
(per-halo Y500c, f_gas, f_star, T_mw). Angular power spectra via Pylians
`Pk_plane`/`XPk_plane` with `BoxSize = fov` in radians (so `k` returned IS
the multipole ℓ). Optional shape noise (`shape_noise_ngal`, `sigma_e`) added
before smoothing; `nu_norm={"map","noise"}`. ν bins extended to 12
(peak_cross edges to 13, the old ν≥6 discard removed) — WORKLOG 2026-06-11.
MAS/CIC correction: `pixelize_z_projection(mas_correct=True)` — interlacing +
CIC deconvolution, opt-in, default off (WORKLOG 2026-06-09, see Discussion
caveats below).

---

## RESULTS

### (1) Fiducial validation: BIND vs TNG300 hydro truth
- **Headline WL closure** (Cℓκκ, N_peak, N_min, V0/V1/V2 with residual
  panels): figure `fig_validation_field.pdf` /
  `figs_raw/paper_lightcone_figs/cell006_out1.png` — residuals mostly
  within a few % (shaded ±band), Cℓκκ residual ~0% flat to ℓ~10⁴, MF
  residuals <1%; N_peak/N_min residuals noisier (up to ~±15-20% at the
  sparse high/low-ν tail — small-number statistics). **Provenance**: WORKLOG
  2026-06-10 "BIND-vs-hydro validation notebook + DMO ray-traced baseline"
  §"Notebook Part II"; figure generated in
  `examples/paper_lightcone_figs.ipynb` §2.4 (built by `_build_paper_nb.py`,
  verified via `nbconvert --execute`, 0 errors — WORKLOG 2026-06-16).
- **Summary numbers** (WORKLOG 2026-06-10 "BIND validated vs hydro + R(ν)
  gas-fraction deep-dive"): "**BIND reproduces TNG300 hydro** (50
  realizations, identical geometry): Cℓκκ BIND/hydro≈1.00, suppression
  tracks hydro, **R(ν) BIND/hydro=0.98–1.06** (rises ×2 with ν), y-profile-
  at-peaks ~5%."
- **Halo-level closure** (Y–M relation, stacked τ profile ratio, pressure vs
  Arnaud+10): figure `fig_validation_halo.pdf` /
  `figs_raw/paper_lightcone_figs/cell005_out1.png` (3-panel — NOTE: the
  on-disk `examples/figures_lightcone/fig_validation_halo.pdf` is a
  **single-panel** f_gas(<r500c)-only figure, apparently a stale/different
  regeneration; the notebook-embedded 3-panel version is the richer,
  authoritative one — use the notebook PNG, not the disk PDF, for this
  figure). Y–M: BIND tracks TNG300 hydro across 1e13–3e14.5 Msun/h with
  overlapping intrinsic-scatter bands. τ_BIND/τ_hydro ratio by mass bin: ~
  1.05–1.15 for M200c 1e13.0–1e13.5/1e13.5–1e14.0, closer to unity for the
  highest bin — consistent with the quoted **"BIND/truth validation (groups
  +11%)"** number from `examples/electron_column.py` (WORKLOG 2026-06-16
  "pivot off Fisher..." item (c)). Pressure profile y(r)/y(r500c) vs
  Arnaud+2010 universal profile across 4 mass bins (10^13.0–10^14.2)
  matches well to ~2 r500c, diverging beyond (small-number/outer-radius
  noise).
- **Per-halo stochasticity** (WORKLOG 2026-06-11 "WL×tSZ publication plan +
  three findings" implemented-section): "σ(logY|M) BIND 0.23–0.44 vs hydro
  0.24–0.41 dex (+3–6% at groups, matched at clusters); σ(f_gas|M)
  identical." Also (WORKLOG 2026-06-15pm "per-halo baryon-feedback atlas"):
  "σ(logY|M) 0.12–0.22 dex (BIND slightly over-dispersed at group scale =
  calibrated generative scatter, vs deterministic BCMs)" — NOTE these two
  σ(logY|M) ranges (0.23–0.44 vs 0.12–0.22 dex) come from different
  analyses/apertures (map-level peak-stacking vs per-halo atlas R500c) —
  present both with their distinct provenance, don't merge.
- **f_gas(M) match**: "BIND f_gas(M) matches truth 2–11%" (WORKLOG
  2026-06-10, same entry as R(ν)); also "BIND f_gas,500(>5e13)=0.137 vs TNG
  truth 0.132 (~4%)" (WORKLOG 2026-06-15pm "per-halo baryon-feedback atlas");
  also "z~0, group scale 1-3e13, R500c: BIND fid f_gas=0.093 (58% cosmic)
  vs truth 0.086 (54% cosmic)" **[RAW / later flagged LOS-contaminated —
  see Results (3)/Discussion]**.
- **Map gallery**: `demo_maps.png` (root and `figs/demo_maps.png`, both main
  tree) — 3-panel BIND κ(z_s=1) / DMO κ / BIND Compton-y, 5×5 deg. Also
  `fig_skymaps.pdf` (2-panel DMO κ / BIND κ + a 3rd residual-Δκ panel,
  generated by `paper_lightcone_figs.ipynb` §2.3, saved as
  `figures_lightcone/fig_skymaps.pdf`).
- **Tomographic cross-spectra**: `figs_raw/fiducial_lightcone_stats/cell013_out0.png`
  shows Cℓ^κy and Cℓ^yy across 5 source-redshift bins z_s=0.5–2.44.

### (2) R(ν) tracks f_gas·T (not a pure gas fraction)
- **Definition + explicit reframe**: `stats.py` docstring: "`R(nu)=<y>/<kappa>`
  (a specific-thermal-energy proxy: `R ∝ Y/M ∝ f_gas·T_mw` — NOT a pure gas
  fraction)". **Provenance**: `src/bind/inference/stats.py` module docstring
  (worktree `lightcone`).
- **Corr(R, f_gas·T) = 0.79**. **Provenance**: WORKLOG 2026-06-10 "BIND
  validated vs hydro..." ("R shown to be a gas fraction: `R_halo=Y/M ∝
  f_gas·T` (corr 0.79)"); reiterated verbatim in `docs/wl_tsz_plan.md` §0.1
  ("our own corr = 0.79").
- **Mechanism**: "the ×2 rise over ν=1→5 is mostly self-similar thermometry
  (T ∝ M^⅔), not gas content. AGN feedback moves f_gas↓ and T↑ with partial
  cancellation" (`wl_tsz_plan.md` §0.1). This motivated adding the τ plane
  so (κ,τ,y) at peaks separately decompose into (mass, f_gas, T̄).
- **Validation number**: R(ν) BIND/hydro=0.98–1.06, rising ×2 with ν —
  see Results (1) above (same WORKLOG entry).
- Figure candidates: `bind_bridge_hero.png` panel B (ejection–heating
  plane, ΔlnM_gas vs ΔlnT=ΔlnY−ΔlnM_gas, coloured by ΔlnY) is the closest
  available figure to the "R decomposes into f_gas and T" argument, though
  it plots the decomposition directly rather than R(ν) itself.
  `fig_peak_tsz.pdf` (Δκ/κ vs ΔlnM_gas, slope=0.080≈f_gas^proj r=0.50; ΔlnY
  vs Δκ/κ) is a per-peak, AGN/SN-coloured version of the same argument.
  **No standalone R(ν)-vs-parameter curve figure was located on disk** in
  the sources read (the notebook that produced the original R(ν) deep-dive,
  `examples/lightcone_comparison.ipynb`, is referenced only in WORKLOG text
  and was not found in the `lightcone` worktree — flag as **GAP**).

### (3) The halo↔field bridge
- **Bridge I (per-halo, per-peak)**: `bind_bridge_hero.png` (also
  `figs_raw` not needed — file exists directly at
  `/mnt/home/mlee1/BIND/figs/bind_bridge_hero.png`). Δκ/κ (WL peak height,
  lightcone) vs ΔlnM_gas (per-halo atlas), **slope = 0.080 ≈ the projected
  gas fraction** ("the slope IS the bridge coefficient" — gas is ~8% of the
  lensing mass), **per-run r = 0.50** (per-peak scatter ~100% of variance =
  latent morphology + multi-halo projection; this *corrected* an earlier
  "19% latent" claim in the same WORKLOG entry). Panel B: ejection–heating
  plane (κ sees ΔlnM_gas only; tSZ sees ΔlnY=ΔlnM_gas+ΔlnT).
  **Provenance**: WORKLOG 2026-06-15pm "halo↔field bridge — 3 figures for
  the unified 'BIND-TNG' paper"; generator `examples/bind_bridge.py`
  (worktree `wl-tsz-bridge`, module docstring, fig 1 "HERO").
- **Bridge II (van Daalen-style, group scale)**: `bind_bridge_fgas_sofl.png`.
  Group f_gas (M200c 1–3e13, R500c, background-subtracted) predicts the
  measured lightcone suppression S(ℓ)=Cℓ^bind/Cℓ^DMO at ℓ~1-2k with **r =
  0.91** (fit `S = 0.57·f_gas + 0.95`, per `paper_decomp_figs.ipynb` §7.3 as
  quoted in WORKLOG 2026-06-18 "split the figure notebook..."). **57
  feedback lightcones** (of the 60-run twobound OAT design; 57 had landed
  for this analysis), coloured by family: AGN (20), SN/wind (25), other
  (12); fiducial f_gas=0.086, lowest 0.056, highest 0.137.  Group scale is
  the "sweet spot" because "the lightcone is group-dominated." **Provenance**:
  same WORKLOG entry as Bridge I.
- **FLAG — "r=0.93" does not attach to "Δν↔ΔM_gas"**: the brief's
  must-appear list says "bridge r=0.93 / r=0.91". The only r≈0.93 found in
  the sources is **Y_500c (not f_gas) as an even tighter predictor of
  S(ℓ)**: "**Y_500c is a tighter S predictor than f_gas (r=0.935 vs
  0.914)**; partial corr shows Y subsumes f_gas (r[f_gas,S|Y]≈0) while
  keeping residual power (r[Y,S|f_gas]≈0.49 = the heating leg)." Provenance:
  WORKLOG 2026-06-18 "anatomy of the f_gas→S(ℓ) bridge", item 3
  "Compton-Y"; figure `fig_bridge_compton_y.pdf` (panel a: "the Compton-Y
  bridge", r=0.93 annotated on-figure; panel b: Y vs f_gas r=0.98; panel c:
  bar chart of corr-with-S(ℓ) for f_gas 0.91, logY 0.93, logT 0.85, Y|f_gas
  partial 0.48, f_gas|Y partial ≈0). The per-peak Δκ/κ-vs-ΔlnM_gas number
  (which IS closest to "Δν↔ΔM_gas") is **r=0.50** (Bridge I above), not
  0.93. Recommend the draft state r=0.91 (f_gas→S(ℓ), the headline) and
  r=0.93/0.935 (Y→S(ℓ), the sharper predictor) — and NOT attribute 0.93 to
  a Δν/ΔM_gas relation, which the sources don't support.
- **Bridge III (response matrix)**: `bind_bridge_response.png`. |corr(field
  probe, halo Δf_gas group)| across 57 runs: **S(ℓ) 0.92, R(ν) 0.92, N_min
  0.90, C_κy 0.87, V1 0.83, V2 0.82** all driven by halo f_gas; **N_pk the
  exception at 0.28** (peak counts are noise-dominated). **Provenance**:
  same WORKLOG entry ("Fig 3 correlation bars").
- **Mass-bin structure of the bridge**: `fig_bridge_mass.pdf` — group:
  r=0.91, α(slope)=0.59; intermediate: r=0.84, α=0.86; cluster: r=0.59,
  α=1.12 — "slope steepens 0.59→0.86→1.12 toward clusters but r loosens
  0.91→0.84→0.59; group scale is the sweet spot." **Provenance**: WORKLOG
  2026-06-18 "anatomy of the f_gas→S(ℓ) bridge", item 2.
- **van Daalen+2020 functional-form analog**: `fig_bridge_vandaalen.pdf`.
  x = renormalized baryon fraction f̃_bar=(gas+star)/M500c/(Ω_b/Ω_m), fit
  −exp(−5.990·f̃−0.5107) (van Daalen+2020 functional form). **TNG-CAMELS
  spans only f̃~0.81–0.98 — the flat saturated top of the curve** (cannot
  reach the observational band 0.55–0.76 or the steep part) — "THAT'S why
  every fit looked linear (we sample the linear tail of a global
  exponential)." WL ΔCℓ/Cℓ vs f̃ tight (r~0.95), steepens with ℓ, crosses
  into enhancement (>0) at small scales beyond van Daalen's
  suppression-only range. **Provenance**: WORKLOG 2026-06-18, item 5.
- **When/why the bridge breaks**: `fig_bridge_residual.pdf` — below ℓ~1.5k
  thermal quantities (Y, K) predict the residual; above, structural
  quantities (c_dm, c_gas, **f_star→r=0.93**) take over; a 2-variable
  bridge (f_gas + c_dm) restores r 0.75→0.92 at ℓ~8k. Off-bridge residual
  is feedback-mode diagnostic: AGN sits above the bridge, SN/wind below, at
  high ℓ. **Provenance**: WORKLOG 2026-06-18, item 4.
- **Sobol-scale extension**: `fig_cl.pdf` — S(ℓ) envelope for fiducial + 1P
  (n=57) + **Sobol (n=253)**; right panel Pearson r[f_gas^group, S(ℓ)] vs ℓ
  for both designs, peaking ~0.86–0.91 near ℓ~1000–1500 (inset scatter
  "Sobol scatter at ℓ=1037, r=0.86"). This is the canonical suite
  description number: **256-node SB35 Sobol design, 253 usable runs**
  (WORKLOG 2026-06-29 §6 fix, line 58: "253 runs + fid, ~34k halos/run over
  20 shells z=0.034–2.444"). `fig_bridge_sobol.pdf` extends the bridge to a
  Spearman r(mass, ℓ/ν) heatmap for S(ℓ), peaks, minima, V0/V1/V2 at Sobol
  scale.

### (4) Completeness: low-mass reuse + resolution gate
- **Capture fractions**: "**28.7%**" within the production circular paste
  (4×R200, median 1.76 Mpc/h), "**51.6%**" at the full 6.25 Mpc/h patch
  footprint, for the 10¹²–10¹³ M200c band; "the remaining ~48% are field
  halos too far from any cluster." **Provenance**: WORKLOG 2026-06-25
  "low-mass (10¹²–10¹³) baryons via reuse-only, not new generation";
  corroborated in `examples/lightcone_lowmass_reuse.py` module docstring
  ("keeps ~29% of 10¹²-10¹³ halos; widening to the full 6.25 Mpc/h patch
  keeps ~52%").
  - Decision: **reuse-only**, accept ~49% (not the full ~52%, footnote:
    49% vs 51.6% — WORKLOG's decision-line text says "accept ~49%" while
    the measured number is 51.6%; treat 51.6% as the precise measured
    figure and "~49%" as the WORKLOG's own rounding in the decision
    sentence), band 10¹²–10¹³, mirrored on truth (not 10¹¹, not full-box
    tiling, not new generation).
  - Per-halo BIND-vs-truth on the captured population (snap_096): **gas
    1.034× / 0.074 dex**, **Y 0.986× / 0.223 dex** (well recovered
    off-center; median host distance 1.87 Mpc/h), stars 0.75×/0.54 dex
    (two-head channel, expected, map-irrelevant).
  - Map-level (widened footprint `--r200_factor 99`, snap_096): "widening
    boosts total tSZ y +4.0% identically for BIND & truth"; "BIND/truth y =
    0.963 at *both* footprints (no new low-mass bias)"; matter total
    unchanged (patch mass-match) → the low-mass WL signal "lives in the
    power-spectrum redistribution, only visible after lux."
  - No dedicated on-disk figure for this result was located under
    `examples/figures_lightcone/` (81 files inspected) — **GAP**. A
    superficially related figure, `examples/figures_ksz/f6b_lowmass.pdf`,
    exists but belongs to the kSZ/Paper-IV analysis (DESI×ACT BGS/ELG
    f_gas(M) with a "patch reuse" shaded region below 1e13) and should NOT
    be reused for Paper I without modification — it carries DESI×ACT
    content outside this paper's scope.
- **Resolution gate**: `examples/resolution_gate.py` (worktree `lightcone`)
  exists with a full docstring plan (TNG300-2-Dark [8×coarser] /
  TNG300-3-Dark [64×coarser], matched halos via shared initial phases, pass
  criterion "median biases of M_gas/Y within a few % for M>10^13.5") but
  **was never executed** — no results appear anywhere in
  `docs/WORKLOG.md` (grepped `resolution gate|resgate|TNG300-2-Dark|
  TNG300-3-Dark`: only the planning mention in `wl_tsz_plan.md` §3.6 and the
  WORKLOG line "Quijote ruled out → TNG300-2/3-Dark resolution gate" as a
  forward-looking decision-gate note). **GAP**: no resolution-gate numbers
  exist to report; state as a stated-but-unexecuted validation step (per
  `wl_tsz_plan.md` §6 validation checklist, row 6 "High-mass extrapolation
  audit ⬜").
  - Related, resolved feasibility finding: **Quijote is ruled out** for the
    bigger-box extension (1 Gpc/h @1024³ → m_p≈8e10 Msun/h → a 10¹³ halo is
    ~120 particles, far out of training distribution).
    `wl_tsz_plan.md` §3.6.

---

## DISCUSSION — caveats sourced

- **TNG-only prior**: explicit in `wl_tsz_plan.md` §6 validation checklist
  row 8 ("TNG-family prior limitation stated (SIMBA/Astrid retrain =
  future) ⬜ honesty section").
- **Mass floor ≥1e13**: `truth_lightcone.py` docstring + `science.py` design
  — painting operates on FoF halos M≥1e13; the 1e12–1e13 band is
  reuse-only (Results 4), and "1e12-13 halos get cosmic f_b in the [DM/τ]
  proxy until the low-mass extension is trained" (`wl_tsz_plan.md` §2, DM
  design-change bullet).
- **τ is velocity-free electron column**: `lightcone_maps.py` docstring:
  "kSZ" here = tau electron column (no gas velocity field; velocity-weighted
  ΔT_kSZ is unbuilt — needs a DMO-velocity surrogate)." kSZ tier-1
  literature justification (Schaan+21, Hadzhiyska+24, Ried Guachalla+25,
  no arXiv IDs given in-repo) in `wl_tsz_plan.md` §2. Explicitly deferred to
  "Paper IV" per this paper's own Discussion framing in the brief.
- **CIC/aliasing upturn at high ℓ**: WORKLOG 2026-06-09 "diagnosed the
  convergence-power 'upturn' + added anti-aliasing capability" — "BIND is
  correct... The upturn is a raw-CIC aliasing artifact in the
  particle→grid projection (`pixelize_z_projection`), present identically
  in DMO and BIND (common-mode → cancels in same-pipeline ratios)."
  `--mas_correct` fix (interlacing + CIC deconvolution, unit-validated to
  0.4% out to 0.8·k_Nyquist) is opt-in, default off; "For the science
  (S(ℓ), peaks, non-Gaussian stats): ratio against same-pipeline DMO and no
  fix is needed; `--mas_correct` is only for absolute κ-vs-kappaTNG
  validation." Visible directly in `bind_bridge_fgas_sofl.png` left panel
  (S(ℓ) curves spike upward above ℓ~2×10⁴).
- **Painting is 2D projected, not 3D fields**: not found as an explicit
  single-sentence caveat anywhere in the sources read — it follows directly
  from the pipeline description (per-halo 2D patches painted onto 2D
  projected slabs; `bind.model` itself outputs 128×128 2D maps per
  `CLAUDE.md`) but should be stated as an inference from the architecture,
  not a quoted claim. Flag as **synthesized, not directly sourced**.
- **eROSITA / mainstream-X-ray f_gas tension** (motivates the Discussion's
  "what a TNG-family prior can and cannot reach"): see Results (3)/(4)
  cross-reference below — **this is also where the brief's mandated "~41%
  cosmic" number needs correcting**.

### FLAG — the brief's "f_gas saturation ~41% cosmic" number is the RAW, later-corrected value
- **Raw/original claim** (WORKLOG 2026-06-15pm "per-halo baryon-feedback
  atlas (Rung 0) + z=0 f_gas saturation result"): z~0, group scale M200c
  1-3e13, R500c: "cosmic f_b=0.159; BIND fid f_gas=0.093 (58%) ≈ truth
  0.086 (54%). **Strongest single-param depletion = 0.065 (41% cosmic),
  WindFreeTravelDensFac-hi**... **No single-parameter TNG excursion reaches
  the eROSITA strong-feedback band (~20-40% cosmic) — saturates at ~41%.**"
- **Same-day correction** (WORKLOG 2026-06-15pm "eROSITA f_gas confrontation
  + projection-contamination fix", immediately following entry): "**Finding**:
  per-halo patches are projected through the FULL slab... so the aperture
  column carries a cosmic-f_b LOS background (whole-patch f_gas → 0.158 =
  Ω_b/Ω_m; Σ(R) plateaus to cosmic by ~2.5 Mpc/h). Raw aperture f_gas was
  LOS-contaminated. **Fix**: observer-style background subtraction from an
  outer annulus (2.5-3 Mpc/h)... **Corrected result**
  (`fgas_erosita.png`, z<0.2, M500c~1.3e13): BIND fid f_gas,500=**0.076** ≈
  TNG truth ≈ **mainstream X-ray (Eckert) to ~9%** — BIND/TNG f_gas is NOT
  anomalous. **2.9× above eROSITA-Popesso** (declining to ~1.2× at
  clusters); strongest single knob → **0.047** (1.8× eROSITA, closes ~40%
  of the gap). ⇒ reaching eROSITA-low needs combined knobs (Sobol) or a
  non-TNG model... **The earlier raw '41%' + raw f_gas-evolution absolutes
  are LOS-contaminated (relative trends robust).**"
- **A third, independently-computed number** exists from a *different*
  mass-bin/aperture choice (M200c 1–2e13, computed directly in the
  `paper_lightcone_figs.ipynb` §6.1 cell, printed output): "min group f_gas
  reached by a single knob: **0.051** (WindFreeTravelDensFac)"; BIND
  fiducial group f_gas = **0.081**, X-ray-standard band 0.06–0.10,
  eROSITA-low (Popesso+24) = **0.026**. Figure: `fig_siegel.pdf`.
- **Recommendation**: do NOT state "z≈0 group f_gas saturates at ~41% of
  cosmic" as a headline result — that is explicitly the WORKLOG's own
  *retracted* number. Use the bg-subtracted, corrected framing instead:
  BIND fiducial group f_gas ≈ 0.076–0.086 (aperture-dependent) matches
  mainstream X-ray (Eckert+16/19) to ~9% but sits 1.8–2.9× above the
  contested eROSITA-Popesso-low value; the strongest single 1P knob closes
  only ~40% of that gap (values 0.047–0.051 depending on aperture),
  motivating the combined-knob Sobol design as the open question. If the
  brief's editor wants the exact "~41%" phrase kept for continuity, it must
  be captioned as the superseded/raw number with the correction stated
  alongside, per the brief's own instruction to flag WORKLOG corrections.

---

## MANDATORY NUMBERS CHECKLIST (brief §"Numbers that MUST appear")

| Brief number | Status | Sourced value | Provenance |
|---|---|---|---|
| BIND≈hydro at fiducial for WL+tSZ | ✅ confirmed | Cℓκκ BIND/hydro≈1.00; R(ν)=0.98–1.06; y-profile-at-peaks ~5% | WORKLOG 2026-06-10 |
| R(ν)∝f_gas·T | ✅ confirmed | corr(R,f_gas·T)=0.79 | WORKLOG 2026-06-10; `stats.py` docstring; `wl_tsz_plan.md` §0.1 |
| bridge r=0.93 | ⚠ **retarget** | r=0.93/0.935 is Y_500c→S(ℓ), NOT Δν↔ΔM_gas (which is r=0.50) | WORKLOG 2026-06-18 (fgas_bridge_explore); see FLAG above |
| bridge r=0.91 | ✅ confirmed | group f_gas→S(ℓ~1500), 57 runs | WORKLOG 2026-06-15pm; `bind_bridge_fgas_sofl.png` |
| f_gas saturation ~41% cosmic | ⚠ **superseded** | raw/LOS-contaminated; corrected framing = 0.076–0.086 fid, 2.9× eROSITA-low, closes ~40% of gap at best single knob | WORKLOG 2026-06-15pm (two entries, correction wins) |
| low-mass capture 28.7%/51.6% | ✅ confirmed | 4×R200 paste = 28.7%; full 6.25 Mpc/h patch = 51.6% | WORKLOG 2026-06-25; `lightcone_lowmass_reuse.py` docstring |
| TNG300 box + n(snapshots)=20 | ✅ confirmed | box=205 Mpc/h (TNG300), 20 snapshots | `science.py:34`; WORKLOG 2026-06-21 |
| shells z=0.034–2.444 | ✅ confirmed | 20 shells | WORKLOG 2026-06-29 |
| 4×R200c paste aperture | ✅ confirmed | `r200_factor=4.0` default, circular Hann taper | `schemas.py`, `pipeline.py` |

---

## FIGURES

All paths absolute; PDFs render fine with `pdftoppm`/`pdftocairo` for
inspection (already done for the ones below) but should be used as vector
PDFs in the final paper where available.

1. **`/mnt/home/mlee1/BIND/figs/demo_maps.png`** — 3-panel: BIND κ (z_s=1),
   DMO κ, BIND Compton-y, 5×5 deg. Shows: the baryonified field is visually
   distinct from DMO; the y-map traces halo cores. Serves: **Methods §2.3
   (lightcone construction) / Results (1) map gallery**. Provenance:
   `paper_lightcone_figs.ipynb` §2.3 loads `demo_maps.npz`
   (`SCI/demo_maps.npz`); identical content also on disk as
   `examples/figures_lightcone` sibling `fig_skymaps.pdf` (DMO/BIND/residual
   variant, 3-panel, generated by the same notebook cell,
   `examples/figures_lightcone/fig_skymaps.pdf`).
2. **`/mnt/home/mlee1/BIND/examples/figures_lightcone/fig_validation_field.pdf`**
   (also `figs_raw/paper_lightcone_figs/cell006_out1.png`) — 6-panel Cℓκκ,
   N_peak, N_min, V0, V1, V2 with residual sub-panels, BIND (solid) vs
   TNG300 hydro (dashed), residuals within a shaded ±band. Shows: the
   headline fiducial closure. Serves: **Results (1)**. Provenance: WORKLOG
   2026-06-10 + `paper_lightcone_figs.ipynb` §2.4 cell 6 (code loads
   `runs/bind/run_0000` vs `runs/truth/run_0000` `Cl_kappa.npz`/
   `peak_counts.npz`/`nongaussian_stats.npz`).
3. **`figs_raw/paper_lightcone_figs/cell005_out1.png`** (fig_validation_halo,
   notebook version — richer than the on-disk single-panel PDF, see note
   above) — 3-panel: Y–M (BIND vs TNG hydro, scatter band), τ_BIND/τ_hydro
   ratio by mass bin (~1.05–1.15), pressure y(r)/y(r500c) vs Arnaud+2010
   across 4 mass bins. Shows: halo-level thermodynamic closure. Serves:
   **Results (1)**. Provenance: same notebook, §2.4 cell 5.
4. **`figs_raw/fiducial_lightcone_stats/cell013_out0.png`** — Cℓ^κy
   (5 tomographic z_s bins) + Cℓ^yy. Shows: cross- and tSZ-auto spectra at
   fiducial. Serves: **Methods (c) map suite / Results (1)**. Provenance:
   `fiducial_lightcone_stats.ipynb` §6 "tSZ × WL cross-correlation (1a)".
5. **`/mnt/home/mlee1/BIND/figs/bind_bridge_hero.png`** — 2-panel: Δκ/κ vs
   ΔlnM_gas per-peak (slope=0.080≈f_gas^proj, r=0.50) + ejection–heating
   plane (ΔlnM_gas vs ΔlnT, coloured ΔlnY). Shows: the per-halo → per-peak
   bridge, and the R(ν) decomposition mechanism. Serves: **Results (2)/(3)**.
   Provenance: WORKLOG 2026-06-15pm "halo↔field bridge"; generator
   `examples/bind_bridge.py`.
6. **`/mnt/home/mlee1/BIND/figs/bind_bridge_fgas_sofl.png`** — 2-panel: S(ℓ)
   suppression curves for fiducial/lowest/highest f_gas runs (left, shows
   the CIC-aliasing high-ℓ upturn explicitly) + S(ℓ~1500) vs group f_gas
   scatter, r=0.91, 57 runs coloured by family (right). Shows: the van
   Daalen-style bridge headline. Serves: **Results (3)**. Provenance: same
   as #5.
7. **`/mnt/home/mlee1/BIND/figs/bind_bridge_response.png`** — bar chart of
   |corr(field probe, Δf_gas)| across 57 runs (S(ℓ) 0.92, R(ν) 0.92, N_min
   0.90, C_κy 0.87, V1 0.83, V2 0.82, N_pk 0.28) + R(ν)-vs-f_gas scatter
   r=0.92. Shows: f_gas drives essentially every field observable except
   raw peak counts. Serves: **Results (2)/(3)**. Provenance: same as #5.
8. **`/mnt/home/mlee1/BIND/examples/figures_lightcone/fig_bridge_vandaalen.pdf`**
   — 2-panel: TNG-CAMELS position on the van Daalen+2020 f̃_bar curve (flat
   saturated top, f̃~0.81–0.98) + WL suppression relation with r=0.96/0.94/
   0.87 at ℓ=1000/3000/6000. Shows: why the bridge looks linear (sampling
   only the saturated regime). Serves: **Results (3)**. Provenance: WORKLOG
   2026-06-18 "anatomy of the f_gas→S(ℓ) bridge", item 5.
9. **`/mnt/home/mlee1/BIND/examples/figures_lightcone/fig_cl.pdf`** — S(ℓ)
   envelopes (fiducial, 1P n=57, Sobol n=253) with LSST-Y10/Euclid noise
   bands + Pearson r[f_gas,S(ℓ)] vs ℓ for both designs (inset scatter,
   Sobol r=0.86 at ℓ=1037). Shows: the bridge is robust from 1P corners to
   the full Sobol cloud; establishes the canonical suite size (256-node
   design, 253 usable runs). Serves: **Results (1)/(3)**, suite description.
10. **`/mnt/home/mlee1/BIND/examples/figures_lightcone/fig_siegel.pdf`** —
    sorted 1P group f_gas bars vs X-ray-standard band (0.06–0.10) and
    eROSITA-low (Popesso+24, 0.026); BIND fiducial=0.081 line; minimum
    single-knob = 0.051. Shows: the corrected eROSITA-tension framing.
    Serves: **Discussion** (f_gas-saturation caveat, corrected). Provenance:
    `paper_lightcone_figs.ipynb` §6.1, printed cell output.
11. **`/mnt/home/mlee1/BIND/examples/figures_lightcone/fig_bridge_compton_y.pdf`**
    — 3-panel: S(ℓ) vs log Y_500c (r=0.93), Y vs f_gas (r=0.98), bar chart
    of raw vs partial correlations (f_gas 0.91, logY 0.93, logT 0.85, Y|
    f_gas 0.48, f_gas|Y ≈0). Shows: Y is a tighter/deeper bridge predictor
    than f_gas alone — resolves the r=0.93 vs r=0.91 confusion (FLAG
    above). Serves: **Results (3)**. Provenance: WORKLOG 2026-06-18, item 3.

**Not used / candidates deliberately excluded** (leave for Draft step to
reconsider or for `leftovers.md`): `fig_fgas.pdf` (f_gas(M,z) 6-redshift
evolution + Δf_gas envelope — Paper II/twobound-flagship material per
`draft.tex` §3.1, not in Paper I's outline); `fig_peaks_minima.pdf`,
`fig_mf.pdf`, `fig_hierarchy.pdf`, `fig_orthogonality.pdf`, `fig_decomp.pdf`,
`fig_pressure.pdf`, `fig_budget.pdf`, `fig_morphology*.pdf` (twobound
response-library figures, Paper II/III territory per `PLAN.md`'s branch
mapping); `fig_lightcone_population.pdf` (§6 z=0.034-vs-lightcone-halo-
population fix — belongs to the WL-latent story, Paper III); `bcm_warp_
comparison_fid.png` (anisotropy/f_aniso — Paper V); `sb35_sobol_figs/
scaling_relations_z0.png` (not inspected in detail — low priority, Sobol
per-halo scaling, likely Paper III).

---

## CAVEATS (consolidated)

Brief's mandatory list, each with what the sources actually support:
1. **TNG-only prior** — sourced (`wl_tsz_plan.md` §6 checklist row 8).
2. **Mass floor** (M≥1e13 for direct painting; 1e12–1e13 reuse-only,
   ~28.7–51.6% captured) — sourced (Results 4).
3. **Velocity-free τ** — sourced (`lightcone_maps.py` docstring, explicit
   "kSZ needs velocity" caveat).
4. **Aliasing at high ℓ** (CIC upturn, common-mode, cancels in ratios,
   `--mas_correct` available but not needed for ratio-based science) —
   sourced (WORKLOG 2026-06-09).
5. **Painting is 2D projected, not 3D fields** — architectural inference,
   not a directly quoted caveat sentence anywhere found; flag as
   synthesized (see Discussion section above).

Additional caveats found in the sources, not in the brief's mandatory list:
- **Cℓ^yy −12% deficit** at fiducial, open/unresolved per `wl_tsz_plan.md`
  §6 checklist row 1 ("Fiducial closure... ✅ done; Cℓyy −12% open → §3.3");
  candidate cause: mean-y completeness (BIND mean-y 1.05e-6 vs Planck
  ~1.6e-6, "≈35% of ⟨y⟩ from M<1e13 + unbound gas" per §3.3).
- **p14 bug** (CV sims: parameter index 14=0 despite CV files listing
  2000) — general CAMELS caveat from `CLAUDE.md`, not lightcone-specific
  but worth a footnote if CV data is referenced anywhere in this paper.
- **High-mass extrapolation untested past training range**: "CAMELS-50
  training tops out ≈ few×10¹⁴" (`wl_tsz_plan.md` §3.5), audit listed as
  ⬜ not done (checklist row 6).
- **Response-closure test (twobound extremes vs CAMELS hydro at matched
  θ) is unchecked**: `wl_tsz_plan.md` §6 checklist row 2, marked ⬜ "THE
  credibility test that derivatives are physics" — i.e. the paper validates
  fiducial closure but NOT that the parametric *response* (derivative) is
  correct against independent CAMELS hydro runs at the same θ. Should be
  stated as an open validation gap in Discussion.
- **The Δν↔ΔM_gas bridge scatter is ~100% of variance at the per-peak
  level** (only the per-run-averaged r=0.50 recovers signal) — i.e. the
  bridge is a population-level, not an individual-object-level, statement.

---

## CITATIONS

### Brief's seed list (verify all in Cite step; none independently confirmed
by in-repo text search beyond what's listed below)
van Daalen+2011, van Daalen+2020; Schneider & Teyssier 2015; Aricò+2020/21;
Chisari+2019; Mead+2021 (HMcode-2020); Villaescusa-Navarro+2021/2023 (CAMELS
+ SB35); Pillepich+2018, Springel+2018, Nelson+2019 (TNG300); Lipman+2022
(flow matching); DES/KiDS/HSC S8 refs; FRB DM (Macquart+2020); Planck y-map
(Planck 2015 XXII). BIND model paper = in prep.

### Confirmed by name in repo sources (add to references.bib wishlist)
- **van Daalen+2020** — explicitly the functional form fit in
  `fig_bridge_vandaalen.pdf` / WORKLOG 2026-06-18 item 5 ("van Daalen+2020
  for WL... their Fig.16... vd fit −exp(−5.990 f̃−0.5107)"). High
  confidence, directly used quantitatively.
- **Schneider & Teyssier 2015 (BCM)** — cited by name in
  `src/bind/inference/spherical.py` context (WORKLOG 2026-06-17 "faithful
  radial-BCM counterfactual": "a real Schneider–Teyssier BCM reproduces").
- **Arnaud et al. 2010** (universal pressure profile) — used quantitatively
  as the comparison curve in `fig_validation_halo`/`draft.tex` §Pressure
  Profiles ("compared to Arnaud+10 universal profile").
- **Duffy et al. 2008** (NFW concentration-mass) — used in
  `halo_atlas.m500c_from_m200c` ("NFW+Duffy") per WORKLOG 2026-06-15pm
  "eROSITA f_gas confrontation" entry.
- **Popesso et al. 2024** — the eROSITA-low f_gas value plotted in
  `fig_siegel.pdf` legend ("eROSITA-low (Popesso+24)") and WORKLOG
  2026-06-15pm.
- **Eckert et al. 2016/2019** — mainstream X-ray f_gas comparison
  ("Eckert+16/+19 slope 0.21 norm 0.131@2e14"), WORKLOG 2026-06-15pm.
- **Lovisari et al. 2015** — additional X-ray f_gas comparison ("Lovisari+15
  slope 0.16 → 0.070@1e13"), same WORKLOG entry.
- **Macquart et al. 2020** — "Macquart term" named directly in `stats.py`
  docstring (FRB DM mean-amplitude nuisance term), matches brief's seed.
- **Planck 2015 XXII** (or a specific Planck y-map release — exact paper
  not pinned in-repo) — MEMORY note `tsz-ymap-normalization` states
  "validated vs Planck" for the mean-y comparison (1.05e-6 vs Planck
  ~1.6e-6, `wl_tsz_plan.md` §3.3); brief's seed suggests Planck 2015 XXII
  specifically — needs Cite-step verification of which Planck y-map paper
  is meant (2015 XXII tSZ or a later release).

### From `docs/wl_tsz_plan.md` "Key references" (explicit arXiv IDs — highest
confidence, hand directly to Cite step)
2506.07432, 2512.02954, 2509.10455, 2410.19905, 2309.07959, 2503.19441,
2503.19442 (listed as "2503.19441/2"), 2602.10065, 2602.12238, 2505.07949,
2406.01672, 2407.20448, 2312.08450, 2406.08540, 2201.08320, 2109.04458,
2602.11279, 2201.12591.

### From `examples/draft.tex` intro (bibkeys only, no inline arXiv IDs —
lower confidence, cross-reference against the arXiv-ID list above before
adding to .bib; several plausibly correspond 1:1, e.g. `DESACTTSZWL2025`
↔ 2506.07432 based on matching "21σ" description)
Siegel2025, Siegeletal2025mnras, DESACTTSZWL2025, BigwoodAmon2024,
Amonetal2023, Salcido2025kSZ, FLAMINGOkSZlensing2024, Martinet2021peaks,
Gatti2022HOScos, Zurcher2022MFs, Marques2024HSC, FLAMINGOscattering2025,
Fong2021TNGbaryons, MartinezConcepcion2024baryons, HarnoisDeraps2021peaks,
Weiss2019MFs, LeeAmon2026 (NOTE: likely the group's own prior BCM-peaks
paper, MNRAS 519, 573 = arXiv:2201.08320 per `wl_tsz_plan.md` §1 — "Our own
BCM-vs-hydro peaks paper (MNRAS 519, 573; 2201.08320) showed BCMs hold only
to ν≲4" — cross-check this maps to `LeeAmon2026`'s in-text description
"biased cosmological inference even when the power spectrum is correctly
reproduced", which matches), FLAMINGOmps2024, Schaye2023FLAMINGO,
SchneiderTeyssier2015 (↔ Schneider & Teyssier 2015, confirmed above),
Schneider2019BCM, Schneider2025baryonification, Gatti2023maplevel,
VN2021fieldemulator (↔ Villaescusa-Navarro, matches brief seed),
Arico2024 (↔ Aricò, matches brief seed "Aricò+2020/21" — check year),
BindPaper1 (self-citation, in prep), vanDaalen2020 (confirmed above),
Schneider2022clusters, Pandey2022joint, PrestonRogers2025.

### kSZ tier-1 literature (named, no arXiv ID, from `wl_tsz_plan.md` §2)
Schaan+21, Hadzhiyska+24, Ried Guachalla+25 — relevant to the "τ is
velocity-free" Discussion caveat.

---

## GAPS (could not source — do not guess)

1. **R(ν) standalone curve figure** — the notebook that produced the
   original "R(ν) BIND/hydro=0.98–1.06" deep-dive
   (`examples/lightcone_comparison.ipynb`, per WORKLOG 2026-06-10) was not
   found in the `lightcone` worktree; only the summary numbers and
   related-but-not-identical figures (`bind_bridge_hero.png`,
   `fig_peak_tsz.pdf`) exist. Use \todo{R(ν) figure} or substitute the
   related figures with a caption caveat.
2. **Resolution gate results** — `examples/resolution_gate.py` exists as a
   fully-specified but never-executed test; no numbers to report. State
   as a planned/future validation step, not a completed result.
3. **Completeness figure** (28.7%/51.6% capture) has no dedicated on-disk
   plot; only text + a script that computes but does not appear to have
   saved a standalone comparison figure into `figures_lightcone/`. The
   Draft step may need to generate a simple schematic from the dossier
   numbers, or use \todo{completeness figure}.
4. **BIND acronym expansion** ("Baryon-Incorporated N-body with Diffusion")
   is asserted only in `examples/draft.tex`, not corroborated by
   `CLAUDE.md` or any other source read. Use with a caveat or omit the
   expansion, referring to it simply as "BIND."
5. **Exact GPU-hour cost comparison** (BIND lightcone vs a full TNG300 run)
   — `draft.tex` §2.5 flags this as an unfilled placeholder ("Note
   computational cost of BIND vs. full hydro: e.g., a single TNG300 run
   costs ~X million CPU-hours; a BIND lightcone costs ~Y GPU-hours") with
   literal X/Y placeholders never filled in the sources read. No actual
   number found anywhere in WORKLOG. Use \todo{} in the draft.
6. **"n_halos=666" precise meaning** — appears in a halo-for-halo
   verification sentence in WORKLOG 2026-06-21 without stating which
   snapshot/subset it refers to; do not present as "the lightcone's total
   halo count" without further clarification (contrast with the separately
   reported 2,933 M200c>1e13 halos at snap_096 alone in the twobound
   catalog).
7. **Data/code availability specifics** (HF `Maxelee/BIND2` weight names,
   exact GitHub release tag) — `CLAUDE.md` names the HF repo and
   `bind-download-weights {fm_two_head,fm_thermo}` but the lightcone
   pipeline's own released-weights identifier (`weights/fm_redshift_thermo`
   appears in `science.py` as a default) was not cross-verified against
   what's actually on HF; flag for the Draft/Verify steps to confirm before
   stating a specific weights name in the Data Availability section.
