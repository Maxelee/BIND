# P4 — BIND kSZ/gas profiles vs DESI DR2 × ACT DR6 + eROSITA — working plan

Created 2026-06-24 (lightcone branch). Companion to `docs/wl_tsz_plan.md` (the
WL × tSZ program). This document corrects the original P4 spec against what BIND
actually has on disk, and lays out a phased, falsifiable program. Kill-gates are
explicit.

## 0. Premise correction (what the original spec got wrong)

The spec assumed "BIND has already ray-traced **kSZ maps** with the true TNG300
velocity field." Two corrections from the code/data:

1. **BIND has no gas velocity field.** Throughout the code, "kSZ" is shorthand
   for the **electron-column optical depth** `tau = sigma_T integral n_e dl`
   (= FRB dispersion measure), the *velocity-independent* leg
   (`src/bind/inference/lightcone_maps.py`, `paint_tauplane.py`). The only
   "velocity" in `model.py` is the flow-matching vector field. A true
   `Delta T_kSZ = -(sigma_T/c) integral n_e v_r dl` product does **not** exist
   yet. It is buildable: the line-of-sight velocities live in the L205 DMO
   snapshots (Gas channel × **DMO** velocities, a halo-model surrogate), but it
   must be built + validated and labelled as a DM-velocity surrogate, not a
   literal hydro-velocity kSZ.

2. **The Sobol set IS on TNG300 and ray-traced** (this the spec got right, my
   first read got wrong): `bind_sb35/` = 256 Sobol nodes painted on the
   `IllustrisTNG/L205n2500TNG_DM` (205 Mpc/h) lightcone, lux-ray-traced to
   kappa/tau/y. So Sobol × ray-traced maps already coexist on TNG300 geometry.

**Consequence:** the velocity-free legs (f_gas, tau & y *profiles*, kappa×tau,
kappa×y) are the headline and need no new physics. The literal kSZ-temperature
confrontation is an optional, clearly-labelled surrogate (Phase 3).

## 1. Reframed title

"Baryon fractions and tau/y profiles from BIND: confronting the SB35 feedback
manifold with DESI×ACT kSZ + eROSITA X-ray gas constraints." Spirit of P4 (which
feedback strengths match the data?) is preserved; the kSZ-temperature estimator
is demoted from premise to optional surrogate.

## 2. Data inventory (verified 2026-06-24)

- `bind_sb35/runs/run_0000..0255/` — per node: `params.npy`, `Cl_{kappa,kappa_y,tau}`,
  `kappa_maps`, `peak_*`, and `snap_029..096/composite_slab*.npz` with per-halo
  `generated_patches` [DM,Gas,Stars], `thermo_patches` [y,T,S,P_e], `halo_masses`,
  `halo_r200`.
- `bind_sb35/analysis_cache/integrated.parquet` — **8.6M halo-instances × 55 cols**:
  `f_gas_{200,500}`, `M_{tot,gas,star,dm}_{200,500}`, `Y_{200,500}`, `T_mw_{200,500}`,
  `M200`, `r200`, `z`, `a`, + all 30 astro params per row. (The f_gas/Y/T science is
  a query on this.)
- snap→z map (stage1 manifests): 96→0.03 … 67→0.50 … 49→1.04 … 38→1.60 … 29→2.44.
- Reusable code: `examples/halo_atlas.py` (f_gas–M atlas + eROSITA Popesso/Eckert
  bands, Y–M, z-evolution), `examples/tau_profiles.py` (per-halo tau(r) stacks,
  BIND vs truth), `examples/profiles.py` (Rung-1 Sigma(r/r500) per field, gas-vs-DM
  extent), `inference/stats.py` (kappa×tau / tau-auto power).

**Caveats baked into the data.** (a) Halo floor M200 ≥ 1e13 → ELG-host regime
(~10^12.5) is mass-floor-limited; BGS/LRG (group–cluster) are robust. (b) BIND
`f_gas` is *total* gas; eROSITA is *hot X-ray* gas; kSZ is *total electrons* —
compare like with like (X-ray band is a lower envelope, not the kSZ target).
(c) p14 / CV caveats per CLAUDE.md do not apply here (SB35, not CV).

## 3. Phased deliverables

**Phase 1 — velocity-free gas confrontation (weeks, low risk).**
- [x] D1 `examples/ksz_fgas_confront.py` — f_gas(M500,z) for BGS/LRG/ELG windows
      across the 256-node Sobol envelope vs Eckert+19 / Popesso+24-eROSITA, plus
      the BGS/ELG f_gas-inversion-vs-A_AGN test. *First result:* at logM500~13.5
      the whole Sobol band sits at f_gas≈0.11–0.15; **0% of nodes reach the
      eROSITA strong-fb band** even at the 2.5% feedback edge → reproduces the
      Siegel/Bigwood tension (modulo the hot-gas vs total-electron caveat).
- [~] D2 stacked **tau** profiles vs Hadzhiyska+26 GNFW — `examples/ksz_tau_gnfw.py`.
      Their density GNFW (Eq 26-27): `rho=f_b rho_cr(z) rho0 (r/x_c r200c)^gamma
      [1+(r/x_c r200c)^alpha]^{-(beta+gamma)/alpha}`, fixed gamma=-0.5, x_c=0.7,
      free {rho0,alpha,beta}, r/r200c, z_eff BGS 0.26 / ELG 1.17, CAP 1'-14',
      Planck18. Forward-project (no Abel) → tau(R/r200c) shape; stack BIND tau by
      **halo mass** (M* is total-halo not central; ELG hosts below the 1e13 floor),
      bg-subtract, compare shape + (alpha,beta) plane across Sobol nodes.
      *First result (16 nodes, snap85 z0.18 + snap46 z1.16):* in the
      resolution-reliable range x~0.2-1.5 BIND's tau is **more extended (shallower)
      than the steep BGS-all beta=7.3 but consistent with massive BGS (M*>11,
      beta~4.8-5.2) and ELG (beta~4.5)**; beta overlaps (BIND 4-7 vs data 4.5-7.3).
      Caveats: inner x<0.2 pixel-limited, outer x>1.5 bg-subtraction artefact, alpha
      degenerate with the fixed core (β is the clean axis).
      **Done (full 256 nodes, clean-α):** reduce stacks tau AND y in one pass, shard
      per (snap,node) → restart-safe MPI batch `run_ksz_tau.sh` (openmpi+srun, the
      `run_sobol_atlas.sh` idiom) or parallel-serial; clean-α fit restricted to the
      data-sensitive band x∈[0.3,1.5] (drops the inner-core bias). *Result (256
      nodes):* in the clean band BIND's tau SHAPE is consistent with the massive BGS
      bins (M*>11, β~4.8-5.2) and ELG (β~4.5), shallower than steep BGS_all (β=7.3);
      in the α–β plane **β overlaps (BIND 4-6 vs data 4.5-7.3)** but α stays offset
      (BIND 1-3 vs data ~0.2). Conclusion: **α≈0.2 is a near-singular GNFW corner
      (it collapses the profile, cf the f_gas test) and is strongly degenerate with
      the fixed core — so the valid comparison is the projected-profile OVERLAY +
      β, not the (α,β) point.** Figs `tau_shape_vs_hadzhiyska.png`,
      `alpha_beta_plane.png`.
- [x] D3 **y (tSZ pressure) leg + temperature** — same `ksz_tau_gnfw.py` pass stacks
      Compton-y; `tau/y ∝ 1/(k_B T_e)` → temperature decomposition (spec Eq. 9).
      *Result:* at fixed logM200~13.6 the ELG-z (1.16) pressure y exceeds BGS-z
      (0.18) (higher ρ_cr(z)); T_e proxy ≈ 1-2×10^7 K (~1-2 keV), declining with
      radius — physically sensible. Fig `y_and_temperature.png`. (Internal/
      self-consistency; no Part-II y target -- compare to ACT tSZ stacks later.)
- [~] D4 **amplitude / f_gas leg.** Robust piece: BIND f_gas/f_b (parquet, physical)
      ≈ **0.78–0.80 within r500** (Sobol 16-84: 0.65–0.89) for BGS/LRG/ELG-mass
      windows — gas-RICH vs the kSZ "low BGS f_gas" finding, consistent with D1.
      Absolute BIND tau(R) amplitude vs feedback in `tau_amplitude.png`.
      **Form CONFIRMED (paper Eq 26-27, user-verbatim):** ρ_gas=f_b ρ_cr(z) ρ0
      (r/x_c r200c)^γ [1+(r/x_c r200c)^α]^{-(β+γ)/α}, γ=-0.5, x_c=0.7, **ρ0
      dimensionless**. `project_tau_absolute()` gives physical τ (τ(0.5r200c)≈1e-5),
      Fig `tau_absolute_vs_hadzhiyska.png`. The earlier "unphysical f_gas" was the
      WRONG metric (integrating a kSZ *fitting function* to r200c is dominated by the
      unconstrained steep small-r extrapolation), NOT a units bug.
      **Real gate = halo-mass matching.** Absolute τ ∝ halo mass; BIND τ(0.5r200c)≈
      3-5e-4 (logM200 13-13.8) vs Hadzhiyska ≈1.2-1.6e-5 — a ~30× gap that is
      **mass-driven**, because the DESI×ACT samples are LOW mass (user: ELG mean
      log M200c≈12.2 h⁻¹, below BIND's 1e13 floor). So BIND can do the SHAPE
      confrontation (mass-robust) but an absolute matched-mass comparison only for
      whatever BGS bins sit ≥1e13. TODO: per-sample M200c (only ELG≈12.2 known) →
      pick the BIND-overlapping bins; LRG (Part I 2604.19744, more massive).
      **M*-matched closure (`--reduce-mstar`, `run_ksz_tau.sh MODE=mstar`):** bin
      by BIND's *painted central* M* (Stars ch within 50 kpc/h ≈ central galaxy;
      varies per node as feedback must) → matches Hadzhiyska's M* bins directly, and
      BIND's own mean M200 sets r200c (TNG SHMR). BIND central M*>11.0/>11.25 →
      mean logM200 13.56/13.81 (sensible BGS hosts). *Result:* at matched M*/M200,
      BIND τ(0.5r200) ≈ 5-6e-4 vs Hadzhiyska GNFW ≈ 2e-5 → **ratio ~27-39×**, BIND
      profile shallower (more extended). ⚠ **Too large to be purely physical** (D1
      f_gas tension is only ~2-4×): the GNFW projection implies f_gas(<r200)~4% of
      cosmic — implausibly low and at odds with the paper's "missing baryons
      RECOVERED" headline → the absolute projection of their dimensionless-ρ0 GNFW is
      under-normalized (or not meant to be volume-integrated). Direction (BIND
      retains gas the data shows ejected = TNG feedback too weak) matches
      Siegel/Bigwood, but the FACTOR needs a **calibration anchor**: the paper's
      reported f_gas per sample, or the measured τ(θ) data points (compare BIND
      directly, bypass the GNFW). Figs `tau_mstar_matched.png`,
      `tau_absolute_vs_hadzhiyska.png`. Central-M* proxy still wants validation vs
      TNG Subfind at fiducial (DMO↔hydro match).
- [x] **D4 RESOLVED — the proper observable** (`examples/ksz_cap_compare.py`). The
      DESI×ACT data are **T_kSZ^CAP**, the compensated-aperture (disk-minus-ring) τ
      in **arcmin²**, NOT the local τ(R) we compared to the GNFW — that mismatch (plus
      the under-normalized GNFW) is the entire spurious ~30×. Applying the SAME CAP
      filter to BIND's M*-matched τ(θ) and overlaying the digitized BGS M*-split
      points → BIND agrees to within a small factor (NOT 30×). **Normalization nailed
      (task 1, paper-confirmed):** (r/r_fid) is the velocity-reconstruction *fidelity*
      coeff (=1 at fiducial; BGS r=0.64/ELG r=0.55) — NO radial scaling; CAP = disk(+1)/
      equal-area-ring(−1) to √2θ_d; **σ_v^true≈300 km/s** ⇒ T_CMB σ_v/c = 2727 µK (my
      first-pass 1176 was a digitization slip). With σ_v=300: **BIND ≈ 2-6× above the
      data at small/mid aperture (2.5-5′), converging to ~1× at 7.5′** — i.e. BIND's
      gas is too CENTRALLY concentrated; the data show it pushed out = TNG feedback
      too weak, ~2-4σ inner (35% errors). Coherent across legs (D1 f_gas 2-4×,
      Siegel/Bigwood). Fig `ksz_cap_matched.png`. Residual caveats: digitized points;
      M* aperture (50 kpc/h); single σ_v for all samples.
- [x] **D4 capstone — σ_v-FREE f~gas confrontation** (`examples/ksz_fgas_profile.py`).
      The papers also report f~gas = f_gas/(Ω_b/Ω_m) vs aperture (their full velocity
      model → ~1 at large θ), which **bypasses σ_v entirely**. BIND's clean **3D**
      f~gas from the per-halo catalogue (no projection): f~gas(<r500)=0.82,
      **f~gas(<r200)=0.88** vs the DESI×ACT BGS data ≈**0.32** at the same aperture →
      **BIND ~2.7× too gas-rich within r200**; the data reach BIND's level only at
      θ≈7-8′ (~3-4 r200) = gas pushed out, BIND retains it centrally. (BIND's
      *projected* cumulative f~gas overshoots cosmic at large θ — 2-halo not removed —
      shown inner-only, shape-only.) The cleanest, normalization-independent statement
      of the result. Fig `ksz_fgas_profile.png`.
- [ ] D3 kappa×tau and kappa×y cross-spectra across Sobol (reuse `stats.py`):
      feedback dependence of the gas-vs-DM cross-correlation amplitude.

**Phase 2 — likelihood / posterior (the new claim).**
- [~] D5 **GP posterior on feedback** — `examples/ksz_posterior.py`. GP (ARD Matérn,
      sklearn) per clean-band shape-bin over the 256 Sobol nodes (CV R²≈0.4-0.7),
      data = MC over Hadzhiyska BGS_Ms11.25 (α,β) errors → shape mean+cov, emcee
      (vectorized) under the Sobol-box prior. *Result:* the kSZ τ-SHAPE alone
      **weakly constrains** feedback — only ~5/30 params at <0.85× prior width
      (RadioFeedbackReorient 0.73, UVBH0beta/WindFreeTravelDensFac/
      WindEnergyReductionExponent/MinWindVel ~0.83-0.85); the rest stay prior-wide.
      Consistent with feedback being ~2-3D ([[sobol-feedback-latent]]); shape-only is
      info-poor → the constraint needs the (mass-matched) ABSOLUTE amplitude + multi-
      tracer/mass. Figs `ksz_posterior_{corner,constraints}_*.png`. NB conditioned on
      a bin whose true host mass may be <1e13 (see D4 mass gate).
- [x] D5-CAP **posterior on the real observable** — `examples/ksz_posterior_cap.py`.
      Data vector = absolute tau^CAP(theta) (7 apertures) for BGS M*>11.25; GP(30
      params -> log tau^CAP) emulates **well** (CV R²≈0.83 at every aperture — much
      better than the shape's 0.4-0.7, the amplitude responds smoothly to feedback).
      emcee vs the digitized DESI×ACT points (35% errors). *Result:* still
      **data-limited** — no param pinned (top UVBHepDeltaz 0.86× prior; clearest
      single signal a ~2σ preference for higher QuasarThreshold; main AGN/SN
      amplitudes flat, e.g. BHFeedbackFactor -0.08σ). The ~2× BIND-vs-data offset is
      within the errors, so the data can't yet demand stronger feedback. **Pipeline
      success (emulator works), constraint limited by kSZ SNR + single bin**, not by
      BIND. Figs `ksz_posterior_cap_{corner,constraints}.png`. Power scales with:
      tighter absolute norm (D1-task), combining M* bins + ELG/LRG, full data cov.
      *Re-run with corrected σ_v=300 (BIND now ~2-6× high):* still data-limited —
      most-constrained WindEnergyIn1e51erg 0.81× prior; clearest single shift
      QuasarThreshold +2.2σ; the main AGN/SN amplitudes move <0.5σ in a degenerate
      combination (no clean "which knob"). Confirms: the kSZ amplitude offset is real
      (~2-4σ inner) but a single BGS bin can't isolate the responsible parameter in
      the ~2-3D feedback subspace — needs multi-bin/tracer + smaller errors.
      *Joint BGS M*>11.0 + M*>11.25 (`--joint`):* the M*-trend helps modestly — best
      param 0.86→0.77 (VariableWindVelFactor), #params<0.85 prior 0→5, constraint
      shifts to the wind/SN sector (VarWindVel, WindEnergy, WindFreeTravelDens) — the
      M*-dependence of gas tracks SN feedback scaling. ⚠ diagonal cov treats the
      nested M* bins as independent → optimistic; ELG unreachable (floor), LRG needs
      Part I data. Real jump needs published cov + LRG + tighter kSZ errors.
- [ ] D5b GP (ARD Matérn) over the Sobol grid on the D2 data vector (tau/y profiles
      + f_gas), emcee/dynesty → posterior on (A_AGN, A_SN, …). First CAMELS-style
      continuous-parameter posterior from kSZ/eROSITA gas data.
- [ ] D5 z-evolution f_gas(z), beta(z) for preferred nodes vs DESI redshift splits.
- [x] **D6 MULTI-PROBE (kSZ τ + tSZ y)** — `examples/ksz_posterior_multiprobe.py`. kSZ
      = gas density, tSZ = pressure (~thermal energy) → breaks the density-temperature
      degeneracy (heating vs ejection). GP(30 params → [τ^CAP, y^CAP]), **realistic
      AR(1) aperture covariance** (not diagonal), three posteriors (kSZ/tSZ/joint).
      *Current DESIxACT SNR (35%):* data-limited (0-1 params <0.85 prior); tSZ adds
      the AGN/thermal-wind directions kSZ misses. *Future SO/CMB-S4xDESI (10%):* the
      data pin **~2 effective feedback directions** — dir1 (post var 0.10) = wind/SN-
      energy + AGN (+VarWindVel −BHRadEff +WindEnergy +QuasarThr), dir2 (var 0.16) =
      SNIa/radio; **adding tSZ tightens both vs kSZ alone**. Shown as a constrained-
      DIRECTIONS corner (feedback is ~2-3D, so per-param corners are prior-dominated/
      noisy — the direction corner is the honest clean figure). Figs
      `ksz_multiprobe_{corner,bars}.png`; paper notebook Fig 6. NB GP training now
      parallel (joblib, no CV) → ~20× faster; sklearn-GP+emcee are CPU (the V100 is
      for BIND painting/training, not this). TODO: published bin-bin cov; κ (WL) as
      the 3rd leg → full (κ,τ,y) decomposition; real ACT tSZ data (forecast→fit).

- [x] **D7 FIELD-LEVEL WL × SZ** — `examples/wl_sz_multiprobe.py` (the RIGHT way to
      use κ). Per-halo κ stack = feedback-blind mass anchor (kept only for the (κ,τ,y)
      CGM decomposition `ksz_thermo_decomp.py`); κ's feedback power is FIELD-LEVEL.
      Built on the assembled `bind_sb35/emulator/emulator_dataset.npz` (123 nodes × 30
      params × 5 z_s; has S(ℓ) suppression, cl_kappa_y/yy/kappa_tau, peaks/minima/MFs/
      PDF/WST/moments, scaling f_gas/Y/T, + per-bin errors). GP(params→stats) + emcee,
      constrained-DIRECTIONS corner. *Result (forecast z_s=2):* WL **auto** (S(ℓ)+high-ν
      peaks) barely constrains (#params<0.85=0); **SZ (κ×y cross + y auto) DOMINATES**
      (#<0.85=4, best 0.58); joint pins 2 dirs to var **0.06/0.11** (wind/SN+AGN). The
      κ×y cross is the single most feedback-sensitive observable (Sobol spread 0.76 vs
      peaks/WST ~0.08). **Lensing's feedback value = mass-weighting of the pressure
      (κ×y), not a standalone probe** (matches the weak κ-auto SBI). Paper Fig 7; memory
      [[wl-sz-kappay-feedback]]. This reframes the program as **multi-probe WL×SZ**, not
      kSZ-only. TODO: real ACT y-map / DES×ACT shear×y data (forecast→fit); the full
      tomographic κ Cl; published covariance.

- [x] **D8 shear×y fit pipeline + tomography + per-halo fold** (overnight autonomous;
      `examples/wl_sheary_fit.py`, `examples/perhalo_plus_kappay.py`). No real shear×y
      data on disk, so built the cosmology-robust fit PIPELINE (real-data hook
      `DATA_NPZ`) and ran a DESxACT-precision forecast. Findings:
      1. **Cosmology degeneracy is the key methodological point.** The κ×y feedback
         signal is amplitude-dominated (gas pressure), degenerate with σ8/Ωm. With an
         overall amplitude MARGINALISED (cosmology-robust), tomographic shear×y pins 2
         feedback dirs to var **0.25/0.28** (modest); the strong var-0.06 (D7) needed
         the absolute amplitude at fixed cosmology. **Feedback from WL×tSZ needs a
         cosmology anchor.** Tomography (5 z_s) helps vs single plane. Paper Fig 8.
      2. **κ×y is redundant with the per-halo y.** Folding field-level κ×y into the
         per-halo (τ+y) CAP posterior gives NO gain (#<0.85 1→1) — both are
         gas-PRESSURE probes. The complementary leg is mass/density (WL κ-auto, X-ray
         f_gas), not more pressure. `perhalo_kappay_corner.png`.
      **Meta-lesson:** multi-probe feedback gains SATURATE — the strong probes (tSZ y,
      κ×y) are mutually redundant and feedback is ~2-D; the real levers are higher SNR,
      a cosmology prior (breaks the κ×y amplitude degeneracy), and pairing pressure
      (SZ) with an INDEPENDENT density/mass probe (kSZ τ, X-ray), not stacking pressure.
- [x] **D9 COSMOLOGY ANCHOR — the self-consistent WL×SZ analysis**
      (`examples/wl_sz_cosmo_anchor.py`, paper Fig 9). Resolves the D8 degeneracy by
      adding the WL κ-AUTO Cl with an explicit cosmology amplitude A~σ8 Ωm^0.5:
      `C^{κκ}∝A²` (cosmic shear pins A), `C^{κy}∝A¹` (freed once A is set). Result:
      κ-auto pins **A=1.00±0.06 → 0.999±0.008 (8× tighter)**, which tightens the κ×y
      feedback dirs from the cosmology-agnostic var 0.25 (D8) to **var 0.15/0.17** —
      recovering most of the way toward the fixed-cosmology ideal 0.06 (D7). **The
      hierarchy fixed-cosmo 0.06 < κ-auto-anchored 0.15 < amplitude-marginalized 0.25
      is the complete, physical story.** shear×y + κ-auto = the self-consistent
      WL×SZ feedback analysis. TODO: real DES×ACT shear×y + DES κ-auto data (the
      pipeline `DATA_NPZ` hook); joint cosmology+feedback (vary cosmology, needs a
      multi-cosmology BIND suite — currently TNG-only).

- [x] **D10 FIRST REAL-DATA fit — BIND shear×y vs DES Y3 × ACT** (not a forecast!).
      Data: Pandey `shivampcosmo/ACTxDESY3` `DES_ACT.fits` → `desact_data.npz`
      (compton_shear 4 src×20 θ = 80 pts @ 26.5σ, ξ± cosmic shear, joint 480² cov,
      DES n(z)); also itrharrison ACT κ_CMB×DESγ SACC (cosmology cross). Pipeline
      (`examples/desact_sheary_realfit.py`): BIND C^{κy} → n(z)-weight per DES bin →
      Hankel J₂ → ξ_γy(θ). **Found+fixed a real bug:** Pylians `XPk_plane` normalises
      its power ≠ `Pk_plane` (constant F≈1.2016e7) → the stored κ×y cross was ~1e7 too
      low (autos were fine; pixel corr(κ,y)=0.52 confirmed maps OK). Patched
      `bind.inference.stats.power_spectrum` (cross now = x.r·√(Pk_plane autos)); the
      FORECASTS are unaffected (relative). **Result (honest, after a self-correction):**
      my first figure ("BIND ~2× high at small θ = feedback too weak") was a PIPELINE
      ERROR — I omitted the y-map instrument beam (BIND's y is beam-free; the data y has
      a ~10' effective beam → suppresses small-scale) and applied no scale cuts. With
      the y-beam + a θ>8' cut, BIND reproduces the data's intermediate-θ peak SHAPE but
      **over-predicts at large θ by ~1.5-2×** (the 2-halo regime). That residual is
      ENTANGLED with cosmology (TNG vs DES σ8, ~1.2×), the exact beam/scale-cuts, and the
      κ×y normalisation — **no clean feedback claim from shear×y yet**. The robust
      feedback signal stays in the inner CGM (kSZ-CAP, X-ray f_gas), not large-θ
      shear×y. Paper Fig 10 (corrected: no-beam dotted vs beamed solid + scale cut).
      TODO: published y-map beam + scale cuts; joint ξ±+compton_shear fit WITH the
      cosmology anchor (marginalise σ8); regenerate the 256-run Cl_kappa_y + emulator
      dataset with the patched code; low-z source planes for bins 1,2.

**Phase 3 — optional kSZ-temperature surrogate (clearly labelled).**
- [ ] D6 `tau × v_r(DMO)` → halo-model `Delta T_kSZ` maps + stacking estimator
      (sign-flip by v_los). For forecast/illustration + connection to halo-model
      projected-field kSZ; NOT a literal hydro-velocity comparison.

## 4. Kill-gates

- **G1 (after D2):** if BIND tau/y GNFW profiles cannot be made consistent with
  *any* Sobol node within Hadzhiyska errors even after the hot-gas/electron and
  aperture corrections → that is the falsification result (spec Regime 3): TNG's
  feedback range is too gas-rich. Write it up as such; do not force-fit.
- **G2 (after D4):** if the posterior is prior-dominated (kSZ/f_gas barely
  constrains the 30-d space beyond the ~2–3 effective directions already known
  from [[sobol-feedback-latent]]) → report the constrained subspace + an upper
  limit, don't oversell a full posterior.
- **G3 (Phase 3):** only build D6 if a reviewer/collaborator needs the literal
  estimator; the science verdict should not depend on it.

## 5. References (corroborated in `wl_tsz_plan.md` / user notes)

Siegel/Bigwood consensus 2509.10455; Kovač+ 2507.07991; Hadzhiyska+26 (DESI×ACT
BGS/ELG GNFW); Ried Guachalla+25 (DESI×ACT tau); Schaan+21, Amodeo+21 (ACT×BOSS
kSZ profiles); Popesso+24, Eckert+19 (X-ray f_gas).
