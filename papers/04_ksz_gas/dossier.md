# Dossier — Paper IV: Gas thermodynamics vs DESI×ACT kSZ + eROSITA

Sources read: `docs/WORKLOG.md` entries 2026-06-24 (×4), 06-25 (×2), 06-26 (×3), 06-15(pm)
(×2, eROSITA confrontation + per-halo Rung-0 atlas); memory files
`ksz-p4-tau-not-velocity.md`, `baryon-atlas-rung0.md`; worktree
`wt/ksz-desi-act/docs/ksz_desi_act_plan.md`; worktree `examples/{paper_ksz_desi_act.ipynb,
paper_ksz_field.ipynb, _build_ksz_paper_nb.py, _build_ksz_field_nb.py, ksz_fgas_confront.py,
ksz_tau_gnfw.py, ksz_cap_compare.py, ksz_fgas_profile.py, ksz_posterior*.py,
ksz_thermo_decomp.py, _reduce_fgas_cap.py, _reduce_fgas_lowmass.py, _reduce_fgas_radial.py,
_reduce_kappa_bgs.py, _reduce_ycap_lrg.py, _reduce_tauy_fiducial.py}`; on-disk
`examples/figures_ksz/` (13 PDFs, main tree, gitignored) — identical figure set to the
notebook's own `f1..f8[a-e]` cells (verified 1:1 by filename/section match).

**Reading note on chronology.** WORKLOG is reverse-chronological (newest first). Within
2026-06-24/25/26 there were *several* corrections to the same headline numbers as bugs were
found; I give the **final, most-corrected** value in Results and flag every retraction
explicitly. Two corrections are important enough that they invalidate specific numbers the
brief asked for — see the boxed notes in Results (1) and (5).

---

## INTRO — sourced framing

- P4 premise correction (from `ksz_desi_act_plan.md` §0, and WORKLOG 2026-06-24 "P4 kicked
  off"): the original spec assumed BIND had ray-traced literal kSZ (ΔT) maps with the true
  TNG velocity field. **Correction: BIND has no gas velocity field.** Throughout the code
  "kSZ" = the velocity-free electron-column optical depth `tau = sigma_T ∫ n_e dl` (=FRB
  dispersion measure), computed in `src/bind/inference/lightcone_maps.py` /
  `paint_tauplane.py`. A literal `ΔT_kSZ = -(σ_T/c)∫n_e v_r dl` does not exist; building one
  needs DMO-velocity surrogate (Phase-3, not built; kill-gate G3). — provenance:
  `ksz_desi_act_plan.md` §0, memory `ksz-p4-tau-not-velocity.md`.
- `bind_sb35` **is** on TNG300 geometry and ray-traced: 256 Sobol nodes painted onto
  `IllustrisTNG/L205n2500TNG_DM` (205 Mpc/h), lux-ray-traced to κ/τ/y — so the velocity-free
  legs (f_gas, τ/y profiles, κ×τ, κ×y) need no new physics. — `ksz_desi_act_plan.md` §0.
- Why single-simulation comparisons can't separate parameter- from model-tension: this is
  the thesis stated directly in the brief and matches the design description in
  `ksz_desi_act_plan.md` §2 ("same halos × 256 feedback nodes").

## METHODS — sourced details

- **Suite recap** (cite Paper I, in prep.): 256-node SB35 Sobol design painted onto shared
  TNG300-DM halos; **253 usable/valid Sobol nodes** for the ray-traced field-level products
  (`emulator_dataset.npz`, per `paper_ksz_field.ipynb` §0: "precomputed for the 253 valid
  Sobol nodes"). Per-halo (non-ray-traced) products use all 256 nodes minus per-analysis
  NaN/finite masks (e.g. D1's f_gas query, D2-D4's τ/y stacks).
- **8.6M-halo integrated catalog**: `bind_sb35/analysis_cache/integrated.parquet` = "8.6M
  halo-instances × 55 cols": `f_gas_{200,500}`, `M_{tot,gas,star,dm}_{200,500}`, `Y_{200,500}`,
  `T_mw_{200,500}`, `M200`, `r200`, `z`, `a`, + all 30 astro params per row. —
  `ksz_desi_act_plan.md` §2, `ksz_fgas_confront.py` docstring, memory
  `ksz-p4-tau-not-velocity.md`.
- **CAP filter stacks**: compensated aperture photometry, disk (+1, r<θ_d) minus
  **equal-area** ring (-1, θ_d<r<√2θ_d) — `examples/ksz_cap_compare.py` docstring,
  `_reduce_fgas_cap.py` docstring (citing Ried Guachalla+25 §III.2.1 verbatim: "the value of
  the CAP filter for the gas at the virial radius with that of the CAP filter for the
  matter"). CAP-ratio f̃_gas = CAP_gas(θ_d)/CAP_mat(θ_d)/(Ω_b/Ω_m) is the exact DESI×ACT
  observable — NOT a cumulative aperture (methods pitfall, see below).
- **GNFW profile comparison** (Hadzhiyska+26 per code comments — see Citations §
  for the year/author ambiguity that needs resolving): density
  `rho_gas(r) = f_b·rho_cr(z)·rho0·(r/(x_c r200c))^gamma·[1+(r/(x_c r200c))^alpha]^(-(beta+gamma)/alpha)`,
  fixed γ=-0.5, x_c=0.7, free {ρ0,α,β}, forward-projected (no Abel inversion) to τ(R/r200c).
  — `examples/ksz_tau_gnfw.py` docstring (Eq. 26-27 of the source paper).
- **σ_v-free f̃_gas estimator**: f̃_gas = f_gas/(Ω_b/Ω_m) — bypasses the survey's velocity
  reconstruction and σ_v calibration entirely; computed two ways — (a) clean 3D from the
  per-halo catalog (no projection) and (b) a continuous projected curve from the painted
  patches, ΣM_gas(<R)/ΣM_tot(<R)/(Ω_b/Ω_m). — `examples/ksz_fgas_profile.py`,
  `_reduce_fgas_radial.py` docstrings.
- **Posterior machinery**: GP (ARD Matérn, sklearn, joblib-parallel) emulator trained on the
  256 Sobol nodes mapping 30 params → CAP observable(s); emcee sampling under the Sobol-box
  prior against the real/mock data + covariance (population likelihood over per-halo/per-bin
  stacks). — `ksz_posterior_cap.py`, `ksz_posterior_multiprobe.py` docstrings.
- **State plainly (mandatory per brief): BIND "kSZ" is the velocity-free electron column τ**
  — confirmed above; amplitude comparisons enter through the τ (CAP-ratio) observable, never
  a reconstructed ΔT.
- **Methods-lesson footnote — mass convention (2.6×) bug** (NOT a result, per brief):
  the tSZ y-CAP originally selected LRG hosts at logM200∈[13.2,13.5] **Msun/h**, but the
  DESI-LRG host mass from ACT CMB-lensing is logM200=13.18 **Msun/h** (Sailer et al. 2024) —
  the old band was ~0.2 dex too high. Because Y∝M^5/3, that alone inflated the y-CAP to a
  spurious **2.6×**. Fixed: band → [13.0,13.35] Msun/h, mean-stack (not median). — WORKLOG
  2026-06-26 "tSZ mass-selection bug"; `_reduce_ycap_lrg.py`.
- **Two further normalization pitfalls worth a methods paragraph:**
  1. Comparing BIND's *local* τ(R) to the forward-projected Hadzhiyska GNFW gave a spurious
     **~30×** "tension" — the DESI×ACT data are actually `T_kSZ^CAP`, the compensated-aperture
     signal in arcmin², a different observable from local τ(R); the GNFW's dimensionless ρ0
     was also under-normalized for volume integration. — WORKLOG 2026-06-24 D4-resolved;
     memory `ksz-p4-tau-not-velocity.md`.
  2. Even after switching to the CAP filter, using a **cumulative** (not compensated) aperture
     f̃_gas keeps the ~50 Mpc/h line-of-sight background that the true CAP ratio is designed to
     cancel — this manufactured an earlier, now-retracted "0/256 nodes reach the data" claim
     (see boxed note in Results 1/2 below). Also found+fixed: an angular-diameter-distance bug
     (`DA(z)` had a spurious extra `/h`, inflating D_A by 1.476×) that shrank θ(r200) and biased
     the derived data f̃_gas(r200) low (0.28 → corrected 0.44–0.60). — WORKLOG 2026-06-25
     "P4 notebook refocus" §(f).

---

## RESULTS

### (1) f_gas–M across the design vs eROSITA: 0% coverage

**Headline (robust, repeatedly confirmed, use as stated):** at logM500 ~ 13.5 the whole
256-node Sobol band sits at **f_gas ≈ 0.11–0.15**; **0% of nodes reach the Popesso+24/eROSITA
strong-feedback band — even at the 2.5% strongest-feedback edge of the design.** This
reproduces the Siegel/Bigwood (arXiv:2509.10455) tension across a *continuous* feedback
parameter space rather than a single simulation.
— Provenance: WORKLOG 2026-06-24 "P4 kicked off — D1 done" (`ksz_fgas_confront.py`);
re-stated identically in `ksz_desi_act_plan.md` §3 Phase 1 D1. Figure: `f5_fgas_erosita`
(notebook cell012_out0.png) — viewed directly; shows BIND Sobol 16–84% band (~0.10–0.155),
median, strongest-fb (2.5%) dotted line all *above* both the Popesso+24 eROSITA strong-fb
curve and (mostly) the Eckert+19 mainstream X-ray curve, over M500c = 10^13–3×10^14 M⊙/h.

> **⚠ BOXED CORRECTION — the brief's "~41% cosmic f_gas saturation" number is SUPERSEDED /
> an artifact. Do not use it.** That number comes from a *different, earlier* dataset — the
> 60-run **twobound one-at-a-time (OAT)** design (not the 256-node Sobol), reduced by
> `examples/halo_atlas.py` — and was explicitly flagged by the *same session* that produced it
> as **LOS-contaminated**: per-halo patches are projected through the *full* slab depth, so a
> raw (non-background-subtracted) aperture's f_gas plateaus toward the cosmic value
> (Ω_b/Ω_m=0.159) regardless of feedback. WORKLOG 2026-06-15(pm) "eROSITA f_gas
> confrontation..." explicitly states: *"the earlier 'saturates ~41% cosmic' was a projection
> artifact."* Memory `baryon-atlas-rung0.md` states in capitals: **"SUPERSEDED: the earlier
> raw-aperture 'f_gas saturates ~41% cosmic, doesn't reach eROSITA' was the LOS-contaminated
> number — ignore it."**
>
> **The corrected (background-subtracted, single-fiducial-run + OAT) number** is different in
> kind: at z<0.2 (M500c~1.3e13), **BIND fiducial f_gas,500 = 0.076 ≈ TNG truth (0.5% level) ≈
> mainstream X-ray (Eckert+19, ~0.070–0.131) to ~9%** — i.e. BIND/TNG is *not* anomalous
> against the mainstream X-ray literature. The tension is specifically with the *contested*
> eROSITA-Popesso low value (2.9× above it at group scale, declining to ~1.2× at cluster
> scale). The **strongest single OAT-parameter excursion** reaches f_gas,500 = 0.047 (1.8×
> eROSITA — closes **~40%** of the gap but does not reach it): "No single-parameter TNG
> excursion reaches the eROSITA strong-feedback band ... saturates at ~41% [cosmic]" is the
> *original* (pre-bg-sub) phrasing of this OAT ceiling result, and the compatible
> **post-bg-sub number the paper should quote instead is 0.047 (1.8× eROSITA-Popesso,
> single-knob) vs 0.076 (fiducial)**, not "41% cosmic." — Provenance: WORKLOG 2026-06-15(pm)
> ×2 ("per-halo baryon-feedback atlas (Rung 0)" + "eROSITA f_gas confrontation... fix"); memory
> `baryon-atlas-rung0.md`.
>
> **Recommendation for the writer:** lead Result (1) with the robust, Sobol-wide **"0% of 256
> nodes reach the eROSITA strong-fb band"** (D1, `ksz_fgas_confront.py`) as the paper's
> headline; if a "reachable ceiling" statement is wanted, use the corrected OAT number
> (single-knob 0.047, 1.8× eROSITA, ~40% of the gap closed) rather than "41% cosmic," and cite
> it explicitly as coming from the earlier one-at-a-time design, not the Sobol suite. Note
> also: I could not verify from the sources whether `integrated.parquet`'s `f_gas_500` column
> (used directly by D1) itself carries the same LOS/background-subtraction treatment as
> `halo_atlas.py`'s corrected pipeline — `ksz_fgas_confront.py` simply reads the pre-computed
> column with no visible bg-subtraction step. The ~0.11–0.15 (D1, Sobol, logM500~13.5) vs
> 0.076 (bg-subtracted OAT fiducial, z<0.2) figures are *not* a clean apples-to-apples (different
> mass bin/z window/sample), so do not present them as contradicting each other, but flag the
> open question to the verifier.

### (2)/(3) τ/y CAP profiles vs GNFW + CAP-aperture reconciliation of the kSZ amplitude

**D2 shape result (256 nodes, clean band x=R/r200c ∈[0.3,1.5]):** BIND's τ shape is
consistent with the *massive* BGS bins (M*>11, β≈4.8–5.2) and ELG (β≈4.5), shallower than the
steep BGS_all fit (β=7.3, Hadzhiyska Table II). In the (α,β) plane, β overlaps (BIND 4–6 vs
data 4.5–7.3) but α is offset (BIND 1–3 vs data ~0.2) — flagged as **not a valid comparison
point**: α≈0.2 is a near-singular GNFW corner degenerate with the fixed core (x_c=0.7); "the
valid comparison is the projected-profile OVERLAY + β, not the (α,β) point." — WORKLOG
2026-06-24 "P4 D2-D4"; `ksz_tau_gnfw.py`. Figures: `tau_shape_vs_hadzhiyska.png`,
`alpha_beta_plane.png` (named in WORKLOG; not independently located as PDFs in
`figures_ksz/` — likely superseded by the later `f3_response_fan`/`f6_ksz_confront` figures
that carry the same physics in the final notebook — see Gaps).

**D3 y/T result:** at fixed logM200~13.6, ELG-z (1.16) pressure y exceeds BGS-z (0.18) at
fixed mass (higher ρ_cr(z)); T_e proxy ≈ 1–2×10^7 K (~1–2 keV), declining with radius.
Figure `y_and_temperature.png` (named, not independently located — see Gaps).

**D4/CAP-aperture reconciliation — the FINAL, corrected amplitude statement (use this one):**
The exact DESI×ACT observable is the **CAP-ratio** f̃_gas = CAP_gas(θ_d)/CAP_mat(θ_d)/(Ω_b/Ω_m)
(Ried Guachalla+25 §III.2.1 — see Citations note on author/year), computed via
`_reduce_fgas_cap.py`. **Fiducial CAP f̃_gas(r200) = 0.77 (M*>11.0 cut) / 0.84 (M*>11.25 cut)
vs data 0.44 / 0.60 → ratios 1.76× / 1.40× (~2σ tension); ~20% of the 256 Sobol nodes are
consistent with the data within its (large) errors (49 nodes at the M*>11.0 cut / 65 nodes at
the M*>11.25 cut, of 256); the single best-fit node has χ²≈0.1.** Verdict: **fiducial is too
gas-rich (~1.4–1.8×) but the confrontation is feedback/data-limited, NOT "0/256 nodes" or
"beyond the manifold."** — Provenance: WORKLOG 2026-06-25 "kSZ paper: CAP-aperture
reconciliation (headline correction)"; confirmed again in WORKLOG 2026-06-26 "figure-revision
pass" (§6, f6_ksz_confront). Figure: `f6_ksz_confront` (notebook cell014_out1.png) — viewed
directly; shows BIND fiducial CAP f̃_gas curves (green M*>11.0, orange M*>11.25) rising from
~0.15–0.2 at θ=1′ to ~0.9–0.95 (≈cosmic) at θ~9–10′, vs data points with error bars that sit
systematically below (~0.3–0.5 at mid-θ), SB35 16–84% feedback band shown, "best-fit node
χ²≈0" label, "CAP-ratio (paper's filter)·σ_v-free·M*-matched" annotation confirming the exact
observable used.

> **Retracted precursor claims (do not use):** an earlier pass using a *cumulative* (not
> compensated) aperture gave "0/256 nodes reach the data," a ~2.5–3.1× tension, and (before an
> angular-diameter-distance bug fix) an even larger ~3× tension with data f̃_gas(r200)~0.28 —
> all superseded by the exact-CAP result above. Also retracted: a raw CAP-τ (not the f̃_gas
> ratio) comparison giving "BIND ~2–6× above the data, ~1× at 7.5′" (WORKLOG 2026-06-24
> D4/CAP-resolved) — later shown (2026-06-25) to be **normalization-dominated** (σ_v,
> velocity-reconstruction fidelity r/r_fid, T_CMB, M_halo all fold into a naive τ=T^CAP/(T_CMB
> σ_v/c) conversion) and explicitly dropped from the paper notebook: *"a naive
> τ=T^CAP/(T_CMB σ_v/c) conversion is INVALID ... Fig 6 CAP panel DROPPED."* The σ_v-free
> f̃_gas ratio above is the only CAP-amplitude number that should appear in the paper.

**σ_v-free 3D capstone** (clean, no projection): BIND clean 3D f̃_gas(<r500)=0.82,
**f̃_gas(<r200)=0.88** (per-halo catalog, no projection) vs DESI×ACT BGS data ≈0.32 at the same
aperture → **BIND ~2.7× too gas-rich within r200** (this is the *3D*, not CAP-ratio, number —
both should appear, labelled distinctly). Data reach BIND's level only at θ≈7–8′ (~3–4 r200).
— WORKLOG 2026-06-24 "D4 capstone"; `ksz_fgas_profile.py`.

**ELG regime (mass AND redshift extension, patch reuse):** at z=1.16 (ELG, own redshift,
snap 46), matched aperture (~2.8 r200, data reach limit), BIND fiducial f̃_gas(2.8 r200)=0.88
(truth 0.77) vs ELG data ~0.46 → **~2×** tension — "the tension holds across mass
(clusters→ELG) AND redshift (z=0.2→1.2)." — WORKLOG 2026-06-25 "P4 notebook refocus" (d).
Figure: `f6b_lowmass` (notebook cell018_out2.png) — viewed directly; two-redshift panel (BGS
z=0.18 left, ELG z=1.16 right), 256 SB35 lines + fiducial + DESI×ACT points with error bars,
patch-reuse region shaded for M200<1e13.

**Mass anchor (κ, de-degenerates the gas-vs-mass question):** BIND CMB-lensing κ (from
total-mass patches, ACT DR6 L<3000 smoothing, bg-subtracted, M*-matched) is **~1.3× the DESI
BGS×ACT κ(θ) data at the 1-halo scale (θ~2.25′)** — vs ~2× in the (cumulative, now-superseded)
f̃_gas metric — so **BIND's halo masses are right to ~30%; the f̃_gas tension is dominantly GAS,
not mass.** — WORKLOG 2026-06-25 §(f); `_reduce_kappa_bgs.py`. Figure `f6c_kappa_mass`
(notebook cell020_out1.png) — viewed directly.

### (4) First feedback posterior from (τ,y) CAP stacks; which parameters move

GP(30 params → log τ^CAP(θ)/log y^CAP) emulator quality: **CV R²≈0.83** at every aperture for
the CAP amplitude (much better than shape-only R²≈0.4–0.7) — "the amplitude responds smoothly
to feedback ... pipeline success (emulator works)." — WORKLOG 2026-06-24 "D4-D5."

Constraint (single BGS bin, data-limited): top parameter UVBHepDeltaz at 0.86× prior width; no
parameter strongly pinned. Joint BGS M*>11.0+M*>11.25: modestly better (best 0.86→0.77×
prior, #params<0.85×prior 0→5, "wind/SN sector" — VariableWindVelFactor,
WindEnergyIn1e51erg, WindFreeTravelDensFac). — WORKLOG 2026-06-24 "D4-D5"; 2026-06-25
"CAP-aperture reconciliation" §6e.

**Expanded to all DESI×ACT M* cuts (§7, `ksz_posterior_allcuts.py`):** 2→5 BGS cuts *does*
widen the constrained set (0→6 params <0.85× prior, min ratio 0.87→0.75 — pulled into the
wind/SN sector) but **+3 ELG cuts (8 points total) barely moves it further** → "latent-limited"
(consistent with the §4 finding that feedback response is intrinsically ~2-D). — WORKLOG
2026-06-26 "figure-revision pass" §7. Figure `f7_posterior` (notebook cell026_out0.png) —
viewed directly; shows posterior sd/prior sd for ~30 params across 3 nested configs (2 BGS /
all 5 BGS / 5 BGS+3 ELG), IMFslope, UVBH0beta, RadioFeedbackFactor most constrained at the
widest cut (~0.75–0.85× prior), most others prior-dominated (~0.9–1.05×).

**"Which feedback fits" (49 kSZ-consistent nodes, exact-CAP definition):** prior-normalized
parameters of the CAP-consistent nodes vs the full prior + fiducial show **only the wind/SN +
IMF sector is pulled** (a ~6-knob combination: VariableWindSpecMomentum, WindEnergyIn1e51erg,
IMFslope, VariableWindVelFactor, WindFreeTravelDensFac, WindEnergyReductionFactor,
RadioFeedbackFactor, BlackHoleFeedbackFactor, SNII_MinMass_Msun, RadioFeedbackReorientationFactor,
MaxSfrTimescale, QuasarThreshold shown) — "individual signs are non-naive" (an earlier draft
mistakenly said "higher wind energy" uniformly; corrected). — WORKLOG 2026-06-26
"figure-revision pass" (f6a). Figure `f6a_params_fit` (notebook cell016_out0.png) — viewed
directly; 12-parameter strip plot, all-256 (grey) vs 49-kSZ-consistent (red) vs fiducial
(star) vs consistent-median (bar).

**MONEY PLOT forecast (§8, future SO/CMB-S4×DESI):** kSZ alone or tSZ alone each leave one
direction prior-wide (var~0.9); **joint kSZ+tSZ pins BOTH directions: dir1 ν≈0.10, dir2
ν≈0.16** (ν = posterior/prior variance) — "complementary, not redundant." dir1 loads on
+VariableW −BlackHole +WindEnerg; dir2 on +SNIa_Rate +WindFreeT +RadioFeed. Best-constrained
forecast direction lies **0.81 inside the §4 latent plane** (vs 0.26 for a random direction) —
i.e. the forecast pins the *same* 2-D manifold the current-data response lives in. — WORKLOG
2026-06-24 "D6 MULTI-PROBE"; 2026-06-25 "P4 notebook refocus" §8. Figure `f8_money_forecast`
(notebook cell028_out0.png) — viewed directly; corner plot, kSZ-only (blue)/tSZ-only
(green)/joint (red) posteriors on the 2 constrained directions, joint clearly tightest.

### (5) Thermo decomposition — what drives τ vs y differences

> **⚠ BOXED CORRECTION — the original "y = f̃_gas·κ·T_e" 3-probe closure and its "too much
> gas, not too hot" headline are RETRACTED.** The original decomposition (WORKLOG
> 2026-06-25 §6d, `ksz_thermo_decomp.py`) reported: density (kSZ, then-current CAP number)
> 2.0× × mass (κ) 1.3× × pressure (tSZ) 2.6× ⇒ implied T_e≈1.0× ("too much GAS, not too HOT").
> This 2.6× tSZ number was then found (WORKLOG 2026-06-26 "tSZ mass-selection bug") to be
> **inflated by the Msun-vs-Msun/h host-mass bug** (§ Methods above). After the fix, WORKLOG
> states explicitly: *"the earlier 'tSZ pulls beyond / inter-probe tension' was the logM bug
   ... §6e reframed ... The 'too much gas not too hot' / 'mildly too hot' framings below are
   RETRACTED"* (memory `ksz-p4-tau-not-velocity.md`, verbatim). The **cross-sample
   density×mass×temperature decomposition itself was dropped from the paper notebook** ("new
   right panel shows the Y∝M^5/3 mass-sensitivity" instead) — there is **no currently valid
   T_e/heating-vs-ejection decomposition number to report**.

**What remains valid and citable for this section:**
- κ (mass, robust to the Y∝M^5/3 bug because it uses total-mass patches, not Y): BIND masses
  right to **~30%** (1.3× at the 1-halo scale) — confirms the gas tension is density, not a
  mass-selection artifact. (Result 3 above.)
- f̃_gas (density, kSZ leg, exact CAP-ratio): fiducial **1.76×/1.40×** the data (M*>11.0/11.25
  cuts) — Result (2)/(3) above.
- y-CAP (pressure, tSZ leg, mass-corrected at logM200=13.18 Msun/h, Sailer+24): BIND **≈1.51×**
  the data (fiducial), with the **±0.1-dex mass systematic alone spanning 1.35–2.05×** —
  "the tSZ amplitude is mass-selection-dominated, NOT a clean feedback probe." At the same
  corrected mass, kSZ and tSZ now *agree* at the strong-feedback edge (MAP ê₁≈−7,
  implied f̃_gas≈0.53–0.54) rather than showing an "inter-probe tension" (the earlier framing,
  retracted). — WORKLOG 2026-06-26 "tSZ mass-selection bug." Figure `f6d_tsz_pressure`
  (notebook cell022_out0.png) — viewed directly; panel (a) shows the ±0.1-dex mass-syst band
  dominating the visual 1.5–2.1× spread and a Spearman ρ=0.80 rank correlation between
  χ²(kSZ) and χ²(tSZ) across nodes (same nodes fit both legs); panel (b) shows BIND's y-CAP
  shape normalized at 2.25′ is **more extended** than the data (peaks 4.75′ vs data 3.5′) —
  a flat-pedestal test confirmed this extension is real, not a background-subtraction
  artifact.
- **2-D latent decomposition (§4, the paper's actual "what drives τ vs y" answer — NOT a
  τ-vs-y split but an inner-vs-outer-gas split):** the 30-d (τ,y) response at the BGS mass bin
  collapses to **2 latent components explaining 97% of the variance** (λ=0.51/0.46, sharp
  knee after 2). Target-rotated to physical axes: ê₁ = **inner gas** f_gas(<R500) (r=+0.95
  with the latent — the wind/SN ejection axis), ê₂ = **outer gas** f_gas(R500→R200) (r=+0.95).
  The two zones are only **r=0.24 correlated** (genuinely independent) and jointly
  reconstruct the plane (R²=0.90/0.96). Total halo mass is common-mode across nodes
  (CV=0.0008) so "gas variation = f_gas variation." — WORKLOG 2026-06-26 "figure-revision
  pass" (f4). ⚠ An intermediate labelling attempt (a gas-blind τ-shape-residual proxy for ê₂)
  was tested and found to be an **overclaim** — "in TNG gas content & profile concentration
  are ~87% degenerate" and gave only partial r=0.70 — before the clean inner/outer-f_gas
  labelling was found; keep only the final, physical labelling. Figure `f4_latent` (notebook
  cell010_out0.png) — viewed directly; 4-panel: (a) scree plot 2 components→97%, (b) loading
  bars naming ~9 parameters per axis (VariableWindVelFac, IMFslope, WindFreeTravelDens,
  WindEnergyIn1e51er dominate both axes with opposite/shared sign patterns), (c)/(d) same
  latent plane colored by inner/outer f_gas respectively, both r=0.95.

### (6) Field-level companion (paper_ksz_field.ipynb — ray-traced multiprobe view)

Built because (user, quoted in WORKLOG) "did we ever use the ray-traced data?" — answer: no,
the per-halo CAP paper (§1–§5 above) uses painted patches only, never the lux ray-traced
maps; this is a deliberate sibling analysis using the assembled `bind_sb35/emulator_dataset.npz`
(**253 valid Sobol nodes**: tomographic C_ℓ^κκ, **C_ℓ^κy**, C_ℓ^yy, C_ℓ^ττ, ready-made
suppression S(ℓ,z_s) vs the paired DMO baseline, peaks/PDF/MF/WST, scaling relations) — no
new reductions. — WORKLOG 2026-06-26 "field-level companion."

**Key contrast with the per-halo result (honest, stated as a contrast in WORKLOG):** the
*field-level* feedback response is **~1-D** (λ₁≈0.95, robust across every probe combination) —
a single gas-ejection amplitude — vs. the per-halo CGM's clean **2-D** (inner+outer gas).
Interpretation given: "the LOS projection + lensing kernel wash out the radial 2nd
dimension." Figure `cell010_out0.png` (§4 field companion) — viewed directly; scree plot
shows component 1 at λ₁=0.95 already above the 95%-variance line, component 2 flat.

**§5 real-data confrontation — BIND shear×y vs real DES Y3 × ACT** (`desact_data.npz`, ξ_γy
via n(z)-weighting + physical ACT DR6 beam (2.4′, NOT an earlier erroneous 10′ "effective
beam") + J₂ Hankel transform, robust window 8′–40′): **BIND is ≈2.5× (DES bin 3, z̄≈0.74) /
2.3× (DES bin 4, z̄≈0.94) above the data** at well-measured scales (read directly off the
figure's text annotations) — "TNG too gas-bound." A user catch found and fixed a bug where an
over-smoothing 10′ beam had shifted BIND's peak to 11′ vs the data's 4.5′; with the correct
2.4′ beam, BIND is shown to be **both too high AND too steep** (over-bound gas), with the 5°
field's low-ℓ cut affecting large θ. — WORKLOG 2026-06-26 "field-level companion." Figure
`cell012_out0.png` — viewed directly (2-panel, DES bins 3 & 4, BIND Sobol 16–84% band + 2.4′
beam solid + no-beam dotted vs DES Y3×ACT black points). Note: the WORKLOG *prose* summary
rounds this to "~1.5–2× high" in one place — **use the more precise on-figure values (2.3–2.5×)
as primary**, and note the prose is an approximate paraphrase, not a separate number.

**§6 the κ×y cross is the feedback driver** (not the κ auto): band-integrated feedback
response D ranks yτ (0.18) > ττ (0.16) > yy (0.135) > κy (0.10) ≫ κκ (0.03) — κ-auto is
"mostly dark matter," feedback lives in the pressure/density cross-correlations. Node-spread
fraction vs fiducial rises steeply at high ℓ for τ,y-bearing spectra but stays flat (<0.2) for
κκ across the whole ℓ range shown. — Figure `cell014_out0.png` — viewed directly.

**§7/§8 forecast:** field-level Fisher gives **~1 dominant constrained direction** (per-param
corner shows only 3 of ~27 params below prior, one clearly dominant at ~0.6× prior — see
figure); a future LSST×CMB-S4 κ×y forecast (×5 Fisher boost vs DES×ACT) pins that single
gas-ejection latent to dir1 ν≈0.05 (dir2, orthogonal, stays wide at ν≈0.11 — "dir1 ≪ dir2").
— Figures `cell016_out0.png`, `cell018_out0.png` — viewed directly.

---

## FIGURES

All paths below are the freshly-extracted PNGs (identical content to the on-disk
`examples/figures_ksz/*.pdf`, verified 1:1 by section heading match — 13 files on disk = 13
notebook output cells in `paper_ksz_desi_act.ipynb`). Extracted to
`papers/04_ksz_gas/figs_raw/{paper_ksz_desi_act,paper_ksz_field}/cellNNN_outM.png` +
`manifest.json` in each dir (cell index / heading / source snippet provenance).

Primary notebook (`paper_ksz_desi_act.ipynb`, 8 sections / 13 output cells):

| file | on-disk PDF equiv. | heading | what it shows | serves section |
|---|---|---|---|---|
| `paper_ksz_desi_act/cell004_out0.png` | `f1_lightcone.pdf` | §1 DATA | 3-panel: snapshot↔z coverage (20 slices), Sobol design in physical log units (ASN1 wind-energy × ASN2 wind-speed, fiducial★), halo mass function w/ BGS/LRG/ELG host bands vs the 1e13 BIND floor | Intro/Methods |
| `paper_ksz_desi_act/cell006_out0.png` | `f2_pipeline.pdf` | §2 METHODS | fiducial τ(R/r200c) stack by mass bin + clean band; M*-matching 2D histogram (SHMR) with the two DESI cuts marked; CAP filter schematic (disk+ring) | Methods |
| `paper_ksz_desi_act/cell008_out0.png` | `f3_response_fan.pdf` | §3 response fan | τ(R) and y(R) for all 256 Sobol nodes colored by f̃_gas,500, fiducial in black, node-spread/fiducial bottom row (core-dominated response) | Results §3 |
| `paper_ksz_desi_act/cell010_out0.png` | `f4_latent.pdf` | §4 latent | scree plot (2 comps=97% var), loading bars (~9 params/axis), latent plane colored by inner-gas / outer-gas f_gas (both r=0.95) | Results §5 (thermo/latent decomposition) |
| `paper_ksz_desi_act/cell012_out0.png` | `f5_fgas_erosita.pdf` | §5 Confrontation I | f_gas,500(M500c): BIND Sobol 16-84% band + median + strongest-fb vs Eckert+19 and Popesso+24-eROSITA; Ω_b/Ω_m line | **Results (1) — headline 0% figure** |
| `paper_ksz_desi_act/cell014_out1.png` | `f6_ksz_confront.pdf` | §6 Confrontation II | CAP-ratio f̃_gas(θ) fiducial curves (M*>11.0/11.25) + SB35 band + best-fit node vs real DESI×ACT data points | **Results (2)/(3) — corrected CAP amplitude figure** |
| `paper_ksz_desi_act/cell016_out0.png` | `f6a_params_fit.pdf` | §6a which feedback fits | strip plot of ~12 params: all-256 vs 49-kSZ-consistent vs fiducial vs consistent-median | Results (4) |
| `paper_ksz_desi_act/cell018_out2.png` | `f6b_lowmass.pdf` | §6b ELG at its z | two-redshift (BGS z=0.18 / ELG z=1.16) f̃_gas(M200) with patch-reuse region, all M* cuts | Results (2)/(3) — mass+z extension |
| `paper_ksz_desi_act/cell020_out1.png` | `f6c_kappa_mass.pdf` | §6c mass anchor | CMB-lensing κ(θ) BIND vs DESI BGS×ACT, 1-halo region shaded | Results (2)/(3) — mass-not-gas |
| `paper_ksz_desi_act/cell022_out0.png` | `f6d_tsz_pressure.pdf` | §6d pressure leg | (a) y-CAP amplitude w/ ±0.1-dex mass-syst band; (b) shape normalized at 2.25′ (BIND too extended) | Results (5) |
| `paper_ksz_desi_act/cell024_out0.png` | `f6e_latent_data.pdf` | §6e data in latent plane | real DESI×ACT data projected into the ê₁/ê₂ manifold (kSZ-only/tSZ-only/joint contours) + 1D posterior on ê₁ | Results (4)/(5) |
| `paper_ksz_desi_act/cell026_out0.png` | `f7_posterior.pdf` | §7 all-cuts posterior | posterior-sd/prior-sd for ~30 params, 3 nested M*-cut configs | Results (4) |
| `paper_ksz_desi_act/cell028_out0.png` | `f8_money_forecast.pdf` | §8 MONEY PLOT | corner plot of 2 constrained directions, kSZ-only/tSZ-only/joint forecast posteriors | Results (4) — forecast |

Field-level companion (`paper_ksz_field.ipynb`, 8 output cells, NOT on disk as separate PDFs —
extracted fresh):

| file | heading | what it shows | serves section |
|---|---|---|---|
| `paper_ksz_field/cell004_out0.png` | §1 DATA (ray-traced) | fiducial ray-traced y map + related panels | Results (6) intro |
| `paper_ksz_field/cell006_out0.png` | §2 METHODS | field statistic + DMO-paired ratio definitions | Results (6) methods |
| `paper_ksz_field/cell008_out0.png` | §3 field response fan | S(ℓ) and C_ℓ^κy across 256 nodes | Results (6) |
| `paper_ksz_field/cell010_out0.png` | §4 ~1 latent | scree plot showing λ₁≈0.95 (field is ~1-D, contrast w/ per-halo 2-D) | **Results (6) — key contrast figure** |
| `paper_ksz_field/cell012_out0.png` | §5 real DES Y3×ACT | ξ_γy(θ) BIND vs real data, 2 DES bins, beam/no-beam/scale-cut shown, "BIND/data≈2.5×/2.3×" | **Results (6) — real-data confrontation** |
| `paper_ksz_field/cell014_out0.png` | §6 κ×y is the driver | (a) band-integrated response ranking yτ>ττ>yy>κy≫κκ; (b) node-spread/fiducial vs ℓ | Results (6) |
| `paper_ksz_field/cell016_out0.png` | §7 field-constrained directions | per-param posterior (prior-dominated) + 1-dominant-direction panel | Results (6) forecast |
| `paper_ksz_field/cell018_out0.png` | §8 MONEY PLOT (field) | forecast ellipse: dir1≪dir2 (LSST×CMB-S4 κ×y) | Results (6) forecast |

**Recommendation (8–10 figures per brief):** primary candidates for the final draft, following
the notebook's own arc: f1 (data/design), f2 (methods/CAP schematic), f3 (response fan), f4
(2-D latent), f5 (eROSITA 0%), f6 (CAP-corrected kSZ amplitude), f6d (tSZ pressure leg, w/
mass-selection caveat baked into the panel), f7 or f8 (posterior/forecast) = 8 from the
primary set, plus 1–2 from the field companion (cell010 "~1-D contrast" + cell012 "real DES×ACT"
or cell014 "κ×y driver") for the mandatory §6 field-level section = 9–10 total.

---

## CAVEATS

**Brief's mandatory list (all sourced):**
1. **Velocity-free τ** — BIND's "kSZ" is the electron-column optical depth, not a
   velocity-weighted temperature; ΔT_kSZ is unbuilt and would need a DMO-velocity surrogate
   (Phase 3, kill-gate G3, not attempted). — `ksz_desi_act_plan.md` §0, §3 Phase 3.
2. **TNG model space only** — the whole exercise (256-node Sobol) samples TNG-*like* feedback
   parametrizations; it cannot speak to non-TNG subgrid models. Stated as the paper's thesis
   framing (brief; consistent with the "model-space vs parameter-space tension" discussion
   point).
3. **Mass floor 1e13 for painted halos** — the TNG300 lightcone halo floor is M200≥1e13; the
   ELG-host regime (~10^12.5, DESI ELG mean logM200≈12.2 h⁻¹ per user-supplied number) sits
   below it. Low-mass (1e12–1e13) reuse via patch cross-matching recovers a **partial** sample
   (28.7% within the 4×R200 production circular paste, 51.6% at the full 6.25 Mpc/h patch
   footprint) — "the remaining ~48% are field halos too far from any cluster," i.e. ELG-like
   field galaxies are systematically under-captured by the reuse trick (captured halos have
   median host distance 1.87–1.9 Mpc/h, so they are near-cluster, not field). Env-bias check
   (full TNG300 hydro catalog): near-host vs field f̃_gas 0.408 vs 0.409 (<0.3% difference) —
   the captured reuse sample is representative *of this particular statistic* despite the
   spatial bias. — WORKLOG 2026-06-25 "low-mass baryons via reuse-only"; `_reduce_fgas_lowmass.py`.
4. **Satellite/miscentering effects in CAP stacks** — explicitly flagged for the tSZ y-CAP
   shape (not amplitude): "the photometric-LRG stack is smoothed by miscentering + photo-z LOS
   spread + the ACT beam (which the authors model and marginalise)" — this is *why* the paper
   compares only the 1-halo y-CAP *amplitude*, never the profile shape, to the tSZ data.
   — memory `ksz-p4-tau-not-velocity.md`. Analogous 2-halo caveat for κ: BIND's per-halo patch
   is isolated (no 2-halo term), so κ falls below the data at θ≳3′ — compare only the 1-halo
   scale.
5. **y-map normalization already physical** — Compton-y is already physical (no extra 1/a²
   weighting needed); previously validated against Planck (per project memory
   `tsz-ymap-normalization.md`, not re-derived in this brief's sources but consistent with all
   ksz-desi-act y-usage, which applies no extra normalization factor).

**Additional caveats found in the sources (not in the brief's mandatory list):**
6. **"0% coverage" / f_gas absolute-number provenance gap** — `ksz_fgas_confront.py` reads
   `f_gas_500` directly from `integrated.parquet` with no visible background-subtraction step
   in that file; I could not confirm from the ksz-desi-act sources whether the parquet's
   `f_gas_500` column itself already carries the LOS-contamination fix later applied in
   `halo_atlas.py` (2026-06-15). Flag for the verifier; do not assert the two pipelines'
   absolute f_gas values are on the same footing.
7. **GNFW (α,β) point comparison is invalid** — α≈0.2 is a near-singular corner of the fixed
   γ=-0.5,x_c=0.7 GNFW form, degenerate with the fixed core; only the profile-overlay + outer
   slope β is a meaningful comparison (D2, `ksz_tau_gnfw.py`).
8. **kSZ-inferred f_gas from the GNFW table is unphysical as transcribed** — forward-integrating
   the Hadzhiyska GNFW density (∝ρ0) to get f_gas gives BGS_all→0 (the parametrization's ρ0
   units/normalization don't support this integral); don't reproduce this calculation in the
   paper — use the paper's own reported f_gas table or the CAP-ratio comparison instead.
   — memory `ksz-p4-tau-not-velocity.md`.
9. **Thermo (κ,τ,y) decomposition retracted** — see boxed note in Results (5); do not report a
   T_e/heating number.
10. **CAP posterior covariance is diagonal/approximate** — the joint M* multi-bin posterior
    "diagonal cov treats the nested M* bins as independent → optimistic"; the multiprobe
    kSZ+tSZ forecast covariance is a "realistic AR(1) aperture covariance," an improvement but
    still not the survey's actual published bin-bin covariance (a listed TODO in
    `ksz_desi_act_plan.md` D6).
11. **BIND f_gas = total gas; eROSITA = hot X-ray gas; kSZ = total electrons** — compare like
    with like; the X-ray band is a lower envelope on the true gas content, not a kSZ target.
    (Mentioned as a standing caveat throughout D1/D2.)
12. **Central-M* proxy not independently validated** — "Central-M* proxy still wants
    validation vs TNG Subfind at fiducial (DMO↔hydro match)" — listed as an open TODO, not yet
    done. — WORKLOG 2026-06-24 "D4 RESOLVED."
13. **σ_v value used throughout (300 km/s) and an earlier digitization slip** — the paper
    normalization used σ_v^true≈300 km/s (T_CMB σ_v/c=2727 µK); an earlier draft used 1176 µK
    from a digitization slip — worth a methods footnote if any raw-τ CAP number were to be
    shown (it is not, per the "Fig 6 CAP panel DROPPED" retraction above), otherwise moot.

---

## CITATIONS

**Seed citations from the brief — status after checking the actual sources:**

| brief's seed | found in sources? | what the sources actually say |
|---|---|---|
| Schaan+2021 (ACT kSZ stacking) | ✅ | notebook closing cell: "earlier stacks Schaan et al. (2021), Amodeo et al. (2021)" |
| Amodeo+2021 | ✅ | as above |
| Hadzhiyska+2026 | ⚠ AMBIGUOUS — see note below | |
| Bigwood+2024 | ⚠ year mismatch | notebook closing cell says "Bigwood et al. (2025)"; `ksz_desi_act_plan.md` §5 groups it as "Siegel/Bigwood consensus 2509.10455" (same ID as Siegel, ambiguous); no independent arXiv ID for Bigwood found anywhere in the sources |
| eROSITA group f_gas (Popesso/Bahar) | ✅ Popesso, ❌ Bahar | `ksz_fgas_confront.py`: `popesso_fgas(M)="Popesso+24 eROSITA hot-gas fraction, f_gas,500 = LOW/strong-feedback edge"`, formula f_gas=2.23e-7·M^0.39; notebook closing cell "Popesso et al. (2024)"; **Bahar is never mentioned anywhere in the sources** — brief's guess is unconfirmed, do not cite Bahar without independent verification |
| DESI (DESI Collab 2024) | ✅ | notebook closing cell: "DESI Collaboration (2024), Hahn et al. (2023, BGS), Zhou et al. (2023, LRG), Raichoor et al. (2023, ELG)" |
| ACT DR6 | ✅ (implicit, used throughout: beam=1.6′/2.4′, L<3000 lensing cut) | no separate ACT DR6 instrument-paper citation found; likely covered by the Hadzhiyska/Ried-Guachalla + Sailer citations |
| Battaglia+2012 GNFW | ❌ NOT FOUND | the GNFW form actually used is explicitly the *Hadzhiyska+26/Ried-Guachalla+25* paper's own fit (Eq. 26-27 per `ksz_tau_gnfw.py`), not Battaglia+2012's; no mention of Battaglia anywhere in the ksz-desi-act sources — do not cite Battaglia for this GNFW without checking the actual paper |
| CAP filter (Ferraro/Schaan) | ❌ NOT FOUND | the CAP filter (disk-minus-equal-area-ring) is implemented and attributed only to Ried Guachalla+25 §III.2.1 in the code comments; no Ferraro or dedicated CAP-filter-origin citation found — verify externally (Schaan et al. 2021 is cited for "earlier kSZ stacks" generally, may be the right anchor, but the sources don't say so explicitly) |
| van Daalen+2020 | ✅ | field companion notebook: "van Daalen et al. (2020) ($f_{gas}$–suppression)"; used for the S(ℓ)=C_ℓ^κ,hydro/C_ℓ^κ,DMO baryonic-suppression definition |
| Siegel+ | ✅ | notebook closing cell: "Siegel et al. (2025, arXiv:2509.10455)"; consistently referenced as "Siegel/Bigwood" or "Siegel/Bigwood consensus" throughout WORKLOG/plan doc |

**⚠ Hadzhiyska+26 / Ried Guachalla+25 — the SAME two arXiv IDs are attributed to TWO
different author names across the sources; this needs resolution before citing.**
- `arXiv:2604.19744` = "Part I" (LRG paper; TABLE III kSZ amplitude rescaling 0.367, "mass
  degenerate" warning) — attributed to "Ried Guachalla et al. 2025" in `docs/method.md`-style
  prose (WORKLOG 2026-06-25 §6c: "Part I (LRG; Ried Guachalla et al. 2025)").
- `arXiv:2604.19745` = "Part II" (BGS/ELG; GNFW Table II, §III.2.1 CAP-ratio definition,
  digitized CAP points) — attributed to **both** "Hadzhiyska+26" (`ksz_tau_gnfw.py` docstring:
  "Hadzhiyska+26 (DESI DR2 x ACT DR6, arXiv:2604.19745)"; `ksz_cap_compare.py`,
  `ksz_fgas_profile.py`, `ksz_posterior*.py`) **and** "Ried Guachalla+25" (`_reduce_fgas_cap.py`
  docstring: "Ried Guachalla+25 (DESI x ACT, arXiv:2604.19745)"; the notebook's own closing
  citation list: "DESI×ACT kSZ: Hadzhiyska et al. (2024), Ried Guachalla et al. (2025)" —
  listing BOTH as if separate references sharing one description).
- The arXiv ID `2604.19745` itself implies a **2026** submission (arXiv YYMM prefix 2604 =
  April 2026), which is inconsistent with both "Hadzhiyska et al. (2024)" and "Ried Guachalla
  et al. (2025)" appearing in the same closing-citation sentence.
- **My read:** this is very likely *one* paper (a two-part DESI×ACT kSZ series) whose actual
  lead author the agent sessions guessed inconsistently before/after finding the real preprint
  — "Hadzhiyska" appears to be an earlier guess (used in the earlier D2-D4/D4-D5/posterior
  code, written 2026-06-24) and "Ried Guachalla" the name found once the actual paper text was
  read closely enough to quote §III.2.1 verbatim (2026-06-25 onward, `_reduce_fgas_cap.py`,
  WORKLOG). **The CITE step MUST verify the actual author/year for arXiv:2604.19744 and
  2604.19745 before the paper goes out** — do not print both names as if citing two different
  papers, and do not trust either year (2024/2025/2026) without checking.

**Other confirmed citations (found directly in sources, not brief seeds):**
- CAMELS: Villaescusa-Navarro et al. (2021).
- IllustrisTNG: Weinberger et al. (2017), Pillepich et al. (2018), Nelson et al. (2019, TNG300
  public data release).
- Eckert et al. (2019) — mainstream X-ray f_gas,500 (weak-feedback edge), formula
  f_gas=0.131·(M/2×10^14)^0.21 (`ksz_fgas_confront.py`).
- Kovač et al. (2025, arXiv:2507.07991) — grouped with the Siegel/Bigwood missing-baryon
  consensus.
- Lin et al. (2025, arXiv:2509.01881) — low-dimensional feedback latents (motivates/parallels
  the §4 2-D latent finding here; this is also the anchor reference for sibling Paper III).
- Constantine (2015) — active-subspaces method (cited alongside the latent/PCA machinery).
- Sailer et al. (2024) — DESI-LRG host mass from ACT CMB-lensing, logM200=13.18 Msun/h (used
  to fix the mass-convention bug).
- tSZ y-CAP data: arXiv:2502.08850 (Zenodo 14706729) — ACT×DESI photometric-LRG Compton-y CAP.
  **Author name not found in the ksz-desi-act sources** — code only cites the arXiv ID, never
  a name; the CITE step must resolve the actual authorship.
- DES Y3 × ACT shear×y (field-level §5, real-data confrontation): **Gatti et al. (2022),
  Pandey et al. (2022)** (field companion notebook closing cell). Data file itself sourced
  from GitHub `shivampcosmo/ACTxDESY3` → `DES_ACT.fits` (a data repository, not itself a
  citable reference — the citable papers are Gatti/Pandey above).
- Flow-matching / OT method (BIND's own engine, likely already cited in Paper I but relevant
  if Methods restates the emulator): Lipman et al. (2023, arXiv:2210.02747), Tong et al.
  (2023, arXiv:2302.00482) — `docs/method.md`.

**Candidate arXiv IDs found only in project memory (NOT independently confirmed against the
ksz-desi-act code/notebooks — treat as leads, not verified citations):** a general "lit
deep-dive" list in memory `baryon-atlas-rung0.md` (dated 2026-06-15, attached to the Rung-0/1/2
per-halo atlas work, not specifically the P4/kSZ notebooks) lists: "eROSITA f_gas 2411.16555,
consensus 2509.10455, strong feedback 2512.02954, kSZ 2407.07152, gas/DM shapes 1003.2270,
FRB-DM-mass tension 2507.16816." Of these, 2509.10455 is independently confirmed above as
Siegel+2025; the others (2411.16555 possibly = Popesso+24; 2512.02954 possibly = Bigwood's
missing arXiv ID; 2407.07152 possibly an earlier kSZ-stacking paper) are **unverified guesses
at best** — the CITE step should search these IDs directly rather than assume the mapping.
