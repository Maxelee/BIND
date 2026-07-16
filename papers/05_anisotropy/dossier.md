# Dossier — Paper V (Letter): anisotropy of the baryonic WL suppression

Miner output for `papers/05_anisotropy/brief.md`. Organized by the brief's section
outline (Intro / Method / Result / Discussion). **Read the "HEADLINE CORRECTION" box
first** — it changes what number the Result section should lead with.

---

## ⚠ HEADLINE CORRECTION (WORKLOG wins over the brief's seed number)

The brief's thesis states "~40–65% of the effect is anisotropic at first order,"
measured with "a faithful radially-symmetrized counterfactual (a radial 'BCM warp')."
The sources show this conflates **two different controls that were run at two
different times**, and the brief's framing attributes the old number to the new
(faithful) control. The actual history:

1. **2026-06-16** (`docs/WORKLOG.md:1234`): first spherical-BCM control built. It
   symmetrizes the **correction** ΔΣ = Σ_hydro − Σ_dmo (`azimuthal_average_patch`,
   `mode="monopole"`) — i.e. it forces the *correction* to be circular.
2. **2026-06-17, first pass** (`WORKLOG:1193`, notebook `wl_anisotropy_paper.ipynb`
   Part II): using control (1), the paper draft measures
   **f_aniso(C_ℓ) ≈ 41% → 48% → 55%** at ℓ = 5000 → 10⁴ → 2×10⁴ (increasing with
   scale), and the cross/auto mechanism (cross/ΔP≈1.39, auto/ΔP≈−0.39). This is the
   source of the memory-file's "~40–65%" framing.
3. **2026-06-17, methodological fix** (`WORKLOG:1172`, "faithful radial-BCM
   counterfactual"): control (1) is identified as flawed — it credits a real
   (Schneider–Teyssier) BCM with **zero** triaxiality-tracing, because a real BCM
   displaces mass *radially* (`r → r+d(r)`) and so a triaxial halo's correction is
   *not* circular even under a faithful BCM. This makes `dX_aniso = X_bind − X_sph`
   from control (1) an **upper bound** that lumps triaxiality-tracing into "feedback
   anisotropy." New primary control **`bcm_warp_patch`** (`mode="bcm_warp"`,
   `src/bind/inference/spherical.py`): pushes each DMO pixel radially so
   `M_dmo(<r) = M_hydro(<r')` (mass-exact, CIC), so triaxiality **co-moves** with the
   mass flow; only genuine angular feedback structure now separates `sph` from
   `bind`.
4. **2026-06-17, corrected measurement** (`WORKLOG:1144`, "three-way kappa
   comparison confirms triaxiality-tracing inflated f_aniso"; script
   `examples/bcm_warp_comparison.py`; fig
   `examples/figures_lightcone/bcm_warp_comparison_fid.png`): re-run on fid,
   z_s=1, with the **faithful** `bcm_warp` control plus the old circular `mono`
   control side by side. Result:
   - Old "40–65%" number = **f_mono = (C_b−C_m)/(C_b−C_d) ≈ 0.5, ~flat in ℓ**
     (this is `bind` vs the *pure-spherical/circular* control — NOT the faithful
     radial-BCM-warp control the brief attributes it to).
   - Corrected **f_aniso = (C_b−C_s)/(C_b−C_d) [faithful radial BCM misses] ≈ 0.45
     at ℓ=3×10³ → 0.24 at ℓ=10⁴ → ~0 at ℓ=2×10⁴** (declining with scale).
   - Triaxiality-tracing **f_tri = (C_s−C_m)/(C_b−C_d) rises 0.08 → 0.25 → 0.61**
     over the same range — i.e. at small scales essentially *all* the apparent
     anisotropy is triaxiality a real BCM reproduces, not missed feedback physics.
   - Real space: r(bind,sph) ≈ r(bind,mono) ≈ 0.987–0.988; the faithful warp control
     barely beats the crude mono control globally (16% vs 16% of κ rms) — its
     advantage over `mono` is concentrated at high ℓ.
   - `docs/WORKLOG.md:1153`, verbatim: "the headline dropped as predicted."

**Net effect on the Letter's headline claim:** the number that should lead the Result
section is the **scale-dependent, corrected f_aniso (≈45% at ℓ~3000, declining to
~0 by ℓ~2×10⁴)**, not a flat "40–65%." The "40–65%" (or the notebook's own "~40–55%",
see below) should be presented explicitly as the earlier, upper-bound estimate from
the circular/monopole control, superseded by the three-way comparison. This is
exactly the brief's own caveat ("control revision pending, number expected to
drop") — the revision has, in fact, already happened and dropped the number, but
**only in the standalone script/figure, not yet propagated back into the notebook's
own field-level chapters** (cross/auto decomposition, Discussion, Conclusions all
still quote the old-control numbers — see caveat list below).

---

## Intro

- **Thesis (sourced):** BCMs (Schneider & Teyssier 2015; Aricò et al. 2020; Mead
  et al. 2021) displace halo mass **radially only**, keeping the correction's
  monopole c₀(r) and discarding its quadrupole c₂(r). BIND lets you build the
  literal spherical/radial counterfactual per halo (no 3D particles, no N-body
  re-run) and difference against the full painted field to isolate anisotropy.
  — provenance: `wl_anisotropy_paper.ipynb` cell 0 (title/abstract markdown);
  `src/bind/inference/spherical.py` module docstring.
- **Novelty claim (sourced, verbatim from notebook abstract):** "This has not been
  measured before, because it requires field-level baryon maps that match
  hydrodynamics halo-by-halo while letting us surgically remove **only** the
  angular structure." — `wl_anisotropy_paper.ipynb` cell 0.

## Method

### The three/four-field construction
- `dmo` (no baryons), `sph`/`mono` (spherical BCM control — **two variants**, see
  correction box above), `bind` (full BIND composite). All built from the **same
  halos and the same per-snapshot realization shifts** so cosmic variance and shape
  noise cancel in the difference. — provenance: `wl_anisotropy_paper.ipynb`
  Methods cell (index 2); `bind.inference.spherical` module docstring;
  `examples/bcm_warp_comparison.py` docstring.
- **Faithful control = `bcm_warp_patch`** (`mode="bcm_warp"`): radial push
  `r→r'` with `M_dmo(<r) = M_hydro(<r')`, CIC, mass-exact; triaxiality co-moves
  with the mass flow. Validated: mass conserved to 1e-5, monopole tracks hydro,
  DMO annular RMS preserved. — `WORKLOG:1172` (2026-06-17 "faithful radial-BCM
  counterfactual").
- **Superseded control = `azimuthal_average_patch`** (`mode="monopole"`):
  azimuthal average of the *correction* ΔΣ, i.e. "monopole-only correction in
  projection" — forces the correction itself circular, crediting a real BCM with
  zero triaxiality-tracing. — `WORKLOG:1172`, `1234`.
- **Validation that `sph`/`mono` is a fair monopole-only counterpart:** stacked-
  cluster c₀(r) of BIND vs spherical control lie on top: **L2 relative residual =
  2.22%, correlation = 0.99998**; residual `bind−sph` is a *pure m≥2* field —
  its monopole is consistent with zero: **RMS|c0|/RMS|c2| = 0.018** (PASS);
  residual monopole as a fraction of the full-correction monopole = **0.019** (i.e.
  the spherical control leaves c0(r) intact to ~2%). — provenance:
  `wl_anisotropy_paper.ipynb` cells 17–18 (printed output), also
  `WORKLOG:1217` ("L2 2.2%, corr 0.99998").
- Also: for the single biggest halo (logM=14.57), the anisotropy carries **97% of
  the correction's pixel rms** — a "pure quadrupole+" case. — cell 17 output.

### The multipole estimator (per-halo, "Pillars")
- Azimuthal Fourier decomposition on physical-radius annuli:
  c_m(r) = ⟨ΔΣ(r,φ) e^{−imφ}⟩_φ; Hankel transform H_m(k) = 2π∫c_m(r) J_m(kr) r dr.
  Halos aligned to their **DMO shape axis** θ_DMO (observationally realizable —
  BCG/cluster shape), *not* the halo's own c₂ axis (self-alignment biases the
  amplitude high). Debiased stacked quadrupole power P₂(k) = ⟨|H₂^Re|²⟩ −
  ⟨|H₂^Im|²⟩ (cross-power subtraction). Headline per-halo statistic: f_quad(k) =
  P₂/(P₀+P₂), a scale-free ratio. — `wl_anisotropy_paper.ipynb` Methods cell.

### The paired field-level statistic + why it is first order (the mechanism)
- For any map statistic X: ΔX_baryon = X_bind − X_dmo; ΔX_aniso = X_bind − X_sph;
  f_aniso(X) = ΔX_aniso/ΔX_baryon.
- **Exact identity** for the power spectrum:
  ΔP = P_bind − P_sph = cross + auto, with
  cross = 2 Re⟨κ_dmo* 𝓕(κ_bind−κ_sph)⟩ (**first order** in the correction) and
  auto = ⟨|𝓕δ_bind|²⟩ − ⟨|𝓕δ_sph|²⟩ (second order), δ ≡ κ−κ_dmo.
  Measured on the fid maps (8 paired realizations, z_s=1), band ℓ∈[3×10³,2×10⁴]:
  **cross/ΔP ≈ +1.39 (first order, suppressive), auto/ΔP ≈ −0.39 (second order,
  wrong sign), closure (cross+auto)/ΔP = 1.000** (rising to ~1.95/−0.95 by
  ℓ~1.8×10⁴). — provenance: `wl_anisotropy_paper.ipynb` cell 24 printed output;
  `WORKLOG:1199-1203` (2026-06-17 "cross-term = first-order"); matches memory file
  `wl-anisotropy-fieldlevel.md`.
  — **⚠ CAVEAT (found in sources, important):** this cross/auto decomposition was
  computed with **`sph` = the old (superseded) monopole-of-correction control**,
  not the faithful `bcm_warp` control. The memory file
  (`wl-anisotropy-fieldlevel.md:35-37`) explicitly flags this as unresolved:
  *"REVISES the punchline below: the 'BCMs miss ~half / first-order cross-term'
  claim was vs the MONOPOLE; vs a faithful radial BCM the missed fraction is ≤45%
  & scale-dependent (→0 small scales). Re-evaluate the cross/auto identity with
  κ_s = radial-BCM, not mono."* No WORKLOG entry after 2026-06-17 shows this
  re-evaluation was done. **The first-order mechanism itself (cross-term
  dominance from DMO-alignment) is plausible and independently motivated by the
  toy model, but has not been re-verified with the corrected control** — flag
  for Discussion as an open item, not a settled result.
- Why the sign/mechanism is physical: the anisotropic correction κ_bind−κ_sph is
  **spatially aligned with the halo's own DMO field** because feedback follows the
  potential — so it correlates with the very matter that lenses (first order),
  rather than acting only through its own (small) auto-power (second order).

### The 60 OAT (one-at-a-time) runs
- 30 CAMELS/TNG feedback parameters, each varied to its min and max bound
  (fiducial held fixed elsewhere) → 60 `twobound` runs (`tb0000`–`tb0059`).
  `docs/WORKLOG.md:1217`-1232 (2026-06-17, "all 60 OAT runs done"): spherical-BCM
  maps finished for fid, truth, tb0000–59 **minus tb0032/tb0043** (2 pending a maps
  re-run at the time of that entry). Notebook confirms `design = decode_design()`
  loads **30 parameters with completed runs** (i.e. 60 runs, min+max per param).
  — provenance: `wl_anisotropy_paper.ipynb` code cell defining `design`
  (`print(f'OAT design: {len(design)} parameters with completed runs')`); output
  of the significance-gating cell: "22/30 parameters move the WL anisotropy above
  the 2-sigma fid noise floor."
- **Note:** `ejection_anisotropy.ipynb` (the companion per-halo-physics notebook)
  separately reports **"57 single-knob OAT runs"** — a similar but not
  numerically identical design/count to the 60 (58-completed) figure above. Both
  numbers are directly sourced (not invented) but the discrepancy (57 vs
  58/60) is unresolved in the sources; use "60 OAT runs (58 completed at time of
  writing)" for the field-level (Cℓ/peaks) results and "57" only if directly
  citing the `ejection_anisotropy.ipynb` per-halo q2-vs-feedback figures.

### Error model
- Paired differencing: realizations share seeds, so scatter = per-realization
  paired-ratio spread, not marginal cosmic variance.
- Bootstrap/jackknife over **N_real = 8** paired realizations (one 205 Mpc/h box,
  8 plane-randomizations) — **caveat, stated by the source itself: these are
  realization/projection scatter, NOT independent cosmic volumes; errors are
  optimistic.**
- Null tests: (i) feeding `(dmo, sph, sph)` → f_aniso ≡ 0 exactly (no manufactured
  signal) — measured **f_aniso = 0.00e+00**; (ii) paired vs unpaired band-scatter
  ratio = **25.8×** (seed-pairing suppresses cosmic variance by this factor,
  confirming the paired signal is real, not chance). — `wl_anisotropy_paper.ipynb`
  cell 28 printed output.
- Alignment dilution for the stacked-cluster forecast: ⟨cos2Δθ⟩ = **0.55**,
  corresponding to Δθ≈28° (Gaussian rms σ≈26°) — **not** the commonly quoted 35°
  (cos70°=0.34 would give a different dilution). Validated by the analytic toy
  (cell 6). — `wl_anisotropy_paper.ipynb` Methods cell + cell 9 markdown;
  `WORKLOG:1213`.

## Result

### Field-level: the corrected (headline) number
See "HEADLINE CORRECTION" box above.
- **Corrected f_aniso(ℓ)** (faithful `bcm_warp` control, fid, z_s=1,
  `examples/bcm_warp_comparison.py`, `WORKLOG:1144`): **≈0.45 (ℓ=3×10³) → 0.24
  (ℓ=10⁴) → ~0 (ℓ=2×10⁴)**. f_tri (triaxiality-tracing, captured by a real BCM):
  **0.08 → 0.25 → 0.61** over the same range. f_mono (=f_aniso+f_tri, the OLD
  "40–65%" number) ≈ 0.5, roughly flat in ℓ.
- Suppression curves at this same measurement: S_bind (BIND) drops well below
  both S_sph (radial BCM) and S_mono (pure spherical) at ℓ≳few×10³; S_sph tracks
  much closer to S_bind than S_mono does, by construction (see
  `bcm_warp_comparison_fid.png`, left panel).

### Field-level: the notebook's own (superseded-control) numbers
(For completeness / provenance — these are the numbers baked into
`wl_anisotropy_paper.ipynb`'s own narrative and figures; per the correction box,
they were computed with the pre-fix circular/monopole `sph` control.)
- Notebook abstract (cell 0): "~40–55% of the small-scale C_ℓ^κκ suppression... is
  carried by structure a spherical BCM can never produce."
- Field 2 (Cℓ), measured result text (cell 25 markdown) + printed bootstrap
  numbers (cell 28): S_bind/S_sph = 0.94/0.97 (ℓ=5000), 0.90/0.95 (10⁴),
  0.91/0.96 (2×10⁴) → **f_aniso ≈ 41% → 48% → 55%** (ℓ=5000→10⁴→2×10⁴,
  *increasing* with ℓ — opposite trend to the corrected measurement).
  Band-averaged [3000,20000]: **f_aniso = 48%** (bootstrap 16–84%: 47–48%),
  significance **z = 52.3σ** (paired, N=8, "OPTIMISTIC — shared box").
- Conclusions (cell 40) restates: f_aniso(Cℓ) ≈ 41→48→55% at ℓ=5000→10⁴→2×10⁴,
  "bootstrap-significant on this lightcone."
- Pillar 1 (per-halo, one-halo correction power) f_quad: Conclusions text claims
  **"f_quad ~ 30–55% at k ≳ few h/Mpc"** — **⚠ this appears to overstate the
  directly-computed figure**: the actual Pillar-1 figure (cell 8,
  `f_quad(k) = ⟨|H2|²⟩/(⟨|H0|²⟩+⟨|H2|²⟩)`) peaks at **~20% for the cluster bin**
  (around k~2 h/Mpc, declining/noisy at higher k) and **~5% for the group bin** —
  nowhere near 30–55% in the plotted curve. Flag this internal
  narrative-vs-figure inconsistency for the Draft/Verify stages; prefer the
  directly-plotted ~5–20% range or use `\todo{}` rather than the narrative's
  30–55%.
- Per-halo peak amplitude ratio (cell 8 printed): peak |aligned c2|/peak |c0| =
  **1.44 (group)**, **0.73 (cluster)** — i.e. per halo the quadrupole correction
  can exceed the monopole correction in amplitude (group scale).
- Non-Gaussian statistics (Field 3, cell 33 printed): peaks χ² vs 0 = 23/55 dof
  (p=1.0); high-ν(>2) anisotropic shift = **−6, i.e. −1% of spherical, z=−0.5σ**
  (conservative, marginal errors → lower bound); minima χ²=7/22 dof (p=1.0),
  high-ν shift **−0, −12% of spherical, z=−0.3σ**. Memory file separately states
  peak counts **436→418 (baryon effect), →424 (spherical-only), anisotropy
  ~43% of the peak baryon effect** — this number was **not** found verbatim in
  the notebook's printed cell outputs inspected here; source only in the memory
  file, flag as lower-confidence / needs the originating cell re-identified if
  used.

### Which feedback channels drive the anisotropy
- Field-4 (Cℓ-anisotropy response, cell 37/38 printed, top responders by
  band-averaged |dA/du| where A = 100×(S_sph−S_bind) in ℓ∈[2000,20000]):
  **BlackHoleRadiativeEfficiency (dA/du=−9.90%), QuasarThreshold (+6.21%),
  IMFslope (+5.53%), SeedBlackHoleMass (+4.02%), VariableWindVelFactor (+3.91%),
  BlackHoleAccretionFactor (+3.36%), BlackHoleEddingtonFactor (+2.96%),
  WindEnergyIn1e51erg (−2.93%)**. Significance gating: fid realization noise on A,
  σ=0.161% → 2σ threshold |dA/du|>0.456%; **22/30 parameters** move the
  anisotropy above this (optimistic) noise floor.
- Pillar 3 (per-halo Q2 response, group scale, cell 13 printed): top knobs
  **VariableWindVelFactor (dQ2/du=−0.065 Msun/pc²), BlackHoleRadiativeEfficiency
  (+0.057), WindEnergyIn1e51erg (−0.032), WindFreeTravelDensFac (+0.029),
  QuasarThreshold (−0.029), SeedBlackHoleMass (−0.029), BlackHoleAccretionFactor
  (−0.025), MaxSfrTimescale (−0.025)**.
- Across both the field-level and per-halo response rankings the consistent
  message: **AGN kinetic-mode/black-hole parameters and wind parameters** are
  the dominant controls of the anisotropic fraction (both amplitude and, per
  `ejection_anisotropy.ipynb`, orientation).
- `ejection_anisotropy.ipynb` (57 OAT runs; per-halo q2, group scale, cell 16
  printed): top knobs controlling κ-residual anisotropy q2: **WindEnergyIn1e51erg
  (dq2k/du=−0.0115, dq2y/du=+0.0561), IMFslope (+0.0111/−0.0489),
  VariableWindVelFactor (−0.0111/+0.0319), WindFreeTravelDensFac
  (+0.0086/−0.0397), WindEnergyReductionFactor (−0.0060/+0.0225)**. Fiducial
  alignment cos2(θ_Δ−θ_DM) (group) = **−0.773** (negative ⇒ ejection
  perpendicular to the DM major axis — bipolar signature, baryons redistribute
  along the halo **minor** axis).
- Ejection-anisotropy amplitude by radius (cell 5, "Figure 1"): WL convergence
  residual q2(κ−κ_dmo) is **mildly anisotropic (3–5%)**, roughly flat/declining
  with r/r200c; tSZ pressure q2(y) is **strongly anisotropic in the outskirts**,
  rising from ~5–12% (inner) to **~35–43%** at r~2.9 r200c.
- Environment jackknife (Test 4c, `ejection_anisotropy.ipynb`): outer-tSZ
  quadrupole is largely LSS/filament-driven, not pure AGN bipolarity — filament
  environment **≈52%** vs void environment **≈29%** at 2.7 r200. The κ (WL)
  quadrupole is the cleaner feedback-only probe; the outer-y quadrupole needs
  filament decontamination.

### Detectability (Pillar 2)
- Forecast S/N for the stacked-cluster tangential-shear quadrupole (with 0.55
  alignment dilution), printed cell 10, **cluster bin only**:
  N_cl=1000: LSST Y10 0.5, Euclid 0.5, Roman HLS 0.7;
  N_cl=3000: LSST 0.8, Euclid 0.8, Roman 1.3;
  N_cl=10000: LSST 1.5, Euclid 1.5, Roman 2.3.
  Group-sample S/N is lower still (figure only, group panel tops out well under
  1 across the plotted N_cl range up to ~1.6×10⁴).
- **⚠ Discrepancy found:** the Conclusions text (cell 40) claims "S/N ~3–5 for
  N_cl~3×10³–10⁴," which is **not supported by the printed/plotted forecast** —
  the actual cluster-bin curve only approaches ~3 (Roman) near the top of the
  plotted range (N_cl≈1.6×10⁴), and LSST/Euclid remain ≲2 even there. Use the
  directly computed numbers above; flag the Conclusions' "3–5" claim as
  unverified/overstated.

## Discussion (sourced argument, from notebook Discussion + Conclusions)

- A spherical BCM has **zero quadrupole ⇒ zero cross term** in the exact
  ΔP=cross+auto identity, so (per the notebook's own — not-yet-corrected-control —
  measurement) it can only fit the true P(ℓ) by **over-ejecting radially**,
  biasing its recovered gas profile by order the anisotropic fraction. Concretely:
  fiducial S_sph≈0.93 at k~17 h/Mpc vs S_bind≈0.86; a spherical model tuned to
  0.86 must inflate ejection to compensate. — cell 39.
- Three testable consequences claimed: (1) gas profile biased even when P(ℓ)
  is right → breaks multi-probe consistency (tSZ y, X-ray f_gas, kSZ/τ, cluster
  lensing); (2) fails non-Gaussian statistics — ties to the team's own earlier
  result that BCMs reproduce hydro peak counts only to ν≲4 (**Lee et al. 2022,
  MNRAS 519, 573; arXiv:2201.08320**) — "the missing ν is the missing
  quadrupole"; (3) won't transfer across feedback models/cosmologies because the
  cross term depends on feedback-geometry/halo-shape alignment.
- Determination: BCMs need an **aligned m=2 correction term** c2(r), calibrated
  the way c0(r) already is; BIND supplies this per halo. A spherical BCM remains
  adequate only for a single 2-pt statistic at ℓ≲few×10³ where the baryon effect
  itself is ≲2%.
- Ties to Paper II (brief's cross-reference): analytic/spherical BCMs cannot
  absorb this term the way a data-driven (2-template) nuisance parametrization
  might — worth an explicit sentence linking to the Paper II N=2-templates
  result (`analysis/wl-cosmo-bias`, not directly re-verified here — cross-check
  with Paper II's dossier).

## Mandatory caveats (brief's list, verified present in sources) + additional found caveats

1. **PRELIMINARY: control revision pending, number expected to drop.** ✅ sourced
   — and per the correction box above, the revision **has already been run**
   (`bcm_warp_comparison.py`, 2026-06-17) and the number **did drop** (to a
   scale-dependent ≈45%→~0), but this has not been propagated back into the
   notebook's own Field-2/Discussion/Conclusions numbers, which still show the
   pre-fix 41→48→55% (increasing with ℓ). The Letter should present the
   corrected, declining-with-scale number as the headline and flag the older
   number as superseded.
2. **Fiducial TNG only (no feedback-dependence of the fraction yet)** — for the
   *corrected* (bcm_warp) measurement, true: `bcm_warp_comparison.py` has only
   been run on `fid` (`docs/WORKLOG.md:1156`, "TODO: rerun `--run tbNNNN` once
   the twobound maps finish for the param dependence of f_aniso"). The
   parameter-dependence figures that DO exist (Field 4, Pillar 3, ejection
   anisotropy vs feedback) all use the **superseded** circular/monopole control,
   not the faithful radial-BCM-warp control — another instance of the same
   not-yet-propagated-fix caveat.
3. **First-order decomposition only** — ✅ sourced; the cross/auto split is exact
   (closure to machine precision) but was computed against the superseded
   control (see Method section caveat above); not re-derived with κ_s=bcm_warp.
4. **N_real=8 realizations of one 205 Mpc/h box** — errors are
   realization/projection scatter, not independent cosmic volumes; "optimistic."
5. **CAMELS-25/SB35 training caps clusters at M200c ≲ 5×10¹⁴ Msun/h**; profiles
   measured at z~0, placed at z_l=0.3 for the Pillar-2 forecast (shape assumed
   ≈self-similar); 128² patches ⇒ ~50 kpc/h resolution floor. — Conclusions
   caveat paragraph (cell 40).
6. Peak/PDF significance uses **per-field marginal errors** (paired cancellation
   ignored) ⇒ conservative, a **lower bound** on the true paired significance.
7. Pillar-2 alignment dilution is a single scalar (0.55); a calibrated
   BCG-misalignment model is explicitly listed as future work, not done.
8. Internal inconsistencies found (not brief-mandated, but should be flagged to
   Draft/Verify): (a) Pillar-1 f_quad Conclusions text (30–55%) vs directly
   plotted figure (~5–20%); (b) Pillar-2 Conclusions S/N claim (~3–5) vs printed
   forecast numbers (≤2.3 at N_cl=10⁴); (c) 60 vs 57 OAT-run counts between the
   two notebooks; (d) the on-disk `examples/figures_lightcone/fig_anisotropy_stats.pdf`
   (found during figure search) makes an apparently unrelated claim ("radial
   features already capture each statistic," "non-radial adds nothing" to a
   feedback-response regression R²) with **no WORKLOG entry or script reference
   found** documenting its origin — recommend NOT citing this figure/claim in
   the Letter without further sourcing (see Gaps).

## FIGURES

| # | path | shows | serves | provenance |
|---|------|-------|--------|------------|
| 1 | `examples/figures_lightcone/bcm_warp_comparison_fid.png` (on disk, main tree) | 3-panel: (a) suppression S(ℓ) for BIND/radial-BCM/pure-spherical; (b) residual-field power C_ℓ^ΔΔ for total-baryon / pure-sph-error / BCM-misses / triaxiality; (c) decomposition fractions f_aniso, f_tri, f_mono(ℓ). **This is the corrected, faithful-control headline figure.** | Result (primary) | `WORKLOG:1144-1157` (2026-06-17 "three-way kappa comparison"); generated by `examples/bcm_warp_comparison.py` |
| 2 | `figs_raw/wl_anisotropy_paper/cell020_out0.png` | 4-panel maps: DMO, spherical-BCM, BIND (full), and the difference κ_bind−κ_sph ("anisotropy") at z_s=1, same halos/seed | Method/Intro visual (painted-vs-warped patch/field visual) | notebook cell 20, heading "Field 1 — the maps" |
| 3 | `figs_raw/wl_anisotropy_paper/cell024_out0.png` | Cross/auto decomposition of ΔP=P_bind−P_sph: ΔCℓ contributions (left) and fraction-of-ΔP (right) vs ℓ, showing cross≈first order dominates, auto≈second order subdominant/wrong-sign | Method (mechanism) — ⚠ built with superseded `sph` control, caveat must accompany | notebook cell 24, "§4.2 — why the anisotropy reaches Cℓ at first order" |
| 4 | `figs_raw/wl_anisotropy_paper/cell037_out0.png` | Field 4: response curves (top-10 params, anisotropic suppression % vs normalized u) + scale-resolved ∂(S_sph−S_bind)/∂u heatmap (param × ℓ) | Result — which feedback channels drive it | notebook cell 37, "Field 4 — the parameter control of anisotropy" |
| 5 | `figs_raw/ejection_anisotropy/cell005_out0.png` | Figure 1: radial q2(κ) [3–5%, mild] and q2(y) [rises to 35–43%, strong] anisotropy profiles, fiducial vs TNG truth, group/cluster bins | Result/Discussion — supporting ejection-anisotropy physics (optional per brief) | `ejection_anisotropy.ipynb` cell 5 |
| 6 | `figs_raw/ejection_anisotropy/cell019_out0.png` | Figure 4: bar chart of which feedback knobs change the *orientation* of ejection (bipolar vs shape-tracing); ★=AGN jet-reorientation factor | Discussion (optional, ejection driver) | `ejection_anisotropy.ipynb` cell 19 |
| 7 | `figs_raw/wl_anisotropy_paper/cell006_out0.png` | Toy-model validation: injected pure aligned quadrupole recovered exactly by the estimator; null test (no signal in → no f_quad out); alignment-dilution cos2Δθ curve validated against the 0.55/28° value | Methods appendix (validation), not core Result | notebook cell 6, "toy example... known answer" |
| 8 | `figs_raw/wl_anisotropy_paper/cell008_out0.png` | Pillar 1: correction profiles c0/c2 (left), f_quad(k) per-halo anisotropic fraction (right) — group ~5%, cluster ~20% peak | Backup/appendix — flagged inconsistency with Conclusions text (30–55%) | notebook cell 8, "Pillar 1" |
| 9 | `figs_raw/wl_anisotropy_paper/cell010_out0.png` | Pillar 2: Stage-IV S/N forecast curves (group/cluster × LSST/Euclid/Roman) vs N_clusters | Discussion (detectability), if space allows | notebook cell 10, "Pillar 2" |
| 10 | `figs_raw/wl_anisotropy_paper/cell032_out0.png` | Field 3: peak counts, f_aniso(ν) for peaks/minima (noisy), convergence-PDF baryon residual (sph−dmo vs bind−dmo) | Backup — noisy middle panel, not recommended as primary | notebook cell 32, "Field 3" |
| — | `examples/figures_lightcone/fig_anisotropy_stats.pdf` (on disk) | Bar charts: feedback-response CV R² with/without "anisotropy" features per statistic (S(ℓ), peaks, V0-V2), and incremental R² — claims non-radial info adds ~nothing | **NOT recommended** — no WORKLOG/script provenance found (see Gaps/caveat 8d) | unresolved |

Not extracted/inspected: remaining `wl_anisotropy_paper.ipynb` output cells not
listed above (17 total extracted, all listed in
`figs_raw/wl_anisotropy_paper/manifest.json`); remaining `ejection_anisotropy.ipynb`
cells (7 total extracted, all listed in `figs_raw/ejection_anisotropy/manifest.json`).

## CITATIONS

**Found directly in the sources (engine/notebook text), verify bibliographic
details independently:**
- Schneider & Teyssier 2015 — BCM origin paper. `wl_anisotropy_paper.ipynb` cell 0;
  `src/bind/inference/spherical.py` docstring.
- Aricò et al. 2020 — BCM. `wl_anisotropy_paper.ipynb` cell 0.
- Mead et al. 2021 — BCM (HMcode-style). `wl_anisotropy_paper.ipynb` cell 0.
- **Lee et al. 2022, MNRAS 519, 573; arXiv:2201.08320** — "our earlier result" that
  BCMs reproduce hydro peak counts only to ν≲4; real, already-published paper (not
  "in prep"), same author group. `wl_anisotropy_paper.ipynb` cell 39/40.
- Planck Intermediate V (2013), Fig. 15 — azimuthal scatter ~20% in stacked
  cluster pressure profile at r500. `ejection_anisotropy.ipynb` cell 13.
- Lau et al. (2011) — X-ray morphology quadrupoles in simulated clusters, ~15–25%
  at r500. `ejection_anisotropy.ipynb` cell 13.
- Biffi et al. (2016) — same context as Lau+2011. `ejection_anisotropy.ipynb`
  cell 13.

**Brief's seed citations — NOT found in the sources searched (repo-wide grep of
both notebooks + `bcm_warp_comparison.py` + `spherical.py`); need external
verification by the Cite stage, or drop:**
- Dai, Feng & Seljak 2018 — no occurrence found.
- Osato+ (halo triaxiality WL) — no occurrence found.
- van Daalen+2020 — only a generic, year-less mention of "van Daalen" (the
  f_gas-suppression correlation, in `ejection_anisotropy.ipynb`); no explicit
  2020 citation with number found in these two notebooks. (Note: van Daalen 2020
  IS cited with specifics elsewhere in the repo, e.g. the f_gas-bridge memory
  file, but not in the Paper-V source material itself — cite with care / verify
  it's the intended paper.)

**Cross-paper (self-citations):** Paper I, Paper II, Paper III (BIND suite,
"in prep." per PLAN.md convention) — Paper II connection (data-driven templates
vs analytic BCMs) is argued in the Discussion above but not literally cross-cited
in the source notebooks (they predate the paper-suite split).

---

## Gaps (could not source — do not guess)

1. **Parameter-dependence of the corrected f_aniso.** The faithful `bcm_warp`
   three-way comparison has only been run on `fid` (z_s=1). No corrected
   (post-fix) OAT/feedback-response figure exists — all "which feedback channels
   drive it" figures (Field 4, Pillar 3, ejection anisotropy) use the superseded
   circular/monopole control. If the Letter's Result section claims feedback
   channels for the *corrected* anisotropy specifically, that is unsourced —
   present it as "which channels drive the (upper-bound) anisotropic signal
   under the earlier control" instead, or `\todo{}`.
2. **Re-derivation of the cross/auto (first-order) mechanism against the
   faithful control.** Flagged explicitly as un-done in the memory file. If the
   Letter wants to claim the *corrected* 45%→~0 signal is still "first order,"
   that specific claim is unsourced (only the old-control 41→48→55% signal was
   shown to be first-order/cross-dominated).
3. **Provenance of `examples/figures_lightcone/fig_anisotropy_stats.pdf`.** No
   WORKLOG entry, no notebook, no script (searched both worktree and main-tree
   `examples/`) references its filename or its "radial features already capture
   each statistic" / "non-radial adds nothing" claim. Do not use in the Letter
   without locating the source, since its content could be misread as
   contradicting the paper's thesis (it answers a different question — SBI
   feedback-parameter-inference feature importance — not the WL-suppression
   anisotropic fraction).
4. **60 vs 57 OAT-run count reconciliation.** Two notebooks report slightly
   different completed-run counts for what is nominally the same 30-parameter
   ×2-bound twobound design. Not reconciled in the sources.
5. **Pillar-1 f_quad "30–55%" and Pillar-2 "S/N~3–5" Conclusions claims** are not
   supported by the directly-computed/plotted cell outputs (see caveat 8a/8b
   above) — treat as internally unverified, use the directly-plotted numbers
   instead, or `\todo{}` if the Conclusions framing is needed verbatim.
6. **Bibliographic details** (journal, volptr, DOI/arXiv) for Schneider &
   Teyssier 2015, Aricò et al. 2020, Mead et al. 2021, Planck Int. V (2013), Lau
   et al. 2011, Biffi et al. 2016 — names/years only are in the sources; full
   entries need the Cite stage (arXiv/ADS lookup).
7. **Peak-count "436→418→424, ~43% anisotropic"** number (memory file only,
   not found in the notebook's own printed cell text) — needs the originating
   cell re-identified before use, or drop.
