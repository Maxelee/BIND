# Evaluating the ejection-heating coupling for a possible Sec. 3c companion panel

**Script:** `papers/01_pipeline/audits/companion3c_eval.py` (self-contained, read-only on
`/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.npz`, canonical 253-run Sobol suite,
group mass bin = index 0, `log10 M500c/(Msun/h) = 13.125`).

## Question

`proto_bridge_hero.py` panel (b) found `r(ln f_gas, ln Y - ln f_gas) = 0.86` at the population
level (253 Sobol runs), where "ejection" = `ln f_gas` (group bin) and "heating" =
`ln Y - ln f_gas` (group bin). In the retired 57-run 1P design these two axes decorrelated,
which originally motivated an ejection-heating plane figure. Before reviving that panel as a
Sec. 3c companion to fig 20(b), or letting fig 20(b) stand alone, we need to know: is the 0.86
coupling a **design-response artifact** of the Sobol prior (its dominant direction moves both
axes together), or an **intrinsic degeneracy** between the two observables — and either way,
does the heating axis add anything the WL/tSZ observables don't already get from ejection alone?

## Results

### 1. Raw coupling — reproduced

`r(ln f_gas, heat) = 0.860` (95% Fisher CI [0.824, 0.889]; bootstrap median 0.860, 95% CI
[0.821, 0.892]; p = 4e-75, n = 253). Spearman rho = 0.820. Matches the prototype's 0.86 exactly.

### 2. Partial correlations

| Control | residual r | 95% CI | R²(ln f_gas \| control) | R²(heat \| control) |
|---|---|---|---|---|
| none (raw) | 0.860 | [0.824, 0.889] | — | — |
| log10(WindEnergyIn1e51erg) alone | **0.860** | [0.824, 0.889] | 0.188 | 0.064 |
| full 30-dim standardized astro vector (OLS) | **0.783** | [0.731, 0.827] | 0.665 | 0.749 |

Controlling for the single dominant feedback lever (WindEnergy) changes the coupling **not at
all** (0.860 → 0.860) — WindEnergy alone only explains 19%/6% of the two axes' variance, so it
is not what is driving the coupling. Controlling for the *entire* 30-dimensional astro design —
even though each axis is individually 65-75% linearly predictable from the 30 parameters — only
knocks r down to 0.783, still enormous (p = 9e-54). **The coupling survives removal of the full
linear Sobol-design response.** This already rules out "shared dominant response direction in
parameter space" as the primary explanation.

### 3. Does the heating axis add WL/tSZ information beyond ejection?

WL suppression `S(ell=5000, z_s=1)`:
- `r(S, ln f_gas) = 0.713`
- `r(S, heat | ln f_gas) = -0.012` (p = 0.85, consistent with zero)
- R²(S ~ ln f_gas) = 0.508 → R²(S ~ ln f_gas + heat) = 0.508 (Δ = +0.0002)

tSZ `Cl_yy(ell≈2981)`:
- `r(Cl_yy, ln f_gas) = 0.728`
- `r(Cl_yy, heat | ln f_gas) = +0.019` (p = 0.76, consistent with zero)
- R²(Cl_yy ~ ln f_gas) = 0.530 → R²(Cl_yy ~ ln f_gas + heat) = 0.530 (Δ = +0.0004)

**Adding the heating axis buys essentially zero additional explanatory power** for either
headline observable once ejection is known — for both the WL suppression amplitude and the tSZ
power spectrum, the incremental R² is in the fourth decimal place and the partial correlation of
heat is statistically indistinguishable from zero.

### 4. PCA of the standardized (ln f_gas, heat) plane

PC1 = 93.0%, PC2 = 7.0% of variance. This is mechanically `(1±|r|)/2` for two standardized
variables (0.930/0.070 exactly), so it is a re-expression of the raw r, not new information —
but it is the right way to answer the "plane vs. line" question the 1P figure posed: **7% in the
second axis is a near-line, not a genuine plane** (a real plane would show something closer to
50/50, or at least a non-trivial double-digit-percent split well above measurement/binning
noise).

### 5. Robustness checks

- **Within-WindEnergy quartiles**: r stays in [0.81, 0.90] across all four quartiles of the
  dominant lever — the coupling is essentially invariant to conditioning on the single strongest
  parameter, consistent with (2a).
- **Single-parameter scan** (30 astro params): no individual parameter dominates either axis
  (largest |r| = 0.42 for WindEnergy on `ln f_gas`, -0.59 for IMFslope on heat) — nothing points
  to a single confounding lever being "secretly" responsible for the 0.86.

### 6. Mechanistic decomposition (the decisive piece)

`r(ln f_gas, ln Y) = 0.988` at the group bin — **`ln f_gas` and `ln Y` are themselves nearly
degenerate**, not just correlated. Physically this makes sense: at fixed mass bin, the
integrated Compton-Y is close to a direct tracer of how much gas is retained (`Y ~ n_e`-weighted
gas content), so ejecting gas (lower `f_gas`) and losing SZ signal (`lower Y`) are close to the
same thing at group scale.

The OLS log-log slope of `ln Y` vs. `ln f_gas` is **1.350**, not 1. But `heat = ln Y - ln f_gas`
implicitly assumes a *unit* slope subtraction. Since the true slope is super-linear (1.35), the
literal `heat` axis is not the orthogonal residual of `Y` on `f_gas` — it retains a
`(slope − 1) = +0.35`-unit echo of `ln f_gas` by construction. The properly orthogonalized
residual of `ln Y` on `ln f_gas` (computed directly) has `r(ln f_gas, residual) = -0.0000` with
`ln f_gas`, as it must by definition — and this residual is numerically the same quantity as the
"heat | ln f_gas" partial used in checks (3): the WL/tSZ correlations there (`-0.012`, `+0.019`,
both consistent with 0) are already the answer for what a *cleanly* orthogonalized heating axis
would show. There is no hidden second signal being masked by the fixed-slope definition.

## Verdict

The r = 0.86 population-level coupling is **predominantly intrinsic, not a Sobol-design
artifact**:

- It is untouched by controlling for the dominant single lever (WindEnergy: 0.860 → 0.860), and
  only partially reduced (0.860 → 0.783, still p ≈ 1e-53) by controlling for the *entire*
  30-dimensional linear design response — even though both axes are 65-75% parameter-predictable
  individually. A pure design-direction artifact would have collapsed toward zero under the
  full-parameter partial; it did not.
- The mechanism is definitional/physical, not sampling-dependent: `ln f_gas` and `ln Y` are
  nearly degenerate at the group mass bin (r = 0.988), and the super-linear (1.35, not 1)
  log-log `Y`–`f_gas` slope means the fixed unit-slope subtraction used to build "heat" bakes in
  a real ejection echo rather than removing it.

The retired 1P design likely decorrelated the two axes not because the underlying degeneracy is
absent, but because a 57-run one-parameter-at-a-time sweep restricts the joint (`f_gas`, `Y`)
range each lever explores in isolation, which can locally mask a relationship that is globally a
near-line across the full, high-dimensional Sobol population. That is a **support/coverage**
difference between the two designs, not evidence that the canonical suite's coupling is spurious.

Separately — and this is the more paper-relevant fact regardless of the artifact-vs-intrinsic
question — the heating axis, honestly orthogonalized, **carries no measurable independent
information about the WL suppression amplitude or the tSZ `Cl_yy`** once ejection is known
(ΔR² ≈ 2–4×10⁻⁴, partial r consistent with 0 at ~0.8σ). Even in the counterfactual world where
the 0.86 coupling *were* purely a design artifact, adding the axis back would not recover any
observable-relevant signal fig 20(b) is missing.

## Recommendation

**(b) — let §3c stand on fig 20 panel (b) alone.** Do not revive a dedicated ejection-heating
companion panel.

Reasoning:
1. On the canonical 253-run suite that the rest of Paper I is built from, the ejection-heating
   "plane" is a 93/7 near-line (PC2 = 7.0%), not the ~50/50 genuine plane the retired 1P figure
   suggested. Reviving the panel would visually claim a two-axis degeneracy-breaking result the
   canonical data does not support.
2. The residual coupling survives aggressive controls (full 30-parameter OLS partial still 0.78),
   so it cannot be dismissed as "just the dominant Sobol direction" and then waved away — it
   reflects a real, near-degenerate `f_gas`–`Y` scaling relation at group mass. A companion panel
   built on it would mostly be re-displaying that scaling relation under a different label, which
   is arguably better shown as the scaling relation itself (already implicit in fig 20) than
   dressed up as an independent "heating" axis.
3. Most decisively: the properly orthogonalized heating residual has essentially zero partial
   correlation with both the WL suppression amplitude and the tSZ power spectrum once ejection is
   known (ΔR² ~ 10⁻⁴). A companion panel's main scientific justification would be "here is a
   second axis of feedback response the WL/tSZ summary in fig 20(b) doesn't capture" — and the
   data says that axis, once honestly defined, does not carry incremental WL/tSZ information at
   this mass bin and these observables.
4. fig 13 (the per-halo profile-stack figure that would have anchored a physically distinct,
   per-halo version of this story) has been removed from the paper, so there is no companion
   scaffolding left to hang a revived panel on; introducing a new population-level panel now
   would need to justify itself entirely on the numbers above, and it does not clear that bar.

No third construction is warranted: the natural "fix" (use the properly orthogonalized heating
residual instead of the literal `ln Y - ln f_gas`) was tested directly via the `heat | ln f_gas`
correlations in check 3 and shows the same null result, so there is no alternate axis definition
that rescues independent structure at this mass bin with these two headline observables.

## Caveats / scope

- This is a linear (Pearson/OLS) analysis at one mass bin (group scale, index 0) and two
  observable choices (`S(ell=5000, z_s=1)`, `Cl_yy(ell≈3000)`). It does not rule out nonlinear
  structure, other mass bins, or other summary statistics (e.g. peak counts, MFs) carrying
  residual heating-axis information — those were out of scope for this audit but could be
  checked cheaply with the same `emulator_dataset_xpkfix.npz` keys if a reviewer pushes back.
- The "full 30-dim OLS partial" is a linear projection; with n = 253 and p = 30 it is
  well-determined (not rank-deficient) but does not capture nonlinear parameter dependence that a
  GP or neural response surface would.
