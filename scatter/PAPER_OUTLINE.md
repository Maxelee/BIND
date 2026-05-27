# BIND Scatter Paper — Structural Outline

**Status:** Populated with placeholders; agent fills numbers when Jacobian completes.
**Note:** All claims marked ⊕ depend on Phase 2 calibration passing.  All claims marked ⊗ require Phase 3 Jacobian to be complete without numerical issues.

---

## 1. Introduction

**Framing:** The next generation of cluster surveys (eROSITA, CMB-S4, Rubin LSST) will constrain cosmology through the cluster abundance and mass-function. The limiting systematic is the *mass–observable* scatter: how much does $M_{\rm gas}$, $M_\star$, or a weak-lensing signal vary at fixed halo mass? Current cluster-cosmology analyses assume this scatter is parameter-independent, i.e., a single number $\sigma_{\rm intrinsic}$ is sufficient. If the scatter depends on feedback parameters, it becomes a nuisance that must be marginalized with a prior — and that prior should come from theory.

**Gap:** Deterministic BCMs (Schneider & Teyssier 2015, Baryonification, etc.) have zero scatter by construction. Hydro suites (TNG, SIMBA, BAHAMAS) have scatter but only at one or a few parameter values. No tool currently maps $\sigma(\theta)$ continuously across the multi-dimensional feedback parameter space.

**BIND fills this gap.** Parameter-conditioned generative model → we can sample from $p(Y | \theta, c_h)$ for any $\theta$.

**Headline claim:** [TO FILL when Jacobian complete]
- If Fig 2 shows separation: "The scatter direction in 35-D feedback space is approximately orthogonal to the mean direction, so the scatter encodes additional feedback information."
- Fallback: "Scatter is parameter-independent at the $<X\%$ level, supporting standard cluster-cosmology assumptions."

---

## 2. BIND Model (brief recap)

**One paragraph.** Cite forthcoming methods paper. BIND (Baryonic INformation Diffusion) is a 2D flow-matching generative model conditioned on DMO dark-matter patches and 35 IllustrisTNG feedback parameters. Trained on the CAMELS simulation suite (IllustrisTNG, L50n512, 1P + CV + SB35). Generates (DM_hydro, Gas, Stars) patches at 128×128 pixels (6.25 Mpc/h), K independent posterior samples for each conditioning.

**Model details:** Two-head design (DM+Gas head + Stars occupancy-density head). EMA weights used for inference. 4-channel output.

---

## 3. Posterior Calibration Against CV Ground Truth

**Status:** Phase 2 PASSED ✓ (all 11 headline observables < 30% scatter error, fiducial θ)

### 3.1 Observable set

[TO FILL] List of 12 headline observables (Table 1). Dropped: $f_b$, $f_b/f_{b,\rm cosmic}$, $R_{\rm closure}/R_{200}$, $\Sigma_{{\rm gas},c}$ — see Appendix A.

### 3.2 Calibration results ⊕

- CV ground truth: 27 simulations × N_h halos, scatter across sims at fiducial $\theta$.
- BIND posterior: K=10 samples per halo at fiducial $\theta$.

**Calibration table** → `scatter/cv_calibration.csv`

Key results:
- $M_{\rm DM}$: BIND_total/truth = [TO FILL: ~1.00 from CSV]
- $M_{\rm gas}$: BIND_total/truth = [TO FILL: ~0.98]
- $M_\star$: BIND_total/truth = [TO FILL: ~0.97]
- $q_{\rm DM}$, $q_{\rm gas}$, $q_\star$: [TO FILL]
- $\Delta q_{\rm DM}$: [TO FILL: ~0.91 (slightly lower, may be acceptable)]

**Conclusion:** BIND reproduces the CV scatter at the fiducial to within [TO FILL: 2–11%] relative error, satisfying our 30% calibration threshold.

### 3.3 Inter vs intra decomposition ⊕

From Fig 4: $\sigma_{\rm inter} \gg \sigma_{\rm intra}$ for mass observables (ratio ~[TO FILL: ~10:1]), confirming that the posterior's spread is dominated by real halo-to-halo variation, not model stochasticity.

For shape observables, $\sigma_{\rm inter} \approx \sigma_{\rm intra}$ (ratio ~[TO FILL: ~1.5:1]), suggesting the model adds non-negligible stochasticity to shapes — this is expected since shapes are less constrained by the DMO conditioning.

---

## 4. Mean and Scatter Jacobian

⊗ (Requires Phase 3 to complete)

### 4.1 Definition

$J^{(\mu)}_{oj} = \frac{\partial \langle \bar Y^{(o)} \rangle}{\partial \theta_j}$,  $\quad$  $J^{(\sigma)}_{oj} = \frac{\partial \log \sigma_{\rm inter}^{(o)}}{\partial \theta_j}$

Computed by central finite differences at $\varepsilon = 0.05$ in normalized $\theta$-space.  $N_h = 200$ CV halos, $K = 10$ samples each.

### 4.2 Top mean-movers ⊗

[TO FILL after Jacobian] For $\log M_{\rm gas}$:
- Top-3 $|J^\mu|$: [parameter names and values]
- Expected: $\Omega_m$ in top-3 (sanity check). Passed: [YES/NO]

For $\log M_\star$:
- Top-3 $|J^\mu|$: [TO FILL]

### 4.3 Top scatter-movers ⊗

For $\sigma_{\rm inter}(\log M_{\rm gas})$:
- Top-3 $|J^\sigma|$: [TO FILL]
- Are they the same as the mean-movers? [TO FILL]

For $\sigma_{\rm inter}(\log M_\star)$:
- Top-3 $|J^\sigma|$: [TO FILL]

---

## 5. Headline Result: Scatter vs Mean

⊗ (Requires Phase 3 + Fig 2)

### 5.1 The fig2 scatter plot

For each of the 35 parameters and 3 headline observables, we plot $J^\sigma_{oj}$ vs $J^\mu_{oj}$. Each point is one (parameter, observable) pair.

**Expected interpretations:**
- Points on the x-axis (large $J^\mu$, small $J^\sigma$): parameters that shift the mean without changing scatter → these are the "cosmological" directions where standard BCMs work well.
- Points on the y-axis (large $J^\sigma$, small $J^\mu$): parameters that change scatter without shifting the mean → unique BIND information.
- Off-diagonal: parameters that affect both.

**Actual result:** [TO FILL] Qualitative description of Fig 2.

### 5.2 Cluster-cosmology implication

If the scatter direction is orthogonal to the mean direction:
- The scatter encodes additional feedback information not accessible from mean relations.
- Including scatter in cluster-cosmology analyses could break the $\Omega_m$–$\sigma_8$ degeneracy along the feedback direction.
- Quantify: the angle between the mean-response and scatter-response vectors in 35-D: cos θ = [TO FILL].

If the scatter direction is collinear with the mean direction:
- See fallback discussion (§0 of PLAN_scatter_paper.md).

---

## 6. Scatter Contours in (A_SN1, A_AGN1) Plane

⊗ (Requires Fig 3)

[TO FILL] 5×5 grid in (A_SN1, A_AGN1) normalized parameter space. Show that $\sigma_{\rm inter}(\log M_{\rm gas})$ is a non-trivial function of these two parameters. Contours suggest [TO FILL: monotone / non-monotone / saddle point].

---

## 7. Robustness and Limitations

Results from Phase 5:

| Check | Result |
|-------|--------|
| K budget (K=5,10,20) | [TO FILL after check] |
| Mass-bin stability | [TO FILL] |
| LOS contamination | [TO FILL] |
| Step-size linearity | [TO FILL] |
| Seed reproducibility | [TO FILL] |

**Limitations:**
- BIND is trained on IllustrisTNG physics only; SN/AGN parameterization is IllustrisTNG-specific.
- 2D projected observables; 3D observables would be cleaner but harder to observe.
- N_h=200 halos gives SE ~ [TO FILL] on $J^\sigma$ — see Table [robustness] for SNR.
- Interpolant: flow-matching (FM) interpolant; stochastic interpolant would give different intra-scatter.

---

## Appendix A: LOS Contamination Check

[Refer to Fig in scatter/robustness/los_contamination.pdf]

The dropped observables ($f_b$, $R_{\rm closure}/R_{200}$, $\Sigma_{\rm gas,c}$) show [TO FILL: similar mean responses but elevated scatter variance], supporting our decision to exclude them from the headline analysis.

---

## Numbers Reference

**To fill in from CSV and Jacobian:**
- Number of halos per mass bin: [from calibration_cv.py output]
- Calibration ratios sigma_BIND/sigma_truth per obs: [from cv_calibration.csv]
- Top-5 J_mean param names and values for M_gas, M_star, dq_DM: [from Jacobian]
- Top-5 J_log_sigma param names and values: [from Jacobian]
- Angle between mean and scatter direction in 35-D: [to compute in figures.py]
- Fig 3 sigma contour range: [from fig3 data]
- Phase 5 SUMMARY.md results

---

## Figures

| Figure | File | Status |
|--------|------|--------|
| Fig 1: Calibration | paper_figures/scatter/fig1_calibration.pdf | ✓ Done |
| Fig 2: Scatter vs mean Jacobian | paper_figures/scatter/fig2_scatter_vs_mean.pdf | ⊗ Pending Jacobian |
| Fig 3: Scatter contours | paper_figures/scatter/fig3_scatter_contours.pdf | ⊗ Pending |
| Fig 4: Inter vs intra | paper_figures/scatter/fig4_inter_vs_intra.pdf | ✓ Done |

---

## Claim Dependency Matrix

| Claim | Depends on Phase |
|-------|-----------------|
| BIND reproduces CV scatter | Phase 2 ⊕ |
| sigma_inter >> sigma_intra for masses | Phase 2 ⊕ |
| Scatter direction ⊥ mean direction | Phase 3 ⊗ |
| Specific params move scatter but not mean | Phase 3 ⊗ |
| Scatter is (non-)linear in A_SN1/A_AGN1 | Phase 4 (Fig 3) ⊗ |
| K=10 sufficient | Phase 5 check 1 |
| Results mass-bin stable | Phase 5 check 2 |
| LOS observables defensibly dropped | Phase 5 check 3 |

---

*Last updated: 2026-05-12. Agent-generated outline. Human will rewrite prose.*
