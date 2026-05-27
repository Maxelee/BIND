---
title: "BIND Scatter Paper — Convincingness Diagnostic Brief (2026-05-13 evening)"
tags: [BIND, Active, Implementation, AgentBrief]
project: BIND
created: 2026-05-13
status: ready_to_execute
related: [BIND Scatter Paper - Phase 5 Synthesis 2026-05-13, BIND Scatter Paper - Diagnostic Autopilot Brief]
---

# BIND Scatter Paper — Convincingness Diagnostic Brief

**Audience.** Claude Sonnet 4.6 in autopilot in the BIND repo, continuing from the Phase 5 synthesis ([[BIND Scatter Paper - Phase 5 Synthesis 2026-05-13]]). All Phase 1 and Phase 4 outputs from this afternoon's diagnostic run are on disk and reusable.

**Why this exists.** The Phase 5 synthesis communicated the contamination finding in a way that implied "BIND is not sensitive to feedback parameters" and did not address the fact that BIND's DMO conditioning input has cosmology implicitly baked in. The user's intuition — that BIND must be sensitive to feedback, and that "varying $\Omega_m$" means more than just changing a conditioning-vector entry — is correct and the diagnostic suite needs to either confirm or refine the Phase 5 conclusions with a tighter set of plots.

**Mission.** Produce 6 figures that, taken together, resolve the user's two specific concerns:
1. *"BIND must be sensitive to feedback — show me."*
2. *"Cosmology is in the DMO, not just the parameter vector — does that change the contamination conclusion?"*

Plus answer: does the contamination pattern survive a cleaner re-analysis, and which observable–parameter pair should headline the paper?

---

## 0  Operating principles

- Reuse all Phase 1/4 outputs from this afternoon. Do not re-run BIND inference unless explicitly required.
- Six figures total. Each is one script. Each script writes a sidecar JSON with the numbers it plotted.
- No paper edits. Plots only.
- Wall-clock target: ≤4 hours. Most figures are plot-from-existing-arrays.
- If any of the figures returns a result that contradicts the Phase 5 verdict (e.g., BIND mean Jacobian on feedback is small, or DMO-vs-conditioning decomposition fully explains the sign flip), **STOP and write `outputs/scatter_diagnostics/CONVINCINGNESS_UPDATE.md`** rather than burying the result.

---

## 1  Figure 1 — "BIND IS sensitive to feedback" affirmative

**Purpose.** Demonstrate that BIND's *mean* response to feedback parameters is large, accurate, and matches truth — independent of whether the *scatter* Jacobian is contaminated.

**Plot.** A two-panel summary:

- **Panel (a):** Bar chart of BIND mean Jacobian $J_{\rm mean}(F_a \mid A_{\rm SN1})$ for all 12 observables, compared side-by-side with the truth 1P mean response $\Delta \log F_a / \Delta A_{\rm SN1}$ from the 1P_p3_* sims. Annotate each bar with the Pearson agreement.
- **Panel (b):** Same as (a), but for $A_{\rm AGN1}$ using 1P_p9_* (or whichever 1P arm matches A_AGN1). If A_AGN1's 1P arm has too few halos or wasn't run, use $A_{\rm SN2}$ instead.

**Output.**
- `figures/scatter_diagnostics/fig_bind_feedback_mean_response.pdf`
- `outputs/scatter_diagnostics/fig_bind_feedback_mean_response.json` (the numbers)

**Pass condition.** For mass/profile observables (M_gas, M_star, Sigma_gas_r*), BIND tracks truth on feedback mean response to within ±20% sign-correct. This figure proves the affirmative claim "BIND is sensitive to feedback."

---

## 2  Figure 2 — The R_j heatmap, properly annotated

**Purpose.** Make the parameter-class contamination pattern visually unambiguous and read in the right scientific language.

**Plot.** A 16-observable × 35-parameter heatmap of $R_j = |J_{\log\sigma_{\rm intra}}| / |J_{\log\sigma_{\rm inter}}|$. Colour map: white at $R_j = 0$, yellow at $R_j = 0.3$, orange at $R_j = 0.5$, deep red at $R_j \geq 1.0$.

- Group the 35 columns by parameter class (cosmology / SN / AGN / IMF / other), with separator lines between groups and a class label strip above the columns.
- Group the 16 observable rows by family (masses / shapes / profiles), with separator lines.
- Overlay text labels at each cell showing $R_j$ to 2 decimals, only for cells where $R_j > 0.3$ (uncluttered).
- Add a right-side panel: median $R_j$ per row across the four headline parameters (the autopilot's existing readout).

**Output.**
- `figures/scatter_diagnostics/fig_Rj_heatmap.pdf`
- `outputs/scatter_diagnostics/fig_Rj_heatmap.json`

**Pass condition.** The figure must show a *visible block structure*: cosmology columns hot for shape observables, feedback columns cool for masses/profiles. If the heatmap is uniformly hot or uniformly cool, the parameter-class interpretation is wrong and we need to revisit the Phase 1 result.

---

## 3  Figure 3 — Cosmology-in-DMO decomposition (the user's key concern)

**Purpose.** Directly test whether the apparent $\Omega_m$ "contamination" is a fixed-DMO conditioning-vector artefact, or a real total-derivative response that includes the DMO structural change.

**Method.** Compute three Jacobians for dq_DM and $f_b$ vs $\Omega_m$, on the same set of $\sim$200 halos:

- $J^{\rm cond}_{\sigma}$: **fixed-DMO FD.** Same halo patches, perturb the $\Omega_m$ entry of the conditioning vector. This is what the autopilot's Phase 1 measured.
- $J^{\rm DMO}_{\sigma}$: **DMO-only FD.** Use 1P-Omega_m-low and 1P-Omega_m-high *sim* DMO patches as input, but set the conditioning vector to the *fiducial* parameter values. This isolates the DMO-structure piece.
- $J^{\rm full}_{\sigma}$: **matched FD.** Use 1P-Omega_m-low and 1P-Omega_m-high sim DMO patches *and* set the conditioning vector to each sim's actual parameters. This is the total response, comparable to the 1P truth FD from Phase 4.

By chain rule (approximately): $J^{\rm full}_{\sigma} \approx J^{\rm cond}_{\sigma} + J^{\rm DMO}_{\sigma}$ (linearised).

**Plot.** For dq_DM and $f_b$ (panel rows) and for $\sigma_{\rm inter}$ and $\sigma_{\rm intra}$ (panel columns), a bar chart with four bars per panel: $J^{\rm cond}$, $J^{\rm DMO}$, $J^{\rm full}$ (the sum), and $J^{\rm truth}$ from Phase 4. Plus error bars.

**Output.**
- `figures/scatter_diagnostics/fig_cosmology_DMO_decomposition.pdf`
- `outputs/scatter_diagnostics/fig_cosmology_DMO_decomposition.json`

**Three diagnostic outcomes.** Document the result; do not pre-commit to interpretation.

| Outcome | Interpretation |
|---|---|
| $J^{\rm full} \approx J^{\rm truth}$ and $J^{\rm DMO} \gg J^{\rm cond}$ | BIND's partial response is fine; the cosmology "contamination" is actually a real DMO-structure response that's missing from the fixed-DMO FD. The sign-flip is a chain-rule artefact, not a model failure. **Headline-relevant: changes the framing significantly.** |
| $J^{\rm full} \approx J^{\rm truth}$ and $J^{\rm DMO} \sim J^{\rm cond}$ | DMO and conditioning-vector contributions are comparable; both matter for the total response. Contamination diagnostic still valid for the partial response. |
| $J^{\rm full} \ne J^{\rm truth}$ (linearisation fails) | The chain-rule decomposition doesn't close; nonlinearity is large. Means the FD is operating outside its linear regime. Different problem. |

**Pass condition.** Numbers are produced; interpretation follows from outcome.

---

## 4  Figure 4 — Truth scatter response for clean observable–parameter pairs

**Purpose.** The Phase 4 cross-check was only run for dq_DM at $\Omega_m$ and $\Omega_b$. The user wants to see whether the *clean* combinations (low R_j) have matching BIND vs truth scatter responses — i.e., whether the headline reframe candidates (f_b | A_SN1, Sigma_gas_r3 | A_SN1) hold up under truth.

**Plot.** Four panels, each a $\sigma_{\rm truth}$ vs parameter plot with BIND's $\sigma_{\rm inter}$ prediction overlaid:

- (a) $\sigma_{\rm truth}(f_b)$ vs $A_{\rm SN1}$ from 1P_p3_*.
- (b) $\sigma_{\rm truth}(\Sigma_{\rm gas, r3})$ vs $A_{\rm SN1}$ from 1P_p3_*.
- (c) $\sigma_{\rm truth}(M_{\rm gas})$ vs $A_{\rm SN1}$ — a sanity-check on a well-determined observable.
- (d) $\sigma_{\rm truth}({\rm dq\_DM})$ vs $A_{\rm SN1}$ — to see if the dq_DM case is salvageable for feedback even if it failed for cosmology.

For each panel: scatter points are truth at each 1P level with bootstrap CIs; overlaid line is BIND's predicted $\sigma_{\rm inter}$ from the FD Jacobian extrapolated linearly.

**Output.**
- `figures/scatter_diagnostics/fig_truth_scatter_vs_ASN1.pdf`
- `outputs/scatter_diagnostics/fig_truth_scatter_vs_ASN1.json`

**Pass condition.** For the clean combinations (panels a, b, c), BIND's predicted scatter trend matches truth in sign and approximate magnitude. If yes → the affirmative headline result is confirmed. If no → even the "clean" combinations have hidden issues and the paper needs deeper revision.

---

## 5  Figure 5 — Where the "mean orthogonal to scatter" framing IS supported

**Purpose.** The original paper concept ("scatter direction is orthogonal to mean direction in parameter space") might still be valid for clean observable–parameter classes. Show this where it holds.

**Plot.** For each of the 12 observables, plot the angle $\theta_{ab} = \arccos\left(\hat J_{\rm mean} \cdot \hat J_{\rm scatter}\right)$ in degrees, where the vectors are over the 35 parameters and normalised. Two versions:

- (a) Using all 35 parameters (raw angle).
- (b) Using only the parameters where $R_j < 0.3$ for that observable (decontaminated angle).

Observables sorted by decontaminated angle. Mark $90°$ (perfect orthogonality) as a reference line.

**Output.**
- `figures/scatter_diagnostics/fig_mean_vs_scatter_orthogonality.pdf`
- `outputs/scatter_diagnostics/fig_mean_vs_scatter_orthogonality.json`

**Pass condition.** At least one observable shows decontaminated angle near $90°$, supporting a "mean and scatter carry orthogonal information" framing for the publishable subset.

---

## 6  Figure 6 — Final headline-candidate comparison

**Purpose.** Side-by-side comparison of the three candidate headlines so the human can pick.

**Plot.** A four-row × four-column grid:

- **Rows** (the three candidates plus a baseline):
  1. dq_DM | $\Omega_m$ — original (retracted)
  2. $f_b$ | $A_{\rm SN1}$ — Option A
  3. $\Sigma_{\rm gas, r3}$ | $A_{\rm SN1}$ — Option B
  4. $M_{\rm gas}$ | $A_{\rm SN1}$ — baseline (well-calibrated, possibly below detection)

- **Columns:**
  1. $J_{\log\sigma_{\rm inter}}$ value with SE
  2. $R_j$ contamination ratio
  3. 1P truth $J$ value (if available from Phases 4 or new from Figure 4)
  4. Verdict colour-coded box: GREEN/YELLOW/RED with one-line reason

**Output.**
- `figures/scatter_diagnostics/fig_headline_candidates.pdf`
- `outputs/scatter_diagnostics/fig_headline_candidates.json`

**Pass condition.** Exactly one row is green; that becomes the headline.

---

## 6.5  Figure 7 — Per-parameter best-signal survey (added on user pushback)

**Purpose.** Address the question: *we have 35 parameters; we've been talking about 6. Are there clean publishable signals on the other 29 that we've been ignoring?*

**Method.** From the Phase 1 `phase1_intra_jacobian.npz`, for each of the 35 parameters $j$, find the observable $a^*(j)$ that maximises a publishability score:

$$\mathrm{score}(j, a) = \mathrm{SNR}(J_{\log\sigma_{\rm inter}, ja}) \times \mathbb{1}[R_{j,a} < 0.5]$$

Score is zero for contaminated combinations; otherwise it's the SNR of the inter-Jacobian. Rank the 35 parameters by their best-observable score.

**Plot.** Horizontal bar chart, 35 rows (one per parameter, sorted by score, descending). Each bar shows the score; bar colour is the R_j of the best observable (green R_j<0.3, yellow 0.3–0.5, grey >0.5 — these latter shouldn't appear by construction). Annotate each bar with the best observable name. Add a vertical line at SNR=2 (detection threshold).

**Output.**
- `figures/scatter_diagnostics/fig_parameter_signal_survey.pdf`
- `outputs/scatter_diagnostics/fig_parameter_signal_survey.json` — full ranking table

**Pass condition.** None — this is a discovery figure. Document the top 10 entries and call out any parameters beyond the headline 6 (Ω_m, Ω_b, A_SN1, A_SN2, A_AGN1, A_AGN2) that have score > 2.

Also produce a side-table: **for each of the 6 parameter classes** listed below, the median R_j and best-observable score:

```python
PARAMETER_CLASSES = {
    "cosmology": ["Omega_m", "Omega_b", "sigma_8", "h", "n_s"],
    "SN_amplitude": ["A_SN1", "A_SN2"],
    "SN_subgrid": ["IMFslope", "WindEnergyIn1e51erg", "WindFreeTravelDens",
                   "VariableWindVelFactor", "VariableWindSpecMomentum",
                   "MaxWindCoolingTime", "MinWindVel", ...],   # use repo's actual names
    "AGN_amplitude": ["A_AGN1", "A_AGN2"],
    "AGN_subgrid": ["BlackHoleFeedbackFactor", "BlackHoleEddingtonFactor",
                    "BlackHoleSeedMass", "ThermalFeedbackEnergyDensity",
                    "RadioModeFactor", "QuasarModeFactor", ...],
    "cooling_ISM": ["FactorEvaporation", "PhotoionisationThreshold",
                    "MaxSfrTimescale", ...],
}
```

(Use the repo's actual parameter naming — these are illustrative. The repo's `param_names` array from `phase1_intra_jacobian.npz` is authoritative.)

This side-table reveals whether the cosmology-vs-feedback dichotomy is real or whether sub-classes (e.g., AGN-subgrid) have their own R_j patterns.

---

## 6.6  Figure 8 — Parameter clustering in mean-Jacobian space (added on user pushback)

**Purpose.** Test whether the 35 parameters cluster into the physically expected classes (cosmology / SN-amplitude / SN-subgrid / AGN-amplitude / AGN-subgrid / cooling) when grouped by their *response signatures* across the 12 observables. If they cluster cleanly, the physical interpretation of the R_j pattern is well-supported. If they don't, the parameter-class framing is post-hoc.

**Method.** For each parameter $j$, take its 12-dimensional mean-Jacobian vector $\vec J_{\rm mean}(j) = (J_{{\rm mean}, ja})_{a=1}^{12}$. Normalise to unit length. Compute pairwise cosine similarity $S_{jk} = \hat J(j) \cdot \hat J(k)$ across all 35 × 35 pairs.

**Plot.** Two-panel figure:
- **Panel (a):** 35 × 35 similarity heatmap, parameter rows and columns ordered by hierarchical clustering on $1 - |S_{jk}|$. Annotate the dendrogram cut at $k=6$ clusters along the rows.
- **Panel (b):** The same heatmap but ordered by the physical parameter classes from §6.5. Side-by-side: do the clusters from (a) match the classes in (b)?

**Output.**
- `figures/scatter_diagnostics/fig_parameter_clustering.pdf`
- `outputs/scatter_diagnostics/fig_parameter_clustering.json` — cluster assignments, agreement metric

**Pass condition.** None — discovery figure. But compute and report the cluster-class agreement: the fraction of parameters whose data-driven cluster (a) matches the assigned physical class (b). High agreement → physical classes are real and the R_j framing generalises. Low agreement → parameter classes are convention not physics.

If the data-driven clustering produces a small "cosmology cluster" and a separate "feedback cluster" that further splits into sub-types, that confirms the Phase 1 cosmology-vs-feedback finding at the all-parameters level and gives the paper its parameter taxonomy.

If the clustering is incoherent (no clean block structure), the cosmology-vs-feedback framing is a post-hoc artefact of only looking at the 6 headline parameters.

---

## 7  Stop conditions

Halt and write `outputs/scatter_diagnostics/CONVINCINGNESS_UPDATE.md` if any of:

1. **Figure 1 fails its pass condition** — BIND's mean response on feedback is small or wrong-sign. Would indicate something is broken about the inference pipeline; not a paper-framing issue.
2. **Figure 3 outcome 1** — $J^{\rm cond} \ll J^{\rm DMO}$ for dq_DM | $\Omega_m$. Means the fixed-DMO FD was the wrong derivative for the cosmology response. This significantly changes the framing of the entire Phase 1 contamination conclusion. Must flag for human re-read before paper writing.
3. **Figure 4 fails for f_b | A_SN1** — truth doesn't show A_SN1-driven scatter response in f_b. Means Option A is no longer the recommended headline.
4. **Compute or repo failure** — required 1P sims not accessible, or BIND inference fails on 1P DMO patches (e.g., normalization mismatch).

---

## 8  File tree

```
bind/
└── scripts/
    ├── fig1_bind_feedback_mean.py
    ├── fig2_Rj_heatmap.py
    ├── fig3_cosmo_DMO_decomposition.py
    ├── fig4_truth_scatter_vs_ASN1.py
    ├── fig5_mean_vs_scatter_orthogonality.py
    ├── fig6_headline_candidates.py
    ├── fig7_parameter_signal_survey.py
    └── fig8_parameter_clustering.py

outputs/scatter_diagnostics/
├── fig_bind_feedback_mean_response.json
├── fig_Rj_heatmap.json
├── fig_cosmology_DMO_decomposition.json
├── fig_truth_scatter_vs_ASN1.json
├── fig_mean_vs_scatter_orthogonality.json
├── fig_headline_candidates.json
├── fig_parameter_signal_survey.json
├── fig_parameter_clustering.json
├── CONVINCINGNESS_REPORT.md          # written at end summarising all 8 figures
└── CONVINCINGNESS_UPDATE.md          # only if a stop condition fires

figures/scatter_diagnostics/
├── fig_bind_feedback_mean_response.pdf
├── fig_Rj_heatmap.pdf
├── fig_cosmology_DMO_decomposition.pdf
├── fig_truth_scatter_vs_ASN1.pdf
├── fig_mean_vs_scatter_orthogonality.pdf
├── fig_headline_candidates.pdf
├── fig_parameter_signal_survey.pdf
└── fig_parameter_clustering.pdf
```

---

## 9  Acceptance criteria

Done when:
1. All six figures exist and have non-placeholder content.
2. `CONVINCINGNESS_REPORT.md` summarises each figure's pass/fail/outcome with the actual numbers.
3. The report ends with a final headline recommendation (one row of Figure 6) and a single-sentence answer to each of the user's three concerns:
   - "BIND is sensitive to feedback because: [numbers from Fig 1]."
   - "The fixed-DMO FD captures [X%] of the cosmology response; the DMO-structure piece contributes [Y%]; the contamination conclusion [does / does not] depend on this distinction."
   - "Of the 35 parameters, [N] have clean publishable scatter signals (R_j < 0.5 with SNR > 2). They are: [list]. The parameter taxonomy from Figure 8 [does / does not] match the physical class structure."

The next human action will be to read the report, look at the figures, decide on the headline, and either commit to the reframed paper or escalate further.

---

## 10  One-line summary for the agent

Six figures that (1) prove BIND uses feedback, (2) decompose the cosmology response into conditioning-vector vs DMO-structure pieces, (3) verify the clean headline candidates against truth, and (4) produce a final go/no-go on the headline reframe. Reuse existing Phase 1/4 outputs; no new BIND inference except for Figure 3's three FD computations and Figure 4's 1P-truth observable extraction.
