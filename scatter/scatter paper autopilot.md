# BIND Scatter Paper — Diagnostic Autopilot Brief

**Audience.** A coding agent (Claude Sonnet 4.6 in autopilot) operating inside the BIND repository, continuing work begun this morning.

**Mission.** A preliminary scatter-paper analysis ran today and produced strong but fragile results. The headline observable (dq_DM) has known calibration and noise-floor issues that may contaminate the headline scatter-Jacobian signal. Execute the priority-ordered diagnostic suite from [[BIND Scatter Paper - Critical Review 2026-05-13]] to determine whether dq_DM survives as the publishable headline, or whether the paper needs reframing around a cleaner observable.

**Pre-existing context** (in repo from earlier today):
- `outputs/scatter_residual/` — morning's residual cross-correlation analysis (CV, K=10, N=1154 halos, 7 observables). Contains `matrices.npz`, `observables.parquet`, `residuals.parquet`.
- `paper_figures/scatter/` — afternoon's preliminary scatter-paper outputs (fig1_calibration, fig2_scatter_vs_mean, fig3_scatter_contours [running], fig4_inter_vs_intra).
- The afternoon's FD-Jacobian pipeline. Find it under whatever name the scripts use; do not regenerate it.

**Five phases, four hard escalation gates, ~6–9 hours of wall-clock end-to-end** (most of which is waiting on compute).

---

## 0  Operating principles

1. **Reuse first, regenerate last.** Morning's K=10 BIND samples and afternoon's FD Jacobian outputs are on disk. Extend them; do not re-run.
2. **Phase-by-phase with explicit gates.** Each phase writes its own report and acceptance criterion. Do not start phase $N+1$ until phase $N$'s gate passes or is intentionally skipped.
3. **Escalate on novel science.** If any of the 4 escalation triggers fire (§9), STOP and write `outputs/scatter_diagnostics/ESCALATION.md`. The human should look at the result before more compute is burned.
4. **Checkpoint incrementally.** Write a one-line progress entry to `outputs/scatter_diagnostics/PROGRESS.log` after every script completes. The human may interrupt mid-run; the next iteration should resume from PROGRESS.
5. **Conservative defaults.** Decisions left ambiguous in this brief follow the defaults table in §8. Do not improvise.
6. **No paper edits in this run.** This is diagnostic work. Paper revisions wait for human decision after Phase 5.

---

## 1  Scientific objective

Decide whether the preliminary scatter-paper headline — "$\Omega_m, \omega_b$ modulate dq_DM scatter at SNR ≈ 3.5–4.2, a cosmologically informative summary statistic invisible to mean-field analyses" — is supported by evidence, contaminated by model-noise modulation, or reflects a real signal that survives multiple cross-checks.

The headline depends on three load-bearing claims; each phase tests one:

| Claim | Phase that tests it | Pass criterion |
|---|---|---|
| BIND-truth scatter calibration is internally consistent | Phase 0 | $P_{aa}({\rm dq\_DM})$ reads sensibly from morning's matrices |
| The reported $J_{\log\sigma_{\rm inter}}$ is not contaminated by noise-floor modulation | Phase 1 (CORE) | $|J_{\log\sigma_{\rm intra}}|/|J_{\log\sigma_{\rm inter}}| < 0.3$ for the headline parameters on dq_DM |
| The scatter response is physical (truth shows the same signal) | Phase 4 | $\sigma({\rm dq\_DM})$ in 1P-truth shrinks at high $\Omega_m$ in the predicted direction |

Pass on Phases 1 and 4 → the paper can stand. Fail either → reframe.

---

## 2  Phase 0 — Consistency check and discovery (~30 min, no new compute)

### 2.1  Pull $P_{aa}({\rm dq\_DM})$ from morning's matrices
Open `outputs/scatter_residual/matrices.npz`. Extract the per-halo Pearson agreement diagonal `P_pair` if present, or recompute from `residuals.parquet` if not. Read off $P_{aa}$ for each of the 7 morning-suite observables. Save as `outputs/scatter_diagnostics/phase0_paa_diagonal.json`:

```json
{
  "P_aa": {"log10_M_DM": ..., "log10_M_gas": ..., ..., "q_DM": ..., "q_gas": ..., "q_star": ...},
  "mean": 0.828,
  "note": "dq_DM is not in the morning suite; this run uses q_DM (DM axis ratio). dq_DM is a *derived* quantity (q_DM − q_DM_DMO or similar); if morning suite has dq_DM, use it; otherwise note the gap and proceed."
}
```

**Important caveat.** The morning's analysis used 7 observables and may not include dq_DM (which is a *misalignment*, not an axis ratio). If dq_DM is missing from the morning suite, document this and proceed; the proxy observation is $P_{aa}(q_{\rm DM})$ and $P_{aa}(q_\star)$, which the morning analysis *does* have. If $\sigma_{\rm intra}({\rm dq\_DM})$ is large, expect $P_{aa}({\rm q\_DM})$ and $P_{aa}({\rm q\_star})$ to be noticeably below the 0.828 average.

### 2.2  Repository discovery
Locate and document in `outputs/scatter_diagnostics/PHASE0_REPO_MAP.md`:
- The afternoon FD-Jacobian script (the one that produced `paper_figures/scatter/fig2_scatter_vs_mean.pdf`). Likely named `scatter_jacobian.py` or `fd_scatter_jacobian.py` or similar.
- The data path where K=10 multi-sample BIND patches live (from morning brief: `outputs/scatter_residual/bind_samples/`).
- The CAMELS-IllustrisTNG 1P truth halo catalogues. 1P consists of 66 sims varying one parameter at a time across the prior range. We will need 1P halos at the extremes of $\Omega_m$ and $\omega_b$.
- The dq_DM computation utility (it's the misalignment observable; locate the function that computes it).

### 2.3  Gate 0
- `phase0_paa_diagonal.json` written.
- `PHASE0_REPO_MAP.md` written.
- The afternoon FD-Jacobian script is identified and its inputs/outputs documented.
- 1P truth halo data is locatable. If not — STOP and report (1P data should exist from earlier BIND validation; missing data is unexpected).

**Decision branch on $P_{aa}$ readout:**
- If $P_{aa}({\rm q\_DM}) \geq 0.7$ AND $P_{aa}({\rm q\_star}) \geq 0.7$: morning and afternoon analyses are *consistent* with high BIND determinism, contamination-via-noise-floor concern is still live but not yet contradicted. Proceed.
- If $P_{aa}({\rm q\_DM}) < 0.5$ or $P_{aa}({\rm q\_star}) < 0.5$: morning result already showed BIND is noise-dominated for shape observables; contamination concern is strongly supported. Proceed but flag heavily.
- If $P_{aa}({\rm q\_DM}) \geq 0.95$: morning result says BIND has near-perfect per-halo determinism even for axis ratios. This contradicts $\sigma_{\rm intra}>\sigma_{\rm inter}$ from this afternoon's Fig 4. **STOP — internal inconsistency; the human needs to look.**

---

## 3  Phase 1 — $J_{\log\sigma_{\rm intra}}$ Jacobian (CORE, 2–4 hours)

**This is the priority-1 diagnostic.** If $\sigma_{\rm intra}$ responds to $\boldsymbol\theta$, the afternoon's headline scatter Jacobian for dq_DM is contaminated by model-noise modulation.

### 3.1  What to compute

For every parameter $j \in \{1,\ldots,35\}$ and every observable $a$ in the afternoon suite (12 observables), compute via correlated-seed FD:

$$J_{\log\sigma_{\rm intra}, ja} = \frac{\log\sigma_{\rm intra,a}(\theta+\epsilon\hat e_j) - \log\sigma_{\rm intra,a}(\theta-\epsilon\hat e_j)}{2\epsilon}$$

where $\sigma_{\rm intra,a}(\theta)$ is the within-halo std of BIND output across $K$ noise seeds at fixed halo, then averaged over halos. Use exactly the same K seeds, halo set, and $\epsilon$ as the afternoon $J_{\log\sigma_{\rm inter}}$ computation, so the two Jacobians are directly comparable.

### 3.2  Implementation

Write `bind/analysis/scatter_intra_jacobian.py`. The function should reuse:
- The multi-sample BIND inference pipeline (already generates K=10 samples per halo at perturbed θ).
- The afternoon Jacobian script's parameter-perturbation infrastructure (read $\boldsymbol\theta_+$, $\boldsymbol\theta_-$, $\epsilon$ from its config).

Critical implementation detail: **$\sigma_{\rm intra}$ is computed across seeds at fixed halo, then aggregated across halos**, not the other way. Two reasonable aggregation choices:

- (preferred) Mean across halos: $\bar\sigma_{\rm intra}(\theta) = \frac{1}{N}\sum_h \sigma_{{\rm intra},h}(\theta)$.
- (alternative) Pooled variance: $\sigma_{\rm intra,pooled}^2 = \frac{1}{N}\sum_h \sigma_{{\rm intra},h}^2$, then square-root.

Use the *preferred* (mean across halos). Either way, document the choice in the output JSON.

### 3.3  Outputs

`outputs/scatter_diagnostics/phase1_intra_jacobian.npz`:
- `J_log_sigma_intra` (12 obs × 35 params) — the intra-Jacobian
- `J_log_sigma_inter` (12 obs × 35 params) — re-read from afternoon's output for direct comparison
- `contamination_ratio` = $|J_{\log\sigma_{\rm intra}}|/|J_{\log\sigma_{\rm inter}}|$ (12 × 35)
- `param_names`, `obs_names`

Plus a figure `figures/scatter_diagnostics/fig_intra_vs_inter_jacobian.pdf`:
- Same 3-panel layout as afternoon Fig 2 (M_gas, M_star, dq_DM).
- x-axis = $J_{\log\sigma_{\rm inter}}$ (afternoon scatter response).
- y-axis = $J_{\log\sigma_{\rm intra}}$ (this phase's noise-floor response).
- Points coloured by parameter family.
- Diagonal line $y = x$ as guide (would indicate "noise-floor entirely explains the inter response").
- Annotate Omega_m, omega_b, A_SN1, A_SN2 by name.

### 3.4  Gate 1 — the headline test

For dq_DM specifically, evaluate:

$$R_j = \frac{|J_{\log\sigma_{\rm intra}, j, {\rm dq\_DM}}|}{|J_{\log\sigma_{\rm inter}, j, {\rm dq\_DM}}|}$$

for $j \in \{\Omega_m, \omega_b, A_{\rm SN1}, A_{\rm SN2}\}$ (the four parameters with afternoon SNR>2 for dq_DM scatter).

| Outcome | Verdict | Action |
|---|---|---|
| $R_j < 0.3$ for all four headline parameters | Headline survives — $\sigma_{\rm intra}$ does not meaningfully respond to θ | Proceed to Phase 2; flag for paper that contamination check passed |
| $0.3 \leq R_j < 0.7$ for one or more | Partial contamination — $\sigma_{\rm intra}$ contributes but is not the dominant cause | Proceed to Phase 4 (1P truth) before deciding headline status. Write a note in `PROGRESS.log`. |
| $R_j \geq 0.7$ for one or more headline parameters | **Headline contaminated** — the apparent scatter signal in dq_DM is plausibly noise-floor modulation | **ESCALATE** (see §9). Write `outputs/scatter_diagnostics/ESCALATION.md` with the numbers. Stop further compute pending human review. |

Report all $R_j$ values regardless of outcome.

---

## 4  Phase 2 — Mass-binned scatter Jacobian (~1 hour, reuses data)

Persistent signal across mass bins is harder to fake than a single global $J_{\log\sigma_{\rm inter}}$ value.

### 4.1  What to compute

Re-compute $J_{\log\sigma_{\rm inter}}$ for the same 12 observables × 35 parameters, but now within each of the three mass bins from afternoon Fig 1:
- bin 1: $\log M_{200c} \in [13.0, 13.5]$
- bin 2: $\log M_{200c} \in [13.5, 14.0]$
- bin 3: $\log M_{200c} \in [14.0, 14.8]$

Reuse the existing afternoon Jacobian outputs — only the aggregation step changes (per-bin std instead of global std).

### 4.2  Outputs

`outputs/scatter_diagnostics/phase2_massbinned_jacobian.npz`:
- `J_log_sigma_inter_per_bin` (3 bins × 12 obs × 35 params)
- `n_halos_per_bin` (3,) — must each be ≥ 100 for statistics. If any bin has < 50 halos, drop it from analysis and document.

Figure `figures/scatter_diagnostics/fig_dqDM_massbinned_J.pdf`: dq_DM only, bar chart of $J_{\log\sigma_{\rm inter}}(\Omega_m), J_{\log\sigma_{\rm inter}}(\omega_b), J_{\log\sigma_{\rm inter}}(A_{\rm SN1}), J_{\log\sigma_{\rm inter}}(A_{\rm SN2})$ in each mass bin.

### 4.3  Gate 2

Pass: at least one of the four headline parameters has consistent sign across all three mass bins with SNR > 2 in at least two bins.

Fail: signs flip or signals are bin-specific (suggests bin-dependent artefact rather than physical effect).

Document either way. This is *evidence* not a stop condition — fail is a "weakens the headline" signal, not an escalation.

---

## 5  Phase 3 — K-sweep on focused subset (~3 hours compute, CONDITIONAL)

**Skip this phase entirely if Gate 1 passed cleanly** ($R_j < 0.3$ everywhere). It is only worth the compute if Phase 1 is in the partial-contamination band ($0.3 \leq R_j < 0.7$).

### 5.1  What to compute

For the CV-fiducial parameter point plus 2 perturbed θ (specifically $\Omega_m^\pm$ at $\pm\epsilon$):
- Generate K=40 BIND samples per halo (instead of K=10). On a focused subset of ~200 halos (random sample stratified by mass) to keep compute tractable.
- Recompute $\sigma_{\rm intra,a}$ and $\sigma_{\rm inter,a}$ for dq_DM, q_DM, q_gas, q_star.
- Recompute $J_{\log\sigma_{\rm inter}}$ and $J_{\log\sigma_{\rm intra}}$ for $\Omega_m$ on this K=40 dataset.

### 5.2  What this tests

If contamination from $\sigma_{\rm intra}/\sqrt K$ is the source of the apparent $J_{\log\sigma_{\rm inter}}$ signal, increasing $K$ from 10 to 40 should *reduce* the apparent $J_{\log\sigma_{\rm inter}}$ signal by ~2× (since the contamination term scales as $1/K$). If the apparent signal is unchanged across K, the signal is physical, not contamination.

### 5.3  Outputs

`outputs/scatter_diagnostics/phase3_K40_sweep.json`:
- $J_{\log\sigma_{\rm inter}}(\Omega_m, {\rm dq\_DM})$ at K=10 (re-read from afternoon) and K=40 (new).
- Ratio K=40 / K=10. Expected if physical: ~1.0. Expected if pure contamination: ~0.5 (since contamination $\propto 1/K$).

### 5.4  Gate 3

This is informational, not a stop condition. Document the ratio and let the synthesis phase weigh it.

---

## 6  Phase 4 — 1P truth cross-check (~1–2 hours)

**This is the strongest single test in the suite** — it sidesteps every BIND-internal noise-floor concern by going directly to hydro truth.

### 6.1  What to compute

CAMELS 1P consists of 66 simulations varying one parameter at a time. Pick the 1P sims with extreme $\Omega_m$ (the lowest and highest values in the 1P sweep). Compute $\sigma_{\rm inter,truth}({\rm dq\_DM})$ in each:

- 1P-Omega_m-low: $\sigma_{\rm phys,low}$ — truth halos at $\Omega_m$ near prior lower bound.
- CV-truth: $\sigma_{\rm phys,fid}$ — same as afternoon Fig 1 (fiducial).
- 1P-Omega_m-high: $\sigma_{\rm phys,high}$ — truth halos at $\Omega_m$ near prior upper bound.

Apply the same mass cut ($M_{200c} > 10^{13}\,M_\odot/h$) and aggregate across halos.

If 1P sims have only one realisation each (typical for CAMELS), $\sigma_{\rm phys}$ is the std of dq_DM across halos *within that sim*. With ~10 halos above the mass cut per sim, the std estimate is noisy — report bootstrap CI.

Repeat for $\omega_b$ extremes if 1P has them.

### 6.2  What this tests

BIND predicts $J_{\log\sigma_{\rm inter}}({\rm dq\_DM}, \Omega_m) = -0.864$ — i.e., at high $\Omega_m$, dq_DM scatter shrinks. Test in truth:
- Predicted: $\sigma_{\rm phys,high}/\sigma_{\rm phys,low} \approx \exp(-0.864 \cdot \Delta\Omega_m / 0.5)$ where $\Delta\Omega_m$ is the 1P extreme range in normalized space. For 1P-low to 1P-high spanning $\sim$$\pm 0.4$ in normalised, the predicted ratio is $\exp(-0.864 \cdot 0.8) \approx 0.50$ — high-$\Omega_m$ scatter should be ~half of low-$\Omega_m$ scatter.
- Observed: read off from truth.

### 6.3  Outputs

`outputs/scatter_diagnostics/phase4_1p_truth.json`:
- $\sigma_{\rm phys,truth}({\rm dq\_DM})$ at low / fid / high $\Omega_m$ and $\omega_b$, with bootstrap CIs.
- Predicted vs observed ratios.

Figure `figures/scatter_diagnostics/fig_dqDM_1ptruth.pdf`: scatter of $\sigma_{\rm phys}$ vs $\Omega_m$ (and vs $\omega_b$) in truth, overlay BIND's predicted curve.

### 6.4  Gate 4 — the strongest test

| Outcome | Verdict |
|---|---|
| Truth shows shrinkage at high $\Omega_m$ in the predicted direction with predicted-magnitude $\pm$ factor of 2 | **Headline corroborated** — independent of all BIND-internal concerns. The paper has a physical signal. |
| Truth shows no $\Omega_m$ dependence in dq_DM scatter (CI consistent with zero) | **Headline contradicted** — BIND's signal is internal artefact. **ESCALATE.** |
| Truth shows shrinkage but at wrong direction or wrong magnitude (e.g., grows at high $\Omega_m$, or shrinks 10× more strongly than predicted) | **Headline partially supported with caveats** — physical signal exists but BIND mis-quantifies it. Continue to Phase 5; do not escalate. |

---

## 7  Phase 5 — Synthesis and verdict (~1 hour)

### 7.1  What to write

`outputs/scatter_diagnostics/REPORT.md` — a 1–2 page synthesis with:
- A summary table of all four phase outcomes.
- The headline numbers: $R_j$ values from Phase 1, mass-bin consistency from Phase 2, K-sweep ratio from Phase 3 (if run), truth signal from Phase 4.
- A verdict on each of the three load-bearing claims (§1) — PASS / PARTIAL / FAIL.
- A recommendation in one of three buckets:

| Recommendation | Trigger |
|---|---|
| **Headline survives** — dq_DM remains the headline; paper publishable as drafted with strengthened calibration framing | Phase 1 PASS, Phase 4 PASS |
| **Reframe needed** — dq_DM moves to a supplementary "even for this hard observable" section; a cleaner observable (Sigma_gas_r0 is the prime candidate per the critical review) becomes the headline | Phase 1 PARTIAL, Phase 4 PARTIAL, or Phase 1 PARTIAL + Phase 4 PASS |
| **Headline retracted** — dq_DM is removed from the headline; paper scope changes substantially | Phase 1 FAIL, or Phase 4 FAIL |

### 7.2  Also write

`outputs/scatter_diagnostics/section_paragraph_calibration.md` — a short paragraph for the paper's methods/validation section honestly characterising the calibration of dq_DM (the 25–43% inter under-reproduction; the $\sigma_{\rm intra}>\sigma_{\rm inter}$ at fiducial). The phrase "almost all bars agree to within ~5–9%" should be corrected with specific numbers for dq_DM.

### 7.3  Gate 5
- REPORT.md exists with verdict.
- Calibration paragraph exists.
- All upstream phase outputs are linked.

---

## 8  Decision defaults (do not deliberate)

| Decision | Default |
|---|---|
| Aggregation of $\sigma_{\rm intra}$ across halos | Mean across halos (not pooled-variance) |
| FD step $\epsilon$ | Whatever the afternoon Jacobian used (read from its config) |
| K for Phase 1 | Reuse K=10 from morning. Do not regenerate. |
| K for Phase 3 (if run) | K=40 on 200 stratified halos at fiducial + $\Omega_m^\pm$ only |
| Mass bin edges | $[13.0, 13.5, 14.0, 14.8]$ as in afternoon Fig 1 |
| 1P truth observable | dq_DM directly. Apply the same mass cut $M_{200c} > 10^{13}\,M_\odot/h$. |
| Bootstrap $B$ | 2000 |
| Whether to run Phase 3 | Only if Phase 1 falls in $0.3 \leq R_j < 0.7$ partial-contamination band |
| Headline parameters | $\Omega_m, \omega_b, A_{\rm SN1}, A_{\rm SN2}$ |
| Paper edits | **None in this run.** Diagnostic only. |

If a default conflicts with a hard-coded value in the BIND repo (e.g., a different $\epsilon$ for FD), use the repo's value and note the discrepancy in REPORT.md.

---

## 9  Stop conditions and escalation

Write `outputs/scatter_diagnostics/ESCALATION.md` and **halt all further compute** if any of these occur:

1. **Phase 0 inconsistency.** $P_{aa}({\rm q\_DM}) \geq 0.95$ — contradicts the afternoon's $\sigma_{\rm intra}>\sigma_{\rm inter}$ finding. Both analyses cannot be right.
2. **Phase 1 contamination.** $R_j \geq 0.7$ for any headline parameter on dq_DM — the noise-floor explains most of the apparent signal.
3. **Phase 1 unexpected sign.** $J_{\log\sigma_{\rm intra}}$ is *positively correlated with* $J_{\log\sigma_{\rm inter}}$ across the parameters — suggests a deeper coupling between BIND noise and the response surface.
4. **Phase 4 truth contradiction.** 1P-truth shows no $\Omega_m$ dependence in $\sigma_{\rm dq\_DM}$ (CI consistent with zero) — BIND's signal is internal artefact.
5. **Compute or repo failure.** Any required artefact missing; multi-sample inference fails; wall-clock projection > 12 hours.
6. **Surprise.** Anything the agent encounters that is not anticipated by this brief and changes the scientific interpretation.

ESCALATION.md must contain: what triggered, the relevant numbers, the suspected interpretation, and what the human needs to look at.

---

## 10  File tree

```
bind/
├── analysis/
│   └── scatter_intra_jacobian.py       # §3 — new
├── scripts/
│   ├── phase0_consistency_check.py     # §2 — new
│   ├── phase1_intra_jacobian_run.py    # §3 — new
│   ├── phase2_massbinned_jacobian.py   # §4 — new
│   ├── phase3_K40_sweep.py             # §5 — new, CONDITIONAL
│   ├── phase4_1p_truth_check.py        # §6 — new
│   └── phase5_synthesise.py            # §7 — new

outputs/scatter_diagnostics/
├── PROGRESS.log                          # checkpointing
├── PHASE0_REPO_MAP.md                    # §2
├── phase0_paa_diagonal.json              # §2
├── phase1_intra_jacobian.npz             # §3
├── phase2_massbinned_jacobian.npz        # §4
├── phase3_K40_sweep.json                 # §5 (if run)
├── phase4_1p_truth.json                  # §6
├── REPORT.md                             # §7
├── section_paragraph_calibration.md      # §7
└── ESCALATION.md                         # only if a stop condition fires

figures/scatter_diagnostics/
├── fig_intra_vs_inter_jacobian.pdf       # §3
├── fig_dqDM_massbinned_J.pdf             # §4
└── fig_dqDM_1ptruth.pdf                  # §6
```

---

## 11  Reference: the science context (one paragraph)

The BIND model is a flow-matching generative model that paints baryonic fields onto DM-only halos. A preliminary paper analysis ran today claiming that cosmological parameters ($\Omega_m, \omega_b$) modulate the inter-halo scatter of baryon–DM misalignment (dq_DM) at SNR ≈ 3.5–4.2, even though they have no measurable effect on the mean misalignment. The signal is striking — but dq_DM is also the worst-calibrated observable in the suite (inter-halo std 25–43% low vs CV truth) and the only observable where BIND's intra-halo model noise exceeds its inter-halo physical scatter at fiducial. The correlated-seed finite-difference scheme used for the Jacobian protects against per-seed noise in differences of means, but **does not** protect against $\theta$-dependent modulation of the noise variance itself. The job of this diagnostic suite is to determine whether the dq_DM scatter signal is real physical response, or whether it is BIND's noise floor changing with $\theta$. The cleanest external test is Phase 4 (1P truth at extreme $\Omega_m$), which goes around BIND entirely.

External anchors (do not require reading):
- [Genel et al. 2019, arXiv:1807.07084](https://arxiv.org/abs/1807.07084) — butterfly amplitudes ~2–25% in galaxy properties at fixed IC; sets the scale for irreducible stochasticity.
- [Farahi & Evrard 2018, arXiv:1711.04922](https://arxiv.org/abs/1711.04922) — multivariate lognormal joint $(M_\star, M_{\rm gas} | M_h)$; not directly applicable to 2D-projected analyses but the framework anchor for residual covariance.

---

## 12  Acceptance criteria

The task is complete when:
1. Gates 0, 1, 2 have either passed or hit an escalation trigger.
2. Phase 4 has run unless an escalation fired earlier.
3. Phase 5 REPORT.md exists with a verdict in one of the three buckets (headline survives / reframe / retract).
4. All output files in §10 exist (with the noted conditionality for Phase 3).
5. PROGRESS.log contains a chronological record of every script that ran.
6. If escalation fired, ESCALATION.md exists with sufficient context for a human to decide what to do.

The next human action after completion will be: read REPORT.md, look at the figures, and decide (a) which observable becomes the paper's headline, and (b) whether the paper needs reframing or can stand.
