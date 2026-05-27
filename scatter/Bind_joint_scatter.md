---
title: "BIND Scatter Joint Structure — Autopilot Brief (2026-05-13 night)"
tags: [BIND, Active, Implementation, AgentBrief]
project: BIND
created: 2026-05-13
status: ready_to_execute
related: [BIND Scatter - Reframe to Joint Structure 2026-05-13, BIND Scatter-Residual Implementation Brief, BIND Scatter Paper - Convincingness Synthesis 2026-05-13]
---

# BIND Scatter Joint Structure — Autopilot Brief

**Audience.** A coding agent (Claude Sonnet 4.6 in autopilot) operating inside the BIND repository, with full familiarity from today's prior runs (morning's `outputs/scatter_residual/` and afternoon's `outputs/scatter_diagnostics/`).

**Mission.** Test whether BIND reproduces the *parameter dependence* of cross-observable residual correlations. Operationalises the reframe in [[BIND Scatter - Reframe to Joint Structure 2026-05-13]]: ask "does the residual correlation matrix $C_{ab}(\theta)$ shift with parameters in truth, and does BIND track the shift" rather than "does $\sigma_{\rm inter}(\theta)$ respond cleanly to parameters." This bypasses the $R_j$ contamination issue and connects to mainstream SHMR-scatter physics.

**Structure.** Pilot phase first (1 parameter, 2 endpoints, single gate). Expand only if pilot passes. Reuse all morning + afternoon code.

**Wall-clock target.** Pilot: ≤4 hours including 30 minutes compute. Full sweep: ≤1 day.

---

## 0  Operating principles

1. **Reuse aggressively.** The morning's `bind/analysis/observables.py`, `bind/analysis/scatter_residual.py` (LOWESS, residuals, correlation matrices), and `bind/scripts/multisample_inference.py` are all directly applicable. Only the parameter-point loop and the comparison code are new.
2. **Pilot before sweep.** Phase 1 tests a single parameter (A_SN1) with one gate. If the gate fails, STOP and report — do not proceed to the full 12-sim sweep.
3. **One figure, one decision.** Each phase ends with a clear go/no-go on a single quantitative criterion.
4. **No paper edits in this run.** Diagnostic analysis only. Paper §4 revisions wait for the full-sweep results and human review.
5. **Checkpoint everything.** Write progress to `outputs/scatter_joint_structure/PROGRESS.log` after each script.

---

## 1  Scientific objective

The morning's residual cross-correlation analysis ([[BIND Scatter-Residual Results 2026-05-13]]) established at the **fiducial CV parameter point** that:
- BIND's per-halo agreement averages $\langle P_{aa}\rangle = +0.828$
- $\|C^T_{\rm fid} - C^G_{\rm fid}\|_F = 0.654$, leading-eigenvector angle $7.1°$
- BIND captures joint physics at fiducial

This brief tests whether the same correlation-structure reproduction holds **as parameters vary**. The metric of interest is the parameter-dependent residual correlation matrix:
$$C_{ab}(\theta) = \rho_{\rm Spearman}(\hat\Delta_a(\theta), \hat\Delta_b(\theta))$$
where $\hat\Delta_a(\theta)$ is the standardised residual computed against the LOWESS mean fit at parameter $\theta$.

The question we are answering: **Does the shift $C^T_{ab}(\theta_+) - C^T_{ab}(\theta_-)$ match the shift $C^G_{ab}(\theta_+) - C^G_{ab}(\theta_-)$?**

If yes: BIND captures parameter-dependent joint physics, not just fiducial joint physics. This is the §4.3 headline finding for the paper.

If no: BIND captures fiducial joint physics but not parameter-dependent joint physics. Still a publishable bounded statement, but the §4.3 framing has to be narrower.

---

## 2  Fixed observable set (no deliberation)

Use exactly the morning's 7 observables (drop $f_b$ from primary, include in supplementary):
```
observables_primary = [
    "log10_M_DM", "log10_M_gas", "log10_M_star",
    "log10_Sigma_gas_c",
    "q_DM", "q_gas", "q_star",
]
```
Same as the morning's brief. The supplementary 8th observable ($f_b$) is included only in the supplementary 8×8 matrix.

---

## 3  Phase 0 — Discovery (~15 minutes)

Verify three things and record in `outputs/scatter_joint_structure/PHASE0.md`:

1. **Locate the 1P-p3 sims**: confirm 1P_p3_n2 and 1P_p3_2 are available with hydro patches. (We know from the user that hydro patches DO vary across 1P sims — the convincingness diagnostic's data-gap claim was wrong.)
2. **Halo counts**: how many halos with $M_{200c} > 10^{13}\,M_\odot/h$ in each of 1P_p3_n2 and 1P_p3_2. Expected ~10 per sim (small box, narrow mass range). Hard floor: ≥ 5 per sim; if fewer, STOP.
3. **BIND inference path for 1P sims**: confirm the inference pipeline accepts 1P DMO patches with the corresponding 1P parameter vectors. The afternoon's convincingness diagnostic ran this for the dq_DM truth check; reuse that infrastructure.

---

## 4  Phase 1 — Pilot on A_SN1 (~3 hours)

### 4.1  Generate BIND samples at the two endpoints

For each of {1P_p3_n2, 1P_p3_2}:
- Load DMO patches for halos with $M_{200c} > 10^{13}\,M_\odot/h$
- Set the conditioning $\boldsymbol\theta$ to the actual 1P-p3 parameter values for that sim (i.e., feed BIND the same θ that the hydro sim was run with)
- Generate $K = 10$ BIND samples per halo with seeds `seed = hash((sim_id, halo_id, sample_id)) & 0xFFFFFFFF`
- Save to `outputs/scatter_joint_structure/bind_samples/<sim_name>/halo<id>/sample<k>.npz`

### 4.2  Compute observables and residuals

For both 1P sims (truth) and BIND samples:
- Run `compute_observables` from `bind/analysis/observables.py` to extract the 7 primary observables.
- For *each parameter endpoint separately*, fit a LOWESS mean $\hat\mu_a(\log M_h)$ on the truth halos at that endpoint (do NOT pool across endpoints — the mean relation itself can shift with $\theta$, and we want residuals against the *local* mean).
- Same for the BIND output at that endpoint.
- Compute standardised residuals $\hat\Delta_a$ for both truth and BIND.
- Save residuals to `outputs/scatter_joint_structure/pilot_residuals_p3_n2.parquet` and `..._p3_2.parquet`.

**Important decision**: the LOWESS mean is fit *per endpoint* on truth, and a separate one is fit on BIND. This isolates correlation shifts from mean shifts. If we used a global mean fit pooled over endpoints + truth + BIND, mean drift would leak into the residual correlations.

### 4.3  Compute correlation matrices

For each endpoint, compute the 7×7 Spearman residual correlation matrix for truth and BIND, plus bootstrap standard errors ($B = 2000$):
- $C^T(\theta_+), C^T(\theta_-)$ from 1P_p3_2 and 1P_p3_n2 truth
- $C^G(\theta_+), C^G(\theta_-)$ from BIND outputs at those endpoints

Compute the parameter-dependent shifts:
- $\Delta C^T = C^T(\theta_+) - C^T(\theta_-)$
- $\Delta C^G = C^G(\theta_+) - C^G(\theta_-)$
- Per-entry bootstrap SE for each

Save to `outputs/scatter_joint_structure/pilot_matrices.npz`.

### 4.4  Pilot figure

`figures/scatter_joint_structure/fig_pilot_ASN1.pdf` — three panels:

- **Panel (a)**: ρ(ΔM_*, ΔM_gas) at three points (1P_p3_n2, CV fiducial from morning's analysis, 1P_p3_2). Truth (black) and BIND (red) lines with bootstrap CI bands. Linked across the x-axis (parameter level).
- **Panel (b)**: ρ(ΔM_*, Δq_DM) same format.
- **Panel (c)**: ρ(ΔM_gas, Δq_DM) same format.

Below the three panels: a small bar chart of all 21 unique off-diagonal $\Delta C^T_{ab}$ entries (sorted by magnitude) with BIND $\Delta C^G_{ab}$ overlaid — gives a one-glance summary of where truth shifts and how well BIND tracks.

### 4.5  Pilot gate

`outputs/scatter_joint_structure/pilot_gate.json` reports:
- $N$ entries of $\Delta C^T$ with bootstrap-significant shift ($|z| > 2$).
- For those entries, the median fractional BIND tracking ($\Delta C^G / \Delta C^T$).
- Sign-agreement count.

**Pass criteria** (all required):
1. At least one $\Delta C^T$ entry has $|z| > 2$. *Interpretation*: truth correlation structure does shift with A_SN1.
2. For at least one bootstrap-significant entry, $|\Delta C^G / \Delta C^T| \geq 0.5$ AND $\mathrm{sign}(\Delta C^G) = \mathrm{sign}(\Delta C^T)$. *Interpretation*: BIND captures at least half the magnitude of at least one shift with the correct sign.

If pass: proceed to Phase 2 (full sweep).

If fail on criterion 1: truth correlation structure doesn't shift with A_SN1 in this sample size. Either A_SN1 doesn't move the joint structure at these mass scales (interesting null), or the 1P-p3 endpoints are too narrow / halo counts too low to detect a shift. Write `outputs/scatter_joint_structure/PILOT_NULL.md` and STOP; human decides whether to widen the endpoints or move to a different parameter.

If fail on criterion 2: truth structure shifts but BIND doesn't track it. **STOP and ESCALATE** — this is a meaningful negative result that changes the paper's claim. Write `outputs/scatter_joint_structure/PILOT_FAIL.md` with the numbers; human review required before further compute.

---

## 5  Phase 2 — Full 6-parameter sweep (~6–8 hours, CONDITIONAL on Phase 1 pass)

Repeat Phase 1's pipeline (§4.1–4.3) for the six standard 1P parameter arms:
| Param | Arm | Low endpoint | High endpoint |
|---|---|---|---|
| $\Omega_m$ | 1P_p1 | n2 | 2 |
| $\sigma_8$ | 1P_p2 | n2 | 2 |
| $A_{\rm SN1}$ | 1P_p3 | n2 (DONE in pilot) | 2 (DONE in pilot) |
| $A_{\rm AGN1}$ | 1P_p4 | n2 | 2 |
| $A_{\rm SN2}$ | 1P_p5 | n2 | 2 |
| $A_{\rm AGN2}$ | 1P_p6 | n2 | 2 |

For each parameter:
- Generate BIND samples at both endpoints (reuse the pilot output for A_SN1).
- Compute $C^T, C^G$ at each endpoint with per-endpoint LOWESS fit.
- Compute $\Delta C^T_j, \Delta C^G_j$ for the parameter.

Save to `outputs/scatter_joint_structure/sweep_matrices.npz`.

### 5.1  Headline figure

`figures/scatter_joint_structure/fig_sweep_headline.pdf` — the headline figure of §4.3:

- **Layout**: 3 rows × 6 columns of panels.
- Rows: three observable pairs — (M_*, M_gas), (M_*, q_DM), (M_gas, q_DM).
- Columns: six 1P parameters.
- Each panel: scatter plot, x = $\Delta C^T_{ab}$ for that (param, observable-pair), y = $\Delta C^G_{ab}$, one point per panel with bootstrap CIs on both axes. Diagonal $y = x$ overlaid. Cosmology columns coloured blue, feedback columns coloured red.

Plus a single summary panel below the grid: scatter of all $3 \times 6 = 18$ points on a single x-vs-y plot, fit a regression line, report $R^2$. **One number** that says how well BIND tracks the truth's parameter-dependent joint structure.

### 5.2  Secondary figure

`figures/scatter_joint_structure/fig_sweep_all_pairs.pdf` — the full $\binom{7}{2}=21$ pairs × 6 parameters scatter. Reveals which observable pairs BIND tracks well and which it misses. Auxiliary to the headline.

### 5.3  Report

`outputs/scatter_joint_structure/REPORT.md` — synthesis with:
- Headline number: $R^2$ from the summary panel of fig_sweep_headline.
- Per-parameter table: how many entries shift in truth (criterion 1), how many BIND tracks (criterion 2), sign agreement rate.
- Per-observable-pair table: same.
- Comparison to morning's fiducial result: does the parameter-dependent tracking match the fiducial $P_{aa}$ tracking, or is there degradation off-fiducial?
- Verdict for §4.3 of the paper: *Strongly supported* / *Partially supported* / *Bounded-failure statement required*.

---

## 6  Decision defaults (no deliberation)

| Decision | Default |
|---|---|
| Pilot parameter | A_SN1 (1P_p3) |
| Pilot endpoints | 1P_p3_n2 (low) and 1P_p3_2 (high) |
| BIND samples per halo | $K = 10$ |
| Mass cut | $M_{200c} > 10^{13}\,M_\odot/h$ |
| Mean fit | LOWESS per endpoint per source (truth and BIND separately) — preserves parameter-dependent mean shifts as a *separate* signal, not absorbed into residuals |
| Correlation method | Spearman |
| Bootstrap $B$ | 2000 |
| Primary observable set | 7 (drop $f_b$ from primary) |
| Headline observable pairs for §4.4 | (M_*, M_gas), (M_*, q_DM), (M_gas, q_DM) |
| Comparison metric | $\Delta C^G / \Delta C^T$ with sign agreement |

---

## 7  Stop conditions

Write `outputs/scatter_joint_structure/STOP_REPORT.md` and halt if any:

1. **Phase 0 fails**: 1P-p3 hydro patches missing or halo counts below floor.
2. **Phase 1 criterion 1 fails**: truth correlation structure doesn't shift with A_SN1 → null result; human decides whether to widen endpoints or change parameter.
3. **Phase 1 criterion 2 fails**: truth shifts but BIND doesn't track → ESCALATE; this changes the paper's §4.3 claim.
4. **Compute or repo failure**: any required data missing; BIND inference fails on 1P DMO patches with non-fiducial θ; wall-clock projection > 12 hours.
5. **Surprise**: a result not anticipated by this brief that changes the paper's framing.

---

## 8  File tree

```
bind/
└── scripts/
    ├── joint_struct_pilot.py            # §4 — pilot inference + matrices + figure
    └── joint_struct_sweep.py            # §5 — full 6-param sweep + headline figure

outputs/scatter_joint_structure/
├── PROGRESS.log
├── PHASE0.md
├── bind_samples/                        # BIND outputs at 1P parameter points
├── pilot_residuals_p3_n2.parquet
├── pilot_residuals_p3_2.parquet
├── pilot_matrices.npz
├── pilot_gate.json
├── sweep_matrices.npz                   # Phase 2
├── REPORT.md                            # Phase 2 final synthesis
├── PILOT_NULL.md  or  PILOT_FAIL.md     # only if pilot fails
└── STOP_REPORT.md                       # only if stop condition fires

figures/scatter_joint_structure/
├── fig_pilot_ASN1.pdf                   # Phase 1
├── fig_sweep_headline.pdf               # Phase 2 — the §4.3 headline figure
└── fig_sweep_all_pairs.pdf              # Phase 2 — auxiliary
```

---

## 9  Reference: science context (one paragraph)

The BIND model paints baryonic fields onto DMO halos. The morning's analysis showed that at fiducial CAMELS parameters, BIND reproduces the cross-observable residual correlation matrix between baryonic properties (per-halo Pearson agreement $P_{aa} = 0.83$; leading-eigenvector angle $7.1°$; Frobenius distance $0.654$). The afternoon's complementary analysis tried to measure the parameter dependence of $\sigma_{\rm inter}$ directly but ran into noise-floor modulation that contaminates the response for shape observables on cosmological parameters. This brief takes the cleaner path: measure how the morning's residual *correlation matrix* shifts with parameters in truth, then ask whether BIND tracks the shift. Because correlations are scale-invariant rank statistics, the noise-floor contamination issue is structurally bypassed. The headline result is the §4.3 figure of the BIND paper: a 3-by-6 grid showing parameter-dependent joint physics tracking across the standard 6 CAMELS 1P arms.

External literature anchors (already in vault):
- [Wechsler & Tinker 2018 ARA&A](https://arxiv.org/abs/1804.03097) — galaxy-halo scatter physics.
- [Behroozi et al. 2019 UniverseMachine](https://arxiv.org/abs/1806.07893) — assembly-bias scatter decomposition.
- [Farahi & Evrard 2018, arXiv:1711.04922](https://arxiv.org/abs/1711.04922) — log-normal joint $(M_\star, M_{\rm gas} | M_h)$ in BAHAMAS/MACSIS; the literature anchor for residual correlation analysis.

---

## 10  Acceptance criteria

Done when:
1. Pilot gate passed AND Phase 2 ran AND REPORT.md exists with a verdict, OR
2. Pilot gate failed AND PILOT_NULL.md or PILOT_FAIL.md exists with sufficient context for human review.
3. All artefact files in §8 exist (with the conditionality noted).
4. PROGRESS.log contains a chronological record of every script that ran.

The next human action after completion will be: read REPORT.md (or the pilot-fail report), look at the headline figure, decide whether §4.3 of the paper goes in as "joint structure tracks across parameter space" or as a bounded statement, and revise the committee meeting deck accordingly.

---

## 11  One-line summary for the agent

Pilot the parameter-dependent residual cross-correlation matrix on A_SN1 (1P_p3 endpoints, K=10 BIND samples, per-endpoint LOWESS); if truth correlation structure shifts and BIND tracks at least half the shift with correct sign, extend to the 6-parameter sweep and produce the §4.3 headline figure ($3 \times 6$ grid + a single summary scatter with $R^2$). Reuse all morning + afternoon code; no paper edits.
