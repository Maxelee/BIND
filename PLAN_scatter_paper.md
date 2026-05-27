# BIND Scatter Paper — Executable Plan

**Target executor:** Sonnet 4.6.
**Working directory:** `/mnt/home/mlee1/vdm_bind2`.
**Estimated wall time:** 2–4 days of GPU + CPU work, plus writing.

---

## 0. Science question and headline claim

**Question.** Does the halo-to-halo *scatter* in cluster-scale baryonic observables encode information about feedback physics beyond what the *mean* relations contain, and how does that scatter depend on the IllustrisTNG subgrid parameters?

**Why BIND is the only tool that can answer it.** Deterministic BCMs have zero scatter by construction. Hydro suites have scatter at only one parameter point (CV) or one realization per parameter (SB35). Only a parameter-conditioned *generative* model can map $\sigma_{\rm intrinsic}(\theta)$ across the 35-D space.

**Headline claim to prove or refute.** Feedback parameters that primarily move the *mean* of $\log M_{\rm gas}(<R_{200}) \mid M_{200}$ are different from feedback parameters that primarily move the *scatter*. The scatter direction in parameter space is approximately orthogonal to the mean direction, and exploiting it could break degeneracies that are inaccessible to deterministic baryonification.

**Fallback claims.** If scatter is parameter-independent: that is itself a publishable validation of the standard cluster-cosmology assumption. If BIND's posterior is uncalibrated against CV ground truth: the paper becomes a posterior-calibration methods paper that fixes the issue. Both are publishable.

---

## 1. Environment

- Python env: same as `fd_jacobian_cv.py` uses (PyTorch + Lightning, h5py, scipy).
- Model: `/mnt/home/mlee1/ceph/fm_runs/fm_two_head/checkpoints/last.ckpt` (loaded via `train.FlowMatchingLit.load_from_checkpoint`).
- Norm stats: `/mnt/home/mlee1/ceph/fm_runs/fm_two_head/norm_stats.npz`.
- CV data: `/mnt/home/mlee1/ceph/fm_testsuite/CV` (27 sims at fiducial cosmology + astro).
- SB35 data root (for off-fiducial scatter measurements where available): same convention; check `data.py` for the exact loader.
- GPU: assume 1× A100 (use the same `device = "cuda"` pattern as `fd_jacobian_cv.py:259`).
- Reuse `observables_from_phys` at `fd_jacobian_cv.py:151` for per-halo observables. Do **not** reinvent.

**LAW reminder.** Never recursively search `/mnt/home` or `/mnt/ceph` at their roots — always scope to `/mnt/home/mlee1/...` or `/mnt/ceph/users/mlee1/...`.

---

## 2. Observable set (final)

Use only observables that survive the LOS-projection critique. **Drop** `f_b`, `f_b_norm`, `R_cl/R_200`, `Sigma_gas_c` from the headline figures (they are LOS-contaminated). Use them only in an appendix robustness check.

Headline observables, all from `observables_from_phys`:

| Symbol | Description | Why it survives projection |
|---|---|---|
| $M_{\rm DM}(<R_{200})$ | projected DM in aperture | dominated by halo at cluster mass |
| $M_{\rm gas}(<R_{200})$ | projected gas in aperture | parameter-response dominated by halo |
| $M_\star(<R_{200})$ | projected stars | compact, low background |
| $\Delta q_{\rm DM}$ | $q_{\rm DM}^{\rm hydro} - q_{\rm DM}^{\rm DMO}$ | differencing cancels background |
| $q_{\rm gas}$ | gas axis ratio | projection-native lensing observable |
| $q_\star$ | stellar axis ratio | projection-native lensing observable |
| $\Sigma(r)$ at 5 log-spaced radii | radial profile | projection-native |

That gives ~11 scalar observables per halo (3 masses, 3 shapes, $\Delta q_{\rm DM}$, 5 profile bins, but adjust as the radial bin counts dictate).

---

## 3. Phases

Each phase has: **goal**, **inputs**, **what to write/run**, **artifacts**, **gating check**. Do not proceed past a phase whose gating check fails — file a note in `PLAN_NOTES.md` and stop for human review.

### Phase 0 — Reconnaissance (1 hr)

**Goal.** Verify the existing infrastructure works and confirm paths.

**Actions.**
1. `cat fd_jacobian_cv.py | head -270` — confirm the model-loading and observable code is intact.
2. `python -c "from train import FlowMatchingLit; print('ok')"` — environment sanity.
3. List CV sim dirs: `ls /mnt/home/mlee1/ceph/fm_testsuite/CV | head` (do not recurse).
4. Inspect one CV halo file structure: read one `.h5` to confirm field layout.

**Artifacts.** None, but record what you found in `PLAN_NOTES.md`.

**Gating check.** Model loads, CV path resolves, halo files have expected shape `(C, 128, 128)`.

---

### Phase 1 — Build the scatter measurement engine (4–8 hr)

**Goal.** A reusable script that, given (param vector $\theta$, set of DMO halo conditionings $\{c_h\}$, $K$ samples per halo), returns a tensor of per-sample observables and a clean variance decomposition.

**File to create.** `scatter/measure_scatter.py`.

**Interface** (specify, do not deviate):
```python
def measure_scatter(
    model_fm, norm_stats, theta_norm, dmo_conds, ls_conds,
    masses, r200_pix, K=10, n_steps=20, device="cuda", batch_size=32,
) -> dict:
    """
    Returns:
        obs_tensor:  (N_h, K, N_obs) float32 — denormalized observables
        obs_names:   list[str] of length N_obs
        masses:      (N_h,) — pass-through for binning
        sigma_inter: (N_obs,) — std across halos of the per-halo mean over K
        sigma_intra: (N_obs,) — mean across halos of the per-halo std over K
        sigma_total: (N_obs,) — total std across the full (N_h * K) ensemble
    """
```

**Implementation notes.**
- Reuse `_sample_fixed_noise` and `observables_from_phys` from `fd_jacobian_cv.py`. Do not reinvent.
- For each halo $h$, draw $K$ **independent** noise tensors $z_{h,1}, \ldots, z_{h,K}$ (the opposite of the Jacobian script, which fixes noise across param perturbations).
- Denormalize the generated `phys_3HW` exactly as `fd_jacobian_cv.run_compute` does (`* sigma + mu`, then exponentiate).
- Variance decomposition (log space for masses, linear for shapes):

  Let $X_{h,k}^{(o)}$ be observable $o$ for halo $h$, sample $k$. For mass observables use $Y = \log_{10} X$; for shapes use $Y = X$.

  $\bar{Y}_h^{(o)} = \frac{1}{K}\sum_k Y_{h,k}^{(o)}$
  $\sigma_{\rm intra}^{(o)} = \langle \mathrm{std}_k(Y_{h,k}^{(o)}) \rangle_h$ — mean over halos of within-halo std
  $\sigma_{\rm inter}^{(o)} = \mathrm{std}_h(\bar{Y}_h^{(o)})$ — std over halo means
  $\sigma_{\rm total}^{(o)} = \mathrm{std}_{(h,k)}(Y_{h,k}^{(o)})$ — pooled
- Identity check: $\sigma_{\rm total}^2 \approx \sigma_{\rm inter}^2 + \sigma_{\rm intra}^2$ to within a few percent.

**Gating check.** Run `measure_scatter` on fiducial $\theta$ with `K=4`, 20 halos. Print all three sigmas. Confirm the identity holds (within 5%). Save the test output as `scatter/test_fiducial_K4.npz`.

---

### Phase 2 — Posterior calibration against CV ground truth (4–6 hr)

**Goal.** Validate that BIND's $\sigma_{\rm inter}$ at fiducial $\theta$ reproduces the empirical scatter measured across the 27 CV simulations.

**This is the most important phase. Do not skip.** If calibration fails, the rest of the paper rests on uncalibrated posteriors. Refer back to the methods-paper hole list.

**File.** `scatter/calibration_cv.py`.

**Actions.**
1. For each of the 27 CV sims, compute observables on the *true* hydro halos (using the same `observables_from_phys`, NOT BIND samples).
2. For matched DMO conditionings, run `measure_scatter` at fiducial $\theta$ with $K \geq 10$.
3. Bin by halo mass: 3 bins ($10^{13}{-}10^{13.5}$, $10^{13.5}{-}10^{14}$, $10^{14}{-}10^{14.8}$).
4. Per mass bin per observable, compute:
   - $\sigma_{\rm CV}^{\rm true}$: empirical scatter across CV sims of the per-halo $Y$
   - $\sigma_{\rm BIND}^{\rm inter}$: BIND's inter scatter
   - $\sigma_{\rm BIND}^{\rm total}$: BIND's total scatter
5. Plot: bar chart per observable per mass bin, three bars (`true`, `BIND inter`, `BIND total`).
6. Save table to `scatter/cv_calibration.csv`.

**Gating check.**
- $|\sigma_{\rm BIND}^{\rm total} - \sigma_{\rm CV}^{\rm true}| / \sigma_{\rm CV}^{\rm true} < 0.3$ for at least 8 of the 11 observables averaged over mass bins.
- **If this fails**: STOP. Write findings to `PLAN_NOTES.md` and flag for human review. Likely fixes: (a) longer training, (b) better CFG, (c) revisit the noise sampling. Do not paper over this.
- **If this passes**: the rest of the analysis is on solid ground.

---

### Phase 3 — Map scatter across the 35-D parameter space (1 day GPU)

**Goal.** Compute $\partial \sigma_{\rm inter} / \partial \theta_j$ and $\partial \langle \bar{Y} \rangle / \partial \theta_j$ for each parameter $j$ and observable $o$, at the SB35 fiducial.

**File.** `scatter/scatter_jacobian.py`. Pattern after `fd_jacobian_cv.py` (compute → merge two-stage).

**Strategy.** Central finite differences on $\sigma_{\rm inter}$:
- For each $j \in \{0, \ldots, 34\}$:
  - Run `measure_scatter` at $\theta^+ = \tilde\theta + \varepsilon e_j$ and $\theta^- = \tilde\theta - \varepsilon e_j$, with $\varepsilon = 0.05$ in normalized units (larger than the Jacobian's $10^{-3}$ because second-moment estimators have higher variance — you need a real signal).
  - Same set of CV halo conditionings for $+$ and $-$ (and same noise seeds **within each side** for variance reduction; *different* between $+$ and $-$ is fine because we're estimating a population statistic).
  - Use $K \geq 10$ per halo, $N_h \geq 100$ halos. This is the budget driver: $35 \times 2 \times 100 \times 10 = 70{,}000$ generations.
- Compute:
  - $\Delta_j \langle\bar Y\rangle^{(o)} = \langle\bar Y\rangle^+ - \langle\bar Y\rangle^-$
  - $\Delta_j \sigma_{\rm inter}^{(o)} = \sigma_{\rm inter}^+ - \sigma_{\rm inter}^-$
  - Normalize by $2\varepsilon$ to get the (one-sided in $\log\sigma$) sensitivities $\partial \log\langle\bar Y\rangle/\partial \theta_j$ and $\partial \log \sigma_{\rm inter}/\partial \theta_j$.
- Save to `scatter/J_mean_and_scatter.npz` with shape `(N_obs, N_params)` for each of `J_mean`, `J_log_sigma`, plus standard errors estimated from a single-sided variance over halos.

**Implementation.**
- Use multi-GPU if available (DDP pattern from `train.py`); otherwise run on 1 GPU and budget ~12 hours.
- Save per-parameter intermediate npz files in `scatter/intermediate/` so a crash mid-run is recoverable.
- Add a `--params` flag so you can run a subset (debugging).

**Gating check.**
- Print, for each observable, the parameter with the largest $|\partial\log\sigma/\partial\theta_j|$ and the largest $|\partial\log\langle\bar Y\rangle/\partial\theta_j|$.
- Sanity: $\Omega_m$ should appear among top-3 movers of $\langle M_{\rm gas}\rangle$. If not, something is wrong.
- The standard errors on $\partial\log\sigma$ should be $< 30\%$ of the mean signal for at least the AGN/SN amplitude parameters. If not, increase $K$ or $N_h$.

---

### Phase 4 — Build the headline figure and the physics interpretation (4–6 hr)

**Goal.** Produce the figure that justifies the paper: scatter-vs-mean response across parameters.

**File.** `scatter/figures.py`, populating `paper_figures/scatter/`.

**Figures to make.**

1. **`scatter/fig1_calibration.pdf`** — Phase 2 output: BIND vs CV-true scatter per observable per mass bin.

2. **`scatter/fig2_scatter_vs_mean.pdf`** — **headline plot.** For each of the 3 most informative observables ($M_{\rm gas}$, $M_\star$, $\Delta q_{\rm DM}$), a scatter plot:
   - x-axis: $\partial \log\langle\bar Y\rangle / \partial \theta_j$
   - y-axis: $\partial \log\sigma_{\rm inter} / \partial \theta_j$
   - Points: one per parameter, colored by parameter group (SN/AGN/cosmo/other), labeled with $\theta_j$ name.
   - Annotate: parameters that move *only* the mean (on x-axis), *only* the scatter (on y-axis), and *both* (off-diagonal).
   - **This is the single plot that justifies the paper.** If it shows clean separation between mean-movers and scatter-movers, the headline claim is supported.

3. **`scatter/fig3_scatter_contours.pdf`** — $\sigma_{\rm inter}(\log M_{\rm gas} | M_{200} = 10^{14})$ as a 2D contour in the $(A_{\rm AGN1}, A_{\rm SN1})$ plane, computed by re-running `measure_scatter` on a 5×5 grid in those two parameters (others fixed at fiducial). Cost: $25 \times 100 \times 10 = 25{,}000$ generations — ~3 hr.

4. **`scatter/fig4_inter_vs_intra.pdf`** — for each observable at fiducial, bar chart of $\sigma_{\rm inter}$ vs $\sigma_{\rm intra}$. Establishes that DMO conditioning explains most scatter.

**Gating check.** Visual inspection. Figures saved as PDF and PNG. Each has a self-explanatory caption draft in `scatter/captions.md`.

---

### Phase 5 — Robustness checks (1 day)

**Goal.** Anticipate referee questions. Run them now, not after submission.

Each check is its own short script under `scatter/robustness/`.

1. **Noise-sample budget.** Does the result stabilize at $K=10$? Re-run Phase 4 figure 2 at $K \in \{5, 10, 20\}$ and overplot. If $K=10$ is converged, document and stop. Otherwise bump $K$ for the final analysis.

2. **Mass-bin stability.** Repeat Phase 3 separately in 3 mass bins. Make a 3-panel version of fig2.

3. **LOS contamination test.** Re-include the dropped observables ($f_b$, $R_{\rm cl}/R_{200}$, $\Sigma_{\rm gas,c}$) and put them on the same plot — they should show *similar* mean responses but *unreliable* scatter responses (or be visibly off-trend). This is the appendix figure that defends the observable-set choice.

4. **Step-size robustness.** Re-run Phase 3 for 3 high-impact parameters at $\varepsilon \in \{0.025, 0.05, 0.1\}$. Show the result is linear in $\varepsilon$ (i.e. we are in the small-perturbation regime).

5. **Seed robustness.** Re-run Phase 3 for 3 high-impact parameters with a different `torch.manual_seed`. Show $\partial\log\sigma/\partial\theta_j$ shifts by less than its error bar.

**Gating check.** All five robustness scripts produce a one-line summary at the end. Collect them in `scatter/robustness/SUMMARY.md`.

---

### Phase 6 — Writing (3–5 days, human)

**Goal.** Produce a paper draft. **The agent should NOT write the prose; this phase is for the human.** The agent should produce the figure-by-figure raw materials and a structured outline.

**Outline file.** `scatter/PAPER_OUTLINE.md`. The agent fills in:
- Numbers (medians, percentiles, $R^2$, etc.) for each claim.
- Figure references and caption drafts.
- A list of every claim that depends on a Phase-2 calibration passing, marked as such.
- A list of every claim that requires Phase 3 to have completed without numerical issues.

Outline structure (the human will rewrite):
1. Intro — frame around cluster cosmology scatter systematic
2. BIND brief recap (one paragraph + cite forthcoming methods paper)
3. Posterior calibration against CV
4. Mean and scatter Jacobian
5. Headline: scatter direction is orthogonal to mean direction (or whatever Phase 4 shows)
6. Cluster cosmology implication
7. Robustness & limitations

---

## 4. Failure modes and human-handoff triggers

The agent should **stop and write a note in `PLAN_NOTES.md`** under any of these conditions:

- Phase 2 gating check fails (calibration is off).
- Phase 3 standard errors are larger than mean signal for >50% of parameters.
- Phase 4 figure 2 shows no separation between mean and scatter responses (i.e., $\partial\log\sigma$ is perfectly collinear with $\partial\log\langle\bar Y\rangle$). In this case the headline claim collapses and the paper needs reframing — see fallback claims in §0.
- Any of the destructive-action conditions in the harness rules.

Do not push to git, do not submit to a queue, do not run anything beyond a single A100 without confirming with the user.

---

## 5. Deliverables checklist

- [ ] `scatter/measure_scatter.py` with `measure_scatter()` interface as specified.
- [ ] `scatter/calibration_cv.py` + `scatter/cv_calibration.csv` + `scatter/fig1_calibration.pdf`.
- [ ] `scatter/scatter_jacobian.py` + `scatter/J_mean_and_scatter.npz`.
- [ ] `scatter/figures.py` + `paper_figures/scatter/fig{1,2,3,4}.pdf`.
- [ ] `scatter/robustness/SUMMARY.md` covering all 5 checks.
- [ ] `scatter/PAPER_OUTLINE.md` populated with numbers and figure references.
- [ ] `PLAN_NOTES.md` with any deviations or unresolved issues.

---

## 6. What this paper is and is not

**Is.** A self-contained science paper that demonstrates BIND can answer a question no other tool can: parameter-dependent intrinsic scatter in cluster baryon content. Connects directly to a leading cluster-cosmology systematic.

**Is not.** The BIND methods paper. The methods paper is a separate manuscript (the current `draft.tex`) and should be cited as "forthcoming". This scatter paper does NOT need to re-validate the model from scratch — the calibration check in Phase 2 is the only validation needed for the scatter result to be credible.

**Order of submission.** Methods paper first, scatter paper second. They can be on arXiv within a week of each other.
