# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

---
## 2026-05-27 — `ksz_project`: pre-SBI information-content gate (1/2/3)

Before committing to NPE/NLE, ran the three information-content gates on the
cube model (`fm_cube_two_head`, cluster-depth τ). New: `param_meta.py` (SB35
names/bounds/aliases), `sensitivity_1p.py` (Gate 1), `information_content.py`
(Gates 2+3). Verdict written to `docs/paper2_ksz_plan.md` §4.H.

- **Gate 1 (1P, isolated variation): GO.** τ responds strongly to *feedback*:
  A_SN1 |Δτ/τ|≈0.81, IMF 0.62, A_SN2 0.47, A_AGN2 0.39, BH_radeff 0.33; 18/34
  responsive params are feedback; BIND reproduces the truth response *sign* on
  every top responder. ⇒ validation_e's "only Ω_m/Ω_b informed" was a
  linear-surrogate + SB35 simultaneous-variation artifact, not insensitivity.
- **Gates 2+3 (SB35 inverse): degenerate.** Held-out R²(θ_j|x): **0 feedback
  params reach 0.1** (best IMF 0.098, truth ceiling 0.097 → observable-limited,
  not BIND). RF doesn't beat linear (~100 sims, data-starved); rich observable
  (annular+scatter, dim 3→18) adds ~1 param. Only Ω_m, Ω_b clear the bar.
- **Verdict:** τ is sensitive-but-degenerate → SBI worth it, but deliverable is
  a low-dim feedback *direction* (SVD/Fig 4), not per-param IDs (= §2 outcome 1).
  Break degeneracy with multi-probe (X-ray/tSZ); train inference on LH (1000)
  not SB35 (~100).

---
## 2026-05-27 — `ksz_project`: validation A–G correctness pass (3 fixes)

Audited A–G for the kSZ τ observable and fixed three real problems (least →
most important). All A–G re-run on `fm_two_head` (CV+1P+Test) exit 0.

- **Fix 3 — consistency + C significance.** Added `tau_utils.per_halo_tau`
  (disk | CAP, scalar/per-halo radii) as the single estimator used by
  A/C/D/E/F (was 3 divergent local copies). Rewrote `validation_c.py` to
  aggregate to **one stacked τ per sim** before the Spearman — within an SB35
  sim all halos share θ, so per-halo pooling was pseudo-replication (N_halos
  not N_sims) with meaningless p-values. Now N=101 sims, honest p. BIND tracks
  truth: Ω_b ρ=+0.36 (truth +0.37), Ω_m −0.35, p12/p4/p25 agree.
- **Fix 2 — real coverage test.** `validation_e.py` now feeds the posterior the
  **real held-out stack `x[i]`**, not `emu.predict(θ_i)`; the old version was
  self-referential and could only ever over-cover. Added SBC ranks
  (Φ(θ_true), uniform iff calibrated) + per-param KS p + out-of-sample residual.
  Honest result: emulator out-of-sample residual ≈ **0.337**; SBC **rejects
  uniformity (KS p≈0) even for the 2 informed params** (Ω_m constraint 0.32,
  Ω_b 0.14) — coverage→1.0 is *over*-dispersion, not calibration. Argues for
  real BIND-in-the-loop NPE/NLE over the linear surrogate. `validation_f.py`
  rebuilt on the same real-data base → Ω_m coverage now degrades 1.00→0.98 as
  σ_v→0.30 (was trivially flat); docstring flags flatness is only meaningful
  for data-informed params.
- **Fix 1 — LOS / canonical observable (the §6.1 risk).** CAP is now the
  default estimator for C/D/E/F (∑w=0 cancels the uniform 50 Mpc/h LOS
  background, mirroring ACT). Loader records `truth_source`/`los_depth_mpc_h`;
  every observable script prints a `[LOS]` banner
  (`fullbox → 50.0 Mpc/h, ~10× cluster depth`) so 2D-vs-cube projection can't be
  silently conflated. Documented the decision in
  [docs/paper2_ksz_plan.md](paper2_ksz_plan.md) §6.1 + a §4 "what this
  establishes" note: **A–C are emulator-fidelity checks at the 50 Mpc/h
  projection, not cluster-depth τ.**
- **Cube validation (ACT-comparable τ).** Ran A–G on the cube suite
  `fm_testsuite_cube` / `fm_cube_two_head` (6.25 Mpc/h LOS ≈ cluster depth; LOS
  banner confirms `truth_source=cube`). Outputs keyed `_fm_cube_two_head`
  alongside the 2D ones. **Key finding:** at cluster depth the background no
  longer dilutes the halo signal, so BIND's over-prediction is more visible —
  D's CAP τ(M) is **+11.6 % above truth at logM≈13.15** (vs +3.2 % for 2D),
  +5–8 % higher mass; A bias +0.03 dex (vs +0.01). The +12 % at the ACT-deficit
  mass scale is a BIND-fidelity floor for the inference error budget. E's
  out-of-sample emulator residual 0.28 (vs 0.34 for 2D); still only Ω_m/Ω_b
  informed, SBC non-uniform.

---
## 2026-05-27 — `ksz_project`: validation E (SBI coverage) + F (v_los robustness)

Extended the kSZ validation pipeline with the remaining two §4 plots:

- **E. Leave-one-out coverage** on the 102-sim Test (SB35) pool. Built a tiny
  dependency-free emulator in `analysis/ksz/inference.py`: per-mass-bin ridge
  regression θ_std → x_k with empirical residual σ, plus analytic Gaussian
  posterior on raw θ (with N(0, prior_std²) prior on standardised θ).
  `validation_e.py` runs LOO at level 0.6827 with multiplicative measurement
  noise (default 5%) and 8 realizations per held-out sim.
- **F. v_los robustness** uses the same emulator with an extra multiplicative
  systematic x_obs = x_true × (1 + ε_v), ε_v ~ N(0, σ_v²), swept across
  σ_v ∈ {0, 0.05, 0.10, 0.20, 0.30}.  Posterior shift / coverage
  degradation as a function of σ_v.
- Wired into `run_ksz_validation_a.sh` (PLOTS default → `ABCDEFG`).

Honest smoke result on fm_two_head / Test / 5 mass bins (3 survive the
≥ 80 %-sim-coverage filter, since SB35 sims have few halos at log M > 14):
**only 2/35 params are data-informed**: p00 (Ω_m, constraint = 0.32) and p06
(Ω_b, constraint = 0.14) where `constraint = 1 − σ_post / σ_prior` in std-θ
space.  All others are prior-dominated by construction — coverage = 1.0 is a
mathematical identity, not calibration.  Both informed params are also
over-covered (1.0 vs 0.68), meaning the ridge-emulator residual σ is wider
than the per-sim true scatter (this is the conservative direction).  F is
correspondingly flat in σ_v: when the data don't constrain a parameter, v_los
systematics can't move the posterior.  Both are useful negative results: the
stacked τ(M) observable alone is rank-3 — to get usable constraints on
subgrid params we need (a) more mass bins (HMF-limited above 10^14 — plot G)
or (b) a richer observable (annular profiles → plot B, or τ × M scaling →
future work).  Recorded in `figures/ksz_validation_e_*.txt`.

---
## 2026-05-27 — `ksz_project`: Paper-2 scoping + validation pipeline A–G

New branch `ksz_project` (off `main`). Adds the Paper-2 scoping doc and a
CPU-only kSZ validation pipeline on top of existing `test_suite/` artifacts.

- [docs/paper2_ksz_plan.md](paper2_ksz_plan.md) — motivation, falsifiable
  deliverable, procedure, 7 validation plots, 6 result figures, risks.
- `analysis/ksz/` — shared loaders (`_io.py`, `tau_utils.py`) and per-plot
  modules. Currently implemented:
  - **A** per-halo τ recovery scatter (BIND vs truth, mass-binned dex stats),
  - **B** annular τ(R/R200) profiles, mass-binned median ± 16–84%,
  - **C** Spearman τ–parameter sensitivity (35-bar BIND vs truth on Test suite,
    CV excluded due to zero param variance),
  - **D** stacked τ(M) in ACT-DR6-like apertures (disk or compensated CAP),
  - **G** HMF coverage / per-sim halo counts (model-independent).
- `run_ksz_validation_a.sh` — single SLURM script (also runs as `bash`) driving
  A–G; `PLOTS=ABCDG` default, `--partition=gen --mem=32G --time=02:00:00`.

Smoke results on `fm_two_head`: A bias ≈ +0.01 dex, scatter 3–7%; D shows a
+5–10% mass-trending overprediction at R_ap=0.5 Mpc/h (CAP); G flags
log M > 14 as HMF-limited at ~2 halos/sim. Next: E (SBI coverage), F (v_los
robustness).

Caveats hit and recorded to memory: CAMELS test-suite dir names are
case-sensitive (`CV / 1P / Test`, not `cv/1p/sb35`); legacy `halo_catalog.npz`
stores R200 under `radii` in **kpc/h** (no `r200s` key).

---

## 2026-05-27 — Repo hygiene, branch reorganization, and agent instructions

**Repo cleanup.** The repo had no `.gitignore`, so ~304 untracked items
(2 GB of caches/outputs/figures, committed `.pyc`) were noise. Added a
`.gitignore` (caches, `outputs/`, figures, `*.npz`/`*.npy`/`*.log`, pycache,
notebook checkpoints, machine-local `.claude/settings.local.json`), untracked
the committed `.pyc` files, and refreshed the tracked paper figures. Untracked
count: 304 → 0.

**Branch reorganization.** Decision: keep `main` a clean trunk and park distinct
analyses on topic branches instead of dumping everything on `main`.
- `main` — core engine (`data/model/train/metrics`, `test_suite/`) + the
  ~890-line engine evolution since the last working-model commit + refreshed
  `paper_figures.ipynb`.
- `feature/3d-cube` — 3D / cube-projection extension.
- `analysis/2d` — scatter package, observables, `project1-7`, CV derivatives.
- `wip` — scratch notebooks, parameter-injection experiments, planning notes.

Notebooks are committed with outputs (per preference). No git remote — local-only.

**Agent instructions.** Added `CLAUDE.md` (architecture + commands + conventions
+ data caveats), this `docs/WORKLOG.md`, and `.github/copilot-instructions.md`
mirroring the project context for GitHub Copilot. Then merged `main` into each
topic branch so they all carry the shared docs, and appended a tailored
`## This branch: …` section to `CLAUDE.md` + the Copilot file on each
(`feature/3d-cube`, `analysis/2d`, `wip`) describing that branch's projects.
`main`'s copy stays generic.
