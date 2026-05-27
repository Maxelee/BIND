# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

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
