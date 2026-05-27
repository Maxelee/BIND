# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

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
