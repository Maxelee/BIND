# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

---

## 2026-06-09 — Circular paste aperture is now the standard (`r200_factor=4.0`)

The BIND composite now defaults to a **circular `4×R200c` paste aperture** instead
of the legacy square Hann taper. Over 26 CV sims (`fm_two_head`), circular removes
the small-scale total-matter `P(k)` deficit: BIND/Truth at `k` 40–70 h/Mpc goes
−10.6% (square) → −0.8% (circular), at the cost of mild +7–11% over-production at
`k` 10–40. The hydro-replaced control shows the square-aperture high-`k` deficit is
an over-smooth-core *model* issue that the tight aperture compensates geometrically.

- Engine: default `r200_factor` 0.0→4.0 in `RunConfig` (`schemas.py`), both CLIs
  (`camels_suite`, `paint`), and `build_bind_composite`. `load_halo_catalog` now reads
  R200c from cached catalogs (legacy `radii` kpc/h as well as `r200s` Mpc/h), so a
  `--repaste` no longer needs the FOF files and preserves halo↔patch ordering.
- Note + example figure: `docs/circular_aperture.md`. Full study (also scale_global
  P(k)-invariance + taper sweeps): `experiments/composite_study/FINDINGS.md`.
- Reversible: `--r200_factor 0` restores square; circular composites rebuild cheaply
  from cached `generated_halos.npz`.

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
