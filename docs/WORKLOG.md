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

## 2026-06-01 — Mutual-information notebook: rebuilt twice to per-sim, null-calibrated + expanded viz

`examples/mutual_information.ipynb` (split out of `paper_figures.ipynb`). Went
through TWO corrections driven by the human's skepticism, both proven in-notebook
(executed end-to-end, 0 errors, 16 figures, 4.1 MB; torch3 venv — from a bare
shell needs `LD_LIBRARY_PATH` → a gcc-13 libstdc++).

1. **Per-sim-means → per-halo** (fixed small-N noise). Then the human flagged a
   strong, unphysical dependence on **UVBHepDeltaz** (HeII-reion redshift width)
   on z=0 stellar mass. Diagnosis: **per-halo MI is pseudo-replicated** — only
   ~101 independent parameter draws (Test/SB35 LH) but ~4272 halos broadcast the
   same params, so KSG reports a ~0.4-bit (Stars)/~0.15 (Gas) *phantom floor on
   every parameter*. UVBHepDeltaz was pure floor.
2. **Per-halo → per-sim, null-calibrated** (the correct fix): aggregate halos →
   per-sim statistic (N=independent sims), report **excess over a shuffle-null**
   with 3σ significance + bootstrap error bars. UVBHepDeltaz excess → **0.000**
   under both per-sim and a block-preserving per-halo null; the surviving signals
   are physical: **Ωm→DM ≈1.79 bit, Ωb→Gas ≈0.79, VariableWindVel/σ8→Stars**.
   BIND reproduces the real excess (Ωb→Gas: truth 0.79 / BIND 0.83). The
   independent unit for parameter MI is the **simulation, not the halo**.

Also expanded from single colorbar heatmaps to a full battery (per the human's
request): §7 per-param profile (bar/sorted/cumulative — 80% of info in ~10
params), §8 param–param MI (LH independence check) + interaction info
(synergy/redundancy) + graph, §9 pointwise/specific information, §10
compressed-rep MI (PCA latent×param + t-SNE), §11 field-space per-pixel MI maps
(Ωb→Gas shows a feedback-regulated central hole; truth≈BIND). Multivariate KSG
estimator added (`ksg_mi`). Caches under `examples/paper_figures/mi_cache/`
(`halo_features_<model>.npz`, `sim_stacked_patches_<model>.npz`).
`paper_figures.ipynb` MI cells left in place. Per branch convention may belong on
`analysis/*`.

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
