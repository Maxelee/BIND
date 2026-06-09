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

## 2026-06-07 — `feature/redshift`: publish the redshift+thermo model to Hugging Face

Released the trained multi-z model (`/mnt/home/mlee1/ceph/fm_runs/fm_redshift`,
`last.ckpt` = epoch 199; `stars_two_head` + `predict_thermo` + `condition_redshift`)
as run **`fm_redshift_thermo`** on the HF weights repo. Slimmed the checkpoint
3988→1994 MB via `bind.tools.slim_checkpoint` (drops optimizer state, keeps
`state_dict`+`ema_state_dict`+hparams; verified it reloads through
`FlowMatchingLit`), then uploaded `last.ckpt`+`norm_stats.npz` to
`mel2260/BIND/fm_redshift_thermo/`.

Corrected a stale repo pointer: the real HF weights repo is **`mel2260/BIND`**
(already hosting `fm_two_head`/`fm_thermo`), not `Maxelee/BIND2`. Updated
`download_weights.py` (`DEFAULT_REPO`, registered the new run in `KNOWN_RUNS`),
`pyproject.toml`, `README.md`, `docs/index.md`, `docs/baryonify.md`. GitHub URLs
(`Maxelee/BIND`) left unchanged. ⚠ z>0 thermo physics still unvalidated — model
card warning not yet added.

## 2026-06-04 — `feature/redshift`: stage + launch the multi-z conditioned training

The multi-z dataset (`/mnt/home/mlee1/ceph/train_data_multiz_128_cpu`,
`process_simulations_multiz.py` output) is on disk: 922 train / 101 test sims,
**7** snapshots each (snap 024/z=4 has no halos >1e13, so it dropped out;
nominal z = 0, 0.21, 0.47, 1.05, 1.48, 2.00, 3.01). Verified sample npz carry
`redshift`/`scale_factor` + 4 thermo maps across z.

- **Prep** (`data_generation/prep_redshift_training.py`, new): one-time
  single-process build of the two recursive file caches
  (`file_list_cache_multiz.txt`: **151,685** train / **15,789** test) +
  `fm_redshift/norm_stats.npz` (stars_two_head + predict_thermo). Rationale:
  on a fresh run all 8 DDP ranks would each rglob ceph + compute stats and race
  on both writes. Gotcha: the interactive Slurm job has a **17.5 GB** cgroup cap;
  the default 10k-sample float64 stack OOM-killed (~12 GB) — prep now defaults to
  `n_stats_samples=4000` (~65M px/channel; the 1 TB training node keeps 10k).
- **Launch**: `REDSHIFT=1 sbatch run_train.sh` → `--condition_redshift
  --predict_thermo --stars_two_head`, out_ch=8, writes `fm_runs/fm_redshift/`.
  (Run by the user; sbatch isn't run from here.)
- **Notebook** (`examples/analysis_redshift.ipynb`, new): mirrors
  `analysis_thermo.ipynb` + a redshift-dependence section — per-z fidelity
  scorecard, amplitude evolution truth-vs-BIND, Y–M evolution, and a
  conditioning-response test (fix structure+params, sweep only the conditioning
  a). Same z>0-thermo-physics caveat applies to absolute high-z amplitudes.

## 2026-05-31 — `feature/redshift`: multi-redshift data + redshift conditioning

New branch off `main` to make redshift a continuous conditioning variable.
Design (decided with the user): condition on **scale factor a=1/(1+z)** via a
dedicated summed embedding (mirroring the time embedding), kept **optional**
(back-compatible); new dataset directory; inference accepts redshift *or* scale
factor.

- **Model/data/train** (`eda1c63`): `UNet(condition_redshift=)` adds a
  sinusoidal→MLP `redshift_emb` summed into AdaGroupNorm conditioning;
  `forward(x,t,params,scale_factor=None)` (defaults a=1 for a redshift model);
  `FlowMatching.loss/sample` thread `scale_factor` (held fixed under CFG).
  `data.py`: `z_to_a`/`a_to_z`, `SNAPSHOT_REDSHIFTS`, `AstroDataset(condition_redshift=)`
  emits per-sample `scale_factor`, `load_file_list(recursive=)` for the nested
  layout. `train.py`: `--condition_redshift`.
- **Data gen** (`457a088`): `data_generation/process_simulations_multiz.py`
  merges the mass + thermo pipelines into one MPI pass over 8 snapshots
  (z=0..4), 1 rotation/halo, nested `train/sim_i/snap_NNN/` with `redshift`/
  `scale_factor` stored. + `run_mpi_multiz.sh`.
- **Inference** (`ff5b237`): `bind.paint(..., redshift=/scale_factor=)` and the
  `bind-paint` CLI flags; `Model` reads `condition_redshift` from the checkpoint.

**Open / needs the user:** (1) ⚠ the z>0 **thermo** comoving→physical factors
(physical density ∝ a⁻³ → pressure/entropy; physical pixel area ∝ a² →
Compton-y) are implemented but **unvalidated** against an independent reference.
(2) Data generation + training are the user's compute steps (MPI/Pylians/GPU —
not runnable here); code is syntax-checked + unit-smoke-tested only.

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
