# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — unreleased (release date TBD)

The release that accompanies the BIND methods paper. It adds the weak-lensing
statistics emulator, the staged large-box painting path, redshift and observable
conditioning, and the in-repo training-data pipeline — and it **changes the
default compositing behaviour of the painting API**, see the breaking notes below.

### ⚠️ Breaking / behavioural changes

- **The circular paste aperture is now the standard everywhere.**
  `r200_factor = 4.0` (circular Hann-tapered aperture of radius `4 × R200c`) plus
  `paste_mode = "shared"` replaced the square whole-patch taper
  (`r200_factor = 0`) and the independent-patch blend (`paste_mode = "average"`)
  as the defaults. Both legacy modes remain reachable explicitly. Any composite
  produced with v0.1.0 defaults is *not* comparable to a v0.2.0 composite.

- **SLURM wrapper scripts were consolidated.** `run_train_two_head.sh` was
  removed (use `run_train.sh`, with `THERMO=1` / `REDSHIFT=1` / `OBS=1` selecting
  the mode) and `run_test_suite_parallel.sh` was renamed to `run_test_suite.sh`
  (the model is selected by the `RUN_DIR` / `MODEL_NAME` / `CHECKPOINT_PATH`
  environment overrides).

### Added

- **`bind.wlemu` — weak-lensing statistics emulator.** 30 SB35 astrophysical
  parameters + source redshift → all κ summary statistics (Cl, PDF, peaks,
  minima, Minkowski functionals V0–V2, scattering coefficients, moments; 383
  dimensions) with GP σ and a single-field covariance. Per-redshift PCA plus
  exact ARD-Matérn-5/2 GPs; **inference is numpy-only** from the packaged
  artifact `src/bind/assets/wlemu_gp.npz` (~3.3 MB, shipped in the wheel).
  Fitting and 13-fold cross-validation live in `bind.wlemu.fit` behind the
  `wlemu-fit` extra. New `bind-wlemu` CLI, `examples/wlemu_tutorial.ipynb`,
  `docs/wl_emulator.md`. Subsequent work added user-chosen ℓ/ν output grids,
  continuous source redshift via PCHIP cross-plane interpolation, a whitened
  sensitivity map, and active-subspace reduction.
- **Staged large-box painting path.** `bind.inference.paint_stages` plus three
  new console scripts — `bind-paint-project` (CPU, optionally MPI: particles →
  z-slab maps + halo cutouts), `bind-paint-generate` (GPU: sample hydro patches),
  `bind-paint-recomposite` (CPU: paste patches into full-box maps). The
  intermediate products are cached, so a composite can be rebuilt with different
  paste settings without re-generating. Exported at top level as
  `bind.project_and_extract`, `bind.generate_from_stage1`,
  `bind.recomposite_slab`, `bind.recomposite_from_saved`. New `mpi` extra.
- **Redshift conditioning.** `bind.train --condition_redshift` conditions the
  UNet on the scale factor `a = 1/(1+z)` through its own sinusoidal→MLP embedding
  summed into `AdaGroupNorm`; a redshift model defaults a missing `scale_factor`
  to `a = 1` (z = 0). Inference accepts `bind.paint(..., redshift=…)` or
  `scale_factor=…`, and `bind-paint --redshift/--scale_factor`. New multi-redshift
  data generator (8 snapshots, z = 0–4, mass + thermo in one pass) and
  `REDSHIFT=1 run_train.sh`. `--exclude_snaps` supports leave-one-redshift-out
  tests. Documented in `docs/redshift.md`.
- **Observable conditioning.** `bind.train --condition_observables` conditions on
  aperture-integrated R200 observables (`Y_200, Mgas_200, Mstar_200, Tx_200,
  K_200, P_200, M_200`) instead of the 35 parameters, with `--mask_observables`
  for partial vectors. `bind-camels-suite` auto-detects such models and measures
  the observables from truth. `docs/observables.md`,
  `examples/analysis_observables.ipynb`.
- **Variational-diffusion / score-matching interpolant** on the same UNet, as an
  alternative to flow matching, with a comparison harness (`validate_vdm.py`,
  `run_train_vdm.sh`, `run_validate_vdm.sh`).
- **Training-data generation pipeline in-repo** (`data_generation/`): the MPI/SLURM
  scripts that build `<DATA_ROOT>` from the CAMELS SB35 suite — mass maps, gas
  thermodynamic maps, the optional 1e12–1e13 low-mass extension, the multi-redshift
  variant, and an optional R200 persist step. Documented in
  `docs/data_generation.md`.
- **`tools/paper_cache/`** — the cache builders and figure producers behind the
  methods-paper figures (metrics, P(k), redshift cache, posterior ensembles, ODE
  convergence, timing benchmark, and the per-figure builders under
  `tools/paper_cache/figures/`).
- **`docs/circular_aperture.md`** — the measured justification for the circular
  paste aperture, with the k-band table that motivates the new defaults.
- Low-mass extrapolation and multi-halo "covering" paint studies
  (`run_lowmass_suite.sh`, `examples/lowmass_extrapolation.ipynb`,
  `examples/covering_*.ipynb`, `examples/offcenter_covering.ipynb`).
- `fm_redshift_thermo` added to the checkpoints known to `bind-download-weights`.
- Release metadata: `CITATION.cff`, `CHANGELOG.md`, `NOTICE.md`, `MANIFEST.in`,
  and a Data & Attribution section in `README.md` covering CAMELS/IllustrisTNG
  provenance and the code-vs-weights licence split.
- **Reproducible sampling.** `--seed` on `bind-paint`, `bind-camels-suite`,
  `bind-paint-generate` and `bind-paint-recomposite`, plus a `seed=` keyword on
  `bind.paint()`, `Model.generate()`, `generate_from_stage1()` and
  `recomposite_from_saved()`. BIND is a generative model, so before this every
  run drew different noise and no published map could be reproduced even with
  identical inputs and weights. Seeding uses a local `torch.Generator`; **with
  no seed the sampler takes literally the previous code path**, so unseeded
  results are unchanged from v0.1.0.
- **Provenance stamping.** Every `summary.json` and output `.npz` now records
  `bind_version`, the checkpoint and norm-stats sha256, and the resolved
  `n_steps` / `r200_factor` / `paste_mode` / `seed`
  (`bind.inference.artifacts.build_provenance`). Read it back with
  `read_provenance()`; it returns `None` for artifacts written before stamping
  existed, which is itself the useful signal. This retires a real archaeology
  problem — one cached results tree holds six model evaluations at three
  `n_steps` and three `r200_factor` values with no record of which is which.
- **First test suite** (`tests/`, 54 tests, ~7 s on CPU with no data, GPU,
  network or checkpoint) and a real CI gate: shell syntax, pinned ruff over
  `src` and `tests`, version consistency, a `pkgutil` walk that imports every
  shipped module, console-script resolution, a wheel-asset check, and pytest.
- **`docs/redshift.md`** and **`docs/reproducing_the_paper.md`** — redshift
  conditioning as a user-facing feature, and a per-figure provenance table
  mapping each manuscript figure to its producing script, cache and model.
- `tools/paper_cache/acceptance_gate.sh` — regenerates the paper's Table 2 from
  the committed cache and requires it back byte-identical.

### Changed

- Version is `0.2.0` in both `pyproject.toml` and `bind.__version__`, with a CI
  check that the two stay in sync.
- **`bind-download-weights` now defaults to the `mel2260/BIND` Hugging Face
  repository** (it was `Maxelee/BIND2` in v0.1.0). Override with `--hf_repo`.
- **Dependency declarations corrected.** `tqdm` is imported at module scope in
  `inference/{pipeline,paint,paint_stages}.py` — all on the `import bind` path —
  but was declared nowhere, so a fresh install produced an unimportable package.
  `tqdm` and `scipy` are now declared, and `requirements.txt` (which was missing
  `Pylians` and `tqdm` entirely) is a labelled mirror of
  `[project.dependencies]`.
- **CI rebuilt.** It now syntax-checks every `run_*.sh`, pins ruff, checks
  version consistency, walks and imports every shipped `bind.*` module via
  `pkgutil` instead of a hand-maintained list, resolves every declared console
  script, asserts the packaged assets are present in the built wheel, and runs
  pytest. Triggers extended to tags and `paper/**`.
- `.gitignore` extended to cover `build/`, `dist/`, egg-info, tool caches and
  LaTeX intermediates.
- `README.md` rewritten for accuracy: the acronym is **Baryonic INpainting with
  Deep learning** (it was previously expanded incorrectly), the model is
  described as halo-centric rather than uniformly tiled, the compositing defaults
  are documented correctly, the CLI list is complete, and the docs link no longer
  points at `bind.readthedocs.io`, which belongs to an unrelated project.

### Fixed

- **`bind-paint-generate` and `bind-paint-recomposite` no longer emit the legacy
  composite.** Both defaulted `--r200_factor 0.0` and exposed no `--paste_mode`,
  while `bind-paint`, `bind-camels-suite`, `RunConfig` and the `paint_stages`
  library functions all defaulted to `4.0` / `"shared"`. Because the CLIs passed
  their argparse defaults explicitly they overrode the library default, and since
  `pipeline` gates shared-content handling on
  `paste_mode == "shared" and r200_factor > 0`, the old value silently disabled
  shared-content handling too. The same repository produced two different
  scientific answers depending on which entry point you used, with no warning.
  Both CLIs are new in 0.2.0, so this is an internal inconsistency corrected
  before release rather than a change to any published interface. The legacy
  configuration is measured at **−10.6% total-matter P(k) at k = 40–70 h/Mpc**
  (versus −0.8% for the circular aperture) in `docs/circular_aperture.md`.

- **`run_train.sh` was unusable.** It had been committed in a syntactically
  invalid state — the `OBS` mode was merged into the `REDSHIFT`/`THERMO` chain
  without closing it, leaving an unterminated `if`, a duplicated `elif THERMO`
  arm and two contradictory `echo` lines. `bash -n run_train.sh` exited 2 at line
  103, so the launcher aborted immediately, while `docs/training.md`,
  `docs/observables.md` and `CLAUDE.md` all instruct users to `sbatch
  run_train.sh`.
- **`REDSHIFT=1 sbatch run_train.sh` silently used the wrong data root.**
  `DATA_ROOT` was defaulted unconditionally *before* the mode chain, which made
  every per-mode default unreachable — a redshift-conditioned model was trained on
  the single-redshift dataset unless `DATA_ROOT` was also exported by hand. Mode
  selection is now one `REDSHIFT`/`OBS`/`THERMO`/else chain, with `DATA_ROOT` and
  `RUN_NAME` resolved after it so an explicit environment value still wins.
- `bind-paint-generate` / `bind-paint-recomposite` compositing defaults (see
  Breaking, above), and a stale `--r200_factor 2.0` example in the
  `paint_recomposite` docstring.
- `ruff check src` is clean. The gate had never passed on this branch; 15 of 16
  errors were import ordering and unused imports, plus a stray whitespace-only
  line and a dead `params_arr`/`params_list` accumulation in
  `compute_norm_stats_cube` (both the 2D and cube paths take parameter bounds
  from the SB35 CSV, never from the sampled files, so removing it cannot change
  any output).
- CI imported `PaintConfig`, which exists on no branch.
- A stale, committed `build/lib/**` tree shadowed `src/` and still carried the
  **old `r200_factor = 0.0` default**; it is now gitignored.
- The suite evaluation dropped `scale_factor`, so redshift-conditioned models
  were evaluated at z = 0 regardless of snapshot; it is now resolved per snapshot.
- Hydro snapshots using the `snapshot_` filename prefix (the L50n512 layout) are
  now accepted by the inference loader.
- Multi-redshift data generation: the Gas target map was missing the
  code→physical `1e10` factor, and the `(sim, snap)` work list is now shuffled so
  MPI ranks are load-balanced.
- The low-mass test manifest is namespaced and its lock file keyed by snapshot,
  so concurrent chunks no longer collide.
- Finite DDIM sampling in the VDM branch.

### Notes for reproducibility

- Released weights live on Hugging Face at
  [`mel2260/BIND`](https://huggingface.co/mel2260/BIND) — `fm_two_head`,
  `fm_thermo`, `fm_redshift_thermo`. **A Hugging Face repository is mutable**, so
  reproducing published numbers requires pinning a revision:
  `bind-download-weights <run> --revision <sha>`.
  **The exact revision SHA for this release is TBD** and will be recorded here.
- The released `fm_two_head` checkpoint is epoch 80 and contains **no**
  `ema_state_dict`; the inference path correspondingly does not apply EMA to it.

## [0.1.0] — 2026-05-29

First tagged release: the packaged `bind` distribution (`src/bind/` layout) with
the flow-matching UNet, the CAMELS training/evaluation pipeline
(`bind.train`, `bind.inference`), the `bind.paint()` deploy API and
`bind-paint` / `bind-camels-suite` CLIs, the two-head Stars parameterization,
the `--predict_thermo` gas-thermodynamic channels, and the `fm_two_head` /
`fm_thermo` checkpoints on Hugging Face.

[0.2.0]: https://github.com/Maxelee/BIND/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Maxelee/BIND/releases/tag/v0.1.0
