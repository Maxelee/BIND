# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

---

## 2026-06-10 — `lightcone`: prior-bound (1P) design + portable off-site generation bundle

Per user: start with one-parameter prior-bound variations (not Sobol) and make
generation Globus-portable to a GPU-rich machine.

- **`design.twobound_design`**: 30 astro params × {min, max} = **60 lightcones**,
  one param varied per row, cosmo at TNG300. Validated.
- **`bind-make-portable`**: builds a self-contained bundle (`design/`, per-run
  `params.npy`, `conditions/` = stage-1 stripped of DMO maps [compressed],
  `weights/`, `generate_halos_portable.sh`, `README.md`). Workflow: Globus bundle
  → GPU box runs `sbatch --array=0-59 generate_halos_portable.sh` (BUNDLE-relative
  paths, needs only `pip install -e BIND` + GPU) → Globus `runs/` back → CPU side
  `DESIGN=twobound` `run_sobol_{paste,lux,stats}.sh --array=0-59`.
  Built `/mnt/home/mlee1/ceph/bind_portable_twobound` (~20 GB: conditions
  dominate; DMO background stays home for compositing).

---

## 2026-06-10 — `lightcone`: GPU/CPU split + staged SLURM scripts matching conventions

Replaced the monolithic `run_science_run.sh` with **staged, convention-matching**
array jobs (cf. run_lightcone_*.sh / lux/run_lux_bind.sh), and split the GPU-heavy
generation from the CPU raytracing so the Sobol generation can run on a GPU-rich
machine:

- **`bind-generate-halos`** (`paint_stages.generate_halos`): GPU generation ONLY —
  model.generate on the stage-1 *cutouts*, saves per-halo patches (no compositing,
  no DMO). Portable: ship cutouts+params+weights to the GPU machine, generate,
  ship the halos back; compositing (DMO background) is CPU-side via
  `bind-paint-recomposite`.
- **Staged scripts** (array index = run index): `run_sobol_generate.sh` (gpu/a100,
  ~4 GPU-hr/run), `run_sobol_paste.sh` (cca: recomposite→lensplane+yplane, deletes
  composites), `run_sobol_lux.sh` (cca, `module restore lux`, `srun -n` scalable —
  raytracing speeds up with cores; uses lux `n_realizations`), `run_sobol_stats.sh`
  (collect→stats→delete lensplanes/raw). Keeps halos/kappa/y/stats only.
- Storage confirmed ample (100 TB); kept full halo patches (no scalars-only mode).
  GPU budget here (~497 hr ≈ 100 pts) → arrays default 0-99 (Sobol is incremental;
  the 256-pt design's first 100 are a valid sub-design). Removed monolithic
  run_science_run.sh / run_science_generate.sh.

---

## 2026-06-10 — `lightcone`: canonical (ray-traced) per-run pipeline + lean data management

Dropped the Born approximation as the production path (kept `assemble_lightcone`
only for quick diagnostics). The science runs now follow the canonical chain
end-to-end per run: **generate halos → composite → lensplanes + y-planes → lux
ray-trace → collect → statistics → delete transients.**

- **Driver** `run_science_run.sh` (SLURM-array, one task per run; resolves
  `RUN_DIR` from `DESIGN`+`SLURM_ARRAY_TASK_ID`). **Keeps** `snap_*/halos.npz`,
  `kappa_maps.npz`, `y_maps.npz`, stats; **deletes** composites, lensplanes, raw
  lux `.dat` (large 4096²-class, regenerable).
- New CLIs: `bind-extract-halos` (slim per-halo arrays — patches+metadata, no
  maps) and `bind-lux-collect` (stack lux `kappa{p}.dat`/`y{p}.dat` →
  `kappa_maps.npz`/`y_maps.npz`, `--delete_raw`).
- lux: `n_realizations` param (Sobol uses fewer than the fiducial 100);
  `main.cpp` loop now reads it.
- Orchestration: `science.plan_runs` writes per-run `params.npy` + the array
  submit command; `bind-science-plan` repointed to it (Born `plan_generate`/
  `plan_maps_stats` retained as library functions, not the default path).

---

## 2026-06-10 — `lightcone`: fiducial stats notebook + R(ν) tSZ-at-WL-peaks data vector

- **Notebook** `examples/fiducial_lightcone_stats.ipynb`: end-to-end fiducial WL+tSZ
  stats from the 100 ray-traced lux κ maps (+ lux y once landed) + composites —
  Cℓκκ tomographic, baryonic suppression (same-pipeline projected matter), peaks/
  minima, PDF/moments/Minkowski, κ×y / Cℓyy, halo scaling, and R(ν).
- **`stats.peak_cross_stats`**: stacks the (geometry-consistent) lux y at WL peaks →
  **R(ν,z_s)=⟨y⟩/⟨κ⟩** (projected gas-fraction proxy ∝ Y/M ∝ f_gas·T_mw), peak
  counts, stacked radial y profile (feedback shape), and Cov(κ_peak,y_peak).
  Validated on real maps: R rises ~2× with ν (z_s=1) and falls with z_s — exactly
  the expected gas-fraction behaviour; radial y profile declines 2e-5→1e-6 core→9′.
- Wired into `bind-lightcone-stats` (writes `peak_cross.npz` when per-source-bin y
  present) → R(ν,θ_astro) is now an emulator target. Made the Born assembler
  (`assemble_lightcone`) emit **tomographic** y (per source plane, like lux) so the
  Sobol runs get R(ν) too. `lux_io.load_y_realizations` reads lux `y{plane}.dat`.
  See [[tsz-ymap-normalization]] (y is physical, summed — no 1/a² weight).

---

## 2026-06-10 — `lightcone`: stats upgrades (Pylians/MFs) + lux tSZ y-emission

Per user requests on the stats and the fiducial end-to-end run:

- **Stats** (`bind.inference.stats`): power spectra now via **Pylians**
  `Pk_plane`/`XPk_plane` (BoxSize=fov_rad → k=ell); `peak_counts` returns peaks
  **and minima** (8-neighbour); replaced Betti with **Minkowski functionals**
  `V0,V1,V2` (differential estimator) in `nongaussian_stats`. Validated on real
  lux maps: Cl rises with z_s, skewness 2.0→0.54 with smoothing, V0 1→0,
  V2(Euler) +ve at high ν, peaks≫minima at |ν|>2.
- **lux ray-traced output reader** `bind.inference.lux_io` (kappa/gamma 1024²
  float64 Fortran-record; planes 26/45/59/70/78 → z_s 0.5/1/1.5/2/2.44).
- **Cleanup:** deleted the 900 non-{26,45,59,70,78} kappa/gamma maps across the
  100 fiducial RT runs (kept 1000).
- **tSZ via lux** (chosen approach): modified `/mnt/home/mlee1/lux`
  (`params.{hpp,cpp}` add `compute_tsz`/`tsz_input_dir`; `raytracing.{hpp,cpp}`
  add `load_y`/`calc_y` + a line-of-sight y accumulator written as `y{plane}.dat`
  with the SAME per-snapshot rot/disp randomization as the lensing planes →
  pixel-consistent with kappa). Syntax-checked (g++ stubs); user recompiles
  (`module restore lux && make`) + reruns. y-planes generated by
  `bind-paint-yplane` (single-field Fortran record, validated lux format) /
  `run_lightcone_yplane.sh`; tSZ config `lux/lux_bind_tsz.ini`.
- New CLIs: `bind-paint-yplane`. ⚠ z>0 thermo a-factors in calc_y still
  unvalidated (see [[lightcone-kappa-upturn-aliasing]] sibling caveat).

---

## 2026-06-10 — `lightcone`: Stage 3 summary statistics + maps/stats orchestration

Completed the 1a/1b analysis path: per-run summary statistics + wiring map
assembly and stats into the orchestration.

- **Stats (`bind.inference.stats` + `bind-lightcone-stats`):** flat-sky
  `power_spectrum`, `cl_kappa` (tomographic auto/cross + BIND/DMO suppression
  `S(ell)`), `cl_kappa_y` (WL×tSZ cross + `C_ell^yy`), `peak_counts`,
  `nongaussian_stats` (PDF, per-scale variance/skewness/kurtosis, Betti
  `beta0/beta1`), `halo_scaling` (per-halo `Y_500c, f_gas, f_star, T_mw` from
  composite patches). Writes `Cl_kappa.npz`, `Cl_kappa_y.npz`, `peak_counts.npz`,
  `nongaussian_stats.npz`, `halo_scaling.npz` per run.
- **Orchestration:** `assemble_lightcone` gained `manifest_root` (per-run
  composites + shared manifests); `bind-lightcone-maps` gained `--manifest_root`;
  `plan_maps_stats` emits `maps_stats_tasks_<design>.db` (1 task/run, maps→stats
  chained, restart-safe). `bind-science-plan` now emits both generate and
  maps+stats task files (n_real=32 fiducial, 8 for 1P/sobol).
- **Validated** on the demo maps + real composites: skewness 1.36→0.53 with
  smoothing (WL non-Gaussianity), S(ell)→0.91, ℓ(ℓ+1)Cℓ/2π~8e-5 @ ℓ=1000;
  halo_scaling **logY–logM corr 0.971** (Y∝M^5/3), f_gas~0.10, f_star~0.014,
  T_mw~5e6 K, positive κ×y and y×y. New CLIs: `bind-lightcone-stats`.

---

## 2026-06-10 — `lightcone`: tSZ y-map compositing + flat-sky kappa/y lightcone assembly

Built Stage 2c (the missing tSZ piece) and a self-contained kappa+y assembler.

- **Thermo compositing:** generalised `paste_halos_2d` to N channels and extended
  `build_bind_composite` with `thermo_patches=` → composites the 4 gas-thermo
  channels (compton_y, T, entropy, P_e) into full-box maps (`composite_thermo`,
  blended like gas: `alpha*canvas`, no DMO background, no scale_global). Saved in
  each `composite_slab*.npz`; wired through generate + recomposite. So
  `composite_slab` now holds the roadmap's 7-channel `composite_maps` content.
- **Lightcone assembly:** `bind.inference.lightcone_maps.assemble_lightcone` +
  `bind-lightcone-maps` CLI. Flat-sky Born stack of the per-snapshot composites
  onto a **common angular grid** (so kappa & y are pixel-aligned for the 1a
  cross-correlation). kappa = sum W(chi_l,chi_s)*delta_scaled per source z; y =
  sum of compton_y planes (additive, no kernel). Periodic-bilinear FOV tiling;
  random per-snapshot shift gives the N_real realizations. Falls back to
  compositing y from saved `thermo_patches` for older composites.
- **Demo (20-snap lightcone, z_s=1, 5deg, 1024px):** kappa std 0.023 (BIND) <
  0.0234 (DMO) — baryonic suppression visible; y mean ~1e-6, peaks ~2e-4 at
  clusters. Maps + figure at `/ceph/bind_science/demo_maps.{npz,png}`.
- Caveats: Born (not lux raytracing); demo mass used the existing square-taper
  composites while y used circular r200_factor=4 — science runs regenerated with
  thermo compositing + r200_factor=4 will be fully consistent. z>0 thermo
  a-factors still unvalidated (see [[lightcone-kappa-upturn-aliasing]] sibling
  CLAUDE.md note).

---

## 2026-06-09 — `lightcone`: science pipeline for 1a/1b — Stage 0 (design) + Stage 2 orchestration

Started the astro-parameter science pipeline (1a: WL+tSZ cross vs astro params;
1b: non-Gaussian WL emulator). No MAS fix needed — science ratios use same-pipeline
DMO so the aliasing artifact cancels (see entry below).

- **Stage 0 (design):** `bind.inference.design` + `bind-design` CLI. Cosmo pinned
  at TNG300 (indices {0,1,6,7,8}); 30 astro params varied in SB35 space (log/linear
  per LogFlag). Wrote `astro_params_{sobol(256),1P(150=5×30),fiducial}.npy` +
  `design.json` to `/mnt/home/mlee1/ceph/bind_science/design`.
- **Key efficiency:** stage 1 (DMO projection + cutouts) is parameter-independent
  → run once per snapshot, shared across all runs; only stage 2 (generate+composite)
  varies. All 20 shared stage-1 dirs already exist under `bind_lightcone_tng` and
  are reused as-is.
- **Stage 2 orchestration:** `bind.inference.science.plan_generate` + `bind-science-plan`
  emit per-run `params.npy` + restart-safe disBatch task files (one
  `bind-paint-generate --params ...` per run×snapshot, **circular taper r200_factor=4**,
  fm_redshift_thermo). Counts: fiducial 20, 1P 3000, sobol 5120 tasks. Added
  `--params` override to `bind-paint-generate`. Submit via `run_science_generate.sh`.
- Output root `/mnt/home/mlee1/ceph/bind_science`; N_sobol=256 (user choices).

Still to build: 2c tSZ y-map LOS assembly (thermo currently only per-halo patches,
not composited — needed for 1a and to avoid keeping TBs of patches), Stage 3 stats
(Cl auto/cross, peaks, PDF/Betti/moments, halo scaling), Stage 4 emulator set.

---

## 2026-06-09 — `lightcone`: diagnosed the convergence-power "upturn" + added anti-aliasing capability

The BIND lightcone κ power spectrum showed a sharp high-ℓ upturn vs kappaTNG.
Traced it (snap096, TNG300) at the **projected-mass level, no raytracing**: summed
the 4 `composite_slab*.npz` into a full-box map and compared 2-D P(k) against the
truth hydro projection and the lightcone's own DMO. Findings:

- **BIND is correct.** `BIND/DMO_lc` (same-pipeline ratio) matches the true
  `hydro/DMO_truth` baryonic suppression to a few % across all scales.
- The upturn is a **raw-CIC aliasing artifact in the particle→grid projection**
  ([`pixelize_z_projection`](../src/bind/inference/pipeline.py)), present *identically*
  in DMO and BIND (common-mode → cancels in same-pipeline ratios). Not the
  generative model, not compositing, not the 4-vs-2 slab count. Deconvolution
  alone makes it worse (it's additive, not a suppressed window).
- Pixel scale was never the issue: generation (128px/6.25 Mpc/h) and composite
  (4198px/205 Mpc/h) are both 0.0488 Mpc/h; the 4198→4096 lensplane step is a
  crop, not a resample — generative fidelity untouched.

Added an opt-in fix (default off, no behaviour change): `pixelize_z_projection(
mas_correct=True)` does **interlacing + CIC deconvolution** (`interlace_combine_2d`,
`cic_window_2d` in `pipeline.py`; unit-validated to 0.4% out to 0.8·k_Ny on
synthetic data). Wired through stage 1 (`project_and_extract(mas_correct=)` /
`bind-paint-project --mas_correct`) which stores a separate **`dmo_aa`** map —
the raw `dmo` stays the model-condition source. Stage 2 / recomposite auto-use
`dmo_aa` as the composite background when present, so the lensplane inherits it
with no lensplane change. Validation tooling: `examples/lightcone_projection_check.py`
+ a cell in `examples/lightcone_diagnostics.ipynb`. **For the science (S(ℓ),
peaks, non-Gaussian stats): ratio against same-pipeline DMO and no fix is
needed; `--mas_correct` is only for absolute κ-vs-kappaTNG validation.**

---

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
