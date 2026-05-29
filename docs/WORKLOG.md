# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

---

## 2026-05-28 — `feature/thermo`: connect the SS-calibration Sobol notebook to the `scatter` framework

Wired `tsz_ss_calibration_sobol.ipynb` to the BIND **scatter** analysis machinery
(`analysis/2d`). Ported the pure-numpy modules `scatter/{__init__,sensitivity,
residual,obs_common}.py` onto this branch (no transitive deps). Three changes:

- **§3 plumbing** — replaced the notebook's inline `src_bootstrap`/`dcor` with the
  shared `scatter.sensitivity` estimators (SRC R² reproduces exactly: 0.557/0.697/
  0.644/0.696) and added `cv_r2_compare`. Decisive check is **linear vs GBM** (gap
  ≲0.05 ⇒ near-linear, SRC faithful); the GP underperforms only because a default
  30-D RBF is starved at n=256 — not evidence of nonlinearity.
- **§6.5 (new)** — diagnoses *why* the per-halo money plot is weak. Per-halo, the
  gas fraction is the best suppression predictor (ρ≈0.63 vs profile metric) and
  Δ_SS (ρ≈0.45) is largely subsumed: partial ρ(Δ_SS,S|f_gas) collapses 0.24→0.10
  while ρ(f_gas,S|Δ_SS) holds at 0.15. Δ_SS is a noisy shadow of gas content.
- **§6.6 (new)** — tests the scatter paper's "concentration is the master predictor"
  on thermo suppression and **falsifies it**: all DMO structural features (`c_core`,
  `r_half`, `q_DMO`, `q_DMO_in`, from new `extract_dmo_structure.py` → ceph
  `sobol_ss_cv/dmo_structure.npz`, aligned 1:1 with the cube) predict suppression
  weakly (|ρ|≲0.23); conditioning the money plot on `c_core` changes nothing
  (0.46≈0.45) and `c_core`⊥`f_gas` (ρ≈0.02). **Key result: two scatter regimes —
  mass-relation scatter is structure-inherited (scatter paper), thermo/suppression
  scatter is feedback-driven (this notebook).** This gives §7's pivot to a baryonic
  observable a mechanism, not just an assertion.

New figs: `ss_perhalo_predictors`, `ss_dmo_structure`. Each new/changed cell
verified against the real cube with the `torch3` venv; full-notebook re-exec
(to repopulate outputs) still TODO.

## 2026-05-28 — `feature/thermo`: Sobol calibration reframed as an observable-conditioned suppression prior

Added **Section 7** to `tsz_ss_calibration_sobol.ipynb`, pivoting the analysis
from the weak *per-halo* money-plot slope (median Spearman ρ≈0.24 — not what a
survey measures) to the **population statistic** WL actually constrains. Per
design, collapse the 1111 halos to median `f_gas`(<R200) and the Y–M amplitude
(`log c0`), then regress design-mean suppression on them: ρ≈0.52 (k≈10.4 Fourier)
to ρ≈0.85 (profile ratio) — **3–4× more predictive** than the per-halo law. BIND's
design-mean suppression **brackets van Daalen+20** TNG/EAGLE/BAHAMAS at k=10.
Deliverable `suppression_prior(obs)` → `P_hydro/P_DMO ± σ_pred≈0.10`
(`S ≈ 2.5·f_gas + 0.55`). New figs `ss_population_prior`, `ss_literature_bracket`.
Motivation: turn tSZ/X-ray observables into a prior on the dominant baryonic
systematic for Stage-IV WL. Caveats: IllustrisTNG-conditional; per-patch supp ≠
full-box P(k). Executed clean with the `torch3` kernel (nbconvert default
`python3` kernel lacks pandas — force `--kernel_name=torch3`).

## 2026-05-28 — `feature/thermo`: expanded Sobol calibration visual diagnostics

Extended `tsz_ss_calibration_sobol.ipynb` with alternative views of the same
Sobol design fit outputs to make interpretation faster and more robust:

- Added a **calibration phase diagram** (`alpha` vs `beta`, colored by
  `sigma_int`, marker size by `|Spearman|`) with extreme-design annotations.
- Added **tornado-style SRC plots** (top signed drivers + 16-84% CI whiskers)
  for each target (`alpha`, `beta`, `sigma_int`, `median_S`).
- Added a **PCA landscape map** of the 30-D feedback cube (PC1/PC2) colored by
  each calibration target to diagnose response smoothness/structure.
- Added a concise “what is the point” interpretation section plus a compact
  summary table (`median`, p16-p84, linear `R2`, significant-driver fraction).

All newly added code cells executed successfully and wrote:
`tsz_ss_sobol_figs/ss_phase_diagram.png`,
`tsz_ss_sobol_figs/ss_tornado_drivers.png`,
`tsz_ss_sobol_figs/ss_pca_landscape.png`.

## 2026-05-28 — `feature/thermo`: Sobol map of the SS-residual→suppression law

New experiment built on the `tsz_wl_calibration.ipynb` "money plot" (CV halos:
deviation from self-similar `Y200 = c0·M200^(5/3)`, `Δ_SS = log10(Y/Y_ss)`,
predicts matter-power suppression `P_hydro/P_DMO(k≈10.4)`; BIND α≈+0.17,
σ_int≈0.086). Exploits BIND's generate-anywhere capability: regenerate the **same
~1111 CV halos across a 256-pt Sobol grid in the 30 astro params** (cosmology
fixed at CV fiducial), refit the relation per grid point, and map which feedback
knobs tune the slope α / zero-point β / scatter σ_int.

- `sobol_ss_generation.py` — deterministic Sobol design (inverted through the
  model's normalized box via `norm_stats`, log-flag aware; cosmo idx {0,1,6,7,8}
  held at fiducial) + chunked GPU campaign reusing `test_suite` generation
  (`load_model_bundle`, `generate_halo_patches`). **Common random numbers**
  (`torch.manual_seed` before each design) isolate the param response. Reduces
  each generation to per-halo obs `[Y200,T,S,P,f_gas,m_gen,supp_k10,supp_prof]`
  (R200 aperture + Pylians `Pk_plane` at k≈10.4, matching the notebook). Saves
  sharded maps+obs to **ceph** (`/mnt/home/mlee1/ceph/sobol_ss_cv/`), resumable;
  `--reduce` → `cube.npz`. Flags: `--n_chunks/--chunk_id --fp16 --no_maps`.
- `run_sobol_ss.sh` — H100 array wrapper (`N_CHUNKS=16 sbatch --array=0-15`),
  user-submitted; design recomputed per task from `--seed` so no manifest/lock.
- `tsz_ss_calibration_sobol.ipynb` — reads `cube.npz`: per-design MLE fit
  (α,β,σ_int), "moving money-plot" panels at α/β extremes, and SRC + distance-
  correlation sensitivity (30 knobs × fit params) with bootstrap CIs.

Validation: applying the reduction+fit to the **existing** `fm_thermo_ema` CV run
**exactly reproduces the figure** (α=+0.174, σ_int=0.086). Smoke-tested 2 designs
end-to-end (k bin 10.432, sim_17 skipped for missing radii). Full 256-pt run is
the user's H100 array; a 32-pt obs-only preview (`…/sobol_ss_cv_dev/`, seed 12345)
exercises the notebook (SRC R² only meaningful at the full design size).

## 2026-05-28 — `feature/thermo`: three tSZ/ICM science notebooks

Built three single-notebook science projects exploiting the self-consistent
mass+thermo emulator (all read `fm_thermo_ema` suite outputs; same per-halo
machinery as `paper_figures_thermo.ipynb`):

- `tsz_wl_calibration.ipynb` — **tSZ as a calibrator of baryonic feedback for
  weak lensing.** Per-halo matter-power suppression `P_hydro/P_DMO` and profile
  ratio `rho_hydro/rho_DMO` (total matter = DM+Gas+Stars; DMO from `dmo_fullbox`;
  total mass conserved to ~0.1-1%, so the ratio isolates redistribution) vs the
  R200-aperture observables (Y/T/P/f_gas), + full-box composite cross-check.
  Preliminary: **pressure is the best single predictor** of suppression
  (partial rho≈+0.43 at fixed M); f_gas weak (≈0.08); individual-halo
  predictability modest (Y+T+M R²≈0.16) — signal is population-level; BIND gives
  slightly *tighter* relations than truth (smoothing).
- `tsz_ym_mass_bias.ipynb` — **Y-M relation & cluster mass bias.** Y-M slope
  ~1.77-1.94 (CV/SB35), above self-similar 5/3. A CV-calibrated Y-M applied
  across SB35 implies a **~22.6% (1σ) mass bias** (range -36..+71%), driven by
  A_AGN1; **BIND predicts the per-sim bias with rho=0.94** (forward-model use).
- `icm_entropy_pressure_feedback.ipynb` — **ICM entropy/pressure thermometer.**
  Central entropy K0 recovered well by BIND (bias +0.02, scatter 0.10 dex,
  rho=0.90 — better than the pixel-level entropy-smoothing caveat implied);
  Pnorm noisier (0.26 dex, rho=0.80). A_AGN1/A_AGN2 push K0 in *opposite*
  directions (rho -0.50 / +0.54); BIND captures it.

All three are build-only (helpers + data contract smoke-tested on subsets; not
run end-to-end over the full suite, and findings above are preliminary). Aperture
= circular R200 (catalog `radii` in kpc/h). Robustness fixes: skip catalogs with
0 halos (SB35_665) or missing `radii` (CV/sim_17).

## 2026-05-28 — `feature/thermo`: thermo paper-figures notebook

Added `paper_figures_thermo.ipynb`, the thermo analogue of
`paper_figures.ipynb`. Consumes the `fm_thermo_ema` test-suite outputs (the
EMA epoch-64 eval ran across all suites: CV 25/27, 1P 141, SB35 102), reading
per-halo truth thermo from `truth_thermo_patches.npz` and the 7-channel
`generated_halos.npz` (`[DM,Gas,Stars,Y,T,S,P]`, physical units on disk).

- Per-halo reduction uses a circular **R200 aperture** (`halo_catalog['radii']`
  is in **kpc/h**; R200 ≈ 7–21 px in the 128-px / 6.25 Mpc/h patch). Integrated
  Compton-Y sums `y·A_pix`; T/S/P are gas-mass-weighted means.
- Figures: T1 field showcase, T2 per-pixel dex bias/scatter, T3 pixel PDF/KS,
  T4 radial profiles by mass bin, **T5 scaling relations Y/T/S/P–M** + Y–T &
  recovery, **T6 Spearman correlation matrices + ideal-gas P∝nT proxy**,
  **T7 35-D parameter response (truth vs BIND)**, **T8 1P butterfly maps** +
  response-amplitude bars, and a dex-metric scorecard from `summary.json`.
- CV p14 override (CAMELS bug) applied to CV rows of the master halo table only.
- Built but not executed this session; data-contract + helpers smoke-tested
  against real CV/1P outputs.

## 2026-05-27 — `feature/thermo`: emulate the 4 gas thermo fields

New branch `feature/thermo`. The training files (no-lowmass) now carry four
extra gas-derived maps — `compton_y`, `temperature`, `entropy`, `pressure`
(added by `make_train_data/add_gas_thermo_maps.py`). Extended the emulator to
generate them jointly with the mass fields via a single flow-matching model
(extended output channels), gated behind `--predict_thermo`. Default
3-channel and `--stars_two_head` paths are untouched.

- **Normalization** (`data.py`): thermo fields are strictly positive with ~4–9
  dex of range, so `log10(1+x)` collapses sub-unity fields (compton_y/pressure)
  to ~0. Use per-channel `log10(max(x, floor))` standardization instead
  (`thermo_forward`/`thermo_inverse`; floor = 0.1th pct, zero-safe). New
  back-compatible `NormStats` fields (`predict_thermo`, `thermo_mean/std/floor`);
  `AstroDataset` appends `N_THERMO` channels after the mass target →
  `[DM, Gas, Stars(/occ,dens), Y, T, S, P]`.
- **model.py**: decoupled the stars-loss weighting from `out_ch==4` (now keys
  off an explicit `stars_two_head` flag) so appended thermo channels (weight 1)
  don't break two-head stars. UNet/FM are otherwise channel-agnostic.
- **train.py**: `--predict_thermo` (requires `fm` + large-scale path; cube data
  has no thermo). `out_ch = (4 if two_head else 3) + 4`.
- **Evaluation** (`test_suite/`): inference denormalizes + returns thermo in the
  trailing channels; thermo is evaluated **per-halo, not composited** (compton_y
  is extensive; T/P/S are intensive mass-weighted means — mass conservation
  N/A). Ported the exact gas-thermo recipe into `pipeline.py`
  (`project_thermo_fullbox`, axis-aligned to match the suite cutout frame) to
  reconstruct per-halo truth thermo patches; `runner.py` reports per-channel
  log10 bias/scatter (`thermo_metrics`).
- **Data gap**: ~0.1% of no-lowmass files lack thermo maps (the thermo job
  skipped some sims). `--predict_thermo` skips them — `compute_norm_stats`
  ignores them and `AstroDataset` resamples a random index on a miss (misses
  cluster by sim). Without this the first `fm_thermo` job died instantly on a
  `KeyError: compton_y`.
- **Validated**: norm-stats save/load + back-compat; dataset emits 7 ch;
  forward/inverse round-trip <1e-4; FM loss/grad for single- and two-head+thermo;
  truth-thermo port reproduces stored `sim_0_halo_0_rot_0` maps to float32
  round-off (median rel err ~1e-6).

Launch: `python train.py --predict_thermo --run_name fm_thermo` (add
`--stars_two_head` to keep the stellar split).

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
