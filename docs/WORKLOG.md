# Work log

Reverse-chronological log of notable sessions: what changed, why, and decisions
worth remembering. Newest entries on top. Keep entries short — link commits and
files rather than restating diffs. (Maintained by Claude Code; see CLAUDE.md.)

---

## 2026-06-04 — Outpainting the inter-halo background (branch `feature/outpainting`)

New investigation: BIND's composite is patchy (baryons only inside pasted halos). Added
`examples/outpaint_walkthrough.ipynb` exploring how to fill the inter-halo background, validated
against truth on CV + held-out SB35.

- **Recipe (train-free):** fill background with `f_b·smooth(DMO)` for Gas, `(1-f_b)·DMO` for DM,
  ~0 for Stars. `f_b=Ω_b/Ω_m` is a params input (mass identity, not a fit); σ≈2.5 px is the physical
  IGM gas-filtering scale (broad plateau — no calibration needed). Gas background `logRMSE 8.8→0.11`.
- **Generalizes off CV:** on 18 held-out SB35 sims, `corr(f_b, true bg gas/DMO)≈1.0`; M2 holds
  `logRMSE≈0.11` blind. Only residual: weak σ–feedback dependence (`MaxSfrTimescale`/wind, |r|~0.5).
- **Total-field P(k) (paper fig5b style + outpaint):** outpaint is a *wash* on the total field
  (DM-dominated, already DMO-filled); its real value is the gas field. Beats gas-extrapolation (3a)
  and kriging (3c), which ignore the DMO map.
- **Small-scale (high-k) deficit diagnosed:** the >1 truth upturn is **stellar condensation** in
  central galaxies, which BIND under-delivers. Per-halo stellar *mass* is right (1.01×); 24% of
  stellar mass is in **sub-threshold (<1e13) halos BIND never paints** (training threshold is 1e13).
  Sampler ruled out (`run_sampler_diag.sh`/`sampler_diag.py`, A100): n_steps converged by ~50, model
  is properly stochastic (single sample 0.82 vs mean-of-8 0.45 at k>40 ⇒ use single realizations).
  High-k power lever = **stellar concentration in painted halos** (per-patch stars ~0.79), *not*
  coverage. **Next time:** test post-hoc stellar sharpening (cheap) before a spectral-loss retrain.

## 2026-06-03 — Two-stage paint (CPU/MPI project → GPU generate); fixes TNG-box OOM

`run_paint_tng.sh` (one-shot `bind.paint` on one A100 node) OOMed: the box load
path (`io_gadget.read_dmo_particles`) concatenates **all** ~33 GB of TNG300-Dark
particle positions on one process, then `Simulation.project()` masks them per
z-slab. Split painting into two SLURM jobs so neither holds the full particle set:

- **Stage 1 — `bind.cli.paint_project` (`run_paint_tng_project.sh`, CPU/MPI).**
  New `bind.inference.paint_stages.project_and_extract`. Each MPI rank reads only
  `files[rank::size]` snapshot chunks, **streams them one at a time** into the
  small z-slab maps (~70 MB/slab), then partials are `MPI.SUM`-reduced onto rank 0,
  which reads the FoF catalog, extracts per-halo DMO cutouts, and writes
  `stage1_slab{NN}.npz` + `stage1_manifest.json`. Streaming alone fixes the OOM
  (peak ≈ one chunk + slab maps); MPI just adds multi-node speed. Runs serially
  with 1 task / no mpi4py.
- **Stage 2 — `bind.cli.paint_generate` (`run_paint_tng_generate.sh`, GPU).**
  `paint_stages.generate_from_stage1` loads the intermediate, runs the sampler,
  composites, and writes the same `composite_slab{NN}.npz`/`summary.json` as
  `bind.paint`. Light on memory; no particle I/O.

**Gotcha (cost real time if forgotten):** Pylians `MASL.MA` is **not** additive
onto a pre-filled field — calling it repeatedly to accumulate chunks silently
*loses mass* (measured ~25%). Deposit each chunk into a fresh zero field and sum
with numpy (`_accumulate_chunk_into_slabs` reuses `_project_zslabs` per chunk).
Verified: streaming == production one-shot to float precision, mass conserved.

**Multiscale physical-scale bug fixed (the important one — correctness, not just a
crash):** the network's 4 input channels are fixed *physical* scales
`[6.25,12.5,25,50] Mpc/h` (training: `data_generation/process_simulations2_cpu.py`
`extract_multiscale_cutouts`, `scales_mpc`). But inference `pipeline.extract_multiscale`
used fixed *pixel* scales `[128,256,512,full_res]` — so its 4th (largest) context
channel was the **whole box**: fine at the native 50 Mpc/h / npix=1024 grid, but on
the 205 Mpc/h TNG box it became a 205 Mpc/h window (OOD for the UNet) **and** crashed
(`npix=4198` not a multiple of 128 → reshape `ValueError`). The first real TNG stage-1
run hit exactly this — *after* a fully successful MPI projection (64 ranks, 92.7s, 2951
halos at M>1e13, confirming mpi4py works). Fix: `extract_multiscale` now takes
`mpc_per_pix` and cuts the `MULTISCALE_MPC=(6.25,12.5,25,50)` windows (capped at the
box), resampled to 128² via `_downsample_square` (exact block-mean when divisible —
**CAMELS box=50/npix=1024 bit-for-bit unchanged** — else area-avg down / bilinear up).
Network inputs stay 128² at the trained scales; only the slab *background* spans the
full box, which is correct (that's where patches get pasted). Both
`extract_halo_cutouts` callers pass `mpc_per_pix=box_size/npix`.

**Stage 3 — re-composite without regenerating (`bind-paint-recomposite` /
`recomposite_slab` / `recomposite_from_saved`).** The sampler output is already
saved (`generated_patches` per `composite_slab*.npz`), so re-blending with new
`taper_frac` / `r200_factor` / `patch_mass_match` re-runs only
`build_bind_composite` — **no GPU**. `recomposite_slab(stage1_npz, generated_npz,
**settings)` returns the bundle for interactive notebook sweeps; verified
idempotent (same settings reproduce the saved composite to max|diff|=0) and
mass-conserving under a circular `r200_factor` paste. Generate now also writes
`condition_sums` (per-halo cutout mass) so future composites are self-recompositable.
Shared `_save_composite_slab` helper used by both generate + recomposite.

**Validated on the real TNG300-Dark box (snap 099).** Stage 1: mass conserved
*exactly* (projected = `pmass × 2500³` to ratio 1.0000), Ωm=0.3090 from the maps
(fiducial 0.3089), 2951 halos M200c 1.0e13–1.0e15 (154 >1e14), R200↔M200c
consistent, cutouts centered on density peaks (98–99%), multiscale context at the
correct [6.25,12.5,25,50] Mpc/h. Stage 2 (fm_two_head, 50 steps, ~5 s/16-halo
batch on one A100): composite mass-conserved per slab, DM_hydro reproduces the
DMO web, Gas/Stars painted only in halos, f_b≈0.10 (feedback-depleted, below
cosmic 0.157). Notebook `examples/paint_tng_results.ipynb` does these checks +
figures; stage-2 cells are race-safe (`safe_load`) so they populate as slabs land.
§7 computes the **matter-power suppression**: sum the z-slabs (masses additive) →
full-box DMO + painted (DM+Gas+Stars) grids → 2D `Pk_plane` ratio (capped at
`k_Nyq=π·npix/L`). Textbook curve: S→1 for k<2, knee at k~3, **~17–20% suppression
(S≈0.80–0.83) by k~20–40 h/Mpc** — consistent with TNG AGN feedback (from the
M>1e13 painted population). Runs in the `bind_env` Jupyter kernel
(`python -m ipykernel install --user --name bind_env`).

**Env:** runs in `~/venvs/BIND_env` — a Python-3.11 `--system-site-packages` venv
built from the module python view (inherits the view's numpy 2.2.4 / torch 2.6
cuda12.5 / h5py; only `bind` + Pylians are pip-installed locally). mpi4py comes
from the `python-mpi` module (no build), matching the view's numpy, so stage 1
loads `module load python openmpi python-mpi` (same modules at create + run time
so PYTHONPATH exposes mpi4py); fallback `module load openmpi && pip install mpi4py`
into BIND_env. Stage 2 just activates BIND_env (torch bundles CUDA). Submit gated:
`jid=$(sbatch --parse run_paint_tng_project.sh);
sbatch --dependency=afterok:$jid run_paint_tng_generate.sh`.

---

## 2026-06-02 — New branch `analysis/pk-decomposition`: field-level halo-masking decomposition of P(k) suppression

Started the "which halos drive the matter-power suppression?" experiment (the
natural next paper after the `scaling_relations` figures). Field-level halo
masking: for each of the 256 Sobol designs, re-composite halo subsets (3 mass
decades × 3 gas-fraction-at-fixed-mass terciles) back into the 50 Mpc/h box and
measure the partial S(k) — reusing the precomputed BIND patches
(`sobol_ss_cv/maps/`, **no re-emulation, no hydro**). New CPU-only tools:
- `tools/partial_supp_sobol.py` — array-ready/resumable; per design stores
  `S_full`, `S_sub` (subset-only paste), `S_loo` (leave-one-out). `full_pk`
  reproduces `box_supp_sobol` `S_true` exactly.
- `tools/fig_partial_supp.py` — variance attribution + 4-panel figure.
- `run_partial_supp.sh` — 16-way CPU SLURM array (**user submits**; org policy).

**Methodology (validated on a 32-design thin slice).** Subset-only paste
OVER-counts a subset's contribution (~1.8×: mass-renorm + patch-overlap
cross-term); LOO UNDER-counts (~0.55×). Their mean — the 2-bracket Shapley
contribution `c_s = ½[(S_sub−1)+(S_full−S_loo)]` — is ~additive (Σ_s c_s ≈
S_full−1, slope 1.11, Σ shares ≈ 115%), giving defensible **absolute** Var[S]
shares via `share_s = Cov(c_s, ΔS_full)/Var(ΔS_full)`. A naive LMG/regression
split washes out to uniform ~11% (the 9 subset partials are collinear — all driven
by the same knobs); use the covariance/Shapley partition, not LMG.

**Prototype result** (k=10, n=32; full 256 pending user Slurm): group decade
**[13,13.5) ≈ 48% of Var[S]**, [13.5,14) ≈ 30%, clusters [14,15) ≈ 21% (the last
from only **51 halos** — high per-halo weight; ~0 mean contribution though). At
fixed mass, **gas-rich halos dominate the variance** (~43–49%) over gas-poor
(~25–30%) — gas content, not just mass, structures Var[S] (direction 2 confirmed).
High WindEnergy *deepens* the group-decade contribution (group dominance
strengthens, doesn't shift to clusters — direction 1). Stable k=3↔k=10. Direction
3 (beyond-R_vir redistribution radius vs θ) not yet built.

**Publication-grade notebook** `pk_suppression_decomposition.ipynb` (Question →
Methods → Results → Discussion, **11 figures**, executed 0 errors on the prototype;
auto-upgrades to the full cache when present). Built to disarm a skeptic: Fig 1
motivates the variance + **honestly places the fiducial** (it is the *8th-pct
strong-feedback tail*, not the floor — fixed a misleading "typical member" framing
after human pushback); Fig 2 pure validation (masking composite == independent box
pipeline to **machine precision**, all designs & k); Fig 4 the Shapley-additivity
credibility plot; Fig 5 headline heatmap; Figs 6–7,9 results w/ bootstrap CIs;
**Fig 8 the param→halo sensitivity map** + **Fig 10 the feedback-strength-axis
robustness** (the 30-D treatment: ~4 of 30 params drive S(k), group/gas-rich
dominance holds weak→strong — replaced the misleading single-knob split); Fig 11 the
actionable synthesis (group $f_{\rm gas}$ most constrains the WL baryonic prior); §5
referee Q&A table. Branch carries the uncommitted `scaling_relations` fixes too;
nothing committed yet.

**Full 256-design run + mechanism check (2026-06-02 eve).** Ran the campaign
(`run_partial_supp.sh` → `--reduce` → `partial_supp_sobol.npz`, 256 designs, clean);
notebook auto-upgraded. Results **robust** vs the 32-pt prototype: by mass
57/35/21% of Var[S(k=10)] (groups/[13.5,14]/clusters), by gas 32/36/44%
(poor/mid/rich), additivity slope 1.12, drivers IMFslope/BHRadEff/WindEnergy
($R^2{\sim}0.7$). **Honest mechanism check (Fig 11) overturned the naive
"swing-voter" reading I'd floated:** at the halo level the per-patch suppression
swing is *uncorrelated* with gas ($\rho{=}0.00$, even at fixed mass). The gas-rich
excess is a **reservoir/magnitude** effect — all subsets are ~coherent with the total
($r{\sim}0.99$) so variance tracks contribution *magnitude* ($\rho{=}0.97$), and it is
**group-scale only** (gas rich/poor var ratio 2.68/1.72/0.89). The genuinely
variance-specific result is **clusters: $\sim$0 mean contribution but $\sim$21% of the
variance** (6× per-halo). Re-framed the Discussion accordingly + fixed a 3× scale bug
in Fig 6B per-halo. Figures `paper_figures/pk_decomp_fig{1..12}_*.png`.

## 2026-06-02 — scaling_relations.ipynb: fixed silent halo-ordering bug + unified into one 4-act story (assembly dropped)

Two-part pass on `scaling_relations.ipynb` (still on `main`, the human's active
notebook; per convention belongs on a topic branch). **(1) Fixed a silent
correctness bug:** the CV scaling-relation loader (cell 10) iterated sims in
*numeric* order (`sim_0,sim_1,sim_2,…`) but the Sobol cube stores its 1111 halos
in *lexicographic* dir order (`sim_0,sim_1,sim_10,…`; sim_17 dropped/no-radii,
sim_27 absent). Both total 1111, so the old `assert len==1111` passed while
per-halo identity was scrambled — every "same halos across designs" result was
wrong. Loader now iterates `sorted(CV_ROOT.iterdir())` (== cube order) and a hard
`np.allclose(_cube['M200'],halo_mass)` assert enforces it (mirrors
`tools/box_supp_sobol.py`). **(2) Made it run + tell one story:** restored the
missing Sobol-cube loader cell (was a NameError cascade:
`_cube/D_norm/ASTRO_NAMES/OBS_NAMES/N_HALO/SUPP/spearmanr`), defined `hi_m/lo_m`,
trimmed the dangling assembly/`S_massonly` NOTE. Rewrote the narrative as a 4-act
arc on two pillars (same halos ⇒ cosmic variance differenced away; field-level
emulator ⇒ paste→P(k)): I Fidelity (Figs 1–2) → II structured scatter / two
populations (Fig 3) → III controlled feedback experiment (Figs 4–5: MI of the 30
knobs on the fiducially-classified gas-rich/poor tails) → IV power spectrum (Fig 6
design fan + new **Fig 7** capstone). Fig 7 (`perhalo_box_synthesis`) shows the
*same* knobs (IMFslope, WindEnergy, BHRadEff) drive both per-halo gas content and
box S(k=10), and design-mean log f_gas predicts box S(k=10) at ρ=+0.54 (p~1e-20).
**Assembly (DMO history) dropped per the human** — this supersedes the prior
entry's assembly "Fig 4"; that analysis is not in the current notebook.
Re-executed end-to-end (torch3 kernel + gcc-13 libstdc++ on `LD_LIBRARY_PATH`),
0 errors, 7 figures.

## 2026-06-02 — scaling_relations.ipynb: "assembly as a hidden 2nd parameter of feedback" (new Fig 4)

Mined the Sobol feedback cube (`/mnt/home/mlee1/ceph/sobol_ss_cv/`: `cube.npz`
256 designs × 1111 CV halos × 8 obs, common-random-noise; `pk_supp_extra.npz`
supp at k=5/10/band; `assembly_table.npz` 3D DMO assembly c_V/λ/σ_v/rhalf/z_form)
+ assembly histories. Added Figure 4 to `scaling_relations.ipynb` (on `main` —
the notebook the human is actively editing; flagged that per convention this
belongs on a topic branch).

Result (executed, fig written to `paper_figures/assembly_feedback_susceptibility.{pdf,png}`):
per-halo OLS response of P_hydro/P_DMO to the 30 astro knobs (median R²=0.68);
dominant drivers BHRadiativeEfficiency / IMFslope / WindEnergy. **At fixed mass
the *mean* suppression is ~assembly-independent (|ρ|≲0.07), but the
*susceptibility* (response derivative) is significantly set by assembly**: early-
forming/concentrated/compact halos resist feedback — z_form ρ=−0.15 (p~1e-6),
c_V ρ=−0.15 (p~1e-7), rhalf ρ=+0.16 (p~1e-7); k=5 even stronger (ρ=−0.26,
p~1e-18). Mass-matched tercile split: high-conc ~10% less susceptible in 8/9
bins. Framed honestly as hypothesis-grade (emulator + 2D + CV cosmology; modest
ρ); motivates a direct 3D hydro/DMO P(k) split-by-formation-time test. The ICM-
observable analogue already exists on `analysis/tsz-icm`
(`assembly_feedback_susceptibility.ipynb`); this is the matter-power / weak-
lensing version.

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
