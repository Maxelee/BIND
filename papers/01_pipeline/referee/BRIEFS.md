# Referee-response agent briefs — 2026-08-12

Deploy-grade briefs for the five referee comments (see `../REFEREE_PLAN.md` for
the triage). Six agents; models per task. Deliverables land in this directory
(`papers/01_pipeline/referee/`) as `*_results.md` / `*_memo.md`, figures as NEW
files in `/mnt/home/mlee1/BIND/imgs/` (paper figure dir — never overwrite
existing names). No agent edits `main.tex`; all draft prose goes in memos.

Shared guardrails (verbatim in every prompt):
- Never execute Slurm commands except read-only `squeue -u mlee1`/`sacct`/`sinfo`/
  `seff`; never sbatch/srun/scancel.
- Never run recursive searches rooted at /mnt/home, /mnt/ceph, /mnt/ceph/users,
  or /mnt/sw; scope to the named dirs, `timeout 60`, bounded depth.
- Python = `/mnt/home/mlee1/venvs/BIND_env/bin/python3`, run from
  `papers/01_pipeline/` when importing its local modules.
- Campaign trees (`bind_lightcone_tng`, `bind_science`, `bind_sb35`,
  `tng_full_validation`) are READ-ONLY; bulk intermediates →
  `/mnt/home/mlee1/ceph/referee_work/<tag>/`.
- No file named `report*.md` (harness-blocked); branch `papers`, no commits.

## Agent 1 — R1-fullhydro (opus): the three-rung ladder figure

Data (first 50 realizations seed-paired everywhere, ladder 1992+7r):
BIND fid = `ceph/bind_lightcone_tng/` (550 reals, map cubes at root);
pasted truth = `ceph/bind_science/runs/truth/run_0000/` (550);
DMO = `ceph/bind_science/runs/dmo/run_0000/`;
full hydro = `ceph/tng_full_validation/runs/hydro_full/run_0000/` (50);
diffuse variant = `.../runs/diffuse/run_0000/`.
Method: recompute all statistics from the raw map cubes with one code path
(`field_cache.py` conventions; released `Cl_kappa_y.npz`/`Cl_tau.npz` caches are
pre-xpkfix and must NOT be used; bind run_0000 `nongaussian_stats.npz` MFs are
stale). Cross-check against `tng_full_validation/analysis/tng_full_validation_summary.md`
band ratios (κ pasted/full 1.003/1.018/1.152; Cl_yy ≤2%; mean-y 0.889→0.719)
before building the figure — stop and diagnose if they don't reproduce.
Covariance: truth 550 reals, Hartlap, ×25/18000 (main.tex Eqs. 8–10).
Deliver: `imgs/fig06b_full_hydro.{png,pdf}` (BIND vs pasted vs full-hydro;
κ statistics + Cl_yy/Cl_ττ with pasted+diffuse; LSST-Y10 bands; 0.8ℓ_Ny marked),
`referee/R1_fullhydro_results.md` (numbers, χ², draft LaTeX ¶ replacing the
"~90% quoted from Lee-2026a" sentence + caption + caveat-1 pairing).

## Agent 2 — R2-noisy (opus): survey-realism twins + detectability

Data: `ceph/bind_sb35/emulator_dataset_nu05n.npz` + `nu05n_shards/` (272 shards;
fid/truth 550, dmo, 256 Sobol; built 2026-08-05); noiseless counterparts
`emulator_dataset_nu05.npz` (+ `_xpkfix`). Conventions: `noisy_grid.py`
docstring is binding (L22 recipe: σ_e=0.26, n_gal=27, noise-before-1′-smoothing,
θ_G = 1/e radius). First: `build_noisy_cache.py --verify`.
Deliver: (a) `imgs/fig05n_field_validation_noisy.{png,pdf}` — fig05 twin, BIND vs
truth residuals under the noisy 550-real LSST-Y10-scaled covariance; (b)
`imgs/fig23n_s3_opener_noisy.{png,pdf}` — S(ℓ) Sobol spread vs noisy LSST/Euclid
precision; (c) `referee/R2_noisy_results.md`: per-statistic table — validation χ²
(noisy vs noiseless) AND response detectability (fraction of 256 nodes >3σ/5σ
from fiducial per statistic under noise; span/σ ratios) — plus drafted
replacement sentences for every detectability claim in main.tex (quote line
numbers; read-only) and the SSC/systematics caveat sentence.

## Agent 3 — R3a-prior (sonnet): surface the across-the-prior validation

Mission: pull the held-out SB35-test validation content of Lee-2026b
(arXiv:2603.11815 — fetch abstract + validation sections; also local
`examples/paper_figures.ipynb` and any eval outputs discoverable under
`~/ceph` top level, one-level ls only) and `examples/enhancement_closure.py`
(+ any results it left). Deliver `referee/R3a_prior_validation_memo.md`: a
drafted §3 closing subsection (¶-level LaTeX) that states the across-the-prior
error budget with concrete numbers (masses / profiles / S(k) closure at held-out
parameter locations, N of held-out sims), a recommendation on whether a compact
imported figure is warranted (spec it if yes, don't build), and provenance for
every number.

## Agent 4 — R3b-texture (opus, GPU): N-sample texture experiment

Question: is the 5–10% gas-spectrum texture systematic single-sample stochastic
(shrinks ∝1/N under sample averaging) and does it grow with feedback strength?
Machinery: `src/bind/cli/{extract_halos,generate_halos,paint_generate,paint_recomposite}.py`
+ `ceph/bind_lightcone_tng/snap_096/{stage1,composite_slab*.npz,summary.json}`
(fiducial conditioning + canonical paint); full-hydro reference slabs
`ceph/tng_full_validation/composites/snap_096/`; pasted-truth slabs need
recomposite from patches (truth run stores patches only).
Stages: (0) recon the per-slab pipeline + seed convention; (1) provenance smoke:
re-generate a few snap_096 halos with the CANONICAL seeds → must reproduce the
stored composite; then fresh seeds; (2) fiducial snap_096, all 4 slabs, N=8
fresh-seed paints (~23k patch generations) on the LOCAL V100S (`--device cuda`,
no Slurm), budget ≤3 h GPU — reduce N if throughput demands; (3) plane-level
Cl/P(k) of gas/y/τ: single-sample vs mean-of-N vs truth → stochastic excess
scaling vs 1/N + residual floor; (4) if GPU budget remains: repeat at the
max-VarWindVelFactor Sobol node (no truth needed — single-vs-averaged excess
only); (5) `referee/R3b_texture_results.md` + figures (referee/figs/) + drafted
rewrite of the §3 immunity ¶. Outputs → `ceph/referee_work/texture/`.
Fallback if the pipeline can't run locally: deliver the runnable driver +
runbook + exact blockers instead. Hard cap ~4 h total.

## Agent 5 — R4-cosmo (sonnet, web): the quantitative fixed-cosmology caveat

Mission: (a) find and VERIFY the exact Elbers et al. paper on feedback–cosmology
non-factorizability of the matter-power baryon suppression (referee: "ξ²
scaling"; likely FLAMINGO-adjacent; currently absent from references.bib — do
not fabricate; if the scaling lives elsewhere, say so); (b) extract the
quantitative coupling and compute a bound: induced error on S(ℓ)≈0.9-level
suppression across a Stage-IV-width cosmology posterior; (c) scoped-grep the
`papers/` tree for the prior ξ² proposal (Paper-3 planning); (d) deliver
`referee/R4_cosmology_caveat_memo.md`: drafted caveat-3 ¶ (states the
cosmology-conditioned-painter asymmetry — SB35 varies 5 cosmological params, the
painter conditions on them; substrate/atlas/latent-map M are TNG300-cosmology —
plus the numeric bound), one §outlook sentence, and the verified bibtex entry.

## Agent 6 — R5-latents (opus): latent-circularity noise floor + rebuttal prep

Question: could λ (measured from the same generated snap_096 composites the
statistics come from — confirmed: `build_atlas_200c.py` reads them) absorb
realization noise and inflate CV R² 0.96-vs-0.6?
Inputs: `family_basis_all.py` (latent measurement + bundle build; read for the
sobol atlas input paths), `family_model.py`, `figs_preview/family_model_bundle.npz`,
`agnostic_lambda_search.py` + `audits/agnostic_lambda_search_obs_2026-08-12b.log`,
atlas caches `ceph/bind_science/halo_atlas/` (fid + truth snap096 + sobol rows).
Tasks: (i) per-node halo bootstrap of the 8 latents → σ_boot(λ) vs across-design
spread; (ii) R²-inflation bound: analytic attenuation + Monte-Carlo λ-jitter
refit under the exact shipped CV protocol (rng(1) folds — reuse the code path);
(iii) truth-atlas cross-check: λ_fid measured from hydro-pasted TRUTH halos
(zero generative noise) → FamilyModel predictions vs the BIND-λ predictions vs
measured fiducial statistics; (iv) leg-1 runbook: fresh-seed snap_096 repaint at
all 256 nodes — exact commands via the paint CLIs, seed-offset convention, cost
estimate (GPU smoke ≤15 min ONLY if nvidia-smi shows the GPU free — Agent 4 has
priority). Deliver `referee/R5_latent_noise_results.md` + drafted §5.4 ¶ with
the bound, + `referee/R5_leg1_runbook.md`.

## Monitoring

All six run in background in parallel; only Agent 4 (and optionally 6's smoke)
touches the GPU. On completion each is verified: deliverable files exist, imgs/
additions are new names only, numbers carry provenance. Follow-ups via
SendMessage to the same agent if a deliverable is deficient.
