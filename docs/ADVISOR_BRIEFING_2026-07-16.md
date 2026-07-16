# Advisor briefing — 2026-07-16

One-page-per-section prep for the 1-hour meeting. Everything below is committed;
branch/commit references at the bottom. Deep detail per session: `docs/WORKLOG.md`.

## TL;DR (the 30-second version)

Since June 10 the project went from "BIND paints single boxes" to a **full weak-lensing +
SZ lightcone program**: BIND now paints multi-snapshot TNG300 lightcones (κ, y, τ maps),
validated against the true TNG300 hydro lightcone, and was run over a **256-point Sobol
sweep of all 30 astrophysical parameters** (the `bind_sb35` suite — same halos, 20
snapshots, ray-traced). On top of that suite sit five science threads and two emulators.
**Two results are paper-grade:** (1) exactly **2 baryon-template nuisance parameters are
necessary and sufficient** to remove the baryonic S8 bias for LSST-Y10 cosmic shear, and
(2) **no point in TNG's 30-dim feedback space reaches the eROSITA strong-feedback band**
— the WL/kSZ/X-ray "strong feedback" tension is model-space, not parameter-space.

## The foundation (everything depends on this)

**Pipeline** (`lightcone` branch, `src/bind/`): multi-snapshot painting → lens planes →
lux multi-plane ray tracing → κ / Compton-y / τ (kSZ, FRB-DM) maps, plus a *truth*
lightcone from TNG300 hydro for apples-to-apples and a paired DMO trace.
**Validation:** BIND ≈ TNG300 hydro for WL and tSZ statistics at fiducial; the halo-mass
response R(ν) tracks **f_gas·T** (thermal energy, not gas fraction alone).
Show: `examples/fiducial_lightcone_stats.ipynb`.

**The SB35 Sobol suite** (the data engine for everything): 256 runs × 20 snapshots,
*identical halos*, 30-dim astro Sobol sweep; ray-traced κ/y per run; per-halo aperture
atlas of **8.6M (θ, halo-observable) rows**; preempt-safe MPI cache machinery.

Dependency map:

```
src/bind (pipeline, emulators)          [lightcone]
   └── fiducial + truth lightcone validation        [lightcone]
         └── SB35 Sobol suite (256 runs, atlases)   [analysis/sobol-sb35]
               ├── transfer functions → cosmo-bias capstone → LSST forecast   [analysis/wl-cosmo-bias]
               ├── feedback latents + SBI (WL & per-halo)                     [analysis/sobol-sb35]
               ├── kSZ / DESI×ACT / eROSITA paper (P4)                        [analysis/ksz-desi-act]
               ├── stat emulator (bind.emulator) + field emulator (bind.wlemu) [lightcone]
               └── 35-dim cosmology rescaling (feasibility done)              [feature/cosmo-rescale]
   ├── WL×tSZ bridge / Fisher / real-data shear×y (fiducial-era)  [analysis/wl-tsz-bridge]
   └── WL anisotropy paper draft                                  [analysis/wl-anisotropy]
```

## The five science threads

### 1. Cosmology bias from baryons — **paper-grade** (`analysis/wl-cosmo-bias`)
The arc: transfer-function ensemble S(ℓ, z_s) over the Sobol suite (paired DMO trace
cancels cosmic variance, r≈0.98; suite brackets both suppression *and* enhancement —
59% of runs enhance at ℓ~10³) → per-run effective (ΔS8, ΔΩm) bias → **capstone**:
- Ignoring baryons: up to **6σ** S8 bias at ℓ_max=5000 (LSST-Y10).
- One nuisance template: insufficient (0.67σ residual). **Two: <0.08σ**, validated held-out.
- Cost: ~3× the σ_S8 floor; BCM-style N≥6 models are pure extra cost.
- Independent corroboration: Lin+2026 finds the same 2-D structure.
- Robertson+2026 reproduction with BIND: full emcee+CCL likelihood, Y1 with N=2 →
  S8 bias −0.37σ at 1.11× cost; beyond-2pt statistics live on a ≤2-3D manifold,
  Minkowski functionals/PDF complementary to κκ.
Show: `examples/cosmo_bias_capstone.py` figures, `examples/lightcone_shear_forecast.py` posteriors.

### 2. Feedback latent space + SBI — **the ML story** (`analysis/sobol-sb35`)
- WL suppression is **2-dimensional** (PCA 98.7% in 2 components): Latent-0 ≈ BH axis,
  Latent-1 ≈ SN axis — independently reproduces "One latent to fit them all" (arXiv:2509.01881).
- The WL latent vs the kSZ gas latent: **leading axis shared (12°), second axis rotated (38°)**
  — κ sees a blend of the gas zones kSZ separates.
- SBI: WL auto-Cl alone barely constrains the 30 params (info is ~3-dim), but pins the
  2-D suppression latent to ~0.17 of the prior cloud; emulator-as-simulator escapes the
  n=253 training-set limit. Per-halo population SBI over 8.6M halos (coverage-calibrated).
- **κ×y is the feedback driver**: WL auto-stats are near-blind; the κ×y cross + tSZ y
  dominate the constraint; the per-halo κ stack is a feedback-blind mass anchor.
- Halo-level anchor: per-halo Born Δκ tracks f_gas (r=+0.62); §6 now uses the *actual*
  lightcone halo population (kernel-weighted catalog).
Show: `examples/paper_lightcone_figs2.ipynb` (the latent paper notebook), `examples/wl_latent_sbi.ipynb`.

### 3. kSZ / gas vs DESI×ACT + eROSITA — **paper-grade tension result** (`analysis/ksz-desi-act`)
Plan: `docs/ksz_desi_act_plan.md`. Two notebooks: `paper_ksz_desi_act.ipynb` (per-halo CAP)
+ `paper_ksz_field.ipynb` (field-level, uses the ray-traced maps).
- **Headline: 0% of the 30-dim Sobol volume reaches the eROSITA strong-feedback f_gas band**
  → the Siegel/Bigwood-style "TNG underestimates feedback" tension cannot be fixed by any
  TNG parameter setting; it's a model-space statement.
- τ/y profiles vs Hadzhiyska+26 GNFW on all 256 nodes; CAP-aperture reconciliation
  (headline correction); first feedback posterior from (τ, y) CAP stacks; σ_v-free f̃_gas.
- First **real-data** confrontation: BIND shear×y vs DES Y3 × ACT (on `analysis/wl-tsz-bridge`).
- Honest caveat: BIND "kSZ" is the velocity-free τ column; a true ΔT_kSZ needs a DMO
  velocity surrogate (open decision below).

### 4. WL×tSZ bridge & Fisher era (`analysis/wl-tsz-bridge`)
The June 10–18 fiducial-era arc that motivated everything: halo↔field bridge
(Δν↔ΔM_gas r=0.93; group f_gas↔S(ℓ) van Daalen relation r=0.91 — Fig. 16 reproduced for
WL), f_gas response model (WindEnergy dominant; groups respond 2.6× clusters), z~0 f_gas
saturation at ~41% of cosmic, unified electron-column/FRB-DM toolkit, Fisher rungs with a
true 2000-map hydro covariance, and the paper skeletons (`paper_wl_tsz.ipynb`, `draft.tex`,
plan `docs/wl_tsz_plan.md`). Mostly superseded-by or feeding threads 1–3, but it's the
narrative spine of "Paper I".

### 5. WL anisotropy (`analysis/wl-anisotropy`)
Is the baryonic suppression isotropic? Field-level counterfactual (radial BCM warp =
faithful control): **~40–65% of the suppression is anisotropic at first order** — but
measured with the old circular control, expected to drop on revision. Paper draft exists
(`wl_anisotropy_paper.ipynb`); parked, needs a decision (below).

## Infrastructure built (enables the next round)

- **`bind.emulator`** — GP statistics emulator: 30 params + z_s → any field statistic
  (Cl/peaks/MFs/PDF/WST/DM/Y–M/κ×y) in milliseconds. Trained on the Sobol suite;
  emulates the suppression S directly. Show: `examples/lightcone_emulator.ipynb`.
- **`bind.wlemu`** — *field-level* generative emulator (θ → κ map) via conditional flow
  matching; engine built + smoke-tested end-to-end, 1024² first-class. **Not yet trained
  at scale** (that's a pending H100 SLURM run).
- **35-dim cosmology rescaling** (`feature/cosmo-rescale`) — Angulo–White rescaling of
  TNG300-Dark adds (Ωm, σ8, Ωb, h, ns) to the Sobol design *without new N-body*.
  Feasibility validated on real TNG300-3-Dark data (mild targets ≤3.4% error; theory D(k)
  scan predicts per-run error). Full plan: `docs/cosmo_rescaling_plan.md`.
- Low-mass completeness: 10¹²–10¹³ halos captured at 28.7% (4×R200) / 51.6% (full patch)
  by *reuse only* — new generation would be 78× GPU cost.
- SHMR scatter on the suite: joint-feedback σ≈0.39 dex ≫ single-knob 0.16 dex
  (compounding), accretion history only ~10% of residual.

## Suggested 60-minute agenda

| min | topic | anchor material |
|----|-------|-----------------|
| 0–5 | The one-slide arc: pipeline → 256-run Sobol suite → 2 paper-grade results | this doc's TL;DR |
| 5–15 | Foundation: lightcone pipeline + BIND≈hydro validation (κ & y maps) | `fiducial_lightcone_stats.ipynb` |
| 15–27 | **Result 1:** cosmo-bias capstone (2 templates N&S) + LSST forecast | capstone + shear-forecast figs |
| 27–38 | **Result 2:** 2-D feedback latent, SBI, κ×y as the feedback probe | `paper_lightcone_figs2.ipynb` |
| 38–47 | **Result 3:** kSZ/eROSITA model-space tension + real-data shear×y | `paper_ksz_desi_act.ipynb` |
| 47–53 | Infrastructure futures: stat & field emulators, 35-dim rescaling | showcase notebooks / plan docs |
| 53–60 | Decisions (below) | — |

## Decisions to put to the advisor

1. **Which paper goes out first?** Candidates: (a) cosmo-bias capstone (closest to
   complete, clean result), (b) kSZ/eROSITA tension (timely, real-data hook),
   (c) unified BIND-TNG methods+bridge paper first to introduce the tool.
2. **Green-light the 35-dim rescaled suite?** Feasibility is validated; open decision D2
   = how to handle the extreme (Ωm=0.1 ∧ σ8=1.0) corner (drop, clamp, or map-level v1.5
   Fourier reweighting).
3. **Spend GPU budget training `bind.wlemu` at scale?** (multi-node H100 run; engine ready.)
4. **kSZ velocity:** build the true ΔT_kSZ via a DMO-velocity surrogate, or keep the
   τ-column framing for P4?
5. **Anisotropy paper:** finish (needs the revised control, number will shrink) or park?
6. Emulator HOS fidelity: κ×y/peaks emulation isn't good enough to stack into SBI yet
   (5.2σ artifact) — invest in better HOS emulation or keep Cl-only?

## Repo / branch map (what was committed today)

Everything below forks off `lightcone` @ `ebeeb28`, so each branch carries the full
pipeline. Nothing is pushed yet (`git push origin <branch>` when ready).

| branch | commit | contents |
|--------|--------|----------|
| `lightcone` | `67a6ab9`..`ebeeb28` | engine (`bind.emulator`, `bind.wlemu`, τ-plane/spherical/truth-halo/paired-stats CLIs, stats.py extension), all SLURM/MPI infra, tutorial + emulator showcase notebooks, fiducial validation, WORKLOG |
| `analysis/wl-tsz-bridge` | `88c5c47` | Paper-I era: bridge, f_gas response, Fisher rungs, dashboards, paper skeletons + `draft.tex`, DES Y3×ACT real-data fit, `docs/wl_tsz_plan.md` |
| `analysis/wl-cosmo-bias` | `df77be3` | transfer functions → bias → capstone → Robertson/LSST forecast (10 engines) |
| `analysis/sobol-sb35` | `c7cf82f` | Sobol atlases, SHMR, feedback latents, all SBI, `paper_lightcone_figs2.ipynb` |
| `analysis/ksz-desi-act` | `4dcbaa8` | P4 kSZ paper: engines, posteriors, `_reduce_*` batch jobs, both paper notebooks, plan doc |
| `analysis/wl-anisotropy` | `6b58f97` | anisotropy paper draft + BCM-warp control + ejection follow-up |
| `feature/cosmo-rescale` | `c230a4e` | AW10 rescaling solver, validation, Sobol error scan, plan doc |

Cross-branch notes: `examples/sobol_ml.py`, `wl_stat_latents.py`, `halo_atlas.py` live on
`analysis/sobol-sb35` and are imported by notebooks there; the kSZ §4 latent kernel is
shared between `sobol-sb35` and `ksz-desi-act` via `paper_ksz_desi_act.ipynb`. Run scripts
at the repo root (committed on `lightcone`) drive engines that live on the analysis
branches — run them from the analysis branch.

## Loose ends (deliberately not committed)

- `examples/merger_tree_11_tng300 (1).ipynb` — stray " (1)" duplicate download; triage or delete.
- `sbi-logs/` + `tng300_dm_mass_history.hdf5` (26 MB generated data) — now gitignored.
- `examples/figures_lightcone/` — figure outputs (gitignored); its two real files
  (`param_families.tex`, `sb35_sobol_analysis.ipynb`) were committed on `analysis/sobol-sb35`.
- Local branch `analysis/paper3b-ejection-heating` is empty (points at old `main`);
  its intended content now lives on `analysis/wl-anisotropy` — safe to delete.
- No branches pushed to `origin` yet.
