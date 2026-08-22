# Paper figure map — 2026-08-06 arc revision (v2)

Author's flow: methods → halo-level validation → map-level validation →
parameter effects + shape families → **pivot to the analytic model**
(construction → latent choice → kernels → model validation → tie back to
families) → astrophysics constraints (corner). The van Daalen material is out
of the main arc.

Files in `figs_v2/` (PDF) + `figs_preview/` (PNG). "Cell" = the §3c cell of
`_build_figures_nb.py` (any `pfig_`/`fig20` name via `_run_subset.py` renders
all of them); "script" = `paper_s3b_clusters.py` (twobound side).

## Locked sections (methods / halo / map validation / parameter effects)

Existing figs 1–8 + 12 arrangement unchanged. Families:

| slot | file | source |
|---|---|---|
| families MAIN | `pfig_s3b_cl_clusters` (script) | ΔC_ℓ families C1–C4 + singleton, r̄ = 0.95–0.98; IMFslope rides C3. |
| families APPENDIX | `pfig_s3b_pdf_clusters` (script) | family consistency on the κ-PDF (one sentence in text). |

## The analytic-model section (the pivot; replaces the vD/bridge material)

Order within the section:

| # | slot | file | role |
|---|---|---|---|
| 1 | construction MAIN | `pfig_s4a_model_construction` (script) | the tutorial formalized: (a) universe → four halo numbers; (b) one ℓ: the partial slope is the coefficient; (c) every ℓ → the four kernels; (d) one knob's ΔS = fingerprint-weighted kernel sum (chain rule, no fitting). Motivates the methodology before any equation. |
| 2 | latent measurement MAIN | `pfig_s4a_latent_measurement` (`paper_latent_measurement.py`) | HOW the four latents are measured: one real hinge-bin halo in four channels (Σ_tot, Σ_gas, Σ_star, T) with R500c/R200/annulus geometry drawn where each computation happens; below, the population step (per-halo values across the 632 hinge-bin halos, median = the latent — reproduces the fiducial's vector 0.801/0.097/0.645/6.833). |
| 2b | latent choice | text + `latent_ablation.py` numbers | why (f̃_bar, f̃_star, c_gas, log T̃): roles + exhaustive C(15,4) ablation (ours top 3.7%, plateau; f̃_star irreplaceable). Appendix table on request. |
| 3 | kernels MAIN | `fig20g_latent_kernels` (cell) | the five coefficient functions × z_s fan + 1σ-impact panel — the centerfold. |
| 4 | buildup MAIN | `pfig_s4a_cv_buildup` (cell) | one CV-R²(ℓ) panel: f̃_bar → +f̃_star → +c_gas → +log T̃ vs the 30-param baseline ("how many numbers is feedback"). |
| 5 | validation MAIN | `pfig_s4b_model_curves` (cell) | **fig09c layout, analytic model instead of the GP**: 7 LOO nodes × 13 statistics, measured (black) vs model (blue dashed), per-panel error stamps. |
| 6 | fiducial demo | `pfig_s4b_reconstruction` (cell) | the out-of-design fiducial + 3 LOO nodes, S(ℓ) + κ-PDF (the "it actually works on a universe the fit never saw" figure). |
| 7 | generality MAIN | `pfig_s4b_generality` (cell) | 9-statistic bars, 3- vs 4-latent — one model, every statistic. |
| 8 | tie-back MAIN | `pfig_family_kernel_bridge` (script) | family shapes = kernel mixtures (r = 0.98–1.00 out-of-design) + fingerprint matrix — closes the loop to C1–C5. |

Appendix for this section: `pfig_s4b_app_zs` (one latent vector, five source
planes), `pfig_s4b_app_epoch` (partition early / budget late; flex slot),
`pfig_s4b_error_budget` (`paper_error_budget.py`: the model error decomposed —
intrinsic vs λ-measurement vs kernel-coefficient vs S-realization noise; at
ℓ≲500 the residual is pure measurement noise, i.e. the model saturates the
data there; reliability R = 0.96–1.00, attenuation ≤4% and only for log T̃).

## Astrophysics constraints (closing section)

| slot | file | note |
|---|---|---|
| MAIN | `fig20i_latent_corner` (cell) | the staged SH-scores corner. **Already in the requested form**: the prior cloud and the held-out test are *measured* latents from the simulations — the 30→4 regression is not used anywhere in this figure. Mandatory internal-recovery framing in the caption prints. |

## Analytic-model validation (`sec:family_validation`) — reworked 2026-08-14

`main.tex` uses **`pfig_fm_model_curves`** (from `family_model_section_figs.py`, the
2026-08-12 agnostic-λ family), *not* the `pfig_s4b_model_curves` cell listed in the
table above — that row predates the agnostic-λ rework. `imgs/` export is a straight
copy: `cp figs_preview/pfig_fm_model_curves.png imgs/`.

**What changed.** Every panel is now divided by a **reference** (`REF_MODE`, default the
measured fiducial; `"mean"` switches to the Sobol design mean). Raw, the panels were
unreadable: P(ν) spans four decades, N_pk three, V₀ runs 1→0, so a few-percent feedback
response was a line width and all seven leave-one-out nodes lay on top of each other.
The residual sub-panel now uses the **same** normalization as the panel above it, so
spread and model error read against one another, with **per-panel** vertical scales (the
response runs ±1% for V₀ to ±80% for S(ℓ); the old shared ±7% hid one and clipped the other).

**Two things to know before editing it.**
- The **`C_ℓ^κκ` panel is gone and nothing was lost**: normalizing cancels the shared DMO
  band spectrum, so `C_ℓ/C_ℓ^ref` is *algebraically identical* to `S(ℓ)/S(ℓ)^ref` — the old
  panels (a) and (b) would have been the same picture twice. Seven statistics, seven panels,
  legend + error summary in the freed eighth slot. Matches main.tex's "seven statistics".
- A **tail mask** is mandatory, not cosmetic: P, N_pk, N_min, V₀, V₁ all fall to zero in
  their tails and **V₂ changes sign**. `REF_FLOOR = 0.05` (bins below 5% of the panel peak
  are dropped and shaded). Measured: at 1% the κ-PDF ratio goes negative and V₂'s zero
  crossing blows up; 3% is stable but leaves boundary spikes (N_min max residual 15.0% →
  4.5% at 5%). S(ℓ) is never masked — its reference never drops below 88% of peak.

**Caveat.** With `REF_MODE="fid"` the reference comes from `amplitude_sets.npz`'s
`<st>__fid_measured`, which derives from the **retired** `bind/run_0000`; its refresh is a
deferred item in `referee/FIDUCIAL_SWAP.md`. `REF_MODE="mean"` avoids this entirely (the
Sobol set was conditioned correctly) at the cost of a less physical reference.

## §3 opener (`sec:astro`) — rebuilt 2026-08-14

| slot | file | LaTeX name | source |
|---|---|---|---|
| §3 opener MAIN | `fig23_s3_opener` (cell) | `imgs/fig23_s3_opener.png` | 3 stacked panels on a shared ℓ axis. (a) S(ℓ) at z_s=1: Sobol 5–95% (grey, 256 nodes), TNG fiducial (black), LSST-Y10 + *Euclid*-like ±1σ as nested **filled** ribbons on the fiducial. (b),(c) detection significance \|S_node−S_fid\|/σ per survey, 5–95% band + median, log y. |

**The `imgs/` export step for this figure is a straight copy** — the notebook
name and the paper name coincide:
`cp figs_preview/fig23_s3_opener.png imgs/fig23_s3_opener.png` (`imgs/` is
gitignored, line 48 of the root `.gitignore`; it is the local/Overleaf build
target only).

**Ensembles (deliberate — see the cell's markdown for the full argument):** the
S(ℓ) *curves* stay on the canonical 50-real seed-paired trees (there is no
N=1000 version of the 256 Sobol nodes, only of the fiducial); the survey *band*
is a **relative** error measured on the N=1000 fiducial campaign
(`bind_n1000/analysis/stream_stats_bind_run_0000.npz`, landed 2026-08-13) and
never enters a ratio against a 50-real quantity. Going 50→1000 **widens** the
band by 8–18%. The cell is **self-contained given the setup cell** (its own
`cl_relerr23`), so it carries no `_run_subset.py` DEPS entry and renders in ~7 s:

```bash
python _run_subset.py fig23_s3_opener
```

## Cut / demoted (2026-08-06 author rulings)

- `fig20c_vandaalen_plane` — CUT from the paper (vD relation survives as text
  motivation for f̃_bar; figure remains in the repo).
- `pfig_s3c_hinge_plane` — CUT (its two stories are carried by the
  construction figure panel (b) and the corner's prior cloud).
- `fig20a_vandaalen_matrix` — not in the new arc; park as appendix/optional
  pending author confirmation.
- `fig20h_theta_to_latents` — appendix/optional (not needed by the corner;
  documents the θ→λ leg's accuracy if a referee asks).
- earlier cuts stand: IMF-highlight overlay, old fig 21, τ-profile
  S(5000)-split.
- `fig23b_s3_bridge_scatter` — **demoted 2026-08-14**, was the §3 opener
  (S(5000) vs group f_gas, WindEnergy-coloured, ruled 2026-08-03). Not cut: it
  makes the point fig 23 does not (the span is *indexed by a gas observable*),
  and it is the §3c bridge in miniature. Its survey gauges are the v1
  Knox-only analytic form at one ℓ bin and are superseded by fig 23's measured
  N=1000 band. Keeps the `fig04_field_validation` DEPS entry (it reuses that
  cell's `cl_relerr`).

## Render commands

```bash
cd papers/01_pipeline
python paper_s3b_clusters.py            # families + bridge + construction
python _run_subset.py pfig_s4b_model_curves   # whole §3c cell (all pfig_/fig20)
```
