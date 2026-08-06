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
| 2 | latent choice | text + `latent_ablation.py` numbers | why (f̃_bar, f̃_star, c_gas, log T̃): roles + exhaustive C(15,4) ablation (ours top 3.7%, plateau; f̃_star irreplaceable). Appendix table on request. |
| 3 | kernels MAIN | `fig20g_latent_kernels` (cell) | the five coefficient functions × z_s fan + 1σ-impact panel — the centerfold. |
| 4 | buildup MAIN | `pfig_s4a_cv_buildup` (cell) | one CV-R²(ℓ) panel: f̃_bar → +f̃_star → +c_gas → +log T̃ vs the 30-param baseline ("how many numbers is feedback"). |
| 5 | validation MAIN | `pfig_s4b_model_curves` (cell) | **fig09c layout, analytic model instead of the GP**: 7 LOO nodes × 13 statistics, measured (black) vs model (blue dashed), per-panel error stamps. |
| 6 | fiducial demo | `pfig_s4b_reconstruction` (cell) | the out-of-design fiducial + 3 LOO nodes, S(ℓ) + κ-PDF (the "it actually works on a universe the fit never saw" figure). |
| 7 | generality MAIN | `pfig_s4b_generality` (cell) | 9-statistic bars, 3- vs 4-latent — one model, every statistic. |
| 8 | tie-back MAIN | `pfig_family_kernel_bridge` (script) | family shapes = kernel mixtures (r = 0.98–1.00 out-of-design) + fingerprint matrix — closes the loop to C1–C5. |

Appendix for this section: `pfig_s4b_app_zs` (one latent vector, five source
planes), `pfig_s4b_app_epoch` (partition early / budget late; flex slot).

## Astrophysics constraints (closing section)

| slot | file | note |
|---|---|---|
| MAIN | `fig20i_latent_corner` (cell) | the staged SH-scores corner. **Already in the requested form**: the prior cloud and the held-out test are *measured* latents from the simulations — the 30→4 regression is not used anywhere in this figure. Mandatory internal-recovery framing in the caption prints. |

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

## Render commands

```bash
cd papers/01_pipeline
python paper_s3b_clusters.py            # families + bridge + construction
python _run_subset.py pfig_s4b_model_curves   # whole §3c cell (all pfig_/fig20)
```
