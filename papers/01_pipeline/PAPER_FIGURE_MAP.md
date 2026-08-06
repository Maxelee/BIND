# Paper figure map — 2026-08-06 author figure plan

Slot → file → caption source. Files live in `figs_v2/` (PDF) + `figs_preview/`
(PNG). "As-is" figures are the existing fig20 family members; "composed"
figures are re-compositions rendered by the PAPER COMPOSITIONS block at the end
of the §3c cell (`_run_subset.py pfig_...` renders them; captions draw from the
same cell's live prints). §3b renders standalone via `paper_s3b_clusters.py`
(twobound caches, no Sobol-cell dependency).

## §3b — response-shape families (1 main, 1 appendix)

| slot | file | source / caption numbers |
|---|---|---|
| MAIN | `pfig_s3b_cl_clusters` | `paper_s3b_clusters.py` (clk leg). C1 n=12 r̄=0.98 (BH/AGN accretion + misc), C2 n=7 r̄=0.95 (wind velocity), C3 n=5 r̄=0.96 (RadioFdbk + **IMFslope** — the mechanism conclusion), C4 n=5 r̄=0.96 (WindEnergy), singleton SNII_MinMass. 30/30 params pass S/N≥3. |
| APPENDIX | `pfig_s3b_pdf_clusters` | same script (pdf leg): families persist across statistics (C1 n=17 r̄=0.97 incl. IMFslope; WindEnergy family intact n=4 r̄=0.98). One text sentence. |

CUT: single-panel IMF-highlight overlay + old fig 21 (superseded by cluster
membership; `imf_shape_clusters.py` remains the machinery reference).

## §3c — the halo-latent bridge (3 mains; 2026-08-06 arc revision)

Arc per the author: parameter effects → family breakdown (§3b) → **the
family↔kernel bridge** → the analytic model (§4a). The vD anchor no longer
carries the narrative.

| slot | file | source |
|---|---|---|
| MAIN 1 | `fig20a_vandaalen_matrix` (as-is) | r(S, f̃_bar) ℓ×M matrix + vD20 box; numbers in FIGURE_NUMBERS. |
| MAIN 2 | `pfig_s3c_hinge_plane` (composed) | (a) = fig20b hinge (r=0.977); (b) = fig20d latent plane (budget × partition, family compass, enhancement rings). |
| MAIN 3 | `pfig_family_kernel_bridge` (`paper_s3b_clusters.py`) | **the families↔coefficients bridge**: chain-rule closure ∂S/∂θ_j = Σᵢ cᵢ(ℓ)·∂λᵢ/∂θ_j — each family's mean twobound shape vs its kernel mixture (r = 1.00/0.99/0.98/0.98/0.86, OUT-OF-DESIGN: Sobol kernels × twobound fingerprints) + the fingerprint matrix Δλ/σ_λ. The families exist because there are only four kernels. |

DEMOTED: `fig20c_vandaalen_plane` — appendix/optional (author: "doesn't
really fit"; the vD relation stays as text motivation for f̃_bar).
CUT: the τ-profile S(5000)-split (fig20b right panel / fig12b) —
arrangement-beyond-budget carried by the plane panel and the kernels.

## §4a — the analytic model (2 mains)

| slot | file | source |
|---|---|---|
| MAIN 1 | `fig20g_latent_kernels` (as-is) | five kernels × z_s fan + 1σ-impact panel — the physics centerfold. |
| MAIN 2 | `pfig_s4a_cv_buildup` (composed) | ONE CV-R²(ℓ) panel: f̃_bar → +f̃_star → +c_gas → +log T̃ (+ 30-raw-param baseline). Medians printed by the cell ("how many numbers is feedback"). |

## §4b — validation & generality (2 mains, 2–3 appendix)

| slot | file | source |
|---|---|---|
| MAIN 1 | `pfig_s4b_reconstruction` (composed) | fig20e (b,c): S(ℓ) for 3 LOO nodes + out-of-design fiducial, κ-PDF subpanel. RMS from fig20e prints (incl. the thermal-kernel robustness breakdown). |
| MAIN 2 | `pfig_s4b_generality` (composed) | per-statistic CV-R² bars, 3- vs 4-latent, 9 statistics — "one model, every statistic". |
| APPENDIX | `pfig_s4b_app_zs` (composed) | fig20f(a) standalone: one low-z latent vector, five source planes. One sentence in text. |
| APPENDIX | `fig20h_theta_to_latents` (as-is) | θ→λ map validation 4-panel; GP/linear CV-R² numbers in text. |
| FLEX (appendix by default) | `pfig_s4b_app_epoch` (composed) | fig20f(b) standalone: partition set early / budget built late / z-matched-worse. Promotable to main per the author's flex ruling. |

## §4c — inference (1 main)

| slot | file | source |
|---|---|---|
| MAIN | `fig20i_latent_corner` (as-is) | the staged SH-scores corner (analytic Student-t, coverage-verified). Mandatory internal-recovery framing sentence in the caption (printed by the cell). |

## Render commands

```bash
cd papers/01_pipeline
python paper_s3b_clusters.py                      # §3b main + appendix
python _run_subset.py pfig_s3c_hinge_plane        # any pfig_/fig20 name runs
                                                  # the whole §3c cell (all
                                                  # fig20* + pfig_* files)
```
