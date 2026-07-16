# Paper IV brief — gas thermodynamics vs DESI×ACT kSZ + eROSITA

**Working title:** The BIND Lightcone Suite IV: Gas thermodynamics across a
30-parameter feedback space, confronted with DESI×ACT kSZ and eROSITA
**Format:** full paper (~10–14 pp). The observational-confrontation paper.

## Thesis
Painting the full 30-parameter TNG feedback design onto the same halos lets
us ask whether ANY TNG-like feedback configuration reproduces the strong gas
depletion reported by recent kSZ (DESI×ACT) and X-ray (eROSITA) analyses.
Answer: no — 0% of the Sobol volume reaches the eROSITA strong-feedback
f_gas band, and the kSZ CAP profiles prefer the depleted edge of the design.
The tension is a statement about the TNG model *space*, not its parameters.
We provide per-halo (θ, τ, y) posteriors from CAP stacks and a σ_v-free
gas-fraction estimator.

## Sources
1. Worktree `wt/ksz-desi-act/docs/ksz_desi_act_plan.md` — THE plan doc
   (decision gates D1–D5); follow its framing.
2. `docs/WORKLOG.md`: 2026-06-24 (×4: P4 kickoff/D1; D2–D4 tau/y vs
   Hadzhiyska+26 GNFW; D4–D5 kSZ amplitude via CAP + first feedback
   posterior + σ_v-free f̃_gas; multi-probe WL×SZ overnight), 06-25 (×2:
   CAP-aperture reconciliation "headline correction"; P4 notebook refocus
   on feedback-latent, 10→8 figs), 06-26 (×3: tSZ mass-selection
   Msun-vs-Msun/h bug "the 2.6×"; figure-revision pass; field-level
   companion paper_ksz_field), 06-15 pm (eROSITA f_gas confrontation +
   projection-contamination fix — the original result).
3. Worktree `wt/ksz-desi-act/examples/`: `paper_ksz_desi_act.ipynb` (primary
   figure notebook, 8-figure arc), `paper_ksz_field.ipynb` (field-level
   companion — fold in as a section or keep as clearly-marked companion
   material), `ksz_fgas_confront.py`, `ksz_tau_gnfw.py`, `ksz_cap_compare.py`,
   `ksz_posterior*.py`, `ksz_thermo_decomp.py`, `ksz_fgas_profile.py`,
   `tau_profiles.py`, `perhalo_plus_kappay.py`, `_reduce_*.py` (methods
   details for the batch reductions: CAP f_gas, BGS κ stacks, LRG y caps).

## Section outline
- **Intro:** the "feedback stronger than TNG" claims (kSZ stacking:
  Schaan+2021/Amodeo+2021→Hadzhiyska+2026, Bigwood+2024; eROSITA groups;
  DESI×ACT); why single-simulation comparisons can't distinguish
  parameter-tension from model-tension; our design: same halos × 256
  feedback nodes.
- **Methods:** suite recap (cite Paper I); per-halo integrated catalog
  (8.6M halos: f_gas, Y, T, τ + 30 params); CAP filter stacks on τ and y
  around DESI-like (BGS/LRG) selections; GNFW profile comparison
  (Hadzhiyska+26); σ_v-free f̃_gas estimator; posterior machinery
  (per-halo population likelihood); **state plainly: BIND "kSZ" is the
  velocity-free electron column τ** — amplitude comparisons enter through
  the τ profile, not ΔT reconstruction.
- **Results:** (1) f_gas–M across the design vs eROSITA band: 0% coverage
  — with the projection-contamination fix; z≈0 f_gas saturates at ~41%
  cosmic across ALL nodes (the design's reachable ceiling); (2) τ/y CAP
  profiles vs GNFW fits across all 256 nodes (post mass-convention fix —
  the corrected comparison, mention the Msun vs Msun/h pitfall in methods);
  (3) CAP-aperture reconciliation of the kSZ amplitude; (4) first feedback
  posterior from (τ, y) CAP stacks; which parameters move; (5) thermo
  decomposition (what drives τ vs y differences); (6) field-level
  companion: ray-traced multiprobe view.
- **Discussion:** model-space vs parameter-space tension; what WOULD reach
  the band (ejection beyond R200 etc.); velocity caveat and the path to
  true ΔT_kSZ (DMO velocity surrogate); selection effects.
- **Conclusion.**

## Numbers that MUST appear (verify against WORKLOG; the 06-25/26 entries
CORRECT earlier numbers — always use the post-correction values)
0% of Sobol volume in the eROSITA strong-feedback band; ~41% cosmic f_gas
saturation; 8.6M-halo catalog; 256 nodes / 253 usable; the corrected
CAP-aperture amplitude statement; the mass-convention (2.6×) bug is a
methods-lesson footnote, not a result.

## Figures (candidates)
Disk: `examples/figures_ksz/` (13 files). Embedded: `paper_ksz_desi_act.ipynb`
(the revised 8-figure arc — extract all, use its structure),
`paper_ksz_field.ipynb`. Target 8–10 following the notebook's arc.

## Seed citations (verify all — several are recent)
Schaan+2021 (ACT kSZ stacking); Amodeo+2021; Hadzhiyska+2026 (verify the
actual reference + year); Bigwood+2024; eROSITA group f_gas (find the actual
paper the WORKLOG's "eROSITA band" refers to — likely Popesso+2024/
Bahar+2024 — verify against ksz_fgas_confront.py comments); DESI
(DESI Collab 2024); ACT DR6; Battaglia+2012 GNFW; CAP filter (Ferraro/
Schaan); van Daalen+2020; Siegel+ (check spelling/year in code comments).

## Mandatory caveats
Velocity-free τ (ΔT_kSZ unbuilt; needs DMO-v surrogate); TNG model space
only; mass floor 1e13 for painted halos (low-mass reuse partial); satellite/
miscentering effects in CAP stacks if noted in WORKLOG; y-map normalization
already physical (no extra 1/a² — validated vs Planck).
