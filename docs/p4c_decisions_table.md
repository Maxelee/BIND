# P4c analysis-decision registry (R9 deliverable)

Every analysis choice in the tSZ/kSZ confrontation chain, labeled **pre-hoc**
(fixed before comparing data to models) or **post-hoc** (made after seeing a
comparison, with the trigger and the re-validation that justifies keeping it).
Companion to `docs/p4c_referee_hardening_plan.md`; verdicts in
`KS/lightcone/verdicts/`.

| # | choice | status | trigger / justification |
|---|--------|--------|------------------------|
| 1 | LRG sample: DESI DR1 SGC spec, z∈[0.4,0.6], WEIGHT-weighted | pre-hoc | plan §2-T2 (matches mock z=0.503) |
| 2 | Mass anchor logM200=13.18 (Sailer+24) for θ200(z) | pre-hoc | plan §1.4; mock tuned to it in P3-LRG |
| 3 | CAP filter = repo `cap_batch` formula + gates | pre-hoc | plan §1.4 |
| 4 | Measurement resolution 0.29296875'/px (BIND-matched), not the map-native 0.5' | **post-hoc** | first M2R review: discrete CAP is resolution-dependent (coarse xb cols returned 0/garbage). Re-validated: fine/coarse transfer ≈1.00 for xb≥0.9; R1a closure measures the resampling operator bias (−4.5→−1.9%) and corrects it with 50% kept as systematic |
| 5 | Aperture rule: xb·θ200 per-object on BOTH sides (data: anchor-z rule; model: same anchor rule via R2 re-stack) | **post-hoc** | first M2R review (grid-convention mix) + R2c (per-halo-θ200 vs anchor rule are different measurements). Node-independent correction (±2%) |
| 6 | Primary data map: deproj_cib_1.7 (baseline = labeled secondary) | **post-hoc** (choice), pre-hoc precedent | baseline ILC innermost points are dust-negative (−4 to −7σ); Liu+2025's own fiducial is deproj. Baseline-data variant always reported (0 consistent) |
| 7 | χ² fit range xb≤1.4 | **post-hoc** scope, pre-hoc convention | adopted from M1's kSZ 1-halo convention for cross-probe symmetry; beyond it BIND's ≥1e13-only painting lacks the 2-halo/diffuse floor (R4 quantifies) |
| 8 | 100-cell jackknife + Hartlap inflation per sample-estimated block | **post-hoc** correction | R0 (referee audit): 30 cells + no Hartlap biased χ² ×1.32; direction conservative, all χ² regenerated |
| 9 | Satellite treatment: per-node HOD stacks, κ-calibrated to the lensing anchor, f_eff=0.08±0.04 (=true f_sat 0.11±0.05 × 0.73 in-catalog factor) | **post-hoc** model upgrade | R2: mock was pure-centrals; naive satellite addition double-counts the anchor mass (recorded as unphysical bound); median-matching is a no-op; fiducial-template transfer invalid (±35–39% node spread) → per-node |
| 10 | Central miscentering ignored | pre-hoc → justified | R2a: ~2% under the 1.6' beam, ≪ CIB band |
| 11 | CIB systematic: correlated Σ_CIB across 11 fine-res variants (diagonal half-band retired) | **post-hoc** correction | R5a: the diagonal band absorbed non-CIB-shaped residuals (χ²_med 13.5→40); per-variant full-analysis range also reported |
| 12 | Mass-anchor nuisance ±0.1 dex (coherent template both sides) | pre-hoc (plan R3), implemented post-R5 | dominant coherent freedom; ∂model~+17%/0.1dex |
| 13 | R7 fidelity correction: model ÷ (1+16.4%±2%) | **post-hoc** measured calibration | truth-lightcone closure at the exact halos/apertures; validates the a-factor production path |
| 14 | kSZ covariance: R6 unit fix (release cov_ksz = raw-amplitude cov, byte-identical to Fig6) + Hartlap | **post-hoc** correction of an external release bug | `_r6_ksz_audit.py` reproduces the original χ² exactly before correcting; bgs110 31→27 (rank ρ=0.97), bgs1125 48→5 |
| 15 | kSZ primary cut bgs110 (bgs1125 = variant) | pre-hoc | M1/P6b convention; post-R6 note: bgs1125 is unstable to the covariance fix, bgs110 is stable |
| 16 | Consistency threshold χ²<dof+2√(2dof) | pre-hoc | M1 convention; threshold scans (slack 1/2/3, p>0.05/0.01) always reported |
| 17 | Headline = tension + ranking coherence, NOT consistent-node counts | **post-hoc** framing, forced | counts swung 0↔119 across defensible error treatments (documented in R5.json); the invariants are the data deficit direction and ρ(kSZ,tSZ)=0.81–0.90 |
| 18 | EBV<0.15 variant reported, not cut | pre-hoc (plan decision item 2 default) | shift = 0.05–0.07σ (negligible) |
| 19 | Nulls: random (10×) + RA-rotated; gate scoped to the fit range | pre-hoc, scope **post-hoc** | random null fails (−3σ, ~3% of signal) only at θ≥4.75', outside the fit range; not Galactic-|b|-correlated (R5d) |
| 20 | DES WL dropped as a constraint (appendix; co-ranking ρ=0.78 kept as corroboration) | **post-hoc** (user decision) | D4: amp-marginalized ξ± accepts all 253 (envelope ≤0.4% vs 11–19% errors); map leg reconstruction-filter-dominated |
| 21 | Liu csv used only as superseded overlay; comparison scales ≤2' | **post-hoc**, resolved | R5c: the release csv's duplicated column IS the pz4 bin (χ²/9=1.7); our pipeline reproduces it at ~1.1σ; bins-bit-identical bug proven |

**Data-side blinding statement for the paper:** choices 4–7 and 11–13 were
made after model-data comparisons, each triggered by a diagnosed estimator or
budget defect, each accompanied by a closure test or variant table rather than
an outcome preference; the headline moved *against* the initially-favorable
result (consistency → tension), which is the opposite of confirmation bias.
