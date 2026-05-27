# Summary table — BIND scatter-residual analysis (CAMELS-TNG CV)

## Sample

- CV halos (post-cut M_200c > 1e13 M_sun/h): **1154**
- BIND samples per halo (K): **10**
- Bootstrap B = **2000** (over halos, with replacement)

## Headline statistics

- ‖C^T − C^G‖_F  (primary 7×7, Spearman): **0.6543**
  - Frobenius null median (split-half truth): 0.3386
  - Frobenius p-value: **0.0045**
- ‖C^T − C^G‖_F  (supplementary 8×8, with log10_f_b): 0.6781
- Angle between leading eigenvectors: **7.05°**
- Top eigenvalue ratio (T/G): 0.879

## Top 3 strongest truth correlations (off-diagonal)

| obs a | obs b | C^T | C^G | z |
|---|---|---|---|---|
| log10_M_DM | log10_M_gas | +0.821 ± 0.011 | +0.880 ± 0.009 | -4.06 |
| log10_M_gas | log10_Sigma_gas_c | +0.717 ± 0.017 | +0.728 ± 0.017 | -0.46 |
| q_DM | q_star | +0.590 ± 0.020 | +0.687 ± 0.017 | -3.66 |

## Per-halo Pearson agreement P_aa (expected ≈ 0)

| observable | P_aa | SE |
|---|---|---|
| log10_M_DM | +0.997 | 0.001 |
| log10_M_gas | +0.948 | 0.008 |
| log10_M_star | +0.799 | 0.032 |
| log10_Sigma_gas_c | +0.768 | 0.029 |
| q_DM | +0.901 | 0.007 |
| q_gas | +0.716 | 0.021 |
| q_star | +0.667 | 0.018 |

## Mass dependence of ρ(ΔM_*, ΔM_gas)

| log10 M200c bin | N | ρ truth | SE truth | ρ BIND | SE BIND |
|---|---|---|---|---|---|
| [13.00, 13.30) | 643 | +0.343 | 0.037 | +0.494 | 0.033 |
| [13.30, 13.70) | 355 | +0.401 | 0.050 | +0.572 | 0.041 |
| [13.70, 14.80) | 156 | +0.464 | 0.071 | +0.588 | 0.057 |

## Pairs flagged at |z| > 2 (off-diagonal C^T − C^G)

| obs a | obs b | C^T | C^G | z |
|---|---|---|---|---|
| log10_M_DM | log10_M_gas | +0.821 | +0.880 | -4.06 |
| log10_M_DM | log10_M_star | +0.464 | +0.613 | -4.54 |
| log10_M_DM | log10_Sigma_gas_c | +0.505 | +0.620 | -3.59 |
| log10_M_gas | log10_M_star | +0.373 | +0.533 | -4.49 |
| log10_M_star | log10_Sigma_gas_c | +0.219 | +0.463 | -6.30 |
| q_DM | q_gas | +0.504 | +0.672 | -5.65 |
| q_DM | q_star | +0.590 | +0.687 | -3.66 |
| q_gas | q_star | +0.242 | +0.434 | -4.96 |
