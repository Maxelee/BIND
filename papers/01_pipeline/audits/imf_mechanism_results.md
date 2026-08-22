# IMF-mechanism checks — results

Engine: `papers/01_pipeline/imf_mechanism.py` (self-contained; does not import
`_build_figures_nb.py`). Data: `papers/01_pipeline/audits/imf_mechanism_data.json`.
Feeds paper §3b.

Physics question (arXiv:2403.10609 mechanism): high-mass-star enrichment → faster
cooling → more BH accretion → more AGN feedback. If `IMFslope` acts mainly *through*
this AGN channel, its response fingerprint should look like `BlackHoleRadiativeEfficiency`'s,
not like the wind parameters' (`VariableWindVelFactor`, `WindEnergyIn1e51erg`,
`VariableWindSpecMomentum`, `WindFreeTravelDensFac`).

**Caveat that applies throughout**: each SB35 parameter's numeric "high" bound does not
necessarily mean "more feedback" in a physically comparable sense across parameters (e.g.
`IMFslope` is a negative power-law index; whether its numeric-high end means a shallower
or steeper IMF, and hence more or fewer massive stars, is a convention question this
script does not resolve). Signed correlations/cosines are reported for transparency, but
the primary, convention-robust reading uses the **rank** of |ρ| (magnitude only) or the
**magnitude** |r| of a shape correlation, which cannot flip sign under an arbitrary
parameter-direction flip.

## (a) Column similarity — 12-statistic importance fingerprints

Rebuilt the exact fig05_param_response convention (`rebin_nan` + `spearman_masked` +
`X_unit`, `emulator_dataset_xpkfix.npz`, 253 runs, z_s = 1) independently, then took each
parameter's **column** of the 12×30 importance grid as its "fingerprint" (12 statistics:
`cl_kappa, suppression, pdf, peak_counts, minima_counts, mf_v0, mf_v1, mf_v2, cl_yy,
cl_tt, cl_kappa_y, cl_kappa_tau`).

|rho|-fingerprint (unsigned magnitude — convention-robust) | cos | rank ρ |
|---|---|---|
| IMFslope vs BHRadiativeEff | **+0.990** | **+0.517** |
| IMFslope vs VarWindVelFactor | +0.871 | −0.734 |
| IMFslope vs WindEnergy1e51erg | +0.921 | −0.441 |
| IMFslope vs VarWindSpecMom | +0.893 | −0.594 |
| IMFslope vs WindFreeTravelDens | +0.922 | −0.748 |
| BHRadiativeEff vs VarWindVelFactor | +0.896 | −0.497 |
| BHRadiativeEff vs WindEnergy1e51erg | +0.927 | −0.699 |
| BHRadiativeEff vs VarWindSpecMom | +0.910 | −0.517 |
| BHRadiativeEff vs WindFreeTravelDens | +0.938 | −0.524 |

- **Cosine similarity is uniformly high (0.87–0.99) and not very discriminating** — every
  parameter that clears fig05's per-parameter null moves most of the same 12 statistics by
  a similar amount, so raw magnitude vectors all point in roughly the same direction. That
  said, IMFslope–BHRadiativeEff (0.990) *is* the single highest pair among the six checked.
- **The rank correlation of the |ρ| fingerprint is the clean, discriminating signal**:
  IMFslope's *ordering* of which of the 12 statistics it hits hardest/weakest correlates
  **positively** with BHRadiativeEff (+0.52) and **negatively** with all four wind
  parameters (−0.44 to −0.75). BHRadiativeEff itself is anti-correlated with all four wind
  parameters too (−0.50 to −0.70). IMFslope groups with BHRadiativeEff, against the winds,
  in this convention-independent metric.

Signed peak-fingerprint cosines/rank-ρ were also computed (see JSON) but are muddied by
the sign-convention caveat above (e.g. IMFslope–BHRadiativeEff signed cos = −0.13, near
zero) — not used as primary evidence.

### Finer per-bin fingerprints: S(ell) and Cl_yy

Compared the full row (not just the peak) of the Spearman map for `suppression` and
`cl_yy`, at both the fig05 rebin (rb=16, 45 bins) and native resolution (724 ℓ-bins):

| stat | pair | cos (rb16) | cos (native) | pearson (native) |
|---|---|---|---|---|
| suppression | IMFslope vs BHRadiativeEff | −0.880 | −0.880 | **+0.645** |
| suppression | IMFslope vs VarWindVelFactor | +0.712 | +0.712 | **−0.963** |
| suppression | IMFslope vs WindEnergy1e51erg | −0.490 | −0.482 | **−0.989** |
| suppression | IMFslope vs VarWindSpecMom | +0.263 | +0.273 | **−0.987** |
| suppression | IMFslope vs WindFreeTravelDens | −0.878 | −0.873 | +0.191 |
| cl_yy | IMFslope vs BHRadiativeEff | −0.849 | −0.851 | −0.725 |
| cl_yy | IMFslope vs VarWindVelFactor | +0.534 | +0.535 | +0.743 |
| cl_yy | IMFslope vs WindEnergy1e51erg | −0.977 | −0.974 | +0.094 |
| cl_yy | IMFslope vs VarWindSpecMom | −0.837 | −0.821 | +0.729 |
| cl_yy | IMFslope vs WindFreeTravelDens | −0.956 | −0.964 | −0.957 |

Taking |pearson| (convention-robust): for **S(ell)** IMFslope's ℓ-shape correlates *more*
strongly with 3 of the 4 wind parameters (0.96–0.99) than with BHRadiativeEff (0.65) — the
*fine* ℓ-dependence within a single spectrum does **not** cleanly separate the AGN
mechanism from the wind mechanism. For **Cl_yy** the picture is mixed (WindEnergy is the
outlier at 0.09; the rest cluster around 0.7–0.96).

**Reading (a)**: the coarse "which of the 12 statistic-types does this parameter hit
hardest" ranking is where IMFslope and BHRadiativeEff cleanly cluster against the winds;
the fine ℓ-by-ℓ shape *within* one spectrum does not show the same clean separation — it
more likely reflects a common gas-ejection/heating scale set by the halo physics itself,
shared across whichever knob is turned.

## (b) Two-bound signatures — real full-map twobound runs

`bind_science/runs/twobound/` = 60 run directories = (nominally) 2 bounds × 30 params.
**All 60 have full map-level products** (`Cl_kappa.npz`, `Cl_kappa_y.npz` [which already
contains `cl_yy`], `kappa_maps.npz`, `y_maps.npz`) — none are patch-only, contrary to the
initial hint in the brief. Self-contained reimplementation of
`dashboard_precompute.g1()`'s one-parameter-changed decode found **57/60 runs decode to a
clean single-parameter perturbation** (see part (c) for the other 3).

For each of IMFslope / BHRadiativeEff / the 4 wind params with a complete low+high bound
pair, defined the **response shape** = `log(Cl_high/Cl_fid) − log(Cl_low/Cl_fid)` vs ℓ,
for both Cl_κκ(z_s=1) ("kk", S(ℓ)-equivalent since it's normalized to the same fiducial)
and Cl_yy ("yy"), using `bind_science/runs/bind/run_0000` as the shared fiducial reference.

| param | bounds | runs | median\|shape_kk\| | median\|shape_yy\| |
|---|---|---|---|---|
| IMFslope | [−2.8, −1.8] | run_0012/13 | 0.154 | 1.039 |
| BHRadiativeEff | [0.05, 0.8] | run_0040/41 | 0.320 | 1.000 |
| VarWindVelFactor | [3.7, 14.8] | run_0004/05 | 0.221 | 0.382 |
| WindEnergy1e51erg | [0.9, 14.4] | run_0000/01 | 0.073 | 1.171 |
| VariableWindSpecMomentum | — | (only 1 bound; see part c) | — | — |
| WindFreeTravelDens | [0.005, 0.5] | run_0020/21 | 0.054 | 0.124 |

Cross-parameter shape correlation (Pearson of the log-response curve vs ℓ, ℓ ∈
[100, 1.5×10⁴], N=206 bins):

| pair | statistic | pearson(shape) |
|---|---|---|
| IMFslope vs BHRadiativeEff | kk | **−0.901** |
| IMFslope vs BHRadiativeEff | yy | **−0.995** |
| IMFslope vs VarWindVelFactor | kk | +0.532 |
| IMFslope vs VarWindVelFactor | yy | −0.490 |
| IMFslope vs WindEnergy1e51erg | kk | −0.833 |
| IMFslope vs WindEnergy1e51erg | yy | −0.998 |
| IMFslope vs WindFreeTravelDens | kk | −0.362 |
| IMFslope vs WindFreeTravelDens | yy | +0.199 |

**Reading (b)** (using |r|, the convention-robust magnitude): IMFslope's ℓ-shape has the
**highest** magnitude-correlation with BHRadiativeEff for *both* Cl_κκ (|r|=0.90, the single
highest of the four) and Cl_yy (|r|=0.995, tied with WindEnergy1e51erg at 0.998). The two
purely-*kinematic* wind parameters (VarWindVelFactor, WindFreeTravelDens — set wind
velocity/loading) show much weaker shape-correlation with IMFslope (|r| = 0.20–0.53) than
the two purely-*energetic* parameters (BHRadiativeEff, WindEnergy1e51erg — set an energy
budget) do (|r| = 0.83–1.00). This independent, full-map check reproduces the same
IMFslope/BHRadiativeEff clustering found in (a), and refines it: IMFslope's imprint looks
like an **energy-injection-type** mechanism (shared with both AGN radiative efficiency and
SN wind energy) rather than a **kinematic** one (wind velocity/loading) — consistent with,
but slightly broader than, a pure AGN-only channel.

## (c) dashboard_cache g1_design.json — 57 vs 60

- `bind_science/dashboard_cache/g1_design.json` has **57 entries**; missing runs:
  `run_0018`, `run_0049`, `run_0053`.
- 27/30 parameters have both bounds decoded; **3 parameters have only one**:
  `VariableWindSpecMomentum`, `UVBH0Deltaz`, `UVBHepDeltaz`.
- **Root cause, verified directly**: for these 3 runs, `params.npy` — and the *design*
  table `twobound_params.npy` itself — is **exactly** the fiducial parameter vector (not a
  decode bug: `np.allclose(actual, fiducial)` is `True` for all three). I.e. the *second*
  bound for those 3 parameters was never actually generated as a distinguishing
  perturbation; one of the two "bound" runs for each of those 3 parameters used the
  fiducial value instead (for `VariableWindSpecMomentum` the one *present* singleton run,
  `run_0019`, is at 4000 vs fiducial 0 — suggesting the *other*, missing bound was meant to
  probe a value ≤ 0 that collapsed onto the fiducial 0 by construction, or a similar
  edge-of-prior issue for the other two).
- Per-run **products** are now complete: verified directly that all 60 twobound run
  directories have `peak_counts_nufid.npz` and `paired_stats.npz` (0 missing of either),
  resolving the earlier 2026-06-15 WORKLOG gap (only 20/60 nufid restats had landed then).
- **Consequence**: re-running the precompute script alone will **not** move 57→60 — the
  data for those exact 3 design slots doesn't exist. Reaching 60 unique 1P-style entries
  requires **3 new twobound runs** at the actual intended (non-fiducial) bound values for
  `VariableWindSpecMomentum`/`UVBH0Deltaz`/`UVBHepDeltaz`, run through the full BIND
  painting + map + nufid/paired-stats pipeline, before the cache rebuild can pick them up.
- **Script location**: `examples/dashboard_precompute.py` is **not in the current working
  tree** (only a stale `examples/__pycache__/dashboard_precompute.cpython-311.pyc`
  remains). Its source lives on git branch `analysis/wl-tsz-bridge`, commit `88c5c47`
  (`git show analysis/wl-tsz-bridge:examples/dashboard_precompute.py`). It is a pure
  reduction over already-materialized `runs/twobound/*` products (`g1()` builds
  `g1_design.json` via the median-decode described above; `g1_stats`/`g1_quick`/`g1_maps`/
  `g1_peakhalo*` consume it) — no map generation of its own, so once the 3 new runs exist
  the rebuild itself is cheap (the 2026-06-15 log entry: "~min" runtime, no SLURM needed).
