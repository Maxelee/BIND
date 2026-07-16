# Paper V brief — (Letter) anisotropy of the baryonic WL suppression

**Working title:** The BIND Lightcone Suite V: How anisotropic is the
baryonic suppression of weak lensing?
**Format:** LETTER (~4–6 pp, 3–4 figures). Explicitly preliminary — frame
as a first field-level measurement with a stated upper-bound caveat.

## Thesis
Baryonification methods correct DMO simulations with *spherically symmetric*
displacement kernels. Using BIND's per-halo generative painting, we build a
faithful radially-symmetrized counterfactual (a radial "BCM warp" of the
painted patches) and measure, at field level, how much of the WL suppression
survives symmetrization: ~40–65% of the effect is anisotropic at first
order. Spherical baryonification therefore misses a structural component of
the signal — though our control is conservative (symmetrized correction =
upper bound; the estimate is expected to drop with the refined control).

## Sources
1. `docs/WORKLOG.md`: 2026-06-16 (×1 "spherical-BCM control"), 06-17 (×4:
   Part II built out — all 60 OAT runs; paper draft "cross-term =
   first-order"; faithful radial-BCM counterfactual; three-way kappa —
   triaxiality-tracing inflated f_aniso), plus the earlier wl-anisotropy
   memory framing.
2. Worktree `wt/wl-anisotropy/examples/`: `wl_anisotropy_paper.ipynb` — this
   IS the paper draft in notebook form (harvest its text + figures + arc),
   `bcm_warp_comparison.py` (the control), `ejection_anisotropy.ipynb`
   (supporting physics: which feedback params drive anisotropic ejection —
   one figure max, or leftovers).

## Section outline (letter structure)
- **Intro (short):** BCM/baryonification assumes sphericity; nobody has
  measured the anisotropic fraction at field level; BIND enables the
  counterfactual.
- **Method:** painted vs radially-warped control patches; why the radial
  warp is the *faithful* control (earlier circular-spherical control
  overestimated — the three-way κ comparison showed triaxiality-tracing
  inflated f_aniso); the 60 one-at-a-time (OAT) runs; field-level κ
  statistics compared.
- **Result:** anisotropic fraction ~40–65% (first order) of the suppression;
  dependence on scale/statistic; which feedback channels drive it
  (ejection anisotropy).
- **Discussion:** implications for BCM-based marginalization (ties to Paper
  II: templates are data-driven, so they absorb this; analytic BCMs may
  not); the number is an upper bound and preliminary — refined control
  expected to lower it.

## Numbers that MUST appear (verify against WORKLOG)
~40–65% first-order anisotropic fraction (with the "measured with the old
circular control, expected to drop" qualifier verbatim in spirit); 60 OAT
runs; the three-way κ comparison conclusion.

## Figures
Embedded in `wl_anisotropy_paper.ipynb` (4.9 MB — extract all, pick 3–4):
painted-vs-warped patch visual, suppression with/without symmetrization,
anisotropic fraction vs scale, (optional) ejection-anisotropy driver figure.
Also check `examples/figures_lightcone/bcm_warp_comparison_fid.png` on disk.

## Seed citations (verify)
Schneider & Teyssier 2015; Schneider+2019; Aricò+2020; Dai, Feng &
Seljak 2018 (potential-gradient descent?) — verify; halo triaxiality WL
refs (e.g., Osato+); van Daalen+2020; Paper I/II in prep.

## Mandatory caveats
PRELIMINARY: control revision pending, number expected to drop; fiducial
TNG only (no feedback-dependence of the fraction yet); first-order
decomposition only.
