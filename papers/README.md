# The BIND Lightcone Suite — Draft Index

Cross-paper coherence pass completed 2026-07-16. Five drafts, all compiling cleanly
(pdflatex → bibtex → pdflatex ×2, zero LaTeX errors, zero undefined references/citations
on the final pass in every paper). Series title, numbering, and sibling-paper `bindI`–`bindV`
bibliography entries were standardized across all five `references.bib` files during this
pass (see "Coherence fixes applied" below).

| # | Directory | Compile | Pages | Figures | `\todo{}` items |
|---|-----------|---------|-------|---------|------------------|
| I | `01_pipeline` | clean | 20 | 13 | 9 |
| II | `02_cosmo_bias` | clean | 21 | 10 | 4 |
| III | `03_latent_sbi` | clean | 26 | 14 | 15 |
| IV | `04_ksz_gas` | clean | 19 | 11 | 8 |
| V | `05_anisotropy` | clean | 10 | 4 | 8 |

---

## Paper I — `01_pipeline`
**The BIND Lightcone Suite I: Baryon-Painted Weak-Lensing and Sunyaev–Zel'dovich
Lightcones from Dark-Matter-Only Simulations**

Introduces BIND, a conditional flow-matching emulator trained on CAMELS-TNG that paints
stochastic, multi-channel baryonic fields (gas/stellar density, Compton-$y$, temperature,
entropy, electron pressure) onto individual dark-matter-only halos, and extends it to a
full cosmological lightcone: per-snapshot painting of TNG300-Dark halos, circular-aperture
compositing, lens-plane construction, and multi-plane ray-tracing producing self-consistent
$\kappa(z_s)$, Compton-$y$, and kSZ optical-depth $\tau$ maps. At fixed fiducial parameters
the BIND lightcone reproduces the true TNG300 hydrodynamical lightcone's WL power spectrum
and Minkowski functionals to a few percent, with peak/minimum counts and halo-level $Y$–$M$
and pressure-profile statistics agreeing within their respective (larger) uncertainties. The
paper shows that the tSZ-to-WL peak ratio $R(\nu)$ is a specific-thermal-energy proxy
($\propto f_{\rm gas}\,T_{\rm mw}$, not gas fraction alone) and establishes the quantitative
halo↔field bridge — group gas fraction and integrated Compton-$Y$ predict the WL suppression
$S(\ell)$ at $r=0.91$ and $r=0.93$ — that the companion papers build on. It closes with the
completeness of the reuse-based low-mass extension and the (not-yet-executed) mass-resolution
validation program. Compiled cleanly: 20 pages.

**Remaining `\todo{}` items** (9; grepped from `main.tex`):
- Locate or regenerate the standalone $R(\nu)$-vs.-feedback-parameter figure (source notebook
  missing from the working tree).
- Fig. 13's family-classification legend (AGN $n{=}18$/SN $n{=}31$/other $n{=}8$) does not
  match Fig. 6's (20/25/12) for the nominally identical 57-run 1P design — unreconciled.
- No completeness schematic figure exists for the low-mass reuse result; generate or omit.
- Cross-check the exact Planck $y$-map release (2015 XXII vs. later) before citing a specific
  reference for the $1.6\times10^{-6}$ comparison value.
- Insert the final GitHub URL/tag at release.
- Confirm the exact released weight identifier for the lightcone/thermo checkpoint used here.
- GPU-hour cost comparison, BIND lightcone vs. full TNG300 hydro run — no number on hand.
- Author list to be finalized; acknowledgments to be finalized.

---

## Paper II — `02_cosmo_bias`
**The BIND Lightcone Suite II: Two Nuisance Parameters Suffice to Marginalize Baryonic
Feedback in LSST-Era Cosmic Shear**

Paints a 256-node Sobol sweep (253 usable) of all 30 IllustrisTNG feedback parameters onto a
shared DMO lightcone at fixed TNG-fiducial cosmology and measures the WL suppression
$S(\ell,z_s)$ against a paired DMO trace. Ignoring baryons biases LSST-Y10 $S_8$ by up to
$6.03\sigma$ at $\ell_{\rm max}=5000$; one data-driven nuisance template leaves up to
$0.67\sigma$ unmarginalized, but the leading two singular modes of the suppression-residual
ensemble collapse the bias to $\le0.073\sigma$ in-sample (and $\le0.08\sigma$ held-out) at a
cost of $\approx3\times$ the statistical floor, confirmed end-to-end by a full non-linear
likelihood forecast reproducing Robertson et al. (2026) ($-0.368\sigma$ residual at
$1.107\times$ cost). Richer $N\ge6$ template ladders add substantial further statistical cost
for a residual bias already an order of magnitude below the LSST-Y10 floor at $N=2$. An
independent geometric (active-subspace) analysis of ten WL+tSZ statistics and an independent
VAE latent-space decomposition (Paper III) both find the same $\lesssim2$–$3$-dimensional
manifold, corroborated by the unrelated method of Lin et al. (2026) on a different summary
statistic. An appendix demonstrates the feasibility of extending the Sobol suite to varying
cosmology via Angulo–White rescaling. Compiled cleanly: 21 pages.

**Remaining `\todo{}` items** (4):
- Render a $(\Omega_{\rm m},\sigma_8,a_1,a_2)$ posterior corner plot from
  `shear_forecast_y1.npz` (not produced under the no-new-analysis rule for this pass).
- Quantitative head-to-head comparison of an optimized scale cut vs. the $N=2$ template basis
  at fixed residual $S_8$ bias — no dedicated analysis exists in this suite.
- Author list to be finalized; acknowledgments to be finalized.

*Coherence note:* during this pass, the Introduction's summary of the $N\ge6$ template ladder
("adding more templates only inflates the statistical error with no further bias reduction")
was found to still carry the pre-correction overstatement even though the Abstract, Results
§4.3, and Conclusions had already been corrected in the per-paper fix pass; reworded to match
the sourced $2$–$3.5\times$ further reduction at $\ell_{\rm max}=5000$.

---

## Paper III — `03_latent_sbi`
**The BIND Lightcone Suite III: The Two-Dimensional Latent Space of Baryonic Feedback, and
What Lensing × SZ Statistics Can Constrain**

Shows the WL suppression $S(\ell)$ collapses onto a 2-D latent space (linear PCA: 98.7% of
variance in two components at $n=216$, consistent at the final $n=253$; corroborated by an
independent $\beta$-TC-VAE), with axes reading cleanly as black-hole and supernova/wind
feedback directions — independently reproducing the Lin et al. (2026) hydro-to-DMO
power-ratio latent found across CAMELS suites. The leading WL axis is nearly identical to the
leading kSZ gas latent axis (canonical correlation 0.98) while the second axis is
substantially rotated (0.79). SBI on the WL auto-spectrum alone leaves a broad 30-parameter
posterior but a tight 2-D suppression-latent posterior (passes SBC). A cautionary methods
result: naively stacking emulated higher-order statistics/$\kappa\times y$ onto the
auto-$C_\ell$ likelihood *degrades* the fit, traced to lower emulator fidelity for those
statistics. Using the actual lightcone halo population (not a single low-$z$ snapshot), the
per-halo lensing perturbation correlates with gas fraction at $r=0.62$, peaking near
$z\approx0.33$. The paper closes with the first direct confrontation of BIND shear$\times y$
against real DES Y3 $\times$ ACT data: BIND over-predicts the correlation by
$\approx2.3$–$2.5\times$, indicating IllustrisTNG feedback is too gas-bound relative to the
observed universe at this level of analysis. Compiled cleanly: 26 pages.

**Remaining `\todo{}` items** (15; largest count in the suite — this is explicitly the most
provisional draft):
- An unresolved 2-D (PCA) vs. $\sim$1-D (field-level, $\kappa\times y$-inclusive) effective-
  dimensionality tension is flagged but not fully reconciled; construction differences
  (\ell-band, $\kappa y$ inclusion) identified, gap not closed.
- The $\sim$8.6M-per-halo population-level SBI posterior (amortized NPE + IID-product
  likelihood + LORO temperature calibration) was designed and partially built but not run to
  a completed GPU result.
- Confirm or re-run the $\kappa\times y$/tSZ multiprobe analysis at $N=253$ (currently
  disclosed as possibly stale at $N=123$).
- Reconcile the Sobol-index vs. Spearman disagreement on the top feedback driver.
- Obtain posterior-$\sigma$/prior-$\sigma$ ratios for 2 of the 6 parameters shown in the NPE
  corner plot to settle whether 4, 5, or 6 are appreciably informed.
- Confirm whether $1.5$–$2\times$ or $2.3$–$2.5\times$ is the intended final DES$\times$ACT
  headline number.
- Two figures need regeneration (emcee marginals overlay; untruncated axis labels) — flagged
  rather than silently fixed, per the no-analysis-rerun rule.
- Coordinate with Paper IV on final ownership of the $\kappa y$-driver and DES$\times$ACT
  confrontation figures.
- Author list, acknowledgments/funding/data-availability to be finalized.

---

## Paper IV — `04_ksz_gas`
**The BIND Lightcone Suite IV: Gas Thermodynamics Across a 30-Parameter Feedback Space,
Confronted with DESI×ACT kSZ and eROSITA**

Paints the same $10^{13}\,M_\odot/h$ TNG300-Dark lightcone halos 256 times (Sobol feedback
sweep) to ask whether the DESI$\times$ACT kSZ / eROSITA "missing baryons" tension reflects the
IllustrisTNG feedback *model* or just an unlucky *parameter* choice. Finds that **zero** of
256 Sobol nodes reach the eROSITA strong-feedback gas-fraction band at any halo mass tested,
including the most extreme feedback tail — the central, most robust result of the paper.
Against the exact DESI$\times$ACT CAP-ratio observable the fiducial run is $1.4$–$1.8\times$
too gas-rich, but $\sim$20% of the 256-node design is statistically consistent with the data
within its errors, so the kSZ tension is comparatively data-limited relative to the eROSITA
one. A CMB-lensing mass anchor confirms the discrepancy is a gas-content, not a
mass-selection, effect. The 30-D $(\tau,y)$ feedback response collapses onto a clean 2-D
inner-gas/outer-gas latent manifold, and a companion field-level (ray-traced) analysis shows
this further collapses to one dominant gas-ejection direction once projected through the
lensing kernel. Repeatedly stresses that BIND's "kSZ" is a velocity-free optical depth, not a
reconstructed temperature decrement. Compiled cleanly: 19 pages.

**Remaining `\todo{}` items** (8):
- Reason for the 3/256 field-level Sobol-node exclusions is untraced in the plan doc/notebook.
- Exact BIND-vs-TNG-truth gas-fraction agreement at the $z<0.2$ mass bin is only qualitatively
  sourced ("$\approx$ TNG truth"); trace a specific percentage if a tighter number is needed.
- Verify the pre-computed Sobol-suite $f_{\rm gas,500}$ column carries the same line-of-sight/
  background-subtraction treatment as the corrected OAT single-knob pipeline before final
  submission.
- One fig01 panel-c label is a stale, un-regenerable in-image caption (documented, not fixed).
- Author list to be finalized; acknowledgments to be finalized.

*Coherence note:* the "closes roughly 40% of the gap to the eROSITA band" single-knob claim
(two occurrences, §Results and §Discussion) used the identical three sourced numbers as
Paper I's now-corrected eROSITA gap-closing statement ($0.076$, $0.047$, $0.026$) and carried
the same closed/remaining-fraction inversion Paper I's fix pass had already caught and
corrected there ($0.029/0.050 = 58\%$ closed, not $40\%$). Corrected both occurrences here to
match Paper I. Also normalized "$z=0.03$–$2.44$" → "$z=0.034$–$2.444$" and the "TNG300-DM"
prose naming → "TNG300-Dark" (two instances) to match Papers I/II's naming convention; the
sourced dataset-directory name in the accompanying `% src:` comment was left unchanged.

---

## Paper V — `05_anisotropy`
**The BIND Lightcone Suite V: How Anisotropic Is the Baryonic Suppression of Weak Lensing?**

A focused Letter using BIND's per-halo painting to build, for the first time, a mass-exact
radial "BCM-warp" spherical counterfactual of a full painted WL field, isolating the fraction
of the convergence-suppression signal that no spherically symmetric baryon-correction model
can reproduce. The corrected, faithful-control fraction is scale-dependent:
$f_{\rm aniso}\approx0.45$ at $\ell=3\times10^3$, falling to $0.24$ at $\ell=10^4$ and
$\approx0$ by $\ell=2\times10^4$ — substantially smaller than an earlier, cruder
azimuthally-symmetric-correction estimate ($f_{\rm mono}\approx0.5$, roughly scale-flat),
which is shown to be dominated at small scales by halo triaxiality a real BCM already
captures rather than by missed feedback physics. An exact cross/auto decomposition of the
power-spectrum difference shows the anisotropic suppression is carried at first order by
alignment between the angular correction and the halo's own DMO lensing signal. The 60-run
one-at-a-time TNG feedback sweep identifies AGN kinetic-mode and stellar-wind parameters as
the strongest controls of the (upper-bound) angular signal, and the fiducial ejection
quadrupole is bipolar and anti-aligned with the halo's DM major axis. Explicitly flags which
parts of the argument (the feedback-driver ranking, the cross/auto mechanism) still rest on
the superseded, cruder spherical control rather than the corrected one. Compiled cleanly:
10 pages — the shortest and most explicitly preliminary draft in the suite.

**Remaining `\todo{}` items** (8):
- A formal bootstrap/significance test for the corrected `bcm_warp`-control $f_{\rm aniso}$
  has not been computed (only the superseded mono-control $z=52.3\sigma$ exists).
- No citation found/available for angular/quadrupole structure in hydrodynamic halo gas maps
  (the one previously attached, van Daalen 2020, was verified not to support the claim and
  was removed rather than replaced with an unverified substitute).
- A quantitative L2/correlation validation of `bcm_warp_patch` itself (of the kind already
  reported for the cruder `mono` control) has not been computed in the available sources.
- Author list to be finalized; acknowledgments to be finalized.

---

## Coherence fixes applied in this pass

1. **`bindI`–`bindV` bibliography entries** standardized across all five `references.bib`
   files to the exact series titles as they appear in each paper's own `\title{}`, with a
   uniform `author = {Lee, Max E.}`, `note = {in preparation}`, `year = {2026}` (previously,
   the same key carried up to four different titles/author-name styles depending on which
   paper's `references.bib` it appeared in, e.g. `bindI` read "BIND I: A conditional
   flow-matching emulator..." in Paper II, "Baryon-painted weak-lensing and SZ
   lightcones..." in Paper III, and a fully different construction in Paper IV).
2. **Paper II, Introduction** — the summary sentence for the $N\ge6$ template ladder still
   read the pre-correction "no further bias reduction," contradicting the already-corrected
   Abstract/Results/Conclusions; reworded to match.
3. **Paper IV** — the "$\sim$40% of the eROSITA gap closed" single-knob claim (two
   occurrences) carried the same closed/remaining-fraction inversion that Paper I's own fix
   pass had already caught and corrected using the identical three sourced numbers; corrected
   to $\sim$58% in both places, consistent with Paper I.
4. **Paper IV** — normalized "$z=0.03$–$2.44$" to "$z=0.034$–$2.444$" (one occurrence) and
   "TNG300-DM" prose references to "TNG300-Dark" (two occurrences) to match the naming used in
   Papers I and II for the identical simulation; the sourced dataset-directory name in the
   accompanying provenance comment was left unchanged.

All five papers were recompiled after these edits
(`pdflatex → bibtex → pdflatex → pdflatex`); all five remain clean (0 LaTeX errors, 0
undefined references/citations, 0 bibtex errors) at unchanged page counts (20/21/26/19/10).

## Items checked and found already consistent

- Simulation-suite description (SB35 Sobol design, 256 nodes/253 usable runs; TNG300-Dark
  DMO lightcones; 20 snapshot shells $z=0.034$–$2.444$; painted-halo mass floor
  $\sim10^{13}\,M_\odot/h$; $4\times R_{200c}$ circular paste) agrees everywhere it is stated
  (Papers I, II, III, IV use the Sobol design; Paper V's 60-run OAT design is a deliberately
  different, smaller sweep and is described as such, not conflated with the Sobol suite).
- The retracted "$\sim$41% of cosmic" gas-fraction saturation figure is flagged, explained
  (line-of-sight/projection-contamination artifact), and explicitly not used as a result in
  both Paper I (§Discussion) and Paper IV (§Results), with matching explanations.
- `bindI`–`bindV` citation keys are exactly balanced against each paper's own
  `references.bib` (every key cited in a paper's `main.tex` is defined in that paper's bib,
  and vice versa) in all five papers.
- No near-verbatim duplicated Methods paragraphs were found; Papers II–V describe the shared
  pipeline machinery briefly and cite Paper I rather than reproducing its construction
  details.
- The Lin et al. (2026) two-dimensional-latent corroboration is characterized consistently
  between Papers II and III (same paper, same "independent method/different summary
  statistic" framing); the two papers use different internal `\citet{}` keys for it
  (`lin2026` vs. `Lin2026latent`), which is permitted — the task's key-consistency
  requirement applies only to the sibling `bindI`–`bindV` self-citations.
