# Paper I brief — BIND lightcones: methods + validation + bridge

**Working title:** The BIND Lightcone Suite I: Baryon-painted weak-lensing and
Sunyaev–Zel'dovich lightcones from dark-matter-only simulations
**Format:** full methods+validation paper (~12–18 pp). This is the anchor the
other four papers cite for the pipeline.

## Thesis
A conditional flow-matching emulator trained on CAMELS-TNG (BIND) can paint
baryonic fields (gas, stars, DM response, Compton-y, electron column) onto
halos of a dark-matter-only TNG300 *lightcone*, producing ray-traced κ, y, and
τ maps that statistically match the true TNG300 hydro lightcone at fiducial
parameters — for ~zero marginal GPU cost per feedback variation. The per-halo
gas response and the field-level WL suppression are quantitatively linked
(the "bridge"), so the tool connects halo gas astrophysics to survey
observables in both directions.

## Sources (read in this order)
1. `docs/WORKLOG.md` entries: 2026-06-10 (×3: truth lightcone, validation
   notebook + DMO baseline, BIND validated + R(ν) deep-dive), 06-11
   (publication plan + three findings), 06-15 pm (halo↔field bridge; f_gas
   confrontation; Rung 1–2 profiles/morphology; synthesis), 06-16 (paper
   notebooks), 06-17 (three-way kappa), 06-18 (probes notebook, tau plane,
   Paper I/II split, f_gas→S(ℓ) bridge anatomy), 06-25 (low-mass reuse),
   plus the 06-21 SB35 Sobol pipeline entry for the suite description.
2. Worktree `wt/wl-tsz-bridge`: `docs/wl_tsz_plan.md` (the Paper I/II/III plan
   — follow its framing), `examples/draft.tex` (an existing partial draft —
   harvest its intro/structure, don't discard), `examples/paper_wl_tsz.ipynb`
   (argument-first paper notebook), `examples/paper_lightcone_figs.ipynb`,
   `examples/paper_methods_figs.ipynb`, `examples/bind_bridge.py`,
   `examples/fgas_bridge_explore.ipynb`, `examples/lightcone_probes.ipynb`.
3. Worktree `wt/lightcone`: `examples/fiducial_lightcone_stats.ipynb`
   (validation stats), `src/bind/inference/{lightcone_maps,truth_lightcone,lensplane,stats}.py`
   (methods details), `examples/lightcone_lowmass_reuse.py`,
   `examples/resolution_gate.py`, `CLAUDE.md` §"What this project is".

## Section outline
- **Intro:** baryonic feedback vs WL surveys (van Daalen suppression; BCM vs
  hydro vs field-level emulation); the gap = per-halo generative painting at
  lightcone scale; contributions list.
- **Methods:** (a) BIND recap (conditional flow matching, 35-param CAMELS
  SB35 conditioning, thermo channels — cite in-prep model paper, keep short);
  (b) lightcone construction: TNG300-Dark multi-snapshot stages, per-halo
  painting, composite pasting (4×R200c circular aperture), lens planes, lux
  multi-plane ray tracing, truth lightcone from TNG300 hydro, paired DMO
  trace; (c) map suite: κ(z_s), Compton-y, τ/electron-column (=FRB DM);
  (d) statistics pipeline (Cl, peaks, PDF, MFs; MAS/CIC correction).
- **Results:** (1) fiducial validation BIND vs hydro truth (WL Cl + peaks +
  PDF; y auto-spectrum; τ) — the headline "matches" figure(s); (2) R(ν)
  response tracks f_gas·T (thermal energy, not gas fraction alone);
  (3) the bridge: Δν↔ΔM_gas r=0.93; group f_gas ↔ S(ℓ) van Daalen relation
  r=0.91 (Fig. 16 analog for WL); z≈0 group f_gas saturates at ~41% of the
  cosmic baryon fraction across the whole feedback design; (4) completeness:
  low-mass (1e12–1e13) reuse capture 28.7% @4×R200 / 51.6% @full patch;
  resolution gate.
- **Discussion:** TNG-prior caveat; ≥1e13 mass floor for painting; τ is
  velocity-free electron column (kSZ needs velocity — Paper IV); CIC/aliasing
  upturn at high ℓ (`--mas_correct`; ratios cancel it); what Papers II–V do
  with the tool.
- **Conclusion** + data/code availability (HF `Maxelee/BIND2`, GitHub).

## Numbers that MUST appear (verify each against WORKLOG before use)
BIND≈hydro at fiducial for WL+tSZ observables; R(ν)∝f_gas·T; bridge r=0.93 /
r=0.91; f_gas saturation ~41% cosmic; low-mass capture 28.7%/51.6%; TNG300
box + n(snapshots)=20, shells z=0.034–2.444; 4×R200c paste aperture.

## Figures (candidates)
`examples/figures_lightcone/` on disk (81 files — inspect, pick the
validation/map images); embedded figs from `paper_wl_tsz.ipynb`,
`paper_lightcone_figs.ipynb`, `paper_methods_figs.ipynb`,
`fiducial_lightcone_stats.ipynb`, `fgas_bridge_explore.ipynb`. Target 8–12
figures: map gallery (κ/y BIND vs truth), validation spectra ratios, R(ν),
bridge scatter plots, van Daalen analog, completeness.

## Seed citations (verify all)
van Daalen+2011, van Daalen+2020; Schneider & Teyssier 2015; Aricò+2020/21;
Chisari+2019; Mead+2021 (HMcode-2020); Villaescusa-Navarro+2021/2023 (CAMELS +
SB35/CAMELS-SAM?); Pillepich+2018, Springel+2018, Nelson+2019 (TNG300);
Lipman+2022 (flow matching); DES/KiDS/HSC S8 refs; FRB DM (Macquart+2020);
Planck y-map (Planck 2015 XXII). BIND model paper = in prep.

## Mandatory caveats
TNG-only prior; mass floor; velocity-free τ; aliasing at high ℓ; painting is
2D projected (not 3D fields).
