# Leftovers — Paper IV (04_ksz_gas)

Results and figures the dossier surfaced but that this draft deliberately did
not include, with one line each on why.

## Figures left out (dossier candidates not used)

- **`f6a_params_fit` (notebook cell016_out0.png)** — strip plot of the ~12
  feedback parameters for the 49 kSZ-CAP-consistent nodes vs the full 256.
  The underlying numbers (which parameter combination is pulled) are reported
  in prose in §3.4 (Results 4); the figure itself was cut to stay within the
  brief's 8–10 figure target, favoring the MONEY PLOT forecast (Fig. 8) as the
  more visually compelling summary of "which directions are constrained."
- **`f6b_lowmass` (notebook cell018_out2.png)** — two-redshift (BGS/ELG)
  f̃_gas(M200) panel showing the tension holds across mass and redshift. The
  headline number (~2× tension at z=1.16) is reported in prose in §3.3
  ("The tension holds across mass and redshift" paragraph); cut as a figure
  to make room for the field-level companion figures, which the brief
  explicitly asks to fold in as a section.
- **`f6c_kappa_mass` (notebook cell020_out1.png)** — CMB-lensing κ(θ) mass
  anchor. The headline number (BIND κ ~1.3× data at 1-halo scale → masses
  right to ~30%) is reported in prose in §3.3 ("The mass anchor" paragraph);
  cut as a figure for the same space reason as f6b.
- **`f6e_latent_data` (notebook cell024_out0.png)** — real DESI×ACT data
  projected into the (ê1, ê2) latent plane with kSZ-only/tSZ-only/joint
  contours. This is a natural pairing with Fig. 4 (the latent itself) and
  Fig. 8 (the forecast), but was cut to stay within the 8–10 figure target;
  its content (kSZ+tSZ agree at the strong-fb edge post mass-fix) is covered
  in prose in §3.5.
- **`f7_posterior` (notebook cell026_out0.png)** — posterior-sd/prior-sd for
  ~30 parameters across the 3 nested M*-cut configurations. The underlying
  numbers (0→6 params <0.85× prior, min ratio 0.87→0.75, ELG cuts barely
  move it) are reported in prose in §3.4; cut in favor of the MONEY PLOT
  (Fig. 8), which makes the "kSZ+tSZ complementary" point more directly.
- **`paper_ksz_field/cell014_out0.png`** (κ×y is the feedback driver, D
  ranking) — the ranking numbers (yτ 0.18 > ττ 0.16 > yy 0.135 > κy 0.10 ≫
  κκ 0.03) are reported in prose in §3.6 ("κ×y, not κ auto, carries the
  feedback signal" paragraph); cut as a figure to keep the field-level
  section to 2 figures (matching the brief's "1-2 from the field companion"
  guidance) rather than 3.
- **`paper_ksz_field/cell016_out0.png`, `cell018_out0.png`** (field-level
  per-param posterior + forecast ellipse) — the headline forecast numbers
  (dir1 ν≈0.05 ≪ dir2 ν≈0.11) are reported in prose in §3.6's closing
  paragraph; cut for the same space reason, and because the per-halo MONEY
  PLOT (Fig. 8) already carries the paper's main forecast punchline.
- **`paper_ksz_field/cell004_out0.png`, `cell006_out0.png`, `cell008_out0.png`**
  (field-level §1 DATA / §2 METHODS / §3 response-fan panels) — these are the
  ray-traced-lightcone analogs of the already-included per-halo Fig. 1
  (lightcone design + HMF), Fig. 2 (methods/CAP schematic), and Fig. 3
  (response fan); shown once (per-halo) rather than twice (per-halo and
  field-level) to stay within the figure budget, since the underlying
  suite/pipeline/response-shape content is identical between the two framings
  and only the field-level *result* (Fig. 9, Fig. 10) differs.

## Results reported in prose only (numbers included, no dedicated figure)

All of the above numbers are stated in the text with `% src:` provenance
comments even though their figures were cut, per the instruction that figures
are a curated subset and numbers should still appear with sourcing.

## Content mentioned in the dossier but not used anywhere (number or figure)

- **GNFW (α,β) point comparison** — the dossier flags this as an invalid
  comparison (α≈0.2 near-singular corner); mentioned only qualitatively in
  §3.2 as a caveat about why the β-only / profile-overlay comparison is used
  instead. The specific numeric α values are given but the (α,β) *point* is
  explicitly not used as evidence, per the dossier's own recommendation.
- **The unphysical forward-integrated GNFW f_gas calculation** (dossier
  caveat 8: "gives BGS_all → 0") — not reproduced anywhere in this paper, as
  instructed by the dossier.
- **σ_v = 300 km/s / T_CMB σ_v/c = 2727 µK, and the 1176 µK digitization
  slip** — mentioned in the dossier as a methods footnote relevant only if a
  raw CAP-τ number were shown; since the raw CAP-τ panel was dropped
  (per the dossier's own note, "Fig 6 CAP panel DROPPED"), this paper does
  not show any raw-τ number and so omits this footnote entirely rather than
  include a footnote about a number that never appears.
- **CAP posterior covariance caveat (diagonal / AR(1) approximate)** — this
  IS included, but only as one line in the Discussion "Selection effects"
  paragraph and the closing sentence of §4.2 (Discussion); not elevated to a
  full caveat bullet in §3.7 to avoid diluting the itemized caveat list
  down to items the brief did not explicitly mandate. (It is still present in
  the text, just not bulleted.)

## Sections/framing choices

- The brief's "~41% cosmic f_gas saturation" MUST-appear number is included
  ONLY as a retracted/superseded claim in §3.1, per the dossier's explicit
  correction guidance ("the 06-25/06-26 entries CORRECT earlier numbers —
  always use the post-correction values"). The corrected replacement number
  (0.047, 1.8× eROSITA, single-knob OAT) is what is used as the paper's
  actual "reachable ceiling" statement. This is a deliberate reading of the
  brief's own instruction to prefer post-correction values over the literal
  wording of its "numbers that MUST appear" list, which itself pre-dates the
  correction.
- DESI Collaboration (2024), Hahn+23 (BGS), Zhou+23 (LRG), Raichoor+23 (ELG)
  are confirmed in the dossier's citation table but not individually cited by
  key in the running text — the paper refers to "DESI" and the BGS/LRG/ELG
  samples generically rather than attaching a citation to every mention, to
  avoid over-citing a survey-selection detail that is not this paper's focus.
  Flagged in refs_needed.md in case a revision wants per-sample citations.
