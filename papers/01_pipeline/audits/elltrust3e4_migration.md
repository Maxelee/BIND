# ELL_TRUST migration: 1.5×10⁴ → 3.0×10⁴ (2026-08-04)

Author ruling implemented this session (evidence in the setup-cell comment of
`_build_figures_nb.py`, reproduced here):

- **(a)** raw `C_ℓ^κκ` log-slope (fiducial + DMO, z_s=1) keeps STEEPENING through
  1.5–3×10⁴ (−2.3→−2.5) and only hardens (−2.5→−1.7, the actual CIC/pixelization
  upturn) in the 3×10⁴–3.7×10⁴ band → measured contamination onset ≈3×10⁴ ≈
  0.8·ℓ_Nyq (ℓ_Nyq=36864).
- **(b)** Sobol response-ratio 16–84 half-widths grow SMOOTHLY (25%→89% from
  5×10³ to 5.3×10⁴) with no cliff — the old "explosion above 1.5×10⁴" claim does
  not reproduce on the corrected (xpkfix/nu05) dataset.

Ruled: axes stop at axis-Nyquist (36864), the 3.7–5.2×10⁴ corner-mode zone is
dropped from every plot; large shaded aliasing regions are removed; RAW-spectrum
panels get a thin vline at the measured onset (3×10⁴) instead; ratio/residual
panels (S(ℓ), residual strips, fig 7 everything) carry no marker and run to
36864 unmarked; and the analysis trust cut (χ² ranges, masks) moves to 3×10⁴
everywhere.

Backup of the pre-migration builder: `audits/_build_figures_nb.pre-elltrust3e4.py`.

## 1. Change-site list

### `papers/01_pipeline/_build_figures_nb.py`

- **Setup cell** — `ELL_TRUST` 1.5e4→**3.0e4**, `ELL_MAX_PLOT` 5.22e4→**36864.0**;
  full comment block rewritten with the (a)/(b) measurement provenance above.
  Added a new shared helper, `ell_trust_marker(ax, label=..., fontsize=...)`,
  drawing the thin vline (+ optional "CIC/pixel upturn (measured, 0.8 ℓ_Nyq)"
  label) — used by every raw-spectrum panel below instead of the old
  `axvspan` shading.
- **Fig 4, panel (a)** (`Cl_κκ`, the only raw-spectrum panel in fig 4 —
  panel (b) is `S(ℓ)`): the old `for _ax in (a, ar): axvspan(...)` + separate
  "CIC\naliasing" text call replaced with `ell_trust_marker(a, ...)` on the
  main axis only (never the residual strip). §2b markdown rewritten: "corner
  mode ℓ=5.2×10⁴ … aliases above ℓ≈1.5×10⁴ (shaded)" → axis-Nyquist +
  measured-onset phrasing.
- **Fig 4b** (6 spectra panels: κκ, yy, ττ, κy, κτ, yτ — all raw, unratioed):
  removed the `for _ax in (a, ar): axvspan(ELL_TRUST, ELL_MAX_PLOT, …)` loop;
  replaced with `ell_trust_marker(a, label=(j==0))` — vline on all six main
  axes, the explanatory label drawn once (panel a) to avoid repeating it six
  times, matching the file's existing "no in-panel clutter" convention for
  this figure. Markdown ("ℓ to the corner mode with the CIC-aliasing region
  shaded") and the "Reading" paragraph (excess-window bounds, see §3 below)
  rewritten.
- **Fig 18** (Appendix A, mechanism of fig 4's high-ℓ `Cl_yy` misfit): the
  "fig-4 high-ℓ excess window" — previously hardcoded `5e3 < ℓ < 1.5e4`
  (`mb18` mask, the `axvspan(5e3, 1.5e4, …)` annotation, its text-label
  x-position, and the print stamp) — now reads `5e3 < ℓ < ELL_TRUST`, i.e. it
  widens to 3×10⁴ automatically with the notebook global. **Flagged, not
  fabricated**: the "+10–20% excess" magnitude quoted in the cell's header
  comment was measured over the OLD, narrower window and has **not** been
  re-verified over the wider one in this pass (fig 18 needs a rerun — see
  §4 pending list).
- **Fig 7** (§3b, all 12 canonical statistics as response RATIOS): markdown
  rewritten ("share one ℓ axis to the full corner mode … CIC-aliasing region
  shaded" → axis-Nyquist, no marker/shading anywhere because every panel here
  is a ratio). Code already had no `axvspan`; only a stale comment referencing
  "the aliasing-region behaviour of S(ℓ) stays visible in figs 4b/12" was
  corrected (that shading no longer exists there either). `mS`/`mE` masks
  already read `ELL <= ELL_TRUST` — inherit the new value automatically,
  verified.
- **Fig 12** (§3d, `S(ℓ)` envelope only — confirmed NOT a raw spectrum):
  removed `ax[0].axvspan(ELL_TRUST, ELL_MAX_PLOT, …)`; comment rewritten to
  state explicitly why no marker is drawn (ratio statistic, cancels the
  artifact). `mS = ELL <= ELL_MAX_PLOT` already inherits the narrower Nyquist
  cutoff automatically.
- **Fig 9c** (GP-predicted vs. true curves, 13 canonical-stat panels): added
  `ell_trust_marker` to the six RAW ell-domain spectrum panels
  (`k in ELL_MASK_HEADS_NB`: cl_kappa, cl_yy, cl_tt, cl_kappa_y,
  cl_kappa_tau, cl_yt) — vline on all six, label on the first (`cl_kappa`)
  only, positioned to avoid the existing per-panel metrics badge and panel
  label. `suppression` (a ratio) and the six nu-domain panels left unmarked.
  Fig 9's own figure (S(ℓ) only) and fig 9b (bar charts, not ell-domain)
  confirmed to need no marker. `mE`, `mEc`, `_ell_trusted` all already
  inherit `ELL_TRUST` — verified, no code change needed.
- **Fig 8** (§4.0 emulator methodology table): "(ell≤1.5×10⁴)" annotations →
  "(ell≤3×10⁴)"; "C-head improvement set" paragraph rewritten to name
  `ELL_MASK_ABOVE` symbolically with the 2026-08-04 provenance. **Table
  itself replaced with the fresh post-refit numbers** (§2 below); the old
  table is kept immediately after, explicitly labeled PRE-MIGRATION, for
  provenance/comparison.
- **Fig 11** (§ redshift closure, ratio panels): confirmed `set_xlim(90,
  ELL_TRUST)` already inherits the new value; no marker needed (ratio); no
  change required.
- **Fig 20** (§3c van Daalen matrix): "five geometric ℓ-bands (ℓ=300–
  1.5×10⁴)" → "300–3×10⁴" (matches `LEDG = np.geomspace(300.0, ELL_TRUST,
  6)`, which already inherited automatically — verified). **Flagged**: the
  per-band Pearson-r numbers quoted in this section's prose were measured on
  the OLD band edges and have not been re-verified on the new ones (fig 20
  needs a rerun).
- **Fig 10** (§4c astro-parameter posterior) **and the §4b″ 50-node coverage
  test** (which shares the identical data-compression/covariance recipe):
  markdown ℓ-range updated (100–1.5×10⁴ → 100–ELL_TRUST=3×10⁴). **A real bug
  was found and fixed here** — see §3 below.
- **§6 "Statistics" prose**: "Raw spectra alias above ℓ∼1.5×10⁴ (ratios
  cancel it)" → measured-onset phrasing citing the 2026-08-04 ruling
  (this is the exact sentence the task named).

### `papers/01_pipeline/audits/map_rescale_check.py`

Standalone audit script whose own header states it reproduces the notebook's
`paired()`/`ELL_TRUST` convention exactly, as a sanity check against the fig-4
cell's printed χ²/dof. Updated `ELL_TRUST` 1.5e4→3.0e4, the `windows` list's
high-ℓ tuple `(5000, 15000)`→`(5000, int(ELL_TRUST))`, the VERDICT f-string,
and the "sanity check" comment (which used to assert equality with the OLD
notebook number, 177 — now explicitly notes the two are not expected to match
post-migration and that *this script's own number is the projection* for a
re-rendered fig 4). **Re-run** (cheap: an 11 MB cached npz, no map data) to
get real, fresh numbers — used directly in the fig 4/4b markdown rewrite
above. See §2 for the numbers.

### `src/bind/emulator/`

- **`core.py`**: `ELL_MASK_ABOVE` 1.5e4 → **3.0e4**; the module comment above
  it rewritten with the full measurement provenance and an explicit note that
  `build_emulator.py` mirrors this constant deliberately (update both) and
  that `emulator_cache.py`'s fingerprint already captures it by live import
  (verified, see below — no code change needed there).
- **`dataset.py`**: `StatSpec.mask_ell_above` docstring updated; all six
  hardcoded `mask_ell_above=1.5e4` entries in `STAT_SPECS` (cl_kappa,
  cl_kappa_y, cl_yy, cl_kappa_tau, cl_tt, cl_yt) → `3.0e4`. **Note**: this
  field only takes effect for datasets *assembled* after this change — the
  currently-released `emulator_dataset_nu05.npz` carries `Target.
  mask_ell_above=None` for every head, so `core.ELL_MASK_ABOVE` (not this
  file) is what actually governed the fit just run. Updated anyway for
  consistency, so a *future* re-assembly does not silently regress to the
  old threshold.
- **`transforms.py`**: docstring "CIC-aliased ell > 1.5e4 tail" →
  "CIC/pixelization-contaminated ell > ELL_MASK_ABOVE tail … 3.0e4 as of
  2026-08-04, was 1.5e4".
- **`papers/01_pipeline/build_emulator.py`**: mirrored `ELL_MASK_ABOVE`
  1.5e4→3.0e4 (module-level constant, deliberately duplicated not imported —
  its own comment says so); module docstring's C-head-improvement-set item 1
  updated to match.

**NOT touched** (deliberately): `audits/spectrum_head_experiment.py`/`.md`,
the archival controlled A/B experiment that originally justified ell-masking.
It is a dated, completed study whose own quoted numbers ("median 0.59 on the
trusted range") were measured under the *then-current* 1.5e4 cut; editing its
`ELL_TRUST` constant without re-running the actual GP fits inside would make
the file internally inconsistent (code says one thing, printed numbers say
another) — worse than leaving it as an honest historical record. Also not
touched: `FIGURE_NUMBERS.md` / `SCORECARD.md`, which are explicitly
"regenerate by re-running the notebook" crib sheets extracted from an
executed run (job 2454934) — several of their lines still say "ell 100-1.5e4"
and will refresh automatically the next time the full notebook executes.

## 2. Refit: before → after (the honest deltas)

`python build_emulator.py --fit` (315 s, CPU, gpgpu backend) then `--verify`
against `emulator_dataset_nu05.npz` (206 train / 50 held-out, seed 0).
`emulator_cache.py`'s fingerprint correctly refused the stale bundle first —
confirmed live: `--verify` against the *old* bundle printed `[emulator_cache]
saved fit does not match this setup (differs in: emulator_config) ->
refitting` and exited 1, *before* any `--fit` was run. The new bundle's
sidecar (`paper1_gp.json`) now records `"ell_mask_above": 30000.0` — the
fingerprint captures the mask value with no code change needed (it imports
`ELL_MASK_ABOVE` from `bind.emulator.core` live).

| head | transform | k (old→new) | frac err OLD (ℓ≤1.5e4) | frac err NEW (ℓ≤3e4) | Δ | R² OLD | R² NEW |
|---|---|---|---|---|---|---|---|
| `suppression` | raw | 12→12 | 3.33% | **3.33%** | unchanged (unmasked ratio) | 0.74 | 0.74 |
| `cl_kappa` (composed, off-diag) | asinh_std | 2→2 | 14.70% | **15.51%** | +0.81pp (+5.5% rel) | −0.02 | −0.02 |
| `pdf` | log1p | 12→12 | 0.94% | **0.94%** | unchanged (nu-domain) | 0.81 | 0.81 |
| `peak_counts` | log1p | 12→12 | 0.70% | **0.70%** | unchanged | 0.51 | 0.51 |
| `minima_counts` | log1p | 12→12 | 0.46% | **0.46%** | unchanged | 0.45 | 0.45 |
| `mf_v0` | raw | 12→12 | 0.13% | **0.13%** | unchanged | 0.77 | 0.77 |
| `mf_v1` | raw | 12→12 | 0.56% | **0.56%** | unchanged | 0.77 | 0.77 |
| `mf_v2` | raw | 12→12 | 0.94% | **0.94%** | unchanged | 0.78 | 0.78 |
| `cl_yy` | log | 12→12 | 4.74% | **7.77%** | +3.03pp (**+64% rel**) | 1.00 | 1.00 |
| `cl_tt` | log | 8→7 | 4.59% | **7.56%** | +2.97pp (**+65% rel**) | 0.87 | 0.85 |
| `cl_kappa_y` | asinh_std | 3→4 | 14.21% | **26.42%** | +12.21pp (**+86% rel**) | −0.02 | −0.02 |
| `cl_kappa_tau` | asinh_std | 4→5 | 19.70% | **32.27%** | +12.57pp (**+64% rel**) | −0.02 | −0.02 |
| `cl_yt` | asinh_std | 3→3 | 18.18% | **27.77%** | +9.59pp (**+53% rel**) | −0.02 | −0.02 |

**Not spun**: every one of the six ell-masked spectrum heads got substantially
worse — roughly +53% to +86% *relative* held-out error, not "some" cost. The
mechanism is exactly `spectrum_head_experiment.md`'s original finding run in
reverse: masking the untrusted tail is *why* these heads' PCA+GP fit well in
the first place (a wider, noisier, larger-dynamic-range bin range pollutes
the compression); moving the mask edge outward to a MORE PHYSICALLY CORRECT
contamination onset necessarily hands the fit more of that noisy range. The
suppression head and all six nu-domain heads (pdf/peak/minima/MFs) are
completely unaffected, as expected (unmasked ratio / not ell-domain at all).
This is a real, deliberate trade-off between trust-boundary correctness and
emulator accuracy on six heads — reported here for the author to weigh, not
absorbed silently.

The fig-8 table in the notebook now carries these fresh numbers as primary,
with the pre-migration table kept immediately below, explicitly labeled, for
comparison.

## 3. A second, independent finding: Hartlap covariance rank bug (found + fixed)

Not on the original task list — found during the sweep and judged serious
enough to fix rather than merely flag. Fig 10's (and the §4b″ 50-node
coverage test's, which shares the same `compress()`/covariance recipe) data
vector was `ℓ∈[100, ELL_TRUST]` rebinned by a hardcoded block size `RBK=8`.
At the OLD `ELL_TRUST=1.5e4` this gave `N_b=25` bins from `n_real=50` paired
realizations (Hartlap factor `(50−25−2)/(50−1)=0.47`, matches the quoted
value). Computed directly on `emulator_dataset_nu05.npz`'s actual 724-point ℓ
grid:

```
ELL_TRUST=1.5e4: 206 raw bins in-range -> RBK=8 -> N_b=25  (OLD, fine)
ELL_TRUST=3.0e4: 415 raw bins in-range -> RBK=8 -> N_b=51  (NEW, BROKEN)
```

At the new `ELL_TRUST` with the *old* `RBK=8`, `N_b=51 > n_real−2=48`, which
makes the Hartlap factor **negative** (`(50−51−2)/(50−1) = −3/49`), silently
flipping the sign of the fitted covariance matrix and corrupting every
downstream likelihood evaluation in both cells (it also breaks the
Percival–Dodelson–Schneider inflation factor a few lines later, whose
denominator needs `n_real − N_b > 4`). This would not have crashed — it would
have produced a plausible-looking but wrong posterior the next time either
cell was executed with the widened `ELL_TRUST`.

**Fix applied**: bumped `RBK` 8→**16** in both the fig-10 cell and the §4b″
coverage-test cell. `415 // 16 = 25` — this restores `N_b=25` **exactly**,
identical to the pre-migration bin count, so `h=0.47`, the Percival/DS
factors, and every other number in that section that depends only on
`(n_real, N_b)=(50,25)` are unaffected; only the per-bin ℓ-*width* of the
compression changes. Added an explicit `assert NRl - NB - 2 > 0` (with a
clear message) right after both cells' `NRl, NB = ....shape` lines, so any
future re-widening of `ELL_TRUST` without a matching `RBK` bump fails loudly
instead of silently corrupting the likelihood. Markdown updated to state the
fix and the reasoning; the posterior itself (emcee run) has **not** been
re-executed in this pass — see pending list.

## 4. Gates

- `ruff check src/bind/emulator` — **clean**.
- `ruff check papers/01_pipeline/build_emulator.py audits/map_rescale_check.py` — clean (not required by the task but checked).
- `python _check_builder.py` — **0/29 code cells with syntax errors**, notebook rebuilds cleanly, all 24 `save(fig, ...)` targets present. Run twice (before and after the fig-8 table rewrite); exit code 0 both times.

## 5. Renders: verified vs. pending

- **`fig12_survey_context`** — rendered via `_run_subset.py fig12_survey_context`
  (8.9 s wall, ~76 MB RSS). **Visually verified** (`figs_preview/
  fig12_survey_context.png`, read and inspected): panel (a) `S(ℓ)` shows no
  shaded region, axis runs to ~3.7×10⁴ (≈36864), no CIC marker present —
  matches the ruling exactly (ratio panel, unmarked, to axis-Nyquist).
- **`fig07_covariation`** — **attempted, OOM-killed**, not rendered. First
  attempt hung at kernel-startup (`do_sys_poll`, defunct child, no progress
  for >3 min) and was killed as unresponsive; a clean retry spawned a new
  kernel that climbed to **13.2 GB RSS** and was killed by the cgroup OOM
  killer (`dmesg`: `oom-kill:constraint=CONSTRAINT_MEMCG …
  oom_memcg=/slurm/uid_2107/job_2457196 … Killed process (python3)
  anon-rss:13234100kB`) — confirms this session's ~13 GB usable-RAM ceiling
  exactly. This contradicts the brief's "(~1–2 GB)" expectation for this
  figure on this node; either the node's available headroom is currently
  smaller than when that was last true, or the nu05-migrated dataset /
  freshly-persisted emulator bundle increased this cell's footprint. **Not
  rendered — pending the next sbatch figure run.**
- **`fig04b_spectra`** — **not attempted**. Given fig07 (expected lighter)
  just OOM'd at the node's hard ceiling, and `fig04b_spectra` structurally
  requires `fig04_field_validation` as a prerequisite (`_run_subset.py`'s
  `DEPS` map: fig04b reads `ell_f`/`f_ell`/`NR`/`paired()`/`KYb`/`KYt` from
  the fig-4 cell), which is independently known to OOM on this node, a
  fig04b attempt would almost certainly also fail. Skipped rather than
  burning another OOM cycle. **Pending the next sbatch figure run.**
- **`fig04_field_validation`, `fig05a_sl_response`** — skipped per the
  original task brief (known OOM on this node).
- **Everything emulator-dependent** (`fig09`, `fig09b`, `fig09c`, `fig15`,
  `fig15b`, `fig10`) and **`fig18`/`fig20`** (whose science-content windows
  widened along with `ELL_TRUST` and are flagged above as not yet
  re-verified) are **pending** a full run — `sbatch run_figures.sbatch` on a
  GPU/high-mem node is the recommended next step, both to visually confirm
  the marker placement on fig 4/4b/9c's raw panels and fig 7/12's absence of
  one, and to regenerate `FIGURE_NUMBERS.md`'s stale "ell 100-1.5e4" lines
  and the fig 18/20 numbers flagged above as unverified over their widened
  windows.

## Summary of what changed vs. what's left

**Done and verified**: the `ELL_TRUST`/`ELL_MAX_PLOT` convention rewrite
across every figure cell and markdown paragraph in `_build_figures_nb.py`
(shading removed, marker added only to raw-spectrum panels, ratio panels
left unmarked and unclipped to axis-Nyquist); the `bind.emulator`
`ELL_MASK_ABOVE` migration (core/dataset/transforms/build_emulator.py, all in
lockstep, fingerprint-safe); the emulator refit (numbers above, reported
honestly); both syntax/lint gates; `fig12`'s render, visually confirmed
correct. **Found and fixed beyond the original task list**: the Hartlap
covariance rank bug in fig 10 / §4b″. **Left pending, explicitly**: `fig07`
(OOM this session), `fig04b`/`fig04`/`fig05a` (OOM), every emulator-figure
render, and the fig 18/20 science-content numbers whose windows widened but
whose underlying values were not re-measured in this pass — all deferred to
the next `sbatch run_figures.sbatch`.
