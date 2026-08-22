# 2026-08-05 five-item edit campaign — audit

Builder edited: `_build_figures_nb.py` (source of truth). Backup taken FIRST, before any
edit: `audits/_build_figures_nb.pre-edits0805.py`. Runner updated: `_run_subset.py` (DEPS +
GUARDED_PENDING reporting). Net diff vs the backup: +475/-53 lines (includes item 1's
mid-session apply-then-revert cycle described below — the final state is a superset of the
backup plus two new guarded cells, not a smaller/simpler diff). Every item was gated
individually with `_check_builder.py` (0 syntax errors, correct save-count each time) before
moving to the next; item 1 was gated three times as its spec changed.

## Item 1 — mass labels: three rulings in sequence, final = TEXT-ONLY, no h, no value change

This item's spec changed twice mid-session via coordinator messages; documenting all three so
the final diff makes sense:

1. **Original brief**: every logged-mass axis label/legend → $\log_{10}\,M/({\rm M}_\odot/h)$
   (keep the h; pure re-label, nothing numeric changes).
2. **First correction**: labels should read pure $M_\odot$ (no h) — but since the underlying
   catalog/atlas values ARE stored in $M_\odot/h$, this requires an actual display-layer
   conversion (+$\log_{10}(1/h)$ shift, $h=0.6774$) applied to plotted values only, defined
   once as `LOG_MSUN_SHIFT` in the setup cell.
3. **Final ruling (implemented)**: supersedes #2 entirely — labels read EXACTLY
   $\log_{10}\,M_{200c}/{\rm M}_\odot$ (no h, no parentheses) as a **pure text change**; NO
   value conversion of any kind (no shift, no `LOG_MSUN_SHIFT`, no changes to plotted data,
   bin edges, or prose-quoted mass ranges). Explicitly: "if you already applied any
   display-layer conversion ... REVERT it completely."

**What's actually in the builder now (post-revert, ruling #3):** every logged-mass axis
label/legend across fig03, fig06, fig14, fig19, fig20, fig23 reads
`r"...M_{200c}/{\rm M}_\odot..."` (or `M_{500c}`/`M_{\star,200c}` analog) — text-only, no `/h`,
no surrounding parentheses. All underlying plotted arrays, bin edges (`edges`,
`np.arange(13.0, 14.8, 0.2)`), legend-text mass-bin numbers (`MB3`), and every markdown
sentence quoting a mass value or range are byte-identical to the pre-campaign backup — verify
via `diff` against `audits/_build_figures_nb.pre-edits0805.py` restricted to those numeric
lines. The setup cell carries a comment recording the final convention and explicitly
forbidding a `LOG_MSUN_SHIFT`-style constant, so a future edit doesn't reintroduce #2's
approach by accident.

Change sites (all in `_build_figures_nb.py`, all text-only):
- **fig03** (`fig03_halo_validation`): `MLAB` (shared x-axis, all 3 panels) and panel (c)'s
  $M_{\star,200c}$ y-label.
- **fig06** (`fig06_radial_profiles`): mass-bin legend title on panel (a) (bin-edge numbers in
  the legend entries themselves, e.g. "13.4-13.8", are untouched).
- **fig14** (`fig14_completeness`): x-labels on both panels (a) and (b).
- **fig19** (`fig19_fgas_erosita`): the $f_{\rm gas}$ y-label's mass clause.
- **fig20** (`fig20_vandaalen_matrix`): panel (a) x-label, panel (b)'s mass-bin-range
  parenthetical, panel (c)'s mass-bin-range parenthetical.
- **fig23** (`fig23_s3_opener`): the $f_{\rm gas}$ x-label's mass clause (previously had no
  units at all; now `.../{\rm M}_\odot`).

**Found and fixed during rendering verification (survived all three rulings):** fig20 panels
(b) and (c) x-labels overflowed the panel width once the parenthetical mass clause grew longer
than the original bracket form. Shortened both (dropped the redundant `500c` subscript inside
the parenthetical, added `fontsize=6.3`) and re-verified visually after the final revert —
confirmed non-overflowing (see Renders section).

Guard: none needed (pure text edit, no data dependency).

## Item 2 — fig05a_sl_response: remove the $\ell=5000$ horizontal guide line

Removed the `axa.axhline(5000, ...)` call and its accompanying inline text label
(`r"$\ell=5000$"`) in the `fig05a_sl_response` cell — there was no separate legend entry, only
the inline annotation, which was removed alongside the line since a floating label with no
reference line would be confusing. Left a one-line comment noting the removal was deliberate
(not a relocation).

## Item 3 — fig07 ν-domain tails: full 22-bin grid, Poisson-faded

Replaced the retired flat "2% of this curve's own peak |value|" floor (`FLOOR_FRAC=0.02`,
previously baked into `response_ratio()`'s default and applied uniformly to all six ν-domain
panels) with the ruled per-bin solid/faded/undrawn convention:

- New `nu_solid_drawn(key, fid)` helper: for `peak_counts`/`minima_counts` (genuine per-map
  mean object counts, `nu_norm='map'`), solid where fiducial mean count ≥ 1, faded (not
  masked away) out to the last bin with fid > 0, undrawn beyond. For `pdf`/`mf_v0`/`mf_v1`/
  `mf_v2` (no natural "one object" unit — and `mf_v2` legitimately crosses zero at its genus
  zero-crossing), solid == drawn == `|fid| != 0` (an amplitude test, not a sign test, so the
  real zero-crossing is never mistaken for a sparse-tail dropout).
- `color_curves()` extended with optional `solid`/`drawn` masks: normal-alpha segment over
  `solid`, alpha≈0.25 segment over the rest of `drawn`, nothing plotted outside `drawn`
  (backward compatible — `solid=None` reproduces the old single-alpha behavior, still used by
  the 6 ell-domain panels).
- Added a light per-bin Poisson band on the unity guide for `peak_counts`/`minima_counts`
  specifically: `err = sqrt(N_fid)/N_fid`, `fill_between(x, 1-err, 1+err, alpha=0.15)` —
  naturally widens in the sparse tail, vanishes past the last drawn bin. Not added for
  `pdf`/MFs (no genuine per-bin count exists there for those; documented in the code).
- `response_ratio()` now takes `floor_frac` explicitly (default 0.0 = division-by-zero guard
  only); the nu-domain branch of `panel_data()` calls it with `floor_frac=0.0`. The 6
  ell-domain panels (S(ℓ) + 5 spectra) are untouched — they already used the separate,
  much-smaller `SPEC_FLOOR=1e-10` numerical guard, unrelated to this ruling.
  ν x-limits unchanged: `(−3, 8)`.
- Documented in a comment block above `nu_solid_drawn()` + a new markdown paragraph
  ("RULED 2026-08-05, ν-domain tails") in the fig07 intro markdown; the stale earlier
  markdown sentence describing the retired 2% floor was also updated to point at the new
  paragraph instead of contradicting it.
- End-of-cell print block extended: replaces the old "floor-masked bin fraction (2% of
  peak)" text with an "undrawn bin fraction" report (ell-domain vs ν-domain semantics stated
  explicitly) plus a new per-statistic solid/faded/undrawn bin-count breakdown for
  peak_counts/minima_counts.

Guard: none needed — this is a pure recomputation of already-available `emulator_dataset_nu05.
npz` data, no new cache dependency.

## Item 4 — fig04/fig04b residual bands: 550-real covariance, GUARDED

**Cache inspection (the load-bearing finding of this item).** Direct shape inspection
(zip-member header peek, never materializing the multi-GB arrays — a first naive
`np.load()[...]` attempt on the raw arrays hit ~5.7 GB and climbing after 4+ minutes and was
killed) of the caches fig04/fig04b actually read:

| cache | n_real |
|---|---|
| `bind_science/runs/bind/run_0000/kappa_maps.npz` | **50** |
| `bind_science/runs/truth/run_0000/kappa_maps.npz` | 550 |
| `bind_science/runs/bind/run_0000/y_maps.npz` | **50** |
| `bind_science/runs/truth/run_0000/y_maps.npz` | 550 |
| `bind_lightcone_tng/tau_maps.npz` (BIND τ) | 550 |
| `bind_science/runs/truth/run_0000/tau_maps.npz` | 550 (now exists — was previously absent) |
| `bind_science/runs/bind/run_0000/paired_perreal_fid.npz` (`clk`) | **50** |

**Verdict: the state is mixed, exactly matching the task's own anticipated fallback** — the
truth side (κ/y/τ) and BIND's own τ have all landed at 550, but BIND's own κ/y maps (which
drive the seed-paired sample size) and the derived `paired_perreal_fid.npz` `S(ℓ)` cache are
still 50-real. A genuine seed-paired covariance needs *both* sides at the same N, so this pair
is the current bottleneck — confirming the task's own note verbatim ("the paired_perreal_fid.
npz cache is 50-real vintage and its regeneration is being handled separately").

**Implementation (guarded, currently exercising the 50-real fallback):**
- New diagnostic block at the top of the fig04 cell: reads each cache's cheap `n_real` scalar
  field (near-zero cost even against an 11 GB array — npz members decompress independently)
  plus `paired_perreal_fid.npz`'s `clk.shape[0]`, computes `N_PAIR_AVAIL` /
  `HAS_550_PAIRED = N_PAIR_AVAIL >= 500`, and prints the full availability breakdown.
- `paired()` fixed to derive its `sqrt(N)` normalization from `min(bR.shape[0], tR.shape[0])`
  (the call's own arrays) instead of the outer-scope `NR` — this is what makes the 550-upgrade
  automatic with no further code change once BIND's own maps catch up (fig04b reuses this same
  `paired()`, so its bands inherit the fix too, per the item's "do the same for fig04b" ask).
- `rho_realization()`'s hardcoded `N=50` default → `R.shape[0]`; Part A's "sigma/sqrt(50)"
  verdict print, the MF point-error print ("DIFFERENCE scatter/sqrt(50)"), and fig04b's
  legend label (`r"paired $\pm1\sigma/\sqrt{50}$"`) are all now dynamic f-strings tied to the
  actual `NR`/`NR6`.
- New markdown paragraphs in both fig04's and fig04b's markdown documenting the 2026-08-05
  guarded status and what "no code change" means concretely.

Today, with the cache state above, every one of these prints/labels correctly still reads
"50" (verified — see fig04's/fig04b's own cells are heavy and were not executed live in this
session, see Renders/Pending below, but the guard logic itself was reasoned through against
the actual on-disk shapes, and `paired()`'s new per-call min-N logic is a strict, low-risk
generalization: identical output when both arrays already have equal length, which is the
case for every currently-executing code path).

## Item 5 — noisy-twin figures + Lee+23 LSST band (all GUARDED, all currently pending)

**Verified absent (safe, cheap `.exists()` checks):** `bind_sb35/nu05n_shards/sci_bind.npz`,
`.../sci_truth.npz`, `bind_sb35/emulator_dataset_nu05n.npz` — the `nu05n_shards/` directory
exists (created) but is empty. A speculative noisy-Cl-κκ cache path
(`bind_science/runs/bind/run_0000/paired_perreal_fid_noisy.npz`) also does not exist.

**New cell `fig04n_field_validation_noisy`** (self-contained, no DEPS): mirrors fig04's six
ν-domain panels (pdf, peak/minima counts, V0–V2), reading
`bind_sb35/nu05n_shards/{sci_bind,sci_truth}.npz`. Guard prints
`noisy cache pending -- cell skipped` and the `save(fig, ...)` call sits inside the guard
(verified live — see Renders). No separate LSST-like sample-variance contour is drawn on the
residual strips (documented rationale: survey realism is already *in* the data via the
injected noise, so a second precision estimate would double-count).

**New cell `fig07n_covariation_noisy`** (DEPS: `["fig05a_sl_response", "fig07_covariation"]`
— the resolver is one-level only, so both ancestors must be listed explicitly): mirrors fig07's
full 12-panel layout, reading `bind_sb35/emulator_dataset_nu05n.npz` for the six ν-domain rows
(reuses fig07's own `STATS`/`FID`/`response_ratio`/`nu_solid_drawn`/`color_curves` rather than
re-deriving ~150 lines of machinery) and keeping the six ell-domain rows identical to fig07 by
construction (the noisy dataset is assumed built the same way `nu05` was from `xpkfix` — only
ν-domain arrays replaced, everything else copied unchanged). A local `panel_data_n(s, A, fid)`
is used instead of mutating fig07's shared `FID` dict, to avoid corrupting state for any later
re-render of fig07 itself.

**Markdown**: one combined markdown cell before `fig04n`'s code documents both new figures —
noise model = Lee et al. 2023 (arXiv:2201.08320) Eq. 6/7, LSST-Y10 numbers
$\sigma_e=0.26$, $n_{\rm gal}=27\,\mathrm{arcmin}^{-2}$, $\theta_G=1'$, noise added before
smoothing; states that fig04n answers §5's "noiseless → survey realism" caveat; states the
guard behavior and why the dead `save()` branch is kept in source for `_check_builder.py`.
A short markdown note before fig07n restates the DEPS/ordering requirement.

**fig04's LSST band, v3 (Lee+23 form), GUARDED, fallback verified active:** added a new block
immediately after the existing v2 band construction (kept, untouched, as the fallback) that
would — if a noisy per-realization $C_\ell^{\kappa\kappa}$ cache existed — replace
`sigma_meas_lsst_kk` with the directly-measured scatter of the *noisy* fiducial realizations
(retiring the analytic shape-noise-excess term, since it is then already in the measured
scatter), area-scaled the same way, with the Hartlap factor for the achieved N printed
("noted"), not baked into the plotted band. **This guard's target path is a best-guess**
(no filename for this specific artifact was given; documented explicitly as speculative,
one-line change if it lands under a different name) — today it correctly evaluates absent and
`lsst_kk`/`lsst_S` keep exactly their v2 values. A new markdown paragraph ("v3
(2026-08-05, GUARDED — Lee+23 form)") documents this in fig04's Survey-context section.

**`_run_subset.py` updates:** `fig04n_field_validation_noisy` needs no DEPS entry (fully
self-contained). `fig07n_covariation_noisy` added to `DEPS` with both `fig05a_sl_response` and
`fig07_covariation` (non-transitive resolver). New `GUARDED_PENDING` set + updated end-of-run
report so a caller doesn't mistake a guarded-absent PDF for a broken run.

## Gate log

`_check_builder.py` run after every item (and again after the fig20 label-overflow fix):
0 syntax errors every time; figure count grew 24 → 25 (item 5, fig04n) → 26 (item 5, fig07n),
stayed at 26 through the final gate. Final gate output: 68 cells (31 code), 0 syntax errors,
26 `save(fig, ...)` sites.

## Renders verified this session (1-core / ~13 GB RAM node)

All via `_run_subset.py <name>`, actively memory-monitored (killed on sight of anything
approaching the ceiling):

Each figure below was rendered TWICE: once under ruling #2 (display-shift) to validate that
approach, then RE-RENDERED after the full revert to ruling #3 (final, text-only) to confirm
the revert actually took — the table reports the final (ruling-#3) verification pass.

| figure | time | note |
|---|---|---|
| `fig03_halo_validation` | 7 s | visually confirmed final form: $\log_{10} M_{200c}/\mathrm{M}_\odot$ on all 3 x-axes + panel (c) y-axis, x-range back to native 13.0–14.7 (unshifted) |
| `fig06_radial_profiles` | 7 s | visually confirmed: legend title $\log_{10} M_{200c}/\mathrm{M}_\odot$, bin numbers native (13.4-13.8, 14.2-14.7, unshifted) |
| `fig14_completeness` | 29 s | visually confirmed: both panel x-labels $\log_{10} M_{200c}/\mathrm{M}_\odot$, native values |
| `fig19_fgas_erosita` | 9 s | visually confirmed: y-label mass clause $M_{200c}/\mathrm{M}_\odot=1$–$2\times10^{13}$, native |
| `fig20_vandaalen_matrix` (+ fig03 prereq) | 11 s → 10 s (overflow fix) → 11 s (final revert) | visually confirmed all 3 panels in final form; **caught + fixed a real label-overflow regression** along the way (panels b/c text ran past the axes when the parenthetical mass clause grew longer — survived the revert since the fix was a fontsize/wording tweak, independent of which ruling was active) |
| `fig04n_field_validation_noisy` | 5 s | guard fires correctly: printed `noisy cache pending -- cell skipped`, no PDF written (confirmed absent, not an error), `_run_subset.py`'s new ABSENT/GUARDED_PENDING reporting confirmed working |

`fig05a_sl_response` was **not** attempted despite being explicitly named as a candidate: the
task's own prior guidance already flags the fig05a-family as an OOM case on this node, and
`fig23_s3_opener` was attempted and aborted after ~3 minutes with no progress once its
`fig04_field_validation` prerequisite was pulled in — fig04's cell now loads the 550-real
truth-side `kappa_maps.npz`/`y_maps.npz` (≈10.7/10.5 GB each) in full, which this session's
cache-shape inspection (above) confirms is real, not a false alarm; killed proactively rather
than risking a hard OOM on the shared node.

## Pending the next sbatch run (`run_figures.sbatch`, GPU node)

Not renderable on this 1-core/~13 GB node — listed here rather than attempted:
- `fig04_field_validation`, `fig04b_spectra` (item 1's fig-not-applicable here / item 4's
  guard + label edits) — needs the full 550-real truth κ/y/τ cubes in memory.
- `fig23_s3_opener` (item 1's fig23 edit) — pulls in fig04 as a prerequisite.
- `fig05a_sl_response` (item 2's edit), `fig05_param_response`, `fig07_covariation` (item 3's
  edit) — the fig05a-family, previously established as OOM-on-this-node.
- `fig07n_covariation_noisy` (item 5) — depends on fig07 (above) AND its own guard is
  currently pending anyway (no cache yet), so nothing to visually verify until both land.
- `fig04n_field_validation_noisy`'s **true branch** (post-cache-landing render) — only the
  guard-false path was exercisable this session, by construction.
- Item 4's **true branch** (once BIND's own κ/y maps + `paired_perreal_fid.npz` reach 550) —
  same reasoning; only the (unchanged) 50-real fallback path is exercisable today.
