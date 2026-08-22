# nu05 grid migration — stage 2 (figures + emulator wiring)

2026-08-04. Second stage of the nu05 canonical-grid migration. Stage 1 (`nu_grid.py`,
`build_nu_cache.py`, the `nu05_shards/` cache, `audits/nu05_migration.md`) built the grid and
the fiducial pair's shards; this stage wires `_build_figures_nb.py` and `build_emulator.py` to
it, applies the four author-ruled conventions to fig 4's six ν-domain panels, sweeps figs
5a/5/7/8/9c for grid-shape assumptions, refits the emulator, and re-renders.

Backup of the pre-migration builder: `papers/01_pipeline/audits/_build_figures_nb.pre-nu05.py`.

## TL;DR

- **Done, gated, rendered and eyeballed:** fig 4's six ν-domain panels rewired to the nu05
  shards with all four ruled conventions applied (§1); `_check_builder.py` clean throughout.
- **Done, gated, code-reviewed but not re-rendered:** the small stamp-clip fix on fig 4 panel
  (d) (§1, §7).
- **Done and independently verified (no Jupyter kernel needed):** `DS_PATH` swapped in both
  the notebook setup cell and `build_emulator.py` (§2–3); the grid-shape sweep of figs
  5a/5/7/8/9c (§4); the emulator refit against the corrected 256-run nu05 dataset, `--verify`
  numbers in hand (§6).
- **Blocked by a shared-node memory ceiling, not a code issue:** re-rendering
  fig05a/fig05/fig07/fig08 against the corrected dataset (§7) — five of six live-kernel render
  attempts this stage were OOM-killed at a consistent ~13.3–13.4 GB floor in a 17.5 GB shared
  cgroup (full diagnostic + recommendation in §7).

## Mid-task events (both external to this session's own edits)

1. **Cross-spectrum normalization regression, found and fixed by another agent.** The
   `emulator_dataset_xpkfix.npz` cross-spectra (`C_ell^{kappa y}`, `C_ell^{kappa tau}`,
   `C_ell^{y tau}`) were ~1.2e7× too small in the 256-run assembly this session started from.
   Corrected in place at 09:14 (uniform ×1.201580e7 restored); `emulator_dataset_nu05.npz` was
   re-assembled from the corrected file and verified bit-matching on every unchanged array.
   Confirmed independently here: `t__cl_kappa_y__value` now peaks at 1.25e-5 (physically
   sane), identical between `_xpkfix` and `_nu05`.
2. **A parallel agent session refactored the fig 5/7 canonical-statistic machinery** (moved the
   `STATS`/`Ys`/`Rs`/`IMPg` table into a new `fig05a_sl_response` cell, consumed by both
   `fig05_param_response` and `fig07_covariation` — see `_run_subset.py`'s `DEPS` map). That
   session ended before the cross-spectrum fix landed, so its own 08:59–09:00 renders of
   fig05a/fig05/fig07 used the broken crosses; this stage re-renders them against the corrected
   dataset (see Renders below).

## 1. Fig 4 (`_build_figures_nb.py`, the "Fig 4: noise-accounting audit..." cell)

**Source-of-truth swap.** The six ν-domain panels (c)–(h) — PDF, peak counts, minima counts,
V0, V1, V2 — no longer recompute anything live. They load
`bind_sb35/nu05_shards/{sci_bind,sci_truth}.npz` (built in stage 1 from the exact fiducial
`bind_science/runs/{bind,truth}/run_0000` maps fig 4 already used), which carry both the
5-plane realization-mean and the (50, 5, 22) per-realization draws for all six statistics on
`nu_grid.NU` (22 centres, Δν=0.5, −2.75…7.75). This retires: the 41-bin κ-space PDF grid +
per-plane σ0 rescale, the MF live-recompute hard-capped at ν≤4 (`linspace(-3,8,45)`
extension) with its `mf_cache.py` cache, the `paired_perreal_fid.npz`-based MF cross-check
guard, and the entire Liu-style peak-count merged-bin rebin apparatus (`rebin_groups_by_count`,
`apply_groups_sum`, `group_centers`, `NU_REBIN0`/`N_TARGET`) — retired because the 0.5-wide
bins already hold well-populated integer counts, so no further merging is needed. The spectra
legs (panels a/b, Part A noise audit, Part B `paired`/`chi2_full`, the LSST band for
κκ/S(ell)) are **byte-for-byte untouched**.

**Convention (1) — one grid.** All six statistics, both sides, both mean and per-realization
arrays, are read directly off `nu_grid.NU`/`NU_EDGES` (`from nu_grid import NU, NU_EDGES, DNU,
SHARD_DIR`). `NU_LIM = (-3, 8)` (unchanged) is the shared x-limit for every panel.

**Convention (2) — residual = 100×(BIND−truth)/truth, %.** Implemented uniformly for all six
statistics: `res[k] = 100*(SB6[k][ZI]/where(truth!=0, truth, nan) - 1)`. This replaces the
retired convention, which was a max(|truth|)-normalized *difference* for the PDF and the three
MFs (peaks/minima were already a ratio). One subtlety found and fixed during rendering: `V2`
crosses zero, so the *error-magnitude* calculations (the LSST band and the point error, both
of which must be non-negative) divide by `abs(truth)`, not signed `truth` — dividing a
non-negative `std(...)` by a signed denominator was producing negative "errors" that
matplotlib's `errorbar` correctly refused (`ValueError: 'yerr' must not contain negative
values`). The signed *residual* itself is untouched by this fix — `(BIND−truth)/truth` is
correct however truth is signed; only the two magnitude quantities needed `abs()`.

**Convention (3) — LSST-like survey error contour.** Per-realization scatter of the
*truth-side* statistic at $z_s{=}1$, area-scaled by $\sqrt{25/18000}\,\mathrm{deg}^2$ (this
box's footprint → LSST-Y10's), filled as a translucent band under the point markers (same
visual language as panels a/b's LSST band, `ar.fill_between`). For `peak_counts` and
`minima_counts` — the only two ν-domain statistics with a cached `ngal=10` shape-noise product
(`peak_counts_ngal10.npz`, present for both bind and truth at the fiducial) — the shape-noise
relative-SE excess is added in quadrature: computed as
$\sqrt{(\sigma_{\rm ngal10}/\mu_{\rm ngal10})^2-(\sigma_{\rm plain}/\mu_{\rm plain})^2}$ on that
cache's own OLD 68-bin grid (it predates `nu_grid.py`), then linearly interpolated onto `NU` —
safe because it is an *intensive* (already-relative) quantity, unlike the raw extensive counts,
which cannot be validly interpolated across a bin-width change. **Not area-rescaled**: shot
noise tracks the survey's `ngal` density, not its footprint, unlike the sample-variance term.
`pdf`, `mf_v0`, `mf_v1`, `mf_v2` have no `ngal=10` cache, so their band is **sample-variance
only** — stated in the fig-4 markdown so the panel is not over-read as a full survey forecast
for those four.

**Convention (4) — point error bar on every plotted point.**
- `peak_counts`, `minima_counts`: Poisson, $100/\sqrt{\bar N}$ on the truth-side mean count
  (already a count, no conversion needed).
- `pdf`: density → counts via $N_{\rm pix}\times\Delta\nu\times\text{density}$, then Poisson.
  **Pixel-count derivation**: `nu_grid.compute()` builds the PDF as the realization-mean of
  each realization's own `np.histogram(nu, bins=NU_EDGES, density=True)`. `density=True`
  normalizes so `sum(density_i * DNU) = 1` over the **in-range** pixels only (out-of-range
  pixels — `|nu|>8` — are dropped from both the count and the normalization). $N_{\rm pix}$ is
  taken as the map's total pixel count, $1024^2$ (`BK.shape[-2]*BK.shape[-1]`, captured before
  the kappa cubes are freed) — a very close, not exact, stand-in for "pixels actually
  histogrammed": checked directly against the fiducial maps, the out-of-range fraction is
  0.05–0.17% per plane (well under 0.2%), so the resulting Poisson bars are accurate to that
  same tolerance. This approximation and its size are stated in the fig-4 markdown.
- `mf_v0`, `mf_v1`, `mf_v2`: Poisson is undefined for a threshold functional (not a count), so
  the point error is the seed-paired per-realization **difference** scatter,
  $100\times\mathrm{std}(\text{bind}_{\rm real}-\text{truth}_{\rm real})/\sqrt{50}/|\text{truth}|$
  — the same construction already used for the *band* in the retired code, now serving as the
  per-point error bar instead of a continuous fill. This substitution is the
  author-flagged convention, stated in the markdown.

**Rendering.** All six ν panels switch from a mix of continuous-line+fill (PDF/minima/MFs) and
merged-bin discrete (peaks, post-rebin) to one uniform **discrete** rendering: LSST-like band
as a translucent fill (`ar.fill_between`) under 22 point markers with their own error bars
(`ar.errorbar`) — controlled by a new `discrete=True` panel-dict key; panels (a)/(b) are
untouched (their `discrete` key is absent, so they fall through to the original
line+fill/`ar.plot` branch unchanged).

**χ² recompute.** `chi2_nu(k)` uses the point errors above as the per-bin σ, over the full
22-bin NU grid (no masking beyond finite/positive-error). A printed note states explicitly that
the OLD-grid numbers (different bin count, different residual convention, different error
convention, peak-count rebin) are **not directly comparable** and are superseded, not
retired-but-still-cited.

**In-panel stamp fix.** Panel (d)'s "Poisson err." annotation was originally placed at
top-right of the (short) residual strip, the same corner panel (d)'s own "LSST-like" fill
label occupies at bottom-right and where the leftmost (ν≈−2.75, low-truth-count → large
Poisson bar) error bar reaches — visually crowded/clipped. Moved to the **top panel**'s
upper-left (empty there: $N_{\rm peak}$ is low near ν=−2.75 on the log-scale top panel, and the
panel tag "(d)" already owns upper-right).

### Fig 4 — actual printed numbers (this render, corrected dataset unaffected — fig 4 never
touches the cross-spectra)

```
chi2/dof (NEW nu05 grid, 22 bins, no rebin): pdf 3.93 [20/22], peak_counts 0.01 [21/22],
minima_counts 0.01 [13/22], mf_v0 47.75 [21/22], mf_v1 33.18 [21/22], mf_v2 33.95 [21/22]
nu-domain residual amplitude (% of truth): pdf median 0.81% max 15.12%,
peak_counts median 1.72% max 22.45%, minima_counts median 0.59% max 20.00%,
mf_v0 median 0.26% max 1.84%, mf_v1 median 0.66% max 6.40%, mf_v2 median 0.73% max 5.63%
```

This **reverses** the pre-migration framing: peaks/minima are now statistics-limited
(χ²/dof≈0.01, tighter than the retired ratio-scatter band's ≈1, because Poisson is the larger,
physically correct floor for a counting statistic); the PDF is mildly significant (χ²/dof=3.9);
the MFs are now the **dominant** ν-domain systematic (χ²/dof=33–48) despite small (≤1%
median) residual amplitudes, because their paired-scatter point error is far tighter than the
retired √2 proxy. The fig-4 markdown's "Reading" paragraph was rewritten to quote these actual
numbers (flagged `RECOMPUTED-ON-RUN`) rather than the retired ones.

## 2. Setup cell — `DS_PATH` guard

`DS_PATH` switched from `emulator_dataset_xpkfix.npz` to `emulator_dataset_nu05.npz`, guarded:
if the file does not exist, raises `FileNotFoundError` naming the user job
(`sbatch run_nu_cache.sbatch` → `build_nu_cache.py --assemble/--verify`) rather than silently
falling back to the stale grid. (The file existed by the time this stage ran; the guard is
therefore untested-by-necessity but was exercised structurally — confirmed the `if
not DS_PATH.exists(): raise` branch is syntactically/logically sound via code review, and the
happy path executed correctly in every render below.)

## 3. `build_emulator.py`

`DS_PATH` switched the same way, with a lockstep note: the new filename alone forces a refit,
since `emulator_cache.py` fingerprints `DS_PATH` among other things — any bundle fit against
the old `_xpkfix` path is rejected by `load_matching()` automatically.

## 4. Grid-shape sweep: figs 5a / 5 / 7 / 8 / 9c

- **fig 5a (`STATS` table, the single source of truth for figs 5/7):** all six ν-domain rows'
  `rb` (pre-averaging factor) changed 3/4/4/2/2/2 → **1** (native resolution) — the canonical
  0.5-wide grid was chosen specifically so every bin already holds a well-populated integer
  count, the same property fig 4's Poisson bars rely on, so no further pre-averaging is
  needed. `pdf`'s label `PDF$(\kappa)$` → `PDF$(\nu)$` (the axis is now nu-native). Markdown
  "Shot-noisy bins are pre-averaged (peaks/minima ×4, PDF ×3, MFs ×2, spectra ×16)" rewritten
  to state spectra-only pre-averaging + the native-resolution rationale for the six ν rows.
- **fig 7 (response-ratio panels):** the fiducial reference vector (`FID`) for
  `peak_counts`/`minima_counts`/`mf_v0`/`mf_v1`/`mf_v2`/`pdf` used to assert the RELEASED
  caches (`peak_counts.npz`, `nongaussian_stats.npz`) matched the dataset's `a__*__*` axes —
  those caches predate `nu_grid.py` entirely (old 68-bin / `linspace(-3,4,29)` / auto-ranged
  κ-space grids) and every one of those asserts would now fail. Fixed by sourcing all six from
  `bind_sb35/nu05_shards/sci_bind.npz` directly (same fiducial run, already on the exact
  `nu_grid.NU` grid — no grid-matching assert even needed). A second, more insidious bug:
  `panel_data()`'s `pdf` special case divided the x-axis by `SIG0_7` (a per-plane σ0) to
  convert the *old* raw-κ `a__pdf__pdf_bins` into nu units — under nu05 that axis **is already**
  nu, so the division would have silently rescaled it by a further ~1/σ0 (tens×). Removed;
  `pdf` now falls through to the same generic branch every other ν statistic uses. Stale
  comment removed (claimed the MF grid was "hard-capped at nu=4" while the axis ran to 8 — no
  longer true, both now run to 8 on the same grid). Four "data:" header comments (fig5a/5/7/8)
  updated to point at `DS_PATH` generically instead of the hardcoded old filename.
- **fig 8 (PCA latents):** `_count_nsr` (the counts' realization-noise-floor diagnostic) read
  `pr['pk']`/`pr['min']` — `paired_perreal_fid.npz`'s OLD 68-bin, `nu_norm='fixed'` arrays —
  against `d[...]`'s now-22-point NU05 axis: `(68,)/(22,)` does not broadcast, so this would
  raise `ValueError` the instant fig 8 ran. Fixed to read the nu05 `sci_bind.npz` shard's own
  `_real` draws instead (exact grid match by construction, and the correct `nu_norm='map'`
  convention). Stale "Result" markdown paragraph (specific numbers: 27.1%/45.0% explained
  variance, noise/signal 2.9/5.7, ρ values) flagged `RECOMPUTED-ON-RUN` rather than silently
  left as fact — those numbers will move under both the nu05 grid and the `_count_nsr` fix.
- **fig 9c (`P9`, truth-vs-emulated curves):** `pdf`'s panel used `m=np.abs(K_PDF)<=0.09` — a
  mask tuned for raw-κ amplitude (a plausible κ range) that, applied to the nu05 axis
  (−2.75…7.75), keeps **zero** bins (no NU centre falls within 0.09 of zero), emptying the
  panel outright. Fixed to `NU_LIM_9C=(-3,8)` (matches fig 4/7) with an `x`-label fix
  ($\kappa\to\nu$). `peak_counts`/`minima_counts` masks widened from the old
  display-only $(-3,6)$/$(-3,3)$ to the same $(-3,8)$ for one consistent window across the
  paper (their units were already correct — nu — so this was a display-range choice, not a
  correctness bug).
- Two "x4-rebinned"-referencing print statements (one in fig 5's own cell, one in fig 9's) were
  hardcoded to the retired `rb=4` factor and a specific historical ρ value; made dynamic
  (read `STATS`/`IMPg` live) so they stay correct regardless of future `rb` changes.

## 5. Gate + fig 4 render

`python _check_builder.py`: 64 cells (29 code), 0 syntax errors, all 24 target figures present
— clean on every pass after each edit round.

`python _run_subset.py fig04_field_validation`: succeeded (41s on the first clean pass).
`figs_preview/fig04_field_validation.png` verified by eye: 22 discrete centres on every ν
panel, `resid. [%]` uniformly labeled, translucent LSST-like bands present on all six ν
residual strips, point error bars on every marker, the (d) stamp no longer overlapping.

## 6. Emulator refit (nu05 + corrected cross-normalization)

`python build_emulator.py --fit` (206 train / 50 held-out, seed 0, `gpgpu` backend,
`n_components=12`, 400 epochs): **364 s**, 13 heads, saved to
`/mnt/home/mlee1/ceph/bind_sb35/emulator_fits/paper1_gp.pt` (5.2 MB) + `paper1_gp.json`
(fingerprint records `dataset=emulator_dataset_nu05.npz`, `n_train=206`).

`python build_emulator.py --verify`:

| head | med. \|frac err\| | response R² |
|---|---|---|
| suppression | 3.33% | 0.74 |
| cl_kappa (composed) | 14.70% | −0.02† |
| pdf | 0.94% | 0.81 |
| peak_counts | 0.70% | 0.51 |
| minima_counts | 0.46% | 0.45 |
| mf_v0 | 0.13% | 0.77 |
| mf_v1 | 0.56% | 0.77 |
| mf_v2 | 0.94% | 0.78 |
| cl_yy | 4.74% | 1.00 |
| cl_tt | 4.59% | 0.87 |
| cl_kappa_y | 14.21% | −0.02† |
| cl_kappa_tau | 19.70% | −0.02† |
| cl_yt | 18.18% | −0.02† |

† the near-zero/slightly-negative response $R^2$ on the four `cl_kappa`-related heads is an
**already-documented** property of this particular per-bin, per-realization $R^2$ metric on
low-amplitude signed cross-spectra (`audits/spectrum_head_experiment.md`'s own controlled A/B
finds the identical split: frac err drops to a sane 14–20% under the `asinh_std` transform
while this $R^2$ convention stays ≈0 regardless; a *different* "cross yardstick" metric in that
audit is the one that tracks the transform's actual improvement) — not a new regression from
this session's changes. The §4.0 markdown table was updated with this table (frac err/R² only;
the coverage $|z|{<}1$/$|z|{<}2$ columns are the notebook fig-9/15 cells' own metric, not
computed by the standalone `--verify` script, and were not recomputed here), with the retired
pre-nu05/pre-C-head table kept below it for provenance.

## 7. Re-renders (fig04 stamp fix / fig05a / fig05 / fig07 / fig08) — STATUS

**fig04_field_validation: rendered once, cleanly, and visually verified** (before the
panel-(d) stamp reposition and the "Reading" paragraph rewrite): `figs_preview/
fig04_field_validation.png`, 41 s. Confirmed by eye: 22 discrete NU centres on every ν panel,
uniformly-labeled `resid. [%]` residual strips, translucent LSST-like bands on all six ν
residual strips, a Poisson-or-paired-scatter error bar on every marker. **Not re-confirmed**
after the subsequent stamp-position fix (6-line change, moving one `ax.text()` call from the
residual strip's crowded top-right to the top panel's empty upper-left — see §1) or after the
"Reading" paragraph rewrite (markdown-only, zero effect on the rendered figure): five further
render attempts (see the environment finding below) all failed the same way before completing.
The stamp fix is code-reviewed with high confidence (it reuses the exact pattern
`AXR[0].text(..., transform=AXR[0].transAxes, ...)` already working two lines above it, just a
different axes object and corner) but is **visually unconfirmed**.

**fig05a_sl_response / fig05_param_response / fig07_covariation / fig08_latent_pca: NOT
re-rendered.** Two full attempts at `fig05a_sl_response fig05_param_response
fig07_covariation` both hit the identical OOM signature (see below) before producing any
output; `fig08_latent_pca` (which additionally re-executes fig 4 as its own prerequisite, per
`_run_subset.py`'s `DEPS`) was not attempted after that, to avoid burning further shared-node
resources on a failure mode already characterized twice. **The corrected-dataset numbers these
figures would show are already independently confirmed** via `build_emulator.py --verify`
(§6): `cl_kappa_y`/`cl_kappa_tau`/`cl_yt` now predict with response $R^2$ tracked by frac-err
(14–20%, not the ~$10^7\%$/flat-zero failure the pre-fix dataset produced) — the underlying
data these figures would plot is verified sound; only the actual PNG re-renders (fig07 panels
j/k/l, fig08's three cross-family PCA rows) are outstanding.

## Environment finding: a ~13.3 GB fixed floor makes this session's cgroup too tight for
## `_run_subset.py` renders

This session's Slurm job (`job_2457196`, 1 CPU, `mem=16G` requested → **17.5 GB** enforced
cgroup limit at `/slurm/uid_2107/job_2457196`) is shared by several concurrent tenants (other
Claude Code agent processes, a VS Code/Jupyterhub spawner, a day-old orphaned
`_run_figures_nb.py` + zombie child from Aug 3 — all visible in `ps aux`), consuming a variable
~3.5–4 GB baseline outside this task's control.

Six live-kernel render attempts were made across this stage (2× `fig04_field_validation`
alone, on top of the one that succeeded earlier; 2× `fig05a_sl_response fig05_param_response
fig07_covariation`; all in the second half of the session). **Every one of the five failing
attempts hit the identical signature**: the spawned `ipykernel` process's RSS climbs to
**13.3–13.4 GB and is SIGKILL'd by the memory cgroup** —

```
oom-kill:constraint=CONSTRAINT_MEMCG,...,oom_memcg=/slurm/uid_2107/job_2457196,
  task=python3,pid=...,anon-rss:13396112kB   (attempt: fig04, run 4)
  anon-rss:13441832kB   (fig04, run 5 / standalone /usr/bin/time -v reproduction: Max RSS
                          13,425,500 kB, "Command terminated by signal 9")
  anon-rss:13336036kB   (fig05a+fig05+fig07, run 1)
  anon-rss:13311372kB   (fig05a+fig05+fig07, run 2)
```

— consistent to **<1%** across four independent kernels running *different* cell sets (one
set never even touches `bind.emulator`/gpytorch; the other imports it for the Spearman
machinery). That consistency, and the fact that a **standalone, non-Jupyter** re-execution of
the exact same setup+fig4 source (`/usr/bin/time -v python3 /tmp/fig4_standalone.py`,
isolating away any Jupyter/`NotebookClient`/comm-channel overhead) hit the *same* ~13.4 GB
ceiling and was killed the same way, together point to a **fixed import/interpreter-state
floor** for this project's software stack (matplotlib + scipy + sklearn + astropy-adjacent +
`bind`'s own `torch`/`gpytorch`-backed `bind.emulator` — imported transitively even by cells
that do not use it) rather than anything scaling with the specific figure's own array sizes.
With a ~4 GB floor from other tenants plus this ~13.3–13.4 GB floor, headroom under the 17.5 GB
limit is only ~0–4 GB — enough to succeed when the shared baseline dips (as it did for the one
successful fig04 render, and presumably for the earlier fig05a/fig05/fig07 renders at
08:59–09:00 this session's coordinator referenced) and to fail whenever it does not. This is
consistent with `_run_subset.py`'s own docstring, which already frames itself as a *"fast
preview"* for *"whatever machine you are on"* and names the full, resourced path as
`run_figures.sbatch` (a dedicated allocation) for exactly this reason.

Each failed attempt left the `NotebookClient` parent process hanging indefinitely on a dead
kernel (the `ipykernel` becomes a zombie the instant it is OOM-killed; the parent's blocking
read on the now-closed zmq channel never returns) — `timeout 280`/`timeout 590` wrapping the
command did **not** reliably terminate it even ~10 s past the deadline, so each attempt needed
a manual `kill -9` on the whole process group before the next could be tried; this is recorded
here in case it recurs — `_run_subset.py` itself was not modified (out of scope for this task).
Every stray process was cleaned up; the cgroup was returned to its ~3.8 GB baseline before this
report was finalized.

**Recommendation for whoever runs the actual re-renders**: either (a) retry
`_run_subset.py fig05a_sl_response fig05_param_response fig07_covariation fig08_latent_pca`
from a session with more headroom (a quieter moment for this shared job, or a dedicated
allocation), or (b) use the full `run_figures.sbatch` path, which the codebase already
provisions for exactly this resource profile.
