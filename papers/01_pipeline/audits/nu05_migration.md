# nu05 migration: canonical Δν=0.5 grid for the six threshold statistics

2026-08-04. Supersedes the earlier `nu8` pass (`linspace(-3, 8, 45)`, 45 centres, dnu=0.25).
The author ruled the canonical convention to be **0.5-wide bins from −3 to 8**: 23 edges,
22 bin centres (−2.75 … 7.75), for all six statistics — `pdf`, `peak_counts`,
`minima_counts`, `mf_v0`, `mf_v1`, `mf_v2` — everywhere in the paper.

Motivation (author): with 0.5-wide bins the counting statistics (peaks, minima, PDF) hold an
**integer number of realization-summed counts per bin**, so each point on those panels
carries a well-defined Poisson error. The narrower 0.25 grid didn't have that property.

Nothing released or from the earlier `nu8` pass was touched: `nu8_shards/` and
`emulator_dataset_nu8.npz` are left exactly as they were, alongside the new `nu05_shards/`
and `emulator_dataset_nu05.npz`. `emulator_dataset_xpkfix.npz` (the ultimate source of
non-ν arrays) is untouched by any of this.

## The edges-vs-centres decision (this is the part that needed care)

The old `nu8` grid conflated two different objects — `NU` (45 "centres") doubled as both
the histogram abscissa (via a derived `NU_EDGES`) *and* the direct MF evaluation points.
The new grid keeps that same *architecture* (one abscissa, `NU`, shared by both families)
but is explicit that the two families consume it differently:

- **`pdf`, `peak_counts`, `minima_counts` are histograms.** A pixel (or local peak/minimum)
  with S/N `nu` is counted into bin `i` iff `NU_EDGES[i] <= nu < NU_EDGES[i+1]` (last bin
  closed at the top edge, i.e. `numpy.histogram`'s convention). These three are always
  computed with the 23-edge array `NU_EDGES`, which bins to 22 counts; the reported
  abscissa is the bin **centre**, `NU[i]`.
- **`mf_v0`, `mf_v1`, `mf_v2` (Minkowski functionals) are threshold functionals, not
  histograms.** `V_k(ν)` is one map-wide number evaluated *at* a threshold ν — nothing is
  binned. These are computed by passing `NU` (the 22 centres) directly as the evaluation
  points to `bind.inference.stats.nongaussian_stats(..., mf_thresholds=NU)`. `NU_EDGES`
  plays no role for the MFs.

Both families end up reporting against the same 22-point `NU` abscissa, which is the whole
point of the exercise (comparable panels), but only the histogram trio actually bins
anything. This distinction pre-existed in the `nu8` code (I verified it by reading
`bind.inference.stats.peak_counts`/`nongaussian_stats`) — `peak_counts(nu_bins=...)` derives
histogram bin centres from edges via `0.5*(nu_bins[1:]+nu_bins[:-1])`, matching how `NU` is
now *defined* as `NU_EDGES` bin centres; `nongaussian_stats(mf_thresholds=...)` evaluates
`minkowski_functionals(nu, mf_thresholds)` at each threshold directly, with no binning. So
the underlying `compute()` logic in `nu_grid.py` needed **no structural changes** — only the
grid constants (`NU_EDGES`, `NU`, `DNU`) and path constants changed; every downstream
consumer of `len(NU)` / `NU_EDGES` in `compute()` picks up 22/23 automatically.

## Files changed

### `papers/01_pipeline/nu_grid.py`
- Module docstring rewritten: states the new canonical grid, the Poisson-error motivation
  for 0.5-wide bins, and a new "EDGES vs CENTRES" section spelling out the histogram-vs-MF
  distinction above. `WHAT IT DOES NOT DO` updated to name both the new and superseded
  shard dirs/dataset files.
- Grid definition changed from
  ```python
  NU = np.linspace(-3.0, 8.0, 45)
  DNU = float(NU[1] - NU[0])                                   # 0.25
  NU_EDGES = np.concatenate([NU - DNU/2, [NU[-1] + DNU/2]])    # 46 edges
  ```
  to
  ```python
  NU_EDGES = np.arange(-3.0, 8.0 + 0.5/2, 0.5)                 # 23 edges, width 0.5
  NU = 0.5 * (NU_EDGES[:-1] + NU_EDGES[1:])                    # 22 centres, -2.75..7.75
  DNU = float(NU[1] - NU[0])                                   # 0.5
  ```
  i.e. edges are now primary and centres are derived from them (previously the reverse),
  which is the more defensible direction given edges are what `numpy.histogram` and
  `peak_counts(nu_bins=...)` actually consume.
- `SHARD_DIR`: `nu8_shards` → `nu05_shards`.
- `DATASET_OUT`: `emulator_dataset_nu8.npz` → `emulator_dataset_nu05.npz`.
- `DATASET_IN` (`emulator_dataset_xpkfix.npz`) unchanged.
- `compute()` body: **unchanged** (verified sufficient — see above). It already consumed
  `NU`/`NU_EDGES`/`len(NU)` generically.

### `papers/01_pipeline/build_nu_cache.py`
- Docstring updated: new grid description, `--assemble` target renamed, note that this
  supersedes (without touching) the `nu8` pass.
- `assemble()`: added `assert len(run_ids) == 256, ...` right after loading
  `DATASET_IN` — the released dataset now carries all 256 Sobol runs (confirmed live:
  `emulator_dataset_xpkfix.npz["run_ids"]` has 256 entries, 0..255), vs the 253 the `nu8`
  sbatch comment referenced. Everything else in `assemble()`/`verify()` is structurally
  unchanged — filenames come from `nu_grid.py` constants, so they resolve to the new
  `nu05_shards/`/`emulator_dataset_nu05.npz` automatically.
- `nu_grid_note` string (written into the assembled dataset) rewritten to describe the new
  grid and explicitly name the histogram-vs-MF-threshold split.
- **NaN-aware `verify()` fix**: present already (comment "bit us 2026-08-03: wst/peak_R
  carry all-NaN rows for uncovered runs", `equal_nan=(a[k].dtype.kind == "f")`) — confirmed
  and left as-is, no changes needed. `--run`, `--assemble`, `--verify` modes all intact.

### `papers/01_pipeline/run_nu_cache.sbatch`
- Header comment rewritten for the new grid (23 edges / 22 centres, Poisson motivation,
  points to `nu_grid.py` for the edges-vs-centres split) and states the earlier
  `nu8_shards/`/`emulator_dataset_nu8.npz` are left in place.
- Enumeration logic **unchanged in structure**: fiducial pair (`bind`, `truth`, `--sci`)
  enqueued first, then every Sobol run pulled live from
  `emulator_dataset_xpkfix.npz["run_ids"]`. Comment updated from "253 of 256" to "all 256"
  (the dataset now has 256 runs; the per-run `--run` calls are idempotent — a run whose
  shard already exists is skipped — so re-running is safe even though the fiducial pair was
  already built locally below).
- disBatch invocation, SLURM resources (`--ntasks=48`, `--mem=600G`, `--time=02:00:00`),
  and the final `--assemble` + `--verify` sequence are untouched, only the final echo path
  renamed to `emulator_dataset_nu05.npz`.
- **Not submitted** — sbatch execution is left to the user per instructions.

`_build_figures_nb.py` and `build_emulator.py` were **not touched**, per instructions —
they're a second stage that consumes the recache once it's assembled.

## Locally built shards (fiducial pair + one Sobol sanity check)

Ran directly on this 1-core node (no Slurm), each taking ~150 s as sized in the sbatch
comment:

```
$ python build_nu_cache.py --run bind --sci      # 149s -> nu05_shards/sci_bind.npz
$ python build_nu_cache.py --run truth --sci     # 150s -> nu05_shards/sci_truth.npz
$ python build_nu_cache.py --run run_0000        # 153s -> nu05_shards/run_0000.npz
```

All three land in `/mnt/home/mlee1/ceph/bind_sb35/nu05_shards/` (a NEW directory; nothing
under `nu8_shards/` was read or written).

| shard | key | shape | dtype | finite | min | max |
|---|---|---|---|---|---|---|
| sci_bind.npz | mf_v0 | (5, 22) | f64 | 100% | 9.94e-05 | 1 |
| sci_bind.npz | mf_v0_real | (50, 5, 22) | f64 | 100% | 7.63e-06 | 1 |
| sci_bind.npz | mf_v1 / mf_v1_real | (5,22) / (50,5,22) | f64 | 100% | 0 | 0.0143 / 0.0149 |
| sci_bind.npz | mf_v2 / mf_v2_real | (5,22) / (50,5,22) | f64 | 100% | −4.2e-4 | 6.6e-4 |
| sci_bind.npz | minima_counts (+_err,_real) | (5,22) / (5,22) / (50,5,22) | f64 | 100% | 0 | 256 (mean) / 323 (real) |
| sci_bind.npz | peak_counts (+_err,_real) | (5,22) / (5,22) / (50,5,22) | f64 | 100% | 0 | 136 (mean) / 175 (real) |
| sci_bind.npz | pdf / pdf_real | (5,22) / (50,5,22) | f64 | 100% | 0 | 0.71 / 0.77 |
| sci_bind.npz | nu | (22,) | f64 | 100% | −2.75 | 7.75 |
| sci_truth.npz | same keys | same shapes | f64 | 100% | (matches sci_bind within a few %, as expected for the fiducial pair) | |
| run_0000.npz | mf_v0/v1/v2, peak_counts(+_err), minima_counts(+_err), pdf, nu | (5,22) / (22,) — **no `_real` arrays** (per_real=False for non-`--sci` runs, by design) | f64 | 100% | — | — |

`sci_bind.npz` = 0.15 MB, `sci_truth.npz` = 0.16 MB (both carry the 50-realization `_real`
draws needed for fig 4's paired ±1σ/√50 bands), `run_0000.npz` = 0.01 MB (mean-only, as
released for the other 253/256 Sobol shards). All three are 100% finite everywhere — no
NaNs, consistent with the earlier full-suite `nu8` run.

## Build sequence for the user (not run)

Full Sobol recache (256 runs + fiducial pair, most already covered locally for `bind`,
`truth`, `run_0000` — the per-run step is idempotent and will skip those three):

```bash
sbatch papers/01_pipeline/run_nu_cache.sbatch
```

This does, in order (identical structure to the `nu8` job): enqueue `bind`/`truth` (`--sci`,
per-realization draws) + all 256 `run_XXXX` shard tasks over disBatch on 48 cores
(`--ntasks=48 --mem=600G --time=02:00:00`, `cca` partition), then automatically runs

```bash
python build_nu_cache.py --assemble   # -> /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_nu05.npz
python build_nu_cache.py --verify     # NaN-aware check: untouched arrays byte-identical,
                                       # new nu-domain arrays 22-wide and finite, axes == NU
```

If only a manual re-run of assemble/verify is needed later (e.g. after topping up a few
stragglers by hand), the two commands above can be run standalone from
`papers/01_pipeline/` with the venv activated — no Slurm required, they're fast
(assemble copies + writes one compressed .npz; verify just diffs two in-memory dicts).

## Open items for the next stage (not in scope here)

- `_build_figures_nb.py` and `build_emulator.py` still point at the `nu8`-era outputs (or
  whatever they currently reference) — updating them to consume
  `emulator_dataset_nu05.npz` and the 22-point `NU` grid is explicitly a second stage per
  the task instructions, not done here.
- The full 256-run recache has not been submitted; only the 2 fiducial + 1 Sobol sanity
  shard exist under `nu05_shards/` so far.
