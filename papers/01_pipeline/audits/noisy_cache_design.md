# Noisy-statistics cache: LSST-Y10 shape noise + smoothing (Lee+22 recipe)

2026-08-05. Builds a NOISY companion to the existing noiseless `nu_grid.py`/`nu05_shards/`
cache: every lightcone target's kappa maps get LSST-Y10 shape noise (Lee, Lu, Haiman, Liu
& Osato 2022, MNRAS 519, 573; arXiv:2201.08320 — the author's own peak-statistics paper,
referred to below as "L22") Eq. 7, then the paper's Eq. 6 Gaussian smoothing, then all six
threshold statistics plus `cl_kappa` on the noisy smoothed map. **No repaint**: only the
released `kappa_maps.npz` cubes are read.

Targets: `fid` = `bind_lightcone_tng/kappa_maps.npz` (550 real, the NEW fiducial
BIND-painted lightcone — **not** the older `bind_science/runs/bind/run_0000`, 50 real,
which `nu_grid.py`/`field_cache.py`/`mf_cache.py` still call "bind"), `truth` =
`bind_science/runs/truth/run_0000` (550 real), `dmo` = `bind_science/runs/dmo/run_0000`
(read dynamically — it happened to also be 550-real by the time this ran, was 50 when the
task was written; the code never assumes either), and all 256 Sobol runs,
`bind_sb35/runs/run_NNNN` (50 real each).

## Files delivered

- `papers/01_pipeline/noisy_grid.py` — recipe constants, seeded noise, smoothing, per-plane
  `kappa_rms` cache, and `compute()`. Mirrors `nu_grid.py`'s structure (imports its `NU`/
  `NU_EDGES` directly rather than redefining them) and docstring discipline.
- `papers/01_pipeline/build_noisy_cache.py` — CLI: `--kappa-rms`, `--run TARGET
  [--real-start/--real-end/--max-real]`, `--merge-chunks TARGET`, `--assemble`, `--verify`.
- `papers/01_pipeline/run_noisy_cache.sbatch` — disBatch over all targets, chunked. **Not
  submitted** (per instructions — the user runs it).
- Local validation shards (already built, see below): `_kappa_rms_fid.npz` (25-real smoke),
  `fid_chunk_00000_00025.npz` (25-real fid smoke shard), `run_0000.npz` (full 50-real Sobol
  shard) — all under `bind_sb35/nu05n_shards/`.

## The recipe, as implemented

**1. Shape noise (L22 Eq. 7), added BEFORE smoothing.** Per pixel, i.i.d. Gaussian, mean 0,

```
sigma^2 = sigma_e^2 / (2 * n_gal * A_pix)
```

`sigma_e = 0.26`, `n_gal = 27 arcmin^-2` (LSST-Y10), `A_pix = (0.29296875 arcmin)^2` (=
`5*60/1024`, these are 5-deg/1024-px maps). This is measured, not `bind.inference.stats`'
own `_noise_sigma_pix` (`sigma_e/sqrt(n_gal*A_pix)`, no factor of 2 — a *different* paper's
convention) — the module docstring flags this explicitly so the two are never conflated.
Numerically: `NOISE_SIGMA_PIX = 0.12077` (kappa units), vs. a typical fiducial per-map kappa
std of ~0.01 — i.e. the noise **dominates** the raw signal per-pixel, exactly as expected
for unsmoothed LSST-depth convergence.

**2. Smoothing (L22 Eq. 6), applied to the now-noisy map — the θ_G/σ conversion.**
L22's window is `W(theta) ~ exp(-theta^2/theta_G^2)` with `theta_G = 1 arcmin`.
**`theta_G` is the 1/e radius of that Gaussian, not its standard deviation.** Rewriting the
same window in the usual `exp(-theta^2/(2 sigma^2))` form (what `scipy.ndimage.
gaussian_filter`'s `sigma` argument expects) gives

```
sigma = theta_G / sqrt(2) = 0.70710678... arcmin = 2.413591... pixels
```

`SMOOTH_SIGMA_ARCMIN`/`SMOOTH_SIGMA_PIX` in `noisy_grid.py` are already this converted
sigma — call sites pass it straight to `gaussian_filter`. Using `theta_G=1'` directly as the
filter sigma (skipping the `/sqrt(2)`) over-smooths by a factor 1.414 and was flagged in the
module docstring as "the single easiest way to silently break this whole cache," since it
would still run and produce plausible-looking, systematically wrong numbers. `mode="wrap"`
matches `bind.inference.stats._gaussian_smooth`'s existing convention elsewhere in this
pipeline. Unlike `nu_grid.py` (PDF unsmoothed, peaks/minima at 2', MF at 1'), **every**
statistic here — PDF, peaks/minima, MFs, and `cl_kappa` — is computed on the **same**
noisy-then-smoothed map, per L22's convention that the analysis happens on the smoothed
field.

**3. ν normalisation — one shared κ_rms per source plane.** `nu = (smoothed -
smoothed.mean()) / kappa_rms[plane]`, with `kappa_rms[plane]` a SINGLE number: the mean,
over fiducial (`fid`) realizations, of each noisy-smoothed map's own std — the direct
analogue of L22's "use kappa_TNG's kappa_rms for everything," and of
`bind.inference.stats`'s `nu_norm="fixed"` convention (shared sigma so a parameter response
isn't partly absorbed by the map's own changing sigma_kappa). This is computed **once**
(`compute_kappa_rms`, cheap — only noise+smoothing+std, no extrema/MF/Cl) from the fiducial
target, cached atomically (`_kappa_rms_fid.npz`, tmp+`os.replace`, safe against concurrent
disBatch tasks), and **required** by every other build: `load_kappa_rms()` raises a clear
`FileNotFoundError` if the cache is missing rather than silently falling back to a per-map
or per-run normalisation — "refuse, don't guess," the same discipline `field_cache.py`/
`mf_cache.py`'s `--verify` gates use elsewhere in this directory. Every shard also carries
its own copy of the array under the `kappa_rms` key.

**4. Seeding — reproducible AND independent per target.** `noise_rng(target, realization,
plane)` seeds `np.random.default_rng` from `(stable_hash(target), realization, plane)`,
where `stable_hash` is a `sha256`-based hash (NOT Python's `hash()`, which is randomized
per-process by `PYTHONHASHSEED` and would make a "reproducible" seed not actually
reproducible across reruns/processes). Different `target` strings ("fid", "truth", "dmo",
"run_0000", …) therefore draw statistically independent noise realizations — verified
locally: `noise_rng("fid",0,0)` and `noise_rng("truth",0,0)` give different draws, while two
calls to `noise_rng("fid",0,0)` give identical draws. This matches a real survey, where BIND
and the hydro truth are not the same "observation" and would not see the same shape-noise
draw.

## Shard schema

Each `--run TARGET` build writes one `.npz` with (n_planes=5 fixed, asserted against each
cube's actual shape; n_nu=22 from `nu_grid.NU`; n_ell=724 from
`bind.inference.stats.power_spectrum`'s natural grid for these maps):

| key | shape | note |
|---|---|---|
| `peak_counts`, `minima_counts`, `pdf`, `mf_v0`, `mf_v1`, `mf_v2` | `(5, 22)` | realization mean |
| `peak_counts_err`, `minima_counts_err` | `(5, 22)` | realization SE (Poisson-adjacent, counts) |
| `*_real` (all six) | `(n_real, 5, 22)` | per-realization draws — **only for fid/truth/dmo** (`per_real=True`); Sobol shards omit these to save space over 256 runs |
| `cl_kappa`, `cl_kappa_err` | `(5, 724)` | per-plane AUTO spectrum of the noisy smoothed map (NOT the `(5,5,724)` tomographic matrix the noiseless `emulator_dataset*.npz` calls `cl_kappa` — different shape, do not conflate) |
| `cl_kappa_real` | `(n_real, 5, 724)` | fid/truth/dmo only |
| `nu`, `cl_ell` | `(22,)`, `(724,)` | shared axes |
| `kappa_rms` | `(5,)` | this shard's copy of the fixed nu denominator |
| `real_start`, `real_end`, `n_real`, `target` | scalars | chunk provenance |

Chunked fid/truth/dmo builds land as `{target}_chunk_{start:05d}_{end:05d}.npz`;
`--merge-chunks TARGET` concatenates them along the realization axis (verifying
contiguity from each chunk's own stored `real_start`/`real_end`, and that every chunk was
built against the *same* `kappa_rms`) and recomputes the mean/err summary from the complete
cube, writing the plain `{target}.npz`.

`--assemble` builds `emulator_dataset_nu05n.npz` — **Sobol runs only** — as a small,
self-contained dataset (not a copy-and-patch of the ~90-key noiseless
`emulator_dataset_nu05.npz`, because this file's `cl_kappa` has a genuinely different shape
from that one's; blanket-copying and overwriting would be actively misleading). It carries
`run_ids`/`X_native`/`X_unit`/`param_names` (copied from `emulator_dataset_nu05.npz` so the
two stay joinable), the `t__{stat}__value/err/valid` + `a__{stat}__{axis}` schema for the
six ν statistics plus `cl_kappa`, the shared `kappa_rms`, and a `noisy_grid_note` documenting
the recipe inline. `--verify` is NaN-aware (mirrors `build_nu_cache.py`): per-stat finite
fraction, shape checks, nu/ell axis checks, and a spot-check that a few raw shards reproduce
their row in the assembled arrays exactly.

## Local validation (this session, single core)

Built with `--max-real 25`/`--kappa-rms --max-real 25` (smoke-sized, not the production
550):

- `python build_noisy_cache.py --kappa-rms --max-real 25` → 85 s.
  `kappa_rms = [0.01601, 0.02063, 0.02509, 0.02886, 0.03170]` (planes 0–4, increasing with
  source redshift — physically sensible).
- `python build_noisy_cache.py --run fid --max-real 25` → 138 s (25 real × 5 planes),
  `fid_chunk_00000_00025.npz`, 855 KB.
- `python build_noisy_cache.py --run run_0000` → 110 s (50 real × 5 planes; the ~5.6 min
  estimate below was conservative — filesystem caching from the earlier fid-family reads
  likely helped). `run_0000.npz`, 70.7 KB — confirmed no `*_real` arrays present (Sobol
  shards correctly use `per_real=False`, mean+err only), `nu`/`cl_ell`/`kappa_rms` all
  present and matching the fid shard's axes.
- Guard rails: `--assemble` with zero Sobol shards present raised
  `FileNotFoundError: 256/256 Sobol shards missing, e.g. [...]` (not a silent partial
  build); `--verify` with no assembled dataset printed `MISSING: .../emulator_dataset_
  nu05n.npz` and returned exit 1. Both as designed.

**Sanity check (deliverable 3): noisy peak counts must exceed noiseless at low ν** (shape
noise manufactures small-amplitude local maxima that dominate the low-S/N bins). Compared
`fid_chunk_00000_00025.npz` (noisy, 25 real, NEW `bind_lightcone_tng` fiducial) against the
existing noiseless `bind_sb35/nu05_shards/sci_bind.npz` (50 real, OLD
`bind_science/runs/bind/run_0000` fiducial — a different underlying realization count/run,
so this is a qualitative check of noise's effect, not a bit-exact one, exactly as the task
asked):

| plane | Σ peak_counts(ν<-0.75), noisy | noiseless | ratio |
|---|---|---|---|
| 0 | 3389 | 407 | 8.3× |
| 1 | 3903 | 392 | 10.0× |
| 2 | 3851 | 391 | 9.9× |
| 3 | 3721 | 382 | 9.7× |
| 4 | 3611 | 378 | 9.6× |

Noisy peak (and minima) counts exceed noiseless by ~8–10× in the low-ν bins across all five
source planes, as expected. `pdf` integrates to 1.0 on `NU_EDGES` (`Σ pdf * Δν = 1.0000`) as
a basic normalisation check.

## Cost estimate for the full run (see `run_noisy_cache.sbatch` for the derivation)

Measured per-(realization,plane) cost on this workstation (1 core): noise+smooth 0.12 s,
peak/minima extrema 0.29 s, Minkowski functionals 0.81 s (dominant — several `np.gradient`
passes per 1024² map), `cl_kappa` power spectrum 0.07 s → **~1.3 s/unit**. Decompressing one
fid/truth/dmo `kappa_maps.npz` (10.67 GB compressed) costs **~110 s regardless of chunk
size** (`savez_compressed` gives no partial/random-access read — confirmed both by direct
timing and by inspecting the zip member's compression type, `ZIP_DEFLATED`). Sobol run
(~1 GB): ~10 s.

- Sobol run (50×5=250 units): 10 + 250×1.3 ≈ 335 s (~5.6 min); 256 runs serial ≈ 23.9 h.
- fid/truth/dmo in ONE task (550×5=2750 units): 110 + 2750×1.3 ≈ 3685 s (~61 min) — over
  the ~30 min/task budget, hence 4 realization-range chunks/target (~138 real each):
  110 + 138×5×1.3 ≈ 1007 s (~16.8 min)/chunk, comfortable margin. 3×4 = 12 heavy tasks.
- `kappa_rms` (noise+smooth+std only, full 550-real fiducial, single un-chunked task):
  110 + 550×5×0.12 ≈ 448 s (~7.5 min).
- **Total serial compute ≈ 27.0 h**; over `--ntasks=40` in `run_noisy_cache.sbatch`,
  ≈ 40–60 min wall time (plus module/merge/assemble overhead) — `--time=03:00:00` for
  margin on a first-run job. Peak memory (worst case, all 12 heavy tasks land in the same
  wave): 12×11.5 GB + 28×1.1 GB ≈ 169 GB, well under the requested `--mem=450G`.

## Command for the user (NOT submitted)

```
sbatch papers/01_pipeline/run_noisy_cache.sbatch
```

Runs `--kappa-rms` (full 550-real fiducial) standalone first, then a disBatch pool of the 12
heavy fid/truth/dmo chunks + 256 Sobol runs, then `--merge-chunks` for fid/truth/dmo,
`--assemble`, `--verify`.

## `paired_perreal_fid.npz`: writer + the regeneration command (deliverable 4, NOT run)

**Writer.** `src/bind/cli/paired_stats.py`'s `_load_or_build_fid()`, invoked by
`bind-paired-stats` / `python -m bind.cli.paired_stats --run_dir <RD> --fid_dir <FID_DIR>`.
`run_paired_stats.sh` (SLURM array) calls this with `RD == FID_DIR` for array task 0 only —
`main()` special-cases `run_dir == fid_dir` to build+cache the fiducial per-realization cube
(`clk`, `sigma0`, `pk`/`min`/`V0`/`V1`/`V2`/`sk` at a **fixed-nu** convention, note: this is
a *third*, different nu convention from both `nu_grid.py`'s and this task's `noisy_grid.py`'s
— don't conflate the three) and return without writing a `paired_stats.npz` response.
Currently `FID_DIR` defaults to `$OUTPUT_ROOT/runs/bind/run_0000` =
`bind_science/runs/bind/run_0000`, which is why `paired_perreal_fid.npz` lives there today
— still the **50-real vintage**, per `_build_figures_nb.py`'s own guarded fig-4b cell
(`RB = SCI/"runs/bind/run_0000"`, checked live via each cache's `n_real` field; as of this
session `_N_BIND_KAPPA` there is still 50, so `HAS_550_PAIRED` stays `False` and fig 4b keeps
its `±1σ/√50` bands).

**Where the NEW 550-real fid data actually is.** `bind_lightcone_tng/kappa_maps.npz` (550
real, confirmed this session) — **not** `bind_science/runs/bind/run_0000` yet. `_perreal`
only reads `kappa_maps.npz` (not `y_maps.npz`) to build `paired_perreal_fid.npz`, so
`bind_lightcone_tng`'s not-yet-550 `y_maps.npz` (still ~954 MB, old-vintage size) does not
block this specific regeneration, even though it would block other parts of fig 4b that need
`y`.

**Gotcha:** `_load_or_build_fid` only checks `cache.exists()`, never whether its `n_real`
matches the underlying cube — re-running the command against a directory that already has a
`paired_perreal_fid.npz` is a silent no-op. The stale file must be removed first.

**Exact command to regenerate from the new 550-real fid data:**

```bash
rm -f /mnt/home/mlee1/ceph/bind_lightcone_tng/paired_perreal_fid.npz   # only if one exists there already

python -u -m bind.cli.paired_stats \
    --run_dir /mnt/home/mlee1/ceph/bind_lightcone_tng \
    --fid_dir /mnt/home/mlee1/ceph/bind_lightcone_tng
```

(`--run_dir == --fid_dir` intentionally — this is what makes `paired_stats.py` build+cache
the fiducial cube and return, rather than also writing a `paired_stats.npz` response for some
other run against it.) `--n_real` is left unset, so `_perreal` uses **all 550** realizations
for `clk` (`(550, 5, 724)`); `sigma0` is unaffected by N — `_sigma0` caps itself to the first
20 realizations regardless (`n=20` default), so it stays cheap even at 550. Not timed here
(explicitly not run), but by the same per-unit costs measured above, expect this to take on
the order of tens of minutes serial (fewer scales than `noisy_grid.py`'s recipe — no shape
noise/smoothing pass — but `nongaussian_stats`'s default 4 smoothing scales plus `peak_counts`
plus the `clk` spectra, over 550×5=2750 units, un-chunked).

**Then**, for fig 4b to actually pick this up: the notebook currently hardcodes
`RB = SCI/"runs/bind/run_0000"`, not `bind_lightcone_tng`. Either redirect that path (once
BIND's own `kappa_maps.npz`/`y_maps.npz` at the `RB` location are themselves refreshed to
550 — a separate retrace step, already flagged in the notebook as "being handled
separately") or copy/symlink the regenerated `bind_lightcone_tng/paired_perreal_fid.npz`
into `bind_science/runs/bind/run_0000/` once that directory's own kappa/y maps reach 550.
The notebook's `HAS_550_PAIRED` gate (checks each cache's own `n_real` field) needs no
further code change once the maps and cache genuinely reach 550 at the path it reads.
