# Reproducing the methods-paper figures

Every figure in the BIND methods paper is produced by a script in
[`tools/paper_cache/`](https://github.com/Maxelee/BIND/tree/main/tools/paper_cache).
The workflow is deliberately split in two:

1. **Cache builders** (slow, GPU or many-CPU, resumable) walk a CAMELS
   suite-evaluation tree and reduce it to a handful of compact artifacts —
   `mass_table.pkl`, `profiles_r200.pkl`, `shapes.pkl`, `pk_fixed.npz`,
   `zcache.npz`, …
2. **Figure scripts** (fast, CPU, load-only) read those artifacts and plot.
   `tools/paper_cache/figures/*.py` are one-file-per-figure; a few figures
   still live in the load-only notebooks under `examples/`.

Almost all of the cost is in step 1. The compact artifacts from step 1 are what
gets published, so the figures can be regenerated without re-running the
emulator.

```{important}
Read the [Provenance](#provenance-and-what-will-not-bit-reproduce) section
before you try to reproduce a *cache*. Re-running the suite evaluation with
today's defaults will **not** bit-reproduce the archived caches, because the
archived main-body evaluation predates the `paste_mode` flag.
```

## Getting the data

The compact caches are being deposited on Zenodo.

```{admonition} DOI: TBD
:class: warning
The Zenodo record is not yet minted. **This is a placeholder — do not cite or
link a DOI until it appears here.** Until then the caches exist only on the
authors' institutional filesystem — contact the authors if you need them before
the record is minted.
```

The artifacts listed below total **194 MB** as measured on disk. The full
working cache is larger — 80.7 MB (`fm_two_head`) + 140.4 MB (`fm_redshift`) +
13.1 MB (`cube_comparison`) = 234 MB, of which 44.6 MB is `fm_two_head/partials`
and other resumable intermediate state the figures never read. Unpack the deposit anywhere and
point one environment variable at it:

```bash
export BIND_PAPER_DATA=/path/to/unpacked/bind_paper_data
```

```{note}
`BIND_PAPER_DATA` is a **convention introduced by this page**, not a variable
any code reads. `tools/paper_cache/paper_config.py` reads `PAPER_CACHE_DIR`
(and `PAPER_SUITE_ROOT` / `PAPER_MODEL_SUBDIR` / `PAPER_MASS_DIR` /
`PAPER_MODEL_TAG` / `PAPER_SNAP`). Derive those from `BIND_PAPER_DATA` as shown below.
```

Expected layout, mirroring `paper_cache/` on ceph:

```
$BIND_PAPER_DATA/
    fm_two_head/                 # main-body cache
        mass_table.pkl  mass_param.pkl
        profiles.pkl    profiles_r200.pkl
        shapes.pkl      field1p.npz
        pk.npz          pk_fixed.npz
        ode_conv_cv.npz ode_conv_sb35.npz
        timing_benchmark.json
        partials/profiles/*.npz          # 267 per-sim files, needed by fig_radial_pct_diff
        posterior_calibration/ensemble_200x100.npz
    fm_redshift/                 # §5.5 redshift cache
        zcache.npz  zsweep.npz  zindex_test.npz
    cube_comparison/             # Appendix: projection-depth characterization
        cube_comparison_table.pkl  cube_comparison_profiles.npz
```

Then, for any main-body figure:

```bash
source /path/to/venv/bin/activate          # needs numpy, pandas, matplotlib, scipy
export PAPER_MODEL_TAG=fm_two_head
export PAPER_CACHE_DIR=$BIND_PAPER_DATA/fm_two_head
export PAPER_MODEL_SUBDIR=fm_two_head
export PAPER_MASS_DIR=mass_threshold_1p000e13
export PAPER_SUITE_ROOT=/unused/for/load-only/figures   # see caveat below

python tools/paper_cache/figures/fig_mass_error.py
```

Figures are written to `examples/paper_figures/` as `{pdf,png}` at 300 dpi.

```{caution}
**Always export all four `PAPER_*` variables, even though only some scripts
check them.** Six of the figure scripts (`f8_field1p_showcase.py`,
`fig2_baryon.py`, `fig5b_total_field_pk.py`, `fig_mass_error.py`,
`fig_p15_sweep.py`, `mass_error_table.py`) fail fast when any is unset. Two
(`f3_scaling_corner_matched.py`, `timing_benchmark.py`) `setdefault` them to the
`fm_two_head` layout. But `f4_shapes.py`, `spearman_sb35.py` and `fig_radial_pct_diff.py` set nothing and
simply inherit `paper_config.py`'s built-in defaults, which point at the
*development* suite (`fm_lowmass` / `fm_thermo_ema`) — so forgetting the exports
there silently plots the wrong model instead of erroring.

Three scripts bypass `PAPER_*` entirely and read absolute paths:
`f9_pit_calibration.py` (override with `PIT_NPZ`),
`fig_ode_convergence.py` (hardcoded, no override), and
`fig_cube_comparison.py` (`PAPER_CUBE_CACHE`, `PAPER_FIG_DIR`, and the optional
`BIND_PAPER_IMGS` for copying the pdf into a local checkout of the manuscript).

`PAPER_SUITE_ROOT` is only *used* by the figures that touch the raw suite tree —
`fig1_showcase_composite`, `fig_cube_comparison`, `flow_matching_diagram`,
`timing_benchmark`. Those cannot be reproduced from the compact deposit alone;
see [What cannot be reproduced](#what-cannot-be-reproduced-from-public-artifacts).
```

## The figure table

`n_steps` is the sampler setting the *generated content* was made with.
`r200_factor` / `paste_mode` apply only to figures that composite patches into a
full-box map; per-halo and per-patch figures are marked `n/a`.

### Main body — model `fm_two_head`

| figure (label) | producing script | cached artifact | model / run | n_steps | r200_factor | paste_mode |
|---|---|---|---|---|---|---|
| `fig:flow_matching` (`flow_matching_diagram`) | `make_flow_matching_fig.py` | CV/sim_0 `halo_cutouts.npz`, `halo_catalog.npz`, `full_maps.npz` + live sampler | **`fm_redshift` @ a=1** (see note) | 20 | n/a | n/a |
| `fig:showcase` (`fig1_showcase_composite`) | `figures/f8_field1p_showcase.py` | CV/sim_0 `generated_halos.npz`, `full_maps.npz`, `halo_catalog.npz`, `halo_cutouts.npz` | `fm_two_head` | 50 | 4.0 | `shared` (rebuilt at plot time) |
| `fig:each_halo_mass_comp` (`fig2_mass_comparison`) | `figures/fig2_baryon.py` | `mass_table.pkl`, `mass_param.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:baryon_fraction` (`obs_baryon_fraction`) | `figures/fig2_baryon.py` | `mass_table.pkl`, `mass_param.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:mass_comp` (`fig_mass_error`) | `figures/fig_mass_error.py` | `mass_table.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `tab:mass_residuals` (`mass_error_table.tex`) | `figures/mass_error_table.py` | `mass_table.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:p15_sweep` | `figures/fig_p15_sweep.py` | `mass_table.pkl`, `mass_param.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:total_density` (`fig_radial_pct_diff`) | `figures/fig_radial_pct_diff.py` | `partials/profiles/*.npz` | `fm_two_head` | 50 | n/a | n/a |
| `fig:axes_ratios` (`shape_1_axisratio`) | `figures/f4_shapes.py` | `shapes.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:shape_perhalo` | `figures/f4_shapes.py` | `shapes.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:pit_perhalo` | `figures/f9_pit_calibration.py` | `posterior_calibration/ensemble_200x100.npz` | `fm_two_head` `last.ckpt` | 50 | n/a | n/a |
| `fig:mass_corr` (`fig3_spearman_param_mass`) | `figures/spearman_sb35.py` | `mass_table.pkl`, `mass_param.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:mass_corr_pooled` | `figures/spearman_sb35.py` | `mass_table.pkl`, `mass_param.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:dens_corr` (`fig3_spearman_profile`) | `figures/spearman_sb35.py` | `profiles_r200.pkl`, `mass_param.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:butterfly` (`fig7_1p_field_response`) | `figures/f8_field1p_showcase.py` | `field1p.npz` | `fm_two_head` | 50 | n/a | n/a |
| `fig:ode_conv` (`fig_ode_convergence`) | `figures/fig_ode_convergence.py` | `ode_conv_cv.npz`, `ode_conv_sb35.npz` | **not recorded** (see note) | 20/50/100/200/400 vs ref 400 | 4.0 | `shared` |
| `fig:suppression` (`fig5b_total_field_pk`) | `figures/fig5b_total_field_pk.py` | `pk_fixed.npz` | `fm_two_head` | 50 | 4.0 | `shared` (repasted by `build_pk_fixed.py`) |
| `tab:timing` | `figures/timing_benchmark.py` | `timing_benchmark.json` (+ live GPU) | `fm_two_head` `last.ckpt` | 50 | 4.0 | `shared` |
| `fig:relations` (`scaling_fits`) | `figures/f3_scaling_corner_matched.py` | `mass_table.pkl`, `mass_param.pkl`, `dmo_sums.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:matched_residuals` (`fig_matched_residuals`) | `figures/f3_scaling_corner_matched.py` | `mass_table.pkl`, `mass_param.pkl`, `dmo_sums.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:corner` (`residual_corner_plot`) | `figures/f3_scaling_corner_matched.py` | `mass_table.pkl`, `mass_param.pkl`, `dmo_sums.pkl` | `fm_two_head` | 50 | n/a | n/a |
| `fig:cube_comp` | `figures/fig_cube_comparison.py` | `cube_comparison_table.pkl`, `cube_comparison_profiles.npz` | `fm_two_head` vs `fm_cube_two_head` | 50 | n/a | n/a |

### §5.5 Redshift dependence — model `fm_redshift`

| figure (label) | producing script | cached artifact | model / run | n_steps | r200_factor | paste_mode |
|---|---|---|---|---|---|---|
| `fig:z_mass_error` (`fig_z_mass_error`) | `make_z_figures.py` | `zcache.npz` | `fm_redshift` `last.ckpt` | 20 | n/a | n/a |
| `fig:z_response` (`fig_z_response`) | `make_z_figures.py` | `zsweep.npz` | `fm_redshift` `last.ckpt` | 20 | n/a | n/a |

`make_z_figures.py` also emits `fig_z_evolution`, `fig_z_radial_pct_diff` and
`tab_z_snapshots.tex`, which are **not** in the current manuscript. Select with
`--figures R1 R4`.

### Notes on two rows

```{note}
**`fig:flow_matching` uses a different model from the rest of the main body.**
`make_flow_matching_fig.py` hardcodes `RUN_DIR = .../fm_runs/fm_redshift` and
`n_steps = 20`, and its module docstring still describes `fm_redshift` as "the
paper spine model". The main body was later moved to `fm_two_head`
(`n_steps = 50`) but this builder was not updated, and the shipped
`flow_matching_diagram.pdf` predates that move. The figure is qualitative — a
noise → data trajectory for one CV halo — so nothing quantitative depends on
it, but the mismatch is real and should be stated rather than papered over.
```

```{note}
**The ODE-convergence caches carry no embedded provenance.** `ode_conv_cv.npz`
and `ode_conv_sb35.npz` store only per-sim spectra keyed by step count. They
live in the `fm_two_head` cache namespace, but `ode_convergence.py`'s built-in
`--ckpt` default points at an `fm_thermo` epoch-064 EMA checkpoint, so the
checkpoint must have been supplied on the command line and is not recoverable
from the artifact. Treat the model behind that figure as *fm_two_head by
placement, unverified by record*.
```

## Rebuilding a cache from scratch

Only needed if you have the CAMELS suite-evaluation trees. Full recipe in
[`tools/paper_cache/README.md`](https://github.com/Maxelee/BIND/blob/main/tools/paper_cache/README.md).

```bash
export PAPER_SUITE_ROOT=/path/to/fm_testsuite
export PAPER_MODEL_SUBDIR=fm_two_head
export PAPER_MASS_DIR=mass_threshold_1p000e13
export PAPER_MODEL_TAG=fm_two_head

POOL=16 bash run_paper_cache.sh                       # CPU metrics + reduce
python tools/paper_cache/build_pk_fixed.py --metric fixed --pool 6
python tools/paper_cache/build_pk_fixed.py --metric fixed --reduce
python tools/paper_cache/build_gpu_insets.py --which all   # GPU
```

And the suite evaluation that feeds it, per suite:

```bash
bind-camels-suite --suite cv --run_dir weights/fm_two_head \
    --checkpoint_path weights/fm_two_head/last.ckpt \
    --model_name fm_two_head --output_root /path/to/fm_testsuite \
    --halo_mass_min 1e13 --n_steps 50 ...
```

The §5.5 caches do not come from a suite evaluation at all —
`build_zcache.py` runs the sampler directly over the held-out split of the
multi-redshift *training* dataset:

```bash
python tools/paper_cache/build_zcache.py --run_name fm_redshift --n_steps 20 --mass_min 1e13 --n_per_z 800
python tools/paper_cache/make_z_figures.py
```

The archived `zcache.npz` holds 4674 held-out patches
(800 each at snapshots 90/82/74/60/52, 566 at 44, 95 at 32, 13 at 24) at
`mass_min = 1e13`, `n_steps = 20`, `run_name = fm_redshift`, `ckpt = last.ckpt`
— all recorded inside the file. `zsweep.npz` sweeps a fixed set of $z=0$ DMO
patches over `z_grid = [0, 0.1, 0.2087, 0.33, 0.4679, 0.7, 1.0452, 1.25,
1.4837, 1.75, 2.002]`, which deliberately includes off-training-grid redshifts.

## Provenance, and what will not bit-reproduce

This is the part to read before claiming a reproduction.

### Main body: the archived evaluation predates `paste_mode`

The main-body cache was built from the `fm_testsuite` evaluation tree. Reading
the archived `summary.json` files under
`<suite>/<sim>/snap_090/mass_threshold_1p000e13/fm_two_head/`:

| suite | summaries | `n_steps` | `r200_factor` | `paste_mode` |
|---|---|---|---|---|
| CV | 27 | 50 (27/27) | 4.0 (26/27; `CV/sim_17` records none) | **absent (0/27)** |
| 1P | 139 | 50 (139/139) | 4.0 (139/139) | **absent (0/139)** |
| SB35 (`Test`) | 102 | 50 (102/102) | 4.0 (102/102) | **absent (0/102)** |

`paste_mode` is absent because the flag did not exist when that evaluation ran.
Its behaviour was the **legacy `average`** overlap handling — the one that
blends independent flow-matching realizations wherever apertures overlap and
loses the sampled high-$k$ power (see {doc}`circular_aperture`). Today every
entry point defaults to `paste_mode="shared"`.

Consequences, stated plainly:

- **Re-running the suite evaluation with today's defaults will not reproduce the
  archived tree.** The generated per-halo patches are stochastic anyway (the
  archived run was unseeded), and the composites additionally differ by the
  paste change. Add `--paste_mode average` to match the archived compositing
  convention; you still will not get bit-identical content.
- **The archived per-sim `composite.npz` files are legacy-paste.** Anything
  reading them inherits that. `pk.npz` (built by
  `build_metric.py --metric pk`) does exactly this — which is why the paper's
  Fig. 5 uses `pk_fixed.npz` instead.
- **The two composited figures are not affected**, because both rebuild the
  composite at plot/cache time from the cached per-halo `generated_halos.npz`
  with `build_bind_composite(..., r200_factor=4.0, paste_mode="shared")`:
  `f8_field1p_showcase.py` for `fig:showcase`, and `build_pk_fixed.py` for
  `fig:suppression`. Every other main-body figure is per-halo or per-patch and
  never touches the paste.
- **The `fm_two_head` cache used analytic $R_{200c}$ apertures.** The
  `fm_testsuite` halo catalogs store the legacy `radii` key in kpc/h, which
  `paper_config.r200_pix_patch` does not read, so it fell back to deriving
  $R_{200c}$ from $M_{200c}$. The radius difference is ≤1.7%.

### §5.5: uniform and fully recorded

The `fm_redshift_suite` evaluation is internally consistent: all **1469**
`summary.json` files under `<suite>/<sim>/snap_*/mass_threshold_1p000e13/fm_redshift/`
(CV 27 × 6 snapshots, 1P 139 × 5, SB35 102 × 6) record
`n_steps = 20`, `r200_factor = 4.0`, `paste_mode = "shared"`, together with the
resolved `scale_factor` and `redshift`. That evaluation is *not* what the §5.5
figures read, though — they read `zcache.npz` / `zsweep.npz`, which
`build_zcache.py` produced by running the sampler directly (also at
`n_steps = 20`).

### Seeding

The archived runs are unseeded: `bind.paint` / `bind-camels-suite` /
`bind-paint` only gained `--seed` after the paper caches were built, and none of
the archived `summary.json` files carry a provenance block. So even holding
every flag fixed, a re-run of the archived configuration draws a different
realization. Statistical quantities (medians over hundreds of halos, $P(k)$
ratios over tens of sims) reproduce; individual patches do not.

New runs are better off: every entry point now accepts `--seed` and stamps a
provenance block — `bind_version`, checkpoint and norm-stats paths with their
sha256s, and the resolved `n_steps` / `r200_factor` / `paste_mode` / `seed` —
into `summary.json` and into every output `.npz` (`provenance` key, read with
`bind.inference.artifacts.read_provenance`). A cache built from here on is
self-describing in the way the archived one is not.

## What cannot be reproduced from public artifacts

The compact deposit is enough for the figure scripts that are load-only.
It is **not** enough for:

| figure / product | why | what it additionally needs |
|---|---|---|
| `fig:showcase` | rebuilds the composite from per-halo patches | CV/sim_0 `generated_halos.npz` + `full_maps.npz` + `halo_catalog.npz` + `halo_cutouts.npz` from the suite tree |
| `fig:flow_matching` | runs the sampler live | GPU + the `fm_redshift` checkpoint + CV/sim_0 cutouts |
| `tab:timing` | measures wall-clock | the specific GPU (an A100-SXM4-80GB 2g.20gb MIG slice) + the suite prep artifacts |
| `fig:cube_comp` | joins two evaluation trees | `fm_testsuite` **and** `fm_testsuite_cube` |
| any `build_gpu_insets.py` product | regeneration | GPU + checkpoints |
| the caches themselves | see above | the full CAMELS SB35/CV/1P DMO + hydro + FoF data and the suite-evaluation trees |

The underlying CAMELS simulations are public
([camels.readthedocs.io](https://camels.readthedocs.io)), so the suite trees can
in principle be regenerated with `bind-camels-suite` — at the cost of the full
evaluation, and, per the provenance section, not bit-identically.
