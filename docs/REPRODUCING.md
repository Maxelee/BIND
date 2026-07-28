# Reproducing `examples/paper_p4c_tsz_ksz.ipynb`

Figure-by-figure map from the P4c paper notebook back to the campaign
products and the engine scripts that produced them. The notebook itself
**never re-measures anything** — it only reads merged `.npz` products and
verdict `.json` files under the products root and re-plots/re-prints from
them (see `examples/_build_p4c_paper_nb.py`, which is the sole editor of
the notebook — do not hand-edit the `.ipynb`).

## Quick start for outsiders

If all you want is the 10 paper figures on screen, and you don't care about
rebuilding any upstream product from raw data, this is the entire pipeline:

1. **Point four env vars at your copy of the products tree.** All four
   default to the historical campaign paths used throughout this doc
   (`/mnt/home/mlee1/ceph/...`), so **on the campaign filesystem you can
   skip this step entirely**:
   ```bash
   export BIND_KSZ_PRODUCTS=/path/to/ksz_confront       # KS: figures/verdicts/products
   export BIND_KSZ_DOWNLOADS=/path/to/downloads          # DL: raw external data (ACT/Planck/DESI/DR5)
   export BIND_SB35_RUNS=/path/to/bind_sb35              # the 253-node Sobol design + per-run outputs
   export BIND_SCIENCE_RUNS=/path/to/bind_science/runs   # the truth-hydro comparison run (R7)
   ```
   (full detail, defaults, and per-script coverage in the next section).
2. **Install the deps.** `pip install -e .` from the repo root (or `source
   /mnt/home/mlee1/venvs/BIND_env/bin/activate` on the campaign
   filesystem, which already has everything) gets you the core `bind`
   package. The figure-chain scripts additionally import ordinary
   pip-installable science packages — `numpy pandas matplotlib scipy
   nbformat jupyter` cover the notebook-rebuild path in step 3 below; a
   full from-scratch *products* rebuild (see "Consolidated rebuild order")
   also touches `healpy pixell pyccl camb scikit-learn emcee cloudpickle
   fitsio astropy`, each used by only one or two specific upstream engines
   (e.g. `pyccl`/`camb` only in `lightcone_external_suppression.py`;
   `healpy`/`pixell` only in `act_ycap_measure.py` and
   `planck_ycap_crosscheck.py`; `emcee`/`cloudpickle` only in
   `_r8_gp_mcmc.py`) — install them only if you're rebuilding that
   specific product.
3. **Regenerate the notebook** from the already-merged products (no raw
   maps touched, ~3 minutes):
   ```bash
   source /mnt/home/mlee1/venvs/BIND_env/bin/activate
   python examples/_build_p4c_paper_nb.py
   jupyter nbconvert --to notebook --execute --inplace examples/paper_p4c_tsz_ksz.ipynb
   ```
   Kernel `bind_env` is declared in the notebook metadata — do **not** pass
   a `--ExecutePreprocessor.kernel_name` override; the generic `python3`
   kernel lacks `scipy`. Verify zero error outputs after execution (this is
   also the acceptance gate for any change — see below).

Everything past this point explains, figure by figure, what each merged
product *is* and how to rebuild it from scratch if it's ever lost. Jump to
"Figure-by-figure" for that map, or straight to "Consolidated rebuild
order" at the bottom for the full from-scratch chain.

## `BIND_KSZ_PRODUCTS` and friends — the products-root convention

All P4c products live under one root, referred to as `KS` throughout the
campaign docs and code:

```
KS = <products_root>                       # default: /mnt/home/mlee1/ceph/bind_science/ksz_confront
LC = KS/lightcone                          # almost everything below is under here
VD = LC/verdicts                           # verdict JSONs (single source of truth for quoted numbers)
```

`examples/act_ycap_measure.py` and `examples/lightcone_m2r_ycap_real.py`
(the P4 conversions) additionally accept an explicit `--products_root`
flag. Resolution order (identical in both scripts):

```
--products_root <path>   >   $BIND_KSZ_PRODUCTS   >   /mnt/home/mlee1/ceph/bind_science/ksz_confront
```

With neither the flag nor the env var set, behavior is byte-identical to
before this change (pure refactor, verified). Set `BIND_KSZ_PRODUCTS` once
to point every subsequent invocation of these two scripts at a different
products root, e.g. a fresh copy for a from-scratch rebuild:

```bash
export BIND_KSZ_PRODUCTS=/path/to/my_ksz_confront
python examples/act_ycap_measure.py --mode dr5 --grid rap --out T1_dr5_snr5.npz
python examples/lightcone_m2r_ycap_real.py
```

**Round-2 T3** (`docs/paper_improvement_plan.md`) extended the same
env-var override to every other engine in the rebuild chain below, plus
three more env vars for the non-products roots those scripts also
hardcode. All four resolve independently and default to the historical
hardcoded path, so with none set, every script's behavior is
byte-identical to before this task (verified per-file: `python -m
py_compile` + an import-time constant check with the env vars set and
unset — `_t2_fig_verdict.py`, which has no `if __name__` guard, was
verified by code inspection only, not executed):

| env var | default | moves |
|---|---|---|
| `BIND_KSZ_PRODUCTS` | `/mnt/home/mlee1/ceph/bind_science/ksz_confront` | `KS` — all lightcone products/figs/verdicts |
| `BIND_KSZ_DOWNLOADS` | `/mnt/home/mlee1/ceph/paper3/B/downloads` | `DL` — the raw external-data release (ACT/Planck y-maps, DR5 clusters, DESI LRG catalogs) |
| `BIND_SB35_RUNS` | `/mnt/home/mlee1/ceph/bind_sb35` | the SB35 Sobol tree (`design/`, `runs/`, `analysis_cache/`) |
| `BIND_SCIENCE_RUNS` | `/mnt/home/mlee1/ceph/bind_science/runs` | the truth-hydro comparison run used by R7 |

```bash
export BIND_KSZ_PRODUCTS=/path/to/my_ksz_confront
export BIND_KSZ_DOWNLOADS=/path/to/my_downloads
export BIND_SB35_RUNS=/path/to/my_bind_sb35
export BIND_SCIENCE_RUNS=/path/to/my_bind_science_runs
```

### Coverage — mechanism per script

Only `act_ycap_measure.py` and `lightcone_m2r_ycap_real.py` gained a CLI
flag (`--products_root`); every other script below is **env-var only** —
either because it has no argparse at all (a straight-line script or a
script whose only CLI is unrelated switches), or, for
`lightcone_hod_stack.py` and `_r8_gp_mcmc.py` specifically, because adding
a flag would be actively wrong (see notes below). No new CLI surface was
added to any script; `--help` still exits 0 unchanged for every script
that already had argparse.

| script | mechanism | roots parameterized |
|---|---|---|
| `act_ycap_measure.py` | `--products_root` flag + env | `KS` (`BIND_KSZ_PRODUCTS`); `DL` (`BIND_KSZ_DOWNLOADS`, env-only) |
| `lightcone_m2r_ycap_real.py` | `--products_root` flag + env | `KS`/`LC` (`BIND_KSZ_PRODUCTS`) |
| `_r1_closure.py` | env-only | `KS` (`BIND_KSZ_PRODUCTS`); `BEAM_TXT` (`BIND_KSZ_DOWNLOADS`) |
| `_r3_mass_anchor.py` | env-only | `KS` |
| `_r4_twohalo.py` | env-only | `KS` |
| `_r5_cib_systematics.py` | env-only | `KS` |
| `_r5c_liu_figure.py` | env-only | `KS` (`LC` rebuilt from it; previously a standalone hardcoded `LC`) |
| `_r6_ksz_audit.py` | env-only | `KS` |
| `_r7_fidelity_closure.py` | env-only | `KS`; `TRUTH_Y` (`BIND_SCIENCE_RUNS`) |
| `_r8_gp_mcmc.py` | env-only, deliberately no flag (see note) | `KS`/`LC`/`FIG_DIR`/`VERDICT_DIR` (`BIND_KSZ_PRODUCTS`); `DESIGN`/`PARQUET` (`BIND_SB35_RUNS`); `SCRATCH`/chain state **untouched** |
| `lightcone_hod_stack.py` | env-only, deliberately no flag (see note) | `KS`/`LC`/`CAT` (`BIND_KSZ_PRODUCTS`); `RUNS` (`BIND_SB35_RUNS`) |
| `lightcone_latent_corner.py` | env-only | `KS`/`LC` (`BIND_KSZ_PRODUCTS`); `DESIGN`/`PARQUET` (`BIND_SB35_RUNS`) |
| `lightcone_capstone_x.py` | env-only | `KS`/`LC` (`BIND_KSZ_PRODUCTS`); `DESIGN` (`BIND_SB35_RUNS`) |
| `lightcone_m1_fgas_desiact.py` | env-only | `KS` (`BIND_KSZ_PRODUCTS`) |
| `_t2_fig_verdict.py` | env-only | `KS` (`BIND_KSZ_PRODUCTS`) |

Notes:
- **Why no `--products_root` flag on `lightcone_hod_stack.py`:** it
  already has argparse (`--kcal_prep`/`--node`/`--merge`/`--explore`), but
  `CONFIG_NPZ`/`SHARD_DIR` — also imported by `_r3_mass_anchor.py` — are
  derived from `LC` at **module-import time**, before argparse runs. A
  flag that only rebinds `KS`/`LC` in the `__main__` block would leave
  those two stale, a silent correctness bug. The env var is resolved at
  import time (before those derived constants are computed), so it is the
  one mechanism that stays consistent everywhere this module's globals
  are read — including by the cross-module import in `_r3_mass_anchor.py`.
- **Why no `--products_root` flag on `_r8_gp_mcmc.py`:** its `--stage`/
  `--leg`/`--budget` CLI drives **live, long-running MCMC chains**;
  per the task brief this file was deliberately kept minimal-risk.
  `SCRATCH`/`INGREDIENTS_PATH` (the emcee `HDFBackend` checkpoint dir for
  the 4 in-flight chains) are **not** derived from `KS` and do **not**
  honor any env var — they remain the literal historical path so chains
  are never silently redirected mid-run.
- **`FID_Y`/`FID_KAPPA`** (`/mnt/home/mlee1/ceph/bind_lightcone_tng/...`,
  appearing in `_r1_closure.py`, `_r7_fidelity_closure.py`,
  `lightcone_hod_stack.py`) are a fifth hardcoded root not covered by any
  of the four env vars above — out of scope for this task (not one of the
  three non-products roots named in the Round-2 T3 brief).
- **`lightcone_m2_ycap_liu.py`** also hardcodes `KS`/`CEPH` at module
  level and is imported (for two unrelated helper functions,
  `mean_theta200_arcmin`/`N_XB`) by `_r6_ksz_audit.py` and
  `lightcone_m1_fgas_desiact.py`, but it is not named anywhere in the
  figure-by-figure map or rebuild order below, so it was left untouched —
  flagged here as a residual gap, not a silent omission.
- Still uncovered (out of this task's scope — not part of the figure
  chain below): the earlier M2–M7 map-level campaign scripts
  (`lightcone_m2_ycap_liu.py`, `lightcone_m3_kappa_anchor.py`, `lightcone_m4_elg_zshell.py`,
  `lightcone_m5_fgas_mass.py`, `lightcone_m6_params.py`, `lightcone_m7_des.py`,
  `lightcone_desi_catalog.py`), the P0–P4 geometry-validation scripts, and
  the various `_reduce_*.py`/`_build_*.py`/`ksz_*.py` helpers.

## External raw-data prerequisites (not reproduced by anything below)

Everything in this section is a pre-downloaded, pre-existing external data
product; no script in this repo fetches it. See `docs/tsz_des_data_plan.md`
§1 for the full verified inventory. In brief:

- `DL = /mnt/home/mlee1/ceph/paper3/B/downloads` — ACT DR6+Planck y-map
  release (baseline ILC + 10 CIB-deprojected variants + apodized mask),
  ACT DR5 SZ cluster catalog (Hilton et al. 2021), DESI DR1 SGC LRG
  spectroscopic catalog + randoms, Liu et al. (2025)'s own DR9 photometric
  LRG catalog (`dr9_lrg_pzbins.fits`). `act_ycap_measure.py` (and
  `_r1_closure.py`'s `BEAM_TXT`) read these via their own `DL`/`BEAM_TXT`
  constants — unaffected by `--products_root`/`$BIND_KSZ_PRODUCTS`, which
  only moves the *output/products* root `KS`, but overridable separately
  via `$BIND_KSZ_DOWNLOADS` (Round-2 T3).
- `KS/desact_zenodo/` — the Ried Guachalla et al. (2025) DESI DR1 BGS ×
  ACT kSZ release (Zenodo 19160138): `Fig8_BGS_BRIGHT-20.2_logm{11.00,11.25}.npz`,
  `Fig6_BGS_BRIGHT-20.2_logm11.00.npz`.
- `$BIND_KSZ_DOWNLOADS/planck_ysz/` — the Planck 2015 SZ product tarball
  (`milca_ymaps.fits` + confidence mask; Planck Collaboration XXII 2016),
  used only by `examples/planck_ycap_crosscheck.py` (round-2 T2, the
  independent-map cross-check — see Fig. 10's section above). Extracted
  from the tarball on first run; not otherwise fetched by anything in this
  repo.
- `examples/figures_ksz2/tsz_liu2025_official.npz` — the official Liu et
  al. (2025) release, gitignored (matches `*.npz`) and locally cached; no
  producing script was found in the current `examples/` tree (likely a
  one-off csv→npz conversion from an earlier session). Treat as a fixed
  external input.
- `LC/T3_theta200_anchors.npz` — small cached scalars (`t200_mock_arcmin`,
  `t200_data_arcmin`); read by `lightcone_m2r_ycap_real.py` and
  `_r7_fidelity_closure.py`. No producing script for this npz was ever
  committed — it was originally computed inline during T3 via
  `theta200_arcmin()` in `act_ycap_measure.py` at the mock/data anchor
  mass+redshift (the θ₂₀₀ anchors described in §2.2/§2.4 of the notebook)
  and then cached here. **Recovery (round-3 gap closure):** the value is
  not actually lost even without that inline computation —
  `lightcone_m2r_ycap_real.py` reads this npz back in and passes both
  scalars straight through, byte-for-byte, into
  `act_ycap_lrg_real.npz`'s `theta200_mock_arcmin`/`theta200_data_arcmin`
  keys (verified equal to 16 significant figures). `examples/_recover_t3_anchors.py`
  reverses that rename to reconstruct the cache from
  `act_ycap_lrg_real.npz` alone: `python examples/_recover_t3_anchors.py
  [--out PATH]` (defaults to overwriting `LC/T3_theta200_anchors.npz` in
  place — pass `--out` to write elsewhere first if you want to diff before
  clobbering).

## Notebook rebuild (the acceptance gate for any change)

```bash
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
python examples/_build_p4c_paper_nb.py && \
    jupyter nbconvert --to notebook --execute --inplace examples/paper_p4c_tsz_ksz.ipynb
```

Kernel `bind_env` is declared in the notebook metadata — do **not** pass a
`--ExecutePreprocessor.kernel_name` override; the generic `python3` kernel
lacks `scipy`. Verify zero error outputs after execution.

## Figure-by-figure

### Fig. 1 — the y-CAP measurement (§2.2)
- **Input product:** `LC/act_ycap_lrg_real.npz` (keys `xb`, `mean_xb_cib17`,
  `err_jk_xb_cib17`, `sig_cib_xb`, `valid_cols`, `theta_rap_arcmin`,
  `mean_rap`, `random_null`, `rotated_null`, `theta200_data_arcmin`, …).
- **Engine chain:**
  1. `examples/act_ycap_measure.py` — the T1/T2/T2f measurement engine.
     Produces the per-catalog `T2_*.npz` (native 0.5′/px) and `T2f_*.npz`
     (fine-res, 0.29296875′/px — matches BIND's pixel) shards consumed by
     step 2. Exact battery run in `run_tsz_des_overnight.sh` (T2, native
     res) and `examples/_t2f_tasks.disbatch` (T2f, fine res, via disBatch:
     `module load disBatch && disBatch examples/_t2f_tasks.disbatch`).
     Representative single calls:
     ```bash
     python examples/act_ycap_measure.py --mode lrg --zmin 0.4 --zmax 0.6 \
         --grid both --map_variant baseline --out T2_lrg_z0406_baseline.npz
     python examples/act_ycap_measure.py --mode lrg --zmin 0.4 --zmax 0.6 \
         --grid both --map_variant cib1.7 --res_arcmin 0.29296875 \
         --out T2f_lrg_z0406_cib1.7.npz
     python examples/act_ycap_measure.py --mode random --zmin 0.4 --zmax 0.6 \
         --grid rap --n_boot 100 --out T2_random_null_z0406.npz
     python examples/act_ycap_measure.py --mode rotated --zmin 0.4 --zmax 0.6 \
         --grid both --out T2_rotated_null_z0406.npz
     ```
  2. `examples/lightcone_m2r_ycap_real.py` — combines the T2/T2f shards
     (baseline + cib1.7 + the 11-variant CIB battery + nulls), applies the
     R1 resampling correction and R5 CIB covariance if present, and writes
     `act_ycap_lrg_real.npz` + `tsz_consistent_nodes.npz` + `figs/M2R_ycap_real.png`
     + `verdicts/T3.json`. Plain invocation, no args: `python
     examples/lightcone_m2r_ycap_real.py`.

### Fig. 2 — pipeline validation (§2.3)
- **Input products:** `LC/T1_dr5_snr5.npz`, `verdicts/T1.json` (metrics
  `yc_tercile_amps_3.5arcmin`, `stack_snr_jk_at_3.5arcmin`),
  `examples/figures_ksz2/tsz_liu2025_official.npz` (external, see above),
  `LC/R5_liu_pz{1..4}_cib1.7.npz`, `verdicts/R5.json` key `r5c_liu_reproduction`.
- **Engine chain:**
  1. `examples/act_ycap_measure.py --mode dr5` (ACT DR5 cluster stack —
     pipeline first-light test) and `--mode liu` / `--mode rotated_liu`
     (Liu et al. 2025's own DR9 photometric-LRG sample, per `pz_bin`
     1–4). No committed run script for these two modes was found (the T1/R5
     battery appears to have been run ad hoc); reconstructed commands per
     the module's own `--mode` semantics:
     ```bash
     python examples/act_ycap_measure.py --mode dr5 --grid rap \
         --snr_min 5.0 --out T1_dr5_snr5.npz
     python examples/act_ycap_measure.py --mode liu --pz_bin 1 \
         --map_variant cib1.7 --out R5_liu_pz1_cib1.7.npz   # repeat pz_bin 2,3,4
     ```
  2. `examples/_r5c_liu_figure.py` — builds the Liu+2025 reproduction
     panel and appends the `r5c_liu_reproduction` metrics into
     `verdicts/R5.json`. Plain invocation: `python examples/_r5c_liu_figure.py`.
  3. No script in the current `examples/` tree writes `verdicts/T1.json`
     (the DR5 tercile-amplitude/SNR metrics `t1m` used in the Fig. 2(a)
     title). It likely predates the R0–R8 hardening refactor. **Recovery
     (round-3 gap closure):** `examples/_recover_t1_verdict.py`
     regenerates the whole `metrics` block purely from `T1_dr5_snr5.npz` +
     `T1_random_null.npz` — `python examples/_recover_t1_verdict.py [--out
     PATH]` (defaults to overwriting `LC/verdicts/T1.json` in place; pass
     `--out` to write elsewhere first if you want to diff before
     clobbering) — and by default also prints a field-by-field diff against
     the live `verdicts/T1.json` (`--no-compare` to skip). Verified: `stack_snr_jk_per_theta`/
     `stack_snr_jk_at_3.5arcmin` (mean/err_jk), `null_chi2_per_dof`
     (`mean @ inv(cov_jk) @ mean / n_theta` on the random-null stack),
     `null_n`, and `boot_vs_jk_err_ratio` all reproduce **exactly**;
     `yc_tercile_amps_3.5arcmin` (fixed_y_c terciles × the 3.5′ CAP column,
     masked by `per_obj_ok`) reproduces to **<0.09% relative** — close but
     not bit-exact, since the original ad hoc script's precise
     tie-breaking/masking convention at the tercile boundary isn't
     otherwise recoverable from the stored products; `yc_scaling_monotonic`
     re-derives trivially from the amps. `batch_vs_pixell_path_max_reldiff`
     is the one metric **not** reproduced (it needs the separate
     `T1_dr5_snr5_pixellpath.npz`, outside this recovery's two declared
     inputs) — the script reads it straight through from the existing
     verdict instead, documented in its own docstring.

### Fig. 3 — error-budget decomposition (§2.5)
- **Input products:** `LC/R5_cib_cov.npz`, `LC/R3_mass_template.npz`,
  `LC/R1_resample_correction.npz`, `LC/R2_hod_model_curves.npz`,
  `verdicts/R7.json` (key `fit_columns_xb_0.46_1.25`), `verdicts/R4.json`.
- **Engine chain** (each phase script is a plain `python examples/_rN_*.py`
  run, no CLI args, in this dependency order):
  1. `examples/_r1_closure.py` → `R1_resample_correction.npz` (R1a/R1b
     resampling-operator closure test).
  2. `examples/lightcone_hod_stack.py --merge` (after the 254-task HOD
     shard sweep via disBatch: `module load disBatch && disBatch
     examples/_hod_tasks.disbatch`, driven end-to-end by
     `run_hod_disbatch.sh`) → `R2_hod_model_curves.npz` (per-node HOD
     satellite-population model curves at f_eff ∈ {0.04, 0.08, 0.12}).
  3. `examples/_r3_mass_anchor.py` → `R3_mass_template.npz` (±0.1 dex
     mass-anchor template).
  4. `examples/_r5_cib_systematics.py` → `R5_cib_cov.npz` +
     `verdicts/R5.json` (correlated CIB covariance from the 11-variant
     battery).
  5. `examples/_r7_fidelity_closure.py` → `verdicts/R7.json` (painting-
     fidelity closure vs. the TNG300-hydro truth lightcone).
  6. `examples/_r4_twohalo.py` → `verdicts/R4.json` (analytic 2-halo /
     unpainted-gas floor).

### Fig. 4 — kSZ confrontation, bgs110 (§3.1)
- **Input products:** `LC/taucap_lightcone.npz`, `LC/capmat_lightcone.npz`
  (per-node/per-realization τ-CAP and matter-CAP shards — heavier map-level
  products from the earlier P5/P6a campaign, `docs/ksz_lightcone_map_plan.md`;
  out of scope for a from-scratch P4 rerun), `LC/ksz_consistent_nodes_r6.npz`,
  `KS/desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm11.00.npz` (external, Zenodo
  19160138).
- **Engine chain:**
  1. `examples/lightcone_m1_fgas_desiact.py` — the original M1 map-level
     f̃_gas(θ) vs. Ried Guachalla et al. (2025) comparison; reads
     `taucap_lightcone.npz`/`capmat_lightcone.npz`, writes the pre-fix
     `ksz_consistent_nodes.npz`. Plain invocation: `python
     examples/lightcone_m1_fgas_desiact.py`.
  2. `examples/_r6_ksz_audit.py` — the R6 covariance-unit fix (the
     released `cov_ksz` is byte-identical to the raw T^CAP amplitude
     covariance, not the f̃_gas-ratio covariance it sits next to); rebuilds
     the consistency cut and writes `ksz_consistent_nodes_r6.npz` +
     `verdicts/R6.json`. Plain invocation: `python examples/_r6_ksz_audit.py`.

### Fig. 5 — the tSZ money plot (§3.2)
- **Input products:** `verdicts/T3.json` (`metrics`), `LC/R2_hod_model_curves.npz`
  + `verdicts/R7.json` (model-side, same `node_mean`/`fid_y` construction as
  Fig. 3), `LC/latent_constraints.npz` (key `chi2_tsz`), the Fig. 1 data
  arrays (`xb`, `m17`, etc.).
- **Engine chain:** `examples/lightcone_m2r_ycap_real.py` (as in Fig. 1;
  writes `verdicts/T3.json` and `tsz_consistent_nodes.npz`), then
  `examples/lightcone_latent_corner.py` (Phase L — builds the 2-D gas-latent
  plane and the importance-weighted `latent_constraints.npz` + `verdicts/L.json`
  from the kSZ χ² in `ksz_consistent_nodes_r6.npz` and the tSZ χ² in
  `tsz_consistent_nodes.npz`; plain invocation `python
  examples/lightcone_latent_corner.py`, run after the HOD merge and T3 in
  `run_hod_disbatch.sh`'s tail).

### Fig. 6 — cross-probe ranking coherence (§3.3)
- **Input product:** `LC/latent_constraints.npz` (keys `chi2_ksz`,
  `chi2_tsz`, `node_ids`).
- **Engine chain:** same as Fig. 5 — `examples/_r6_ksz_audit.py` (kSZ side)
  → `examples/lightcone_m2r_ycap_real.py` (tSZ side) →
  `examples/lightcone_latent_corner.py` (Phase L, combines both into one
  npz).

### Figs. 7, 8 — gas-plane posterior and parameter forest (§3.4)
- **Input products:** `LC/r8_posterior.npz` (chains `chain_ksz`,
  `chain_tsz`, `chain_joint`, gas-plane projections `gas_fin_*`/`gas_fout_*`,
  `names`, `prior_q`), `LC/latent_constraints.npz` (key `w_joint`, for the
  Fig. 7 node-weight scatter overlay), `verdicts/R8.json`
  (`gas_plane`, `look_elsewhere_null`, `r4_two_halo_a2h_nuisance`).
- **Engine:** `examples/_r8_gp_mcmc.py` (Phase R8 — GP emulators over the
  253-node design feeding a 33-parameter `emcee` MCMC, 4 legs: kSZ/tSZ/joint/
  joint_noA2h). Staged interface (`--stage {all,gp,chain,process,finalize}`);
  the committed production path is two SLURM scripts:
  ```bash
  sbatch run_r8_chains.sh      # fits GPs if missing, runs all 4 chains concurrently
  sbatch run_r8_finalize.sh    # post-processes the 4 chains -> r8_posterior.npz, verdicts/R8.json, figures
  ```
  Both are resumable/idempotent (emcee `HDFBackend` checkpoints under
  `LC/r8_state/`). A single-process rerun is also possible via `--stage all`
  but is far slower (see the scripts' own comments re: wall-clock budget).

### Fig. 9 — consistency counts by treatment (§5, the honesty figure)
- **Input products:** verdict-only — `verdicts/R5.json` (key
  `counts_by_treatment`, from `examples/_r5_cib_systematics.py`) and
  `verdicts/T3.json` (key `metrics.counts`, from
  `examples/lightcone_m2r_ycap_real.py`). No additional npz.
- **Engine chain:** `examples/_r5_cib_systematics.py` (writes the
  diagonal-vs-correlated-vs-dust-template count comparison into
  `counts_by_treatment`) and `examples/lightcone_m2r_ycap_real.py` (writes
  the final-budget `threshold_scan`/counts into `T3.json`). Both plain
  invocations, no args.

### Fig. 10 — weak-lensing suppression implication (§4.1)
- **Input products:** per-run `Cl_kappa.npz` from
  `$BIND_SB35_RUNS/runs/run_NNNN/` (one per SB35 node), the paired DMO
  trace `bind_science/runs/dmo/run_0000/Cl_kappa.npz` and truth-hydro trace
  `bind_science/runs/truth/run_0000/Cl_kappa.npz` (both under
  `$BIND_SCIENCE_RUNS/..`), `LC/latent_constraints.npz` (keys `node_ids`,
  `w_joint`, `f_in`, `chi2_tsz`), `LC/ksz_consistent_nodes_r6.npz` (key
  `node_ids_bgs110`), the cache `LC/Sell_zs1_253.npz` (keys `ell`, `S`,
  `node_ids`), and `LC/external_suppression_curves.npz`.
- **Engine chain:**
  1. The Fig 10 notebook cell itself assembles `Sell_zs1_253.npz` on first
     run (and just loads it thereafter): for each of the 253 SB35 nodes it
     reads that node's own `Cl_kappa.npz` (source-redshift index 1 = z_s=1),
     divides by the shared DMO trace at the same z_s, and caches the
     resulting 253×n_ell suppression-curve matrix. This logic is
     in-notebook only (`examples/_build_p4c_paper_nb.py`, "Fig 10" cell) —
     it is not a separate `examples/*.py` script, but the recipe is adapted
     from `examples/lightcone_transfer.py` on branch `analysis/wl-cosmo-bias`.
     No CLI invocation; rerun via the notebook rebuild (see Quick start).
  2. `examples/lightcone_external_suppression.py` (round-2 track T1,
     `docs/paper_improvement_plan.md`) — produces
     `LC/external_suppression_curves.npz`, the **non-TNG literature**
     baryon-suppression curves overplotted on the same axis: BCM
     (Schneider & Teyssier 2015, via `pyccl.BaryonsSchneider15`), van
     Daalen et al. (2019/2020) evaluated at our own joint-posterior gas
     fraction (`pyccl.BaryonsvanDaalen19`), Amon & Efstathiou (2022)
     $A_{\rm mod}=0.82$ (hand-built from `P_mod = P_lin +
     A_{\rm mod}(P_{\rm nl}-P_{\rm lin})$), and HMcode-2020 at the BAHAMAS
     $T_{\rm AGN}=7.8$ calibration (via `camb`'s `mead2020_feedback`
     variant) — all at the Limber-projected $z_s=1$ WL convergence power,
     TNG300 cosmology ($\Omega_m=0.3089$, $\Omega_b=0.0486$, $h=0.6774$,
     $\sigma_8=0.8159$, $n_s=0.9667$), on the same $\ell$ grid as
     `Sell_zs1_253.npz`. A CAMELS SIMBA/Astrid curve was deliberately
     **skipped** — no committed path to a CAMELS power-spectrum product was
     found (see the script's own docstring). Plain invocation, no args:
     `python examples/lightcone_external_suppression.py`. Requires `pyccl`
     and (for the HMcode-2020 curve only) `camb`; the script degrades
     gracefully and just skips a curve if either import is unavailable.

Two more round-2/round-3 validation products live alongside the figure
chain but aren't wired into any single numbered figure — they surface as
printed cross-checks in the notebook text (Planck) or are a standalone
robustness appendix not yet read by the notebook at all (covariance
robustness):

- **Planck MILCA independent y-map cross-check** (round-2 track T2, §2.3 —
  "External validation of the measurement pipeline" — and the abstract).
  **Input products:** `LC/planck_crosscheck.npz` (the printed comparison
  table: `theta_arcmin`, `cap_planck`/`cap_act_smoothed` + errors, `chi2`,
  `dof`, `pte`, `n_gal`), plus its two intermediate stacks
  `LC/planck_crosscheck_planck_side.npz` and
  `LC/planck_crosscheck_act_side.npz`. **Engine:**
  `examples/planck_ycap_crosscheck.py` — CAP-stacks an independent 2015
  Planck MILCA all-sky Compton-y map (NSIDE=2048, ~10′ beam) at the same
  DESI LRGs used in Fig. 1, beam-matches the primary ACT map to Planck's
  resolution (Gaussian smoothing in quadrature, $\sqrt{10^2-1.6^2}=9.87'$)
  reusing `act_ycap_measure.py`'s own CAP filter unmodified, and computes a
  two-map $\chi^2$/PTE over the large-aperture ($\theta\ge4'$) bins where
  Planck's beam still resolves the signal (result: $\chi^2=6.5/5$,
  PTE≈0.26). Staged CLI: `python examples/planck_ycap_crosscheck.py [--step
  {extract,measure,compare}]` (no args runs all three stages); requires
  `healpy` and the Planck 2015 SZ product tarball under
  `$BIND_KSZ_DOWNLOADS/planck_ysz/` (extracted on first run). See the
  script's own module docstring for the full method (independent-systematics
  argument, mask gating, jackknife covariance) and the R2.00-mask-bug
  workaround it documents.
- **Per-block covariance robustness** (round-3 polish, `LC/covariance_robustness.json`).
  Two read-only sensitivity checks on the two external-covariance legs
  flagged in §2.5's per-block covariance statement: (1) a Hartlap-factor
  *scan* over hypothetical sample counts $n\in\{30,50,100,200,500\}$ applied
  to the kSZ leg's external (Ried Guachalla+25) covariance block, since its
  true $n$ was never published (R6 flags this as unquantifiable, not
  applied) — reports how much the consistent-node set/ranking move; (2) an
  exact rebuild of T3's tSZ $\chi^2$ for the TNG fiducial and best-fit node
  with the jackknife data covariance swapped for the bootstrap one
  (`cov_boot`, $n=400$, Hartlap-debiased), to check the verdict isn't an
  artifact of the jackknife estimator. **Engine:**
  `examples/_r3a_covariance_robustness.py` (imports `_r6_ksz_audit.py`'s
  own helpers to reproduce its chi2 chain byte-for-byte before perturbing
  it — read-only, writes no other product). Plain invocation, no args:
  `python examples/_r3a_covariance_robustness.py`. **Not currently read by
  the paper notebook or its builder** — a standalone robustness appendix,
  not one of the 10 numbered figures' inputs.

## Consolidated rebuild order

For a full from-scratch products rebuild (assuming the raw external data in
`DL` and `KS/desact_zenodo/` are already in place):

```
act_ycap_measure.py  (T1 dr5/liu modes, T2 + T2f LRG/null/CIB battery)
  -> _t2_fig_verdict.py                         (T2.json)
  -> _r5c_liu_figure.py                          (R5 Liu reproduction panel)
  -> lightcone_m2r_ycap_real.py                  (T3: act_ycap_lrg_real.npz, tsz_consistent_nodes.npz)
  -> _r1_closure.py                              (R1_resample_correction.npz)
  -> lightcone_hod_stack.py (+ _hod_tasks.disbatch)  (R2_hod_model_curves.npz)
  -> _r3_mass_anchor.py                          (R3_mass_template.npz)
  -> _r5_cib_systematics.py                      (R5_cib_cov.npz, verdicts/R5.json)
  -> _r7_fidelity_closure.py                     (verdicts/R7.json)
  -> _r4_twohalo.py                              (verdicts/R4.json)
  -> lightcone_m1_fgas_desiact.py                (ksz_consistent_nodes.npz)
  -> _r6_ksz_audit.py                            (ksz_consistent_nodes_r6.npz, verdicts/R6.json)
  -> lightcone_m2r_ycap_real.py  (rerun, now with R1-R7 products present)
  -> lightcone_latent_corner.py                  (latent_constraints.npz, verdicts/L.json)
  -> _r8_gp_mcmc.py  (run_r8_chains.sh + run_r8_finalize.sh)  (r8_posterior.npz, verdicts/R8.json)
  -> lightcone_capstone_x.py                     (verdicts/X.json)

(independent of the chain above — no shared inputs besides the SB35 run tree / LC root;
 run any time before the notebook build; Sell_zs1_253.npz is instead built in-notebook, see Fig. 10 above)
  -> lightcone_external_suppression.py           (external_suppression_curves.npz)     [Fig. 10]
  -> planck_ycap_crosscheck.py                   (planck_crosscheck*.npz)              [§2.3 cross-check]
  -> _r3a_covariance_robustness.py               (covariance_robustness.json)          [robustness appendix, not read by the notebook]

-> examples/_build_p4c_paper_nb.py && jupyter nbconvert --to notebook --execute --inplace examples/paper_p4c_tsz_ksz.ipynb
```

`lightcone_m2r_ycap_real.py` is deliberately listed twice: the corrections
(R1/R2/R5/R7) it looks for are optional (`if r1_path.exists(): ... else:
print("WARNING: ... uncorrected")`), so it degrades gracefully on a first
pass and should be rerun once those products exist to pick up the full
error budget used by the notebook's Figs. 3 and 5.
