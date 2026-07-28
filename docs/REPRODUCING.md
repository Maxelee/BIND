# Reproducing `examples/paper_p4c_tsz_ksz.ipynb`

Figure-by-figure map from the P4c paper notebook back to the campaign
products and the engine scripts that produced them. The notebook itself
**never re-measures anything** — it only reads merged `.npz` products and
verdict `.json` files under the products root and re-plots/re-prints from
them (see `examples/_build_p4c_paper_nb.py`, which is the sole editor of
the notebook — do not hand-edit the `.ipynb`).

## `BIND_KSZ_PRODUCTS` — the products-root convention

All P4c products live under one root, referred to as `KS` throughout the
campaign docs and code:

```
KS = <products_root>                       # default: /mnt/home/mlee1/ceph/bind_science/ksz_confront
LC = KS/lightcone                          # almost everything below is under here
VD = LC/verdicts                           # verdict JSONs (single source of truth for quoted numbers)
```

As of this task (P4, `docs/paper_improvement_plan.md`), the two engines an
outside reproducer would actually run — `examples/act_ycap_measure.py` and
`examples/lightcone_m2r_ycap_real.py` — accept an explicit
`--products_root` flag. Resolution order (identical in both scripts):

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

**Caveat:** only these two engines were converted under P4. The notebook
builder itself and the other ~15 campaign scripts referenced below
(`_r1_closure.py`, `_r3_mass_anchor.py`, `_r5_cib_systematics.py`,
`_r6_ksz_audit.py`, `_r7_fidelity_closure.py`, `_r8_gp_mcmc.py`,
`lightcone_hod_stack.py`, `lightcone_latent_corner.py`, `lightcone_m1_fgas_desiact.py`,
etc.) still hardcode `KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")`
at module level. A full from-scratch rebuild under a different root would
need those hardcoded paths patched too — out of scope for this task.

## External raw-data prerequisites (not reproduced by anything below)

Everything in this section is a pre-downloaded, pre-existing external data
product; no script in this repo fetches it. See `docs/tsz_des_data_plan.md`
§1 for the full verified inventory. In brief:

- `DL = /mnt/home/mlee1/ceph/paper3/B/downloads` — ACT DR6+Planck y-map
  release (baseline ILC + 10 CIB-deprojected variants + apodized mask),
  ACT DR5 SZ cluster catalog (Hilton et al. 2021), DESI DR1 SGC LRG
  spectroscopic catalog + randoms, Liu et al. (2025)'s own DR9 photometric
  LRG catalog (`dr9_lrg_pzbins.fits`). `act_ycap_measure.py` reads these
  directly via its own `DL` constant (unaffected by `--products_root`,
  which only moves the *output/products* root `KS`).
- `KS/desact_zenodo/` — the Ried Guachalla et al. (2025) DESI DR1 BGS ×
  ACT kSZ release (Zenodo 19160138): `Fig8_BGS_BRIGHT-20.2_logm{11.00,11.25}.npz`,
  `Fig6_BGS_BRIGHT-20.2_logm11.00.npz`.
- `examples/figures_ksz2/tsz_liu2025_official.npz` — the official Liu et
  al. (2025) release, gitignored (matches `*.npz`) and locally cached; no
  producing script was found in the current `examples/` tree (likely a
  one-off csv→npz conversion from an earlier session). Treat as a fixed
  external input.
- `LC/T3_theta200_anchors.npz` — small cached scalars
  (`theta200_mock_arcmin`, `theta200_data_arcmin`); read by
  `lightcone_m2r_ycap_real.py` and `_r7_fidelity_closure.py` but no
  producing script was found either — flagged as a reproducibility gap
  (the values are the mock/data θ₂₀₀ anchors described in §2.2/§2.4 of the
  notebook and should be cheap to recompute from `theta200_arcmin()` in
  `act_ycap_measure.py` at the two anchor masses/redshifts if the cache is
  lost).

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
  3. **Gap:** no script in the current `examples/` tree writes
     `verdicts/T1.json` (the DR5 tercile-amplitude/SNR metrics `t1m` used
     in the Fig. 2(a) title). It likely predates the R0–R8 hardening
     refactor. If lost, those two numbers would need to be recomputed
     directly from `T1_dr5_snr5.npz` (S/N at 3.5′, and the CAP amplitude
     terciled by the DR5 catalog's own `fixed_y_c` column, which
     `act_ycap_measure.py --mode dr5` already stores in the npz).

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
-> examples/_build_p4c_paper_nb.py && jupyter nbconvert --to notebook --execute --inplace examples/paper_p4c_tsz_ksz.ipynb
```

`lightcone_m2r_ycap_real.py` is deliberately listed twice: the corrections
(R1/R2/R5/R7) it looks for are optional (`if r1_path.exists(): ... else:
print("WARNING: ... uncorrected")`), so it degrades gracefully on a first
pass and should be rerun once those products exist to pick up the full
error budget used by the notebook's Figs. 3 and 5.
