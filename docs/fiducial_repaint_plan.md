# Fiducial lightcone repaint at the correct TNG300 cosmology — runbook

**Status:** designed, reviewed, hardened; **not run**. Every Slurm job here is submitted by the author.
**Date:** 2026-08-13 (rev. 2, after three adversarial reviews — see §12 TRIAGE LOG)
**Scripts:** `repaint/` · **Gate:** `repaint/verify_repaint.py`
**New tree:** `/mnt/home/mlee1/ceph/bind_lightcone_tng_fixed` (`$REPAINT_ROOT`)
**Released trees:** read-only inputs, never modified.

---

## 1. Decision record: what was wrong, and why

### 1.1 The bug

Every snapshot of the released fiducial lightcone
(`/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_*/stage1/{params,fiducial_params}.npy`)
carries the **CAMELS SB35 fiducial cosmology** instead of TNG300's. The 30 astrophysical
entries were always correct; exactly five entries are wrong:

| idx | parameter | released (CAMELS) | correct (TNG300) | ratio | Δ in normalised model-input units |
|----:|-----------|------------------:|-----------------:|------:|-----------------------------------:|
| 0 | `Omega0` | 0.3000 | 0.3089 | 1.02967 | +0.02225 |
| 1 | `sigma8` | 0.8000 | 0.8159 | 1.01987 | **+0.03975** (largest) |
| 6 | `OmegaBaryon` | 0.0490 | 0.0486 | 0.99184 | −0.01000 |
| 7 | `HubbleParam` | 0.6711 | 0.6774 | 1.00939 | +0.01575 |
| 8 | `n_s` | 0.9624 | 0.9667 | 1.00447 | +0.01075 |

`Ob/Om` was **1.038141×** too high (0.163333 vs 0.157332). The model reads the baryon
budget essentially linearly, so plane power went as the square: **1.038141² = +7.77%
predicted vs +7.69% measured** on the gas/tau planes.

**It was `Ob/Om`, not `Om`, that did the damage.** That single sentence is why the permanent
guard checks three header entries and not one (§2).

### 1.2 Root cause, end to end (audit A, verified)

1. `run_lightcone_project.sh` wrote `bind.fiducial_params()` — the **CAMELS SB35** fiducial,
   read from the `FiducialVal` column of `src/bind/assets/SB35_param_minmax.csv` — into
   `$STAGE1_DIR/fiducial_params.npy` and passed it as `--params` to stage 1.
2. Stage 1 re-saved that vector as `stage1/params.npy` (`paint_stages.py:266`) and named it
   in the manifest (`:337`).
3. `run_lightcone_generate.sh:66-72` never passed `--params`, so `generate_from_stage1` fell
   back to the stage-1 file (`paint_stages.py:460-462`) and the sampler was conditioned on it.
4. `run_sobol_generate.sh:45` **does** pass `--params "$RUN_DIR/params.npy"` — which is why
   every `bind_science` / `bind_sb35` run built off the *same* poisoned stage-1 root is
   nevertheless **correct**.

The five cosmology params genuinely reach the network: the deployed checkpoint
(`weights/fm_redshift_thermo`, `n_params=35`) drops no index, and `OmegaBaryon`/`Omega0`
have the 1st/3rd largest first-layer column norms of all 35 inputs.

### 1.3 What is *not* affected

* **Stage 1 (DMO conditioning)** — projection, cutouts and the halo catalogue never use the
  params vector. Cosmology-independent, reusable byte-for-byte. *(This is also why there is
  nothing to gain by re-running `run_lightcone_project.sh`, and why that script now refuses
  to overwrite an existing stage 1 — see §2.)*
* **Lensing geometry** — `paint_lensplane` reads `Omega_m = 0.3089` from
  `stage1_manifest.json`, which came from the TNG300-Dark snapshot *header*, not from
  `params.npy`. Distances and the lensing kernel were always right.
* **Everything outside `bind_lightcone_tng`** — verified correct in
  `bind_science/runs/{fiducial,twobound}/…` and `bind_sb35/runs/…`.

### 1.4 The evidence that fixes the expected size of the correction

Three `twobound` runs (`run_0018`, `run_0049`, `run_0053`) are fiducial-astro paints at the
**correct** cosmology on the same halos, same checkpoint, same seed ladder. Measured against
the released fiducial:

| quantity | released → correct | note |
|---|---|---|
| per-halo `m_gas,500c` | **+4.97%** high (20.9 replica-σ) | slightly super-linear vs the naive +3.81% |
| per-halo total M500c | −0.03% | `patch_mass_match` pins the 3-channel total |
| `C_l^ττ` (ℓ>1000) | **+7.9%** high | matches `(Ob/Om)² = 1.0777` |
| `C_l^κκ` (z_s=1, ℓ=1e4–3e4) | **+2.2%** | measured new/old = 1.0222/1.0220/1.0228 |
| `C_l^yy` | ~1–3% | the ~−20% y deficit is *model* bias, not this |
| f_gas,500c vs TNG truth | 1.0930 → **1.0447** | halves; ~exact at 1e14–1e15 M⊙ |
| τ vs TNG truth (ℓ=300–1000) | +8.1% → **+0.4%** | the headline physics win |

> **Correction from review.** An earlier revision quoted the small-scale κ change as
> "−1.9% (ℓ>1e4)" in this table and "≈1.029" in §6 — two different numbers for the same
> quantity, neither matching the data. The measured value is **+2.2%** (see §6 for the full
> per-ℓ table, remeasured directly from the trio's `Cl_kappa.npz`). Use 1.022 everywhere.

---

## 2. The permanent code fix (already applied — read this before touching anything)

| file | change |
|---|---|
| `src/bind/params.py` | new `TNG300_COSMOLOGY` dict + `tng300_params()` (fiducial astro on the TNG300 cosmology); `fiducial_params()` unchanged, docstring + module warning now say it is the **CAMELS** fiducial |
| `src/bind/__init__.py` | exports `tng300_params`, `TNG300_COSMOLOGY` |
| `src/bind/inference/paint_stages.py` | `_check_cosmology()` compares **all three** header-checkable entries — `Omega0` (idx 0), `OmegaBaryon` (idx 6), `HubbleParam` (idx 7) — at `COSMOLOGY_RTOL=1e-3`, and reports `Ω_b/Ω_m` in the message. Stage 1 **RAISES** (escape hatch `allow_cosmology_mismatch=True`); `generate_from_stage1`/`generate_halos` **warn** (parameter sweeps legitimately paint off-substrate cosmologies), and there only on `Omega0`, because the stage-1 manifest records `Omega_m` alone |
| `src/bind/cli/paint_project.py` | `--allow_cosmology_mismatch` |
| `run_lightcone_project.sh`, `run_paint_tng_project.sh` | write `bind.tng300_params()` to `stage1/tng300_params.npy`; **plus a no-overwrite guard**: refuse when the output root resolves inside any released/campaign tree, and refuse to overwrite an existing `stage1_manifest.json` without `FORCE=1` |

**Why the three-entry guard matters.** TNG300-Dark's `snapdir_096` header carries
`Omega0=0.3089, OmegaBaryon=0.0486, HubbleParam=0.6774` (verified). An `Omega0`-only check
passes a vector with the right `Ω_m` and CAMELS' `Ω_b=0.049` — which reproduces this exact
bug. Verified by unit test: the old check said PASS, the new one raises with
`Ω_b/Ω_m would be 0.158627 against the substrate's 0.157332 (ratio 1.008230)`.

**Why the project scripts needed a guard.** `run_lightcone_project.sh` still defaults
`OUTPUT_ROOT` to the released tree and writes `stage1_slab*.npz` **in place**. Since this
runbook tells the reader the fix lives in that script, "let me regenerate stage 1 correctly"
is the most likely operator error in the whole campaign — and it corrupts *two* trees at
once, because `repaint/run_repaint_stage1_link.sh` symlinks those exact slab files into the
new tree. `bind.tng300_params()` reproduces `bind_science/runs/fiducial/run_0000/params.npy`
bit-for-bit (verified); the stage-1 gate would have fired on **20/20** snapshots.

**Not applied — author decision (see §8):** seeding the sampler.

---

## 3. What gets rebuilt, and what is reused

| product | released size | action |
|---|---:|---|
| `snap_*/stage1/stage1_slab*.npz` (DMO cutouts + halos) | 14.5 GiB | **REUSE** — symlinked into the new tree |
| `lightcone_transforms.json` | tiny | **REUSE** — symlinked (same sky geometry ⇒ paired) |
| `snap_*/stage1/params.npy` | 408 B ×20 | **REPLACE** (corrected, real files in the new tree) |
| `snap_*/composite_slab*.npz` | 22–24 GiB | **REGENERATE** (GPU) — **now larger, see below** |
| `lensplanes/` (lenspot + yplane + tauplane) | 75.2 GB | **REGENERATE** (CPU) — all three types |
| `rt_output/` raw | 115.3 GB @550 | **REGENERATE** (lux); optional to keep |
| `{kappa,y,tau}_maps.npz` | 31.4 GB @550 | **REGENERATE** |
| `Cl_*`, `peak_*`, `nongaussian_*`, `nu05_stats`, `dm_stats`, `halo_scaling`, `paired_perreal_fid` | 17 MB | **REGENERATE** |
| `lensplanes_4198/`, `mass_lensplanes/`, `haloplane_trace/`, `_haloplane_work/` | ~90 GB | **ORPHANED** — no consumer on `papers` |

**The composites will be bigger than the released ones.** The released
`composite_slab*.npz` contain **no `composite_thermo`** (verified by npz listing: `dmo`,
`composite`, `alpha`, `patch_scales`, `scale_global`, `coverage_pct`, `n_halos`, `slab_idx`,
`n_slabs`, `box_size`, `halo_centers`, `halo_masses`, `halo_r200`, `generated_patches`,
`thermo_patches`, `condition_sums`). The current code writes a `(4, 4198, 4198)` float32
`composite_thermo` into every slab (`pipeline.py:803`, `_save_composite_slab:441`) — 269 MiB
raw, storing at roughly **+45 to +135 MiB per slab** (bracketed by the measured in-file
compression ratios: `composite` 2.74×, `alpha` 30×; `composite_thermo` is alpha-masked with
no DMO background, so it should sit near the good end). Released slabs are ~0.36–0.41 GiB
each. **Expect ~26–35 GiB of composites, not 22–24.**

> **Do this:** after the Tier-0 `snap_096` run, `ls -la $REPAINT_ROOT/snap_096/composite_slab*.npz`
> and update the table below before committing to the full array. Also check the ceph *quota*
> (the filesystem shows PB of free space; the user quota was never re-measured).

**Storage for the new tree** (composites bracketed at 26–35 GiB)

| option | composites | planes | raw rt | maps | total |
|---|---:|---:|---:|---:|---:|
| N_REAL=50, `KEEP_RAW=0` | 26–35 | 75 | 0 | 2.9 | **~105–115 GB** |
| N_REAL=50, `KEEP_RAW=1` (default) | 26–35 | 75 | 10.5 | 2.9 | **~115–125 GB** |
| N_REAL=550, `KEEP_RAW=0` | 26–35 | 75 | 0 | 31 | **~132–142 GB** |
| N_REAL=550, `KEEP_RAW=1` | 26–35 | 75 | 115 | 31 | **~247–257 GB** |

Deleting `lensplanes/` after the trace saves a further 75 GB but forecloses cheap re-traces.
`KEEP_RAW=1` is also what makes the 50-real sibling (stage 5 task 7) buildable.

---

## 4. Cost per stage

| stage | script | resources | wall | compute | new bytes |
|---|---|---|---|---|---|
| 0 params | `make_corrected_params.py` | login node | seconds | — | 20 KB |
| 0b baseline | `verify_repaint.py --record_baseline` | login node | **~15 s** | — | 200 KB |
| 1 link | `run_repaint_stage1_link.sh` | login node | seconds | — | ~0 (symlinks) |
| 2 generate | `run_repaint_generate.sh` | gpu, array 0-19, 1 GPU + **8 cpu** each | **~1–2.5 h** | 4.0–4.7 GPU-hr | 26–35 GiB |
| 2b recomposite | `run_repaint_recomposite.sh` | cca/icelake, array 0-19, 8 cpu | ≤1 h | ~3 core-hr | rewrites in place |
| 3 planes | `run_repaint_planes.sh` | cca cascadelake\|skylake, array 0-19, exclusive | ≤3 h (cap); ~1–2 min of *work* per task | ~20 node-hr held | ~75 GB |
| 4 lux, N=50 | `run_repaint_lux.sh` | cca, 4 nodes/192 ranks, 48 h wall | **2.4 h** | **~465 core-hr** | 2.9 GB (+10.5 GB raw) |
| 4 lux, N=550 | same, 11 chunks of 50 | 4 nodes/192 ranks, 48 h wall | **~26 h, ONE submission** | **~5100 core-hr** | 31 GB (+115 GB raw) |
| 5 stats | `run_repaint_stats.sh` | cca/icelake, array 0-7, 360 GB | ≤8 h | ~15 core-hr | 20 MB |

**Measured rates, not estimates.** 0.43 s/halo/A100 (33,678 halos over 20 snapshots;
snap_096 = 2,933). One 50-realization lux chunk = **02:25:20** on 4 nodes / 192 ranks
(`sacct -j 2441237`; the lux step alone 02:09:46), collect ~14 s/realization.

**Three scheduling facts that were wrong in rev. 1 and are now corrected in the scripts:**

1. **The `gpu` QOS caps the user at `cpu=40, gres/gpu=4`** (`sacctmgr show qos`). With
   `--cpus-per-task=16` only *two* array tasks could run at once. Now `--cpus-per-task=8`,
   which lands exactly on both caps (4×8=32 CPU, 4 GPU) and roughly halves the wall.
   "~25 min for a 20-way array" was never achievable; expect 1–2.5 h.
2. **`pcn-16-06` is the ONLY a100 node and it was DRAINED** (`State=IDLE+DRAIN`,
   `Reason="health cuda != 6"`, 2026-08-12). **Check before submitting:**
   ```bash
   sinfo -p gpu -O NodeList,StateLong,Gres,Reason
   ```
   If still drained, use the idle V100S nodes with fp32 (see §5). This is a hard blocker on
   the campaign starting, not a nicety.
3. **A 5 h wall was wrong for stage 4.** The released 550-real trace ran as **one** job
   (`sacct -j 2457493`: `Timelimit=3-00:00:00`, `Elapsed=22:39:55`, 4 nodes) and cca's
   `MaxTime` is 7 days. Stage 4 now asks for **48 h**, so N_REAL=550 completes in a single
   submission. Slurm does **not** auto-requeue on TIMEOUT (only on preemption/node failure),
   so a short wall meant hand-babysitting *and* ~150 core-hr discarded per cycle. The loop
   additionally refuses to start a chunk that cannot fit in the remaining allocation.

**Queue reality at design time.** `squeue -u mlee1` showed **72 live N1000 jobs**, both cca
arrays pending on `QOSMaxNodePerUserLimit`. Stages 3 and 4 will queue behind them. Stage 3
now accepts `cascadelake|skylake` (360 eligible nodes instead of 216), as does stage 4; both
can also be sent to the preempt lane, and stage 4 is chunk-claimed so the preempt and cca
lanes may run **concurrently** and will divide the chunks between them.

---

## 5. The submit chain (author runs these, in order)

```bash
cd /mnt/home/mlee1/BIND
export REPAINT_ROOT=/mnt/home/mlee1/ceph/bind_lightcone_tng_fixed   # or your own fresh tree

# ── PRE-FLIGHT (do not skip) ─────────────────────────────────────────────────
sinfo -p gpu -O NodeList,StateLong,Gres,Reason     # is an a100 available?
squeue -u mlee1 | wc -l                            # how deep is the N1000 queue?
df -h /mnt/home/mlee1/ceph                         # and check your ceph QUOTA

# ── stage 0: corrected vector + released-tree baseline (no Slurm, seconds) ────
python repaint/make_corrected_params.py            # asserts bit-equality, prints the diff
python repaint/verify_repaint.py --record_baseline # 493 released files + 80 content anchors

# ── stage 1: link the DMO conditioning (no Slurm, seconds) ───────────────────
bash repaint/run_repaint_stage1_link.sh
python repaint/verify_repaint.py --skip_power      # ══ GATE 1 ══ must print GATE: PASS

# ── stage 2: GPU paint ───────────────────────────────────────────────────────
# a100 available:
sbatch --array=0 repaint/run_repaint_generate.sh                     # Tier 0: snap_096
# a100 DRAINED — fall back to the idle V100S (NO_AMP=1 is MANDATORY, sm_70 has no bf16):
NO_AMP=1 sbatch --array=0 --constraint=v100s repaint/run_repaint_generate.sh

python repaint/verify_repaint.py                   # ══ GATE 2 ══ physics on snap_096
ls -la "$REPAINT_ROOT"/snap_096/composite_slab*.npz # ← update the §3 storage table NOW

JID_GEN=$(sbatch --parsable --array=1-19 repaint/run_repaint_generate.sh)   # the other 19
#   NB --array=1-19, NOT 0-19: `generate_complete` is evaluated once at job start with no
#   lock held, so a full 0-19 array launched while Tier 0 is still running would repaint
#   snap_096 concurrently into the same composite_slab*.npz files.

# ── stage 2b: circular re-composite (r200_factor = 4.0, matching the release) ─
JID_RC=$(sbatch --parsable --dependency=afterok:$JID_GEN repaint/run_repaint_recomposite.sh)
#   NB afterok on the WHOLE job, not aftercorr: stage 2 was submitted as --array=1-19
#   (Tier 0 covered snap_096 separately), so a per-task aftercorr chain from a 0-19
#   dependent would wait forever on a source task 0 that does not exist. Stage 2b is
#   ≤1 h and idempotent, so the lost pipelining costs nothing.
#   Skip stage 2b entirely if you set GENERATE_R200=4.0 in stage 2. Either way stage 3
#   hard-asserts summary.json r200_factor == 4.0 before writing a single plane.

# ── stage 3: lenspot + yplane + tauplane ─────────────────────────────────────
JID_PL=$(sbatch --parsable --dependency=afterok:$JID_RC repaint/run_repaint_planes.sh)

# ── stage 4: lux trace + collect ─────────────────────────────────────────────
sbatch --dependency=afterok:$JID_PL repaint/run_repaint_lux.sh      # N_REAL=50 (default)
#   Resubmit the SAME command to resume after a preemption; finished chunks are kept.
#   Optional second lane, safe to run at the same time (chunks are claimed atomically):
#     sbatch --partition=preempt --qos=preempt repaint/run_repaint_lux.sh
#   Release parity later — this WORKS now, it is not a no-op:
#     N_REAL=550 sbatch repaint/run_repaint_lux.sh    # traces chunks 1..10, re-collects
#   Do NOT use FORCE=1 to extend: FORCE only re-collects. FORCE_CHUNKS=1 would additionally
#   re-trace chunk 0 for nothing (~465 core-hr).

# ── stage 5: statistics (only once .trace_complete exists) ───────────────────
ls "$REPAINT_ROOT/.trace_complete" && sbatch repaint/run_repaint_stats.sh   # array 0-7

# ── final gate ───────────────────────────────────────────────────────────────
python repaint/verify_repaint.py                   # ══ GATE 3 ══ full, all checks
```

**Dependency caveat.** `run_repaint_lux.sh` exits **0** when chunks remain (so a requeue
chain is not marked failed). Do **not** hang stage 5 off it with `afterok` — check
`.trace_complete` first, as above.

**Every stage is idempotent and guarded.** Re-submitting skips finished snapshots/chunks.
All stages refuse to run if any path they write resolves inside a released tree
(`repaint_guard`), require every write directory to live **inside** `$REPAINT_ROOT`
(`repaint_assert_under_root`), refuse any directory containing a symlink into a released
tree (`repaint_assert_no_released_symlinks`), and share a campaign-parameter lock
(`$REPAINT_ROOT/.campaign_params`) so a half-repaint cannot mix two seed ladders or grids.
The realization count is tracked separately in `.n_real` / `.stats_n_real`, because extending
50 → 550 is an intended workflow.

### Verification gate between each stage

| after | run | must show |
|---|---|---|
| stage 1 | `verify_repaint.py --skip_power` | GATE 1: `a-params` 20/20 PASS, `d-untouched` all PASS, stage-1 entries are symlinks |
| stage 2 Tier 0 | `verify_repaint.py` (all 4 slabs, the default) | GATE 2: `b-physics` gas mean ≈0.963, gas band power **0.91–0.93** (see §6), mass mid-k ≈1 |
| stage 2b | (automatic) each task re-reads its own `summary.json` and fails on a mismatch | `r200_factor=4.0 taper_frac=0.15` |
| stage 3 | (automatic) per-task hard gate on `summary.json`, exact plane sizes, no released symlinks | 80 each of lenspot/yplane/tauplane at 671088648 / 134217736 B |
| stage 4 | `cat $REPAINT_ROOT/.n_real` | equals your `N_REAL`; `.trace_complete` exists |
| stage 5 | §6 cross-check table | `C_l^ττ` and `C_l^κκ` ratios inside the measured bands **at ℓ>1000** |
| all | `verify_repaint.py` | GATE 3: every row PASS |

---

## 6. Verification gates

`repaint/verify_repaint.py` is the gate. Hard PASS/FAIL table, non-zero exit on any FAIL.

| check | what it asserts | tolerance |
|---|---|---|
| **(a) params** | all 20 snapshots hold the corrected vector **bit-for-bit** vs `bind_science/runs/fiducial/run_0000/params.npy`, and `params[0]` agrees with the manifest's header `Omega_m` | exact / 0.1% |
| **(b) physics** | gas mean surface density ratio new/old = 1/1.038141 = **0.9633**; gas band power ≈ **0.92** (see below); total-mass **mid-k** band power = 1 (the *taper* check); total-mass low-k band power = 1; total-mass sum = 1 (*provenance only*) | 1.5% / **5%** / 5% / 1% / 0.2% |
| **(c) geometry** | per-snapshot halo counts and byte-identical `halo_centers`/`halo_masses`/`halo_r200` vs the released tree | exact |
| **(d) untouched** | recorded size+mtime inventory of **493** released files unchanged — **including all 241 `lensplanes/` files** — plus a first+last-4 KB content anchor on the 80 `lenspot*.dat`; the released `params.npy` **still** holds the buggy CAMELS vector; the new tree is outside every released tree; stage-1 entries are symlinks, not copies | exact |

**On the gas band-power expectation.** `POWER_RATIO_EXPECTED = 1/1.038141² = 0.92787` is the
*flat-rescale* prediction. The correction is not flat — σ8/n_s/h/Ω_m move too and the gas
*shape* changes — so the true ratio is k-dependent and sits slightly below it. Recompositing
the three trio replicas onto the shared `snap_096/slab00` stage 1 gave **0.9124 / 0.9231 /
0.9134** in this gate's band (k∈[20,524)), with strong k-dependence: 0.915 / 0.911 / 0.856 /
0.778 in k = [20,100) / [100,300) / [300,800) / [800,1500). The tolerance was therefore
widened 3% → **5%**, giving the band [0.881, 0.974]. **Discrimination is unharmed:** an
uncorrected repaint gives exactly **1.000**, still 2.6% above the upper edge. Read the printed
value as a measurement — GATE 2 should land near 0.91–0.93; 1.00 means the fix did not take.

**On the total-mass sum check.** It is a tautology and is now *labelled* as one:
`build_bind_composite` sets `scale_global = dmo.sum()/composite.sum()` and multiplies it in
(`pipeline.py:788-789`), so every composite's total equals its own DMO total **by
construction** — measured on the released `snap_096/slab00`: 1.000002 — and both trees share
the same symlinked DMO. It is kept as a cheap *provenance* assertion (it fires if the new tree
stopped sharing the released DMO, or if `patch_mass_match` was turned off), never as evidence
that the mass channel is unchanged. The **mid-k** mass band power is the check that actually
moves: a square taper shifts it by tens of percent, so it is the guard against a skipped
stage 2b.

### Physics cross-checks after stages 4/5 — **measured**, not predicted

Rev. 1 quoted "`C_l^ττ` ≈ 0.922–0.928, flat in ℓ" and "`C_l^κκ` ≈ 1.029 at ℓ>1e4". Both were
wrong, and a legitimate repaint would have violated them. Remeasured directly from
`bind_science/runs/twobound/run_{0018,0049,0053}/Cl_{tau,kappa}.npz` against
`bind_lightcone_tng/*`:

**`C_l^ττ` new/old**

| ℓ band | r0018 | r0049 | r0053 |
|---|---:|---:|---:|
| 100–300 | 0.9638 | 0.9487 | 0.9493 |
| 300–1000 | 0.9340 | 0.9291 | 0.9314 |
| **1000–3000** | 0.9276 | 0.9258 | 0.9274 |
| **3000–10000** | 0.9254 | 0.9226 | 0.9249 |
| **10000–30000** | 0.9232 | 0.9206 | 0.9207 |

**`C_l^κκ` (z_s = 1) new/old**

| ℓ band | r0018 | r0049 | r0053 |
|---|---:|---:|---:|
| 100–300 | 0.9559 | 0.9560 | 0.9560 |
| 300–1000 | 0.9906 | 0.9909 | 0.9909 |
| **1000–3000** | 0.9924 | 0.9925 | 0.9932 |
| **3000–10000** | 1.0049 | 1.0042 | 1.0042 |
| **10000–30000** | 1.0222 | 1.0220 | 1.0228 |

**Read only ℓ > 1000 as a stop/go criterion.** The low-ℓ excursion (κ 0.956 at ℓ=100–300,
identical to four digits across all three independent replicas) is **not** physics: it is the
50-real-numerator-vs-550-real-denominator sampling artifact — the trio has 50 realizations,
`bind_lightcone_tng` has 550. That is the same defect §10.1 carries, and stage 5 task 6 now
produces the paired denominator that removes it.

**Stop and diagnose if,** at ℓ>1000: `C_l^ττ` is outside 0.918–0.932, or `C_l^κκ`(z_s=1) is
outside 0.988–0.997 (ℓ 1e3–3e3) / 1.000–1.010 (3e3–1e4) / 1.017–1.027 (1e4–3e4). Those bands
are the trio's measured spread ±0.005.

Also expect: `C_l^yy` new/old ≈ 1 within paint noise (0.5–4%); fiducial `S(ℓ)` trough
0.8905 → **0.9047** (absolute values subject to the denominator caveat above).

---

## 7. No-overwrite and rollback policy

* **In code:** `repaint_guard()` (shell) and `assert_writable()` (python) resolve **both**
  the symlink-followed and the `--no-symlinks` form of a path and refuse anything inside
  `bind_lightcone_tng`, `bind_science`, `bind_sb35`, `bind_n1000`, `tng_full_validation`.
  Relative paths are refused outright (they would resolve against whatever cwd the job has,
  and `repaint_preamble` `cd`s partway through). Verified against trailing slashes, `..`
  round-trips through a released tree, the raw `/mnt/sdceph/...` alias, and the empty string.
* **`repaint_assert_under_root()`** additionally requires every write directory to be *inside*
  `$REPAINT_ROOT`. This is what stops an inherited/exported `LENSPLANE_DIR` from redirecting
  the lux stage at the released planes — which would have ray-traced the **old,
  wrong-cosmology** planes into the new tree and passed every downstream gate silently.
* **`repaint_assert_no_released_symlinks()`** refuses any campaign directory containing a
  symlink into a released tree, because `bind.inference.lensplane` writes with a plain
  `open(path,"wb")` (lines 193/214/401) which **follows symlinks and truncates the target**.
  `REUSE_MASS_PLANES` — which planted exactly such links — has been **removed**; do not
  reintroduce it. It saved a few node-minutes (measured plane cost: 25–50 s/task) against a
  53.7 GB irreversible loss.
* **SAFETY INVARIANT (re-verify on any lux rebuild).** With `simulation_format=PreProjected`
  the current lux build only *reads* `LP_output_dir`/`tsz_input_dir`/`tau_input_dir`; all
  writes go to `RT_output_dir`, because `lenspot Lenspot(...)` is commented out at
  `main.cpp:47` (`lenspot::write_phi`, `lenspot.cpp:451`, would otherwise
  `fopen("%s/lenspot%02d.dat", LPoutput_dir, "wb")`). This comment now appears in both
  `run_repaint_planes.sh` and `run_repaint_lux.sh`, mirroring `n1000_body.sh:279-283`.
* **Never** `FORCE=1` against a released path; there is no code path that writes there at all.
* **Rollback = delete `$REPAINT_ROOT`.** The released products are untouched and remain the
  published artefacts until the paper is explicitly re-pointed.

### Switching the paper over

Rev. 1 claimed "the figure code reads the lightcone from a handful of constants" and listed
seven files. **That is materially incomplete.** `papers/01_pipeline/_build_figures_nb.py`
resolves the lightcone through **two** roots:

* `LC = CEPH/"bind_lightcone_tng"` (line 127) — one constant, easy.
* `SCI/"runs/bind/run_0000"` — **20 inline occurrences** (lines 287, 328, 1643, 3340, 3963,
  4759, 4787, 6236, 6573, 6595, 7737, 8493, 8499, …), covering every
  `paired_perreal_fid.npz` band, `Cl_kappa.npz`, `kappa_maps.npz`, `y_maps.npz`,
  `nongaussian_stats.npz`, `nu05_stats.npz` and the peak-count family.

`runs/bind/run_0000` is a **50-realization** sibling (`kappa_maps.npz` is
`(50, 5, 1024, 1024)`; the released lightcone is `(550, …)`) and it also holds `y_maps.npz`,
whose amplitude moves by the full ~3.8%. It is *not* κ-only and *not* sub-percent, so it
belongs on the **Must** list, not the "Should" list.

**Do this, in one reviewable commit after GATE 3:**

1. Introduce a single `FID = Path(os.environ.get("BIND_FID", <new or old path>))` constant and
   route **both** the `LC` sites and all 20 `runs/bind/run_0000` sites through it, keeping the
   old path reachable via the env override.
2. Supply the drop-in replacement for `runs/bind/run_0000`:
   * at `N_REAL=50` the repaint tree **is** that product; point `FID` at `$REPAINT_ROOT`;
   * at `N_REAL=550` use `$REPAINT_ROOT/n50`, which stage 5 task 7 builds by re-collecting
     the first 50 realizations plus all five stats variants and `nu05_stats.npz`.
3. Also update the other lightcone constants: `field_cache.py:38`, `build_atlas_200c.py:27`,
   `build_perhalo_profiles.py:38`, `noisy_grid.py:96`, `shell_straddle.py:55`,
   `paper_latent_measurement.py:33`.

`verify_repaint.py --record_baseline` must be run **before** the first job so the "untouched"
check has something to compare against.

---

## 8. Reproducibility: the sampler is unseeded (author decision)

`src/bind/model.py:454` draws `torch.randn(...)` from the **global, unseeded** RNG, and
nothing in the paint path seeds it. Consequences:

* The repaint is **not** a reproduction of the released maps and **cannot** be — even at
  identical parameters two paints differ by ~30.6% per-patch RMS. Only *ensemble statistics*
  are comparable between old and new. (This is also why the 3 twobound replicas are
  legitimate independent replicas of "the" fiducial, and why the GATE-2 physics check now
  defaults to all four slabs: 666 halos carries ~1% replica scatter against a 1.5% tolerance,
  2933 halos is ~2.1× tighter and costs nothing.)
* Paint noise is far below cosmic variance — per-paint scatter of a 50-real mean is
  0.02–0.15% on `C_l^κκ` and 0.13–0.88% on `C_l^ττ`, versus 0.6–4.8% cosmic-variance s.e. —
  so one replica suffices scientifically.
* The new *released product* would inherit the same irreproducibility.
* The fp32 fallback (`NO_AMP=1` on V100S) is scientifically harmless for the same reason —
  but it is recorded in `snap_*/.repaint_provenance.json` (amp, GPU model, host, job id) so it
  is never invisible.

**Recommended optional pre-repaint change (author decision, ~15 lines):** thread a
`torch.Generator` through the sampler, exactly as `bind.wlemu` already does
(`src/bind/wlemu/model.py:259`; `src/bind/wlemu/sample.py:90-98`):

1. `FlowMatching.sample(..., generator=None)` → pass to the `torch.randn` at `model.py:454`.
2. `Model.generate(..., seed=None)` (`inference/paint.py:327-400`) → build the generator once
   and pass it per batch.
3. `--seed` on `bind.cli.paint_generate` / `generate_halos`, recorded in `summary.json`.

*Risk:* changing the call to `torch.randn(..., generator=g)` **changes the RNG stream**, so
paints made after the change differ from paints made before it — statistically irrelevant but
it means the change must land **before** the repaint if the new product is to be the
reproducible one. Seed deterministically per `(snapshot, slab, batch)` rather than once per
process. If declined, state plainly in the paper that the painted fiducial is one draw from
the model's conditional distribution.

---

## 9. The twobound-trio shortcut: verdict and how it is used

`bind_science/runs/twobound/run_{0018,0049,0053}` are fiducial-astro paints at the **correct**
TNG300 cosmology (params byte-identical to `runs/fiducial/run_0000`), same shared stage-1
cutouts, same checkpoint (`last.ckpt`, unchanged since Jun 7), same halos (33,678, 80/80
slabs), same paste settings (`taper_frac 0.15`, `r200_factor 4.0`, `patch_mass_match`), and
lux-traced with `RT_SEED=1992` on the fiducial's own transforms. Seed pairing is verified on
the maps: `corr(tb_r, dmo_r)` = 0.986 diagonal vs 0.002–0.006 off-diagonal, and
`bind/run_0000` κ[0] is **bit-identical** to the released 550-real κ[0].

**Verdict: use it now, but it does not replace the repaint.**

*Use it for (immediately, at zero compute):*

1. **Pre-registering the gate thresholds** — done; §6 and §1.4 come from it.
2. **Writing the corrected numbers into the text while the repaint runs.**
3. **A paint-noise systematic** — the N=3 spread (0.02–0.15% on `C_l^κκ`) is the honest error
   bar on "which replica is the fiducial".

*It cannot replace the repaint because:* it stores **patches only** (no composites, no DMO
background, no lensplanes ⇒ no hero panels, no plane-level diagnostics); it has 50
realizations; it has **no** per-realization `field_cache` shards (Figs 5/6 need paired
±1σ/√50 bands), no `dm_stats`, no `peak_*_multi/_ngal10`, no `paired_perreal_fid`; and every
downstream consumer points at `bind_lightcone_tng` / `runs/bind/run_0000`.

*Caution:* the trio's cached `nongaussian_stats.npz` disagrees **in sign** with a direct
recomputation from its own maps (cached variance ratio 0.988–0.993 vs recomputed 1.001–1.004).
Do **not** use the cached non-gaussian/MF/PDF npz for trio-vs-fiducial deltas; recompute from
the map cubes. The same stale-cache caveat applies to `bind_science/runs/bind/run_0000`.

*Also note:* an "N=3 mean" is legitimate only for **statistics**, never for maps.

---

## 10. What changes in the paper

### 10.1 Headline

The measured fiducial `S(ℓ)` trough moves **0.8905 → 0.9047** (+0.0142 ≈ 35 replica-σ), while
the family-basis model's prediction at the corrected latents moves only 0.8950 → 0.8987. The
closure residual therefore goes **+0.0045 → −0.0060**: the magnitude is ~unchanged but the
**sign flips**. The fiducial-closure narrative must be re-derived, not merely re-numbered.

*Absolute values here carry the known 550-real-DMO-denominator caveat.* `S(ℓ)` is **not**
stored: `bind.inference.stats` writes `suppression`/`cl_dmo` only when a `kappa_dmo` cube is
passed in-process (`stats.py:103-112`), and the released `Cl_kappa.npz` carries only
`ell`/`cl`/`cl_err`, so downstream code forms `S(ℓ)` by dividing by
`bind_science/runs/dmo/run_0000`, whose cube is `(550, 5, 1024, 1024)`. **Stage 5 task 6 now
writes `$REPAINT_ROOT/Cl_kappa_dmo_n50.npz` from the first 50 (seed-paired) DMO realizations.**
Recompute every absolute `S(ℓ)` against it; the *differences* were always robust.

### 10.2 The τ leg gets much better; the small-scale κ closure gets worse

* τ vs TNG truth at ℓ = 300–1000: **+8.1% → +0.4%**; f_gas,500c bias 9.3% → 4.5%
  (≈exact at 1e14–1e15 M⊙). This is the strongest argument for the repaint.
  **Convention note:** every τ number quoted from this campaign is against a **patches-only**
  truth. See §10.6.
* κ: `|S − S_truth|` grows from 0.0067 → 0.0098 (ℓ 1e3–3e3), 0.0071 → 0.0200 (3e3–1e4) and
  0.0006 → 0.0256 (1e4–3e4). The released tree's excellent small-scale κ closure was partly a
  **cancellation** between the too-high baryon fraction (which softens small-scale power) and
  an intrinsic model over-concentration. This needs a physics narrative before publication; a
  DM-vs-gas channel decomposition of the composite would settle it.
* `C_l^yy` is essentially unchanged — the ~−20% y deficit is **model bias**, independent of
  this bug, and that statement in the paper stands. **But say what it is measured against:**
  `paint_tauplane` bakes in the proper-area factor `a_ℓ` (`_tau_per_gas_pixel`) while
  `paint_yplane` applies none, so BIND y is the *legacy comoving-area* convention while the
  truth lightcone y is *physical* (proper-area), with physical = legacy/a². A BIND-vs-truth y
  comparison that has **not** been rescaled by 1/a² per plane would show a large negative
  offset from convention alone. Confirm the −20% was measured after that correction (as the
  diffuse-gas campaign did: `runs/hydro_full_legacyy` vs the rescaled re-trace) and state so.
  **The repaint neither fixes nor worsens this** — say that explicitly so it is not read as
  cleared.

### 10.3 Latents

All 8 shipped latents move, but modestly in design units (σ_design): `c_gas[14.0–14.3]`
−0.311, `logY_ss[13.4–13.6]` −0.164, `logY[13.0–13.2]` −0.078, `logPe[14.0–14.3]` −0.032,
`logT[13.0–13.2]` +0.066, `c_gas[13.2–13.4]` −0.059, `f_star[13.2–13.4]` +0.021,
`logPe[13.4–13.6]` −0.010. Direction: f_star and logT up, logY/logPe/c_gas/logY_ss down.
**The family-basis model itself is unaffected** — it is trained entirely on the 256-run SB35
Sobol design and the twobound shape families, all at the correct cosmology. Only the
out-of-design fiducial evaluation point `lam_fid` and the measured fiducial curves change.

### 10.4 Figures

*Strong (plotted physics moves):* fig03, fig03a, fig04, fig04b, fig06, fig07, fig11, fig19,
fig20a, fig20d, fig20e, `pfig_s4a_latent_measurement`, `pfig_s4b_model_curves`,
`pfig_s4b_reconstruction`, **fig18_highell_localize** (sits exactly in the ℓ>1e4 band where κ
does move, by the measured +2.2%).
*Weak (κ-only, <0.3% below ℓ=4e3):* fig00_hero, fig08, fig10, fig12, fig15, fig15b, fig23.
*Unaffected:* fig01 (stage-1 DMO only) and everything driven purely by the SB35 Sobol design
and the emulator trained on it (figs 5/5a/9/9b/9c/16).

### 10.5 Caches and products to regenerate afterwards

**Must (produced by the campaign itself, stage 5):**
`Cl_kappa`, `Cl_kappa_y`, `Cl_tau`, `dm_stats`, `peak_counts{,_multi,_ngal10,_nufid}`,
`peak_cross{,…}`, `nongaussian_stats`, `halo_scaling`, `scaling_relations`,
`paired_perreal_fid` (task 4, **pinned at `--n_real 50`** so the fixed-ν σ₀ convention matches
the released `runs/bind/run_0000` cache — 1.5 MB = 50 reals), **`nu05_stats.npz` (task 5)**,
**`Cl_kappa_dmo_n50.npz` (task 6)**, **the `n50/` sibling (task 7, only when N_REAL>50)**.

> `nu05_stats.npz` is the canonical 22-point ν grid (dν=0.5, −2.75..7.75) consumed by
> `family_basis_all.py:115` and `refresh_amplitude_sets_mf22.py:32`. It is **not** written by
> `bind.cli.lightcone_stats` (which writes `nongaussian_stats.npz` on the legacy 29-point
> ν=−3..4 grid) — it comes from `nu_grid.compute`, via
> `papers/01_pipeline/remeasure_twobound_nu05.py`. Without task 5 a re-pointed fiducial would
> silently mix the **old** fiducial's nu05 with new everything else.

**Must (fiducial-derived, outside the campaign):**
`bind_science/halo_atlas/fid_snap{029..096}.npz` · `bind_science/profiles/fid_snap*.npz` and
`perhalo_fid_snap096.npz` · `bind_science/field_cache/{field_stats_fid.npz, shards/*_bind.npz}`
(the τ leg especially) · the family-model `lam_fid` and `figs_preview/family_model_bundle.npz`
fiducial point · **`bind_science/runs/bind/run_0000`** (see §7: 20 inline reads, and it holds
`y_maps.npz`, so it is *not* κ-only).

**Must (convention coupling — easy to miss):**
* every `*_nufid.npz` in `twobound`/`sb35`: the fixed-ν convention normalises by **one σ from
  the fiducial**, and the fiducial changes ⇒ re-run `run_nufid_restats.sh` with
  `FID_DIR=$REPAINT_ROOT` (or `$REPAINT_ROOT/n50`).
* every `twobound/run_*/paired_stats.npz`: `clk_resp` is a response **against the fiducial**
  per-realization cube ⇒ re-run `run_paired_stats.sh` with `FID_DIR=$REPAINT_ROOT` after
  stage 5 task 4 has written the new `paired_perreal_fid.npz`.
* `bind_sb35/nu05_shards/sci_bind.npz` and every per-run fiducial `nu05_stats.npz`.

**Should (κ-only, sub-percent but inside the trusted ℓ range):**
`bind_science/mf_cache/mf_nu8_snap096.npz` (bind side) · `bind_sb35/nu05n_shards/fid*.npz` +
`_kappa_rms_fid.npz`.

**Unaffected:** `bind_science/latent_drivers.npz`, `bind_sb35/emulator_fits/paper1_gp.pt`,
the SB35 Sobol design, the family-basis shape bundle, the DMO arm everywhere.

**N1000 coordination (blocking for that campaign, and sequencing-sensitive):** task 0
(`bind` arm) symlinks the master's mis-conditioned lenspot planes and paints y/τ from the
master's mis-conditioned composites. `bind_n1000/bind/run_0000/{kappa,y,tau}_maps.npz`
(56.4 GB) is already written and `.trace_complete`. It needs a `FORCE=1` redo against the new
master, launched as `FID=$REPAINT_ROOT` (`n1000_body.sh:32` accepts the override) so it
consumes the corrected lightcone.

> **Sequencing rule.** A `FORCE=1` redo does `rm -rf $CLAIM` (`n1000_body.sh:115`). At the time
> of writing `squeue -u mlee1` shows **72 live n1000 jobs** (arrays 2459672, 2459732) saturating
> cca at `QOSMaxNodePerUserLimit`. **Wait for the 319-task campaign to drain before the redo**,
> or it will race the running campaign's claim arbitration. Nothing in `repaint/` performs this
> write — `bind_n1000` stays on the read-only list and `n1000_body.sh` owns that tree.

`bind_n1000/analysis/peaks_progress.npz` should be checked for the same contamination. Also
re-derive the diffuse-gas variants under `tng_full_validation`, built against the old fiducial.

### 10.6 The repaint is pasted-only, by design — state this in the paper

`n1000_body.sh` sets `DIFFUSE=1` by default and calls `bind.cli.paint_diffuse_composite`
(f_b default 0.0486/0.3089, `--y_convention physical`) **before** painting the y/τ planes, and
`docs/diffuse_gas_validation.md` records that the pasted τ carries only 16–37% of the true mean
column while the diffuse add-on recovers it to 1.0–1.2%.

**This repaint deliberately does NOT apply the diffuse correction.** Its job is to replace the
released fiducial one variable at a time: the only thing that changes is the conditioning
cosmology. Adding the diffuse term at the same time would confound the two effects and make
the old/new comparison in §1.4 and §6 meaningless. Consequences to write down:

* Every τ number quoted from this campaign is **against a patches-only truth**, including the
  §10.2 headline "+8.1% → +0.4%".
* The new master is therefore **convention-inconsistent with `bind_n1000`**, which is
  `DIFFUSE=1`. That is intentional and matches the released master's convention.
* The N1000 `bind`-arm `FORCE=1` redo will re-apply `DIFFUSE=1` on top of the new composites,
  so the n1000 arm stays internally consistent.

---

## 11. Decision knobs — decide these before submitting

| # | knob | recommendation | why it matters now |
|---|---|---|---|
| 1 | **`N_REAL` 50 vs 550** | **50** (now the `repaint_env.sh` default) | 2.4 h / ~465 core-hr vs ~26 h / ~5100 core-hr. 50 matches `runs/bind/run_0000` and the twobound trio, so every paired comparison and every mean-level number works. Extending later is now genuinely incremental (it was a silent no-op before). If you go 550, stage 5 task 7 builds the 50-real sibling the figures need. |
| 2 | **Seed the sampler first?** (§8) | author's call; it must land **before** the repaint to be worth anything | Otherwise the new released product is as irreproducible as the old one. |
| 3 | **Twobound-trio interim use** | **yes, now** — for gate thresholds, the corrected numbers in the text, and the paint-noise systematic | Zero compute. It cannot replace the repaint (§9). |
| 4 | **Reuse released mass planes?** | **removed — not available** | It planted symlinks that a later write would have followed into the released tree, destroying 53.7 GB. It saved a few node-minutes. |
| 5 | **Keep raw `rt_output`?** | **`KEEP_RAW=1`** (default) | 115 GB at 550 reals, but it makes the 50→550 extension, any re-collect, and the `n50/` sibling free. |
| 6 | **GPU lane** | a100 if undrained, else `NO_AMP=1 --constraint=v100s` | The a100 was drained at design time; check `sinfo` first. fp32 is harmless but is recorded in `.repaint_provenance.json`. |
| 7 | **Small-scale κ narrative** (§10.2) | needed **before** the corrected numbers go into the text | The closure gets *worse*; the cancellation story needs a physics explanation. |
| 8 | **Re-point the paper when?** | one reviewable commit after GATE 3, routing `LC` **and all 20 `runs/bind/run_0000` sites** through a single `FID` constant | Not a drip of per-figure edits. |
| 9 | **Ceph quota** | measure before stage 2 | ~115–257 GB depending on knobs 1 and 5; the filesystem has PB free but the user quota was never checked. |

---

## 12. TRIAGE LOG — reviewer issues, accepted and rejected

Three adversarial reviews (destructive / physics / ops lenses) raised 36 issues, several
duplicated across lenses. Every claim was re-verified against the actual files before acting.

### Accepted and fixed

| # | severity | issue | verification | fix |
|---|---|---|---|---|
| 1 | blocker ×3 | `REUSE_MASS_PLANES=1` symlinked released `lenspot*.dat`/`config.dat` into a directory the campaign writes to; `lensplane.py` writes with `open(path,"wb")`, which follows symlinks and truncates the target | confirmed by reading `lensplane.py:193/214/401` | **branch deleted entirely**; added `repaint_assert_no_released_symlinks()` (tested: refuses a planted link, allows harmless ones) before *and* after every plane write; copied the lux `SAFETY INVARIANT` comment into both plane and lux stages |
| 2 | blocker | lux stage guarded `RT_DIR` but not `LP_DIR`; an exported `LENSPLANE_DIR` would ray-trace the **old** planes into the new tree and pass every gate | code read; nothing downstream checks plane provenance | `repaint_assert_under_root` + no-released-symlinks on `LP_DIR`, and the resolved value is recorded in `$REPAINT_ROOT/.lp_dir` and re-asserted on every job |
| 3 | blocker | no per-chunk claim; the runbook tells the author to resubmit and to use a second lane, so two jobs would `rm -rf` the same scratch and a half-written `.dat` could be `mv`'d into `rt_output` | code read; `n1000_body.sh:110-128` is the working model | atomic `mkdir` claim per chunk with owner file + self-heal; job-unique `_work/rt_chunk<c>_<jobid>`; terminal cleanup removes **only this job's** dirs; `RELEASE_STALE_CLAIMS=1` escape |
| 4 | blocker | `--constraint=a100` pins to the only a100 node, which is **drained** | `scontrol show node pcn-16-06` → `State=IDLE+DRAIN`, `Reason="health cuda != 6"` | documented pre-flight `sinfo` check; added `NO_AMP` knob wired to the verified `--no_amp` flag; V100S fallback documented; fp32 recorded in `.repaint_provenance.json` |
| 5 | major ×3 | `.trace_complete` was existence-keyed, so the documented 50→550 extension was a silent no-op that printed "already complete"; the only escape (`FORCE=1`) also re-traced chunk 0 | logic traced by hand | marker now keyed on `.n_real`; `FORCE` (re-collect) split from `FORCE_CHUNKS` (discard chunks); stats stage tracks `.stats_n_real` and re-runs when the trace grows |
| 6 | major ×2 | 5 h wall with unlimited chunks ⇒ every job killed mid-chunk, ~150 core-hr discarded per submission; the header's own arithmetic was self-inconsistent | `sacct -j 2441237` (chunk = 02:25:20), `sacct -j 2457493` (released 550-real trace = one job, 3-day limit, 22:39:55) | wall → **48 h**; `--open-mode=append`; `--ntasks-per-node=48`; constraint widened to `cascadelake\|skylake`; loop refuses to start a chunk that will not fit in the remaining allocation |
| 7 | major | `run_lightcone_project.sh` defaults `OUTPUT_ROOT` to the released tree, has no guard, and would overwrite the stage-1 slabs the repaint symlinks | read the (session-modified) script | inline guard added to it **and** `run_paint_tng_project.sh`: refuse released roots, refuse an existing `stage1_manifest.json` without `FORCE=1`. Both branches tested standalone |
| 8 | major | `verify_repaint.py`'s "untouched" inventory covered everything **except** `lensplanes/` — the one asset actually at risk | code read | inventory extended by 241 explicit paths (252 → **493** files) plus a first+last-4 KB content anchor on the 80 `lenspot*.dat`; tested: baseline records 493/493 + 80 anchors in 15 s, and the anchor detects a same-size in-place rewrite |
| 9 | major | planes stage never asserted the composites carried the final paste geometry; because the new code writes `composite_thermo` and `paint_yplane._yslab` prefers it, `--r200_factor` becomes a **no-op** for y on the new tree | verified: released `composite_slab00.npz` has **no** `composite_thermo`; `pipeline.py:803` writes one; `cli/paint_yplane.py:49-52` prefers it | hard gate on `summary.json` `r200_factor`/`taper_frac` at the top of stage 3; stage 2b re-asserts its own output; `verify_repaint` gains a mid-k mass-power band (5%) as the taper check |
| 10 | major | §6 cross-check bands were wrong at low ℓ and self-contradictory for κ | **independently remeasured** from the trio's `Cl_{tau,kappa}.npz`: τ 0.9638/0.9487/0.9493 at ℓ=100–300 (vs the stated "0.922–0.928 flat"); κ(z_s=1) 1.0222/1.0220/1.0228 at ℓ=1e4–3e4 (vs "1.029" in §6 and "1.019" in §1.4) | §6 replaced with the measured per-ℓ tables; stop/go restricted to ℓ>1000; §1.4 reconciled to 1.022; the low-ℓ excursion attributed to the 50-vs-550 denominator artifact |
| 11 | major | `_check_cosmology` compared only `params[0]`, but the damage was `Ω_b/Ω_m`; a vector with the right `Ω_m` and CAMELS' `Ω_b` reproduces the bug and passes | verified TNG300-Dark `snapdir_096` header carries `Omega0`, `OmegaBaryon`, `HubbleParam` | now compares idx 0/6/7, skips absent header entries, reports `Ω_b/Ω_m`; `params.py` comment corrected. Unit-tested: old check PASS, new check raises on the sneaky vector |
| 12 | major | §7 claimed "a handful of constants"; `_build_figures_nb.py` reads `runs/bind/run_0000` at ~15 sites and the campaign made no replacement | measured: **20** occurrences; `runs/bind/run_0000/kappa_maps.npz` is `(50, …)` and holds `y_maps.npz` | §7 rewritten with the single-`FID`-constant plan; run_0000 moved to the **Must** list; stage 5 task 7 builds the `n50/` drop-in sibling |
| 13 | major | stats stage produced no `nu05_stats.npz` (canonical 22-point ν grid) | verified it exists in `runs/bind/run_0000` (Aug 12) and every twobound run, written by `nu_grid.compute`, not `lightcone_stats` | new task 5; import path tested (`nu_grid.compute`, NU n=22, −2.75..7.75) |
| 14 | major | no realization-matched DMO denominator ⇒ the campaign reproduces the documented 550-vs-50 defect | verified `runs/dmo/run_0000/kappa_maps.npz` is `(550, 5, 1024, 1024)` and `stats.py:103-112` only writes suppression in-process | new task 6 writes `Cl_kappa_dmo_n50.npz` from the first 50 (seed-paired) DMO reals; `S.cl_kappa` signature verified |
| 15 | major ×2 | `run_repaint_recomposite.sh` and `run_repaint_planes.sh` delegated to scripts whose own default `OUTPUT_ROOT` is the released tree and which write in place | read `run_lightcone_recomposite.sh:42,64-68` and the three plane scripts | all four delegations replaced with direct CLI calls using explicit absolute paths (flags reproduced exactly, including task-0's `--all_snap_dirs`) |
| 16 | major | repaint is pasted-only while `n1000_body.sh` defaults `DIFFUSE=1`, and the runbook never said so | verified `n1000_body.sh:50,176-183` | **doc fix, not a code change** — new §10.6 states pasted-only is deliberate (one variable at a time), that τ numbers are against patches-only truth, and that the n1000 redo re-applies `DIFFUSE=1` |
| 17 | major | gpu `--cpus-per-task=16` allowed only 2 concurrent tasks | `sacctmgr show qos` → gpu `MaxTRESPU cpu=40, gres/gpu=4` | `--cpus-per-task=8` (4 concurrent, both caps saturated); §4 cost row corrected to ~1–2.5 h |
| 18 | minor | `have_all()` used `-s`, which follows symlinks and accepts a truncated plane | plane sizes verified: lenspot 671088648 B, y/tau 134217736 B, 80 of each | exact-size + not-a-symlink test, sizes derived from `LP_GRID` in `repaint_env.sh` |
| 19 | minor ×2 | `n=$(ls … \| wc -l)` under `set -euo pipefail` dies silently on a non-matching glob | **reproduced**: exit 2, zero output | replaced with a `nullglob` array; tested — prints `ERROR: 0/80 …` and exits 1 |
| 20 | minor | `verify_repaint --slabs` defaulted to `[0]` = 666 halos vs a 1.5% tolerance, with ~30.6% per-patch replica RMS | released `summary.json`: slab 0 = 666 of 2933 halos | default → `[0,1,2,3]`; `--slabs 0` documented as the memory-capped fallback |
| 21 | minor | stats stage was the only one without `repaint_lock` | `grep` count 0 | `repaint_lock` added |
| 22 | minor | `paired_stats` ran without `--n_real`, silently changing the fixed-ν σ₀ convention at N_REAL=550 | flag verified present (`paired_stats.py:122`); released cache is 1.5 MB = 50 reals | `PAIRED_N_REAL=50` passed explicitly |
| 23 | minor | `TOL_MASS_SUM` is a tautology | verified `pipeline.py:788-789` sets `scale_global = dmo.sum()/composite.sum()`; released slab measures 1.000002 | kept but **relabelled** as a provenance assertion in code and in §6; the mid-k mass band power added as the check that actually moves |
| 24 | minor | gas band-power expectation off-centre (trio measures 0.912–0.923 vs 0.928 expected) | not independently remeasured (requires recompositing the trio = heavy compute) | tolerance widened 3% → **5%**, keeping the pre-registered physics prediction as the centre; both the prediction and the reviewer's measured trio values recorded in code and §6. Uncorrected still reads 1.000, so discrimination is unharmed |
| 25 | minor | storage estimate stale — released composites lack `composite_thermo` | verified by npz key listing; measured in-file compression ratios | §3 rebracketed to 26–35 GiB with an explicit "measure after Tier 0" instruction |
| 26 | minor | §10.2 did not note the y proper-vs-comoving-area convention asymmetry | consistent with `docs/diffuse_gas_validation.md` | sentence added to §10.2 |
| 27 | minor | bind_n1000 policy/runbook contradiction | verified 72 live n1000 jobs and `n1000_body.sh:32,115` | kept `bind_n1000` read-only for `repaint/` (nothing there writes it); §10.5 now states the redo is `n1000_body.sh`'s job, must use `FID=$REPAINT_ROOT`, and must wait for the 319-task campaign to drain |
| 28 | minor | Tier-0 → full-array race on snap_096 | `generate_complete` holds no lock | submit chain uses `--array=1-19`; rationale documented in-script |
| 29 | minor | stage 3 pinned to `cascadelake` while cca is saturated | `squeue`: both arrays pending on `QOSMaxNodePerUserLimit` | constraint widened to `cascadelake\|skylake` on stages 3 and 4; preempt lane documented |
| 30 | minor | guard hardening (a) relative paths validated against a different cwd than they are used in; (e) unguarded glob in the link script | `repaint_preamble` guards before `cd "$BIND_REPO"`; glob confirmed unguarded | (a) non-absolute paths refused outright, all four path vars checked before `cd`; (e) `shopt -s nullglob` + assert slab count == manifest `n_slabs` |

### Rejected, with evidence

| # | claim | verdict |
|---|---|---|
| R1 | "A `..` traversal through the `/mnt/home/mlee1/ceph` symlink escapes the guard — `/mnt/home/mlee1/ceph/../mlee1/ceph/bind_sb35/runs` returns ALLOWED." | **Overstated.** ALLOWED is *correct*: that path resolves to `/mnt/sdceph/users/mlee1/ceph/bind_sb35/runs` (symlink form) or `/mnt/home/mlee1/mlee1/ceph/…` (textual form), and **neither exists** — `/mnt/sdceph/users/mlee1/ceph` is not a directory (verified). It is a junk path, not a route into a released tree. The genuine `..` hazard (`…/bind_science/runs/../../bind_sb35`) was **already** refused. The `--no-symlinks` double-resolve was added anyway as belt-and-braces, and the real fix in this area was rejecting relative paths. |
| R2 | "Add `GENERATE_R200` to the `repaint_lock` line." | **Rejected.** `GENERATE_R200` is defined only in stage 2; stages 3–5 never set it, so putting it in the shared lock line would make every later stage's lock mismatch. It also has two *legitimate* settings (0.0 + recomposite, or 4.0 one-step) that must converge on the same product. The invariant that actually matters — what taper the composites on disk carry — is recorded in `summary.json` and is now hard-asserted by stage 3 before any plane is written. Rationale recorded in `repaint_env.sh`. |
| R3 | "Add a `DIFFUSE` knob to the plane stage, mirroring `n1000_body.sh`." | **Rejected as a code change, accepted as a doc change.** The repaint's purpose is to change exactly one variable (the conditioning cosmology) relative to the released master; applying the diffuse correction simultaneously would confound the old/new comparison that §1.4 and §6 rest on. Documented explicitly in the new §10.6 instead, including that the n1000 `bind`-arm redo re-applies `DIFFUSE=1`. |
| R4 | "Re-centre the gas band-power expectation on the measured 0.916." | **Partially rejected.** The measurement (recompositing three trio replicas) was not independently reproduced here, and re-centring a pre-registered threshold on an unverified number is exactly the failure mode pre-registration exists to prevent. The physics prediction stays as the centre and the tolerance was widened to 5% so the reported values sit comfortably inside; both numbers are recorded so GATE 2 can be *read*, not just thresholded. Re-centre later if the author reproduces the trio recomposite. |
| R5 | "Default `MAX_CHUNKS_PER_JOB=2` so the job stops before a 5 h wall." | **Superseded.** The premise (a 5 h wall) was itself the defect: the released 550-real trace ran as one job under a 3-day limit and cca allows 7 days. Raising the wall to 48 h plus a remaining-time check is strictly better — N_REAL=550 now finishes in one submission instead of six. `MAX_CHUNKS_PER_JOB` is retained as an optional knob, default 0. |
| R6 | "Collapse stage 3 into a single non-array job." | **Not adopted.** Real, but the 20-way array is also the resume unit and each task holds an exclusive node for only ~1–2 minutes of work. Widening the constraint to `cascadelake\|skylake` (360 vs 216 eligible nodes) and documenting the preempt lane addresses the queueing pressure without giving up per-snapshot restartability. |
