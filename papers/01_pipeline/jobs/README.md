# SLURM handoff pack — BIND Lightcone Paper I compute chain

Authored 2026-08-03. Nothing in this pack was submitted by the authoring agent except
where explicitly noted below as **already running** (discovered, not started, during
verification). You submit everything else by hand. All commands assume:

```bash
cd /mnt/home/mlee1/BIND
source /mnt/home/mlee1/venvs/BIND_env/bin/activate   # only needed for commands run
                                                       # outside an sbatch script
```

Every script/path named below was read directly and confirmed to exist before being
quoted here (see the per-job "verified" notes). Helper scripts referenced in this
README live alongside it in `papers/01_pipeline/jobs/`.

**Ordering**: Jobs 1→4 are a strict chain (the §4 emulator-repaint chain from
`TODO_EXECUTION_PLAN.md`). Jobs 5–7 are independent of that chain and of each other.

| Job | What | Status |
|---|---|---|
| 1 | Repaint run_0114/0115/0117 | **ALREADY RUNNING** — see callout below |
| 2 | Per-run caches + 256-run dataset reassembly | blocked on 1 |
| 3 | `run_emulator_fit.sbatch` | blocked on 2 |
| 4 | `run_figures.sbatch` | blocked on 3 (+ `EDITS_READY`) |
| 5 | 550-realization covariance sets (fid/truth/dmo) | independent, ready now |
| 6 | New Image 2 full-node completion | **deferred** — decision pending |
| 7 | Quick-win fiducial κ re-collect (100 real) | optional, default OFF |

---

## ⚠ Job 1 status: ALREADY SUBMITTED — do not resubmit

While preparing this pack (`squeue -u mlee1`, 2026-08-03 ~11:56) the fix chain for
Job 1 turned out to already be in flight:

```
JOBID         PARTITION  NAME           STATE    TIME   NODES  NODELIST
2457490_114   cca        bind_sobol_lc  RUNNING  7:42   4      pcn-11-[67-70]
2457490_115   cca        bind_sobol_lc  RUNNING  7:42   4      pcn-9-[64-67]
2457490_117   cca        bind_sobol_lc  RUNNING  7:42   4      pcn-11-[05-08]
```

and each run's `_work/` scratch had already been recreated (fresh, ~67 GB each,
timestamped 11:50) — i.e. someone already ran the `rm -rf _work` + `sbatch
--array=114,115,117 run_sobol_lightcone.sh` sequence below just before this pack was
written. **Do not run the commands in this section again** — the run dirs are
mid-repaint; a second `rm -rf` on the live `_work/` or a second `sbatch` racing the
same run directories would corrupt or waste the in-flight job.

Check status yourself with (read-only, scoped to your own jobs):

```bash
squeue -u mlee1 -j 2457490_114,2457490_115,2457490_117
```

Job header says `--time=05:00:00`; started ~11:50, so expect completion **≈16:30–17:00**
same day. Once all three finish, verify before moving to Job 2:

```bash
for r in run_0114 run_0115 run_0117; do
  echo "== $r =="
  ls -la /mnt/home/mlee1/ceph/bind_sb35/runs/$r/{kappa,y,tau}_maps.npz \
         /mnt/home/mlee1/ceph/bind_sb35/runs/$r/Cl_kappa.npz \
         /mnt/home/mlee1/ceph/bind_sb35/runs/$r/Cl_kappa_y.npz \
         /mnt/home/mlee1/ceph/bind_sb35/runs/$r/Cl_tau.npz
  ls -ld /mnt/home/mlee1/ceph/bind_sb35/runs/$r/_work 2>&1   # want: "No such file or directory"
done
```

The rest of this section documents Job 1 in full for the record (e.g. if a future
Sobol node needs the same repair) — it is **not** something to act on right now.

### Job 1 — Repaint the 3 corrupted Sobol nodes (reference only, currently in flight)

**Purpose.** `run_0114`/`run_0115`/`run_0117` under `bind_sb35/runs/` each have only
`params.npy` + healthy `snap_NNN/composite_slab*.npz` per-halo patches (verified: real
files, 9–20 MB each) but no `kappa_maps.npz`/`Cl_*.npz` — a 2026-06-23 crash left a
stale `_work/` scratch dir behind. `run_sobol_lightcone.sh`'s per-snapshot recomposite
step has an **existence-only** idempotency check:

```bash
if [[ ! -f "$WORK/snap_${s3}/composite_slab00.npz" ]]; then
    python -u -m bind.cli.paint_recomposite ...
fi
```

The crash left a **0-byte** `_work/snap_080/composite_slab00.npz` (confirmed by `ls
-la`, run_0114) that satisfies this check, so the recompose step is skipped and the
next stage (`paint_lensplane`) crashes with `EOFError` trying to `np.load` the empty
file. Deleting `_work/` forces a clean recompose on rerun.

**Commands** (already executed once — shown for reproducibility):
```bash
# 1. verify current corrupted state (read-only)
ls -la /mnt/home/mlee1/ceph/bind_sb35/runs/run_0114/_work
ls -la /mnt/home/mlee1/ceph/bind_sb35/runs/run_0115/_work
ls -la /mnt/home/mlee1/ceph/bind_sb35/runs/run_0117/_work

# 2. clear the stale scratch (destructive -- ONLY these three _work/ dirs)
rm -rf /mnt/home/mlee1/ceph/bind_sb35/runs/run_0114/_work
rm -rf /mnt/home/mlee1/ceph/bind_sb35/runs/run_0115/_work
rm -rf /mnt/home/mlee1/ceph/bind_sb35/runs/run_0117/_work

# 3. verify it's gone
ls /mnt/home/mlee1/ceph/bind_sb35/runs/run_0114/_work 2>&1
ls /mnt/home/mlee1/ceph/bind_sb35/runs/run_0115/_work 2>&1
ls /mnt/home/mlee1/ceph/bind_sb35/runs/run_0117/_work 2>&1

# 4. resubmit the repaint (array index == Sobol run number)
cd /mnt/home/mlee1/BIND
sbatch --array=114,115,117 run_sobol_lightcone.sh
```

**Wall time / resources.** Script header: 4 nodes × 48 tasks (192 total), exclusive,
cascadelake, `--time=05:00:00` per array index. Passing an explicit `--array=114,115,117`
on the `sbatch` command line replaces the script's default `0-255%8` throttle entirely,
so all three launch as independent 4-node allocations (12 nodes total) — expect
**~5h wall** if the scheduler seats all three concurrently (it did: see above).

**Outputs to verify:** per run, `kappa_maps.npz`, `y_maps.npz`, `tau_maps.npz`,
`Cl_kappa.npz`, `Cl_kappa_y.npz`, `Cl_tau.npz` under `bind_sb35/runs/run_0NNN/`, and
`_work/` gone (the script's last line is `rm -rf "$WORK"` on success).

**Unblocks:** Job 2 (256-run dataset), and transitively Jobs 3–4.

---

## Job 2 — Per-run caches for the 3 repaired runs + 256-run dataset reassembly

**Purpose.** `bind-emulator-assemble` walks `bind_sb35/runs/run_*` keeping every run
with `Cl_kappa.npz` present (`require="Cl_kappa.npz"` in
`src/bind/emulator/dataset.py:265`) — confirmed by reading `assemble()` directly. Once
Job 1 lands, a plain rerun with no code changes picks up all 256 runs. Separately,
`build_nu_cache.py` (the common-ν-grid cache for figs 4/5/7/8) is keyed **per Sobol
run** and needs new shards for the 3 previously-missing runs before it can reassemble
`emulator_dataset_nu8.npz` on the full 256.

**Correction to the naive assumption that mf/field caches also need per-run shards:**
reading `mf_cache.py` and `field_cache.py` directly shows neither is keyed by Sobol
run at all —

```python
# mf_cache.py
SIDES = ("bind", "truth")                       # the fiducial pair only

# field_cache.py
TASKS = [("kk", "bind"), ("kk", "truth"), ...]  # side in {"bind","truth"} only
```

— both cache the fiducial validation pair (`bind_science/runs/{bind,truth}/run_0000`),
never `bind_sb35` Sobol runs. **`run_mf_cache.sbatch` / `run_field_cache.sbatch` do not
need to be touched for this repaint** — they are already built (confirmed on disk,
`bind_science/mf_cache/mf_nu8_snap096.npz` and `bind_science/field_cache/
field_stats_fid.npz`, both from 2026-07-30) and are unaffected by the Sobol repaint.
Only `run_nu_cache.sbatch`'s per-run shard step needs action.

Verified current state: `emulator_dataset_xpkfix.npz` has exactly 253 runs, missing
`{114, 115, 117}` (checked by loading it directly); `bind_sb35/nu8_shards/` has exactly
255 shards (253 Sobol + the `bind`/`truth` fiducial-pair shards, 0 for the 3 corrupted
runs).

**Commands** — a ready-made helper script does the whole chain with a prereq guard and
a safety backup of the dataset before overwriting it:

```bash
cd /mnt/home/mlee1/BIND
sbatch papers/01_pipeline/jobs/job2_repair_caches.sbatch
```

which runs, in order:
```bash
# 1. nu8 shards for exactly the 3 repaired runs (skips if a shard already exists)
python build_nu_cache.py --run run_0114
python build_nu_cache.py --run run_0115
python build_nu_cache.py --run run_0117

# 2. back up the 253-run dataset, then reassemble in place -> 256 runs
cp emulator_dataset_xpkfix.npz emulator_dataset_xpkfix.bak253.npz
bind-emulator-assemble --out /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.npz
# (uses the bind.emulator.dataset defaults: runs_dir=bind_sb35/runs,
#  dmo_dir=bind_science/runs/dmo/run_0000, parquet=bind_sb35/analysis_cache/integrated.parquet,
#  scaling_snap=96 -- all confirmed via `assemble()`'s own defaults, unchanged from history)

# 3. reassemble + verify the nu8 dataset on the now-256-run xpkfix file
python build_nu_cache.py --assemble
python build_nu_cache.py --verify
```

If you'd rather run it interactively instead of via sbatch, the same 3 steps work
directly from `papers/01_pipeline/` after `source .../BIND_env/bin/activate` — nu8
shard cost is measured at ~159 s/run (comment in `run_nu_cache.sbatch`), so 3 runs is
~8 minutes serial; the reassemble steps are seconds.

**Wall time / resources.** Trivial: 1 node, 4 cores, 32 GB, `--time=00:30:00` is a
generous ceiling (measured cost ≈10 min total).

**Outputs to verify:**
```bash
ls -la /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.npz        # size should grow (~127-130 MB)
ls -la /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.bak253.npz # 253-run backup, 126,252,975 bytes
ls -la /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_nu8.npz
python3 -c "
import numpy as np
d = np.load('/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.npz', allow_pickle=True)
print('n runs:', len(d['run_ids']))   # want 256
"
```

**Unblocks:** Job 3 (the fit's cache fingerprint changes — `n_train`/`train_idx` shift
from 203/253-50 to 206/256-50 — so `run_emulator_fit.sbatch` auto-refits with no code
change).

---

## Job 3 — `run_emulator_fit.sbatch` (GP emulator refit)

**Purpose.** Fit + persist the 13-head Paper I GP emulator bundle (head list re-ruled
2026-08-03 fork A: the 12-head "S(ℓ) primary" set plus `cl_kappa` trained directly on
map-measured spectra — `build_emulator.py` STATS mirrors the notebook's 13-head `STATS_EMU`)
(`emulator_fits/paper1_gp.pt` + sidecar `paper1_gp.json`) that fig 9 and every
downstream figure load via `emulator_cache.load_matching()`. The fit config (dataset,
stats list, seed-0 train/test split, backend `gpgpu`, `n_components=12`,
`{epochs:400, lr:0.1}`) is fingerprinted; since Job 2 changes `n_runs` 253→256, the
seed-0 test draw (`N_TEST=50`) shifts and `train_idx_sha256_16` no longer matches the
currently-saved bundle (`n_train: 203` today) — the refit is **auto-forced**, no
`--force` flag needed or exists.

**Commands:**
```bash
cd /mnt/home/mlee1/BIND
sbatch papers/01_pipeline/run_emulator_fit.sbatch
```
which runs `python build_emulator.py --fit` then `--fit`'s companion `--verify` inside
the same job.

**Wall time / resources.** Header: 1 node, 1×A100 GPU, 16 CPUs, 64 GB, `--time=01:00:00`
(comment: "the whole fit is a couple of minutes" — not parallelised across heads by
design, since `Emulator.save()` serialises the whole object at once).

**Gotcha to carry forward** (`emulator_cache.py:_fingerprint`): for any dataset file
**≥5 MB**, the recorded `dataset_sha256_16` hashes only the **filename**, not its
bytes (`emulator_dataset_xpkfix.npz` at ~127 MB always takes this branch). This run's
refit is correctly forced by the `n_train`/`train_idx` fields changing — but if the
dataset file is ever regenerated in place with the **same run count** (e.g. a stats
bugfix that doesn't add/remove runs), `load_matching()` cannot detect that content
changed and will silently reuse the stale bundle. Worth remembering the next time
`emulator_dataset_xpkfix.npz` is rebuilt for a reason other than run count.

**Outputs to verify:**
```bash
cat /mnt/home/mlee1/ceph/bind_sb35/emulator_fits/paper1_gp.json   # want "n_train": 206
ls -la /mnt/home/mlee1/ceph/bind_sb35/emulator_fits/paper1_gp.pt  # mtime should update
```
plus the job log (`ceph/logs/emu_fit_<jobid>.out`) should show 13/13 heads fit, then
the `--verify` block with no `MISSING head` / `SHAPE MISMATCH` lines and exit 0.

**Unblocks:** Job 4 (the notebook loads this exact bundle for fig 9 and every
downstream emulator-driven figure/number).

---

## Job 4 — `run_figures.sbatch` (full figure/number rebuild)

**Purpose.** Regenerates `paper1_figures.ipynb` from `_build_figures_nb.py` and
executes it end-to-end with the BIND venv kernel (figures land in `figs_v2/` +
`figs_preview/`), now against the 256-run emulator from Job 3. This is also where the
§4 TODO verification pass happens, per `TODO_EXECUTION_PLAN.md`.

**Commands:**
```bash
cd /mnt/home/mlee1/BIND
sbatch papers/01_pipeline/run_figures.sbatch
```

**Gate — check before submitting.** The job waits up to 4h (1-min polls) for
`papers/01_pipeline/EDITS_READY` to exist, then runs unconditionally once it does — it
does **not** check the sentinel's age. As of this writing `EDITS_READY` (and
`RUN_DONE`, containing job id `2454934`) are both **stale**, dated 2026-07-30 08:13,
i.e. from the *previous* figure-overhaul round, not the current TODO-driven edit
campaign (Workstream 1 in `TODO_EXECUTION_PLAN.md`). Confirm Workstream 1's final
"haiku bookkeeping" step has refreshed `EDITS_READY` (or refresh it yourself once you
have confirmed the builder edits landed) before relying on the gate — otherwise this
job will execute against a half-finished builder the moment it happens to see the old
sentinel.
```bash
ls -la /mnt/home/mlee1/BIND/papers/01_pipeline/EDITS_READY   # check mtime is AFTER Workstream 1 finished
```

**Wall time / resources.** Header: 1 node, 1×A100 GPU, 16 CPUs, 128 GB,
`--time=08:00:00` (GPU accelerates the gpytorch fits inside the notebook; CPUs serve
the 50-node coverage test + bootstrap loops).

**Outputs to verify:**
```bash
cat /mnt/home/mlee1/BIND/papers/01_pipeline/RUN_DONE     # new SLURM job id, not the stale 2454934
# RUN_FAILED instead of RUN_DONE means the gate timed out or the notebook errored
ls -la /mnt/home/mlee1/BIND/papers/01_pipeline/figs_v2/ | tail -20
```

**Unblocks:** the §4 adversarial verification pass (diff vs backup, item-by-item TODO
verdict) that follows in Workstream 1.

---

## Job 5 — Covariance sets: 550-realization retrace (fiducial / truth / DMO)

**Purpose.** Extend the three seed-paired lightcone traces from 50 (fiducial/truth) or
100-raw-but-50-collected (see Job 7) realizations to a common **550-realization**
target (TODO floor: 250) for a measured-realization covariance (needed for the
LSST-Y10 band and other covariance-dependent figure items in Workstream 1 step 4).
Truth's replane additionally produces its **first-ever** `tau_maps.npz`/`Cl_tau.npz`,
unblocking the truth-τ / `C_ℓ^{yτ}` comparison noted as a historical gap (truth
predates `paint_tauplane`).

**Does lux support an offset, or does it always trace `r = 0..N-1`?** Read all five
lux-ini-writing scripts in this repo (`run_sobol_lightcone.sh`, `run_sobol_lux.sh`,
`run_lightcone_tau.sh`, `run_twobound_lightcone.sh`, and the ini block duplicated in
each): every one of them exposes exactly `RT_random_seed` (fixed at 1992) and
`n_realizations = $N_REAL`, with the realization loop written as
`for i in $(seq 1 "$N_REAL")`. **No script in this repo exposes a start-index / seed-
offset knob** — the only parameter controlling which realizations get traced is the
total count. Combined with the known seed pairing (`1992 + 7r` per realization,
recorded in project memory from `rt_output/runNNN/config.dat` tails), a fresh
`N_REAL=550` run reproduces realizations 1–50 bit-for-bit (10% redundant) rather than
being able to append 500 new ones to the existing 50/100. **A fresh 550-realization
full retrace per target is therefore the only option exposed by this codebase, and is
the clean choice** — it's also what's needed anyway for truth (no tau leg exists yet
at any realization count) and for fiducial (to keep κ/y/τ paired at the same N, see
Job 7's caveat).

**Seed convention — do not change:** `RT_random_seed` stays at its default `1992` in
every command below. This is what keeps fiducial/truth/DMO realizations paired
(cosmic variance cancels realization-by-realization); changing it breaks the pairing
for every downstream closure test.

### 5a — Truth: one-time replane (prerequisite for 5c)

Truth (`bind_science/runs/truth/run_0000`) retains per-halo hydro patches
(`snap_NNN/composite_slab*.npz`, verified on disk) but its `lensplanes/`/`rt/` were
cleaned after the original build (verified: neither directory exists, and there is no
`tau_maps.npz` — only `kappa_maps.npz` + `y_maps.npz` at 50 realizations each).
`run_sobol_paste.sh` rebuilds `lensplanes/` from the retained patches and **already
includes the τ-plane step** (`paint_tauplane`), so one pass gets kappa+y+tau lensplanes
in place for the first time.

```bash
cd /mnt/home/mlee1/BIND
DESIGN=truth OUTPUT_ROOT=/mnt/home/mlee1/ceph/bind_science \
STAGE1_ROOT=/mnt/home/mlee1/ceph/bind_lightcone_tng \
sbatch --array=0-0 run_sobol_paste.sh
```

Wall time: header default (1 node, icelake, 16 cpus, 128 GB, `--time=04:00:00`) is
already generous for a *single* array task (the header's normal use is 100 array tasks
sharing that ceiling) — expect well under an hour based on the per-snapshot paint cost
in the analogous Sobol-repaint step. Verify: `bind_science/runs/truth/run_0000/
lensplanes/` populated with `config.dat`, `lenspot*.dat`, `yplane*.dat`,
`tauplane*.dat` (20 snapshots × 4 planes each, per the shared geometry).

### 5b — Fiducial: fresh 550-realization κ+y+τ retrace

Fiducial (`bind_lightcone_tng`) already has `lensplanes/` (70 GB, geometry-only,
independent of realization count) and 100 raw `rt_output/runNNN` dirs, but its
collected `kappa_maps.npz`/`y_maps.npz`/`tau_maps.npz` are all still 50-realization
(970/954/932 MB respectively, i.e. ~19.4 MB/realization/map-type — measured directly).
`run_lightcone_tau.sh` is the "add/refresh the τ leg on the existing fiducial" script;
setting `N_REAL=550` makes it retrace all three fields together, fresh:

```bash
cd /mnt/home/mlee1/BIND
N_REAL=550 sbatch --time=3-00:00:00 run_lightcone_tau.sh
```

**⚠ disk note specific to this script**: unlike the Sobol/twobound/DESIGN-based
scripts, `run_lightcone_tau.sh` does **not** clean up `rt_output/` afterward (by
design — the fiducial retains its raw trace). Raw `rt_output/` is currently 18.27 GB
for 100 realizations (~183 MB/realization, includes shear γ maps not just κ/y/τ) — at
550 realizations expect **~100 GB** of raw `rt_output/`, i.e. an **additional ~83 GB**
beyond the final compressed-npz growth counted below. Nothing here deletes old raw
realizations either; if you want to reclaim that space after verifying the new
`kappa_maps.npz`/`y_maps.npz`/`tau_maps.npz`, that's a manual `rm` decision for you to
make once the new collected files are confirmed good — not automated here.

### 5c — Truth: fresh 550-realization κ+y+τ retrace (after 5a)

```bash
cd /mnt/home/mlee1/BIND
DESIGN=truth OUTPUT_ROOT=/mnt/home/mlee1/ceph/bind_science N_REAL=550 \
sbatch --time=3-00:00:00 --array=0-0 run_sobol_lux.sh
# then, after the above completes:
DESIGN=truth OUTPUT_ROOT=/mnt/home/mlee1/ceph/bind_science N_REAL=550 \
sbatch --time=12:00:00 --array=0-0 run_sobol_stats.sh
```
`run_sobol_stats.sh`'s default `KEEP_RAW=0` deletes `lensplanes/`/`rt/`/`lux.ini`
after collecting — no lingering raw-trace disk growth for truth (unlike 5b).

### 5d — DMO: fresh 550-realization κ-only retrace (independent of 5a–5c)

DMO (`bind_science/runs/dmo/run_0000`) is κ-only by construction (no baryons ⇒ no
y/τ). **CORRECTION (2026-08-04)**: the original pack claimed "no replane needed" —
WRONG. `run_sobol_stats.sh`'s default cleanup deleted the DMO `lensplanes/` after the
original 50-realization collection, and the 2026-08-03 5d submission crashed at lux
startup on the missing `lensplanes/config.dat` (task 0 exit 1; the other 191 ranks
hung, burning 4 nodes for ~18 h until cancelled). The replane MUST run first:
```bash
cd /mnt/home/mlee1/BIND
sbatch run_dmo_lensplane.sh        # 20 array tasks, 1 node/16 cpu each, ~2 h
```
then the two-command pattern from `run_dmo_lensplane.sh`'s own header, with
`N_REAL=550` added:
**SECOND CORRECTION (2026-08-04)**: `COMPUTE_TAU=False` is REQUIRED too — run_sobol_lux.sh
has separate COMPUTE_TSZ and COMPUTE_TAU switches (both default True; the header comment
predates the tau machinery). Without it lux demands tauplane01.dat, which the DMO replane
correctly does not build, and dies at startup with hung ranks (job 2457765).
```bash
cd /mnt/home/mlee1/BIND
COMPUTE_TSZ=False COMPUTE_TAU=False DESIGN=dmo OUTPUT_ROOT=/mnt/home/mlee1/ceph/bind_science N_REAL=550 \
sbatch --time=3-00:00:00 --array=0-0 run_sobol_lux.sh
# then:
DESIGN=dmo OUTPUT_ROOT=/mnt/home/mlee1/ceph/bind_science N_REAL=550 \
sbatch --time=12:00:00 --array=0-0 run_sobol_stats.sh
```
Expect this to run **faster** than 5b/5c (no tSZ-y or τ ray-trace overhead, kappa
only) — the `--time=3-00:00:00` ceiling below is conservative, not a prediction.

### Convenience: submit all of 5a–5d with dependency chaining

```bash
cd /mnt/home/mlee1/BIND
bash papers/01_pipeline/jobs/submit_job5_covariance.sh          # all of fid+truth+dmo
# or individually:
bash papers/01_pipeline/jobs/submit_job5_covariance.sh fid
bash papers/01_pipeline/jobs/submit_job5_covariance.sh truth
bash papers/01_pipeline/jobs/submit_job5_covariance.sh dmo
```
This script only calls `sbatch` — it does not itself get run by anyone but you; read
it before running to see exactly what it submits (it prints every job id as it goes,
and chains truth's replane→retrace→stats via `--dependency=afterok`).

**Wall time / resources (scaled from the measured 4-node/192-task/4-5h-per-50-
realizations baseline for the lux+collect+stats stages).** 550 vs 50 realizations is
an **11×** scale-up ⇒ naive linear scaling gives **~44–55h** per lux-trace job. Both
`cca` and `gpu` partitions report `MaxTime=7-00:00:00` (`scontrol show partition`,
confirmed), so a single job comfortably fits; the `--time=3-00:00:00` (72h) requested
above is a safety margin above the estimate, not a hard requirement. Collect+stats
scale with data volume but are cheap relative to ray-tracing; `--time=12:00:00` for
those steps is generous headroom, not a measured number.

**Disk.** Final collected products only, at 550 realizations (measured ~19.4
MB/realization/map-type):
| target | map types | final size |
|---|---|---|
| fiducial | κ, y, τ | ~32 GB |
| truth | κ, y, τ (τ new) | ~32 GB |
| DMO | κ only | ~10.7 GB |
| **total** | | **~75 GB** |

This replaces (not just adds to) the current smaller files (~2.85 GB fid + ~1.94 GB
truth + ~0.97 GB DMO at 50 real today), so net new usage from the *final* products is
~69 GB. **Add ~83 GB** on top for fiducial's un-cleaned raw `rt_output/` growth (see
5b) if you don't manually prune it. `df -h /mnt/home/mlee1/ceph` currently shows 7.0 PB
free — headroom is not a concern, this is purely for quota bookkeeping.

**Outputs to verify** (each map-type file should individually be ~10.7 GB at 550
realizations — measured at 19.4 MB/realization — so ~32 GB across all three fiducial
files, ~32 GB across all three truth files, ~10.7 GB for DMO's single kappa file,
matching the ~75 GB total in the table above):
```bash
ls -la /mnt/home/mlee1/ceph/bind_lightcone_tng/{kappa,y,tau}_maps.npz
ls -la /mnt/home/mlee1/ceph/bind_lightcone_tng/Cl_tau.npz
ls -la /mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/{kappa,y,tau}_maps.npz
ls -la /mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/Cl_tau.npz    # new -- did not exist before
ls -la /mnt/home/mlee1/ceph/bind_science/runs/dmo/run_0000/kappa_maps.npz
```

**Unblocks:** measured-realization covariance for the LSST-Y10 band item in
Workstream 1 step 4; truth's first `tau_maps.npz`/`Cl_tau.npz` (unblocks the fig04b
truth-τ overlay and the `C_ℓ^{yτ}` truth comparison, both currently guarded on file
existence per the TODO plan so they'll auto-upgrade once these land).

---

## Job 6 — New Image 2 full-node completion (deferred)

**Status: decision-gated, nothing to submit yet.** Per `TODO_EXECUTION_PLAN.md`, New
Image 2 (the 177/253 DM/FRB panel gap) fate — "§6 compact vs cut to Paper IV" — is
explicitly listed as one of the decisions reserved for you; a decision memo is being
prepared separately (`prototypes` workstream). Nothing is actioned here. Once a
direction is chosen, this section of the pack should be filled in with the specific
compute needed to complete it (likely a targeted extension of the existing halo/FRB
DM pipeline to the missing 76/253 nodes — TBD pending the decision).

---

## Job 7 — Optional quick win: fiducial κ re-collect to 100 realizations (default OFF)

**Purpose.** `bind_lightcone_tng/rt_output/` already holds 100 raw realizations
(confirmed: `ls rt_output/run*` → 100 dirs), but the collected `kappa_maps.npz` only
reflects the first 50 (970 MB = 50 × 19.4 MB, confirmed). Realizations 51–100 were
only ever kappa-only (per project memory, "runs 051–100 are κ-only" — no y/τ ray-trace
was done for those, since `y_maps.npz`/`tau_maps.npz` are also both 50-real sized
today). Re-collecting to 100 realizations costs **no new ray-tracing**, just a
collect+stats pass over data that already exists on disk:

```bash
cd /mnt/home/mlee1/BIND
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
python -m bind.cli.lux_collect --rt_root /mnt/home/mlee1/ceph/bind_lightcone_tng/rt_output \
    --output_dir /mnt/home/mlee1/ceph/bind_lightcone_tng --n_real 100 --fov_deg 5.0
python -m bind.cli.lightcone_stats --run_dir /mnt/home/mlee1/ceph/bind_lightcone_tng --no_halo_scaling
```
(cheap CPU-only, no MPI/lux needed — run directly or wrap in a trivial 1-node sbatch,
`--time=01:00:00` is generous.)

**⚠ Caveat — why this defaults OFF.** This doubles `kappa_maps.npz` to 100
realizations while `y_maps.npz`/`tau_maps.npz` stay at 50, because realizations 51–100
were never y/τ-traced. **This breaks the same-N pairing across κ/y/τ** that every
paired closure test and joint κy/κτ statistic in the figure package assumes (they
index all three by the same realization number). Any code that currently assumes
`kappa_maps["kappa"].shape[0] == y_maps["y"].shape[0] == tau_maps["tau"].shape[0]`
would need an explicit guard or would silently misalign 50 unmatched realizations.

**Recommendation:** if Job 5b (fiducial full 550-realization retrace, all three
fields together) is done, **skip this entirely** — 5b already gives a larger,
self-consistently-paired κ/y/τ set and makes this quick win redundant. Only worth
doing if you want a same-day κ-only number bump without waiting on 5b's ~2-day retrace,
and are prepared to keep the y/τ side at 50 in the meantime.

---

## Appendix: files in this pack

- `README.md` — this file.
- `job2_repair_caches.sbatch` — Job 2's cache-repair + dataset-reassembly chain
  (prereq-guarded, backs up the dataset before overwriting).
- `submit_job5_covariance.sh` — Job 5's four `sbatch` invocations with truth's
  replane→retrace→stats dependency chained via `--dependency=afterok`; prints every
  job id as it submits. Not run by anyone but you.

## Appendix: source scripts referenced (all verified to exist by direct read/`ls`)

| script | location | role |
|---|---|---|
| `run_sobol_lightcone.sh` | repo root | Job 1 — one Sobol node, full recompose→paint→trace→collect→stats |
| `build_nu_cache.py`, `nu_grid.py` | `papers/01_pipeline/` | Job 2 — per-run ν-grid shards + dataset reassembly |
| `mf_cache.py`, `build_mf_cache.py` | `papers/01_pipeline/` | fiducial-pair only, unaffected by Job 2 |
| `field_cache.py`, `build_field_cache.py` | `papers/01_pipeline/` | fiducial-pair only, unaffected by Job 2 |
| `src/bind/cli/emulator_assemble.py` | package | `bind-emulator-assemble` entry point |
| `src/bind/emulator/dataset.py` | package | `assemble()`, `DEFAULT_RUNS/DMO/PARQUET` |
| `build_emulator.py`, `emulator_cache.py`, `run_emulator_fit.sbatch` | `papers/01_pipeline/` | Job 3 |
| `_build_figures_nb.py`, `_run_figures_nb.py`, `run_figures.sbatch` | `papers/01_pipeline/` | Job 4 |
| `run_sobol_paste.sh`, `run_sobol_lux.sh`, `run_sobol_stats.sh` | repo root | Job 5a/5c/5d — DESIGN-parameterized paste/trace/collect |
| `run_lightcone_tau.sh` | repo root | Job 5b — fiducial-specific tau-add/retrace |
| `run_dmo_lensplane.sh` | repo root | reference for the DMO lensplane-once pattern |
| `run_twobound_lightcone.sh` | repo root | cross-checked to confirm the lux-ini schema has no offset knob |
