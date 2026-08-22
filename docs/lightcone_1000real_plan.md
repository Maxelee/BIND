# N1000 campaign: 1000 lightcone realizations per run

**Status: READY TO SUBMIT — 2026-08-10.** Scope decided: everything at N=1000, full
map cubes kept, maximum cluster usage, output to a **fresh tree** at
`/mnt/home/mlee1/ceph/bind_n1000` — the existing 50-real products under
`bind_science/` and `bind_sb35/` are never touched (the current paper keeps using
them; the N1000 tree is for the future release).

All rates measured from Aug 3–5 job history; scripts adversarially reviewed
(seed math, requeue convergence, data safety) with all findings fixed. Verified in
lux source: with `simulation_format = PreProjected` the current lux build only
*reads* the lensplane dirs — all writes go to `RT_output_dir` — so pointing the
`bind` trace and the seed gate at the sacred master lensplanes is safe.

## What runs

319 runs → `bind_n1000/{bind, dmo, truth, twobound/run_0000..0059, sb35/run_0000..0255}`,
each traced to 1000 seed-paired realizations (canonical ladder `seed = 1992 + 7r`,
identical plane geometry across every category at fixed r, so ratios cancel the
shared-box cosmic variance realization-by-realization).

Realizations are traced in 8 chunks of 125 via the **seed-offset property**
(verified in `lux/main.cpp`: realization *i* draws from a fresh RNG seeded
`base + 7i` and nothing else): chunk *c* uses base seed `1992 + 7·125·c`, its
local `run001..run125` are renamed to global `run0001..run1000`. The review
derived chunked ≡ unchunked exactly, for every r in 1..1000.

## The scripts

| File | Role |
|---|---|
| `n1000_seed_gate.sh` | **Submit first.** Traces 2 reals at base seed 2342 and byte-compares `config.dat` against master realizations 51/52 (`bind_lightcone_tng/rt_output`). PASS = chunking preserves the ladder. ~15 min, 1 node. |
| `n1000_body.sh` | Shared per-run body: manifest (index→run), campaign-params lock, atomic claims, paint planes (recomposite from the old tree's patches, read-only), chunked lux trace with per-chunk sentinels, collect, verify, cleanup. Fully requeue-convergent: every crash point resumes to the correct final state. |
| `run_n1000_cca.sh` | Guaranteed lane: cca QOS, 33 concurrent 3-node icelake jobs (99 nodes / 6,336 cores — pinned just under the per-user caps node=100/cpu=6400/jobs=50). Forward order: bind, dmo, truth first. |
| `run_n1000_preempt.sh` | Opportunistic lane: preempt QOS (no per-user TRES cap), 4×48-core nodes (`cascadelake\|skylake`), `--requeue`, 72 h wall, `%40` throttle (bounds transient disk). **Reverse** manifest order; claims prevent double-tracing where the lanes meet. |
| `run_n1000_stats.sh` | Post-trace per-run stats (Cℓ, peaks+minima, PDF/Minkowski, dm_stats; halo_scaling skipped — n_real-independent, lives in the 50-real tree). Tasks whose run isn't traced yet exit 0; resubmit to sweep. |

Safety/consistency machinery (added after the adversarial review):
`.campaign_params` lock (both lanes must resolve identical RT_SEED/N_REAL/CHUNK/grids
or exit loudly), claim self-heal for the mkdir/owner crash window, `FORCE=1`
retakes stale claims, `GSL_RNG_TYPE/GSL_RNG_SEED` unset before lux (generator-type
pinning — why one gate environment certifies both lanes), stage-3 collect
convergence after a kill during `--delete_raw`, and the `lux_io.py` numeric
run-dir sort (a real >999-realization ordering bug, fixed and unit-tested).

## Submission runbook

```bash
# 0. gate (~15 min) — read the log, require "SEED GATE PASSED"
sbatch n1000_seed_gate.sh

# 1. both lanes, same day
sbatch run_n1000_cca.sh
sbatch run_n1000_preempt.sh

# 2. stats sweeps as runs land (idempotent; rerun until no task skips)
sbatch run_n1000_stats.sh

# progress at a glance
ls /mnt/home/mlee1/ceph/bind_n1000/*/run_*/.trace_complete 2>/dev/null | wc -l   # of 319
squeue -u $USER -n bind_n1000_cca,bind_n1000_pre -h | wc -l

# recovery
#  - preempted preempt-lane task: automatic (requeue + chunk sentinels).
#  - TIMEOUT / node failure / permanently dead task: resubmit just that index, e.g.
#      sbatch --array=147 run_n1000_cca.sh
#    after releasing its claim if owned by the dead job:
#      rm -r /mnt/home/mlee1/ceph/bind_n1000/.claims/task_0147
#  - full-array resubmits are cheap: completed tasks exit in seconds on .trace_complete.
```

While the campaign runs, any other cca-QOS job under this account queues behind
the 6,336-core lane — plan interactive work accordingly (preempt/gen still work).

## Budget and forecast

- Total ≈ 2.38M core-hours (~50k 48-core-node-hours): 140 s/real full κ+y+τ trace
  (measured, 5 jobs, linear in N), 90 s/real DMO κ-only, ~14 s/real stats.
- cca lane alone at its 6,400-core cap: **~16 days**. With the preempt lane
  harvesting idle 48-core nodes, realistic finish **~9–16 days**; a ~1-week finish
  requires the preempt lane to average ~100+ extra nodes (possible when the
  cluster is quiet, not promisable).
- Storage: ~+18 TB final (full κ/y/τ cubes: ~58 GB per full run, 19.4 GB dmo)
  on sdceph (user at ~110 TB, no per-user quota attr, 6.8 PB free). Transient
  scratch ≈ 285 GB per in-flight run (planes 75 GB + raw ≤210 GB), bounded by
  33 + 40 concurrent runs ≈ **~21 TB peak** — cleaned per run at collect.
- Timing sanity: cca lane 48 h wall vs ~41 h/run; preempt lane 72 h (older cores
  +10–15%, and TIMEOUT is not auto-requeued — margin instead).

## Statistical payoff (why 1000)

Peak/minima Monte-Carlo noise ×√20 ≈ 4.5 smaller; a ~70-bin peak/minima covariance
becomes usable (Hartlap ≈ 0.93 at N=1000; not even invertible at 50). Ceiling: all
realizations recycle the one 205 Mpc/h box — the gain is measurement noise, not
cosmic variance; exactly right for the paired/ratio statistics, and 10³–10⁴ reals
from one box is established practice (kappaTNG).

## After the campaign

The 50-real tree stays canonical for Paper I. For release work on the N1000 tree,
remember the known consumer hardcodes if any figure code is pointed at it
(fig 7's `_load_fields_fid(n_real=50)` literal, the tutorial's `√50`, the ~7
direct DMO `Cl_kappa.npz` loads that bypass `dmo_paired_cl()` — all mapped in the
2026-08-10 review; none matter while consumers read the old tree). A
`cov_convergence`-style re-audit at N=1000 and a per-real low-ℓ pairing check
(ρ>0.99 across bind/dmo/truth) should gate the release.

## Decisions log (2026-08-10)

Full maps everywhere · everything at 1000 · max throughput (both lanes at the
caps) · fresh `bind_n1000` tree, no overwrites · stray twobound `_work` scratch
(8 × 75 GB, runs 0000–0007) verified regenerable and deleted (~601 GB freed).
