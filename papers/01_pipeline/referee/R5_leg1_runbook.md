# R5 leg 1 — independent re-paint of `snap_096` at all 256 Sobol nodes

**Status: PREPARED, NOT RUN.** No painting, no `sbatch`, nothing submitted.
This document specifies the decisive experiment for referee comment 5 so the
author can submit it. Every path, command and constant below was verified
read-only on 2026-08-12.

Companion memo with the results already in hand: `R5_latent_noise_results.md`.

---

## 1. What the experiment measures, and why it is the clean one

The referee's concern: $\lambda$ is measured from the same generated maps
whose statistics it predicts, so the fit could be partly predicting
realization noise from itself.

Write the end-to-end CV score as

```
R2(lambda) = R2_honest  -  f_noise  -  f_hat  +  2c
```

* `f_noise` = Var(node-specific realization noise in the statistic) / Var(design)
* `f_hat`  = Var(that noise propagated into the prediction through lambda) / Var(design)
* `2c`     = 2 Cov(statistic noise, prediction noise) / SS — **the referee's
  channel**, the only term that *inflates* the score.

Re-measuring $\lambda$ from an **independent paint** gives a $\lambda'$ whose
noise is uncorrelated with the statistics by construction, so

```
R2(lambda) - R2(lambda') = 2c        (exactly, to first order)
```

with `f_noise` and `f_hat` identical on both sides. **That difference is the
number the referee asked for, and nothing else in the campaign can produce
it**: no reshuffling of the existing halos can separate absorption from
attenuation, because the shared-draw covariance `Cov(n, eps)` is invariant
under halo subsampling (halving the halos doubles the latent noise variance
but leaves the covariance with the map unchanged), so a split-half test has
exactly zero power on `2c`. Only a fresh draw does.

**Expected effect size** (from the 3-replica measurement in the memo):
`2c = +2.8e-4` for $S(\ell)$, ceiling `1.0e-3`. Design the run to resolve
that, not to see a dramatic drop — see §7.

---

## 2. Crucial fact: there is no seed to change

The flow-matching sampler is **unseeded**. `src/bind/model.py:454`
(`FlowMatching.sample`):

```python
x = torch.randn(B, self.out_channels,
                condition.shape[2], condition.shape[3], device=device)
```

No `generator=`, no `seed` in the signature, and no `torch.manual_seed` /
`default_rng` anywhere on the call path
`bind.cli.generate_halos → bind.inference.paint_stages.generate_halos →
bind.Model.generate → FlowMatching.sample`. (The `seed` hits elsewhere in
`src/bind/` are the Sobol *design* seed in `inference/design.py:80`, the
lightcone realization seeds, and training seeds — all unrelated.)

Consequences:

1. **"Fresh seed set" is simply "run it again."** Every invocation into a new
   `--output_dir` already draws independent noise. No code change is needed
   to run leg 1.
2. The 256 campaign nodes were painted in separate array tasks, hence
   **already have mutually independent draws** — there is no common-mode
   cancellation across nodes. (Verified empirically: twobound runs
   0018/0049/0053 have bit-identical 35-parameter vectors yet differ at
   ~30% per-patch RMS; see `R3b_texture_results.md` and the memo's §3.)
3. **Reproducibility fix to propose upstream** (do this *with* leg 1 so the
   re-paint is itself reproducible): thread an optional `seed` through
   `FlowMatching.sample` and `bind.Model.generate`, exactly as
   `src/bind/wlemu/sample.py:90-91` already does —

   ```python
   gen = torch.Generator(device=self.device).manual_seed(int(seed))
   ...
   x = torch.randn(..., device=device, generator=gen)
   ```

   and expose `--seed` on `bind-generate-halos`. A collision-free convention
   for this experiment: `seed = 7_000_000 + 1000*run_id + slab_idx`
   (`run_id` 0–255, `slab_idx` 0–3); the campaign's own paints used no seed
   at all, so no value can collide with them. Record the seed in the output
   npz.

---

## 3. Inputs (all verified present)

| what | path | note |
|---|---|---|
| stage-1 cutouts | `/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096/stage1/` | shared, parameter-independent, 1.05 GB; **read-only** |
| per-node parameters | `/mnt/home/mlee1/ceph/bind_sb35/runs/run_{0000..0255}/params.npy` | 35-dim, 408 B each |
| checkpoint | `/mnt/home/mlee1/BIND/weights/fm_redshift_thermo/{last.ckpt,norm_stats.npz}` | 1.99 GB; redshift-conditioned + thermo — the checkpoint the campaign used (`docs/WORKLOG.md` L1167) |
| existing paint (comparison arm) | `/mnt/home/mlee1/ceph/bind_sb35/runs/run_NNNN/snap_096/composite_slab{00..03}.npz` | 1.08 GB/node; **read-only** |
| existing latents | `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz` | `sobol_*` rows, (256, 2933) |

`snap_096` holds **2933 halos in 4 slabs** (666 / 723 / 743 / 801), identical
across all 256 nodes.

Redshift conditioning: the campaign script passed **no** `--redshift` /
`--scale_factor` for `snap_096`; check `run_sobol_generate.sh`'s per-snapshot
loop before launching and match it exactly, or the re-paint will differ by
more than the draw.

---

## 4. The command chain

Per node (this is the whole of stage 2 — `bind-paint-project` and
`bind-paint-recomposite` are **not** needed, because leg 1 only re-measures
$\lambda$; the statistics $D$ stay as they are):

```bash
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
python -u -m bind.cli.generate_halos \
    --stage1_dir /mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096/stage1 \
    --params     /mnt/home/mlee1/ceph/bind_sb35/runs/run_${R}/params.npy \
    --run_dir    /mnt/home/mlee1/BIND/weights/fm_redshift_thermo \
    --output_dir /mnt/home/mlee1/ceph/referee_work/r5_repaint/runs/run_${R}/snap_096 \
    --device auto
# (console-script equivalent: `bind-generate-halos`, pyproject L47-72)
```

Writes `composite_slab{00..03}.npz` with `generated_patches (n,3,128,128)`,
`thermo_patches (n,4,128,128)`, `halo_centers`, `halo_masses`, `halo_r200`,
`condition_sums`, `n_halos`, `slab_idx`, `n_slabs`, `box_size` — the same key
set `build_atlas_200c.py::reduce_file` consumes.

**Never write into `bind_sb35/runs/`.** `generate_halos()` overwrites
unconditionally (the "skip if exists" guard lives only in the bash wrapper).

---

## 5. Storage layout

```
/mnt/home/mlee1/ceph/referee_work/r5_repaint/
├── runs/run_0000..run_0255/snap_096/composite_slab{00..03}.npz   # transient, 1.08 GB each
├── atlas/run_NNNN_snap096.npz                                    # per-node reduction, ~260 kB
├── atlas_cube_snap096_repaint.npz                                # consolidated (256, 2933) cube
├── logs/
└── PROVENANCE.json      # checkpoint sha, git rev, seeds, timestamps, node ids
```

Full retention = **277 GB**. Recommended: reduce-then-delete inside each array
task (`--purge_patches` in the wrapper), keeping the patches for **8 nodes only**
(~9 GB) as a spot-check sample. Peak transient footprint at concurrency 16 is
then ~18 GB, and the permanent product is ~66 MB.

---

## 6. Reduction step (`build_atlas_200c.py`-equivalent for Sobol)

The campaign's own reducer is `examples/sobol_atlas_mpi.py`, which lives on
branch **`analysis/sobol-sb35`** (not on `papers`). Recover it without a
branch switch:

```bash
git show analysis/sobol-sb35:examples/sobol_atlas_mpi.py \
    > papers/01_pipeline/referee/work/r5_sobol_atlas_mpi.py
git show analysis/sobol-sb35:examples/halo_atlas.py \
    > papers/01_pipeline/referee/work/r5_halo_atlas.py
```

It hardcodes its roots (`SOBOL_RUNS = bind_sb35/runs`,
`SOBOL_CACHE = bind_sb35/analysis_cache/halo_atlas`,
`CUBE_DIR = .../atlas_cubes`, L50-58). In the **copy** — never the campaign
file — repoint those three constants at `referee_work/r5_repaint/`. Then:

```bash
# reduce (MPI; module load ONLY openmpi, mpi4py is in the venv)
srun -n 64 python -u referee/work/r5_sobol_atlas_mpi.py --reduce --snaps 96
# consolidate into the (256, 2933) cube
srun -N1 -n1 python -u referee/work/r5_sobol_atlas_mpi.py --consolidate --snaps 96
```

The reduction primitive is `reduce_run_snap`, byte-identical in convention to
`build_atlas_200c.py::reduce_file` (`PIX = 6.25/128`, `R500_FAC = 0.659`,
background annulus `2.5–3.0 Mpc/h`, both apertures). Its cost is ~1.5 h on
2 nodes × 64 ranks for all 256×20 (run, snap) pairs; **snap_096 alone is 1/20
of that — under 10 min.**

**Gate before believing anything:** re-reduce 8 *original* nodes through the
copied reducer and require bit-identical agreement with the corresponding
`sobol_*` rows of `atlas_cube_snap096.npz`. If that fails, the discrepancy is
in the reduction, not the paint, and the experiment is void.

---

## 7. Analysis and the acceptance criterion

```bash
# lambda' from the fresh cube, everything else untouched
python -u referee/work/r5_repro.py --lat_cube <repaint cube>   # (add the flag)
```

Concretely, in `referee/work/r5_common.py` swap only `CUBE`, keep
`measure_lambda`, `report_folds(256, seed=1)`, the shipped
`family_model_bundle.npz` basis/mean and the shipped `__amps`, and report

```
Delta R2(stat) = R2(lambda) - R2(lambda')
```

per statistic. This is a **paired** comparison — same folds, same basis, same
amplitudes, same $D$ — so its error bar is far smaller than the $\pm6\times10^{-4}$
fold-redraw scatter quoted in the paper.

Expected: `Delta R2(clk) ≈ +3e-4`, with `|Delta R2| < 1e-3` for every
statistic except peaks/minima (expect up to `+7e-3` there).

* **`Delta R2 <~ 1e-3` for `clk`** → noise absorption is <0.3% of the
  0.381 $\lambda$-over-$\theta$ gap. Referee satisfied; quote the number.
* **`Delta R2 > 0.01` for `clk`** → the 3-replica estimate was wrong by
  30x; escalate, and re-examine the reduction gate of §6 first.
* Note the sign convention: a *negative* `Delta R2` is possible and harmless
  (the 3-replica measurement already shows negative values for PDF, $V_0$,
  $V_2$); it just means the noise realizations happened to anti-align.

**Half-leg option (25% cost, ~3σ on the effect).** Re-paint a random 64 of
the 256 nodes, refit the $\lambda\to a$ map on the untouched 192, and compare
held-out residuals for the 64 with $\lambda$ vs $\lambda'$. That gives
$64\times24$ paired residuals against the 3×24 available today — a ~20x
gain in degrees of freedom, enough to resolve `2c = 2.8e-4` at roughly 3σ
(estimated s.e. `9.5e-5`). The full leg reaches ~6σ. **Recommended first
move**, because a null at 3σ already settles the referee's point and costs a
quarter of the GPU time.

---

## 8. Cost

Throughput anchor: `run_sobol_generate.sh` L19 records **~4 GPU-hr per Sobol
node for all 20 lightcone snapshots**, which total **33,678 halos** (counted
from the per-snapshot fiducial atlases). That is **0.43 s/halo on an A100**
at the campaign's `--n_steps 50`, `--batch_size 16`.

> **The local V100S smoke test was not possible.** This session runs on
> compute node `pcn-1-67`, where `nvidia-smi` reports *"failed because it
> couldn't communicate with the NVIDIA driver"* and `/dev/nvidia*` does not
> exist — there is no GPU attached to this shell at all, so I could neither
> measure gen/s nor check whether another agent is using the workstation
> V100S. **All numbers below are extrapolated from the campaign's own
> documented rate**, not measured. The author should re-time one node before
> trusting the wall-clock plan.

| quantity | value |
|---|---|
| halos re-painted | 2933 × 256 = **750,848** |
| per node (A100) | 2933 × 0.43 s ≈ **21 min** (+ ~2 min model load / CUDA init) |
| total GPU time (A100) | **≈ 89 GPU-hours** |
| wall clock, `--array=0-255%16` | ≈ **6 h** |
| wall clock, half-leg `%16`, 64 nodes | ≈ **1.5 h** (22 GPU-hr) |
| local V100S serial (if free) | ~45 min/node → **8 days** full leg; use only for a smoke test or the half-leg (~2 days) |
| transient disk | 277 GB retained, **~18 GB** with reduce-and-delete |
| CPU reduction | < 10 min on 2 nodes × 64 ranks |

Optional 31% saving: only halos in the four latent mass bins
(`13.0–13.6` and `14.0–14.3`, a median 2012 of 2933 halos per node) affect
$\lambda$. Realising it needs a halo-subset argument in `generate_halos`,
which does not exist today — probably not worth the code change.

---

## 9. SLURM array sketch — **for the author to submit; I never submit**

Save as `run_r5_repaint.sh` at the repo root, review, then
`sbatch run_r5_repaint.sh`. Guidance: <https://wiki.flatironinstitute.org/SCC/Software/Slurm>

```bash
#!/bin/bash
#SBATCH --job-name=r5_repaint
#SBATCH --output=/mnt/home/mlee1/ceph/referee_work/r5_repaint/logs/%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/referee_work/r5_repaint/logs/%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=128G
#SBATCH --time=00:45:00
#SBATCH --array=0-255%16        # half-leg: --array=0-63%16
set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND

R=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
SB35=/mnt/home/mlee1/ceph/bind_sb35
OUTROOT=/mnt/home/mlee1/ceph/referee_work/r5_repaint
OUT="${OUTROOT}/runs/run_${R}/snap_096"
mkdir -p "${OUT}" "${OUTROOT}/atlas" "${OUTROOT}/logs"

# idempotency: skip a node that already produced all four slabs
if [[ -f "${OUT}/composite_slab03.npz" ]]; then
    echo "run_${R} already painted; skipping"; exit 0
fi

python -u -m bind.cli.generate_halos \
    --stage1_dir /mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096/stage1 \
    --params     "${SB35}/runs/run_${R}/params.npy" \
    --run_dir    /mnt/home/mlee1/BIND/weights/fm_redshift_thermo \
    --output_dir "${OUT}" \
    --device auto

# reduce immediately, then drop the 1.1 GB of patches (keep the first 8 nodes)
python -u papers/01_pipeline/referee/work/r5_reduce_one.py \
    --run_dir "${OUTROOT}/runs/run_${R}" --snap 96 \
    --out "${OUTROOT}/atlas/run_${R}_snap096.npz"
if (( SLURM_ARRAY_TASK_ID >= 8 )); then rm -f "${OUT}"/composite_slab*.npz; fi
```

`r5_reduce_one.py` is a ~20-line wrapper around
`reduce_run_snap` from the copied `r5_halo_atlas.py` — write it when the run
is scheduled, or drop the per-task reduction and run the MPI reducer of §6
once at the end (then keep all 277 GB until it finishes).

**Do not** reuse `run_sb35_generate.sh` from ceph: it `cd`s to
`/mnt/home/mlee1/vdm_bind2` and reads `bind_sb35/{conditions,weights}/`, none
of which exist any more.

---

## 10. Pre-flight checklist

1. `ls /mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096/stage1/` → manifest + 4 slabs.
2. `sha256sum weights/fm_redshift_thermo/last.ckpt` → record in `PROVENANCE.json`.
3. Confirm the campaign's per-snapshot `--redshift` handling for `snap_096` in
   `run_sobol_generate.sh` and match it.
4. Paint **one** node interactively, reduce it, and check
   $|\lambda' - \lambda| \lesssim 3\,\sigma_{\rm rep}$ per latent
   (`sigma_rep` in `r5_replicas.npz`). A gross mismatch means a
   conditioning difference, not draw noise — stop and diagnose.
5. Run the §6 reduction gate on 8 original nodes.
6. Only then launch the array.
