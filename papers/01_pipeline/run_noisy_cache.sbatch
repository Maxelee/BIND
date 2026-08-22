#!/bin/bash
#SBATCH --job-name=noisy_cache
#SBATCH --output=/mnt/home/mlee1/ceph/logs/noisy_cache_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/noisy_cache_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=450G
#SBATCH --time=03:00:00

# ── LSST-Y10 noisy-statistics cache (Lee+22 arXiv:2201.08320 Eq.6/7 recipe) ───────
# Adds LSST-Y10 shape noise + a 1'-theta_G Gaussian smoothing to the RELEASED kappa
# cubes (no repaint) and recomputes peak_counts/minima_counts/pdf/mf_v0/v1/v2/
# cl_kappa on the noisy smoothed maps, for every lightcone target: the fiducial
# BIND-painted suite (bind_lightcone_tng, 550 real), the hydro truth
# (bind_science/runs/truth/run_0000, 550 real), DMO (bind_science/runs/dmo/run_0000,
# whatever it currently holds), and all 256 Sobol runs (bind_sb35/runs/run_NNNN, 50
# real each). See noisy_grid.py's docstring for the full recipe (theta_G/sigma
# conversion, shape-noise formula, kappa_rms convention, seeding).
#
#   sbatch papers/01_pipeline/run_noisy_cache.sbatch          <- DO NOT SUBMIT for me;
#                                                                 the user runs this.
#
# SIZING (measured on this workstation, 1 core; cca cores may differ by an O(1)
# factor but not in kind):
#   - per (realization, plane): noise+smooth 0.12 s, peak/minima extrema 0.29 s,
#     Minkowski functionals 0.81 s (the dominant cost -- several np.gradient passes
#     per 1024^2 map), cl_kappa power spectrum 0.07 s  ->  ~1.3 s/unit.
#   - decompressing one fid/truth/dmo kappa_maps.npz (10.67 GB compressed, ~11.5 GB
#     of float32) costs ~110 s REGARDLESS of how many realizations a chunk task
#     keeps (savez_compressed gives no random access -- np.load decompresses the
#     whole member); a Sobol run's ~1 GB cube costs ~10 s.
#   - Sobol run (50 real x 5 planes = 250 units): 10 + 250*1.3 ~= 335 s (~5.6 min).
#     256 runs serial ~= 23.9 h.
#   - fid/truth/dmo (550 real x 5 planes = 2750 units) in ONE task would be
#     110 + 2750*1.3 ~= 3685 s (~61 min) -- over the ~30 min/task budget, so each
#     target is split into 4 realization-range chunks of ~138 real:
#     110 + 138*5*1.3 ~= 1007 s (~16.8 min) per chunk, comfortably under 30 min.
#     3 targets x 4 chunks = 12 heavy tasks; plus ONE kappa_rms pass over the full
#     550-real fiducial cube (noise+smooth+std only, no extrema/MF/Cl):
#     110 + 550*5*0.12 ~= 448 s (~7.5 min), run FIRST and standalone (every other
#     task's nu normalisation depends on its output).
#   - total serial compute ~= 7.5 min (kappa_rms) + 12*16.8 min (heavy chunks) +
#     256*5.6 min (Sobol) ~= 27.0 h. Over --ntasks=40 concurrent slots that is
#     ~40-60 min wall time (plus module/merge/assemble overhead) -- hence
#     --time=03:00:00 for margin on a shared, first-run job.
#   - peak memory: a heavy chunk task holds one full ~11.5 GB kappa array (chunking
#     saves compute time, NOT decompression memory, since np.load has no partial-
#     read for a compressed npz); a Sobol task holds ~1.1 GB. Worst case (all 12
#     heavy tasks land in the same wave) ~= 12*11.5 + 28*1.1 ~= 169 GB, well under
#     --mem=450G.
#
# Nothing released, and nothing in nu05_shards/ (the noiseless cache), is touched.
# Shards land in bind_sb35/nu05n_shards/; the assembled dataset is a NEW file,
# emulator_dataset_nu05n.npz.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND/papers/01_pipeline
mkdir -p /mnt/home/mlee1/ceph/logs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module -q load modules/2.4-20250724
module -q load disBatch

# ── Step 0: kappa_rms, standalone, BEFORE anything else (refuse-don't-guess: every
# other task's nu normalisation reads this cache and raises if it is missing). ──
echo "[kappa_rms] building from the full 550-real fiducial suite"
python build_noisy_cache.py --kappa-rms

# ── Step 1: build the disBatch task list — 12 heavy realization-range chunks
# (fid/truth/dmo, chunk count/size derived from each cube's ACTUAL realization
# count, not assumed to be 550) + 256 Sobol runs. ──
TASKS=$(mktemp /tmp/noisy_cache_tasks.XXXXXX)
trap 'rm -f "$TASKS"' EXIT

N_CHUNKS=4
python - "$N_CHUNKS" <<'PY' >> "$TASKS"
import sys
import numpy as np
import noisy_grid as ng

n_chunks = int(sys.argv[1])
for target in ("fid", "truth", "dmo"):
    n_real = ng.n_real_of(ng.kappa_path(target))
    bounds = np.linspace(0, n_real, n_chunks + 1, dtype=int)
    for s, e in zip(bounds[:-1], bounds[1:]):
        print(f"cd /mnt/home/mlee1/BIND/papers/01_pipeline && "
              f"python build_noisy_cache.py --run {target} "
              f"--real-start {s} --real-end {e}")

d = np.load(ng.DATASET_IN, allow_pickle=True)
for r in d["run_ids"]:
    print(f"cd /mnt/home/mlee1/BIND/papers/01_pipeline && "
          f"python build_noisy_cache.py --run run_{int(r):04d}")
PY

echo "[disBatch] $(wc -l < "$TASKS") tasks over ${SLURM_NTASKS} slots"
disBatch -e "$TASKS"          # -e: exit non-zero if any task failed

# ── Step 2: stitch the fid/truth/dmo chunks back into one shard each, then
# assemble the Sobol-only dataset and verify it. ──
for target in fid truth dmo; do
    python build_noisy_cache.py --merge-chunks "$target"
done
python build_noisy_cache.py --assemble
python build_noisy_cache.py --verify
echo "[done] /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_nu05n.npz"
echo "[done] /mnt/home/mlee1/ceph/bind_sb35/nu05n_shards/{fid,truth,dmo}.npz"
