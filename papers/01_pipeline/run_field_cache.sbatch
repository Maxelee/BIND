#!/bin/bash
#SBATCH --job-name=field_cache
#SBATCH --output=/mnt/home/mlee1/ceph/logs/field_cache_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/field_cache_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=01:00:00

# ── Per-realization field statistics for Paper I figs 4 and 4b ────────────────
# Both figures currently rebuild, from the raw 1024^2 cubes on EVERY notebook run:
# ~750 power_spectrum calls (auto + cross, 50 realizations x 5 source planes),
# peak/minima counts and the kappa PDF, plus five ~1 GB npz decompressions. This
# builds all of it once into ~15 MB of per-realization arrays.
#
# 10 completely independent tasks — (product, side) — packed into ONE allocation by
# disBatch rather than submitted as 10 jobs.
#
#   sbatch papers/01_pipeline/run_field_cache.sbatch
#
# --mem is 300G because up to 10 tasks decompress a ~1 GB cube concurrently (the
# cross tasks hold two). Serial fallback, ~10-15 min:
#   for t in "kk bind" "kk truth" "yy bind" "yy truth" "ky bind" "ky truth" \
#            "tt bind" "kt bind" "counts bind" "counts truth"; do
#     set -- $t; python build_field_cache.py --product $1 --side $2; done
#   python build_field_cache.py --merge

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND/papers/01_pipeline
mkdir -p /mnt/home/mlee1/ceph/logs

# disBatch comes from Lmod, not the venv — load AFTER activating so the venv python
# stays first on PATH for the tasks themselves.
module -q load modules/2.4-20250724
module -q load disBatch

# Gate: the recomputation must reproduce caches built by a DIFFERENT pipeline
# (paired_perreal_fid.npz on the bind side, peak_counts.npz on truth) before we
# spend the allocation. A convention drift would otherwise move every curve in fig 4.
echo "[verify] recomputation vs independently-built caches"
python build_field_cache.py --verify

TASKS=$(mktemp /tmp/field_cache_tasks.XXXXXX)
trap 'rm -f "$TASKS"' EXIT
for t in "kk bind" "kk truth" "yy bind" "yy truth" "ky bind" "ky truth" \
         "tt bind" "kt bind" "counts bind" "counts truth"; do
    set -- $t
    echo "cd /mnt/home/mlee1/BIND/papers/01_pipeline && python build_field_cache.py --product $1 --side $2" >> "$TASKS"
done

echo "[disBatch] 10 tasks over ${SLURM_NTASKS} slots"
disBatch -e "$TASKS"          # -e: exit non-zero if any task failed

python build_field_cache.py --merge
echo "[done] cache at /mnt/home/mlee1/ceph/bind_science/field_cache/field_stats_fid.npz"
