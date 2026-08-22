#!/bin/bash
#SBATCH --job-name=mf_cache
#SBATCH --output=/mnt/home/mlee1/ceph/logs/mf_cache_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/mf_cache_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=00:40:00

# ── Extended-grid Minkowski cache for Paper I fig 4 ───────────────────────────
# fig 4 needs V0/V1/V2 on nu in [-3, 8]; the released nongaussian_stats.npz only
# caches linspace(-3, 4, 29). Recomputing inline costs ~2 min of compute plus two
# ~1 GB kappa-cube decompressions on EVERY notebook run. This builds the cache once.
#
# The work is 10 completely independent tasks (2 sides x 5 source planes), which is
# exactly disBatch's use case: it packs them into THIS allocation rather than
# creating 10 jobs, so it is cheap for both the user and the scheduler.
#
#   sbatch papers/01_pipeline/run_mf_cache.sbatch
#
# Serial fallback if you would rather not use disBatch (~2-3 min total):
#   for s in bind truth; do for z in 0 1 2 3 4; do
#     python build_mf_cache.py --side $s --plane $z; done; done
#   python build_mf_cache.py --merge

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND/papers/01_pipeline
mkdir -p /mnt/home/mlee1/ceph/logs

# disBatch comes from Lmod, NOT the venv — load it AFTER activating so the venv's
# python stays first on PATH for the tasks themselves.
module -q load modules/2.4-20250724
module -q load disBatch

# Guard: the recomputation must reproduce the RELEASED 29-bin V0/V1/V2 before we
# spend the allocation building a cache on the extended grid. A convention drift
# in nongaussian_stats would otherwise move every MF curve in fig 4 silently.
echo "[verify] recomputation vs released 29-bin grid"
python build_mf_cache.py --verify

TASKS=$(mktemp /tmp/mf_cache_tasks.XXXXXX)
trap 'rm -f "$TASKS"' EXIT
for side in bind truth; do
    for z in 0 1 2 3 4; do
        echo "cd /mnt/home/mlee1/BIND/papers/01_pipeline && python build_mf_cache.py --side $side --plane $z" >> "$TASKS"
    done
done

echo "[disBatch] 10 tasks over ${SLURM_NTASKS} slots"
disBatch -e "$TASKS"          # -e: exit non-zero if any task failed

python build_mf_cache.py --merge
echo "[done] cache at /mnt/home/mlee1/ceph/bind_science/mf_cache/mf_nu8_snap096.npz"
