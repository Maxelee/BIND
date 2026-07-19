#!/bin/bash
#SBATCH --job-name=b4_sb35_hs
#SBATCH --output=/mnt/home/mlee1/ceph/logs/b4_sb35_hs_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/b4_sb35_hs_%A_%a.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-B4 Sobol extension, stage 0: halo_scaling backfill over the SB35 suite ─
# 253 runs strided over 16 array tasks (~16 runs x ~12 min ≈ 3.2 h/task).
# Skip-if-exists → safe to resubmit until complete. Must FINISH before the
# assembly step of the Sobol grid (the grid job itself is independent).
#
# ⛔ HUMAN CHECKPOINT — submit by hand:
#   sbatch /mnt/home/mlee1/BIND-paper3b/analysis/paper3b/scripts/run_b4_sobol_halo_scaling.sh

set -euo pipefail
source /mnt/home/mlee1/venvs/paper3b_popeye/bin/activate
cd /mnt/home/mlee1/BIND-paper3b
mkdir -p /mnt/home/mlee1/ceph/logs

python -u -m analysis.paper3b.scripts.run_b4_sobol_halo_scaling \
    --task-id "${SLURM_ARRAY_TASK_ID}" --n-tasks "${SLURM_ARRAY_TASK_COUNT}"
echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
