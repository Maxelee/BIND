#!/bin/bash
#SBATCH --job-name=scatter_fig3
#SBATCH --output=/mnt/home/mlee1/ceph/logs/scatter_fig3_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/scatter_fig3_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Computes the 5×5 (A_SN1, A_AGN1) scatter grid and produces fig3.
# Run after J_mean_and_scatter.npz exists (can run in parallel with merge job).
#
#   sbatch --dependency=afterok:${JAC_JOBID} run_scatter_fig3.sh

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

export PYTHONUNBUFFERED=1

N_GRID=${N_GRID:-5}
K=${K:-10}
N_STEPS=${N_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_HALOS=${MAX_HALOS:-100}

mkdir -p paper_figures/scatter /mnt/home/mlee1/ceph/logs

echo "=== Fig 3: scatter contours (${N_GRID}x${N_GRID} grid, K=$K) ==="
python scatter/figures.py --fig3 \
    --fig3_grid   "$N_GRID" \
    --fig3_K      "$K" \
    --fig3_steps  "$N_STEPS" \
    --fig3_batch  "$BATCH_SIZE" \
    --fig3_halos  "$MAX_HALOS"

echo "=== Fig 3 complete ==="
