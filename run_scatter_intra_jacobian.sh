#!/bin/bash
#SBATCH --job-name=scatter_intra_jac
#SBATCH --output=/mnt/home/mlee1/ceph/logs/scatter_intra_jac_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/scatter_intra_jac_%A_%a.err
#SBATCH --time=1:30:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-6   # 7 shards × 5 params each = 35 params total

# Phase 1 diagnostic: J_log_sigma_intra Jacobian
#
# Uses IDENTICAL K, eps, noise_seed, subset_seed as the afternoon
# scatter_jacobian.py run so both Jacobians are directly comparable.
#
# Submit:
#   INTRA_JOBID=$(sbatch --parsable run_scatter_intra_jacobian.sh)
#   sbatch --dependency=afterok:${INTRA_JOBID} run_scatter_intra_merge.sh

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

export PYTHONUNBUFFERED=1

# ── Configuration — must match the afternoon scatter_jacobian.sh run ──────────
N_CHUNKS=7
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

EPS=${EPS:-0.05}
K=${K:-5}
N_STEPS=${N_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
NOISE_SEED=${NOISE_SEED:-42}
SUBSET_SEED=${SUBSET_SEED:-0}

SHARD_OUT="scatter/intermediate_intra/shard_${CHUNK_ID}.npz"
INT_DIR="scatter/intermediate_intra"

mkdir -p scatter/intermediate_intra /mnt/home/mlee1/ceph/logs

echo "=== [intra_jac shard $CHUNK_ID/$N_CHUNKS] writing $SHARD_OUT ==="

python scatter/scatter_intra_jacobian.py compute \
    --n_chunks    "$N_CHUNKS" \
    --chunk_id    "$CHUNK_ID" \
    --output      "$SHARD_OUT" \
    --int_dir     "$INT_DIR" \
    --eps         "$EPS" \
    --K           "$K" \
    --n_steps     "$N_STEPS" \
    --batch_size  "$BATCH_SIZE" \
    --noise_seed  "$NOISE_SEED" \
    --subset_seed "$SUBSET_SEED"

echo "=== [intra_jac shard $CHUNK_ID] done ==="
