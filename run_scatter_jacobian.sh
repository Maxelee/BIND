#!/bin/bash
#SBATCH --job-name=scatter_jac
#SBATCH --output=/mnt/home/mlee1/ceph/logs/scatter_jac_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/scatter_jac_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-6   # 7 shards × 5 params each = 35 params total

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

export PYTHONUNBUFFERED=1

# ── Configuration ──────────────────────────────────────────────────────────────
N_CHUNKS=7
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

EPS=${EPS:-0.05}
K=${K:-5}
N_STEPS=${N_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
NOISE_SEED=${NOISE_SEED:-42}
SUBSET_SEED=${SUBSET_SEED:-0}

SHARD_OUT="scatter/intermediate/shard_${CHUNK_ID}.npz"

mkdir -p scatter/intermediate /mnt/home/mlee1/ceph/logs

echo "=== [shard $CHUNK_ID/$N_CHUNKS] writing $SHARD_OUT ==="

python scatter/scatter_jacobian.py compute \
    --n_chunks "$N_CHUNKS" \
    --chunk_id "$CHUNK_ID" \
    --output   "$SHARD_OUT" \
    --eps      "$EPS" \
    --K        "$K" \
    --n_steps  "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --noise_seed "$NOISE_SEED" \
    --subset_seed "$SUBSET_SEED"

echo "=== [shard $CHUNK_ID] done ==="

# ── Merge: run once all 7 shards are complete ──────────────────────────────────
# Submit with:
#   sbatch --dependency=afterok:<JOBID> run_scatter_merge.sh
