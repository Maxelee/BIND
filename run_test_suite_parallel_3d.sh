#!/bin/bash
#SBATCH --job-name=fm3d_testsuite_par
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm3d_testsuite_par_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm3d_testsuite_par_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-9   # set to 0-(N_CHUNKS-1); override with --array at sbatch time

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

# ── Configuration ─────────────────────────────────────────────────────────────
# N_CHUNKS must match the --array upper bound + 1 (e.g. --array=0-9 → N_CHUNKS=10)
N_CHUNKS=${N_CHUNKS:-10}
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs_3d/fm3d_two_head_v2}
MODEL_NAME=${MODEL_NAME:-fm3d_two_head_v2}

# Optional: set CHECKPOINT_PATH to use a specific .ckpt (defaults to last.ckpt)
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}

DATA_ROOT=${DATA_ROOT:-/mnt/home/mlee1/ceph/train_data_1024/train_3d}
SPLIT=${SPLIT:-test}

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/fm3d_testsuite}

N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
DEVICE=${DEVICE:-auto}

mkdir -p "$OUTPUT_ROOT" /mnt/home/mlee1/ceph/logs

# ── Build extra flags ─────────────────────────────────────────────────────────
EXTRA_FLAGS=()
[[ "${REGENERATE:-0}" == "1" ]] && EXTRA_FLAGS+=(--regenerate)
[[ "${NO_AMP:-0}"     == "1" ]] && EXTRA_FLAGS+=(--no_amp)
[[ -n "${CHECKPOINT_PATH}"   ]] && EXTRA_FLAGS+=(--checkpoint_path "$CHECKPOINT_PATH")

echo "=== [chunk $CHUNK_ID/$N_CHUNKS] Running 3D generation ==="

/mnt/home/mlee1/venvs/torch3/bin/python run_test_suite_3d.py \
    --data_root    "$DATA_ROOT" \
    --split        "$SPLIT" \
    --run_dir      "$RUN_DIR" \
    --model_name   "$MODEL_NAME" \
    --output_root  "$OUTPUT_ROOT" \
    --n_steps      "$N_STEPS" \
    --batch_size   "$BATCH_SIZE" \
    --device       "$DEVICE" \
    --n_chunks     "$N_CHUNKS" \
    --chunk_id     "$CHUNK_ID" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "=== [chunk $CHUNK_ID] Done ==="
