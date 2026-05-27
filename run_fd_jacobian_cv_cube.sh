#!/bin/bash
#SBATCH --job-name=fd_jac_cv_cube
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fd_jac_cv_cube_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fd_jac_cv_cube_%A_%a.err
#SBATCH --time=6:00:00
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

# Force unbuffered stdout so SLURM .out files update in real time
export PYTHONUNBUFFERED=1

# ── Configuration ─────────────────────────────────────────────────────────────
N_CHUNKS=${N_CHUNKS:-7}
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_cube_two_head}
CV_ROOT=${CV_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite_cube/CV}

OUT_DIR=${OUT_DIR:-/mnt/home/mlee1/vdm_bind2/analysis_physics_cache}
SHARD_PREFIX=${SHARD_PREFIX:-proj6_cv_fd_scatter_cube_shard}

N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
EPS=${EPS:-1e-3}
MAX_HALOS=${MAX_HALOS:-}     # blank = use all CV halos
NOISE_SEED=${NOISE_SEED:-42}
SUBSET_SEED=${SUBSET_SEED:-0}

mkdir -p "$OUT_DIR" /mnt/home/mlee1/ceph/logs

EXTRA=()
[[ -n "$MAX_HALOS" ]] && EXTRA+=(--max_halos "$MAX_HALOS")

OUTPUT="$OUT_DIR/${SHARD_PREFIX}${CHUNK_ID}.npz"

echo "=== [shard $CHUNK_ID/$N_CHUNKS] writing $OUTPUT ==="

python fd_jacobian_cv.py \
    --run_dir "$RUN_DIR" \
    --cv_root "$CV_ROOT" \
    --output "$OUTPUT" \
    --n_chunks "$N_CHUNKS" \
    --chunk_id "$CHUNK_ID" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --eps "$EPS" \
    --noise_seed "$NOISE_SEED" \
    --subset_seed "$SUBSET_SEED" \
    --cube \
    "${EXTRA[@]}"

echo "=== [shard $CHUNK_ID] done ==="

# ── Merge: launch ONE final task with --dependency=afterok:$JOBID ────────────
# (or run by hand once all array tasks complete)
#
#   python fd_jacobian_cv.py --merge \
#       --shard_glob "$OUT_DIR/${SHARD_PREFIX}*.npz" \
#       --output     "$OUT_DIR/proj6_cv_fd_scatter_cube_fm_cube_two_head.npz"
