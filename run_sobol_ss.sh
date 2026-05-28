#!/bin/bash
#SBATCH --job-name=sobol_ss
#SBATCH --output=/mnt/home/mlee1/ceph/logs/sobol_ss_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/sobol_ss_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-15   # set to 0-(N_CHUNKS-1); override with --array at sbatch time

# Sobol map of the self-similar-residual -> matter-suppression calibration.
# Regenerates the ~1154 CV halos across a Sobol grid in the 30 astro params
# (cosmology fixed at the CV fiducial) so fig_money can be refit per grid point.
#
# Each array task recomputes the identical Sobol design from SEED and processes
# only its --chunk_id slice (design_ids = range(N_DESIGN)[chunk_id::N_CHUNKS]),
# so there is NO manifest/lock step. Outputs (per-design map + obs shards) land
# on ceph and are resumable (a design with both shards present is skipped).
#
# Submit examples (this script cannot sbatch itself):
#   16-way H100 array (default):
#     N_CHUNKS=16 sbatch --array=0-15 run_sobol_ss.sh
#   then reduce once the array finishes (CPU, seconds):
#     python sobol_ss_generation.py --reduce --out_root /mnt/home/mlee1/ceph/sobol_ss_cv
# Useful overrides: SEED, N_DESIGN, N_STEPS, BATCH_SIZE, OUT_ROOT, FP16=1, REGEN=1.

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

N_CHUNKS=${N_CHUNKS:-16}                    # must match --array upper bound + 1
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

SEED=${SEED:-12345}
N_DESIGN=${N_DESIGN:-256}
N_STEPS=${N_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_ROOT=${OUT_ROOT:-/mnt/home/mlee1/ceph/sobol_ss_cv}
RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_thermo}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$RUN_DIR/checkpoints/kept/keep_epoch064_ema.ckpt}

mkdir -p "$OUT_ROOT" /mnt/home/mlee1/ceph/logs

EXTRA_FLAGS=()
[[ "${FP16:-0}" == "1" ]]  && EXTRA_FLAGS+=(--fp16)
[[ "${REGEN:-0}" == "1" ]] && EXTRA_FLAGS+=(--regenerate)

echo "=== [chunk $CHUNK_ID/$N_CHUNKS] seed=$SEED n_design=$N_DESIGN ckpt=$CHECKPOINT_PATH ==="

python sobol_ss_generation.py \
    --out_root "$OUT_ROOT" \
    --run_dir "$RUN_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --seed "$SEED" \
    --n_design "$N_DESIGN" \
    --n_chunks "$N_CHUNKS" \
    --chunk_id "$CHUNK_ID" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "=== [chunk $CHUNK_ID] Done ==="
