#!/bin/bash
#SBATCH --job-name=fd_jac_thermo
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fd_jac_thermo_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fd_jac_thermo_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-6   # 7 shards x 5 params each = 35 params total

# Local finite-difference Jacobian of the THERMO observables (tSZ Y200 + X-ray SX,
# plus the mass/scaling-relation stats) w.r.t. all 35 params at the CV fiducial,
# fixed-noise central differences -- the gas side of the gas->S(k) WL chain on
# rigorous local-FD footing. fm_thermo model (out_channels=8 -> 7 physical).
# Submit:  sbatch run_fd_jacobian_thermo.sh   ; then merge (command at bottom).

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
export PYTHONUNBUFFERED=1

N_CHUNKS=${N_CHUNKS:-7}
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}
RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_thermo}
CV_ROOT=${CV_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite/CV}
OUT_DIR=${OUT_DIR:-/mnt/home/mlee1/vdm_bind2/analysis_physics_cache}
SHARD_PREFIX=${SHARD_PREFIX:-thermo_cv_fd_shard}
N_STEPS=${N_STEPS:-20}        # matches the fm_thermo cube generation
BATCH_SIZE=${BATCH_SIZE:-32}
EPS=${EPS:-1e-3}
MAX_HALOS=${MAX_HALOS:-}      # blank = all CV halos
NOISE_SEED=${NOISE_SEED:-42}
SUBSET_SEED=${SUBSET_SEED:-0}

mkdir -p "$OUT_DIR" /mnt/home/mlee1/ceph/logs
EXTRA=()
[[ -n "$MAX_HALOS" ]] && EXTRA+=(--max_halos "$MAX_HALOS")
OUTPUT="$OUT_DIR/${SHARD_PREFIX}${CHUNK_ID}.npz"
echo "=== [shard $CHUNK_ID/$N_CHUNKS] $RUN_DIR -> $OUTPUT ==="

python tools/fd_jacobian_thermo.py \
    --run_dir "$RUN_DIR" --cv_root "$CV_ROOT" --output "$OUTPUT" \
    --n_chunks "$N_CHUNKS" --chunk_id "$CHUNK_ID" \
    --n_steps "$N_STEPS" --batch_size "$BATCH_SIZE" --eps "$EPS" \
    --noise_seed "$NOISE_SEED" --subset_seed "$SUBSET_SEED" "${EXTRA[@]}"

echo "=== [shard $CHUNK_ID] done ==="

# After all 7 array tasks finish, merge:
#   python tools/fd_jacobian_thermo.py --merge \
#       --shard_glob "$OUT_DIR/thermo_cv_fd_shard*.npz" \
#       --output     "$OUT_DIR/thermo_cv_fd_fm_thermo.npz"
