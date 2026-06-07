#!/bin/bash
#SBATCH --job-name=fd_sk_box
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fd_sk_box_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fd_sk_box_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-6   # 7 shards x 5 params each = 35 params

# Local finite-difference dS(k)/dtheta of the BOX matter-power suppression at the CV
# fiducial -- the WL-target side of the gas->S(k) chain on rigorous local-FD footing.
# Generates fm_thermo mass-channel patches at theta_fid+-Delta (fixed noise) and reuses
# box_supp_sobol paste+Pk (validated to reproduce S_true exactly). Submit; then merge.

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
export PYTHONUNBUFFERED=1

N_CHUNKS=${N_CHUNKS:-7}
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}
RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_thermo}
OUT_DIR=${OUT_DIR:-/mnt/home/mlee1/vdm_bind2/analysis_physics_cache}
N_STEPS=${N_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-32}
EPS=${EPS:-1e-3}

mkdir -p "$OUT_DIR" /mnt/home/mlee1/ceph/logs
OUTPUT="$OUT_DIR/fd_sk_box_shard${CHUNK_ID}.npz"
echo "=== [shard $CHUNK_ID/$N_CHUNKS] -> $OUTPUT ==="

python tools/fd_sk_box.py \
    --run_dir "$RUN_DIR" --output "$OUTPUT" \
    --n_chunks "$N_CHUNKS" --chunk_id "$CHUNK_ID" \
    --n_steps "$N_STEPS" --batch_size "$BATCH_SIZE" --eps "$EPS"

echo "=== [shard $CHUNK_ID] done ==="

# After all 7 array tasks finish, merge:
#   python tools/fd_sk_box.py --merge \
#       --shard_glob "$OUT_DIR/fd_sk_box_shard*.npz" \
#       --output     "$OUT_DIR/fd_sk_box_fm_thermo.npz"
