#!/bin/bash
#SBATCH --job-name=fm_train
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm_train_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G

# Unified training launcher for the 2D flow-matching emulator (8x H100, DDP).
# One script, three modes selected by env toggles:
#
#   Mass only (DM_hydro, Gas, Stars), z=0 10-rotation data:
#     sbatch run_train.sh
#   + 4 gas-thermo fields (compton_y, temperature, entropy, pressure), z=0 data:
#     THERMO=1 sbatch run_train.sh
#   Multi-redshift: mass + thermo + scale-factor conditioning, multi-z data:
#     REDSHIFT=1 sbatch run_train.sh
#
# --predict_thermo appends 4 output channels (with --stars_two_head: out_ch =
# 4 + 4 = 8). --condition_redshift adds scale-factor a=1/(1+z) conditioning.
# REDSHIFT=1 implies thermo (the multi-z dataset always carries thermo channels)
# and points at the multi-z data path. A fresh norm_stats.npz is computed per run.
#
# Env overrides: RUN_NAME, DATA_ROOT, OUTPUT_DIR, MAX_EPOCHS. Any extra args are
# passed through to bind.train (e.g. `sbatch run_train.sh --exclude_cosmo_params`).

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

THERMO=${THERMO:-0}
REDSHIFT=${REDSHIFT:-0}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/fm_runs}
MAX_EPOCHS=${MAX_EPOCHS:-200}

EXTRA_FLAGS=()
if [[ "$REDSHIFT" == "1" ]]; then
    # Multi-redshift dataset always carries thermo channels -> predict both.
    EXTRA_FLAGS+=(--condition_redshift --predict_thermo)
    DATA_ROOT=${DATA_ROOT:-/mnt/home/mlee1/ceph/train_data_multiz_128_cpu}
    RUN_NAME=${RUN_NAME:-fm_redshift}
elif [[ "$THERMO" == "1" ]]; then
    EXTRA_FLAGS+=(--predict_thermo)
    DATA_ROOT=${DATA_ROOT:-/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu}
    RUN_NAME=${RUN_NAME:-fm_thermo}
else
    DATA_ROOT=${DATA_ROOT:-/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu}
    RUN_NAME=${RUN_NAME:-fm_two_head}
fi

echo "=== training run_name=$RUN_NAME thermo=$THERMO redshift=$REDSHIFT data=$DATA_ROOT ==="

srun python -m bind.train \
    --data_root "$DATA_ROOT" \
    --batch_size 64 \
    --num_workers 8 \
    --base_ch 128 \
    --n_blocks 2 \
    --emb_dim 512 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --interpolant fm \
    --lr 1e-4 \
    --max_epochs "$MAX_EPOCHS" \
    --stars_two_head \
    "${EXTRA_FLAGS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    "$@"
