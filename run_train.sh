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
#   Mass only (DM_hydro, Gas, Stars), 35-param conditioning:
#     sbatch run_train.sh
#   Joint mass + 4 gas-thermo fields (compton_y, temperature, entropy, pressure):
#     THERMO=1 sbatch run_train.sh
#   Observable conditioning (mass + thermo output, but conditioned on the
#   aperture-integrated R200 observables instead of the 35 params):
#     OBS=1 sbatch run_train.sh
#
# --predict_thermo appends 4 output channels (with --stars_two_head: out_ch =
# 4 + 4 = 8) and computes a fresh thermo-aware norm_stats.npz on first launch.
# --condition_observables swaps the 35 params for N_OBS observables (and implies
# thermo here, since the outputs should include the thermo fields). Both require
# --interpolant fm and the large-scale (rotated2_128) data path.
#
# Env overrides: RUN_NAME, DATA_ROOT, OUTPUT_DIR, MAX_EPOCHS. Any extra args are
# passed through to bind.train (e.g. `sbatch run_train.sh --exclude_cosmo_params`).

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

THERMO=${THERMO:-0}
OBS=${OBS:-0}
MASK=${MASK:-0}
DATA_ROOT=${DATA_ROOT:-/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/fm_runs}
MAX_EPOCHS=${MAX_EPOCHS:-200}

EXTRA_FLAGS=()
if [[ "$OBS" == "1" ]]; then
    # Observable conditioning, with mass+thermo outputs.
    EXTRA_FLAGS+=(--condition_observables --predict_thermo)
    if [[ "$MASK" == "1" ]]; then
        # Input-dropout: tolerate a missing subset of observables at inference.
        EXTRA_FLAGS+=(--mask_observables)
        RUN_NAME=${RUN_NAME:-fm_observables_masked}
    else
        RUN_NAME=${RUN_NAME:-fm_observables}
    fi
elif [[ "$THERMO" == "1" ]]; then
    EXTRA_FLAGS+=(--predict_thermo)
    RUN_NAME=${RUN_NAME:-fm_thermo}
else
    RUN_NAME=${RUN_NAME:-fm_two_head}
fi

echo "=== training run_name=$RUN_NAME thermo=$THERMO obs=$OBS data=$DATA_ROOT ==="

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
