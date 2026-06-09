#!/bin/bash
#SBATCH --job-name=fm_core_ft
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm_core_ft_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm_core_ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G

# Fine-tune fm_two_head with the core-weighted loss so the flow stops smoothing the
# sparse co-located cores (the high-k P(k) fix). Loads epoch047 WEIGHTS only
# (--init_from), fresh low-LR schedule + the new --core_weight loss; writes to a NEW
# run dir (fm_two_head_core) so fm_two_head is untouched.
#
#   sbatch run_finetune_core.sh
#   CORE_WEIGHT=5 MAX_EPOCHS=30 sbatch run_finetune_core.sh
#
# Validate after with bind_vs_truth_patches.py (T_stars up?) + the composite S(k).

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
export LD_LIBRARY_PATH=/mnt/sw/nix/store/1fycpximqhwabw5a9z5myspx61lpx8gi-gcc-11.5.0/lib64:${LD_LIBRARY_PATH:-}
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

DATA_ROOT=${DATA_ROOT:-/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/fm_runs}
RUN_NAME=${RUN_NAME:-fm_two_head_core}
INIT_FROM=${INIT_FROM:-/mnt/home/mlee1/ceph/fm_runs/fm_two_head/checkpoints/epoch047-val_loss0.2138.ckpt}
CORE_WEIGHT=${CORE_WEIGHT:-3.0}
CORE_THRESH=${CORE_THRESH:-1.5}
MAX_EPOCHS=${MAX_EPOCHS:-30}
LR=${LR:-2e-5}   # low LR: sharpen cores without relearning goals 1-4

echo "=== core fine-tune: init=$INIT_FROM core_weight=$CORE_WEIGHT lr=$LR -> $RUN_NAME ==="

srun python -m bind.train \
    --data_root "$DATA_ROOT" \
    --batch_size 64 --num_workers 8 \
    --base_ch 128 --n_blocks 2 --emb_dim 512 \
    --dropout 0.1 --cfg_dropout 0.1 \
    --interpolant fm --stars_two_head \
    --lr "$LR" --max_epochs "$MAX_EPOCHS" \
    --core_weight "$CORE_WEIGHT" --core_thresh "$CORE_THRESH" \
    --init_from "$INIT_FROM" \
    --output_dir "$OUTPUT_DIR" --run_name "$RUN_NAME" \
    "$@"
