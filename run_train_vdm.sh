#!/bin/bash
#SBATCH --job-name=vdm_train
#SBATCH --output=/mnt/home/mlee1/ceph/logs/vdm_train_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/vdm_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G

# Variational diffusion / score-matching emulator on the SAME UNet/data/hparams as
# fm_two_head. The ONLY differences vs run_train.sh: --interpolant vdm (eps-prediction
# score matching instead of flow matching) and NO --stars_two_head (the two-head
# stellar split breaks the Gaussian diffusion forward process -> single-head 3ch).
# Fresh per-run norm_stats (single-head) is computed automatically in the run dir.
#
#   sbatch run_train_vdm.sh
#   RUN_NAME=vdm MAX_EPOCHS=200 sbatch run_train_vdm.sh
#
# Goal: clean VDM-vs-FM comparison (expectation: VDM nails small-scale P(k) but shows
# a stellar mass bias -> then fixed by per-channel mass calibration).

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
export LD_LIBRARY_PATH=/mnt/sw/nix/store/1fycpximqhwabw5a9z5myspx61lpx8gi-gcc-11.5.0/lib64:${LD_LIBRARY_PATH:-}
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

DATA_ROOT=${DATA_ROOT:-/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/fm_runs}
RUN_NAME=${RUN_NAME:-vdm}
MAX_EPOCHS=${MAX_EPOCHS:-200}

echo "=== VDM training (score matching, single-head) run_name=$RUN_NAME ==="

srun python -m bind.train \
    --data_root "$DATA_ROOT" \
    --batch_size 64 \
    --num_workers 8 \
    --base_ch 128 \
    --n_blocks 2 \
    --emb_dim 512 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --interpolant vdm \
    --lr 1e-4 \
    --max_epochs "$MAX_EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    "$@"
