#!/bin/bash
#SBATCH --job-name=refiner_adv
#SBATCH --output=/mnt/home/mlee1/ceph/logs/refiner_adv_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/refiner_adv_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Adversarial BIND patch refiner (PatchGAN). Writes ceph/refiner_two_head_adv/
# refiner.pt -- a NEW dir, no training run / weights touched.
#   sbatch run_refiner_adv.sh
#   LAMBDA_ADV=0.2 EPOCHS=120 sbatch run_refiner_adv.sh

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
export LD_LIBRARY_PATH=/mnt/sw/nix/store/1fycpximqhwabw5a9z5myspx61lpx8gi-gcc-11.5.0/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_KERNEL_CACHE_PATH="${TMPDIR:-$HOME/.cache}/torch_kernels"; mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"
cd /mnt/home/mlee1/vdm_bind2

python -u refiner_train_adv.py \
    --epochs "${EPOCHS:-80}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --lambda_l1 "${LAMBDA_L1:-0.3}" \
    --lambda_spec "${LAMBDA_SPEC:-2.0}" \
    --lambda_fm "${LAMBDA_FM:-5.0}" \
    --lambda_adv "${LAMBDA_ADV:-0.0}" \
    --adv_warmup "${ADV_WARMUP:-5}" \
    --disc_channels "${DISC_CHANNELS:-1,2}" \
    --lr "${LR:-2e-4}"
