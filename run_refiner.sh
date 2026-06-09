#!/bin/bash
#SBATCH --job-name=refiner
#SBATCH --output=/mnt/home/mlee1/ceph/logs/refiner_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/refiner_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Train the BIND patch refiner (small residual CNN, L1 + spectral loss).
# Writes ceph/refiner_two_head/refiner.pt -- a NEW dir, no training run touched.
#   sbatch run_refiner.sh
#   LAMBDA_SPEC=3 EPOCHS=80 sbatch run_refiner.sh   # if high-k under-corrected

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
export LD_LIBRARY_PATH=/mnt/sw/nix/store/1fycpximqhwabw5a9z5myspx61lpx8gi-gcc-11.5.0/lib64:${LD_LIBRARY_PATH:-}
cd /mnt/home/mlee1/vdm_bind2

python -u refiner_train.py \
    --epochs "${EPOCHS:-20}" \
    --batch_size "${BATCH_SIZE:-64}" \
    --lambda_spec "${LAMBDA_SPEC:-1000.0}" \
    --lr "${LR:-2e-4}"
