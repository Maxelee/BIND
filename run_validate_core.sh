#!/bin/bash
#SBATCH --job-name=val_core
#SBATCH --output=/mnt/home/mlee1/ceph/logs/val_core_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/val_core_%j.err
#SBATCH --time=00:40:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Is the core-weighted fine-tune sharpening the cores yet? Generates patches with
# baseline (epoch047) + the in-progress fm_two_head_core/last.ckpt and reports
# T(k), core concentration, mass, KS vs truth. Read-only; safe to run repeatedly
# while training runs.
#   sbatch run_validate_core.sh
#   CORE_CKPT=last.ckpt N_TEST=300 sbatch run_validate_core.sh

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
export LD_LIBRARY_PATH=/mnt/sw/nix/store/1fycpximqhwabw5a9z5myspx61lpx8gi-gcc-11.5.0/lib64:${LD_LIBRARY_PATH:-}
cd /mnt/home/mlee1/vdm_bind2

python validate_core_finetune.py \
    --n_test "${N_TEST:-200}" \
    --n_steps "${N_STEPS:-50}" \
    --baseline_ckpt "${BASELINE_CKPT:-epoch047-val_loss0.2138.ckpt}" \
    --core_ckpt "${CORE_CKPT:-last.ckpt}" \
    --out "${OUT:-ceph/fm_diag/core_finetune_progress.png}"
