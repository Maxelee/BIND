#!/bin/bash
#SBATCH --job-name=cv0_sb35_fast
#SBATCH --output=/mnt/home/mlee1/ceph/logs/cv0_sb35_fast_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/cv0_sb35_fast_%j.err
#SBATCH --time=1:30:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

python cv_sim0_sb35_injection_fast.py \
    --cv_sim_id  0 \
    --batch_size 128 \
    --n_steps    50 \
    --device     auto \
    "$@"

echo "=== Done ==="
