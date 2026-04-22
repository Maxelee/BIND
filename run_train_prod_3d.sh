#!/bin/bash
#SBATCH --job-name=fm3d_prod
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm3d_prod_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm3d_prod_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

mkdir -p /mnt/home/mlee1/ceph/logs

# Production FM 3D training on full 128^3 volumes.
srun python train_3d.py \
    --data_root /mnt/home/mlee1/ceph/train_data_1024/train_3d \
    --batch_size 1 \
    --num_workers 8 \
    --crop_size 128 \
    --n_stats_samples 2048 \
    --base_ch 16 \
    --n_blocks 2 \
    --emb_dim 256 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --interpolant fm \
    --lr 1e-4 \
    --max_epochs 200 \
    --output_dir /mnt/home/mlee1/ceph/fm_runs_3d \
    --run_name fm3d_prod128
