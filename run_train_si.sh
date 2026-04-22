#!/bin/bash
#SBATCH --job-name=si3d_train
#SBATCH --output=/mnt/home/mlee1/ceph/logs/si3d_train_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/si3d_train_%j.err
#SBATCH --time=48:00:00
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

# Diagnostic: one line per rank showing task-to-GPU mapping.
srun --ntasks-per-node=4 --gpus-per-node=4 bash -lc 'echo "diag rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"'

srun python train_3d.py \
    --data_root /mnt/home/mlee1/ceph/train_data_1024/train_3d \
    --batch_size 1 \
    --num_workers 8 \
    --crop_size 128 \
    --n_stats_samples 256 \
    --base_ch 16 \
    --n_blocks 2 \
    --emb_dim 256 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --lr 1e-4 \
    --max_epochs 200 \
    --interpolant si \
    --sigma 0.5 \
    --output_dir /mnt/home/mlee1/ceph/fm_runs_3d \
    --run_name si3d_sigma05
