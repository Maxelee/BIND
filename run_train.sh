#!/bin/bash
#SBATCH --job-name=fm_train
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm_train_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm_train_%j.err
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

srun python train.py \
    --data_root /mnt/home/mlee1/ceph/train_data_rotated2_128_cpu \
    --batch_size 64 \
    --num_workers 8 \
    --base_ch 128 \
    --n_blocks 2 \
    --emb_dim 512 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --interpolant fm \
    --lr 1e-4 \
    --max_epochs 200 \
    --output_dir /mnt/home/mlee1/ceph/fm_runs \
    --run_name fm_v2
