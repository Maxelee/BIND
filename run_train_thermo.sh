#!/bin/bash
#SBATCH --job-name=fm2d_thermo
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm2d_thermo_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm2d_thermo_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G

# Joint emulator: mass fields + 4 gas thermo fields (compton_y, temperature,
# entropy, pressure). --predict_thermo appends 4 output channels; combined with
# --stars_two_head this gives out_ch = 4 + 4 = 8 (drop --stars_two_head for
# single-head stars -> out_ch = 3 + 4 = 7). Requires --interpolant fm and the
# large-scale (rotated2_128) data path. Computes a fresh norm_stats.npz with
# thermo stats in the run dir on first launch.

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
    --stars_two_head \
    --predict_thermo \
    --output_dir /mnt/home/mlee1/ceph/fm_runs \
    --run_name fm_thermo
