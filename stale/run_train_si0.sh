#!/bin/bash
#SBATCH --job-name=fm2d_si_sigma0
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm2d_si_sigma0_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm2d_si_sigma0_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

mkdir -p /mnt/home/mlee1/ceph/logs

# Deterministic stochastic-interpolant 2D run: sigma=0 makes the bridge
# fully deterministic (DMO -> hydro, no noise injection along the path).
# Mirrors run_train_param_norm.sh so the only difference vs the FM
# baseline is the source distribution (DMO field instead of Gaussian
# noise). The earlier sigma=0.5 SI run trained poorly; this isolates
# whether the issue was the source distribution or the noise injection.
srun python train.py \
    --data_root /mnt/home/mlee1/ceph/train_data_rotated2_128_cpu \
    --batch_size 64 \
    --num_workers 8 \
    --base_ch 128 \
    --n_blocks 2 \
    --emb_dim 512 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --interpolant si \
    --sigma 0.0 \
    --lr 1e-4 \
    --max_epochs 200 \
    --output_dir /mnt/home/mlee1/ceph/fm_runs \
    --run_name si_sigma0
