#!/bin/bash
#SBATCH --job-name=sobol_explorer
#SBATCH --output=/mnt/home/mlee1/ceph/logs/sobol_explorer_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/sobol_explorer_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

mkdir -p /mnt/home/mlee1/ceph/logs

OUT_DIR=/mnt/home/mlee1/ceph/website_assets
mkdir -p "$OUT_DIR"

python generate_sobol_explorer.py \
    --n_samples 1024 \
    --n_1p_steps 32 \
    --n_steps 50 \
    --batch_size 32 \
    --img_px 128 \
    --cache_npz "$OUT_DIR/sobol_generated.npz" \
    --output "$OUT_DIR/halo_explorer.html"

echo "Done. Output: $OUT_DIR/halo_explorer.html"
