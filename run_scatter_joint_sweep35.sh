#!/bin/bash
#SBATCH --job-name=joint_sweep35
#SBATCH --output=/mnt/home/mlee1/ceph/logs/joint_sweep35_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/joint_sweep35_%j.err
#SBATCH --time=2:30:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Full 35-parameter joint scatter structure sweep.
# Runs BIND inference at the lo/hi 1P endpoints for each CAMELS SB35 parameter
# and computes the Spearman-correlation ΔC_T vs ΔC_G comparison.
#
# Arms 1-6 (Omega_m, sigma8, A_SN1, A_AGN1, A_SN2, A_AGN2) are loaded from
# existing caches automatically; arms 7-35 run fresh (~3 min each, ~90 min total).
#
# Outputs:
#   outputs/scatter_joint_structure/sweep35_arm_{param}.npz  (per-arm cache)
#   outputs/scatter_joint_structure/sweep35_results.npz      (aggregated)
#   outputs/scatter_joint_structure/REPORT35.md
#   figures/scatter_joint_structure/fig_sweep35_headline.pdf/.png
#
# Usage:
#   sbatch run_scatter_joint_sweep35.sh          # fresh run (uses --resume)
#   sbatch run_scatter_joint_sweep35.sh --start N  # resume from arm N (1-based)

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

export PYTHONUNBUFFERED=1

mkdir -p outputs/scatter_joint_structure figures/scatter_joint_structure \
         /mnt/home/mlee1/ceph/logs

# Pass any extra CLI args (e.g. --start N) straight through
EXTRA_ARGS="${@}"

echo "=== joint_struct_sweep_35.py START (job ${SLURM_JOB_ID:-local}) ==="
python scatter/joint_struct_sweep_35.py --resume ${EXTRA_ARGS}
echo "=== joint_struct_sweep_35.py DONE ==="
