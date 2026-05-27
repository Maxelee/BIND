#!/bin/bash
#SBATCH --job-name=scatter_robust
#SBATCH --output=/mnt/home/mlee1/ceph/logs/scatter_robust_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/scatter_robust_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Runs all 5 Phase-5 robustness checks sequentially.
# Check 2 (mass bins) and check 1 K=20 pass are expensive (~4 hr each).
# Run after J_mean_and_scatter.npz exists.
#
#   sbatch --dependency=afterok:${MERGE_JOBID} run_scatter_robustness.sh

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

export PYTHONUNBUFFERED=1

mkdir -p scatter/robustness /mnt/home/mlee1/ceph/logs

echo "=== Phase 5: Robustness checks ==="
python scatter/robustness/run_all_robustness.py

echo "=== Robustness complete — see scatter/robustness/SUMMARY.md ==="
