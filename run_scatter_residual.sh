#!/bin/bash
#SBATCH --job-name=scatter_residual
#SBATCH --output=/mnt/home/mlee1/ceph/logs/scatter_residual_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/scatter_residual_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Runs the BIND scatter-residual cross-correlation analysis end-to-end.
# Uses the pre-cached CV truth + BIND-K10 observables produced by
# scatter/calibration_cv.py; no GPU work needed.
#
#   sbatch run_scatter_residual.sh

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

export PYTHONUNBUFFERED=1

mkdir -p scatter/scatter_residual paper_figures/scatter_residual /mnt/home/mlee1/ceph/logs

echo "=== Phases 1-3: build table, residuals, matrices ==="
python scatter/residual_pipeline.py

echo "=== Phase 4: figures + summary table + writeup ==="
python scatter/residual_figures.py

echo "=== Done ==="
