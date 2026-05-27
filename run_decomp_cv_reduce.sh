#!/bin/bash
#SBATCH --job-name=decomp_cv_reduce
#SBATCH --output=/mnt/home/mlee1/ceph/logs/decomp_cv_reduce_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/decomp_cv_reduce_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Reduce step: concatenate the per-chunk partial cubes (over the halo axis) and run the
# decomposition + Sobol + plots. CPU-only — no GPU, no model. Run after the generate array
# finishes (or submit with: sbatch --dependency=afterok:<gen_array_jobid> run_decomp_cv_reduce.sh).
#
#   MODE=axes  sbatch run_decomp_cv_reduce.sh
#   MODE=joint sbatch run_decomp_cv_reduce.sh

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
export PYTHONUNBUFFERED=1

MODE=${MODE:-axes}
N_BOOT=${N_BOOT:-200}

mkdir -p /mnt/home/mlee1/ceph/logs outputs/scatter_diagnostics figures/scatter_diagnostics
echo "=== CV reduce: mode=$MODE ==="
python -m scatter.scatter_decomposition \
    --base cv --phase reduce --mode "$MODE" --n-boot "$N_BOOT"
echo "=== reduce done ==="
echo "  -> outputs/scatter_diagnostics/scatter_decomposition${MODE:+_}*_cv.{json,npz}"
