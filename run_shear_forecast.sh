#!/bin/bash
#SBATCH --job-name=bind_shear_forecast
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_shear_forecast_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_shear_forecast_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Full-likelihood LSST-Y1 cosmic-shear forecast with a BIND baryon nuisance ──
# Production (converged) emcee chain for examples/lightcone_shear_forecast.py.
# CPU only (CCL + emcee).  The likelihood is CCL-halofit-bound (~0.26 s/eval),
# so this parallelises across cores via emcee's multiprocessing Pool.
#
# 48 walkers x 8000 steps on 32 cores ~ a few hours; tune --steps to taste and
# check the acceptance fraction / autocorr in the printed summary.  Writes
# examples/figures_lightcone/shear_forecast_y1.npz.

set -euo pipefail
cd /mnt/home/mlee1/BIND
source .venv/bin/activate 2>/dev/null || true   # adjust to your venv activation

export OMP_NUM_THREADS=1                          # let emcee own the parallelism
python examples/lightcone_shear_forecast.py \
    --walkers 48 --steps 8000 --burn 2000 --cores "${SLURM_CPUS_PER_TASK:-32}"
