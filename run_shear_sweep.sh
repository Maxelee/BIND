#!/bin/bash
#SBATCH --job-name=bind_shear_sweep
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_shear_sweep_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_shear_sweep_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Step 2: necessity/sufficiency + truth/test sweep ──────────────────────────
# examples/lightcone_shear_sweep.py does TWO things:
#   1. a FAST deterministic linear necessity/sufficiency over ALL 253 runs
#      (-> shear_sweep_linear.npz), the primary quantitative result, ~instant;
#   2. a sampled-posterior truth/test sweep (raw + projection-corrected bias) on a
#      representative weak->strong SUBSET of --nruns runs (-> shear_sweep.{npz,png}).
#
# The MCMC is CCL-halofit-bound (~0.26 s/eval).  IMPORTANT efficiency rule: emcee's
# stretch move evaluates n_walkers/2 logposts per half-step, so --walkers must be
# >= 2*cores for the Pool to be saturated (else half the cores idle).  The sweep
# CHECKPOINTS after every run and RESUMES from shear_sweep.npz, so a preempt/timeout
# only costs the in-flight run.  Cost ~ nruns*3chains*walkers*steps*0.26/cores.
# Tune --nruns up once you've confirmed the wall time on your partition.

set -euo pipefail
cd /mnt/home/mlee1/BIND
source .venv/bin/activate 2>/dev/null || true   # adjust to your venv activation

export OMP_NUM_THREADS=1                          # let emcee own the parallelism
python -u examples/lightcone_shear_sweep.py \
    --nruns 12 --ntmpl 2 --models cosmo,templates --steps 1200 --burn 300 \
    --walkers 128 --cores "${SLURM_CPUS_PER_TASK:-64}"
