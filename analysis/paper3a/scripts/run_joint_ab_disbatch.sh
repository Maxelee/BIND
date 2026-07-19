#!/bin/bash
#SBATCH --job-name=joint_ab
#SBATCH --output=/mnt/home/mlee1/ceph/logs/joint_ab_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/joint_ab_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── JOINT_AB_PLAN step 4: recovery battery + THE joint A+B fit, one job ─────
# Stage 1: disBatch fans the 16 battery injections (32-dim: astro + f_sat +
#          sigma_pos; synthetic data through fgas + kSZ + B covariances).
# Stage 2: assemble + score the battery (amended binomial + KS criteria).
# Stage 3: disBatch runs the 4 data-fit seeds (DE moves, 128 x 30k); every
#          fit task re-checks the battery gate itself, so a FAILED battery
#          stops the fits, not the job.
# Stage 4: assemble (cross-chain R-hat, per-block MAP chi2, the
#          (dln M_gas, dln T) posterior, sigma_pos posterior, B
#          posterior-predictive p).
#
# NOTE before submitting: if the 62-unit model_grid_tfwiener.npz (with the
# bind anchor) has been re-rsync'd, the B block picks it up automatically —
# re-run `python -m analysis.paper3a.inference.bblock` once to confirm the
# chi2 anchors still reproduce, then submit.
#
# ⛔ Submit:  sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_joint_ab_disbatch.sh

set -euo pipefail

module load gcc disBatch
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2-paper3a
mkdir -p /mnt/home/mlee1/ceph/logs/joint_ab

echo "=== stage 0: B-block validation snapshot (recorded in the job log) ==="
python -u -m analysis.paper3a.inference.bblock | tail -40

echo "=== stage 1: joint recovery battery (16 tasks) ==="
disBatch -p /mnt/home/mlee1/ceph/logs/joint_ab/db_battery_ \
    analysis/paper3a/scripts/joint_ab_battery.disbatch

echo "=== stage 2: assemble + score the battery ==="
python -u analysis/paper3a/scripts/run_joint_ab_recovery.py --assemble

echo "=== stage 3: the joint fit (4 seeds) ==="
disBatch -p /mnt/home/mlee1/ceph/logs/joint_ab/db_fits_ \
    analysis/paper3a/scripts/joint_ab_fits.disbatch

echo "=== stage 4: assemble the fit ==="
python -u analysis/paper3a/scripts/run_joint_ab_fit.py --assemble

echo "DONE"
