#!/bin/bash
#SBATCH --job-name=bind_perhalo_sbi
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_perhalo_sbi_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_perhalo_sbi_%j.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Per-halo SBI: train the amortized NPE q(theta | x_halo) and sample the ─────
# population posterior at the TNG300 fiducial (the 30-param "money" corner).
# Single GPU task; ~8.6M (theta, Y_200) pairs reduced to a few×10^5 by --subsample.
#
# Env overrides: OUT (results dir), SUBSAMPLE, OBSERVABLES, SNAPS, MCMC_STEPS.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND/examples
mkdir -p /mnt/home/mlee1/ceph/logs

OUT="${OUT:-/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/perhalo_sbi_results}"
SUBSAMPLE="${SUBSAMPLE:-1500000}"
MCMC_STEPS="${MCMC_STEPS:-3000}"
OBSERVABLES="${OBSERVABLES:-logM200 a logY200}"

python train_perhalo_sbi.py \
    --device cuda \
    --out "$OUT" \
    --subsample "$SUBSAMPLE" \
    --mcmc_steps "$MCMC_STEPS" \
    --observables $OBSERVABLES \
    ${SNAPS:+--snaps $SNAPS}

echo "perhalo SBI done -> $OUT"
