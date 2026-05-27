#!/bin/bash
#SBATCH --job-name=cosmo_rescale
#SBATCH --output=/mnt/home/mlee1/ceph/logs/cosmo_rescale_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/cosmo_rescale_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Diagnostic #2: does the emulator rescale baryon content with cosmology from the
# parameter vector alone (fixed-DMO), or does cosmology require the DMO field?
# Compares TRUTH vs GEN(correct DMO) vs GEN(fixed DMO) across the 1P cosmology levels.
#
# Submit:
#   sbatch run_validate_cosmo.sh
#   PARAMS="Omega_m" sbatch run_validate_cosmo.sh

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
export PYTHONUNBUFFERED=1

PARAMS=${PARAMS:-"Omega_m sigma8"}
N_HALOS=${N_HALOS:-49}
K=${K:-12}
N_STEPS=${N_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-16}

mkdir -p /mnt/home/mlee1/ceph/logs figures/scatter_diagnostics outputs/scatter_diagnostics

echo "=== Cosmology rescaling diagnostic: params=[$PARAMS] N_halos=$N_HALOS K=$K ==="
python -m scatter.validate_cosmo_rescaling \
    --params     $PARAMS \
    --n-halos    "$N_HALOS" \
    --k          "$K" \
    --n-steps    "$N_STEPS" \
    --batch-size "$BATCH_SIZE"

echo "=== done ==="
echo "  outputs/scatter_diagnostics/validate_cosmo_rescaling.json"
echo "  figures/scatter_diagnostics/validate_cosmo_<param>.{pdf,png}"
