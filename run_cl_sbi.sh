#!/bin/bash
#SBATCH --job-name=bind_cl_sbi
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_cl_sbi_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_cl_sbi_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── SBI on the WL auto power spectrum ─────────────────────────────────────────
# Trains a neural posterior estimator (sbi NPE) that maps the tomographic kappa
# auto-spectrum suppression -> P(theta | x) over the 30 SB35 astro params, then
# evaluates at the fiducial lensing statistics and writes a 30-param corner.
# The flow is small (~120 sims); a single GPU finishes in well under a minute.
#
# Env overrides: DATASET, FIDUCIAL_CL, ELL_MAX, N_ELL_BINS, N_PCA, N_SAMPLES.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND

python -u examples/lightcone_cl_sbi.py \
    --device cuda \
    --dataset    "${DATASET:-/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz}" \
    --fiducial_cl "${FIDUCIAL_CL:-/mnt/home/mlee1/ceph/bind_science/runs/bind/run_0000/Cl_kappa.npz}" \
    --ell_max    "${ELL_MAX:-5000}" \
    --n_ell_bins "${N_ELL_BINS:-10}" \
    --n_pca      "${N_PCA:-0}" \
    --n_samples  "${N_SAMPLES:-50000}"
