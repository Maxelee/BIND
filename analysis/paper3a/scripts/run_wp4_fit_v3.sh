#!/bin/bash
#SBATCH --job-name=wp4_fit_v3
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp4_fit_v3_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp4_fit_v3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A4 v3 emulator fit (production + outdesign, then 8-fold CV) ──────────
# Single-task job (the GP fits share memory and torch threads well; the
# ~2 h k-fold is the long pole and must survive logout — hence sbatch, not
# a login/background run). Inputs: gasemu_dataset_v3.npz (409-dim vectors
# incl. the Sigma(R) + own-patch blocks). Outputs (ceph wp4_emulator/):
#   gasemu_gp_v3.npz         production artifact (numpy-only, A5-ready)
#   gasemu_outdesign_v3.npz  twobound + fiducial out-of-design test
#   gasemu_kfold_v3.npz      8-fold held-out predictions
#
# ⛔ HUMAN CHECKPOINT — Max submits:
#   sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_wp4_fit_v3.sh
#
# Next session picks up with:
#   python analysis/paper3a/scripts/wp4_validate_emulator.py   (point at *_v3)

set -euo pipefail

module load gcc
source /mnt/home/mlee1/venvs/torch3/bin/activate
export OMP_NUM_THREADS=16
cd /mnt/home/mlee1/vdm_bind2-paper3a

WP4=/mnt/ceph/users/mlee1/paper3/A/wp4_emulator

python -m analysis.paper3a.emulator.fit \
    --dataset "$WP4/gasemu_dataset_v3.npz" \
    --artifact-out "$WP4/gasemu_gp_v3.npz" \
    --outdesign-out "$WP4/gasemu_outdesign_v3.npz" \
    --n-pca 24

python -m analysis.paper3a.emulator.fit \
    --dataset "$WP4/gasemu_dataset_v3.npz" \
    --kfold 8 \
    --validation-out "$WP4/gasemu_kfold_v3.npz" \
    --n-pca 24
