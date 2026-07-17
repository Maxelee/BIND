#!/bin/bash
#SBATCH --job-name=b4_grid
#SBATCH --output=/mnt/home/mlee1/ceph/logs/b4_grid_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/b4_grid_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --time=03:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-B4: the model grid — matched mock measurement over the atlas ──────────
# MPI over atlas runs (62 units: 60 twobound + bind fiducial + TNG truth).
# Per unit: 50 realizations x 8 noise seeds = 400 survey-realistic patches
# through the frozen chain (DES n(z) amplitude-corrected plane weights + shape
# noise + PEAK_DEFINITION smoothing/peaks/nu + ACT beam + the IMPORTED B2
# thumbnail/CAP operators). One npz per unit ->
# ~/ceph/paper3/B/wp4_mocks/grid/. Fully resumable: finished units are
# skipped on relaunch (--overwrite to force).
#
# PREREQ (once, serial, ~1 min — already run interactively if
# weights_desy3.npz exists): python -m analysis.paper3b.scripts.build_b4_weights
#
# Memory: each rank holds one unit's kappa (1.05 GB f32) + y slice (~1.05 GB)
# + float64 working copies -> ~4 GB/rank peak, ~190 GB/node worst case;
# 360G gives slack (B3 OOM lesson). Wall: measured 13.4 s/patch on the
# interactive smoke (pixell order-3 thumbnail interpolation dominates — the
# same operator the data chain used, so it is NOT changed for mocks) -> 400
# patches/unit ~ 90 min. 2 nodes x 48 ranks = 96 > 62 units -> 1 unit/rank,
# ~1.6 h + load; 03:00:00 covers it.
#
# ⛔ HUMAN CHECKPOINT — submit by hand:
#   sbatch /mnt/home/mlee1/BIND-paper3b/analysis/paper3b/scripts/run_b4_grid.sh
#
# Weight-scheme sensitivity variants (bind fiducial only, cheap, AFTER the
# main grid — quantifies the plane-weighting systematic in REPORT.md):
#   sbatch ... run_b4_grid.sh --categories bind --weight-scheme variant8
#   sbatch ... run_b4_grid.sh --categories bind --weight-scheme plain

set -euo pipefail

# mpi4py from the python-mpi module (venv re-prepended so the module's h5py
# cannot shadow the venv's — same pattern as run_wp2_truth_validation.sh).
module load python openmpi python-mpi

VENV=${VENV:-/mnt/home/mlee1/venvs/paper3b_popeye}
source "$VENV/bin/activate"
export PYTHONPATH="$VENV/lib/python3.11/site-packages:${PYTHONPATH:-}"

mkdir -p /mnt/home/mlee1/ceph/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd /mnt/home/mlee1/BIND-paper3b
mpirun -np "$SLURM_NTASKS" python -u -m analysis.paper3b.scripts.run_b4_grid "$@"

echo "=== B4 grid done ==="
