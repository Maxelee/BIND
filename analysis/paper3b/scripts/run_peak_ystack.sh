#!/bin/bash
#SBATCH --job-name=b2_peakystack
#SBATCH --output=/mnt/home/mlee1/ceph/logs/b2_peakystack_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/b2_peakystack_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-B2: the measurement — ACT y stacked at DES κ-peak positions ───────────
# MPI over peaks (one 48-rank node): {wiener, glimpse} sm2am catalogs x
# {fiducial, CIB-deprojected} y maps + permuted-nu random nulls, per the
# frozen MEASUREMENT_SPEC (plans repo). Ranks stride the peak list (thumbnail
# extraction + per-peak CAP), per-peak tables gather and per-bin stamp sums
# reduce to rank 0, which runs the jackknife and writes
# ~/ceph/paper3/B/wp2_measurement/ (stack_*.npz + measurement_summary.json).
# Deterministic: identical output to the serial path (verified bit-exact on a
# 2-rank smoke); re-running overwrites in place.
#
# Memory: each rank holds one 1.78 GB y map + ~100 MB workspace -> ~90 GB/node
# well under --mem=240G. Wall: ~113k thumbnails / 48 ranks + 4 map loads ->
# ~10 min. NO GPU.
#
# ⛔ HUMAN CHECKPOINT — submit by hand:
#   sbatch /mnt/home/mlee1/BIND-paper3b/analysis/paper3b/scripts/run_peak_ystack.sh

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
mpirun -np "$SLURM_NTASKS" python -u analysis/paper3b/scripts/run_peak_ystack.py "$@"

echo "=== B2 peak-y stack done ==="
