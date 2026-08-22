#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=cascadelake
#SBATCH -J wlemu_cache
#SBATCH -N 6
#SBATCH -n 48
#SBATCH --exclusive
#SBATCH -o /mnt/home/mlee1/ceph/logs/wlemu_cache_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/wlemu_cache_%j.err
#SBATCH -t 02:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WL field-level emulator: STEP 1 (CPU, OpenMPI) — build the κ cache ─────────
# Pools the SB35 Sobol κ suite (run_NNNN/kappa_maps.npz) into one memmap that
# bind.wlemu reads.  Standalone script (numpy + mpi4py, no bind/CLI): each rank
# pools its [rank::size] runs into disjoint rows; the κ normalisation is an
# MPI.Allreduce over a streamed two-pass reduction.  Idempotent — if kappa.npy is
# already written it skips the fill and only (re)computes norm.npz.
#
#   sbatch run_wlemu_cache.sh                  # native 1024² (~130 GB cache)
#   RESOLUTION=256 sbatch run_wlemu_cache.sh   # fast-iteration res (~8 GB)
#
# Env overrides: RESOLUTION, OUT, RUNS_DIR, N_REAL.

set -euo pipefail

module load python openmpi python-mpi
source ~/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

RESOLUTION=${RESOLUTION:-1024}
OUT=${OUT:-/mnt/home/mlee1/ceph/bind_sb35/wlemu_cache_${RESOLUTION}}
RUNS_DIR=${RUNS_DIR:-/mnt/home/mlee1/ceph/bind_sb35/runs}
N_REAL=${N_REAL:-0}

echo "=== wlemu cache | job ${SLURM_JOB_ID} | ${SLURM_NNODES} nodes × ${SLURM_NTASKS} ranks | res ${RESOLUTION}² ==="

srun python -u build_wlemu_cache.py \
    --out "$OUT" \
    --runs_dir "$RUNS_DIR" \
    --resolution "$RESOLUTION" \
    --n_real "$N_REAL"

echo "=== done → $OUT ==="
