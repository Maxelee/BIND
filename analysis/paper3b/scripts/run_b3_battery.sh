#!/bin/bash
#SBATCH --job-name=b3_battery
#SBATCH --output=/mnt/home/mlee1/ceph/logs/b3_battery_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/b3_battery_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --time=04:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-B3: null/systematics battery (NULL_CRITERIA.md §1–§4) ─────────────────
# MPI over peaks (48 ranks, same layout as the B2 measurement job):
#   §1 null ensemble        2 variants x 32 seeded random-position realizations
#   §2 RA-shifted stacks    2 variants x {±15°, ±30°} (ACT-interior re-cut)
#   §3 B-mode (nullB) peak build + stack on the fiducial y map
#   §4 CIB band             2 variants x the 10-map deprojection spanning set
# ~104 collective stacks + 11 map loads -> ~2–2.5 h wall. Criteria were
# pre-registered in the plans repo NULL_CRITERIA.md BEFORE this job existed;
# battery_summary.json records the evaluations. §5/§6 already ran inline.
#
# ⛔ HUMAN CHECKPOINT — submit by hand:
#   sbatch /mnt/home/mlee1/BIND-paper3b/analysis/paper3b/scripts/run_b3_battery.sh

set -euo pipefail

module load python openmpi python-mpi

VENV=${VENV:-/mnt/home/mlee1/venvs/paper3b_popeye}
source "$VENV/bin/activate"
export PYTHONPATH="$VENV/lib/python3.11/site-packages:${PYTHONPATH:-}"

mkdir -p /mnt/home/mlee1/ceph/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd /mnt/home/mlee1/BIND-paper3b
mpirun -np "$SLURM_NTASKS" python -u analysis/paper3b/scripts/run_b3_battery.py "$@"

echo "=== B3 battery done ==="
