#!/bin/bash
#SBATCH --job-name=dmo_history
#SBATCH --output=/mnt/home/mlee1/ceph/logs/dmo_history_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/dmo_history_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── DMO accretion-history catalog for the SHMR scatter study ──────────────────
# Section 1 of examples/shmr_accretion_scatter.ipynb, parallelised over cores.
# Single node, multiprocessing (no MPI) — ~2900 halos in ~1 min on 64 cores.
# Needs illustris_python in the venv:  pip install illustris_python
#
# Submit:   sbatch run_dmo_history.sh
# Override: MASS_CUT, BASE_SNAP, OUT.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

MASS_CUT=${MASS_CUT:-12.9}
BASE_SNAP=${BASE_SNAP:-96}
OUT=${OUT:-/mnt/home/mlee1/ceph/tng300_dm_mass_history/tng300dm_mass_history.hdf5}

python examples/build_dmo_history.py \
    --out "$OUT" \
    --base-snap "$BASE_SNAP" \
    --mass-cut "$MASS_CUT" \
    --nproc "${SLURM_CPUS_PER_TASK:-64}"

echo "=== done -> $OUT ==="
