#!/bin/bash
#SBATCH --job-name=b4_variants
#SBATCH --output=/mnt/home/mlee1/ceph/logs/b4_variants_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/b4_variants_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-B4/B5: the three MODEL_FREEZE variant scales, one small allocation ────
# Sequential (bind-only + glimpse cross-check; 1-2 units each at n_seeds=32
# ~100 min/unit): variant8 + plain weight schemes on the Wiener comparison
# chain, then the GLIMPSE-transfer cross-check (bind + truth). Right-sized
# replacement for reusing the 2-node grid sbatch on 1-2 units.
#
# ⛔ HUMAN CHECKPOINT — submit by hand:
#   sbatch /mnt/home/mlee1/BIND-paper3b/analysis/paper3b/scripts/run_b4_variants.sh
#
# Outputs -> wp4_mocks/grid_tfwiener_variant8/, grid_tfwiener_plain/,
#            grid_tfglimpse/  (assembled + hashed into MODEL_FREEZE.md).

set -euo pipefail

module load python openmpi python-mpi
VENV=${VENV:-/mnt/home/mlee1/venvs/paper3b_popeye}
source "$VENV/bin/activate"
export PYTHONPATH="$VENV/lib/python3.11/site-packages:${PYTHONPATH:-}"
mkdir -p /mnt/home/mlee1/ceph/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /mnt/home/mlee1/BIND-paper3b

run() { mpirun -np 2 python -u -m analysis.paper3b.scripts.run_b4_grid \
        --n-seeds 32 "$@"; }

run --categories bind --transfer wiener --weight-scheme variant8
run --categories bind --transfer wiener --weight-scheme plain
run --categories bind truth --transfer glimpse

echo "=== B4 variants done ==="
