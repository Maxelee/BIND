#!/bin/bash
#SBATCH --job-name=scatter_merge
#SBATCH --output=/mnt/home/mlee1/ceph/logs/scatter_merge_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/scatter_merge_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Merges scatter Jacobian shards → J_mean_and_scatter.npz
# Then produces fig2, fig4, fills PAPER_OUTLINE numbers.
# Submit after all scatter_jac array tasks finish:
#
#   JAC_JOBID=$(sbatch --parsable run_scatter_jacobian.sh)
#   sbatch --dependency=afterok:${JAC_JOBID} run_scatter_merge.sh

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

export PYTHONUNBUFFERED=1

JAC_OUT="scatter/J_mean_and_scatter.npz"
SHARD_GLOB="scatter/intermediate/shard_*.npz"

echo "=== Merging shards ==="
python scatter/scatter_jacobian.py merge \
    --shard_glob "$SHARD_GLOB" \
    --output     "$JAC_OUT"

echo "=== Gating check ==="
python scatter/scatter_jacobian.py summary --input "$JAC_OUT"

echo "=== Fig 2: scatter vs mean Jacobian ==="
python scatter/figures.py --fig2 --jac "$JAC_OUT"

echo "=== Fig 4: inter vs intra ==="
python scatter/figures.py --fig4

echo "=== Fill PAPER_OUTLINE numbers ==="
python scatter/fill_outline_numbers.py

echo "=== Phase 1: build phase1_intra_jacobian.npz from combined shard data ==="
python scatter/scatter_intra_jacobian.py merge \
    --shard_glob "$SHARD_GLOB" \
    --output     "outputs/scatter_diagnostics/phase1_intra_jacobian.npz"

echo "=== Phase 1: generate intra vs inter figure ==="
python scatter/scatter_intra_jacobian.py figures \
    --input "outputs/scatter_diagnostics/phase1_intra_jacobian.npz"

echo "$(date '+%Y-%m-%d %H:%M') | Phase 1 complete via merged scatter_jacobian shards" \
    >> outputs/scatter_diagnostics/PROGRESS.log

echo "=== Merge + post-processing complete ==="
