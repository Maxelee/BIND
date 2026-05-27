#!/bin/bash
#SBATCH --job-name=scatter_intra_merge
#SBATCH --output=/mnt/home/mlee1/ceph/logs/scatter_intra_merge_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/scatter_intra_merge_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Phase 1: merge intra-σ Jacobian shards, compute contamination ratios, make figure.
#
# Submit after all intra_jac array tasks finish:
#   INTRA_JOBID=$(sbatch --parsable run_scatter_intra_jacobian.sh)
#   sbatch --dependency=afterok:${INTRA_JOBID} run_scatter_intra_merge.sh

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

export PYTHONUNBUFFERED=1

SHARD_GLOB="scatter/intermediate_intra/shard_*.npz"
MERGED_OUT="outputs/scatter_diagnostics/phase1_intra_jacobian.npz"

echo "=== Merging intra-Jacobian shards ==="
python scatter/scatter_intra_jacobian.py merge \
    --shard_glob "$SHARD_GLOB" \
    --output     "$MERGED_OUT"

echo "=== Generating fig_intra_vs_inter_jacobian ==="
python scatter/scatter_intra_jacobian.py figures \
    --input "$MERGED_OUT"

echo "=== Updating PROGRESS.log ==="
echo "$(date '+%Y-%m-%d %H:%M') | Phase 1 complete — merged intra Jacobian; gate1 report in outputs/scatter_diagnostics/phase1_gate1_report.json" \
    >> outputs/scatter_diagnostics/PROGRESS.log

echo "=== Phase 1 merge complete ==="
echo "Output: $MERGED_OUT"
echo "Figure: figures/scatter_diagnostics/fig_intra_vs_inter_jacobian.pdf"
echo "Gate 1: outputs/scatter_diagnostics/phase1_gate1_report.json"
