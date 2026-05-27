#!/bin/bash
#SBATCH --job-name=scatter_post
#SBATCH --output=/mnt/home/mlee1/ceph/logs/scatter_post_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/scatter_post_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# All post-Jacobian steps in one job: merge, fig2, fig3, robustness, outline.
# Prefer the modular scripts (run_scatter_merge.sh, run_scatter_fig3.sh,
# run_scatter_robustness.sh) when you want per-step control.
#
# Usage:
#   JAC_JOBID=$(sbatch --parsable run_scatter_jacobian.sh)
#   sbatch --dependency=afterok:${JAC_JOBID} run_post_jacobian.sh [--skip_fig3] [--skip_robust]

set -euo pipefail
cd /mnt/home/mlee1/vdm_bind2
source /mnt/home/mlee1/venvs/torch3/bin/activate
export PYTHONUNBUFFERED=1

mkdir -p paper_figures/scatter scatter/robustness /mnt/home/mlee1/ceph/logs

SKIP_FIG3=0
SKIP_ROBUST=0
for arg in "$@"; do
  [[ "$arg" == "--skip_fig3" ]] && SKIP_FIG3=1
  [[ "$arg" == "--skip_robust" ]] && SKIP_ROBUST=1
done

JAC=scatter/J_mean_and_scatter.npz
SHARD_GLOB="scatter/intermediate/shard_*.npz"

# 0. Merge shards if J_mean_and_scatter.npz doesn't already exist
if [[ ! -f "$JAC" ]]; then
  echo "=== [0/5] Merging shards ==="
  python scatter/scatter_jacobian.py merge \
      --shard_glob "$SHARD_GLOB" \
      --output     "$JAC"
else
  echo "=== [0/5] $JAC already exists — skipping merge ==="
fi

echo "========================================"
echo "=== Post-Jacobian pipeline starting ==="
echo "========================================"

# 1. Gating check on Jacobian
echo -e "\n=== [1/5] Jacobian gating check ==="
python scatter/scatter_jacobian.py summary --input "$JAC"

# 2. Fig 2: scatter vs mean (fast, just plotting)
echo -e "\n=== [2/5] Fig 2: scatter vs mean Jacobian plot ==="
python scatter/figures.py --fig2 --jac "$JAC"

# 3. Fig 3: scatter contours in (A_SN1, A_AGN1) plane (~2 hr)
if [[ $SKIP_FIG3 -eq 0 ]]; then
  echo -e "\n=== [3/5] Fig 3: scatter contours grid (may take ~2 hr) ==="
  python scatter/figures.py --fig3
else
  echo -e "\n=== [3/5] Fig 3: SKIPPED (--skip_fig3) ==="
fi

# 4. Robustness checks
if [[ $SKIP_ROBUST -eq 0 ]]; then
  echo -e "\n=== [4/5] Robustness checks ==="
  python scatter/robustness/run_all_robustness.py
else
  echo -e "\n=== [4/5] Robustness: SKIPPED (--skip_robust) ==="
fi

# 5. Fill in PAPER_OUTLINE numbers
echo -e "\n=== [5/5] Fill PAPER_OUTLINE.md with numbers ==="
python scatter/fill_outline_numbers.py

echo -e "\n=== Post-Jacobian pipeline complete ==="
echo "Figures in: paper_figures/scatter/"
echo "Outline:    scatter/PAPER_OUTLINE.md"
echo "Robustness: scatter/robustness/SUMMARY.md"
