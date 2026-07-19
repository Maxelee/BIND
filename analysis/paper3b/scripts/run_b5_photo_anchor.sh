#!/bin/bash
#SBATCH --job-name=b5_photo_anchor
#SBATCH --output=/mnt/home/mlee1/ceph/logs/b5_photo_anchor_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/b5_photo_anchor_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-B5 ladder item 5: photometric anchor stack (DESI Main LRG x DR6 y) ────
# 120k galaxies/bin x 4 photo-z bins through the frozen thumbnail+CAP chain.
# Per-bin checkpoints (photo_anchor_bin<b>.npz) → resumable: bins already
# completed by any previous run are skipped, so this is safe to RESUBMIT
# after a timeout/crash. Do NOT deliberately run two instances at once —
# an in-flight bin would be double-computed and the summary json is
# last-writer-wins (2026-07-18 validation; checkpoints are now atomic and
# corrupt-tolerant, and each bin has an independent rng stream). Outputs ->
# ~/ceph/paper3/B/wp5_inference/photo_anchor/{photo_anchor_summary.json,png}
#
# ⛔ HUMAN CHECKPOINT — submit by hand:
#   sbatch /mnt/home/mlee1/BIND-paper3b/analysis/paper3b/scripts/run_b5_photo_anchor.sh

set -euo pipefail
source /mnt/home/mlee1/venvs/paper3b_popeye/bin/activate
cd /mnt/home/mlee1/BIND-paper3b/analysis
mkdir -p /mnt/home/mlee1/ceph/logs

python -u -m paper3b.scripts.run_b5_photo_anchor --chunk 10000
echo "=== photo anchor done ==="
