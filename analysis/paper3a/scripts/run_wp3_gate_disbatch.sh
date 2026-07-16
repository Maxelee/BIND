#!/bin/bash
#SBATCH --job-name=wp3_gate_db
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp3_gate_db_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp3_gate_db_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A3 gate operator evaluation via disBatch (recommended path) ───────────
# Replaces the 317-task Slurm array (run_wp3_gate.sh, kept as fallback) with
# ONE allocation: disBatch dynamically feeds all 317 independent tasks
# (wp3_gate.disbatch) into 64 slots on a single node. Sizing: each task is
# ~30 min on ~1 core and ~2 GB (measured on twobound run_0044), so
# 317 tasks / 64 slots ~= 5 waves ~= 2.5-3 h wall, well inside --time.
#
# ⛔ HUMAN CHECKPOINT — Max submits:
#   mkdir -p /mnt/home/mlee1/ceph/logs/wp3_gate
#   sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_wp3_gate_disbatch.sh
#
# Per-task logs: /mnt/home/mlee1/ceph/logs/wp3_gate/task_<idx>.log
# Retry stragglers/failures (also fine to just resubmit — the driver skips
# finished outputs):
#   disBatch -r /mnt/home/mlee1/ceph/logs/wp3_gate/db_*_status.txt -R \
#       analysis/paper3a/scripts/wp3_gate.disbatch

set -euo pipefail

module load disBatch
mkdir -p /mnt/home/mlee1/ceph/logs/wp3_gate
cd /mnt/home/mlee1/vdm_bind2-paper3a

disBatch -p /mnt/home/mlee1/ceph/logs/wp3_gate/db analysis/paper3a/scripts/wp3_gate.disbatch
