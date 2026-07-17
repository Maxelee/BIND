#!/bin/bash
#SBATCH --job-name=wp4_v3_db
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp4_v3_db_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp4_v3_db_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A4 v3 operator recompute via disBatch ────────────────────────────────
# Same shape as the v2 gate job (317 tasks, 64 slots, one node): recomputes
# every design point with the added ksz_binN_stack_own columns (the
# own/neighbor split the velocity-decorrelation forward model needs;
# emulator/forward.py). Output: operator_tables_v3/. Per-task cost is
# slightly above v2 (~30 min + isolated-map CAPs), so the same --time holds.
#
# ⛔ HUMAN CHECKPOINT — Max submits:
#   mkdir -p /mnt/home/mlee1/ceph/logs/wp4_v3
#   sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_wp4_v3_disbatch.sh
#
# Per-task logs: /mnt/home/mlee1/ceph/logs/wp4_v3/task_<idx>.log
# Retry stragglers (or just resubmit — the driver skips finished outputs):
#   disBatch -r /mnt/home/mlee1/ceph/logs/wp4_v3/db_*_status.txt -R \
#       analysis/paper3a/scripts/wp3_gate_v3.disbatch
#
# After it lands, rebuild the A4 training set on v3 and refit (adds the own
# columns to the forward model's own_frac): see emulator/REPORT pointers.

set -euo pipefail

module load disBatch
mkdir -p /mnt/home/mlee1/ceph/logs/wp4_v3
cd /mnt/home/mlee1/vdm_bind2-paper3a

disBatch -p /mnt/home/mlee1/ceph/logs/wp4_v3/db analysis/paper3a/scripts/wp3_gate_v3.disbatch
