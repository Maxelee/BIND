#!/bin/bash
#SBATCH --job-name=wp3_gate
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp3_gate_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp3_gate_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-316
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A3 gate operator evaluation (rusty, CPU) ──────────────────────────────
# One task per painted design point: 256 SB35 Sobol (0-255) + 60 twobound 1P
# extremes (256-315) + the fiducial (316; skipped harmlessly until its paint
# lands). Each task pastes gas composites and evaluates the WP-A2 operators
# at 6 gate snapshots -> compact npz tables under
# /mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables/.
# Resumable: existing outputs are skipped, so re-submitting after adding the
# fiducial (or after failures) only computes what is missing.
#
# ⛔ HUMAN CHECKPOINT — Max submits:
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_wp3_gate.sh

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2-paper3a
mkdir -p /mnt/home/mlee1/ceph/logs

python -u analysis/paper3a/scripts/run_gate_operators.py "$SLURM_ARRAY_TASK_ID"
