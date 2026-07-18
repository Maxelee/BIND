#!/bin/bash
#SBATCH --job-name=wp6_statsemu
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp6_statsemu_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp6_statsemu_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A6 statsemu: training fan-out + merge + posterior push-through ───────
# Stage 1: disBatch — 23 per-target tasks (production PCA+GP fit + 8-fold
#          CV each; the GP cost is per-target-independent, so one node
#          finishes the whole battery in one ~30-40 min wave).
# Stage 2: merge the per-target parts into statsemu_gp.npz +
#          statsemu_gp_validation.json (per-target frac err AND
#          SEM-relative err from out-of-fold predictions).
# Stage 3: run_a6_posterior_stats.py — A5 FINAL posterior envelopes for
#          every target + the wlemu-vs-statsemu z_s=1 suppression
#          cross-check + the envelope figure.
#
# ⛔ Submit:  sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_wp6_statsemu_disbatch.sh

set -euo pipefail

module load gcc disBatch
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2-paper3a
mkdir -p /mnt/home/mlee1/ceph/logs/wp6_statsemu

echo "=== stage 1: per-target fits + 8-fold CV (23 tasks) ==="
disBatch -p /mnt/home/mlee1/ceph/logs/wp6_statsemu/db_ \
    analysis/paper3a/scripts/wp6_statsemu.disbatch

echo "=== stage 2: merge parts ==="
python -u -m analysis.paper3a.emulator.statsemu merge

echo "=== stage 3: A5-posterior push-through + wlemu cross-check ==="
python -u analysis/paper3a/scripts/run_a6_posterior_stats.py

echo "DONE"
