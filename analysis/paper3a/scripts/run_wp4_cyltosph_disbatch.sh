#!/bin/bash
#SBATCH --job-name=wp4_c2s_db
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp4_c2s_db_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp4_c2s_db_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A4 CylToSph feedback dependence via disBatch ─────────────────────────
# 177 tasks (one per CAMELS L50n512/1P sim), each streaming ~16 snapshot
# chunks (~8 GB total read, few hundred MB resident) for per-halo
# sphere/cylinder gas sums at z=0. Fiducial smoke on a rusty worker:
# ~25 min/sim (ceph-I/O bound) -> 177 tasks / 32 slots ~ 6 waves ~ 2.5-3 h.
#
# ⛔ HUMAN CHECKPOINT — Max submits:
#   mkdir -p /mnt/home/mlee1/ceph/logs/wp4_cyltosph
#   sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_wp4_cyltosph_disbatch.sh
#
# Per-task logs: /mnt/home/mlee1/ceph/logs/wp4_cyltosph/task_<id>.log
# Retry stragglers (or just resubmit — the script skips finished outputs):
#   disBatch -r /mnt/home/mlee1/ceph/logs/wp4_cyltosph/db_*_status.txt -R \
#       analysis/paper3a/scripts/camels1p_cyltosph.disbatch

set -euo pipefail

module load disBatch
mkdir -p /mnt/home/mlee1/ceph/logs/wp4_cyltosph
cd /mnt/home/mlee1/vdm_bind2-paper3a

disBatch -p /mnt/home/mlee1/ceph/logs/wp4_cyltosph/db analysis/paper3a/scripts/camels1p_cyltosph.disbatch
