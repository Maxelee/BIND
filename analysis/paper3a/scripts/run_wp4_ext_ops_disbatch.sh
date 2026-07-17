#!/bin/bash
#SBATCH --job-name=wp4_ext_ops
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp4_ext_ops_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp4_ext_ops_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A4 v3 operators for the extension design (256 tasks) ─────────────────
# Stage 2 of the densification chain (after run_wp4_paint_ext.sh): evaluates
# the v3 gate operators on the newly painted Sobol points 256-511, writing
# sb35_run_0256..0511 tables into operator_tables_v3/ alongside the
# originals. Same cost profile as the v3 job (finished in ~75 min).
#
# Submitted by Max as part of the dependency chain (see run_wp4_paint_ext.sh
# header). Retry stragglers:
#   disBatch -r /mnt/home/mlee1/ceph/logs/wp4_ext_ops/db_*_status.txt -R \
#       analysis/paper3a/scripts/wp4_gate_ext.disbatch

set -euo pipefail

module load disBatch
mkdir -p /mnt/home/mlee1/ceph/logs/wp4_ext_ops
cd /mnt/home/mlee1/vdm_bind2-paper3a

disBatch -p /mnt/home/mlee1/ceph/logs/wp4_ext_ops/db analysis/paper3a/scripts/wp4_gate_ext.disbatch
