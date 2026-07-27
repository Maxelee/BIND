#!/bin/bash
#SBATCH --job-name=t2f_fineres
#SBATCH --output=/mnt/home/mlee1/ceph/logs/t2f_fineres_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/t2f_fineres_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=2
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── R0+R1c (docs/p4c_referee_hardening_plan.md): the FULL T2 battery at the
# BIND-matched 0.29296875'/px, with per-object CAP storage + 100-cell
# jackknife (engine patch 2026-07-28). --force overwrites the two existing
# T2f files (measured pre-patch, 30 cells, no per-object arrays).
# 15 concurrent tasks x ~25GB (map+prefilter+mask per task) — one fat node.
module load disBatch
disBatch -p /mnt/home/mlee1/ceph/logs/t2f_db /mnt/home/mlee1/BIND-ksz2/examples/_t2f_tasks.disbatch
echo "T2F DISBATCH DONE"
