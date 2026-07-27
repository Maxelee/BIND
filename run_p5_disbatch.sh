#!/bin/bash
#SBATCH --job-name=bind_p5_cap
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_p5_cap_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_p5_cap_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── P5 of docs/ksz_lightcone_map_plan.md: 256-run CAP-stack sweep via disBatch ──
# 2048 tasks (256 runs x 8 {map,sample,beam} combos), each ~1-3 min; 256 slots
# -> ~15-25 min wall. Idempotent (existing shards skip instantly) — safe to
# resubmit to mop up any failures. Task file: examples/_p5_tasks.disbatch.
# Check afterwards:  grep -c "run[0-9]" <(ls .../lightcone/shards)  -> expect 2048.

set -euo pipefail
TASKFILE=${TASKFILE:-/mnt/home/mlee1/BIND-ksz2/examples/_p5_tasks.disbatch}
# ELG pass: TASKFILE=/mnt/home/mlee1/BIND-ksz2/examples/_p5_elg_tasks.disbatch sbatch run_p5_disbatch.sh
# M5 massbin pass (506 tasks = 253 runs x {massbin85,massbin46}, ~1 min/task,
# well inside the 01:00:00 limit above):
#   TASKFILE=/mnt/home/mlee1/BIND-ksz2/examples/_p5_massbin_tasks.disbatch sbatch run_p5_disbatch.sh
module load disBatch
mkdir -p /mnt/home/mlee1/ceph/logs /mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone/disbatch_logs
cd /mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone/disbatch_logs
disBatch "$TASKFILE"
echo "=== disBatch done; shard count: ==="
ls /mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone/shards | grep -c 'run[0-9]' || true
