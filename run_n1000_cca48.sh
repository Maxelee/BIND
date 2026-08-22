#!/bin/bash
#SBATCH --job-name=bind_n1000_c48
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_n1000_c48_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_n1000_c48_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint="cascadelake|skylake"
#SBATCH --nodes=4
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --time=60:00:00
#SBATCH --array=100-250
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=FAIL

# ── N1000 campaign, cca lane B: 48-core pools (cascadelake/skylake) ───────────
# Same guaranteed cca QOS as run_n1000_cca.sh but on the 216 cascadelake + 144
# skylake nodes (4 x 48 = the same 192 MPI ranks; 140 s/real was MEASURED on
# cascadelake).  Purpose: fill the per-user caps (node=100, cpu=6400) with
# whichever node type is free — the icelake lane alone is placement-limited.
# The QOS caps jointly bound BOTH cca arrays (cpu=6400 -> <=33 running jobs
# total), so this adds no transient-disk exposure; excess tasks simply pend on
# the QOS limit.  60 h wall (skylake ~10% slower than the 41 h cascadelake run).
#
# Covers manifest slice 100-250 (sb35 run_0037..run_0187): the forward icelake
# lane and the reverse preempt lane approach from the ends; per-task claim dirs
# make any meeting point safe.  MaxSubmitPU=500 on the cca QOS is why this is a
# slice, not the full 0-318 — resubmit other ranges as the queues drain.
#
# Submit:   sbatch run_n1000_cca48.sh
mkdir -p /mnt/home/mlee1/ceph/logs
: "${SLURM_ARRAY_TASK_ID:?run_n1000_cca48.sh must run as a SLURM array task}"
TASK_IDX="$SLURM_ARRAY_TASK_ID"
export TASK_IDX
source /mnt/home/mlee1/BIND/n1000_body.sh
