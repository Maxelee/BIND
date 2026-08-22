#!/bin/bash
#SBATCH --job-name=bind_n1000_pre
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_n1000_pre_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_n1000_pre_%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint="cascadelake|skylake"
#SBATCH --nodes=4
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --time=72:00:00
#SBATCH --array=0-318%40
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=FAIL

# ── N1000 campaign, OPPORTUNISTIC lane (preempt QOS — no per-user node cap) ───
# Harvests idle capacity on the 48-core node pools (4 nodes x 48 = the same 192
# MPI ranks as the cca lane).  Preemption is expected and safe: --requeue +
# per-chunk sentinels mean a killed job resumes at its last finished 125-real
# chunk (max ~5 h redone).  Task order is REVERSED (sb35 tail first) so this
# lane meets the forward-marching cca lane in the middle; the atomic claim dirs
# in $OUT_ROOT/.claims prevent double-tracing when they meet.
#
# Submit:   sbatch run_n1000_preempt.sh
# A task that exits immediately with "claimed by job ..." did no work — that run
# is owned by the other lane.  After a permanent failure (not preemption),
# release the run with:  rm -r <OUT_ROOT>/.claims/task_NNNN
# 72h wall (vs cca's 48h): the older 48-core pools run ~10-15% slower and a
# TIMEOUT — unlike a preemption — is NOT auto-requeued; the margin plus chunk
# sentinels make a timeout a resubmit-and-resume, never lost work.  %40 bounds
# in-flight transient disk (each active run holds ~285 GB of planes+raw scratch).
mkdir -p /mnt/home/mlee1/ceph/logs
: "${SLURM_ARRAY_TASK_ID:?run_n1000_preempt.sh must run as a SLURM array task}"
TASK_IDX=$(( 318 - SLURM_ARRAY_TASK_ID ))
export TASK_IDX
source /mnt/home/mlee1/BIND/n1000_body.sh
