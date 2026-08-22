#!/bin/bash
#SBATCH --job-name=bind_n1000_cca
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_n1000_cca_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_n1000_cca_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=3
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --array=0-318%33
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=FAIL

# ── N1000 campaign, GUARANTEED lane (cca QOS) ─────────────────────────────────
# 33 concurrent 3-node icelake jobs = 99 nodes / 6,336 cores — pinned just under
# the per-user cca caps (node=100, cpu=6400, MaxJobsPU=50).  192 MPI ranks per
# job, same rank count as every timed production trace.  Forward task order:
# bind/dmo/truth land first, then twobound, then sb35.
# The preempt lane (run_n1000_preempt.sh) walks the same manifest in reverse;
# per-task claim dirs keep the two lanes from ever tracing the same run.
#
# Submit:   sbatch run_n1000_cca.sh
# Monitor:  squeue -u $USER -n bind_n1000_cca
# NB: %33 x 3 nodes = 6,336 of the 6,400-core per-user cca cap — while the
# campaign runs, any OTHER cca-QOS job under this account will queue behind it.
mkdir -p /mnt/home/mlee1/ceph/logs
: "${SLURM_ARRAY_TASK_ID:?run_n1000_cca.sh must run as a SLURM array task}"
TASK_IDX="$SLURM_ARRAY_TASK_ID"
export TASK_IDX
source /mnt/home/mlee1/BIND/n1000_body.sh
