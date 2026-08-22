#!/bin/bash
#SBATCH --job-name=bind_sobol_reduce
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_sobol_reduce_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_sobol_reduce_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL
# ──────────────────────────────────────────────────────────────────────────────
# BIND sb35 Sobol halo-aperture reduction on the *preempt* partition.
#
# The work is the per-(run,snap) aperture reduction in bind_mpi_worker.py.  It is
# preempt-safe: every key is written to its own shard under
# analysis_cache/shards/ and a key with an existing shard is skipped, so a
# preempted+requeued job just resumes.  --requeue (above) lets Slurm restart it
# automatically after a preemption; nothing is lost.
#
# One-time, BEFORE the first submit (cheap, run on a login node):
#     python bind_mpi_worker.py migrate     # seed shards from the old integrated.pkl
#
# Submit (you run this; Claude does not submit Slurm jobs):
#     mkdir -p /mnt/home/mlee1/ceph/logs
#     sbatch submit_bind_mpi.sh
#
# AFTER it finishes (cheap, run on a login node):
#     python bind_mpi_worker.py merge       # shards -> integrated.parquet / .pkl
#
# Env overrides: BIND_BUNDLE, BIND_COND_DIR (passed through to the worker).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# No MPI module needed: the reduction is embarrassingly parallel and uses srun's
# SPMD env (SLURM_PROCID / SLURM_NTASKS) to split work, not an MPI library.  This
# avoids the mpi4py<->launcher mismatch seen on this system.
source ~/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

echo "=== bind sobol reduce | job ${SLURM_JOB_ID} | nodes=${SLURM_NNODES} ranks=${SLURM_NTASKS} ==="
echo "    host=$(hostname)  start=$(date)"

# Heavy, restartable step.  srun launches ${SLURM_NTASKS} independent tasks; each
# reduces all_keys[SLURM_PROCID::SLURM_NTASKS], skipping keys that already have a
# shard.  Race-free (disjoint slices) and resumes after a preemption+requeue.
srun python bind_mpi_worker.py build

echo "=== done: $(date) ==="
echo "Next: run 'python bind_mpi_worker.py merge' on a login node to assemble the table."
