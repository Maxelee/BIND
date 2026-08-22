#!/bin/bash
#SBATCH --job-name=bind_ksz_tau
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_ksz_tau_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_ksz_tau_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=icelake
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── P4 / D2-D3: BIND stacked tau + y profiles for the kSZ confrontation ────────
# Stacks per-halo tau (electron column) AND y (tSZ) profiles in x=R/r200c by halo
# mass, across the 256 SB35 Sobol runs, for the BGS-redshift (snap 85, z~0.18) and
# ELG-redshift (snap 46, z~1.16) slices -> input for examples/ksz_tau_gnfw.py
# --plot (vs Hadzhiyska+26 GNFW). One pass produces both legs.
#
# MPI (mpi4py) partitions the (snap, node) tasks across ranks; each rank writes a
# per-(snap,node) shard, so the job is embarrassingly parallel and RESTART-SAFE —
# a preemption just means resubmit; finished shards are skipped. A final
# single-rank step consolidates shards into the per-snap npz the plotter reads.
#
# Per the MPI-venv convention: load ONLY openmpi; mpi4py lives in the venv (do NOT
# load python-mpi — it ABI-clashes with the venv).
#
# Submit (you run this; Claude does not submit Slurm jobs):
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch run_ksz_tau.sh                    # MODE=mass: halo-mass-binned tau+y
#   MODE=mstar sbatch run_ksz_tau.sh         # central-M*-binned (Hadzhiyska match)
#   MODE=both  sbatch run_ksz_tau.sh         # both passes
# Resume after a preemption: just resubmit; finished shards are skipped.
# Subsets / recompute:
#   SNAPS="85" sbatch run_ksz_tau.sh        # one redshift slice
#   FORCE=1   sbatch run_ksz_tau.sh         # recompute all shards
#
# Env overrides: MODE (mass|mstar|both; default mass), SNAPS (comma-sep; default
# "85,46"; mstar only needs the BGS slice "85"), NODES (default "0-255"), FORCE=1.

set -euo pipefail

module load openmpi                                  # ONLY openmpi (mpi4py is in the venv)
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

MODE=${MODE:-mass}
SNAPS=${SNAPS:-85,46}
NODES=${NODES:-0-255}
FORCE_FLAG=""; [[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

run_pass () {                                        # $1=reduce flag, $2=consolidate flag, $3=label
    echo "=== ksz ${3} reduce | nodes=${SLURM_NNODES} tasks=${SLURM_NTASKS} | snaps='${SNAPS}' force='${FORCE:-0}' | $(date) ==="
    srun -n "${SLURM_NTASKS}" python -u examples/ksz_tau_gnfw.py "$1" --mpi \
        --snaps "${SNAPS}" --nodes "${NODES}" ${FORCE_FLAG}
    echo "=== consolidate (single rank) | $(date) ==="
    srun -N1 -n1 python -u examples/ksz_tau_gnfw.py "$2" --snaps "${SNAPS}" --nodes "${NODES}"
}

[[ "$MODE" == "mass"  || "$MODE" == "both" ]] && run_pass --reduce        --consolidate        "tau+y (halo-mass)"
[[ "$MODE" == "mstar" || "$MODE" == "both" ]] && run_pass --reduce-mstar  --consolidate-mstar  "tau+y (central-M*)"

echo "=== ksz done (MODE=$MODE) | $(date) ==="
ls -la /mnt/home/mlee1/ceph/bind_science/ksz_confront/bind_{tauy,mstar}_xprof_snap*.npz 2>/dev/null
