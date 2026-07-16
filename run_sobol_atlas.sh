#!/bin/bash
#SBATCH --job-name=bind_sobol_atlas
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_sobol_atlas_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_sobol_atlas_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=icelake
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:30:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND SB35 Sobol per-halo atlas (CPU/MPI on the preempt queue) ──────────────
# Reduces every per-halo composite_slab*.npz across the 256 Sobol runs x 20
# snapshots into integrated aperture quantities (f_gas, Y-M, T/K/Pe, baryon
# closure), using the SAME reduction as halo_atlas.py so the Sobol caches are
# directly comparable to the existing twobound caches (bind_science/halo_atlas).
#
# MPI (mpi4py) partitions the 256x20 = 5120 (run, snap) tasks across ranks; each
# rank writes its own per-(run,snap) npz, so the job is embarrassingly parallel
# and RESTART-SAFE — a preemption just means resubmit, finished caches are
# skipped.  A final single-rank step consolidates the per-(run,snap) caches into
# notebook-ready per-snapshot cubes (256 sobol + 60 twobound + fid, halo-aligned)
# plus the design matrix.
#
# Per the MPI-venv convention: load ONLY openmpi; mpi4py lives in the venv (do
# NOT load python-mpi — it ABI-clashes with the venv).
#
# Submit (you run this; Claude does not submit Slurm jobs):
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch run_sobol_atlas.sh
# Resume after a preemption: just `sbatch run_sobol_atlas.sh` again.
# Subsets / recompute:
#   SNAPS="96 90 85" sbatch run_sobol_atlas.sh        # only some snapshots
#   FORCE=1 sbatch run_sobol_atlas.sh                 # recompute all caches
#
# Env overrides: SNAPS (space-separated; default all 20), FORCE=1.

set -euo pipefail

module load openmpi                                  # ONLY openmpi (mpi4py is in the venv)
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

SNAPS=${SNAPS:-}                                     # empty -> script default (all 20)
FORCE_FLAG=""; [[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

echo "=== sobol atlas reduce | nodes=${SLURM_NNODES} tasks=${SLURM_NTASKS} | snaps='${SNAPS:-all}' force='${FORCE:-0}' | $(date) ==="
# shellcheck disable=SC2086  # SNAPS / FORCE_FLAG are intentional word-split args
srun -n "${SLURM_NTASKS}" python -u examples/sobol_atlas_mpi.py --reduce \
    ${SNAPS:+--snaps $SNAPS} $FORCE_FLAG

echo "=== consolidate -> per-snap cubes (single rank) | $(date) ==="
# shellcheck disable=SC2086
srun -N1 -n1 python -u examples/sobol_atlas_mpi.py --consolidate ${SNAPS:+--snaps $SNAPS}

echo "=== sobol atlas done | $(date) ==="
ls -la /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/ 2>/dev/null | tail -5
