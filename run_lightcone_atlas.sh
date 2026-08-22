#!/bin/bash
#SBATCH --job-name=bind_lc_atlas
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_lc_atlas_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_lc_atlas_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=icelake
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND lightcone map-statistics atlas (CPU/MPI on the preempt queue) ─────────
# Reads each run's {kappa,y,tau}_maps.npz (n_real, n_src, 1024, 1024) and computes
# a uniform, response-correct statistics vector with bind.inference.stats — power
# spectra (kappa tomographic, kxy, yy, kxtau, tautau, yxtau), non-Gaussian (PDF,
# moments, Minkowski V0/V1/V2), peak/minima counts, and tSZ-at-peaks R(nu).  Peaks
# use nu_norm='fixed' wrt the FIDUCIAL map rms so cross-run differences are the
# parametric response, not absorbed by per-map sigma_kappa.
#
# MPI partitions the runs (256 sobol + 60 twobound + fid) across ranks; each rank
# writes its own per-run npz.  RESTART-SAFE and INCREMENTAL: a run whose cache
# exists is skipped, and a run not yet ray-traced is skipped (no map) — so this can
# be run repeatedly as the Sobol raytrace lands runs, then a final pass + the
# single-rank consolidate builds the cube.
#
# Per the MPI-venv convention: load ONLY openmpi; mpi4py is in the venv.
#
# Submit (you run this; Claude does not submit Slurm jobs):
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch run_lightcone_atlas.sh
# Re-run as more Sobol runs finish ray-tracing (skips done, picks up new):
#   sbatch run_lightcone_atlas.sh
# Subsets / recompute:
#   SETS="sobol" sbatch run_lightcone_atlas.sh        # only the Sobol runs
#   FORCE=1 sbatch run_lightcone_atlas.sh             # recompute all caches
#
# Env overrides: SETS (subset of "sobol tb fid"; default all three), FORCE=1.

set -euo pipefail

module load openmpi                                  # ONLY openmpi (mpi4py in venv)
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

SETS=${SETS:-sobol tb fid}
FORCE_FLAG=""; [[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

echo "=== lc atlas recompute | nodes=${SLURM_NNODES} tasks=${SLURM_NTASKS} | sets='${SETS}' force='${FORCE:-0}' | $(date) ==="
# shellcheck disable=SC2086  # SETS / FORCE_FLAG are intentional word-split args
srun -n "${SLURM_NTASKS}" python -u examples/lightcone_atlas_mpi.py --recompute \
    --sets $SETS $FORCE_FLAG

echo "=== consolidate -> lightcone_atlas.npz (single rank) | $(date) ==="
srun -N1 -n1 python -u examples/lightcone_atlas_mpi.py --consolidate

echo "=== lc atlas done | $(date) ==="
ls -la /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/lightcone_cubes/ 2>/dev/null
