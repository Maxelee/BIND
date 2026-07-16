#!/bin/bash
#SBATCH --job-name=wp2_truth_val
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp2_truth_val_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp2_truth_val_%A_%a.err
#SBATCH --partition=cca
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A2 Popeye half: TNG300 truth validation (plan task 5-6) ────────────────
# One array task per validation snapshot:  0->096  1->071  2->067  3->063
# (z ~ 0.03 / 0.42 / 0.50 / 0.60 — the last three double as the z>0 thermo
# verdict, SHARED_CONTEXT caveat 6). CPU only. NO GPU.
#
# PARALLEL (MPI): each array task is an OpenMPI job — 3 nodes x 48 ranks = 144
# ranks. The two ~600-file map-reduces (transformed-frame slab projection and
# the CylToSph cKDTree pass) stride their files across all ranks and
# MPI-Reduce to rank 0; the operator/bootstrap midsection is rank-0 serial.
# Per-rank memory ~= a pool worker (1.7 GB slab accumulator + file transient),
# so 48 ranks/node fits well inside --mem=360G on the 768 GB cca nodes.
# Launched without mpirun the old single-node ProcessPool path still works
# (set BIND_NO_MPI=1 to force it even under mpirun).
#
# NO PAINT NEEDED: the painted side defaults to the EXISTING fiducial paint at
# bind_lightcone_tng (per-halo composites + co-located stage1 conditions). The
# painted-side path is smoke-verified on those real composites for all 4 snaps.
# (A fresh paint is only needed for the >=8-draw multi-sample — task 6 — which is
# OFF by default here; enable it by setting MULTISAMPLE_ROOT to an 8-draw paint.)
# Restart-safe at snapshot granularity (each writes its own summary + sigma_model).
#
# ⛔ HUMAN CHECKPOINT — submit by hand:
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch /mnt/home/mlee1/BIND-paper3a/analysis/paper3a/scripts/run_wp2_truth_validation.sh
#
# This runs in paper3b_popeye (CPU only — no CUDA init, so its cu130 torch is
# fine here, unlike the paint). mpi4py comes from the python-mpi MODULE, not
# the venv — hence the module loads + PYTHONPATH re-order below.

set -euo pipefail

# mpi4py lives in the python-mpi module (built against openmpi/4.1.8). Load
# BEFORE activating the venv so the venv's bin/ wins on PATH.
module load python openmpi python-mpi

VENV=${VENV:-/mnt/home/mlee1/venvs/paper3b_popeye}
source "$VENV/bin/activate"

# python-mpi's PYTHONPATH dir also ships its own h5py, and PYTHONPATH beats
# venv site-packages — prepend the venv so ONLY mpi4py (absent from the venv)
# falls through to the module; h5py/numpy/scipy stay the venv's tested builds.
export PYTHONPATH="$VENV/lib/python3.11/site-packages:${PYTHONPATH:-}"

mkdir -p /mnt/home/mlee1/ceph/logs

# One thread per rank — parallelism is rank-level (one MPI rank per core);
# multithreaded BLAS on top of 48 ranks/node would oversubscribe the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mlee1/BIND-paper3a
cd "$REPO"

# Painted composites + their stage1 conditions both come from the existing
# fiducial paint (the driver's defaults); override only to point at a re-paint.
PAINTED_ROOT=${PAINTED_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
CONDITIONS_ROOT=${CONDITIONS_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
OUT_DIR=${OUT_DIR:-/mnt/home/mlee1/ceph/paper3/A/wp2_validation}

ARGS=(--array-index "$SLURM_ARRAY_TASK_ID"
      --painted-root "$PAINTED_ROOT"
      --conditions-root "$CONDITIONS_ROOT"
      --stage1-subdir stage1
      --out-dir "$OUT_DIR")
# Opt-in task 6: set MULTISAMPLE_ROOT to a dir with snap_063_s{0..7} draws.
[[ -n "${MULTISAMPLE_ROOT:-}" ]] && ARGS+=(--multisample-root "$MULTISAMPLE_ROOT")

# mpirun (not srun): works regardless of the cluster's Slurm PMI default and
# forwards the venv environment to all ranks under the Slurm allocation.
mpirun -np "$SLURM_NTASKS" python -u analysis/paper3a/scripts/run_wp2_truth_validation.py "${ARGS[@]}"

echo "=== WP-A2 truth validation task ${SLURM_ARRAY_TASK_ID} done ==="
