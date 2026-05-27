#!/bin/bash
#SBATCH --job-name=assembly_3d_mpi
#SBATCH --output=/mnt/home/mlee1/ceph/logs/assembly_3d_mpi_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/assembly_3d_mpi_%j.err
#SBATCH --time=1:30:00
#SBATCH --partition=preempt
#SBATCH -q preempt
#SBATCH --nodes=1
#SBATCH --ntasks=27            # one rank per CV sim (round-robin if fewer)
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# MPI version of the assembly attribution: the 27 independent CV sims are spread across ranks,
# each rank traces formation time + reads catalogs for its sims, results gathered to rank 0 to
# correlate + plot. Uses the MODULE python stack (NOT the torch3 venv) — module-built mpi4py/h5py
# must not be mixed with the venv. assembly_3d.py needs only numpy/scipy/h5py/matplotlib, all
# present in the module env (verified).
#
#   sbatch run_assembly_3d_mpi.sh
#   (or fewer ranks: edit --ntasks; sims are distributed round-robin)

set -euo pipefail

module -q purge
module -q load python openmpi python-mpi hdf5

cd /mnt/home/mlee1/vdm_bind2
export PYTHONUNBUFFERED=1

CV_MAX=${CV_MAX:-27}
M200_MIN=${M200_MIN:-1e13}
SNAP_MIN=${SNAP_MIN:-33}

mkdir -p /mnt/home/mlee1/ceph/logs figures/scatter_diagnostics outputs/scatter_diagnostics
echo "=== Assembly 3D (MPI): $SLURM_NTASKS ranks, CV_0..$((CV_MAX-1)), snap_min=$SNAP_MIN ==="
srun python -m scatter.assembly_3d --mpi \
    --cv-max "$CV_MAX" --m200-min "$M200_MIN" --snap-min "$SNAP_MIN"
echo "=== done -> outputs/scatter_diagnostics/assembly_3d.json + figures ==="
