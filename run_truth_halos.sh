#!/bin/bash
#SBATCH --job-name=bind_truth
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_truth_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_truth_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --array=1-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND TRUTH lightcone: TNG300 hydro patches at the DMO halos (MPI, CPU) ─────
# Array index = snapshot.  Projects the actual hydro fields (DM_hydro, Gas, Stars
# + compton_y/T/entropy/P_e) at the SAME M>=10^13 DMO halos + lightcone transform
# BIND uses, into the composite_slab patch format -> runs/truth/run_0000/snap_*.
# Then the truth lightcone is built with the SAME pipeline:
#     DESIGN=truth sbatch --array=0-0 run_sobol_paste.sh
#     DESIGN=truth sbatch --array=0-0 run_sobol_lux.sh
#     DESIGN=truth sbatch --array=0-0 run_sobol_stats.sh
#
# Env overrides: OUTPUT_ROOT, HYDRO_ROOT, DMO_GROUPS_ROOT, STAGE1_ROOT.

set -euo pipefail

# MPI runtime ONLY (do NOT load python/python-mpi — their py3.10 h5py shadows the
# venv's py3.11 h5py and breaks the import). mpi4py is installed in the venv,
# built against this openmpi.
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
HYDRO_ROOT=${HYDRO_ROOT:-/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output}
DMO_GROUPS_ROOT=${DMO_GROUPS_ROOT:-/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output}
STAGE1_ROOT=${STAGE1_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
TRANSFORMS=${TRANSFORMS:-$STAGE1_ROOT/lightcone_transforms.json}
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

IDX=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID 0..19}
SNAP=${SNAPSHOTS[$IDX]}
SNAP3=$(printf '%03d' "$SNAP")
RUN_DIR="$OUTPUT_ROOT/runs/truth/run_0000"

srun python -u -m bind.cli.truth_halos \
    --hydro_snapshot "$HYDRO_ROOT" \
    --dmo_group_catalog "$DMO_GROUPS_ROOT/groups_${SNAP3}" \
    --snapshot_index "$SNAP" \
    --transforms "$TRANSFORMS" \
    --transforms_snap_idx "$IDX" \
    --output_dir "$RUN_DIR/snap_${SNAP3}"

echo "=== truth halos done for snap ${SNAP3} (task ${IDX}) ==="
