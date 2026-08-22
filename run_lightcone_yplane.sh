#!/bin/bash
#SBATCH --job-name=bind_yplane
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_lc_yplane_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_lc_yplane_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL


set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
LENSPLANE_DIR=${LENSPLANE_DIR:-$OUTPUT_ROOT/lensplanes}
N_SNAPS=${N_SNAPS:-20}
LP_GRID=${LP_GRID:-4096}
R200_FACTOR=${R200_FACTOR:-4.0}
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs "$LENSPLANE_DIR"

IDX=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID 0..19}
SNAP=${SNAPSHOTS[$IDX]}
SNAP3=$(printf '%03d' "$SNAP")

srun python -u -m bind.cli.paint_yplane \
    --generate_dir "$OUTPUT_ROOT/snap_${SNAP3}" \
    --output_dir   "$LENSPLANE_DIR" \
    --lc_snap_idx  "$IDX" \
    --lc_n_snaps   "$N_SNAPS" \
    --lp_grid      "$LP_GRID" \
    --r200_factor  "$R200_FACTOR"

echo "=== y-planes done for snap ${SNAP3} (task ${IDX}). ==="
