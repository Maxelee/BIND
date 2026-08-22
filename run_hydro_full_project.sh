#!/bin/bash
#SBATCH --job-name=bind_hydrofull_proj
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_hydrofull_proj_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_hydrofull_proj_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=FAIL

# ── TNG300 full-hydro lightcone slab projection (validation, stage A) ─────────
# One array task = one lightcone snapshot: stream all 600 hydro chunk files
# (gas + DM + stars + BH, ~1.5e10+1.45e10 particles), apply the shared lightcone
# transform, CIC-deposit into the native 4198^2 x 4-slab geometry, and write
# drop-in composite_slab{NN}.npz (full-box mass channels + full-box compton-y at
# the legacy comoving-area convention) for the paint/lux stages.
# Inputs are strictly read-only (sgenel's snapshots + the fiducial stage1).
#
# Env overrides: OUT_ROOT, HYDRO_ROOT, FID, Y_CONVENTION.

set -euo pipefail

# no module load: mpi4py lives in the venv (py3.11); python-mpi's h5py would clash
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

HYDRO_ROOT=${HYDRO_ROOT:-/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output}
FID=${FID:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
OUT_ROOT=${OUT_ROOT:-/mnt/home/mlee1/ceph/tng_full_validation}
Y_CONVENTION=${Y_CONVENTION:-physical}   # truth lightcone thermo is physical-area (truth_lightcone.py)
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

IDX=${SLURM_ARRAY_TASK_ID:?run as a SLURM array task 0..19}
S3=$(printf '%03d' "${SNAPSHOTS[$IDX]}")
OUT="$OUT_ROOT/composites/snap_${S3}"

if [[ -f "$OUT/summary.json" ]]; then
  echo "=== snap ${S3} already projected — skipping ==="; exit 0
fi
mkdir -p "$OUT"

echo "=== hydro-full projection | snap ${S3} (lc_snap_idx=$IDX) | $(date) ==="
srun -n "${SLURM_NTASKS}" python -u -m bind.cli.paint_project_hydro \
    --hydro_snapdir "$HYDRO_ROOT/snapdir_${S3}" --snapshot_index "${SNAPSHOTS[$IDX]}" \
    --stage1_dir "$FID/snap_${S3}/stage1" \
    --transforms "$FID/lightcone_transforms.json" --lc_snap_idx "$IDX" \
    --output_dir "$OUT" --y_convention "$Y_CONVENTION"
echo "=== snap ${S3} done | $(date) ==="
