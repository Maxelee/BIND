#!/bin/bash
#SBATCH --job-name=bind_lc_project
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_lc_project_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_lc_project_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=rome
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND lightcone, STAGE 1 (CPU / MPI): project + halos + cutouts ────────────
# Processes 20 IllustrisTNG DM snapshots in a SLURM array.  Each array task
# reads one snapshot, projects particles into z-slab maps via MPI, extracts
# per-halo DMO cutouts, and writes stage1_slab*.npz + stage1_manifest.json.
# The scale factor a=1/(1+z) is recorded in the manifest from the snapshot
# Header/Time attribute so stage 2 can condition on the correct redshift.
#
# Submit both stages as a dependency chain:
#   jid=$(sbatch --parsable run_lightcone_project.sh)
#   sbatch --dependency=aftercorr:$jid run_lightcone_generate.sh
#
# Env overrides: SIM_ROOT, OUTPUT_ROOT, HALO_MASS_MIN.

set -euo pipefail

module load python openmpi python-mpi
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Snapshot list (20 snapshots for the lightcone) ────────────────────────────
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)
SNAPSHOT=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

# ── Configuration ─────────────────────────────────────────────────────────────
SIM_ROOT=${SIM_ROOT:-/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
HALO_MASS_MIN=${HALO_MASS_MIN:-1e13}

SNAP3=$(printf '%03d' "$SNAPSHOT")
STAGE1_DIR="$OUTPUT_ROOT/snap_${SNAP3}/stage1"

SNAPDIR="$SIM_ROOT/snapdir_${SNAP3}"
GROUPDIR="$SIM_ROOT/groups_${SNAP3}"

if [[ ! -d "$SNAPDIR" ]]; then
    echo "ERROR: snapshot directory not found: $SNAPDIR" >&2
    exit 1
fi
if [[ ! -d "$GROUPDIR" ]]; then
    echo "ERROR: group catalog directory not found: $GROUPDIR" >&2
    exit 1
fi

mkdir -p "$STAGE1_DIR"

# ── 35-dim fiducial IllustrisTNG parameter vector ─────────────────────────────
PARAMS_FILE="$STAGE1_DIR/fiducial_params.npy"
python - "$PARAMS_FILE" <<'PY'
import sys
import numpy as np
import bind

np.save(sys.argv[1], bind.fiducial_params().astype(np.float64))
print(f"[params] wrote fiducial 35-dim vector -> {sys.argv[1]}")
PY

# ── Stage 1: MPI projection + cutouts ─────────────────────────────────────────
echo "=== BIND lightcone stage 1 (project): array task ${SLURM_ARRAY_TASK_ID} / snapshot ${SNAP3} ==="
echo "    snapshot=$SNAPDIR"
echo "    groups=$GROUPDIR"
echo "    halo_mass_min=$HALO_MASS_MIN  ntasks=${SLURM_NTASKS:-?}"
echo "    stage1_out=$STAGE1_DIR"

srun python -u -m bind.cli.paint_project \
    --snapshot "$SNAPDIR" \
    --group_catalog "$GROUPDIR" \
    --snapshot_index "$SNAPSHOT" \
    --params "$PARAMS_FILE" \
    --output_dir "$STAGE1_DIR" \
    --halo_mass_min "$HALO_MASS_MIN" \
    "$@"

echo "=== Stage 1 done for snap ${SNAP3}. Intermediate in $STAGE1_DIR ==="
