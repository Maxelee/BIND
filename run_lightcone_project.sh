#!/bin/bash
#SBATCH --job-name=bind_lc_project
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_lc_project_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_lc_project_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
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

# ── Lightcone transforms (rotation/translation/flip per snapshot) ──────────────
# Generated once by task 0; all other tasks wait for the JSON to appear.
LC_SEED=${LC_SEED:-2020}
BOX_SIZE=205.0   # IllustrisTNG L205 in Mpc/h
N_SNAPS=20
TRANSFORMS_FILE="$OUTPUT_ROOT/lightcone_transforms.json"

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
    python - "$TRANSFORMS_FILE" "$LC_SEED" "$BOX_SIZE" "$N_SNAPS" <<'PY'
import sys
from bind.inference.lightcone_transforms import LightconeTransforms

out_path, seed, box_size, n_snaps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
t = LightconeTransforms.from_seed(n_snaps, seed, box_size, random_proj_dir=True)
t.save(out_path)
print(f"[transforms] seed={seed} n_snaps={n_snaps} box={box_size} Mpc/h -> {out_path}")
print(f"[transforms] proj_dirs={t.proj_dirs.tolist()}")
PY
else
    # Wait for task 0 to write the transforms file (up to 5 min)
    echo "[transforms] waiting for $TRANSFORMS_FILE ..."
    for i in $(seq 1 60); do
        [[ -f "$TRANSFORMS_FILE" ]] && break
        sleep 5
    done
    if [[ ! -f "$TRANSFORMS_FILE" ]]; then
        echo "ERROR: transforms file never appeared: $TRANSFORMS_FILE" >&2
        exit 1
    fi
fi

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
echo "    transforms=$TRANSFORMS_FILE  snap_idx=${SLURM_ARRAY_TASK_ID}"

srun python -u -m bind.cli.paint_project \
    --snapshot "$SNAPDIR" \
    --group_catalog "$GROUPDIR" \
    --snapshot_index "$SNAPSHOT" \
    --params "$PARAMS_FILE" \
    --output_dir "$STAGE1_DIR" \
    --halo_mass_min "$HALO_MASS_MIN" \
    --transforms "$TRANSFORMS_FILE" \
    --transforms_snap_idx "${SLURM_ARRAY_TASK_ID}"

echo "=== Stage 1 done for snap ${SNAP3}. Intermediate in $STAGE1_DIR ==="
