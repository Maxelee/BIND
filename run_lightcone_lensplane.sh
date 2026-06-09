#!/bin/bash
#SBATCH --job-name=bind_lc_lensplane
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_lc_lensplane_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_lc_lensplane_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND lightcone, STAGE 3 (CPU): composite mass maps → lux lensplanes ────────
# Converts the per-slab composite_slab*.npz files written by stage 2 into lux
# lenspot{PP:02d}.dat files.  Array task 0 additionally writes config.dat.
#
# Three-stage submit chain:
#   jid1=$(sbatch --parsable run_lightcone_project.sh)
#   jid2=$(sbatch --parsable --dependency=aftercorr:$jid1 run_lightcone_generate.sh)
#   sbatch --dependency=aftercorr:$jid2 run_lightcone_lensplane.sh
#
# Env overrides: OUTPUT_ROOT, LENSPLANE_DIR, TRANSFORMS_FILE, LC_SEED,
#                PLANES_PER_SNAPSHOT, SNAP_STACK.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Snapshot list (must match run_lightcone_project.sh, ordered low-z→high-z) ──
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)
SNAPSHOT=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}
N_SNAPS=${#SNAPSHOTS[@]}

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
LENSPLANE_DIR=${LENSPLANE_DIR:-$OUTPUT_ROOT/lensplanes}
TRANSFORMS_FILE=${TRANSFORMS_FILE:-$OUTPUT_ROOT/lightcone_transforms.json}
PLANES_PER_SNAPSHOT=${PLANES_PER_SNAPSHOT:-}   # empty = infer from manifest n_slabs
# SNAP_STACK: comma-separated true/false per snapshot; empty = all false
# Example for lux.ini convention (first 10 not stacked, last 10 stacked):
#   SNAP_STACK="false,false,false,false,false,false,false,false,false,false,true,true,true,true,true,true,true,true,true,true"
SNAP_STACK=${SNAP_STACK:-}

SNAP3=$(printf '%03d' "$SNAPSHOT")
STAGE1_DIR="$OUTPUT_ROOT/snap_${SNAP3}/stage1"
GENERATE_DIR="$OUTPUT_ROOT/snap_${SNAP3}"

if [[ ! -f "$STAGE1_DIR/stage1_manifest.json" ]]; then
    echo "ERROR: stage-1 manifest not found: $STAGE1_DIR/stage1_manifest.json" >&2
    exit 1
fi

if [[ ! -f "$TRANSFORMS_FILE" ]]; then
    echo "ERROR: transforms file not found: $TRANSFORMS_FILE" >&2
    echo "       Run run_lightcone_project.sh first." >&2
    exit 1
fi

mkdir -p "$LENSPLANE_DIR"

echo "=== BIND lensplane: array task ${SLURM_ARRAY_TASK_ID} / snapshot ${SNAP3} ==="
echo "    generate_dir=$GENERATE_DIR"
echo "    stage1_dir=$STAGE1_DIR"
echo "    lensplane_dir=$LENSPLANE_DIR"
echo "    transforms=$TRANSFORMS_FILE"
echo "    lc_snap_idx=${SLURM_ARRAY_TASK_ID} / lc_n_snaps=$N_SNAPS"

# ── Build --all_snap_dirs list (only needed for task 0 which writes config.dat) ─
ALL_SNAP_DIRS_ARGS=()
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
    for s in "${SNAPSHOTS[@]}"; do
        SP3=$(printf '%03d' "$s")
        ALL_SNAP_DIRS_ARGS+=("$OUTPUT_ROOT/snap_${SP3}")
    done
fi

# ── Build optional flags ───────────────────────────────────────────────────────
OPT_ARGS=()
if [[ -n "$PLANES_PER_SNAPSHOT" ]]; then
    OPT_ARGS+=(--planes_per_snapshot "$PLANES_PER_SNAPSHOT")
fi
if [[ -n "$SNAP_STACK" ]]; then
    OPT_ARGS+=(--snap_stack "$SNAP_STACK")
fi
if [[ ${#ALL_SNAP_DIRS_ARGS[@]} -gt 0 ]]; then
    OPT_ARGS+=(--all_snap_dirs "${ALL_SNAP_DIRS_ARGS[@]}")
fi

python -u -m bind.cli.paint_lensplane \
    --generate_dir "$GENERATE_DIR" \
    --stage1_dir "$STAGE1_DIR" \
    --output_dir "$LENSPLANE_DIR" \
    --transforms "$TRANSFORMS_FILE" \
    --lc_snap_idx "${SLURM_ARRAY_TASK_ID}" \
    --lc_n_snaps "$N_SNAPS" \
    --lp_grid "${LP_GRID:-4096}" \
    "${OPT_ARGS[@]}"

echo "=== Lensplane done for snap ${SNAP3} (task ${SLURM_ARRAY_TASK_ID}). ==="
