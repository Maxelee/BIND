#!/bin/bash
#SBATCH --job-name=bind_tng
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_tng_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_tng_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# Apply BIND to the full IllustrisTNG_DM box (TNG300-1-Dark, 205 Mpc/h,
# ~2.74e9 DM particles, 75 chunk files) at a single snapshot via bind.paint().
#
# DMO snapshot + FoF catalog -> per-halo painted hydro patches composited back
# into per-z-slab maps ([DM_hydro, Gas, Stars]).  The model conditions on the
# 35-dim fiducial IllustrisTNG parameter vector (the full TNG box was run at
# fiducial feedback), which this script writes out before painting.
#
# Default snapshot is 99 (z~=0), the only snapshot currently staged in SIM_ROOT
# and the redshift the main-branch fm_two_head model was trained at.  Override
# SNAPSHOT once another snapshot (e.g. 90) has been transferred to SIM_ROOT.
#
#   sbatch run_paint_tng.sh
#   SNAPSHOT=90 sbatch run_paint_tng.sh          # once snapdir_090/groups_090 exist
#   HALO_MASS_MIN=5e12 N_STEPS=20 sbatch run_paint_tng.sh
#
# Env overrides: SIM_ROOT, SNAPSHOT, RUN_DIR, OUTPUT_DIR, HALO_MASS_MIN,
# N_STEPS, BATCH_SIZE.  Any extra args pass straight through to bind.cli.paint.
#
# NOTE: reading ~33 GB of particle positions across 75 chunks is memory-heavy;
# --mem=256G leaves headroom for the concatenate peak + the z-slab projection.

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Configuration ─────────────────────────────────────────────────────────────
SIM_ROOT=${SIM_ROOT:-/mnt/home/mlee1/ceph/IllustrisTNG_DM/output}
SNAPSHOT=${SNAPSHOT:-99}
RUN_DIR=${RUN_DIR:-weights/fm_two_head}
HALO_MASS_MIN=${HALO_MASS_MIN:-1e13}
N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}

SNAP3=$(printf '%03d' "$SNAPSHOT")
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/bind_illustristng/snap_${SNAP3}}

SNAPDIR="$SIM_ROOT/snapdir_${SNAP3}"
GROUPDIR="$SIM_ROOT/groups_${SNAP3}"

if [[ ! -d "$SNAPDIR" ]]; then
    echo "ERROR: snapshot directory not found: $SNAPDIR" >&2
    echo "       (override SIM_ROOT/SNAPSHOT, or stage the snapshot first)" >&2
    exit 1
fi
if [[ ! -d "$GROUPDIR" ]]; then
    echo "ERROR: group catalog directory not found: $GROUPDIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── 35-dim fiducial IllustrisTNG parameter vector ─────────────────────────────
PARAMS_FILE="$OUTPUT_DIR/fiducial_params.npy"
python - "$PARAMS_FILE" <<'PY'
import sys
import numpy as np
import bind

np.save(sys.argv[1], bind.fiducial_params().astype(np.float64))
print(f"[params] wrote fiducial 35-dim vector -> {sys.argv[1]}")
PY

# ── Paint ─────────────────────────────────────────────────────────────────────
echo "=== BIND paint IllustrisTNG_DM snapshot ${SNAP3} ==="
echo "    snapshot=$SNAPDIR"
echo "    groups=$GROUPDIR"
echo "    run_dir=$RUN_DIR  halo_mass_min=$HALO_MASS_MIN  n_steps=$N_STEPS"
echo "    output=$OUTPUT_DIR"

python -m bind.cli.paint \
    --snapshot "$SNAPDIR" \
    --group_catalog "$GROUPDIR" \
    --snapshot_index "$SNAPSHOT" \
    --params "$PARAMS_FILE" \
    --run_dir "$RUN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --halo_mass_min "$HALO_MASS_MIN" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --device auto \
    "$@"

echo "=== Done. Output in $OUTPUT_DIR ==="
