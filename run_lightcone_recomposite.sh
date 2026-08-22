#!/bin/bash
#SBATCH --job-name=bind_lc_repaste
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_lc_repaste_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_lc_repaste_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND lightcone, STAGE 2b (CPU-only): re-composite with circular apertures ─
# All 20 snapshots already have stage-2 composites with saved generated_patches.
# This re-runs only build_bind_composite with r200_factor=4.0 (circular, 4×R200c)
# — no GPU, no flow-matching sampler needed.  Output overwrites the existing
# composite_slab*.npz files in-place (safe: patches are loaded into RAM before
# the file is overwritten).
#
# Submit after verifying all snapshots have composite_slab*.npz with patches:
#   sbatch run_lightcone_recomposite.sh
#
# Then re-run lensplane generation:
#   sbatch --dependency=afterok:$JOBID run_lightcone_lensplane.sh
#
# Env overrides: OUTPUT_ROOT, R200_FACTOR.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Snapshot list (must match run_lightcone_project.sh) ───────────────────────
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)
SNAPSHOT=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
R200_FACTOR=${R200_FACTOR:-4.0}

SNAP3=$(printf '%03d' "$SNAPSHOT")
SNAP_DIR="$OUTPUT_ROOT/snap_${SNAP3}"
STAGE1_DIR="$SNAP_DIR/stage1"

if [[ ! -f "$STAGE1_DIR/stage1_manifest.json" ]]; then
    echo "ERROR: stage-1 manifest not found: $STAGE1_DIR/stage1_manifest.json" >&2
    exit 1
fi
if [[ ! -f "$SNAP_DIR/composite_slab00.npz" ]]; then
    echo "ERROR: no stage-2 composites in $SNAP_DIR — run run_lightcone_generate.sh first" >&2
    exit 1
fi

echo "=== BIND lightcone re-composite: array task ${SLURM_ARRAY_TASK_ID} / snapshot ${SNAP3} ==="
echo "    snap_dir=$SNAP_DIR"
echo "    stage1_dir=$STAGE1_DIR"
echo "    r200_factor=$R200_FACTOR  (circular paste, 4×R200c)"
echo "    output: overwrite in-place in $SNAP_DIR"

python -u -m bind.cli.paint_recomposite \
    --stage1_dir "$STAGE1_DIR" \
    --generated_dir "$SNAP_DIR" \
    --output_dir "$SNAP_DIR" \
    --r200_factor "$R200_FACTOR"

echo "=== Re-composite done for snap ${SNAP3}. ==="
