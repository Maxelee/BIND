#!/bin/bash
#SBATCH --job-name=bind_lc_generate
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_lc_generate_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_lc_generate_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND lightcone, STAGE 2 (GPU): generation + compositing ───────────────────
# Runs the fm_redshift_thermo flow-matching sampler for each of the 20 lightcone
# snapshots.  The scale factor a=1/(1+z) is read automatically from the
# stage1_manifest.json written by run_lightcone_project.sh, so no manual
# redshift specification is needed.
#
# Submit as a dependency on the project array:
#   jid=$(sbatch --parsable run_lightcone_project.sh)
#   sbatch --dependency=aftercorr:$jid run_lightcone_generate.sh
#
# Or run standalone after stage 1 is complete:
#   sbatch run_lightcone_generate.sh
#
# Env overrides: OUTPUT_ROOT, RUN_DIR, N_STEPS, BATCH_SIZE.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Snapshot list (must match run_lightcone_project.sh) ───────────────────────
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)
SNAPSHOT=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
RUN_DIR=${RUN_DIR:-weights/fm_redshift_thermo}
N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}

SNAP3=$(printf '%03d' "$SNAPSHOT")
STAGE1_DIR="$OUTPUT_ROOT/snap_${SNAP3}/stage1"
OUTPUT_DIR="$OUTPUT_ROOT/snap_${SNAP3}"

if [[ ! -f "$STAGE1_DIR/stage1_manifest.json" ]]; then
    echo "ERROR: stage-1 manifest not found: $STAGE1_DIR/stage1_manifest.json" >&2
    echo "       Run run_lightcone_project.sh first (or check OUTPUT_ROOT)." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== BIND lightcone stage 2 (generate): array task ${SLURM_ARRAY_TASK_ID} / snapshot ${SNAP3} ==="
echo "    stage1_dir=$STAGE1_DIR"
echo "    run_dir=$RUN_DIR  n_steps=$N_STEPS  batch_size=$BATCH_SIZE"
echo "    output=$OUTPUT_DIR"
echo "    (redshift read from stage1 manifest)"

python -u -m bind.cli.paint_generate \
    --stage1_dir "$STAGE1_DIR" \
    --run_dir "$RUN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --device auto \
    "$@"

echo "=== Done for snap ${SNAP3}. Output in $OUTPUT_DIR ==="
