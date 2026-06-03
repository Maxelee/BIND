#!/bin/bash
#SBATCH --job-name=bind_tng_generate
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_tng_generate_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_tng_generate_%j.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# ── BIND paint, STAGE 2 (GPU): generation + compositing ───────────────────────
# Consumes the stage-1 intermediate from run_paint_tng_project.sh: loads the saved
# per-halo DMO cutouts, runs the flow-matching sampler on the GPU, composites the
# painted patches back into per-slab maps, and writes composite_slab{NN}.npz +
# summary.json (identical artifacts to bind-paint).  No particle I/O happens here,
# so this is light on memory — the heavy load/projection is already done on CPU.
#
#   STAGE1_DIR=/path/to/stage1 sbatch run_paint_tng_generate.sh
#
# Or chained after stage 1:
#   jid=$(sbatch --parse run_paint_tng_project.sh)
#   sbatch --dependency=afterok:$jid run_paint_tng_generate.sh
#
# Env overrides: SNAPSHOT, STAGE1_DIR, RUN_DIR, OUTPUT_DIR, N_STEPS, BATCH_SIZE.
# Any extra args pass straight through to bind.cli.paint_generate.

set -euo pipefail

# BIND_env's torch is a CUDA 12.5 build (bundles its own runtime), so no `cuda`
# module is needed.  No MPI here — generation is single-process on one GPU.
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Configuration ─────────────────────────────────────────────────────────────
SNAPSHOT=${SNAPSHOT:-99}
RUN_DIR=${RUN_DIR:-weights/fm_two_head}
N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}

SNAP3=$(printf '%03d' "$SNAPSHOT")
STAGE1_DIR=${STAGE1_DIR:-/mnt/home/mlee1/ceph/bind_illustristng/snap_${SNAP3}/stage1}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/bind_illustristng/snap_${SNAP3}}

if [[ ! -f "$STAGE1_DIR/stage1_manifest.json" ]]; then
    echo "ERROR: stage-1 manifest not found: $STAGE1_DIR/stage1_manifest.json" >&2
    echo "       Run run_paint_tng_project.sh first (or set STAGE1_DIR)." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== BIND paint stage 2 (generate): snapshot ${SNAP3} ==="
echo "    stage1_dir=$STAGE1_DIR"
echo "    run_dir=$RUN_DIR  n_steps=$N_STEPS  batch_size=$BATCH_SIZE"
echo "    output=$OUTPUT_DIR"

python -u -m bind.cli.paint_generate \
    --stage1_dir "$STAGE1_DIR" \
    --run_dir "$RUN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --device auto \
    "$@"

echo "=== Done. Output in $OUTPUT_DIR ==="
