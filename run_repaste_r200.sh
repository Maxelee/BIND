#!/bin/bash
#SBATCH --job-name=fm_repaste_r200
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm_repaste_r200_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm_repaste_r200_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-9   # set to 0-(N_CHUNKS-1); override with --array at sbatch time

# Repaste CV + 1P test suites with a circular 2×R200c mask.
# Reuses existing generated_halos.npz — no model inference needed.
# Overwrites composite.npz in-place (square → circular) so downstream
# notebooks that load composite.npz get the updated maps automatically.

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

# ── Configuration ─────────────────────────────────────────────────────────────
N_CHUNKS=${N_CHUNKS:-10}
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
TEST_MANIFEST="$MANIFEST_DIR/sb35_test_manifest.json"

RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_two_head}
MODEL_NAME=${MODEL_NAME:-fm_two_head_no_pmm}

SNAPSHOT=${SNAPSHOT:-90}
NPIX=${NPIX:-1024}
PATCH_PIX=${PATCH_PIX:-128}
HALO_MASS_MIN=${HALO_MASS_MIN:-1e13}

# Circular paste radius as a multiple of R200c
R200_FACTOR=${R200_FACTOR:-2.0}

DEVICE=${DEVICE:-auto}

CV_FOF_ROOT=${CV_FOF_ROOT:-/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/CV}
ONEP_FOF_ROOT=${ONEP_FOF_ROOT:-/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/1P}

mkdir -p /mnt/home/mlee1/ceph/logs

if [[ ! -f "$TEST_MANIFEST" ]]; then
    echo "ERROR: manifest not found at $TEST_MANIFEST — run run_test_suite_parallel.sh first" >&2
    exit 1
fi

echo "=== [chunk $CHUNK_ID/$N_CHUNKS] Circular repaste at ${R200_FACTOR}×R200c ==="

/mnt/home/mlee1/venvs/torch3/bin/python run_test_suite.py \
    --suite all \
    --test_manifest "$TEST_MANIFEST" \
    --snapshot "$SNAPSHOT" \
    --npix "$NPIX" \
    --patch_pix "$PATCH_PIX" \
    --halo_mass_min "$HALO_MASS_MIN" \
    --run_dir "$RUN_DIR" \
    --model_name "$MODEL_NAME" \
    --output_root "$OUTPUT_ROOT" \
    --device "$DEVICE" \
    --max_workers 1 \
    --cv_fof_root "$CV_FOF_ROOT" \
    --onep_fof_root "$ONEP_FOF_ROOT" \
    --n_chunks "$N_CHUNKS" \
    --chunk_id "$CHUNK_ID" \
    --repaste \
    --r200_factor "$R200_FACTOR" \
    --skip_truth \
    "$@"

echo "=== [chunk $CHUNK_ID] Done ==="
