#!/bin/bash
#SBATCH --job-name=cv_sb35_inject
#SBATCH --output=/mnt/home/mlee1/ceph/logs/cv_sb35_inject_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/cv_sb35_inject_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Cross-injection experiment:
#   For each CV DMO halo (pre-extracted cutouts), run BIND with every SB35
#   parameter vector and compute gas P(k). Tests whether the CV P(k) bias
#   is driven by the DMO field structure or by the parameter conditioning.
#
# Outputs: $OUTPUT_ROOT/cv_dmo_sb35_params_injection.npz
#
# Override defaults via environment variables, e.g.:
#   CV_SIM_IDS=0,1,2 sbatch run_cv_sb35_params_injection.sh

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite}
RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_two_head}
MANIFEST=${MANIFEST:-$OUTPUT_ROOT/manifests/sb35_test_manifest.json}
OUTPUT=${OUTPUT:-$OUTPUT_ROOT/cv_dmo_sb35_params_injection.npz}

N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-auto}

# Limit to a few CV sims and SB35 vectors for a quick run;
# unset / set to empty string to use all.
CV_SIM_IDS=${CV_SIM_IDS:-}     # e.g. "0,1,2,3,4"
MAX_SB35=${MAX_SB35:-}         # e.g. "102" for all

EXTRA_FLAGS=()
[[ -n "$CV_SIM_IDS" ]] && EXTRA_FLAGS+=(--cv_sim_ids "$CV_SIM_IDS")
[[ -n "$MAX_SB35"   ]] && EXTRA_FLAGS+=(--max_sb35 "$MAX_SB35")

echo "=== CV × SB35 parameter injection ==="
echo "  RUN_DIR:  $RUN_DIR"
echo "  MANIFEST: $MANIFEST"
echo "  OUTPUT:   $OUTPUT"
echo "  DEVICE:   $DEVICE"
[[ -n "$CV_SIM_IDS" ]] && echo "  CV sims:  $CV_SIM_IDS" || echo "  CV sims:  all"
[[ -n "$MAX_SB35"   ]] && echo "  SB35:     first $MAX_SB35" || echo "  SB35:     all"

python cv_dmo_sb35_params_injection.py \
    --run_dir        "$RUN_DIR" \
    --manifest       "$MANIFEST" \
    --output         "$OUTPUT" \
    --n_steps        "$N_STEPS" \
    --batch_size     "$BATCH_SIZE" \
    --device         "$DEVICE" \
    "${EXTRA_FLAGS[@]}"

echo "=== Done ==="
