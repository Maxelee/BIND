#!/bin/bash
#SBATCH --job-name=enh_closure
#SBATCH --output=/mnt/home/mlee1/ceph/logs/enh_closure_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/enh_closure_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# Enhancement-branch closure test: does real CAMELS-TNG hydro (L50n512, SB35,
# weak-feedback corner) ENHANCE small-scale matter power relative to its
# paired DMO run, and does BIND reproduce it at the same Sobol nodes?
# Runs examples/enhancement_closure.py -- see that file's module docstring
# for the full pipeline/provenance writeup; this wrapper only supplies the
# SLURM resources + env-var overrides, mirroring run_test_suite.sh /
# papers/01_pipeline/run_figures.sbatch conventions.
#
# *** THIS JOB MUST RUN ON RUSTY, NOT POPEYE. *** The CAMELS raw data
# (SB35_DM_ROOT/SB35_HYDRO_ROOT/SB35_GROUP_ROOT below) lives on Rusty ceph;
# the recon session that produced this script ran on Popeye (sdceph), where
# none of it resolves. The script's verification step (run first, below)
# re-checks all of this on the compute node and aborts loudly (exit 2)
# before any GPU work is requested if the data still isn't reachable.
#
# Model weights: RUN_DIR/MODEL_NAME/CHECKPOINT_PATH, same convention as
# run_test_suite.sh. Default RUN_DIR points at this repo's released weights
# (weights/fm_two_head) -- populate with `bind-download-weights --run
# fm_two_head` on Rusty if not already present, or override to reuse
# whatever checkpoint is already on disk, e.g.:
#   RUN_DIR=/mnt/home/mlee1/BIND/weights/fm_redshift_thermo \
#   MODEL_NAME=fm_redshift_thermo sbatch run_enhancement_closure.sbatch
#
#   sbatch run_enhancement_closure.sbatch
#   SIMS=SB35_5,SB35_1004 EXISTING_EVAL_ROOT=/mnt/home/mlee1/ceph/fm_testsuite \
#       sbatch run_enhancement_closure.sbatch

set -euo pipefail

VENV=${VENV:-/mnt/home/mlee1/venvs/BIND_env}
source "$VENV/bin/activate"
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Configuration (all overridable at `sbatch` time via env vars) ─────────────
SIMS=${SIMS:-}                                   # empty -> script's own hardcoded default list
SNAPSHOT=${SNAPSHOT:-90}
BOX_SIZE=${BOX_SIZE:-50.0}
NPIX=${NPIX:-1024}
PATCH_PIX=${PATCH_PIX:-128}
HALO_MASS_MIN=${HALO_MASS_MIN:-1e13}

# SB35 raw-data roots -- recorded in run_test_suite.sh, same variable names so
# a corrected path can be injected here without editing either script. Left
# UNSET by default (not given a bash-side default) so
# examples/enhancement_closure.py's own candidate-probing runs: it already
# tries these exact run_test_suite.sh paths first, then a /mnt/ceph twin for
# the two that live under the ~/Sims symlink, then (for the param file only)
# the bundled package asset. Set any of the four before `sbatch` to override
# with one explicit path and skip probing for that root.

# Prior bind-camels-suite eval outputs to reuse (skip repainting sims already
# evaluated there) + where this run's fresh paints/results land.
EXISTING_EVAL_ROOT=${EXISTING_EVAL_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/enhancement_closure}

# Model selection -- run_test_suite.sh's own recorded default is fm_two_head
# (RUN_DIR under its old ceph fm_runs/ tree); here RUN_DIR instead defaults to
# this repo's released weights/ layout (flat: <run>/last.ckpt).
RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/BIND/weights/fm_two_head}
MODEL_NAME=${MODEL_NAME:-fm_two_head}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}

N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}
DEVICE=${DEVICE:-auto}

mkdir -p "$OUTPUT_ROOT"

# cuDNN lazy-dlopen fix -- the nix torch build dlopen()s libcudnn_cnn_train.so.8
# by bare soname at backward time and DT_RUNPATH is not consulted; put torch's
# cuDNN dir on LD_LIBRARY_PATH (full story in run_wlemu_train.sh).
CUDNN_DIR=$(python - <<'PY'
import os, subprocess, torch
tlib = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so")
for line in subprocess.check_output(["ldd", tlib], text=True).splitlines():
    if "libcudnn.so.8" in line and "=>" in line:
        print(os.path.dirname(os.path.realpath(line.split("=>")[1].split("(")[0].strip())))
        break
PY
)
if [[ -n "$CUDNN_DIR" && -e "$CUDNN_DIR/libcudnn_cnn_train.so.8" ]]; then
    export LD_LIBRARY_PATH="$CUDNN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[env] cuDNN libs on LD_LIBRARY_PATH: $CUDNN_DIR"
fi

# Only pass the SB35 root/file flags through if the user actually set them --
# an explicit --flag bypasses this script's candidate-probing entirely, which
# is correct for a deliberate override but would be wrong as an always-on
# default (it would silently disable the fallback the verification step
# relies on). Leaving them unset lets examples/enhancement_closure.py fall
# back to its own recorded-default + candidate-probing logic.
EXTRA_FLAGS=()
[[ -n "${SIMS:-}" ]]            && EXTRA_FLAGS+=(--sims "$SIMS")
[[ -n "${SB35_PARAM_FILE:-}" ]] && EXTRA_FLAGS+=(--sb35_param_file "$SB35_PARAM_FILE")
[[ -n "${SB35_DM_ROOT:-}" ]]    && EXTRA_FLAGS+=(--sb35_dm_root "$SB35_DM_ROOT")
[[ -n "${SB35_HYDRO_ROOT:-}" ]] && EXTRA_FLAGS+=(--sb35_hydro_root "$SB35_HYDRO_ROOT")
[[ -n "${SB35_GROUP_ROOT:-}" ]] && EXTRA_FLAGS+=(--sb35_group_root "$SB35_GROUP_ROOT")
[[ -n "${CHECKPOINT_PATH:-}" ]] && EXTRA_FLAGS+=(--checkpoint_path "$CHECKPOINT_PATH")

echo "=== enhancement_closure: run_dir=$RUN_DIR model=$MODEL_NAME output_root=$OUTPUT_ROOT ==="

# The script's own verification step runs first (inside main(), before any
# model/GPU work) and exits 2 with a loud message if the SB35 raw data isn't
# reachable and no reusable prior eval exists -- nothing further to gate here.
python examples/enhancement_closure.py \
    --snapshot "$SNAPSHOT" \
    --box_size "$BOX_SIZE" \
    --npix "$NPIX" \
    --patch_pix "$PATCH_PIX" \
    --halo_mass_min "$HALO_MASS_MIN" \
    --existing_eval_root "$EXISTING_EVAL_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --run_dir "$RUN_DIR" \
    --model_name "$MODEL_NAME" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    "${EXTRA_FLAGS[@]}"

echo "=== enhancement_closure: done ==="
