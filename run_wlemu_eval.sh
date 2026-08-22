#!/bin/bash
#SBATCH --job-name=wlemu_eval
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wlemu_eval_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wlemu_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WL field-level emulator: evaluate a trained κ checkpoint on a single GPU ────
# Loads a flow-matching κ checkpoint, generates convergence maps for the held-out
# (validation) SB35 parameter points, and compares to the cached truth maps:
# angular power spectrum C_ell, 1-point PDF, map moments + example panels.
# Inference is forward-only (no backward), so it fits one A100/V100; bf16 autocast
# is used automatically on Ampere+, fp32 elsewhere.
#
#   sbatch run_wlemu_eval.sh                              # best.pt of the default run
#   CHECKPOINT=.../last.pt sbatch run_wlemu_eval.sh       # evaluate the latest instead
#   N_RUNS=8 N_MAPS=32 sbatch run_wlemu_eval.sh           # heavier, more statistics
#
# Env overrides: RESOLUTION, CACHE, RUN_NAME, RUNS_DIR, CHECKPOINT, OUTPUT_DIR,
#                WEIGHTS, N_RUNS, N_MAPS, Z_IDX, N_STEPS, CFG_SCALE, GEN_BATCH.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Same cuDNN sub-library path fix as run_wlemu_train.sh (harmless for inference,
# but keeps the env identical and future-proofs any backward use).
CUDNN_DIR=$(python - <<'PY'
import os, subprocess, torch
tlib = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so")
for line in subprocess.check_output(["ldd", tlib], text=True).splitlines():
    if "libcudnn.so.8" in line and "=>" in line:
        print(os.path.dirname(os.path.realpath(line.split("=>")[1].split("(")[0].strip())))
        break
PY
)
[[ -n "$CUDNN_DIR" ]] && export LD_LIBRARY_PATH="$CUDNN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RESOLUTION=${RESOLUTION:-1024}
CACHE=${CACHE:-/mnt/home/mlee1/ceph/bind_sb35/wlemu_cache_${RESOLUTION}}
RUN_NAME=${RUN_NAME:-fm_kappa_${RESOLUTION}}
RUNS_DIR=${RUNS_DIR:-/mnt/home/mlee1/ceph/bind_sb35/wlemu_runs}
CHECKPOINT=${CHECKPOINT:-$RUNS_DIR/$RUN_NAME/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-$RUNS_DIR/$RUN_NAME/eval}
WEIGHTS=${WEIGHTS:-ema}

# Quick-look defaults: 3 held-out runs × 3 source planes × 8 maps × 40 ODE steps
# ≈ 72 map-generations (~a few minutes on one A100).  Scale up for fuller stats,
# e.g. N_RUNS=8 N_MAPS=32 Z_IDX='' N_STEPS=50 sbatch run_wlemu_eval.sh
N_RUNS=${N_RUNS:-3}
N_MAPS=${N_MAPS:-8}
Z_IDX=${Z_IDX:-0,2,4}     # source planes; '' = all 5; default spans z=0.5,1.5,2.44
N_STEPS=${N_STEPS:-40}
CFG_SCALE=${CFG_SCALE:-1.0}
GEN_BATCH=${GEN_BATCH:-8}

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    echo "       (training writes best.pt/last.pt every VAL_EVERY steps — wait for the first val,"  >&2
    echo "        or set CHECKPOINT=/path/to/last.pt)" >&2
    exit 1
fi

echo "=== wlemu eval | job ${SLURM_JOB_ID} | res ${RESOLUTION}² ==="
echo "    checkpoint=$CHECKPOINT  (weights=$WEIGHTS)"
echo "    cache=$CACHE  -> output=$OUTPUT_DIR"
echo "    n_runs=$N_RUNS  n_maps=$N_MAPS  z_idx='${Z_IDX:-all}'  n_steps=$N_STEPS  cfg=$CFG_SCALE"

srun python -u examples/wlemu_eval.py \
    --checkpoint "$CHECKPOINT" \
    --cache "$CACHE" \
    --output_dir "$OUTPUT_DIR" \
    --weights "$WEIGHTS" \
    --n_runs "$N_RUNS" \
    --n_maps "$N_MAPS" \
    --z_idx "$Z_IDX" \
    --n_steps "$N_STEPS" \
    --cfg_scale "$CFG_SCALE" \
    --gen_batch "$GEN_BATCH"

echo "=== eval done → $OUTPUT_DIR (cl_compare.png / pdf_compare.png / maps_example.png / summary.json) ==="
