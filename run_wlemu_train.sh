#!/bin/bash
#SBATCH --job-name=wlemu_train
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wlemu_train_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wlemu_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WL field-level emulator: STEP 2 (GPU) — train θ→κ conditional flow matching ─
# Same launch shape as run_train.sh: single node, one srun task per GPU, DDP via
# torch.distributed.  bind.wlemu.train reads its rank from SLURM's env
# (SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS), so no torchrun is needed.
# Reads the pooled cache from run_wlemu_cache.sh; model resolution = the cache's.
#
#   sbatch run_wlemu_train.sh                  # native 1024² (matches default cache)
#   RESOLUTION=256 sbatch run_wlemu_train.sh   # fast-iteration res
#
# To use A100s instead, change --constraint=a100 and the --gpus-per-node /
# --ntasks-per-node count (A100 nodes have 4 GPUs) and lower --mem.
#
# Env overrides: RESOLUTION, CACHE, RUN_NAME, OUTPUT_DIR, BASE_CH, BATCH_SIZE,
#                GRAD_ACCUM, EPOCHS, LR.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# cuDNN 8.9.x in the Flatiron nix torch build keeps its split sub-libraries
# (libcudnn_cnn_train.so.8, ...) in the nix store next to libcudnn.so.8, but the
# main shim dlopen()s them by bare soname at *backward* time and nix's DT_RUNPATH
# is not consulted for that lazy load.  Result: forward works, the first
# loss.backward() dies with "libcudnn_cnn_train.so.8: cannot open shared object
# file" -> "FIND/GET was unable to find an engine to execute this computation".
# Put cuDNN's lib dir (the one torch was built against) on LD_LIBRARY_PATH so the
# lazy dlopen resolves.  Derived from the live torch so it survives store-hash bumps.
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
else
    echo "WARNING: could not locate libcudnn_cnn_train.so.8 via torch; backward may fail" >&2
fi

RESOLUTION=${RESOLUTION:-1024}
CACHE=${CACHE:-/mnt/home/mlee1/ceph/bind_sb35/wlemu_cache_${RESOLUTION}}
RUN_NAME=${RUN_NAME:-fm_kappa_${RESOLUTION}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/bind_sb35/wlemu_runs}
EPOCHS=${EPOCHS:-400}
LR=${LR:-2e-4}
VAL_EVERY=${VAL_EVERY:-500}   # steps between val + best.pt/last.pt checkpoints

# Per-resolution defaults (per-GPU micro-batch / accumulation / width).
# 1024² needs activation checkpointing + a small micro-batch; 256² fits big batches.
if [[ "$RESOLUTION" -ge 1024 ]]; then
    BASE_CH=${BASE_CH:-96};  BATCH_SIZE=${BATCH_SIZE:-2};  GRAD_ACCUM=${GRAD_ACCUM:-4};  CKPT="--grad_checkpoint"
elif [[ "$RESOLUTION" -ge 512 ]]; then
    BASE_CH=${BASE_CH:-128}; BATCH_SIZE=${BATCH_SIZE:-6};  GRAD_ACCUM=${GRAD_ACCUM:-2};  CKPT="--grad_checkpoint"
else
    BASE_CH=${BASE_CH:-128}; BATCH_SIZE=${BATCH_SIZE:-32}; GRAD_ACCUM=${GRAD_ACCUM:-1};  CKPT=""
fi

if [[ ! -f "$CACHE/kappa.npy" || ! -f "$CACHE/norm.npz" ]]; then
    echo "ERROR: finished cache not found at $CACHE — run run_wlemu_cache.sh first." >&2
    exit 1
fi

echo "=== wlemu train | job ${SLURM_JOB_ID} | res ${RESOLUTION}² | ranks=${SLURM_NTASKS} base_ch=${BASE_CH} ==="
echo "    cache=$CACHE  run=$RUN_NAME"
echo "    per-GPU batch=$BATCH_SIZE  grad_accum=$GRAD_ACCUM  -> global=$((BATCH_SIZE*SLURM_NTASKS*GRAD_ACCUM))"

srun python -u -m bind.wlemu.train \
    --cache "$CACHE" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --base_ch "$BASE_CH" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    --val_every "$VAL_EVERY" \
    $CKPT

echo "=== Training done → $OUTPUT_DIR/$RUN_NAME (best.pt / last.pt) ==="
