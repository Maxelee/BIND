#!/bin/bash
#SBATCH --job-name=bind_emu_train
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_emu_train_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_emu_train_%j.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Emulator stage 3: train the lightcone-statistics emulator (GPU) ────────────
# Assembles the dataset from the run suite (if not already cached) and fits the
# emulator on the allocated GPU.  All three torch backends use the GPU here: the
# MLP ensemble, the conditional normalizing flow, and the GPU exact GP (gpgpu,
# the recommended backend, batched over PCA components via gpytorch).  The
# sklearn "gp" backend is CPU-only — use "gpgpu" on this GPU node instead.
#
# Env overrides: DATASET, RUNS_DIR, OUT_DIR, BACKENDS ("mlp gpgpu flow"),
# N_COMPONENTS, EPOCHS, EXTRA (raw flags for bind.cli.emulate train).

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

RUNS_DIR=${RUNS_DIR:-/mnt/home/mlee1/ceph/bind_sb35/runs}
OUT_DIR=${OUT_DIR:-/mnt/home/mlee1/ceph/bind_sb35/emulator}
DATASET=${DATASET:-$OUT_DIR/emulator_dataset.npz}
BACKENDS=${BACKENDS:-mlp gpgpu flow}
N_COMPONENTS=${N_COMPONENTS:-20}
mkdir -p "$OUT_DIR"

# (re)assemble the training table from whatever runs are currently complete
python -u -m bind.cli.emulator_assemble --runs_dir "$RUNS_DIR" --out "$DATASET"

EPOCH_OPT=(); [[ -n "${EPOCHS:-}" ]] && EPOCH_OPT+=(--epochs "$EPOCHS")
for B in $BACKENDS; do
    echo "=== training backend: $B ==="
    python -u -m bind.cli.emulate train --dataset "$DATASET" \
        --backend "$B" --n_components "$N_COMPONENTS" \
        --out "$OUT_DIR/lightcone_emulator_$B.pt" "${EPOCH_OPT[@]}" ${EXTRA:-}
done
echo "=== emulator bundles written to $OUT_DIR ==="
