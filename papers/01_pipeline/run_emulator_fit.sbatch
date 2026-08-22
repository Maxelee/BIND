#!/bin/bash
#SBATCH --job-name=emu_fit
#SBATCH --output=/mnt/home/mlee1/ceph/logs/emu_fit_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/emu_fit_%j.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# ── Fit and PERSIST the Paper I Gaussian-process emulator ────────────────────
# The notebook refits the 13 GP heads on every run. Emulator.save()/load() already
# exist and round-trip bit-exactly; this just uses them. The saved bundle is also
# the paper's release artifact -- "theta -> any released statistic in milliseconds"
# is only a real claim if the trained object ships.
#
# NOT parallelised with disBatch, deliberately: the heads share one dataset load and
# the gpytorch backend already batches across PCA components on the GPU, so the whole
# fit is a couple of minutes. Splitting it would mean 13 separate bundles to merge,
# and Emulator.save() serialises the whole object at once.
#
#   sbatch papers/01_pipeline/run_emulator_fit.sbatch
#
# The fit is keyed to the exact train/test split (see emulator_cache.py). If fig 9's
# split, statistic list or backend settings change, the notebook detects the mismatch
# and refits rather than silently validating on training data.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND/papers/01_pipeline
mkdir -p /mnt/home/mlee1/ceph/logs

# cuDNN lazy-dlopen fix -- same story as run_figures.sbatch: the nix torch build
# dlopen()s libcudnn_cnn_train.so.8 by bare soname and DT_RUNPATH is not consulted.
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
fi

python build_emulator.py --fit
python build_emulator.py --verify
echo "[done] bundle at /mnt/home/mlee1/ceph/bind_sb35/emulator_fits/paper1_gp.pt"
