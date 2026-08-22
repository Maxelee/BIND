#!/bin/bash
#SBATCH --job-name=paper1_figs
#SBATCH --output=/mnt/home/mlee1/ceph/logs/paper1_figs_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/paper1_figs_%j.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Paper I figure package: rebuild + execute paper1_figures.ipynb ─────────────
# Regenerates the notebook from _build_figures_nb.py, then executes it with the
# BIND venv kernel (figs land in figs_v2/ + figs_preview/). The GPU accelerates
# the gpytorch emulator fits (bind.emulator auto-selects cuda); the CPUs serve
# the 50-node coverage test and bootstrap loops (cells read SLURM_CPUS_PER_TASK).
# Writes RUN_DONE / RUN_FAILED sentinels in papers/01_pipeline/ on exit.
#
#   sbatch papers/01_pipeline/run_figures.sbatch

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND/papers/01_pipeline
mkdir -p /mnt/home/mlee1/ceph/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NB_CELL_TIMEOUT=${NB_CELL_TIMEOUT:-10800}

# cuDNN lazy-dlopen fix — the nix torch build dlopen()s libcudnn_cnn_train.so.8
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

rm -f RUN_DONE RUN_FAILED

# Gate: the figure-builder edit campaign creates EDITS_READY when the builder is
# final and syntax-checked. Wait up to 4 h (1-min checks) so a job that starts
# before the edits finish idles briefly instead of executing a half-edited
# builder; bail out cleanly if the gate never opens.
for _ in $(seq 1 240); do
    [[ -f EDITS_READY ]] && break
    sleep 60
done
if [[ ! -f EDITS_READY ]]; then
    echo "EDITS_READY never appeared after 4 h — not executing a half-edited builder" >&2
    echo "gate-timeout ${SLURM_JOB_ID:-local}" > RUN_FAILED
    exit 1
fi

python _build_figures_nb.py
if python _run_figures_nb.py; then
    echo "${SLURM_JOB_ID:-local}" > RUN_DONE
else
    echo "${SLURM_JOB_ID:-local}" > RUN_FAILED
    exit 1
fi
