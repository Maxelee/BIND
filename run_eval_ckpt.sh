#!/bin/bash
#SBATCH --job-name=ckpt_highk
#SBATCH --output=/mnt/home/mlee1/ceph/logs/ckpt_highk_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/ckpt_highk_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Compare fm_two_head checkpoints on high-k transfer T(k). Read-only on the
# checkpoints; writes a fresh npz to ceph/fm_diag/ckpt_highk/. Does NOT touch
# any training run / released weights.
#
#   sbatch run_eval_ckpt.sh
#   RUN_DIR=/mnt/home/mlee1/ceph/fm_runs/fm_two_head N_SIMS=4 sbatch run_eval_ckpt.sh

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
# torch3 activate no longer carries the gcc-11 libstdc++ on LD_LIBRARY_PATH
# (the pinned hash 404'd); add it so `import pandas`/`import bind` works on the
# compute node. See project_torch3_libstdcxx_fix.
export LD_LIBRARY_PATH=/mnt/sw/nix/store/1fycpximqhwabw5a9z5myspx61lpx8gi-gcc-11.5.0/lib64:${LD_LIBRARY_PATH:-}
cd /mnt/home/mlee1/vdm_bind2

RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_two_head}
N_SIMS=${N_SIMS:-4}
N_STEPS=${N_STEPS:-50}
OUT=${OUT:-ceph/fm_diag/ckpt_highk/$(basename "$RUN_DIR").npz}

python eval_ckpt_highk.py \
    --run_dir "$RUN_DIR" \
    --n_sims "$N_SIMS" \
    --n_steps "$N_STEPS" \
    --out "$OUT"
