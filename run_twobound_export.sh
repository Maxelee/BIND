#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=cascadelake
#SBATCH -J twobound_export
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -o /mnt/home/mlee1/ceph/logs/twobound_export_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/twobound_export_%j.err
#SBATCH -t 01:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Pool the per-run twobound κ suite into one export pair ──────────────────────
# Reads <RUNS_DIR>/run_NNNN/{kappa_maps.npz,params.npy} and streams them into a
# single twobound_kappa.npz (kappa + params + metadata) and twobound_params.npy.
# The κ array is ~58 GB (60 runs × 50 real × 5 z_s × 1024²) but is streamed
# run-by-run, so peak memory stays ~1 GB — 16 GB is plenty of headroom.
#
#   sbatch run_twobound_export.sh
#   RUNS_DIR=/path/to/runs sbatch run_twobound_export.sh
#
# Env overrides: RUNS_DIR, OUT_DIR.

set -euo pipefail

module load python
source ~/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

RUNS_DIR=${RUNS_DIR:-/mnt/home/mlee1/ceph/bind_science/runs/twobound}
OUT_DIR=${OUT_DIR:-$RUNS_DIR}

echo "=== twobound export | job ${SLURM_JOB_ID:-local} | runs ${RUNS_DIR} ==="

python -u combine_twobound_kappa.py --runs_dir "$RUNS_DIR" --out_dir "$OUT_DIR"

echo "=== done → $OUT_DIR/{twobound_kappa.npz,twobound_params.npy} ==="
