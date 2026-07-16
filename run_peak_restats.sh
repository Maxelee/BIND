#!/bin/bash
#SBATCH --job-name=bind_peak_restats
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_peak_restats_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_peak_restats_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-8
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Peak-stats regeneration (post stats.py ν/noise/multi-scale upgrade) ───────
# Array index = run_dir × 3 + variant.  Variants per run dir:
#   0: --smoothing_arcmin 2                       → peak_{counts,cross}.npz
#      (extended ν to 12, legacy shapes — notebook-compatible)
#   1: --smoothing_arcmin 2 1 5 8 --out_suffix _multi → *_multi.npz (scale axis;
#      peak_cross uses the FIRST scale, 2')
#   2: --smoothing_arcmin 2 --shape_noise_ngal $NGAL → *_ngal<N>.npz
#      (ν vs smoothed-noise rms — survey S/N convention)
# Default run dirs: the fiducial trio (bind, truth, dmo run_0000) → --array=0-8.
# Other designs:  DESIGN=twobound sbatch --array=0-179 run_peak_restats.sh
# Env overrides: OUTPUT_ROOT, DESIGN, NGAL (default 10; e.g. 27 for LSST-Y10).

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
NGAL=${NGAL:-10}

if [[ -n "${DESIGN:-}" ]]; then
    mapfile -t RUN_DIRS < <(ls -d "$OUTPUT_ROOT/runs/$DESIGN"/run_* | sort)
else
    RUN_DIRS=("$OUTPUT_ROOT/runs/bind/run_0000"
              "$OUTPUT_ROOT/runs/truth/run_0000"
              "$OUTPUT_ROOT/runs/dmo/run_0000")
fi

RD="${RUN_DIRS[$((SLURM_ARRAY_TASK_ID / 3))]}"
VAR=$((SLURM_ARRAY_TASK_ID % 3))

case $VAR in
  0) OPTS=(--smoothing_arcmin 2) ;;
  1) OPTS=(--smoothing_arcmin 2 1 5 8 --out_suffix _multi) ;;
  2) OPTS=(--smoothing_arcmin 2 --shape_noise_ngal "$NGAL") ;;
esac

# NUFID_FROM=<run dir>: fixed-nu convention (nu = kappa/sigma_fid; -> *_nufid.npz)
[[ -n "${NUFID_FROM:-}" ]] && OPTS+=(--nu_sigma0_from "$NUFID_FROM")

echo "=== peak restats: $RD (variant $VAR) ==="
python -u -m bind.cli.lightcone_stats --run_dir "$RD" --peaks_only "${OPTS[@]}"
echo "=== done: $RD variant $VAR ==="
