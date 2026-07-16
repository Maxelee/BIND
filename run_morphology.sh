#!/bin/bash
#SBATCH --job-name=bind_morphology
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_morphology_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_morphology_%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=0-61
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Rung 2: per-halo component morphology (shapes / offsets) ──────────────────
# Per run x snap: median per-M200c-bin ellipticity |e| of gas/DM/star, gas–DM
# centroid offset, and gas–DM alignment (LOS-bg-subtracted moments within 1.5 r500)
# -> bind_science/morphology/<run>_snap<NNN>.npz. Restart-safe (skips existing).
# Headline: gas is rounder than DM; feedback offsets/rounds it. (FRB-DM figure
# reuses the Rung-1 profiles cache — no reduction needed for it.)
#
# Array (one run per task, all 20 snaps, ~3 min/run):
#   tasks 0..59 = twobound run_0000..run_0059;  60 = fiducial;  61 = TNG300 truth
#
#   sbatch run_morphology.sh                      # full grid
#   SNAPS="96 90 85" sbatch run_morphology.sh     # z<0.2 only
#   FORCE=1 sbatch run_morphology.sh              # recompute
#
# Afterwards (workstation):  python examples/morphology.py --plot
# Env overrides: OUTPUT_ROOT, SNAPS (space-separated snaps), FORCE=1.

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
SNAPS=${SNAPS:-}
FORCE_FLAG=""; [[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

t=$SLURM_ARRAY_TASK_ID
if   [[ "$t" -lt 60 ]]; then RUN=$(printf 'run_%04d' "$t")
elif [[ "$t" -eq 60 ]]; then RUN=fid
elif [[ "$t" -eq 61 ]]; then RUN=truth
else echo "task $t out of range (0-61)"; exit 1
fi

echo "=== morphology: $RUN  (snaps='${SNAPS:-all}' force='${FORCE:-0}') ==="
# shellcheck disable=SC2086  # SNAPS / FORCE_FLAG are intentional word-split args
python -u examples/morphology.py --reduce --runs "$RUN" ${SNAPS:+--snaps $SNAPS} $FORCE_FLAG
echo "=== done: $RUN ==="
