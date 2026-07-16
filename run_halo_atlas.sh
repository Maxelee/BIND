#!/bin/bash
#SBATCH --job-name=bind_halo_atlas
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_halo_atlas_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_halo_atlas_%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-61
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Per-halo baryon-feedback atlas (Rung 0) ───────────────────────────────────
# Reduces every composite_slab*.npz (per-halo DM/gas/star mass + y/T/S/Pe patches)
# into integrated aperture quantities -> f_gas(M|z,theta), Y-M + scatter, baryon
# closure. One small npz per run x snap in <OUTPUT_ROOT>/halo_atlas/ (restart-safe:
# existing caches are skipped, so this co-exists with the workstation snap_096 run).
#
# Array (one run per task, all 20 snaps each, ~7 min/run):
#   tasks  0..59 = twobound run_0000..run_0059
#   task     60  = fiducial lightcone (bind_lightcone_tng)
#   task     61  = TNG300 hydro truth (runs/truth/run_0000)
#
#   sbatch run_halo_atlas.sh                        # full grid (all runs, all snaps)
#   sbatch --array=0-59 run_halo_atlas.sh           # twobound runs only
#   SNAPS="96 90 85" FORCE=1 sbatch run_halo_atlas.sh  # z<0.2, recompute (eROSITA fig)
#   FORCE=1 sbatch run_halo_atlas.sh                # recompute whole grid (e.g. new fields)
#
# Afterwards (cheap, on a workstation):
#   python examples/halo_atlas.py --plot --plot_z --plot_erosita
# Env overrides: OUTPUT_ROOT, SNAPS (space-separated snaps), FORCE=1 (pass --force).

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
SNAPS=${SNAPS:-}                              # empty -> all 20 (script default)

t=$SLURM_ARRAY_TASK_ID
if   [[ "$t" -lt 60 ]]; then RUN=$(printf 'run_%04d' "$t")
elif [[ "$t" -eq 60 ]]; then RUN=fid
elif [[ "$t" -eq 61 ]]; then RUN=truth
else echo "task $t out of range (0-61)"; exit 1
fi

FORCE_FLAG=""; [[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"
echo "=== halo atlas: $RUN  (snaps='${SNAPS:-all}' force='${FORCE:-0}') ==="
# shellcheck disable=SC2086  # SNAPS / FORCE_FLAG are intentional word-split args
python -u examples/halo_atlas.py --reduce --runs "$RUN" ${SNAPS:+--snaps $SNAPS} $FORCE_FLAG
echo "=== done: $RUN ==="
