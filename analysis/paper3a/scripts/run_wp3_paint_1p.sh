#!/bin/bash
#SBATCH --job-name=wp3_paint_1p
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp3_paint_1p_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp3_paint_1p_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A3 fiducial paint (rusty GPU) ──────────────────────────────────────────
# 2026-07-16: the painted 1P extremes already exist (bind_portable_twobound,
# 30 params x 2 bounds x 20 snaps — Max), so the only gap for the A3 gate is
# the TNG-fiducial run: one array task over the same 20 TNG300 lightcone
# snapshots with identical conditions/weights/sampler settings as the Sobol
# and twobound bundles.
#
# Prepare the run tree first (CPU, seconds; prints the manifest + array range):
#   python analysis/paper3a/scripts/make_wp3_1p_bundle.py --fiducial-only
# (Without --fiducial-only it builds 5-level trajectories for 6 feedback
# params = 31 runs, kept in reserve if the gate needs finer 1P sampling —
# then set --array=0-30.)
#
# ⛔ HUMAN CHECKPOINT — Max submits:
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_wp3_paint_1p.sh
#
# Environment/repo note (updated 2026-07-16 after job 6624841 failed):
# bind.cli.generate_halos exists ONLY on the `lightcone` branch. The June
# Sobol run worked because /mnt/home/mlee1/vdm_bind2 happened to be on
# `lightcone` then; it is now on `feature/wl-emu`, so BIND_env's editable
# install no longer sees the module. Fix: a dedicated read-only worktree
# `/mnt/home/mlee1/vdm_bind2-lightcone` (branch `lightcone`, 6a38e18 — the
# June generation vintage) put ahead of the editable install via PYTHONPATH.
# Max's active checkout is never touched, and this script no longer depends
# on whatever branch it happens to be on. Restart-safe: (run, snap) pairs
# with an existing composite_slab00.npz are skipped.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/vdm_bind2-lightcone
export PYTHONPATH=/mnt/home/mlee1/vdm_bind2-lightcone/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p /mnt/home/mlee1/ceph/logs

BUNDLE=${BUNDLE:-/mnt/home/mlee1/ceph/bind_sb35}
RUN_ROOT=${RUN_ROOT:-/mnt/home/mlee1/ceph/paper3/A/wp3_gate/1p_runs}
WEIGHTS=${WEIGHTS:-$BUNDLE/weights/fm_redshift_thermo}
N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}

SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

RUN_DIR="$RUN_ROOT/$(printf 'run_%04d' "$SLURM_ARRAY_TASK_ID")"
[[ -f "$RUN_DIR/params.npy" ]] || { echo "missing $RUN_DIR/params.npy — run make_wp3_1p_bundle.py first" >&2; exit 1; }

echo "=== WP-A3 1P paint: run ${SLURM_ARRAY_TASK_ID} ==="
echo "    run_dir=$RUN_DIR  weights=$WEIGHTS  n_steps=$N_STEPS  batch_size=$BATCH_SIZE"

for SNAP in "${SNAPSHOTS[@]}"; do
  S3=$(printf '%03d' "$SNAP")
  OUT="$RUN_DIR/snap_${S3}"
  if [[ -f "$OUT/composite_slab00.npz" ]]; then
    echo "  snap ${S3}: already done, skipping"
    continue
  fi
  python -u -m bind.cli.generate_halos \
      --stage1_dir "$BUNDLE/conditions/snap_${S3}" \
      --params "$RUN_DIR/params.npy" \
      --run_dir "$WEIGHTS" \
      --output_dir "$OUT" \
      --n_steps "$N_STEPS" \
      --batch_size "$BATCH_SIZE" \
      --device auto
done

echo "=== 1P paint done: $RUN_DIR ==="
