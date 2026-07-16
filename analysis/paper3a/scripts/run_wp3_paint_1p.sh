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
#SBATCH --array=0-30
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A3 1P trajectory paint (rusty GPU) ─────────────────────────────────────
# Paints the gate's 1P runs (fiducial + 6 feedback params x 5 levels = 31 runs,
# see manifest.json) over the same 20 TNG300 lightcone snapshots as the SB35
# Sobol bundle, with identical conditions, weights, and sampler settings, so
# Sobol envelope and 1P trajectories are directly comparable.
#
# Prepare the run tree first (CPU, seconds; prints the manifest + array range):
#   python analysis/paper3a/scripts/make_wp3_1p_bundle.py
# then review the printed parameter list before submitting (the "6 key
# feedback params" default is WP-A3's proposal — edit with --params ... if
# you want a different set).
#
# ⛔ HUMAN CHECKPOINT — Max submits:
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_wp3_paint_1p.sh
#
# Environment/repo note: this intentionally mirrors run_sb35_generate.sh
# (BIND_env venv + /mnt/home/mlee1/vdm_bind2 checkout) — the exact
# combination that painted the 256-run Sobol bundle on 2026-06-18/19.
# bind.cli.generate_halos does not exist on the analysis/paper3a-gas-
# calibration topic branch (built from main); do not "fix" this script to
# run from the worktree. Restart-safe: (run, snap) pairs with an existing
# composite_slab00.npz are skipped.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/vdm_bind2
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
