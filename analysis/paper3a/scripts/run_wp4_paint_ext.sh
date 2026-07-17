#!/bin/bash
#SBATCH --job-name=wp4_paint_ext
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp4_paint_ext_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp4_paint_ext_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=256-511%32
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A4 design densification: paint Sobol points 256-511 (rusty GPU) ──────
# Same machinery/env as run_wp3_paint_1p.sh (lightcone worktree @6a38e18 via
# PYTHONPATH; conditions + fm_redshift_thermo weights shared from bind_sb35).
# ONLY the six gate snapshots (not all 20): ~6 x 15-20 min per run on an
# a100 -> comfortably inside --time. %32 throttles GPU usage; raise/lower to
# taste. Restart-safe: painted (run, snap) pairs are skipped, so a resubmit
# after any failure only paints what is missing.
#
# Prepare run dirs first (CPU, seconds — already run in session 4c):
#   python analysis/paper3a/scripts/make_wp4_sobol_ext.py
#
# ⛔ HUMAN CHECKPOINT — Max submits (full unattended chain):
#   cd /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts
#   jid1=$(sbatch --parsable run_wp4_paint_ext.sh)
#   jid2=$(sbatch --parsable --dependency=afterok:$jid1 run_wp4_ext_ops_disbatch.sh)
#   sbatch --dependency=afterok:$jid2 run_wp4_fit_ext.sh
# (afterok: each stage starts only if the previous one fully succeeded; on a
# partial failure, resubmit the failed stage — everything is resumable —
# and the dependency releases on the resubmitted job's completion instead.)

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/vdm_bind2-lightcone
export PYTHONPATH=/mnt/home/mlee1/vdm_bind2-lightcone/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p /mnt/home/mlee1/ceph/logs

BUNDLE=/mnt/ceph/users/mlee1/bind_sb35
RUN_ROOT=/mnt/ceph/users/mlee1/bind_sb35_ext/runs
WEIGHTS=$BUNDLE/weights/fm_redshift_thermo
N_STEPS=50
BATCH_SIZE=16

GATE_SNAPS=(96 71 67 63 56 49)

RUN_DIR="$RUN_ROOT/$(printf 'run_%04d' "$SLURM_ARRAY_TASK_ID")"
[[ -f "$RUN_DIR/params.npy" ]] || { echo "missing $RUN_DIR/params.npy — run make_wp4_sobol_ext.py first" >&2; exit 1; }

echo "=== WP-A4 ext paint: run ${SLURM_ARRAY_TASK_ID} (gate snaps only) ==="

for SNAP in "${GATE_SNAPS[@]}"; do
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

echo "=== ext paint done: $RUN_DIR ==="
