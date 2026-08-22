#!/bin/bash
#SBATCH --job-name=bind_sobol_gen
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_sobol_gen_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_sobol_gen_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-99
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND science STAGE 2 (GPU): generate halos for a Sobol design ─────────────
# Array index = run index.  Each task runs the flow-matching sampler over the 20
# lightcone snapshots for one parameter point (~4 GPU hr/run) and saves ONLY the
# per-halo patches (composite_slab*.npz, no maps).  Portable: needs only the
# stage-1 cutouts + per-run params + weights — point STAGE1_ROOT/WEIGHTS at the
# shipped copies to run this on a GPU-rich machine, then ship the halos back.
#
# Env overrides: DESIGN, OUTPUT_ROOT, STAGE1_ROOT, WEIGHTS.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

DESIGN=${DESIGN:-sobol}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
STAGE1_ROOT=${STAGE1_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}   # cutouts (shared/shipped)
WEIGHTS=${WEIGHTS:-weights/fm_redshift_thermo}
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

RUN_DIR="$OUTPUT_ROOT/runs/$DESIGN/$(printf 'run_%04d' "$SLURM_ARRAY_TASK_ID")"
[[ -f "$RUN_DIR/params.npy" ]] || { echo "missing $RUN_DIR/params.npy (run bind-science-plan)"; exit 1; }

for SNAP in "${SNAPSHOTS[@]}"; do
  SNAP3=$(printf '%03d' "$SNAP")
  python -u -m bind.cli.generate_halos \
      --stage1_dir "$STAGE1_ROOT/snap_${SNAP3}/stage1" \
      --params "$RUN_DIR/params.npy" \
      --run_dir "$WEIGHTS" \
      --output_dir "$RUN_DIR/snap_${SNAP3}" \
      --device auto
done

echo "=== generated halos for $RUN_DIR ==="
