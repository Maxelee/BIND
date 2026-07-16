#!/bin/bash
#SBATCH --job-name=wp2_multisample
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp2_multisample_%A.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp2_multisample_%A.err
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

# ── WP-A2 multi-sample paint (task 6) — OPTIONAL, snap 063 only ───────────────
# The single-draw fiducial paint ALREADY EXISTS (bind_lightcone_tng/snap_NNN),
# and the core truth validation reuses it — NO paint is needed for Sigma_model /
# CylToSph / x_e / Y-M. This script paints ONLY the extra generative draws at
# snap 063 that the multi-sample-convergence check (SHARED_CONTEXT caveat 4)
# needs, using the SAME conditions + params (cosmology 0.3/0.8) as the existing
# fiducial composite so the draws are directly comparable.
#
# ⛔ HUMAN CHECKPOINT — Max submits (only if you want the task-6 number):
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch /mnt/home/mlee1/BIND-paper3a/analysis/paper3a/scripts/run_wp2_fiducial_paint.sh
# then re-run the snap-063 validation with
#   MULTISAMPLE_ROOT=/mnt/home/mlee1/ceph/paper3/A/wp2_multisample \
#     sbatch --array=3-3 analysis/paper3a/scripts/run_wp2_truth_validation.sh
#
# Environment: BIND_env (torch 2.6.0 / CUDA 12.5, matches Popeye's ~12.6 driver
# and is what painted the twobound/fiducial runs) + the lightcone bind source via
# PYTHONPATH. ⚠ do NOT use paper3b_popeye (torch cu130 -> "driver too old", the
# cause of the failed job 2451133). Restart-safe: existing draws are skipped.
#
# ⚠ generate_halos has no --seed, so the N draws come from separate processes; if
# they come out identical the sampler is deterministically re-seeded (the
# validation's multi_sample_convergence will show frac_dev ~ 0 across k) — then
# add a --seed to generate_halos or draw in one process.

set -euo pipefail

VENV=${VENV:-/mnt/home/mlee1/venvs/BIND_env}
LIGHTCONE_SRC=${LIGHTCONE_SRC:-/mnt/home/mlee1/BIND-lightcone/src}
source "$VENV/bin/activate"
export PYTHONPATH="$LIGHTCONE_SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /mnt/home/mlee1/ceph/logs

FID=${FID:-/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_063}   # existing fiducial paint (conditions + params)
STAGE1="$FID/stage1"
PARAMS="$STAGE1/params.npy"
WEIGHTS=${WEIGHTS:-/mnt/home/mlee1/ceph/bind_portable_twobound/weights/fm_redshift_thermo}
OUT_ROOT=${OUT_ROOT:-/mnt/home/mlee1/ceph/paper3/A/wp2_multisample}
N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}
N_MULTISAMPLE=${N_MULTISAMPLE:-8}

[[ -f "$PARAMS" ]] || { echo "missing $PARAMS" >&2; exit 1; }
python -c "import bind.cli.generate_halos" 2>/dev/null || {
  echo "bind.cli.generate_halos not importable — check LIGHTCONE_SRC=$LIGHTCONE_SRC and VENV=$VENV" >&2; exit 1; }

echo "=== WP-A2 multi-sample paint: snap 063, ${N_MULTISAMPLE} draws ==="
for K in $(seq 0 $((N_MULTISAMPLE - 1))); do
  OUT="$OUT_ROOT/snap_063_s${K}"
  if [[ -f "$OUT/composite_slab00.npz" ]]; then echo "  draw ${K}: done, skip"; continue; fi
  python -u -m bind.cli.generate_halos \
      --stage1_dir "$STAGE1" \
      --params "$PARAMS" \
      --run_dir "$WEIGHTS" \
      --output_dir "$OUT" \
      --n_steps "$N_STEPS" --batch_size "$BATCH_SIZE" --device auto
done
echo "=== multi-sample paint done: $OUT_ROOT ==="
