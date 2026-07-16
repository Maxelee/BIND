#!/bin/bash
#SBATCH --job-name=wp2_fid_paint
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp2_fid_paint_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp2_fid_paint_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-0
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A2 Popeye half: TNG-fiducial paint for the truth validation ────────────
# Paints BIND at the TNG-fiducial parameters (where TNG300-hydro is the ground
# truth) on the 4 validation snapshots, reusing the bind_portable_twobound
# conditions + weights already on Popeye (shared DMO across theta). Also paints
# 8 independent generative draws at snap 063 for the multi-sample check (task 6).
#
# Prep first (CPU, seconds):
#   python analysis/paper3a/scripts/make_wp2_fiducial_bundle.py
#
# ⛔ HUMAN CHECKPOINT — Max submits:
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch /mnt/home/mlee1/BIND-paper3a/analysis/paper3a/scripts/run_wp2_fiducial_paint.sh
#
# Environment: bind.cli.generate_halos lives on the `lightcone` branch (the
# redshift-conditioned model). This uses the paper3b_popeye venv (torch/h5py/
# Pylians) with the lightcone SOURCE shadowed in via PYTHONPATH from a dedicated
# detached worktree — no dependency on whatever branch /mnt/home/mlee1/BIND is
# currently on (Max's live checkout is left untouched). Restart-safe: (draw,snap)
# with an existing composite_slab00.npz are skipped.
#
# GPU note: Popeye a100 node is pcn-16-06 (6× a100-pcie-40gb, 1024 GB RAM). If
# the a100 constraint queues too long, v100s-pcie-32gb also fits (drop
# --constraint, add --constraint=v100 & reduce --batch_size if OOM).

set -euo pipefail

VENV=${VENV:-/mnt/home/mlee1/venvs/paper3b_popeye}
LIGHTCONE_SRC=${LIGHTCONE_SRC:-/mnt/home/mlee1/BIND-lightcone/src}
source "$VENV/bin/activate"
export PYTHONPATH="$LIGHTCONE_SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /mnt/home/mlee1/ceph/logs

BUNDLE=${BUNDLE:-/mnt/home/mlee1/ceph/bind_portable_twobound}
RUN_ROOT=${RUN_ROOT:-/mnt/home/mlee1/ceph/paper3/A/wp2_fiducial}
WEIGHTS=${WEIGHTS:-$BUNDLE/weights/fm_redshift_thermo}
N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}
N_MULTISAMPLE=${N_MULTISAMPLE:-8}

RUN_DIR="$RUN_ROOT/run_0000"
[[ -f "$RUN_DIR/params.npy" ]] || { echo "missing $RUN_DIR/params.npy — run make_wp2_fiducial_bundle.py first" >&2; exit 1; }

python -c "import bind.cli.generate_halos" 2>/dev/null || {
  echo "bind.cli.generate_halos not importable — check LIGHTCONE_SRC=$LIGHTCONE_SRC" >&2; exit 1; }

paint_one () {  # paint_one <snap3> <output_dir>
  local s3="$1" out="$2"
  if [[ -f "$out/composite_slab00.npz" ]]; then echo "  $out: done, skip"; return; fi
  python -u -m bind.cli.generate_halos \
      --stage1_dir "$BUNDLE/conditions/snap_${s3}" \
      --params "$RUN_DIR/params.npy" \
      --run_dir "$WEIGHTS" \
      --output_dir "$out" \
      --n_steps "$N_STEPS" --batch_size "$BATCH_SIZE" --device auto
}

echo "=== WP-A2 fiducial paint (single draw) — snaps 096 071 067 063 ==="
for S3 in 096 071 067 063; do
  paint_one "$S3" "$RUN_DIR/snap_${S3}"
done

echo "=== WP-A2 multi-sample paint — snap 063, ${N_MULTISAMPLE} draws (task 6) ==="
# generate_halos has no --seed; independent draws come from separate processes.
# If two draws come out identical, the sampler is deterministically re-seeded —
# escalate (add a --seed to generate_halos, or draw in one process). The truth
# validation's multi_sample_convergence will surface this (frac_dev ~ 0 across k).
for K in $(seq 0 $((N_MULTISAMPLE - 1))); do
  paint_one 063 "$RUN_DIR/snap_063_s${K}"
done

echo "=== fiducial paint done: $RUN_DIR ==="
