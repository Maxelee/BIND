#!/bin/bash
#SBATCH --job-name=bind_sobol_paste
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_sobol_paste_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_sobol_paste_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND science STAGE 3 (CPU): paste halos -> lensplanes + y-planes ──────────
# Array index = run index.  For each of the 20 snapshots: re-composite the
# B-generated halos with the (A-side) DMO background, convert to lux lensplanes +
# y-planes, then delete the transient composite.  Keeps run_<i>/lensplanes/;
# the per-halo patches (run_<i>/snap_*/composite_slab*.npz) are left in place.
#
# Env overrides: DESIGN, OUTPUT_ROOT, STAGE1_ROOT, R200_FACTOR, LP_GRID.

set -euo pipefail

# serial CPU (no MPI / no h5py here) — just the venv; do NOT load python-mpi.
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

DESIGN=${DESIGN:-sobol}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
STAGE1_ROOT=${STAGE1_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
TRANSFORMS=${TRANSFORMS:-$STAGE1_ROOT/lightcone_transforms.json}
R200_FACTOR=${R200_FACTOR:-4.0}
LP_GRID=${LP_GRID:-4096}
N_SNAPS=20
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

RUN_DIR="$OUTPUT_ROOT/runs/$DESIGN/$(printf 'run_%04d' "$SLURM_ARRAY_TASK_ID")"
LP_DIR="$RUN_DIR/lensplanes"
TMP="$RUN_DIR/_composite_tmp"
mkdir -p "$LP_DIR" "$TMP"

# all_snap_dirs (stage-1 dirs, for config.dat at lc_snap_idx 0)
ALL_SNAP=(); for s in "${SNAPSHOTS[@]}"; do ALL_SNAP+=("$STAGE1_ROOT/snap_$(printf '%03d' "$s")"); done

for IDX in $(seq 0 $((N_SNAPS-1))); do
  SNAP3=$(printf '%03d' "${SNAPSHOTS[$IDX]}")
  S1="$STAGE1_ROOT/snap_${SNAP3}/stage1"
  # 1. recomposite (DMO background + B-generated halos) -> transient composite maps
  python -u -m bind.cli.paint_recomposite \
      --stage1_dir "$S1" --generated_dir "$RUN_DIR/snap_${SNAP3}" \
      --output_dir "$TMP/snap_${SNAP3}" --r200_factor "$R200_FACTOR" --no_save_patches
  # 2. lensplanes (+ config.dat at IDX 0) and y-planes
  EXTRA=(); [[ "$IDX" -eq 0 ]] && EXTRA=(--all_snap_dirs "${ALL_SNAP[@]}")
  python -u -m bind.cli.paint_lensplane \
      --generate_dir "$TMP/snap_${SNAP3}" --stage1_dir "$S1" --output_dir "$LP_DIR" \
      --transforms "$TRANSFORMS" --lc_snap_idx "$IDX" --lc_n_snaps "$N_SNAPS" \
      --lp_grid "$LP_GRID" "${EXTRA[@]}"
  python -u -m bind.cli.paint_yplane \
      --generate_dir "$TMP/snap_${SNAP3}" --stage1_dir "$S1" --output_dir "$LP_DIR" \
      --lc_snap_idx "$IDX" --lc_n_snaps "$N_SNAPS" --lp_grid "$LP_GRID" --r200_factor "$R200_FACTOR"
  # 3. tau-planes (kSZ/FRB electron column from the gas channel) for lux compute_tau
  python -u -m bind.cli.paint_tauplane \
      --generate_dir "$TMP/snap_${SNAP3}" --stage1_dir "$S1" --output_dir "$LP_DIR" \
      --lc_snap_idx "$IDX" --lc_n_snaps "$N_SNAPS" --lp_grid "$LP_GRID"
  rm -rf "$TMP/snap_${SNAP3}"
done
rmdir "$TMP" 2>/dev/null || true

echo "=== lensplanes + y-planes + tau-planes ready for $RUN_DIR ==="
