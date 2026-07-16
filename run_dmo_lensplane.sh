#!/bin/bash
#SBATCH --job-name=bind_dmo_lp
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_dmo_lp_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_dmo_lp_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── DMO baseline: ray-traced convergence with NO baryons ──────────────────────
# Array index = snapshot.  Builds lensplanes directly from the stage-1 DMO maps
# (same slabs/transforms/geometry as BIND & truth) into runs/dmo/run_0000.  Then:
#     COMPUTE_TSZ=False DESIGN=dmo sbatch --array=0-0 run_sobol_lux.sh
#     DESIGN=dmo        sbatch --array=0-0 run_sobol_stats.sh
# giving DMO kappa for the Cℓ-level baryonic suppression (P_BIND/P_DMO, P_hydro/P_DMO).
#
# Env overrides: OUTPUT_ROOT, STAGE1_ROOT, LP_GRID.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
STAGE1_ROOT=${STAGE1_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
TRANSFORMS=${TRANSFORMS:-$STAGE1_ROOT/lightcone_transforms.json}
LP_GRID=${LP_GRID:-4096}
N_SNAPS=20
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

IDX=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID 0..19}
SNAP3=$(printf '%03d' "${SNAPSHOTS[$IDX]}")
S1="$STAGE1_ROOT/snap_${SNAP3}/stage1"
LP_DIR="$OUTPUT_ROOT/runs/dmo/run_0000/lensplanes"
mkdir -p "$LP_DIR"

ALL_SNAP=(); for s in "${SNAPSHOTS[@]}"; do ALL_SNAP+=("$STAGE1_ROOT/snap_$(printf '%03d' "$s")"); done
EXTRA=(); [[ "$IDX" -eq 0 ]] && EXTRA=(--all_snap_dirs "${ALL_SNAP[@]}")

python -u -m bind.cli.paint_lensplane --dmo \
    --stage1_dir "$S1" --output_dir "$LP_DIR" \
    --transforms "$TRANSFORMS" --lc_snap_idx "$IDX" --lc_n_snaps "$N_SNAPS" \
    --lp_grid "$LP_GRID" "${EXTRA[@]}"

echo "=== DMO lensplanes done for snap ${SNAP3} (task ${IDX}) ==="
