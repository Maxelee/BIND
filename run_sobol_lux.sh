#!/bin/bash
#SBATCH --job-name=bind_sobol_lux
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_sobol_lux_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_sobol_lux_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=4
#SBATCH --ntasks=192
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND science STAGE 4 (CPU/MPI): lux ray-trace kappa + tSZ-y per run ───────
# Array index = run index.  Writes a per-run lux ini (LP/RT dirs -> this run) and
# ray-traces N_REAL random realizations, emitting kappa + y at the 5 source
# planes.  Ray-tracing scales with cores — raise --nodes/--ntasks for speed.
#
# Env overrides: DESIGN, OUTPUT_ROOT, N_REAL, LUX_BIN, FOV_DEG, LP_GRID, RT_GRID.

set -euo pipefail

module restore lux
cd /mnt/home/mlee1/lux
mkdir -p /mnt/home/mlee1/ceph/logs

DESIGN=${DESIGN:-sobol}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
N_REAL=${N_REAL:-8}
LUX_BIN=${LUX_BIN:-/mnt/home/mlee1/lux/lux}
FOV_DEG=${FOV_DEG:-5.0}
LP_GRID=${LP_GRID:-4096}
RT_GRID=${RT_GRID:-1024}

RUN_DIR="$OUTPUT_ROOT/runs/$DESIGN/$(printf 'run_%04d' "$SLURM_ARRAY_TASK_ID")"
LP_DIR="$RUN_DIR/lensplanes"
RT_DIR="$RUN_DIR/rt"
INI="$RUN_DIR/lux.ini"
mkdir -p "$RT_DIR"

cat > "$INI" <<EOF
LP_output_dir = $LP_DIR
RT_output_dir = $RT_DIR
tsz_input_dir = $LP_DIR
input_dir = $LP_DIR
LP_grid = $LP_GRID
RT_grid = $RT_GRID
planes_per_snapshot = 4
angle = $FOV_DEG
simulation_format = PreProjected
snapshot_list = 96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29
snapshot_stack = false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false
output_planes = 26, 45, 59, 70, 78
projection_direction = -1
translation_rotation = False
RT_random_seed = 1992
RT_randomization = True
compute_tsz = True
n_realizations = $N_REAL
verbose = True
EOF

echo "=== lux: $RUN_DIR  (N_REAL=$N_REAL, nodes=${SLURM_NNODES}, tasks=${SLURM_NTASKS}) ==="
srun -n "${SLURM_NTASKS}" "$LUX_BIN" "$INI"
echo "=== lux done for $RUN_DIR ==="
