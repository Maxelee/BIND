#!/bin/bash
#SBATCH --job-name=bind_sobol_stats
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_sobol_stats_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_sobol_stats_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-99
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND science STAGE 5 (CPU): collect lux maps -> kappa/y npz -> statistics ─
# Array index = run index.  Stacks the raw lux .dat into kappa_maps.npz /
# y_maps.npz (deleting the raw .dat), computes all summary stats (Cℓ, peaks,
# Minkowski, R(ν), halo scaling), then deletes the transient lensplanes.
# KEPT per run: snap_*/composite_slab*.npz (halos), kappa_maps.npz, y_maps.npz,
# Cl_*, peak_*, nongaussian_*, halo_scaling.npz.
#
# Env overrides: DESIGN, OUTPUT_ROOT, FOV_DEG.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

DESIGN=${DESIGN:-sobol}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
FOV_DEG=${FOV_DEG:-5.0}

RUN_DIR="$OUTPUT_ROOT/runs/$DESIGN/$(printf 'run_%04d' "$SLURM_ARRAY_TASK_ID")"

python -u -m bind.cli.lux_collect --rt_root "$RUN_DIR/rt" --output_dir "$RUN_DIR" \
    --fov_deg "$FOV_DEG" --delete_raw
python -u -m bind.cli.lightcone_stats --run_dir "$RUN_DIR" --snap_root "$RUN_DIR"

# drop the large transients; halos/kappa/y/stats are kept
rm -rf "$RUN_DIR/lensplanes" "$RUN_DIR/rt" "$RUN_DIR/lux.ini"
echo "=== stats done; transients cleaned for $RUN_DIR ==="
