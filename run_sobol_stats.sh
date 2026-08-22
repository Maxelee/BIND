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
# Env overrides: DESIGN, OUTPUT_ROOT, FOV_DEG, SMOOTHING ("1 2 5 8"), NGAL
# (shape noise gal/arcmin^2 -> writes peak_*_ngal<N>.npz; noiseless files kept),
# SIGMA_E, STATS_EXTRA (raw extra flags for bind.cli.lightcone_stats).

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

DESIGN=${DESIGN:-sobol}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
FOV_DEG=${FOV_DEG:-5.0}

RUN_DIR="$OUTPUT_ROOT/runs/$DESIGN/$(printf 'run_%04d' "$SLURM_ARRAY_TASK_ID")"
RT_ROOT=${RT_ROOT:-$RUN_DIR/rt}        # BIND fiducial: bind_lightcone_tng/rt_output
SNAP_ROOT=${SNAP_ROOT:-$RUN_DIR}       # composites for halo_scaling
N_REAL=${N_REAL:-}                     # cap realizations loaded (memory)
KEEP_RAW=${KEEP_RAW:-0}                # 1 = keep raw .dat + transients (e.g. BIND fiducial)
mkdir -p "$RUN_DIR"

COLLECT_OPT=(); [[ "$KEEP_RAW" == "0" ]] && COLLECT_OPT+=(--delete_raw)
[[ -n "$N_REAL" ]] && COLLECT_OPT+=(--n_real "$N_REAL")

SMOOTHING=${SMOOTHING:-2.0}            # peak smoothing scale(s), e.g. "1 2 5 8"
NGAL=${NGAL:-}                         # shape-noise source density (off if empty)
SIGMA_E=${SIGMA_E:-0.26}
STATS_OPT=(--smoothing_arcmin $SMOOTHING)
[[ -n "$NGAL" ]] && STATS_OPT+=(--shape_noise_ngal "$NGAL" --sigma_e "$SIGMA_E")
STATS_OPT+=(${STATS_EXTRA:-})

python -u -m bind.cli.lux_collect --rt_root "$RT_ROOT" --output_dir "$RUN_DIR" \
    --fov_deg "$FOV_DEG" "${COLLECT_OPT[@]}"
python -u -m bind.cli.lightcone_stats --run_dir "$RUN_DIR" --snap_root "$SNAP_ROOT" \
    "${STATS_OPT[@]}"

if [[ "$KEEP_RAW" == "0" ]]; then
    rm -rf "$RUN_DIR/lensplanes" "$RUN_DIR/rt" "$RUN_DIR/lux.ini"
    echo "=== stats done; transients cleaned for $RUN_DIR ==="
else
    echo "=== stats done; raw kept (KEEP_RAW=1) for $RUN_DIR ==="
fi
