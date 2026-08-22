#!/bin/bash
#SBATCH --job-name=bind_val_trace
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_val_trace_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_val_trace_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=3
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── TNG300 validation traces (stage B): paint planes -> lux -> collect -> stats ─
# Array task 0 = VARIANT hydro_full  (full-box hydro composites from stage A)
# Array task 1 = VARIANT diffuse     (truth pasted composites + f_b*DMO*(1-alpha)
#                                     diffuse gas at T=1e4K, built inline)
# Both trace N_REAL=50 realizations with the canonical base seed 1992 — the SAME
# plane geometry per realization as every existing run, so the comparison vs the
# truth trace (first 50 of its 550) is paired realization-by-realization.
# kappa is expected UNCHANGED for 'diffuse' (mass is moved between channels, not
# added); tau/y gain the diffuse column.
#
# Submit AFTER run_hydro_full_project.sh completes (task 0 checks and exits if
# composites are missing; task 1 is independent of stage A).
#
# Env overrides: OUT_ROOT, TRUTH, FID, N_REAL, T_DIFFUSE_K, F_B, PAINT_PARALLEL.

set -euo pipefail

module restore lux
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

FID=${FID:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
TRUTH=${TRUTH:-/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000}
OUT_ROOT=${OUT_ROOT:-/mnt/home/mlee1/ceph/tng_full_validation}
N_REAL=${N_REAL:-50}
LP_GRID=${LP_GRID:-4096}
RT_GRID=${RT_GRID:-1024}
FOV_DEG=${FOV_DEG:-5.0}
RT_SEED=1992                       # canonical ladder — MUST stay paired
T_DIFFUSE_K=${T_DIFFUSE_K:-1e4}
F_B=${F_B:-}                       # empty = CLI default (0.0486/0.3089)
LUX_BIN=${LUX_BIN:-/mnt/home/mlee1/lux/lux}
PAINT_PARALLEL=${PAINT_PARALLEL:-6}
N_SNAPS=20
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

case "${SLURM_ARRAY_TASK_ID:?run as array 0-1}" in
  0) VARIANT=hydro_full ;;
  1) VARIANT=diffuse ;;
  *) echo "ERROR: array index must be 0 (hydro_full) or 1 (diffuse)" >&2; exit 1 ;;
esac
OUT_RUN="$OUT_ROOT/runs/$VARIANT/run_0000"
WORK="$OUT_RUN/_work"
LP_DIR="$WORK/lensplanes"
RT_DIR="$WORK/rt"
unset GSL_RNG_TYPE GSL_RNG_SEED || true

if [[ -f "$OUT_RUN/tau_maps.npz" ]]; then
  echo "=== $VARIANT already traced — skipping ==="; exit 0
fi
mkdir -p "$LP_DIR" "$RT_DIR" "$OUT_RUN"

ALL_FID_SNAPS=()
for s in "${SNAPSHOTS[@]}"; do ALL_FID_SNAPS+=("$FID/snap_$(printf '%03d' "$s")"); done

manifest_a() {   # scale_factor of snapshot $1 from the fiducial stage1 manifest
  python -c "import json,sys; print(json.load(open(sys.argv[1]))['scale_factor'])" \
      "$FID/snap_$1/stage1/stage1_manifest.json"
}

echo "=== validation trace | $VARIANT | N_REAL=$N_REAL seed=$RT_SEED | $(date) ==="

# ── stage 0/1: composites + planes, per snapshot ──────────────────────────────
paint_one() {
  local idx=$1 s3 comp log a_l
  s3=$(printf '%03d' "${SNAPSHOTS[$idx]}")
  log="$WORK/paint_snap_${s3}.log"
  if [[ "$VARIANT" == hydro_full ]]; then
    comp="$OUT_ROOT/composites/snap_${s3}"
    [[ -f "$comp/summary.json" ]] || { echo "ERROR: missing stage-A composites $comp" >&2; return 1; }
  else
    # truth stores patches only -> recomposite first (gives composite/alpha/dmo),
    # then add the diffuse component outside the paste mask
    local base="$WORK/composites_base/snap_${s3}"
    comp="$WORK/composites/snap_${s3}"
    if [[ ! -f "$comp/summary.json" ]]; then
      python -u -m bind.cli.paint_recomposite \
          --stage1_dir "$FID/snap_${s3}/stage1" --generated_dir "$TRUTH/snap_${s3}" \
          --output_dir "$base" > "$log" 2>&1
      a_l=$(manifest_a "$s3")
      FB_OPT=(); [[ -n "$F_B" ]] && FB_OPT=(--f_b "$F_B")
      python -u -m bind.cli.paint_diffuse_composite \
          --composite_dir "$base" --output_dir "$comp" \
          --scale_factor "$a_l" --t_diffuse_K "$T_DIFFUSE_K" \
          --y_convention legacy "${FB_OPT[@]}" >> "$log" 2>&1
      rm -rf "$base"
    fi
  fi
  python -u -m bind.cli.paint_lensplane \
      --generate_dir "$comp" --stage1_dir "$FID/snap_${s3}/stage1" \
      --output_dir "$LP_DIR" --transforms "$FID/lightcone_transforms.json" \
      --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" --lp_grid "$LP_GRID" \
      --all_snap_dirs "${ALL_FID_SNAPS[@]}" >> "$log" 2>&1
  python -u -m bind.cli.paint_yplane \
      --generate_dir "$comp" --stage1_dir "$FID/snap_${s3}/stage1" \
      --output_dir "$LP_DIR" --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" \
      --lp_grid "$LP_GRID" >> "$log" 2>&1
  python -u -m bind.cli.paint_tauplane \
      --generate_dir "$comp" --stage1_dir "$FID/snap_${s3}/stage1" \
      --output_dir "$LP_DIR" --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" \
      --lp_grid "$LP_GRID" >> "$log" 2>&1
  echo "[paint] $VARIANT snap ${s3} done"
}

n_lp() { ls "$LP_DIR"/"$1"*.dat 2>/dev/null | wc -l; }
if [[ "$(n_lp lenspot)" -eq $((N_SNAPS*4)) && "$(n_lp yplane)" -eq $((N_SNAPS*4)) && "$(n_lp tauplane)" -eq $((N_SNAPS*4)) ]]; then
  echo "=== [1/4] planes already complete ==="
else
  echo "=== [1/4] composites + planes (20 snapshots, -P $PAINT_PARALLEL) ==="
  rm -rf "$LP_DIR"; mkdir -p "$LP_DIR"
  for I in $(seq 0 $((N_SNAPS-1))); do
    paint_one "$I" &
    while (( $(jobs -r | wc -l) >= PAINT_PARALLEL )); do wait -n; done
  done
  wait
  [[ "$(n_lp lenspot)" -eq $((N_SNAPS*4)) && "$(n_lp yplane)" -eq $((N_SNAPS*4)) && "$(n_lp tauplane)" -eq $((N_SNAPS*4)) ]] \
      || { echo "ERROR: incomplete planes" >&2; exit 1; }
fi

# ── stage 2: lux trace, canonical seeds ───────────────────────────────────────
INI="$WORK/lux.ini"
cat > "$INI" <<EOF
LP_output_dir = $LP_DIR
RT_output_dir = $RT_DIR
tsz_input_dir = $LP_DIR
tau_input_dir = $LP_DIR
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
RT_random_seed = $RT_SEED
RT_randomization = True
compute_tsz = True
compute_tau = True
n_realizations = $N_REAL
verbose = True
EOF
for i in $(seq 1 "$N_REAL"); do mkdir -p "$RT_DIR/$(printf 'run%03d' "$i")"; done
echo "=== [2/4] lux trace ($VARIANT, N_REAL=$N_REAL) | $(date) ==="
srun -n "${SLURM_NTASKS}" "$LUX_BIN" "$INI"

# ── stage 3/4: collect + stats ────────────────────────────────────────────────
echo "=== [3/4] collect -> $OUT_RUN ==="
srun -N1 -n1 python -u -m bind.cli.lux_collect \
    --rt_root "$RT_DIR" --output_dir "$OUT_RUN" --n_real "$N_REAL" \
    --fov_deg "$FOV_DEG" --delete_raw
echo "=== [4/4] stats ==="
srun -N1 -n1 python -u -m bind.cli.lightcone_stats --run_dir "$OUT_RUN" --no_halo_scaling

rm -rf "$WORK"
echo "=== $VARIANT trace complete | $(date) ==="
ls -la "$OUT_RUN"/*_maps.npz
