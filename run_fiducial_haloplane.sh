#!/bin/bash
#SBATCH --job-name=bind_fid_haloplane
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_fid_haloplane_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_fid_haloplane_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=4
#SBATCH --ntasks=192
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── P2 of docs/ksz_lightcone_map_plan.md: fiducial haloplane+massplane co-trace ─
# ONE lux trace of the FIDUCIAL lightcone with two synthetic fields riding the
# tsz/tau slots (same seed 1992, same 50 realizations, same lensing geometry):
#   yplane  slot <- halo-indicator plane (log10 M200 in R200 disks, >=1e12 FoF)
#   tauplane slot <- total-matter column ("mass-as-tau")
# Products (into $FID/haloplane_trace/):
#   haloplane_maps.npz  — traced halo indicator  (gold-standard halo sky pixels)
#   massplane_maps.npz  — traced matter column   (shared CAP_mat denominator)
#   kappa_retrace_maps.npz — must equal the original kappa_maps.npz (free
#                            end-to-end proof the seed/geometry reproduces)
# The dedicated plane dir $NEW_LP (~21 GB) is kept for possible re-traces;
# delete it after P2 passes. Idempotent: skips painting/tracing where outputs
# exist. Env overrides: N_REAL, RT_SEED, FOV_DEG, LP_GRID, RT_GRID, PAINT_PARALLEL.

set -euo pipefail

FID=/mnt/home/mlee1/ceph/bind_lightcone_tng
FOF_ROOT=/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output
TRANSFORMS="$FID/lightcone_transforms.json"
NEW_LP="$FID/mass_lensplanes"
WORK="$FID/_haloplane_work"
RT_DIR="$WORK/rt_output"
OUT="$FID/haloplane_trace"
N_SNAPS=20
N_REAL=${N_REAL:-50}
LP_GRID=${LP_GRID:-4096}
RT_GRID=${RT_GRID:-1024}
FOV_DEG=${FOV_DEG:-5.0}
RT_SEED=${RT_SEED:-1992}                 # MUST match the fiducial/sobol traces
LUX_BIN=${LUX_BIN:-/mnt/home/mlee1/lux/lux}
PAINT_PARALLEL=${PAINT_PARALLEL:-6}
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

module restore lux
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
# the venv's editable bind points at ~/BIND; the two paint CLIs are (still)
# untracked in this checkout only -> PYTHONPATH wins over the editable install
export PYTHONPATH=/mnt/home/mlee1/BIND-ksz2/src${PYTHONPATH:+:$PYTHONPATH}
cd /mnt/home/mlee1/BIND-ksz2
mkdir -p /mnt/home/mlee1/ceph/logs "$NEW_LP" "$RT_DIR" "$OUT"

if [[ -f "$OUT/haloplane_maps.npz" ]]; then
  echo "=== haloplane_maps.npz exists — nothing to do ==="; exit 0
fi

echo "=== P2 fiducial haloplane+massplane co-trace | N_REAL=$N_REAL seed=$RT_SEED | $(date) ==="

# ── [1/4] paint halo-indicator (yplane) + mass (tauplane) per snapshot ─────────
paint_snapshot() {
  local idx=$1 s3
  s3=$(printf '%03d' "${SNAPSHOTS[$idx]}")
  local log="$WORK/paint_snap_${s3}.log"
  local p0=$(printf '%02d' $((idx * 4 + 1)))
  if [[ ! -f "$NEW_LP/yplane${p0}.dat" ]]; then
    python -u -m bind.cli.paint_haloplane \
        --stage1_dir "$FID/snap_${s3}/stage1" --output_dir "$NEW_LP" \
        --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" --lp_grid "$LP_GRID" \
        --transforms "$TRANSFORMS" --fof_root "$FOF_ROOT" \
        --snapshot "${SNAPSHOTS[$idx]}" > "$log" 2>&1
  fi
  if [[ ! -f "$NEW_LP/tauplane${p0}.dat" ]]; then
    python -u -m bind.cli.paint_massplane \
        --generate_dir "$FID/snap_${s3}" --stage1_dir "$FID/snap_${s3}/stage1" \
        --output_dir "$NEW_LP" --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" \
        --lp_grid "$LP_GRID" >> "$log" 2>&1
  fi
  echo "[paint] snap ${s3} done"
}

echo "=== [1/4] paint halo/mass planes (20 snapshots, -P $PAINT_PARALLEL) ==="
for IDX in $(seq 0 $((N_SNAPS - 1))); do
  paint_snapshot "$IDX" &
  while (( $(jobs -r | wc -l) >= PAINT_PARALLEL )); do wait -n; done
done
wait

# reuse the fiducial lensing potential + geometry verbatim
ln -sf "$FID"/lensplanes/lenspot*.dat "$NEW_LP"/
ln -sf "$FID"/lensplanes/config.dat "$NEW_LP"/config.dat

n_y=$(ls "$NEW_LP"/yplane*.dat 2>/dev/null | wc -l)
n_t=$(ls "$NEW_LP"/tauplane*.dat 2>/dev/null | wc -l)
n_l=$(ls "$NEW_LP"/lenspot*.dat 2>/dev/null | wc -l)
echo "    planes: yplane=$n_y tauplane=$n_t lenspot=$n_l (expect 80 each)"
[[ "$n_y" -eq 80 && "$n_t" -eq 80 && "$n_l" -eq 80 && -e "$NEW_LP/config.dat" ]] \
  || { echo "ERROR: missing planes/config" >&2; exit 1; }

# ── [2/4] lux trace (identical ini to the sobol/fiducial traces, new dirs) ─────
INI="$WORK/lux_haloplane.ini"
cat > "$INI" <<EOF
LP_output_dir = $NEW_LP
RT_output_dir = $RT_DIR
tsz_input_dir = $NEW_LP
tau_input_dir = $NEW_LP
input_dir = $NEW_LP
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
echo "=== [2/4] lux trace (N_REAL=$N_REAL, tasks=${SLURM_NTASKS:-?}) ==="
srun -n "${SLURM_NTASKS}" "$LUX_BIN" "$INI"

# ── [3/4] collect + rename ─────────────────────────────────────────────────────
echo "=== [3/4] collect -> $OUT ==="
srun -N1 -n1 python -u -m bind.cli.lux_collect \
    --rt_root "$RT_DIR" --output_dir "$OUT" --n_real "$N_REAL" --fov_deg "$FOV_DEG"
mv "$OUT/y_maps.npz"     "$OUT/haloplane_maps.npz"
mv "$OUT/tau_maps.npz"   "$OUT/massplane_maps.npz"
mv "$OUT/kappa_maps.npz" "$OUT/kappa_retrace_maps.npz"

# ── [4/4] free end-to-end check: re-traced kappa must match the original ───────
echo "=== [4/4] kappa retrace check ==="
srun -N1 -n1 python - <<'PY'
import numpy as np
a = np.load("/mnt/home/mlee1/ceph/bind_lightcone_tng/kappa_maps.npz")["kappa"]
b = np.load("/mnt/home/mlee1/ceph/bind_lightcone_tng/haloplane_trace/kappa_retrace_maps.npz")["kappa"]
d = np.abs(a - b).max()
rms = float(np.sqrt(np.mean((a - b) ** 2)))
ok = d < 1e-6 * max(1.0, float(np.abs(a).max()))
print(f"kappa retrace: max|diff|={d:.3e} rms={rms:.3e} -> {'PASS' if ok else 'FAIL'}")
print("(FAIL means the trace did NOT reproduce the original geometry — investigate before using the haloplane products)")
PY

rm -rf "$WORK"
echo "=== done; products: ==="
ls -la "$OUT"
echo "=== $(date) ==="
