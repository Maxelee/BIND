#!/bin/bash
#SBATCH --job-name=bind_sobol_lc
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_sobol_lc_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_sobol_lc_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=4
#SBATCH --ntasks=192
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH --array=0-255%8
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND SB35 Sobol lightcone: full lux ray-trace per parameter run ────────────
# One SLURM array task = one Sobol run (run_NNNN of 256).  IDENTICAL to the
# twobound pipeline (run_twobound_lightcone.sh) — only the run bundle and the
# array range differ.  The 256 Sobol halo sets were generated off-cluster from
# the SAME fiducial DMO conditions (the run_lightcone_project.sh stage1 cutouts),
# so each run stores only generated_patches/thermo_patches in
# snap_<NNN>/composite_slab*.npz (no pasted composites, no lensplanes, no
# rt_output).  Verified halo-for-halo: generated halo_centers == fiducial stage1.
#
#   1. recomposite patches -> composite + composite_thermo   (reuses the FIDUCIAL
#      snap_<NNN>/stage1 cutouts+manifest; the DMO lightcone + halos are identical
#      across all runs, verified halo-for-halo).
#   2. paint lenspot + yplane + tauplane                      (reuses the FIDUCIAL
#      lightcone_transforms.json; config.dat written from the shared geometry).
#   3. lux trace  compute_tsz=True, compute_tau=True          (SAME RT_SEED and
#      transforms as the fiducial -> the kappa/y/tau differences between runs are
#      PURELY parametric; cosmic variance cancels realization-by-realization).
#   4. collect -> {kappa,y,tau}_maps.npz   (self-consistent kappa+y+tau set on one
#      geometry, written into the run dir).
#   5. stats   -> Cl_kappa / Cl_kappa_y / Cl_tau.npz.
#   6. clean up the bulky composites/lensplanes/rt_output (keep only the npz).
#
# Idempotent: a run whose tau_maps.npz already exists is skipped; recomposite/
# trace resume if partial.  Throttle concurrency with the array '%N' (disk: each
# run needs ~75-140 GB transient under $WORK).
#
# Env overrides: RUNS_ROOT, FID, N_REAL, LP_GRID, RT_GRID, FOV_DEG, RT_SEED,
#                COMPUTE_TSZ, LUX_BIN, WORK_ROOT, PAINT_PARALLEL, FORCE.

set -euo pipefail

# ── shared paths / config ─────────────────────────────────────────────────────
RUNS_ROOT=${RUNS_ROOT:-/mnt/home/mlee1/ceph/bind_sb35/runs}   # 256 Sobol runs
FID=${FID:-/mnt/home/mlee1/ceph/bind_lightcone_tng}     # shared geometry source
TRANSFORMS="$FID/lightcone_transforms.json"
N_SNAPS=20
N_REAL=${N_REAL:-50}
LP_GRID=${LP_GRID:-4096}
RT_GRID=${RT_GRID:-1024}
FOV_DEG=${FOV_DEG:-5.0}
RT_SEED=${RT_SEED:-1992}                 # MUST match the fiducial for a paired response
COMPUTE_TSZ=${COMPUTE_TSZ:-True}
LUX_BIN=${LUX_BIN:-/mnt/home/mlee1/lux/lux}
PAINT_PARALLEL=${PAINT_PARALLEL:-6}      # snapshots painted concurrently on the head node
FORCE=${FORCE:-0}                        # 1 = re-trace even if tau_maps.npz exists
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

RUN=$(printf 'run_%04d' "${SLURM_ARRAY_TASK_ID}")
RUN_DIR="$RUNS_ROOT/$RUN"
WORK="${WORK_ROOT:-$RUN_DIR}/_work"      # scratch for composites + lensplanes + rt_output
LP_DIR="$WORK/lensplanes"
RT_DIR="$WORK/rt_output"

module restore lux                                   # openmpi etc. for srun-launched lux
source /mnt/home/mlee1/venvs/BIND_env/bin/activate   # bind CLIs (plain venv, rpath'd lux)
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

if [[ ! -d "$RUN_DIR" ]]; then echo "ERROR: no run dir $RUN_DIR" >&2; exit 1; fi
if [[ -f "$RUN_DIR/tau_maps.npz" && "$FORCE" != "1" ]]; then
  echo "=== $RUN already has tau_maps.npz — skipping (FORCE=1 to redo) ==="; exit 0
fi
mkdir -p "$LP_DIR" "$RT_DIR"

# all-snapshot FIDUCIAL generate dirs (for paint_lensplane's config.dat on snap 0)
ALL_FID_SNAPS=()
for s in "${SNAPSHOTS[@]}"; do ALL_FID_SNAPS+=("$FID/snap_$(printf '%03d' "$s")"); done

echo "=== sobol lightcone | $RUN | N_REAL=$N_REAL seed=$RT_SEED | $(date) ==="

# ── per-snapshot: recomposite + paint lenspot/yplane/tauplane ──────────────────
paint_snapshot() {
  local idx=$1 s3 fid_snap gen_snap log
  s3=$(printf '%03d' "${SNAPSHOTS[$idx]}")
  fid_snap="$FID/snap_${s3}"; gen_snap="$RUN_DIR/snap_${s3}"
  log="$WORK/paint_snap_${s3}.log"

  # 1. recomposite this run's patches onto the shared DMO background (skip if done)
  if [[ ! -f "$WORK/snap_${s3}/composite_slab00.npz" ]]; then
    python -u -m bind.cli.paint_recomposite \
        --stage1_dir "$fid_snap/stage1" --generated_dir "$gen_snap" \
        --output_dir "$WORK/snap_${s3}" > "$log" 2>&1
  fi
  # 2. lenspot (writes config.dat on idx 0) + yplane + tauplane
  python -u -m bind.cli.paint_lensplane \
      --generate_dir "$WORK/snap_${s3}" --stage1_dir "$fid_snap/stage1" \
      --output_dir "$LP_DIR" --transforms "$TRANSFORMS" \
      --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" --lp_grid "$LP_GRID" \
      --all_snap_dirs "${ALL_FID_SNAPS[@]}" >> "$log" 2>&1
  python -u -m bind.cli.paint_yplane \
      --generate_dir "$WORK/snap_${s3}" --stage1_dir "$fid_snap/stage1" \
      --output_dir "$LP_DIR" --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" \
      --lp_grid "$LP_GRID" >> "$log" 2>&1
  python -u -m bind.cli.paint_tauplane \
      --generate_dir "$WORK/snap_${s3}" --stage1_dir "$fid_snap/stage1" \
      --output_dir "$LP_DIR" --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" \
      --lp_grid "$LP_GRID" >> "$log" 2>&1
  # 3. composites are big and only needed to build the planes — drop them now
  rm -rf "$WORK/snap_${s3}"
  echo "[paint] snap ${s3} done"
}

echo "=== [1/4] recomposite + paint planes (20 snapshots, -P $PAINT_PARALLEL) ==="
for IDX in $(seq 0 $((N_SNAPS - 1))); do
  paint_snapshot "$IDX" &
  while (( $(jobs -r | wc -l) >= PAINT_PARALLEL )); do wait -n; done
done
wait
n_lenspot=$(ls "$LP_DIR"/lenspot*.dat 2>/dev/null | wc -l)
echo "    painted $n_lenspot lenspot planes (expect $((N_SNAPS * 4)))"
[[ "$n_lenspot" -eq $((N_SNAPS * 4)) ]] || { echo "ERROR: missing lensplanes" >&2; exit 1; }

# ── lux re-trace (MPI across the whole allocation) ─────────────────────────────
INI="$WORK/lux_sobol.ini"
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
compute_tsz = $COMPUTE_TSZ
compute_tau = True
n_realizations = $N_REAL
verbose = True
EOF
for i in $(seq 1 "$N_REAL"); do mkdir -p "$RT_DIR/$(printf 'run%03d' "$i")"; done
echo "=== [2/4] lux trace ($RUN, N_REAL=$N_REAL, nodes=${SLURM_NNODES:-?}, tasks=${SLURM_NTASKS:-?}) ==="
srun -n "${SLURM_NTASKS}" "$LUX_BIN" "$INI"

# ── collect + stats into the run dir ──────────────────────────────────────────
echo "=== [3/4] collect -> $RUN_DIR ==="
srun -N1 -n1 python -u -m bind.cli.lux_collect \
    --rt_root "$RT_DIR" --output_dir "$RUN_DIR" --n_real "$N_REAL" --fov_deg "$FOV_DEG"
echo "=== [4/4] stats -> Cl_*.npz ==="
srun -N1 -n1 python -u -m bind.cli.lightcone_stats --run_dir "$RUN_DIR" --no_halo_scaling

# ── clean up bulky intermediates (keep only the npz products) ──────────────────
rm -rf "$WORK"
echo "=== $RUN done; products: ==="
ls -la "$RUN_DIR"/{kappa,y,tau}_maps.npz "$RUN_DIR"/Cl_tau.npz 2>/dev/null
echo "=== $(date) ==="
