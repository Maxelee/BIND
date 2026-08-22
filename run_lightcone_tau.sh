#!/bin/bash
#SBATCH --job-name=bind_lc_tau
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_lc_tau_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_lc_tau_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=4
#SBATCH --ntasks=192
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND lightcone: add the kSZ/FRB electron-column (tau) leg to an existing ───
# kappa+y lightcone, end-to-end in ONE allocation:
#   1. paint tau-planes from the retained composites into the lensplane dir
#   2. lux re-trace with compute_tau=True  (kappa/y reproduce; tau is new)
#   3. bind-lux-collect    -> tau_maps.npz (+ kappa/y)
#   4. bind-lightcone-stats -> Cl_tau.npz  (kappa x tau, tau x tau, + y x tau)
#
# The tau science (kappa x tau, tau x tau, y x tau) is self-consistent because
# all three maps come from the SAME re-trace.  compute_tsz=True regenerates y on
# the same geometry as the new kappa/tau (set False only if RT_SEED matches the
# run that made your existing y and you want to keep it / save time).
#
# Env overrides: LC, N_SNAPS, LP_GRID, RT_GRID, N_REAL, FOV_DEG, RT_SEED,
#                COMPUTE_TSZ, LUX_BIN.

set -euo pipefail

LC=${LC:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
LP_DIR="$LC/lensplanes"
RT_DIR="$LC/rt_output"
N_SNAPS=${N_SNAPS:-20}
LP_GRID=${LP_GRID:-4096}
RT_GRID=${RT_GRID:-1024}
N_REAL=${N_REAL:-50}                 # realizations lux traces AND stats collects
FOV_DEG=${FOV_DEG:-5.0}
RT_SEED=${RT_SEED:-1992}             # MUST match the seed that made the existing kappa/y
COMPUTE_TSZ=${COMPUTE_TSZ:-True}     # regenerate y on the new geometry (safe default)
LUX_BIN=${LUX_BIN:-/mnt/home/mlee1/lux/lux}
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

module restore lux                                   # openmpi etc. for srun-launched lux
source /mnt/home/mlee1/venvs/BIND_env/bin/activate   # bind CLIs (plain venv, rpath'd lux)
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs
# lux does NOT create its own run<NNN>/ output dirs — ensure they exist
for i in $(seq 1 "$N_REAL"); do mkdir -p "$RT_DIR/$(printf 'run%03d' "$i")"; done

# ── 1. paint the tau-planes into the existing lensplane dir ───────────────────
echo "=== [1/4] painting tau-planes -> $LP_DIR ==="
for IDX in $(seq 0 $((N_SNAPS - 1))); do
  S3=$(printf '%03d' "${SNAPSHOTS[$IDX]}")
  python -u -m bind.cli.paint_tauplane \
      --generate_dir "$LC/snap_${S3}" --output_dir "$LP_DIR" \
      --lc_snap_idx "$IDX" --lc_n_snaps "$N_SNAPS" --lp_grid "$LP_GRID"
done

# ── 2. lux re-trace with compute_tau (MPI across the whole allocation) ─────────
INI="$LC/lux_tau.ini"
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
echo "=== [2/4] lux re-trace (compute_tau=True, N_REAL=$N_REAL, nodes=${SLURM_NNODES}, tasks=${SLURM_NTASKS}) ==="
srun -n "${SLURM_NTASKS}" "$LUX_BIN" "$INI"

# ── 3. collect lux maps -> tau_maps.npz (+ kappa/y) ───────────────────────────
echo "=== [3/4] collecting maps -> $LC ==="
srun -N1 -n1 python -u -m bind.cli.lux_collect \
    --rt_root "$RT_DIR" --output_dir "$LC" --n_real "$N_REAL" --fov_deg "$FOV_DEG"

# ── 4. statistics -> Cl_tau.npz (also (re)writes Cl_kappa / Cl_kappa_y here) ───
echo "=== [4/4] statistics -> Cl_tau.npz ==="
srun -N1 -n1 python -u -m bind.cli.lightcone_stats --run_dir "$LC" --no_halo_scaling

echo "=== tau leg done for $LC ==="
ls -la "$LC"/tau_maps.npz "$LC"/Cl_tau.npz
