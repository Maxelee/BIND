#!/bin/bash
#SBATCH --job-name=bind_emu_stats
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_emu_stats_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_emu_stats_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Emulator stage 1b: backfill WST + DM field stats over an existing suite ────
# The original stats stage wrote Cl_*/peaks/nongaussian/peak_cross but NOT the
# wavelet scattering transform (wst.npz) or the dispersion-measure stats
# (dm_stats.npz).  This GPU array backfills both onto every run whose maps are on
# disk, skip-if-exists (so it is safe to re-run while the suite keeps generating,
# and it picks up new runs automatically).  WST is GPU-bound; one GPU per task,
# each task strides over the run list.
#
# Env overrides: RUNS_DIR (default the SB35 Sobol suite), WST_N_REAL, WST_J,
# WST_L, OVERWRITE=1, EXTRA (raw flags for bind.cli.emulator_stats).

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

RUNS_DIR=${RUNS_DIR:-/mnt/home/mlee1/ceph/bind_sb35/runs}
WST_N_REAL=${WST_N_REAL:-10}
WST_J=${WST_J:-4}
WST_L=${WST_L:-4}
NTASKS=${SLURM_ARRAY_TASK_COUNT:-16}
TID=${SLURM_ARRAY_TASK_ID:-0}

OPT=(--wst_n_real "$WST_N_REAL" --wst_J "$WST_J" --wst_L "$WST_L" --device cuda)
[[ "${OVERWRITE:-0}" == "1" ]] && OPT+=(--overwrite)
OPT+=(${EXTRA:-})

# Stride: this task handles runs TID, TID+NTASKS, TID+2*NTASKS, ...
mapfile -t RUNS < <(ls -d "$RUNS_DIR"/run_* 2>/dev/null | sort)
echo "=== $((${#RUNS[@]})) runs found; task $TID/$NTASKS striding ==="
i=$TID
while (( i < ${#RUNS[@]} )); do
    RD="${RUNS[$i]}"
    if [[ -f "$RD/kappa_maps.npz" || -f "$RD/tau_maps.npz" ]]; then
        echo "--- [$i] $RD ---"
        python -u -m bind.cli.emulator_stats --run_dir "$RD" "${OPT[@]}" || echo "  (failed: $RD)"
    fi
    i=$(( i + NTASKS ))
done
echo "=== task $TID done ==="
