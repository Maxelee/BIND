#!/bin/bash
#SBATCH --job-name=ksz_sobol
#SBATCH --output=/mnt/home/mlee1/ceph/logs/ksz_sobol_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/ksz_sobol_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-31   # set to 0-(N_SHARDS-1); override with --array at sbatch time

# Generate the fixed-halo Sobol τ cube for kSZ SBI (analysis.ksz.gen_sobol_taucube).
# Each array task paints all CV halos at a contiguous block of Sobol designs.
# N_SHARDS must equal the --array size; N_DESIGN should be divisible by N_SHARDS.
#
#   sbatch --array=0-31 run_ksz_sobol.sh                 # 32 shards
#   N_DESIGN=4096 N_SHARDS=32 sbatch --array=0-31 run_ksz_sobol.sh
#
# After the array finishes, reduce + train NPE (CPU is fine for both):
#   bash run_ksz_sobol.sh reduce
#   bash run_ksz_sobol.sh npe

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Configuration (env-overridable) ──────────────────────────────────────────
RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_cube_two_head}
# IMPORTANT: use the SAME checkpoint that produced fm_testsuite_cube so the
# Sobol τ cube is consistent with the validated cube results.
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoints/last.ckpt}
TESTSUITE_ROOT=${TESTSUITE_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite_cube}
SUITE=${SUITE:-CV}
MODEL_NAME=${MODEL_NAME:-fm_cube_two_head}

N_DESIGN=${N_DESIGN:-4096}
N_SHARDS=${N_SHARDS:-32}
N_DRAWS=${N_DRAWS:-8}
N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-256}
APERTURE=${APERTURE:-cap}
R_AP_MPC_H=${R_AP_MPC_H:-0.5}
MASS_BINS=${MASS_BINS:-"1e13 3e13 1e14 1e15"}
SEED=${SEED:-0}

OUT_DIR=${OUT_DIR:-/mnt/home/mlee1/ceph/ksz_sobol/$MODEL_NAME}
REDUCED_OUT=${REDUCED_OUT:-analysis_physics_cache/ksz_sobol_taucube_${MODEL_NAME}.npz}
NPE_OUT=${NPE_OUT:-analysis_physics_cache/npe_${MODEL_NAME}}
OBSERVABLE=${OBSERVABLE:-rich}

MODE=${1:-generate}

# ── reduce / npe convenience modes ───────────────────────────────────────────
if [[ "$MODE" == "reduce" ]]; then
    python -m analysis.ksz.gen_sobol_taucube --reduce \
        --out_dir "$OUT_DIR" --reduced_out "$REDUCED_OUT"
    exit 0
fi
if [[ "$MODE" == "npe" ]]; then
    python -m analysis.ksz.npe_tau \
        --cube "$REDUCED_OUT" --observable "$OBSERVABLE" --out_dir "$NPE_OUT"
    exit 0
fi

# ── generate (default; one shard per array task) ─────────────────────────────
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
PER=$(( (N_DESIGN + N_SHARDS - 1) / N_SHARDS ))
START=$(( SHARD_ID * PER ))
END=$(( START + PER ))
if (( END > N_DESIGN )); then END=$N_DESIGN; fi
echo "[shard $SHARD_ID] designs [$START, $END) of $N_DESIGN"

python -m analysis.ksz.gen_sobol_taucube \
    --run_dir "$RUN_DIR" --checkpoint "$CHECKPOINT" \
    --testsuite_root "$TESTSUITE_ROOT" --suite "$SUITE" \
    --n_design "$N_DESIGN" --design_start "$START" --design_end "$END" \
    --n_draws "$N_DRAWS" --n_steps "$N_STEPS" --batch_size "$BATCH_SIZE" --use_amp \
    --aperture "$APERTURE" --r_ap_mpc_h "$R_AP_MPC_H" --mass_bins $MASS_BINS \
    --seed "$SEED" --out_dir "$OUT_DIR"
