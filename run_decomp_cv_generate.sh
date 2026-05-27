#!/bin/bash
#SBATCH --job-name=decomp_cv_gen
#SBATCH --output=/mnt/home/mlee1/ceph/logs/decomp_cv_gen_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/decomp_cv_gen_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-11         # 12 GPU tasks; CHUNK*12 must cover all CV halos (~1200-1300)

# GPU-parallel generation of the decomposition cubes over the full CV halo set (~1100 halos).
# Each array task generates one halo chunk -> a partial cube; run the reduce job afterwards.
#
#   MODE=axes  sbatch run_decomp_cv_generate.sh        # then: MODE=axes  sbatch run_decomp_cv_reduce.sh
#   MODE=joint sbatch run_decomp_cv_generate.sh        # then: MODE=joint sbatch run_decomp_cv_reduce.sh
# Tune CHUNK so CHUNK * (array size) >= total CV halos (~1100). Extra/empty chunks self-skip.

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
export PYTHONUNBUFFERED=1

MODE=${MODE:-axes}            # axes | joint
CHUNK=${CHUNK:-120}           # halos per task (120*10=1200 >= ~1100)
K=${K:-12}
N_DESIGN=${N_DESIGN:-128}     # joint only
N_STEPS=${N_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-16}
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

mkdir -p /mnt/home/mlee1/ceph/logs outputs/scatter_diagnostics figures/scatter_diagnostics
echo "=== CV generate: mode=$MODE chunk task $SLURM_ARRAY_TASK_ID -> halos [$START : $((START+CHUNK))] ==="
python -m scatter.scatter_decomposition \
    --base cv --phase generate --mode "$MODE" \
    --halo-start "$START" --halo-count "$CHUNK" \
    --k "$K" --n-design "$N_DESIGN" --n-steps "$N_STEPS" --batch-size "$BATCH_SIZE"
echo "=== chunk done ==="
