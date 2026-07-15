#!/bin/bash
#SBATCH --job-name=paper_cache
#SBATCH --output=/mnt/home/mlee1/ceph/logs/paper_cache_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/paper_cache_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=ccm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-7   # set to 0-(N_CHUNKS-1)

# Build the parallel per-metric caches for the BIND2 paper figures
# (tools/paper_cache/build_metric.py), then reduce them into the compact
# artifacts the load-only notebooks read.
#
# CPU only. Reads the suite-eval outputs under PAPER_SUITE_ROOT and writes to
# PAPER_CACHE_DIR (both default to the dev fm_thermo/fm_lowmass locations; export
# the spine values below to build the paper's fm_redshift @ z=0 cache).
#
# Three ways to run
# -----------------
#   # (a) LOCAL — one node, multiprocessing pool, build + field1p + reduce:
#   POOL=16 bash run_paper_cache.sh
#
#   # (b) SLURM build array (one chunk per task, all metrics), then a dependent reduce:
#   jid=$(N_CHUNKS=8 sbatch --parse --array=0-7 run_paper_cache.sh)
#   REDUCE=1 sbatch --dependency=afterok:$jid run_paper_cache.sh
#
#   # Spine flip (Phase III): prepend these to any of the above
#   export PAPER_SUITE_ROOT=/mnt/home/mlee1/ceph/fm_redshift_suite
#   export PAPER_MODEL_SUBDIR=fm_redshift_ema
#   export PAPER_MODEL_TAG=fm_redshift

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2/tools/paper_cache

METRICS=${METRICS:-"mass profiles profiles_r200 pk shapes thermo"}
N_CHUNKS=${N_CHUNKS:-8}
POOL=${POOL:-1}
CHUNK=${SLURM_ARRAY_TASK_ID:-}

mkdir -p /mnt/home/mlee1/ceph/logs

echo "=== paper_cache  metrics='$METRICS'  cache=$(python -c 'import paper_config as C; print(C.CACHE_DIR)') ==="

if [[ "${REDUCE:-0}" == "1" ]]; then
    for m in $METRICS; do python build_metric.py --metric "$m" --reduce; done
    python build_metric.py --metric field1p
    echo "=== reduce done ==="
    exit 0
fi

if [[ -z "$CHUNK" ]]; then
    # LOCAL: build everything with a pool, field1p, then reduce.
    for m in $METRICS; do python build_metric.py --metric "$m" --pool "$POOL"; done
    python build_metric.py --metric field1p
    for m in $METRICS; do python build_metric.py --metric "$m" --reduce; done
    echo "=== local build+reduce done ==="
else
    # SLURM array: this task builds its chunk for every metric (resumable).
    for m in $METRICS; do
        python build_metric.py --metric "$m" --chunk "$CHUNK" --n-chunks "$N_CHUNKS" --pool "$POOL"
    done
    echo "=== [chunk $CHUNK/$N_CHUNKS] build done (submit REDUCE=1 job afterok) ==="
fi
