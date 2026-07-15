#!/bin/bash
#SBATCH --job-name=pk_fixed
#SBATCH --output=/mnt/home/mlee1/ceph/logs/pk_fixed_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/pk_fixed_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-7   # set to 0-(N_CHUNKS-1)

# Rebuild the corrected full-box P(k) cache (tools/paper_cache/build_pk_fixed.py,
# `fixed` metric): shared-paste >=1e13 / >=1e12 BIND composites + the NEW hr_ge13
# hydro-replace control (truth patches of the same >=1e13 halos, same shared
# paste) used by Fig 5. Partials that predate hr_ge13 are recomputed
# automatically; up-to-date ones are skipped, so the array is resumable.
#
# CPU only. Reads suite-eval outputs under PAPER_SUITE_ROOT, writes partials +
# pk_fixed.npz into PAPER_CACHE_DIR (defaults = the dev fm_thermo/fm_lowmass
# locations; export the spine values as in run_paper_cache.sh to flip).
#
# How to run
# ----------
#   # (a) SLURM build array, then a dependent reduce (single task):
#   jid=$(N_CHUNKS=8 sbatch --parsable --array=0-7 run_pk_fixed.sh)
#   REDUCE=1 sbatch --dependency=afterok:$jid --array=0 --cpus-per-task=2 --time=00:30:00 run_pk_fixed.sh
#
#   # (b) LOCAL — one node, multiprocessing pool, build + reduce:
#   POOL=16 bash run_pk_fixed.sh

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2/tools/paper_cache

N_CHUNKS=${N_CHUNKS:-8}
POOL=${POOL:-${SLURM_CPUS_PER_TASK:-1}}
CHUNK=${SLURM_ARRAY_TASK_ID:-}

mkdir -p /mnt/home/mlee1/ceph/logs

echo "=== pk_fixed  cache=$(python -c 'import paper_config as C; print(C.CACHE_DIR)') ==="

if [[ "${REDUCE:-0}" == "1" ]]; then
    python build_pk_fixed.py --reduce
    echo "=== reduce done (pk_fixed.npz) ==="
    exit 0
fi

if [[ -z "$CHUNK" ]]; then
    # LOCAL: build everything with a pool, then reduce.
    python build_pk_fixed.py --metric fixed --pool "$POOL"
    python build_pk_fixed.py --reduce
    echo "=== local build+reduce done ==="
else
    # SLURM array: this task builds its chunk (resumable).
    python build_pk_fixed.py --metric fixed --chunk "$CHUNK" --n-chunks "$N_CHUNKS" --pool "$POOL"
    echo "=== [chunk $CHUNK/$N_CHUNKS] build done (submit REDUCE=1 job afterok) ==="
fi
