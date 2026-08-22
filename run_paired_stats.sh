#!/bin/bash
#SBATCH --job-name=bind_paired_stats
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_paired_stats_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_paired_stats_%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-60
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Paired (shared-sky) response errors for WL statistics ─────────────────────
# The realizations of a run and the fiducial are rotations of the SAME box, so
# cosmic variance cancels in the per-realization ratio; the paired error is ~10x
# smaller than the marginal one stored in Cl_kappa/peak_counts. Writes
# <run>/paired_stats.npz (response + paired error for clk, pk, min, V0/1/2, sk).
#
# Array: task 0 = the fiducial (builds the shared per-realization cache
#        <FID_DIR>/paired_perreal_fid.npz that all other tasks reuse);
#        tasks 1..60 = twobound run_0000..run_0059. Tasks >0 wait for the cache,
#        then fall back to building it themselves (atomic) if it never appears.
#
#   sbatch run_paired_stats.sh
#   sbatch --array=1-60 run_paired_stats.sh   # if the cache already exists
#
# Afterwards: python examples/dashboard_precompute.py  (+ re-run the notebook).
# Env overrides: OUTPUT_ROOT, DESIGN (default twobound), FID_DIR.

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
DESIGN=${DESIGN:-twobound}
FID_DIR=${FID_DIR:-$OUTPUT_ROOT/runs/bind/run_0000}
CACHE="$FID_DIR/paired_perreal_fid.npz"

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    RD="$FID_DIR"                       # builds the fiducial cache, writes nothing else
else
    RD="$OUTPUT_ROOT/runs/$DESIGN/$(printf 'run_%04d' $((SLURM_ARRAY_TASK_ID - 1)))"
    # let task 0 build the shared cache first (fall back to self-build if it stalls)
    for i in $(seq 1 120); do [[ -f "$CACHE" ]] && break; sleep 10; done
fi

if [[ ! -f "$RD/kappa_maps.npz" ]]; then
    echo "=== $RD: kappa_maps.npz not there — skipping (resubmit later)"; exit 0
fi

echo "=== paired stats: $RD  (fid $FID_DIR) ==="
python -u -m bind.cli.paired_stats --run_dir "$RD" --fid_dir "$FID_DIR"
echo "=== done: $RD ==="
