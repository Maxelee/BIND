#!/bin/bash
#SBATCH --job-name=bind_nufid_restats
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_nufid_restats_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_nufid_restats_%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt               # Sets Quality of Service to preemptible, if required
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-60
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Fixed-ν peak statistics: ν = κ_sm / σ_fid (ONE σ per source bin, from the
# fiducial), so parameter responses are not absorbed by per-map normalization.
# Writes peak_counts_nufid.npz / peak_cross_nufid.npz alongside the defaults.
#
# Array: task 0 = the fiducial itself (defines/uses its own σ);
#        tasks 1..60 = twobound run_0000..run_0059 (skipped gracefully until
#        their kappa_maps.npz lands — just resubmit later for the stragglers).
#
#   sbatch run_nufid_restats.sh
#   sbatch --array=13-60 run_nufid_restats.sh     # only the late arrivals
#
# Afterwards: python examples/dashboard_precompute.py  (+ re-run the dashboard).
# Env overrides: OUTPUT_ROOT, DESIGN (default twobound), FID_DIR.

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/bind_science}
DESIGN=${DESIGN:-twobound}
FID_DIR=${FID_DIR:-$OUTPUT_ROOT/runs/bind/run_0000}

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    RD="$FID_DIR"
else
    RD="$OUTPUT_ROOT/runs/$DESIGN/$(printf 'run_%04d' $((SLURM_ARRAY_TASK_ID - 1)))"
fi

if [[ ! -f "$RD/kappa_maps.npz" ]]; then
    echo "=== $RD: kappa_maps.npz not there yet — skipping (resubmit this task later)"
    exit 0
fi

echo "=== fixed-nu restats: $RD  (sigma0 from $FID_DIR) ==="
python -u -m bind.cli.lightcone_stats --run_dir "$RD" --peaks_only \
    --nu_sigma0_from "$FID_DIR"
echo "=== done: $RD ==="
