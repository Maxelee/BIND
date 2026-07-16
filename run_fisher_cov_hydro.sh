#!/bin/bash
#SBATCH --job-name=fisher_cov_hydro
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fisher_cov_hydro_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fisher_cov_hydro_%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Robust WL Fisher covariance from the 2000 hydro-replacement κ maps ────────
# One task per lens-plane set LP_00..LP_19 (100 ray-traced maps each) → the 18
# κ-only data-vector bins (C_ℓ^κκ, N_pk, N_min, V2) on the z_s=1 plane (kappa23),
# saved as a part file. 2000 realizations → Hartlap 0.985 (vs 0.39 for the 50-map
# BIND estimate) → fixes the covariance-noise that inflated the BHRadEff ranking.
#
#   sbatch run_fisher_cov_hydro.sh                          # the 20 LPs (~2.5 min each)
#   # then, once the array finishes (cheap, instant):
#   python examples/fisher_cov_hydro.py --combine           # -> fisher_cov_hydro.npz
#
# κ-only (these maps have no y): fixes the WL block; the tSZ block still needs y maps.

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

echo "=== fisher_cov_hydro: LP_$(printf '%02d' "$SLURM_ARRAY_TASK_ID") ==="
python -u examples/fisher_cov_hydro.py --lp_idx "$SLURM_ARRAY_TASK_ID"
echo "=== done LP $SLURM_ARRAY_TASK_ID ==="
