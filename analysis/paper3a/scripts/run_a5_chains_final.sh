#!/bin/bash
#SBATCH --job-name=a5_final
#SBATCH --output=/mnt/home/mlee1/ceph/logs/a5_final_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/a5_final_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A5 convergence-grade chains: 4 seeds in parallel + assembly ──────────
# Decision 2 (Max, 2026-07-18): publication-grade re-run before ruling on
# the exclusion framing (1) and A6 (3). DE moves fix the 6% acceptance;
# four independent seeds give a real cross-chain R-hat. ~2-3 h/seed in
# parallel + minutes of assembly.
#
# ⛔ Submit:  sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_a5_chains_final.sh

set -euo pipefail

module load gcc
source /mnt/home/mlee1/venvs/torch3/bin/activate
export OMP_NUM_THREADS=8
cd /mnt/home/mlee1/vdm_bind2-paper3a

for K in 0 1 2 3; do
  srun --exclusive -n1 -c8 python -u analysis/paper3a/scripts/run_a5_chains_final.py $K &
done
wait

python -u analysis/paper3a/scripts/run_a5_chains_final.py --assemble
