#!/bin/bash
#SBATCH --job-name=a5_subsets
#SBATCH --output=/mnt/home/mlee1/ceph/logs/a5_subsets_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/a5_subsets_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A5 task 5: recovery batteries + probe-subset fits, one job ───────────
# Stage 1: disBatch fans the 32 battery injections (16 kszonly + 16 joint)
#          across the node, ~15-20 min/task single-core.
# Stage 2: assemble + score both batteries (amended binomial + KS criteria).
# Stage 3: disBatch runs the 8 data-fit chains ({kszonly, joint} x 4 seeds,
#          DE moves, 128 x 30k — the decision-2 convergence convention);
#          every fit task re-checks its battery gate itself, so a FAILED
#          battery stops the fits, not the job.
# Stage 4: assemble both fits (cross-chain R-hat, MAP chi2 per block,
#          boundary diagnostic, reduced summaries).
#
# disBatch is the right parallelism here (not OpenMPI): each injection /
# seed is an independent single-process emcee run with a vectorized
# likelihood — there is nothing to message-pass.
#
# ⛔ Submit:  sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_a5_subsets_disbatch.sh

set -euo pipefail

module load gcc disBatch
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2-paper3a
mkdir -p /mnt/home/mlee1/ceph/logs/a5_subsets

echo "=== stage 1: recovery batteries (32 tasks) ==="
disBatch -p /mnt/home/mlee1/ceph/logs/a5_subsets/db_battery_ \
    analysis/paper3a/scripts/a5_subsets_battery.disbatch

echo "=== stage 2: assemble batteries ==="
python -u analysis/paper3a/scripts/run_a5_recovery_subsets.py --which kszonly --assemble
python -u analysis/paper3a/scripts/run_a5_recovery_subsets.py --which joint --assemble

echo "=== stage 3: subset fits (8 chains) ==="
disBatch -p /mnt/home/mlee1/ceph/logs/a5_subsets/db_fits_ \
    analysis/paper3a/scripts/a5_subsets_fits.disbatch

echo "=== stage 4: assemble fits ==="
python -u analysis/paper3a/scripts/run_a5_subsets.py --which kszonly --assemble
python -u analysis/paper3a/scripts/run_a5_subsets.py --which joint --assemble

echo "DONE"
