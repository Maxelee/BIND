#!/bin/bash
#SBATCH --job-name=a8_variants
#SBATCH --output=/mnt/home/mlee1/ceph/logs/a8_variants_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/a8_variants_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A8 stream 1: the three systematics variants that need FRESH CHAINS ───
#
# WHY THIS JOB EXISTS. The cheap route was tried first and failed honestly.
# `run_a8_sysgrid.py` importance-reweights the frozen joint chain, which works
# for variations that shift or mildly widen a tier (central_only ESS 3807,
# rsat_wide ESS 1620 — both clean, coordinates move <= 0.30 sigma). It
# COLLAPSES for the three variations that substantially WIDEN a covariance
# tier: ESS 24 / 130 / 141 out of 8000. That is structural, not a bug — the
# widened posterior is broader than the fiducial proposal being reweighted
# from, so the target has mass where the proposal has none. The pre-registered
# ESS >= 500 gate caught it; the +9 sigma movements those rows report are IS
# artifacts and are NOT quotable. Only fresh chains can assess them.
#
# 12 tasks = 3 variants x 4 seeds, fanned by disBatch inside this one
# allocation (each task single-core; emcee here is vectorized single-process,
# never MPI). Same recipe as the fiducial joint fit: 128 walkers x 30k, DE
# move mixture, and every task re-checks the a5_recovery_jointab gate itself,
# so a failed battery stops the fits rather than the job.
#
#   emul2x        Sigma_theory doubled in ALL THREE blocks (kSZ emul_frac,
#                 fgas emul_frac, and the B block's propagated GP errors)
#   fgas_model2x  fgas PAINT_BIAS_RESID + C2S_TRANSFER_SYS doubled
#   b_coordsys2x  B's SLOPE_SYS + OFFSET_SYS coordinate-systematic tier doubled
#
# AFTER IT LANDS: re-run `python analysis/paper3a/scripts/run_a8_sysgrid.py`.
# It picks the chains up and replaces the three UNRELIABLE rows with
# directly-measured movements. Expect the true movements to be MUCH smaller
# than the IS artifacts — widening errors normally broadens a posterior
# without moving its median — but that is the thing to be demonstrated, not
# assumed.
#
# ⛔ Submit:  sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_a8_variants_disbatch.sh

set -euo pipefail

module load gcc disBatch
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2-paper3a
mkdir -p /mnt/home/mlee1/ceph/logs/joint_ab

echo "=== WP-A8 systematics variants: 12 chains (3 variants x 4 seeds) ==="
disBatch -p /mnt/home/mlee1/ceph/logs/joint_ab/db_a8_ \
    analysis/paper3a/scripts/a8_variants.disbatch

echo "=== chains written; re-scoring the systematics grid ==="
python -u analysis/paper3a/scripts/run_a8_sysgrid.py

echo "DONE"
