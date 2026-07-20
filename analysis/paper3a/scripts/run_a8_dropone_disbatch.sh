#!/bin/bash
#SBATCH --job-name=a8_dropone
#SBATCH --output=/mnt/home/mlee1/ceph/logs/a8_dropone_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/a8_dropone_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A8: the plan-v7 DROP-ONE pair (TK register items #15 and #16) ────────
#
# WHAT THIS IS NOT. It is not a re-run of job 6636045. That job ran the three
# error-WIDENING variants (emul2x, fgas_model2x, b_coordsys2x, 4 seeds each,
# COMPLETED 2026-07-19T22:11) and its chains are on disk and already scored
# into a8_sysgrid.json. Those twelve are NOT repeated here. `a8_variants.
# disbatch` grew to 16 lines when no_ksz was added after 6636045 launched, so
# resubmitting that file would redo all twelve for nothing; this file carries
# only the eight tasks that have never run.
#
#   no_ksz    the kSZ block dropped  -> f_gas + B alone   (TK #15)
#   no_fgas   the fgas block dropped -> kSZ  + B alone    (TK #16)
#
# WHY FRESH CHAINS. Dropping a likelihood term BROADENS the posterior, which
# is the same direction that collapsed importance sampling for the widening
# variants (ESS 24/130/141 of 8000): the target has mass where the fiducial
# proposal has none. Reweighting cannot reach either row. plan v7's other two
# rows (drop-B, and kSZ alone) already exist as frozen subset chains in
# ab_synthesis.json -- these two complete the drop-one set, so the paper can
# say which probe carries the exclusion rather than asserting it.
#
# no_ksz exists because Siegel and Bigwood both omit M500 >~ 10^13.3 from
# their primary analyses on a spec-vs-photo amplitude discrepancy, while our
# kSZ leg uses the spectroscopic m3 at 13.41 -- inside that boundary. Whether
# the exclusion survives without the contested bin is a question the referee
# will ask.
#
# Both variants were verified in-process before submission: each zeroes its
# own block's loglike EXACTLY and leaves the other two bit-identical, so the
# chain geometry (32 columns, both nuisances) is unchanged and the drop is
# the only difference.
#
# 8 tasks fanned by disBatch inside one allocation, each single-core (emcee
# here is vectorized single-process -- never MPI). Same recipe as the
# fiducial fit: 128 walkers x 30k, DE move mixture, and every task re-checks
# the a5_recovery_jointab gate itself, so a failed battery stops the fits
# rather than the job. Reference: the 12-task run took 1h51m wall.
#
# AFTER IT LANDS: re-run `python analysis/paper3a/scripts/run_a8_sysgrid.py`
# (the driver does it below). The fresh-chain loop now picks up no_ksz and
# no_fgas alongside the three widening variants and writes them as
# drop-one rows, replacing the "prepared on request, not run" placeholder in
# drop_probes_note.
#
# ⛔ Submit:  sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_a8_dropone_disbatch.sh

set -euo pipefail

module load gcc disBatch
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2-paper3a
mkdir -p /mnt/home/mlee1/ceph/logs/joint_ab

echo "=== WP-A8 drop-one pair: 8 chains (no_ksz, no_fgas x 4 seeds) ==="
disBatch -p /mnt/home/mlee1/ceph/logs/joint_ab/db_a8_dropone_ \
    analysis/paper3a/scripts/a8_droponebatch.disbatch

echo "=== chains written; re-scoring the systematics grid ==="
python -u analysis/paper3a/scripts/run_a8_sysgrid.py

echo "=== per-variant chi2 at each variant's own MAP ==="
python -u analysis/paper3a/scripts/run_a8_variant_chi2.py

echo "DONE"
