#!/bin/bash
#SBATCH --job-name=a5_fit
#SBATCH --output=/mnt/home/mlee1/ceph/logs/a5_fit_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/a5_fit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A5 unattended chain: recovery battery (if not yet PASSed) → data fit ──
# Skips the battery when wp5_chains/a5_recovery.json already exists (e.g. the
# in-session background run finished); the fit refuses to run unless the
# battery PASSed — the pre-registration is enforced in code, not by memory.
#
# ⛔ HUMAN CHECKPOINT stays intact: the fit ARCHIVES the posterior +
# diagnostics; nothing feeds A6 until Max reviews a5_fit_summary.json.
#
# Submit:  sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_a5_battery_fit.sh

set -euo pipefail

module load gcc
source /mnt/home/mlee1/venvs/torch3/bin/activate
export OMP_NUM_THREADS=16
cd /mnt/home/mlee1/vdm_bind2-paper3a

CHAINS=/mnt/ceph/users/mlee1/paper3/A/wp5_chains

if [[ ! -f "$CHAINS/a5_recovery.json" ]]; then
  python -u analysis/paper3a/scripts/run_a5_recovery.py
else
  echo "recovery battery output exists — skipping to the fit gate"
fi

python -u analysis/paper3a/scripts/run_a5_fit.py
