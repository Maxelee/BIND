#!/bin/bash
#SBATCH --job-name=paired550
#SBATCH --output=/mnt/home/mlee1/ceph/logs/paired550_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/paired550_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# ── Regenerate paired_perreal_fid.npz from the 550-realization fiducial ───────
# The 50-real vintage cache is what still pins fig 4/4b's paired bands at
# ±1sigma/sqrt(50); the builder derives N from the arrays, so this file landing
# at 550 upgrades the bands automatically at the next figure run.
# Serial by design (bind.cli.paired_stats predates the 550 era and has no
# chunk/merge mode; ~2-3 h here is fine inside an allocation). The rm handles
# the CLI's existence-only cache check.
#
#   sbatch papers/01_pipeline/jobs/run_paired_stats.sbatch

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

rm -f /mnt/home/mlee1/ceph/bind_lightcone_tng/paired_perreal_fid.npz
python -u -m bind.cli.paired_stats \
    --run_dir /mnt/home/mlee1/ceph/bind_lightcone_tng \
    --fid_dir /mnt/home/mlee1/ceph/bind_lightcone_tng
echo "[done] $(ls -la /mnt/home/mlee1/ceph/bind_lightcone_tng/paired_perreal_fid.npz)"
