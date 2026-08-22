#!/bin/bash
#SBATCH --job-name=noisy_asm
#SBATCH --output=/mnt/home/mlee1/ceph/logs/noisy_asm_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/noisy_asm_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ── Assemble + verify the noisy-statistics dataset after the shard job ───────
# Chain behind run_noisy_cache.sbatch:
#   sbatch --dependency=afterok:<shard_jobid> papers/01_pipeline/jobs/run_noisy_assemble.sbatch

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND/papers/01_pipeline

python build_noisy_cache.py --assemble
python build_noisy_cache.py --verify
