#!/bin/bash
#SBATCH --job-name=r8_finalize
#SBATCH --output=/mnt/home/mlee1/ceph/logs/r8_finalize_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/r8_finalize_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# R8 finalize: sampling completed in job 2454139 (all legs at 739-1200 tau,
# 15-24x past the 50-tau gate); this post-processes the 4 chains and writes
# figures/npz/verdict. Each leg's processing (~20 GB ceph chain read +
# autocorr FFTs) takes ~30 min and needs node memory (workstation cgroup
# OOM'd at 10GB; sequential 1h job 2454177 timed out at 2/4 legs) -- so the
# 4 legs run as PARALLEL srun steps into per-leg caches
# (r8_state/r8_processed_<leg>.npz), then finalize assembles from the caches.
# Resumable: completed leg caches are reused on rerun (keyed on step count).
# Per-step --mem is required (without it the first step claims the whole
# job allocation and serializes the rest -- hit in job 2454136).
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
E=/mnt/home/mlee1/BIND-ksz2/examples/_r8_gp_mcmc.py
for leg in ksz tsz joint joint_noA2h; do
  srun --exact -n1 -c2 --mem=38G python $E --stage process --leg $leg &
done
wait
# finalize reuses the 4 caches (fast); any leg whose process step failed is
# redone sequentially here (cache miss fallback), covered by the 3h wall.
python $E --stage finalize
echo "R8 FINALIZE DONE"
