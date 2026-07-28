#!/bin/bash
#SBATCH --job-name=r8_finalize
#SBATCH --output=/mnt/home/mlee1/ceph/logs/r8_finalize_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/r8_finalize_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# R8 finalize only: sampling completed in job 2454139 (all legs at 739-1200
# tau, 15-24x past the 50-tau gate) but the 8h wall cut the finalize stage;
# emcee HDFBackend reads the full ~20GB chain per leg before thinning, so
# this needs node memory (the workstation cgroup OOM'd at 10GB).
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
python /mnt/home/mlee1/BIND-ksz2/examples/_r8_gp_mcmc.py --stage finalize
echo "R8 FINALIZE DONE"
