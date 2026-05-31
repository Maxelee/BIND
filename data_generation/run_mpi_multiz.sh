#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=rome
#SBATCH -J make_multiz
#SBATCH -N 8
#SBATCH -n 128
#SBATCH --exclusive
#SBATCH -o OUTPUT_MULTIZ.o%j
#SBATCH -e OUTPUT_MULTIZ.e%j
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -t 6-23:15:00

# Multi-redshift training data: mass + gas-thermo maps in one pass, 1 rotation
# per halo, for 8 snapshots (z = 0 .. 4). Output:
#   train_data_multiz_128_cpu/{train,test}/sim_<i>/snap_<NNN>/sim_<i>_halo_<k>_rot_0.npz
# Work is distributed over (sim, snapshot) pairs; resumable (skips complete halos).
# Smoke test (one rank, two sims, two snapshots):
#   srun -n 1 python3 -u process_simulations_multiz.py \
#       --start_sim 0 --end_sim 2 --snapshots 90,24

module load openmpi
module load python
module load python-mpi
module load hdf5

source /mnt/home/mlee1/venvs/gen_train_data/bin/activate

srun -n 128 python3 -u process_simulations_multiz.py --start_sim 0 --end_sim 1024
