#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=rome
#SBATCH -J add_thermo_maps
#SBATCH -N 8
#SBATCH -n 128
#SBATCH --exclusive
#SBATCH -o OUTPUT_THERMO.o%j
#SBATCH -e OUTPUT_THERMO.e%j
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -t 6-00:00:00

module load openmpi
module load python
module load python-mpi
module load hdf5

source /mnt/home/mlee1/venvs/gen_train_data/bin/activate

# Full run over all 1024 sims.
# For a smoke test, replace the last line with:
#   srun -n 1 python3 -u add_gas_thermo_maps.py --only_sims 0,1
srun -n 128 python3 -u add_gas_thermo_maps.py
