#!/bin/bash
#SBATCH --job-name=partial_supp
#SBATCH --output=/mnt/home/mlee1/ceph/logs/partial_supp_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/partial_supp_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=gen            # CPU-only job; set to your CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-15               # set to 0-(N_CHUNKS-1)
#
# Field-level halo-MASKING decomposition of the matter-power suppression across
# the 256-point Sobol feedback cube.  For each design we re-composite halo
# subsets (mass decade x gas-fraction tercile) back into the DMO box and measure
# the partial S(k).  CPU-only: reuses the precomputed BIND patches in
# sobol_ss_cv/maps/, so NO GPU and NO re-emulation.  ~16 designs/chunk, a few min.
#
# Submit (you run this — org policy: Claude does not submit Slurm):
#   N_CHUNKS=16 sbatch --array=0-15 run_partial_supp.sh
# Then, once every chunk has finished, merge the shards into one cache:
#   source /mnt/home/mlee1/venvs/torch3/bin/activate
#   export LD_LIBRARY_PATH=/mnt/sw/nix/store/pxl9ndpq5yassys69ibkr3kcz34jh385-gcc-13.3.0/lib64:$LD_LIBRARY_PATH
#   python tools/partial_supp_sobol.py --reduce
# Figure:
#   python tools/fig_partial_supp.py --cache partial_supp_sobol.npz

set -e
source /mnt/home/mlee1/venvs/torch3/bin/activate
# torch3 venv needs a gcc-13 libstdc++ (GLIBCXX_3.4.29) on the library path:
export LD_LIBRARY_PATH=/mnt/sw/nix/store/pxl9ndpq5yassys69ibkr3kcz34jh385-gcc-13.3.0/lib64:$LD_LIBRARY_PATH
cd /mnt/home/mlee1/vdm_bind2

N_CHUNKS=${N_CHUNKS:-16}            # must match --array upper bound + 1
python tools/partial_supp_sobol.py --designs all \
    --n_chunks "${N_CHUNKS}" --chunk_id "${SLURM_ARRAY_TASK_ID}"
