#!/bin/bash
#SBATCH --job-name=wp4_fit_ext
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp4_fit_ext_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp4_fit_ext_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A4 512-run refit (v4): dataset rebuild + fit + k-fold ────────────────
# Stage 3 of the densification chain: the operator_tables_v3 glob now holds
# 512 sb35 runs (256 original + 256 extension); the dataset builder picks
# them all up automatically. Outputs versioned _v4 (the 512-run generation);
# the v3 artifacts stay untouched for provenance. Next session validates
# (wp4_validate_emulator.py --suffix _v4), checks the group-bin error
# against the predicted ~2.5%, and flips the default artifact if it holds.
# n=512 GPs: fit ~4x the v3 cost -> 8 h walltime is generous.

set -euo pipefail

module load gcc
source /mnt/home/mlee1/venvs/torch3/bin/activate
export OMP_NUM_THREADS=16
cd /mnt/home/mlee1/vdm_bind2-paper3a

WP4=/mnt/ceph/users/mlee1/paper3/A/wp4_emulator

python -c "
from pathlib import Path
from analysis.paper3a.emulator import dataset as ds
ds.build(tables_dir=Path('/mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables_v3'),
         out_path=Path('$WP4/gasemu_dataset_v4.npz'))
"

python -m analysis.paper3a.emulator.fit \
    --dataset "$WP4/gasemu_dataset_v4.npz" \
    --artifact-out "$WP4/gasemu_gp_v4.npz" \
    --outdesign-out "$WP4/gasemu_outdesign_v4.npz" \
    --n-pca 24

python -m analysis.paper3a.emulator.fit \
    --dataset "$WP4/gasemu_dataset_v4.npz" \
    --kfold 8 \
    --validation-out "$WP4/gasemu_kfold_v4.npz" \
    --n-pca 24
