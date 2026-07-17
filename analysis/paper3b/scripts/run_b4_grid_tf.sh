#!/bin/bash
#SBATCH --job-name=b4_grid_tf
#SBATCH --output=/mnt/home/mlee1/ceph/logs/b4_grid_tf_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/b4_grid_tf_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --time=03:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-B4/B5: the TRANSFER-MATCHED model grid (the B5 comparison chain) ──────
# Identical to run_b4_grid.sh except the noisy mock kappa is filtered with the
# empirical Wiener reconstruction transfer T(ell) before smoothing (B5 design
# decision, DESIGN_DECISION_FILTER.md; transfer_desy3.npz derived 2026-07-17
# from the data map's pseudo-C_ell over the common footprint / bind-fiducial
# mock spectrum — kappa 2-pt only, no y, no stacks). Outputs ->
# ~/ceph/paper3/B/wp4_mocks/grid_tfwiener/. Per-unit npz now also carries the
# per-peak (nu, CAP-Y) tables (~40 MB/unit) so any re-binning is post-hoc.
# Resumable: finished units are skipped on relaunch.
#
# PREREQ (already run 2026-07-17): python -m analysis.paper3b.scripts.build_b5_transfer
#
# ⛔ HUMAN CHECKPOINT — submit by hand:
#   sbatch /mnt/home/mlee1/BIND-paper3b/analysis/paper3b/scripts/run_b4_grid_tf.sh
#
# 2026-07-17 (post first tf run, PRE any central-value look): the comparison
# grid is re-run at n_seeds=32 — the 8-seed tf run measured worst
# MC/sigma_stat = 1.42 in the nu 4-12 bin (median 272 peaks/unit) and its
# ~9%/cell MC noise made the fixed R2 tolerance trip statistically (max
# 1.74 sigma over 25 cells — no significant deviation). x4 patches at ~27 ->
# ~100 min/unit still fits 03:00. The 8-seed run's record is kept in
# MODEL_FREEZE.md as interim.
#   sbatch ... run_b4_grid_tf.sh --n-seeds 32 --overwrite
# AFTER that grid: variants below (bind fiducial only, cheap — quantify the
# plane-weight systematic and the GLIMPSE-transfer scale ON THE COMPARISON
# CHAIN), then finalize MODEL_FREEZE.md:
#   sbatch ... run_b4_grid_tf.sh --n-seeds 32 --categories bind --weight-scheme variant8
#   sbatch ... run_b4_grid_tf.sh --n-seeds 32 --categories bind --weight-scheme plain
#   sbatch ... run_b4_grid_tf.sh --n-seeds 32 --categories bind truth --transfer glimpse
# (the last overrides --transfer; GLIMPSE is a qualitative cross-check only —
# a linear T cannot mimic the sparsity prior, validated in
# b5_transfer_summary.json: tfglimpse mock 0.022 vs data 0.061 nu>=4/deg2.)

set -euo pipefail

# mpi4py from the python-mpi module (venv re-prepended so the module's h5py
# cannot shadow the venv's — same pattern as run_wp2_truth_validation.sh).
module load python openmpi python-mpi

VENV=${VENV:-/mnt/home/mlee1/venvs/paper3b_popeye}
source "$VENV/bin/activate"
export PYTHONPATH="$VENV/lib/python3.11/site-packages:${PYTHONPATH:-}"

mkdir -p /mnt/home/mlee1/ceph/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd /mnt/home/mlee1/BIND-paper3b
mpirun -np "$SLURM_NTASKS" python -u -m analysis.paper3b.scripts.run_b4_grid \
    --transfer wiener "$@"

echo "=== B4 transfer-matched grid done ==="
