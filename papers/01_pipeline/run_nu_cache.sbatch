#!/bin/bash
#SBATCH --job-name=nu_cache
#SBATCH --output=/mnt/home/mlee1/ceph/logs/nu_cache_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/nu_cache_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=600G
#SBATCH --time=02:00:00

# ── One common nu axis for figs 4, 5, 7 and 8 (canonical 0.5-wide-bin grid) ──────────
# Until now those figures plotted against nu on THREE different grids: Minkowski
# functionals on linspace(-3,4,29), peaks/minima on linspace(-4.875,11.875,68), and the
# PDF on a +-6-sigma kappa grid with 41 bins. An earlier pass unified them onto
# linspace(-3,8,45) (nu8_shards/); the author has since ruled the canonical grid to be
# 0.5-wide bins from -3 to 8 (NU_EDGES: 23 edges; NU: 22 centres, -2.75..7.75), so that
# the counting statistics (peaks/minima/PDF) hold an integer count per bin with a
# well-defined Poisson error. This recomputes all six threshold statistics on that grid
# for every run, so the released dataset and every figure share one axis (see nu_grid.py
# for the edges-vs-centres distinction and the caveat that nu is still a per-statistic
# S/N variable, because the three families use different smoothing scales).
#
#   sbatch papers/01_pipeline/run_nu_cache.sbatch
#
# SIZING (measured, not guessed): one run = 50 realizations x 5 source planes = 159 s on
# 1 core, so 255 runs is 11.2 h serial. cca nodes are 48+ cores / 768 GB, and each task
# holds one ~1 GB kappa cube, so 48 concurrent tasks need ~48 GB and finish in ~6 waves,
# i.e. ~17 min. OMP_NUM_THREADS=1 stops 48 numpy processes from oversubscribing the node.
# (Bin count does not materially change runtime -- it is dominated by smoothing/extrema
# finding on the kappa maps, not by the histogram width.)
#
# Nothing released is overwritten: shards land in bind_sb35/nu05_shards/ and --assemble
# writes a NEW emulator_dataset_nu05.npz, copying every non-nu array unchanged. The
# earlier nu8_shards/ and emulator_dataset_nu8.npz are left in place, untouched.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND/papers/01_pipeline
mkdir -p /mnt/home/mlee1/ceph/logs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module -q load modules/2.4-20250724
module -q load disBatch

TASKS=$(mktemp /tmp/nu_cache_tasks.XXXXXX)
trap 'rm -f "$TASKS"' EXIT

# the fiducial pair first (fig 4 needs these, and they carry per-realization draws)
for s in bind truth; do
    echo "cd /mnt/home/mlee1/BIND/papers/01_pipeline && python build_nu_cache.py --run $s --sci" >> "$TASKS"
done
# every Sobol run in the released dataset (all 256; each --run call is idempotent and
# skips shards already built, e.g. by an earlier local/partial pass)
python - <<'PY' >> "$TASKS"
import numpy as np
d = np.load('/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.npz', allow_pickle=True)
for r in d['run_ids']:
    print(f"cd /mnt/home/mlee1/BIND/papers/01_pipeline && "
          f"python build_nu_cache.py --run run_{int(r):04d}")
PY

echo "[disBatch] $(wc -l < "$TASKS") tasks over ${SLURM_NTASKS} slots"
disBatch -e "$TASKS"          # -e: exit non-zero if any task failed

python build_nu_cache.py --assemble
python build_nu_cache.py --verify
echo "[done] /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_nu05.npz"
