#!/bin/bash
#SBATCH --job-name=bind_tng_project
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_tng_project_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_tng_project_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=rome
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── BIND paint, STAGE 1 (CPU / MPI): project + halos + cutouts ─────────────────
# The memory-heavy half of painting the full IllustrisTNG_DM box, split out so it
# runs across many CPU nodes via MPI instead of OOMing one GPU node.  Each rank
# reads only its subset of the 75 snapshot chunks, streams them one at a time, and
# accumulates into the small z-slab maps (~70 MB/slab); the partial maps are
# MPI.SUM-reduced onto rank 0, which extracts per-halo DMO cutouts and writes the
# stage-1 intermediate.  Stage 2 (run_paint_tng_generate.sh) then runs the model
# on one GPU.  Submit gated:
#
#   jid=$(sbatch --parse run_paint_tng_project.sh)
#   sbatch --dependency=afterok:$jid run_paint_tng_generate.sh
#
# Env overrides: SIM_ROOT, SNAPSHOT, STAGE1_DIR, HALO_MASS_MIN.
# Any extra args pass straight through to bind.cli.paint_project.
#
# BIND_env is a Python-3.11 venv (--system-site-packages) built from the module
# python view, so mpi4py comes straight from the `python-mpi` module (no build)
# and matches the view's numpy that bind/Pylians/torch are built against.  Load
# the SAME modules at runtime as at creation so PYTHONPATH exposes mpi4py.
# Fallback if `import mpi4py` fails: `module load openmpi && pip install mpi4py`
# into BIND_env.

set -euo pipefail

module load openmpi/4.1.8   # mpi4py now lives in the venv (py3.11); python-mpi's py3.10 h5py would clash
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Configuration ─────────────────────────────────────────────────────────────
SIM_ROOT=${SIM_ROOT:-/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output}
SNAPSHOT=${SNAPSHOT:-99}
HALO_MASS_MIN=${HALO_MASS_MIN:-1e13}

SNAP3=$(printf '%03d' "$SNAPSHOT")
STAGE1_DIR=${STAGE1_DIR:-/mnt/home/mlee1/ceph/bind_illustristng/snap_${SNAP3}/stage1}

SNAPDIR="$SIM_ROOT/snapdir_${SNAP3}"
GROUPDIR="$SIM_ROOT/groups_${SNAP3}"

if [[ ! -d "$SNAPDIR" ]]; then
    echo "ERROR: snapshot directory not found: $SNAPDIR" >&2
    exit 1
fi
if [[ ! -d "$GROUPDIR" ]]; then
    echo "ERROR: group catalog directory not found: $GROUPDIR" >&2
    exit 1
fi

mkdir -p "$STAGE1_DIR"

# ── 35-dim fiducial IllustrisTNG parameter vector ─────────────────────────────
PARAMS_FILE="$STAGE1_DIR/fiducial_params.npy"
python - "$PARAMS_FILE" <<'PY'
import sys
import numpy as np
import bind

np.save(sys.argv[1], bind.fiducial_params().astype(np.float64))
print(f"[params] wrote fiducial 35-dim vector -> {sys.argv[1]}")
PY

# ── Stage 1: MPI projection + cutouts ─────────────────────────────────────────
echo "=== BIND paint stage 1 (project): snapshot ${SNAP3} ==="
echo "    snapshot=$SNAPDIR"
echo "    groups=$GROUPDIR"
echo "    halo_mass_min=$HALO_MASS_MIN  ntasks=${SLURM_NTASKS:-?}"
echo "    stage1_out=$STAGE1_DIR"

srun python -u -m bind.cli.paint_project \
    --snapshot "$SNAPDIR" \
    --group_catalog "$GROUPDIR" \
    --snapshot_index "$SNAPSHOT" \
    --params "$PARAMS_FILE" \
    --output_dir "$STAGE1_DIR" \
    --halo_mass_min "$HALO_MASS_MIN" \
    "$@"

echo "=== Stage 1 done. Intermediate in $STAGE1_DIR ==="
echo "    Next: STAGE1_DIR=$STAGE1_DIR sbatch run_paint_tng_generate.sh"
