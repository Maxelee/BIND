#!/bin/bash
#SBATCH --job-name=bind_tng_project
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_tng_project_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_tng_project_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

source ~/venvs/torch3/bin/activate

srun -n1 python3 -u ./tools/stack_gas_column.py
