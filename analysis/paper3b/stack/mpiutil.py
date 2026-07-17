"""MPI detection for the WP-B2 stacking driver (same contract as the A-side).

Mirrors `analysis/paper3a/observables/truth_projection.mpi_comm` (not imported
from there — that module drags the bind/torch chain): env-based detection so
plain runs never import mpi4py, `BIND_NO_MPI=1` opt-out, and a hard failure
when a multi-task launch wires up as singletons (which would race N copies of
the full measurement onto the same output files).
"""

from __future__ import annotations

import os


def mpi_comm():
    """COMM_WORLD when running as a multi-rank MPI job, else None."""
    if os.environ.get("BIND_NO_MPI"):
        return None
    n_env = (os.environ.get("OMPI_COMM_WORLD_SIZE")
             or os.environ.get("PMI_SIZE")
             or os.environ.get("SLURM_STEP_NUM_TASKS"))
    if n_env is None or int(n_env) < 2:
        return None
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_size() < int(n_env):
        raise RuntimeError(
            f"launched with {n_env} tasks but MPI world size is {comm.Get_size()} — "
            "MPI did not wire up; launch with mpirun (or srun --mpi=pmix)")
    return comm
