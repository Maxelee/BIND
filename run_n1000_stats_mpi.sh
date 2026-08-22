#!/bin/bash
#SBATCH --job-name=bind_n1000_stmpi
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_n1000_stmpi_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_n1000_stmpi_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --array=0-318
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=FAIL

# ── N1000 stats, MPI over realizations (one array task = one run) ─────────────
# The realization loop is embarrassingly parallel — every statistic is a mean
# over realizations — so 64 ranks on one node turn a ~3 h serial run into a few
# minutes.  Ranks 0/1/2 decompress kappa/y/tau into node-local scratch (the npz
# members are DEFLATE, so there is no random access into them), then every rank
# memmaps the uncompressed cubes and processes reals[rank::64], accumulating
# sum and sum-of-squares; rank 0 reduces and writes the same npz products the
# serial CLI writes.  peak_cross (R(nu)) is NOT produced — unused downstream.
#
# FIRST RUN / DEBUGGING: cap the realizations to make a fast smoke test, e.g.
#   N_REAL=32 sbatch --array=2 run_n1000_stats_mpi.sh     # truth, ~1 min
# then compare against the serial product before trusting a full sweep.
#
# Env: OUT_ROOT, NU_SIGMA0, SMOOTHING, N_REAL (cap), NGAL, SIGMA_E, FORCE,
#      SCRATCH (node-local staging dir, default /tmp).

set -euo pipefail

# mpi4py lives in the venv (py3.11); do NOT module-load python-mpi (h5py clash)
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

# one thread per rank: 64 ranks x multithreaded BLAS/Pylians oversubscribes the node
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BIND_PK_THREADS=${BIND_PK_THREADS:-1}

OUT_ROOT=${OUT_ROOT:-/mnt/home/mlee1/ceph/bind_n1000}
FORCE=${FORCE:-0}
NU_SIGMA0=${NU_SIGMA0:-/mnt/home/mlee1/ceph/bind_n1000/analysis/nu_sigma0_bind.npz}
SMOOTHING=${SMOOTHING:-2.0}
NGAL=${NGAL:-}
SIGMA_E=${SIGMA_E:-0.26}
N_REAL=${N_REAL:-}
SCRATCH=${SCRATCH:-/tmp/n1000_stats_$SLURM_JOB_ID}
: "${SLURM_ARRAY_TASK_ID:?run_n1000_stats_mpi.sh must run as a SLURM array task}"
IDX="$SLURM_ARRAY_TASK_ID"

if   (( IDX == 0 )); then OUT_RUN="$OUT_ROOT/bind/run_0000"
elif (( IDX == 1 )); then OUT_RUN="$OUT_ROOT/dmo/run_0000"
elif (( IDX == 2 )); then OUT_RUN="$OUT_ROOT/truth/run_0000"
elif (( IDX <= 62 )); then OUT_RUN="$OUT_ROOT/twobound/$(printf 'run_%04d' $((IDX - 3)))"
elif (( IDX <= 318 )); then OUT_RUN="$OUT_ROOT/sb35/$(printf 'run_%04d' $((IDX - 63)))"
else echo "ERROR: task index $IDX out of range" >&2; exit 1
fi
# RUN_DIR_OVERRIDE lets a smoke test write into a scratch run dir instead of a
# real one -- NEVER point a capped (N_REAL=...) run at a real run dir, it would
# overwrite good 1000-realization products with a short-run version.
OUT_RUN=${RUN_DIR_OVERRIDE:-$OUT_RUN}

if [[ ! -f "$OUT_RUN/.trace_complete" ]]; then
  echo "=== $OUT_RUN not traced yet — skipping (resubmit this index later) ==="; exit 0
fi

# same convention-aware guard as the serial scripts: recompute when a product is
# missing, truncated, on the old nu axis, or built against a different fiducial
stats_ok() {
  python - "$OUT_RUN" "$NU_SIGMA0" <<'PY' 2>/dev/null
import sys, numpy as np
from pathlib import Path
from bind.inference.stats import NU_CANON
run, table = Path(sys.argv[1]), sys.argv[2]
want_fid = ""
want_sig0u = None
if table:
    with np.load(table) as t:
        want_fid = str(t["source_run"])
        # the VALUES too: regenerating the table in place keeps source_run
        # identical while moving the sigma0 that nu was normalised by
        want_sig0u = np.asarray(t["sigma_unsmoothed"], float).ravel()
for f in ("Cl_kappa.npz", "peak_counts.npz", "nongaussian_stats.npz"):
    p = run / f
    if not p.exists():
        sys.exit(1)
    try:
        with np.load(p) as d:
            for k in d.files:
                _ = d[k].shape
            if f != "Cl_kappa.npz":
                if str(d["nu_grid"]) != "canon":
                    sys.exit(1)
                if want_fid and str(d["nu_fiducial"]) != want_fid:
                    sys.exit(1)
                axes = ([d["nu"]] if f.startswith("peak") else [d["pdf_bins"], d["mf_nu"]])
                for a in axes:
                    if a.shape != NU_CANON.shape or not np.allclose(a, NU_CANON):
                        sys.exit(1)
                if want_sig0u is not None:
                    got = np.asarray(d["nu_sigma0_unsmoothed_table"], float).ravel()
                    if got.size and (got.shape != want_sig0u.shape
                                     or not np.allclose(got, want_sig0u, rtol=1e-9)):
                        sys.exit(1)
    except Exception:
        sys.exit(1)
sys.exit(0)
PY
}
if [[ "$FORCE" != "1" ]] && stats_ok; then
  echo "=== $OUT_RUN already has valid stats — skipping (FORCE=1 to redo) ==="; exit 0
fi

OPT=(--run_dir "$OUT_RUN" --smoothing_arcmin $SMOOTHING --nu_grid canon --scratch "$SCRATCH")
if [[ -n "$NU_SIGMA0" ]]; then
  [[ -f "$NU_SIGMA0" ]] || { echo "ERROR: NU_SIGMA0 table $NU_SIGMA0 not found" >&2; exit 1; }
  OPT+=(--nu_sigma0_from "$NU_SIGMA0" --nu_norm fixed --out_suffix "")
fi
[[ -n "$NGAL"   ]] && OPT+=(--shape_noise_ngal "$NGAL" --sigma_e "$SIGMA_E")
[[ -n "$N_REAL" ]] && OPT+=(--n_real "$N_REAL")

echo "=== n1000 stats (MPI) | task $IDX -> $OUT_RUN | ranks=${SLURM_NTASKS} | $(date) ==="
srun -n "${SLURM_NTASKS}" python -u -m bind.cli.lightcone_stats_mpi "${OPT[@]}"
rm -rf "$SCRATCH"
echo "=== stats done for $OUT_RUN | $(date) ==="
