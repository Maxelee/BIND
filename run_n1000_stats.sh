#!/bin/bash
#SBATCH --job-name=bind_n1000_stats
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_n1000_stats_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_n1000_stats_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --array=0-318
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=FAIL

# ── N1000 campaign: per-run summary statistics over the 1000-real map cubes ───
# Run AFTER the trace lanes: each task computes Cl_kappa / Cl_kappa_y / Cl_tau,
# peak counts + minima, nongaussian (PDF/Minkowski), dm_stats for one run in the
# N1000 tree.  Same manifest indexing as n1000_body.sh.  Tasks whose run has no
# .trace_complete yet exit 0 immediately — resubmit the array (or the missing
# indices) as later runs land.  halo_scaling is skipped: per-halo products are
# n_real-independent and live in the 50-real tree.
#
# Env overrides: OUT_ROOT, SMOOTHING ("1 2 5 8"), NGAL, SIGMA_E, STATS_EXTRA, FORCE.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

OUT_ROOT=${OUT_ROOT:-/mnt/home/mlee1/ceph/bind_n1000}
FORCE=${FORCE:-0}
# Fiducial kappa_rms table defining nu = kappa/sigma_fid for EVERY run
# (bind.cli.nu_sigma0).  Empty => per-map sigma (NOT a common axis).
NU_SIGMA0=${NU_SIGMA0:-/mnt/home/mlee1/ceph/bind_n1000/analysis/nu_sigma0_bind.npz}
: "${SLURM_ARRAY_TASK_ID:?run_n1000_stats.sh must run as a SLURM array task}"
IDX="$SLURM_ARRAY_TASK_ID"

if   (( IDX == 0 )); then OUT_RUN="$OUT_ROOT/bind/run_0000"
elif (( IDX == 1 )); then OUT_RUN="$OUT_ROOT/dmo/run_0000"
elif (( IDX == 2 )); then OUT_RUN="$OUT_ROOT/truth/run_0000"
elif (( IDX <= 62 )); then OUT_RUN="$OUT_ROOT/twobound/$(printf 'run_%04d' $((IDX - 3)))"
elif (( IDX <= 318 )); then OUT_RUN="$OUT_ROOT/sb35/$(printf 'run_%04d' $((IDX - 63)))"
else echo "ERROR: task index $IDX out of range" >&2; exit 1
fi

if [[ ! -f "$OUT_RUN/.trace_complete" ]]; then
  echo "=== $OUT_RUN not traced yet — skipping (resubmit this index later) ==="; exit 0
fi
# Skip only on a product that actually LOADS: a job killed mid-savez (preemption,
# timeout) leaves a truncated npz, and an existence-only check would treat that
# corrupt file as finished forever.
stats_ok() {
  python - "$OUT_RUN" "$NU_SIGMA0" <<'PY' 2>/dev/null
import sys, numpy as np
from pathlib import Path
from bind.inference.stats import NU_CANON
run, table = Path(sys.argv[1]), sys.argv[2]
want_fid = ""
if table:
    with np.load(table) as t:
        want_fid = str(t["source_run"])
for f in ("Cl_kappa.npz", "peak_counts.npz", "nongaussian_stats.npz"):
    p = run / f
    if not p.exists():
        sys.exit(1)
    try:
        with np.load(p) as d:
            for k in d.files:
                _ = d[k].shape
            if f != "Cl_kappa.npz":        # nu-binned products carry provenance
                if str(d["nu_grid"]) != "canon":
                    sys.exit(1)            # written under the old convention
                if want_fid and str(d["nu_fiducial"]) != want_fid:
                    sys.exit(1)            # fiducial sigma0 changed
                # compare the ACTUAL axis, not just the tag: catches any change
                # to NU_CANON itself (e.g. the 21-bin -> 22-centre fix)
                axes = ([d["nu"]] if f.startswith("peak")
                        else [d["pdf_bins"], d["mf_nu"]])
                for a in axes:
                    if a.shape != NU_CANON.shape or not np.allclose(a, NU_CANON):
                        sys.exit(1)
    except Exception:
        sys.exit(1)                        # missing provenance or truncated file
sys.exit(0)
PY
}
if [[ "$FORCE" != "1" ]] && stats_ok; then
  echo "=== $OUT_RUN already has valid stats — skipping (FORCE=1 to redo) ==="; exit 0
fi

SMOOTHING=${SMOOTHING:-2.0}
NGAL=${NGAL:-}
SIGMA_E=${SIGMA_E:-0.26}
STATS_OPT=(--smoothing_arcmin $SMOOTHING --no_halo_scaling --nu_grid canon)
if [[ -n "$NU_SIGMA0" ]]; then
  [[ -f "$NU_SIGMA0" ]] || { echo "ERROR: NU_SIGMA0 table $NU_SIGMA0 not found" >&2; exit 1; }
  # --out_suffix '' keeps the canonical peak_counts.npz name; the nu convention
  # is recorded INSIDE the file (nu_grid/nu_fiducial) rather than in its name
  STATS_OPT+=(--nu_sigma0_from "$NU_SIGMA0" --nu_norm fixed --out_suffix "")
fi
[[ -n "$NGAL" ]] && STATS_OPT+=(--shape_noise_ngal "$NGAL" --sigma_e "$SIGMA_E")
STATS_OPT+=(${STATS_EXTRA:-})

echo "=== n1000 stats | task $IDX -> $OUT_RUN | $(date) ==="
python -u -m bind.cli.lightcone_stats --run_dir "$OUT_RUN" "${STATS_OPT[@]}"
echo "=== stats done for $OUT_RUN | $(date) ==="
