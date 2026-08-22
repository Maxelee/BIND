#!/bin/bash
# ── N1000 stats sweep from an INTERACTIVE allocation ──────────────────────────
# Runs bind.cli.lightcone_stats_mpi over every FINISHED run in the N1000 tree,
# one run at a time, each using all ranks of the current allocation.  Products
# (Cl_kappa / Cl_kappa_y / Cl_tau / dm_stats / peak_counts / nongaussian_stats)
# are written straight into the run directory on ceph — no extra copy step.
#
#   salloc -p cca -C icelake -N1 -n100 --exclusive -t 12:00:00
#   ./n1000_stats_sweep.sh                  # every traced run missing valid stats
#   ./n1000_stats_sweep.sh 0 1 2 21         # only these manifest indices
#   FORCE=1 ./n1000_stats_sweep.sh 0 1 2 21 # recompute even if valid
#   DRY=1 ./n1000_stats_sweep.sh            # list what WOULD run, then exit
#
# SPEC=1 prints the same work list as a compact sbatch --array spec instead of
# running anything (no allocation needed), so the SLURM lane submits exactly the
# runs that need work.  Both QOSs cap SUBMITTED tasks (pending + running) at 500
# per user and every array ELEMENT counts, so an --array=0-318 is refused while
# the trace lanes hold hundreds of pending elements; MAX=<n> truncates the spec
# to the slots you actually have free (check with
# `squeue -u $USER -r -h -o "%q|%t" | sort | uniq -c`):
#   sbatch --array=$(SPEC=1 MAX=180 ./n1000_stats_sweep.sh) \
#          --partition=preempt --qos=preempt --requeue --open-mode=append \
#          --constraint="icelake|cascadelake|skylake" run_n1000_stats_mpi.sh
#
# Manifest indices match n1000_body.sh / run_n1000_stats_mpi.sh:
#   0 bind | 1 dmo | 2 truth | 3..62 twobound run_0000..0059 | 63..318 sb35 run_0000..0255
#
# The sweep is idempotent and interruptible: a run whose products already load,
# carry the canonical nu axis and match the fiducial sigma0 table is skipped, so
# re-running after a Ctrl-C (or as more traces land) only does what is left.
# For the whole tree at once prefer the SLURM array instead — same work, many
# nodes in parallel:  sbatch run_n1000_stats_mpi.sh
#
# Env: OUT_ROOT, NU_SIGMA0, SMOOTHING, NGAL, SIGMA_E, N_REAL (cap: debug only),
#      SCRATCH, NTASKS, FORCE, DRY.

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND

# one thread per rank: N ranks x multithreaded BLAS/Pylians oversubscribes the node
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BIND_PK_THREADS=${BIND_PK_THREADS:-1}

OUT_ROOT=${OUT_ROOT:-/mnt/home/mlee1/ceph/bind_n1000}
NU_SIGMA0=${NU_SIGMA0:-$OUT_ROOT/analysis/nu_sigma0_bind.npz}
SMOOTHING=${SMOOTHING:-2.0}
NGAL=${NGAL:-}
SIGMA_E=${SIGMA_E:-0.26}
N_REAL=${N_REAL:-}
FORCE=${FORCE:-0}
DRY=${DRY:-0}
SPEC=${SPEC:-0}
MAX=${MAX:-0}                     # SPEC: keep at most this many indices (0 = all)
NTASKS=${NTASKS:-${SLURM_NTASKS:-100}}
# per-invocation scratch: the CLI names its staging dir after run_dir.name, and
# twobound/run_0000 and sb35/run_0000 share that name — two concurrent sweeps on
# one node would otherwise collide
SCRATCH=${SCRATCH:-/tmp/n1000_stats_${SLURM_JOB_ID:-$$}}
LOGDIR="$OUT_ROOT/analysis/stats_logs"

[[ -f "$NU_SIGMA0" ]] || { echo "ERROR: sigma0 table $NU_SIGMA0 not found" >&2; exit 1; }
# SPEC/DRY only read the tree — they are useful from a login shell
if [[ "$SPEC" != "1" && "$DRY" != "1" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: no allocation — run this inside salloc/srun" >&2; exit 1; }
  mkdir -p "$LOGDIR" "$SCRATCH"
fi

run_dir_for() {   # manifest index -> run directory
  local i=$1
  if   (( i == 0 )); then echo "$OUT_ROOT/bind/run_0000"
  elif (( i == 1 )); then echo "$OUT_ROOT/dmo/run_0000"
  elif (( i == 2 )); then echo "$OUT_ROOT/truth/run_0000"
  elif (( i <= 62 )); then printf '%s/twobound/run_%04d\n' "$OUT_ROOT" $((i - 3))
  elif (( i <= 318 )); then printf '%s/sb35/run_%04d\n' "$OUT_ROOT" $((i - 63))
  else echo "ERROR: index $i out of range (0..318)" >&2; return 1
  fi
}

# ── build the work list ───────────────────────────────────────────────────────
# Traced (.trace_complete) and not already carrying valid products.  "Valid"
# means the npz actually LOADS and carries the right nu provenance: a job killed
# mid-savez leaves a truncated file that an existence check would treat as
# finished forever.  Same guard as run_n1000_stats{,_mpi}.sh — but the whole
# scan runs in ONE interpreter (319 python startups took ~15 min).
IDXS=("$@")
(( ${#IDXS[@]} )) || IDXS=($(seq 0 318))
mapfile -t TODO < <(python - "$OUT_ROOT" "$NU_SIGMA0" "$FORCE" "${IDXS[@]}" <<'PY'
import sys, numpy as np
from pathlib import Path
from bind.inference.stats import NU_CANON
root, table, force = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
with np.load(table) as t:
    want_fid = str(t["source_run"])
    # The VALUES, not just the path: regenerating nu_sigma0_bind.npz in place
    # (say from 1000 reals instead of 50) leaves source_run identical while the
    # sigma0 every product normalised nu by has moved.  sigma_unsmoothed is
    # scale-independent, so one comparison covers any --smoothing_arcmin.
    want_sig0u = np.asarray(t["sigma_unsmoothed"], float).ravel()

def run_dir(i):
    if i == 0: return root / "bind/run_0000"
    if i == 1: return root / "dmo/run_0000"
    if i == 2: return root / "truth/run_0000"
    if i <= 62: return root / f"twobound/run_{i - 3:04d}"
    return root / f"sb35/run_{i - 63:04d}"

def stats_ok(run):
    for f in ("Cl_kappa.npz", "peak_counts.npz", "nongaussian_stats.npz"):
        p = run / f
        if not p.exists():
            return False
        try:
            with np.load(p) as d:
                for k in d.files:
                    _ = d[k].shape
                if f != "Cl_kappa.npz":
                    if str(d["nu_grid"]) != "canon" or str(d["nu_fiducial"]) != want_fid:
                        return False
                    axes = [d["nu"]] if f.startswith("peak") else [d["pdf_bins"], d["mf_nu"]]
                    for a in axes:
                        if a.shape != NU_CANON.shape or not np.allclose(a, NU_CANON):
                            return False
                    got = np.asarray(d["nu_sigma0_unsmoothed_table"], float).ravel()
                    if got.size and (got.shape != want_sig0u.shape
                                     or not np.allclose(got, want_sig0u, rtol=1e-9)):
                        return False              # table regenerated since this run
        except Exception:
            return False
    return True

for i in (int(x) for x in sys.argv[4:]):
    if not 0 <= i <= 318:
        print(f"index {i} out of range (0..318)", file=sys.stderr)
        sys.exit(1)
    rd = run_dir(i)
    if not (rd / ".trace_complete").exists():
        continue
    if force or not stats_ok(rd):
        print(i)
PY
)
if [[ "$SPEC" == "1" ]]; then
  # compact "a,b-c,d" for sbatch --array, truncated to MAX entries
  (( ${#TODO[@]} )) || { echo "ERROR: nothing to do — no array to submit" >&2; exit 1; }
  (( MAX > 0 && ${#TODO[@]} > MAX )) && TODO=("${TODO[@]:0:$MAX}")
  printf '%s\n' "${TODO[@]}" | awk '
    NR == 1 { s = p = $1; next }
    $1 == p + 1 { p = $1; next }
    { printf "%s%s", (o++ ? "," : ""), (s == p ? s : s "-" p); s = p = $1 }
    END { printf "%s%s\n", (o++ ? "," : ""), (s == p ? s : s "-" p) }'
  exit 0
fi

echo "=== n1000 stats sweep | ${#TODO[@]} run(s) to do | ranks=$NTASKS | $(date) ==="
(( ${#TODO[@]} )) || { echo "nothing to do"; exit 0; }
for I in "${TODO[@]}"; do echo "    task $I -> $(run_dir_for "$I")"; done
[[ "$DRY" == "1" ]] && { echo "(DRY=1 — nothing run)"; exit 0; }

# staging needs ~3x the realization cube uncompressed (~65 GB at N_REAL=1000)
AVAIL_GB=$(df -BG --output=avail "$SCRATCH" | tail -1 | tr -dcs '0-9' ' ')
(( AVAIL_GB >= 80 )) || echo "WARNING: only ${AVAIL_GB}G free on $SCRATCH (staging wants ~65G)"

# ── sweep ─────────────────────────────────────────────────────────────────────
OK=0; FAILED=()
for I in "${TODO[@]}"; do
  RD=$(run_dir_for "$I")
  TAG=$(basename "$(dirname "$RD")")_$(basename "$RD")
  LOG="$LOGDIR/$TAG.log"
  OPT=(--run_dir "$RD" --smoothing_arcmin $SMOOTHING --nu_grid canon
       --nu_sigma0_from "$NU_SIGMA0" --nu_norm fixed --out_suffix ""
       --scratch "$SCRATCH")
  [[ -n "$NGAL"   ]] && OPT+=(--shape_noise_ngal "$NGAL" --sigma_e "$SIGMA_E")
  [[ -n "$N_REAL" ]] && OPT+=(--n_real "$N_REAL")   # DEBUG ONLY: overwrites the
                                                    # 1000-real products with a cap
  echo "--- [$((OK + ${#FAILED[@]} + 1))/${#TODO[@]}] task $I -> $RD | $(date +%T) ---"
  if srun -n "$NTASKS" python -u -m bind.cli.lightcone_stats_mpi "${OPT[@]}" 2>&1 | tee "$LOG"; then
    OK=$((OK + 1))
  else
    echo "!!! task $I FAILED (see $LOG)"
    FAILED+=("$I")
  fi
  rm -rf "$SCRATCH"/n1000stats_* 2>/dev/null || true   # belt-and-braces
done

rm -rf "$SCRATCH"
echo "=== sweep done | $OK ok, ${#FAILED[@]} failed | $(date) ==="
(( ${#FAILED[@]} == 0 )) || { echo "failed indices: ${FAILED[*]}"; exit 1; }
