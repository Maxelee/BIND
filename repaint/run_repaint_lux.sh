#!/bin/bash
#SBATCH --job-name=repaint_lux
#SBATCH --output=/mnt/home/mlee1/ceph/logs/repaint_lux_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/repaint_lux_%j.err
#SBATCH --partition=cca
#SBATCH --constraint="cascadelake|skylake"
#SBATCH --nodes=4
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── FIDUCIAL REPAINT, STAGE 4 (CPU/MPI): lux ray-trace + collect ──────────────
# Traces kappa + y + tau from the repainted planes into N_REAL realizations and
# collects them into {kappa,y,tau}_maps.npz in the new tree.
#
# Same lux .ini as run_lightcone_tau.sh (same geometry, same output_planes, same
# projection_direction, RT_randomization=True) with the chunked, claim-protected,
# requeue-safe driver from n1000_body.sh.
#
# WALL CLOCK.  One 50-realization chunk measured 02:25:20 on 4 nodes / 192 ranks
# (sacct -j 2441237; the lux step alone 02:09:46).  The released 550-realization
# trace ran as ONE job with a 3-day limit (sacct -j 2457493: Elapsed 22:39:55), and
# cca's MaxTime is 7 days — so this job asks for 48 h, enough for all 11 chunks of
# an N_REAL=550 trace in a single submission.  Chunking is kept purely for
# preemption safety, not to fit a short wall.  The loop ALSO refuses to start a
# chunk that will not fit in the remaining allocation, so a timeout can never
# discard partial work (the old 5 h wall threw away ~150 core-hr per submission).
#
# CONCURRENCY.  Every chunk is claimed atomically (mkdir) before any core is spent
# and the scratch dir is job-unique, so two jobs (e.g. a cca job and a preempt job)
# can safely run at the same time and will divide the chunks between them.  This
# mirrors n1000_body.sh, which is the model for this driver.
#
# SEED LADDER — DO NOT CHANGE.  Chunk c covers global realizations
# [c*CHUNK+1 .. (c+1)*CHUNK] with base seed RT_SEED + 7*c*CHUNK, so realization r
# always gets seed 1992 + 7*(r-1): exactly the ladder of the released 50- and
# 550-realization traces.  The repaint is therefore seed-PAIRED, realization by
# realization, with the released DMO / truth / twobound traces — which is what
# makes S(ell) = C_l^kappa / C_l^DMO a shared-sky ratio with cancelling cosmic
# variance.  (The n1000 seed gate proved chunking reproduces an unchunked trace
# bit-identically.)
#
# REALIZATION COUNT (explicit decision, env knob N_REAL; default 50):
#   N_REAL=50   gate run, ~2.4 h wall, ~465 core-hr, ~2.9 GB of maps.
#               Enough for every mean-level number in the paper; matches the
#               50-real bind_science/runs/bind/run_0000 sibling and the twobound
#               trio, so all paired comparisons work.
#   N_REAL=550  release parity, ~26 h wall, ~5100 core-hr, ~31 GB of maps + ~115 GB
#               of raw rt_output if KEEP_RAW=1.  Needed only to reproduce the
#               released covariance/per-realization products at full precision.
#   EXTENDING 50 -> 550 IS INCREMENTAL AND WORKS: the completion marker records the
#   count it was written for, so `N_REAL=550 sbatch ...` after a finished
#   N_REAL=50 run traces chunks 1..10 only and re-collects.  Do NOT use FORCE=1 for
#   this (FORCE re-collects; FORCE_CHUNKS=1 would additionally re-trace chunk 0 for
#   nothing, ~465 core-hr).
#
# Submit (author only), after stage 3; resubmit the SAME command to resume:
#     sbatch --dependency=afterok:$JID_PL repaint/run_repaint_lux.sh
#     N_REAL=550 sbatch repaint/run_repaint_lux.sh          # release parity
# Preemptible lane (safe to run concurrently with the above):
#     sbatch --partition=preempt --qos=preempt repaint/run_repaint_lux.sh
#
# Env: see repaint/repaint_env.sh; plus MAX_CHUNKS_PER_JOB (default 0 = as many as
# fit), RELEASE_STALE_CLAIMS (default 0; 1 = drop unfinished claims first).

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=repaint_env.sh
source "$HERE/repaint_env.sh"

RELEASE_STALE_CLAIMS=${RELEASE_STALE_CLAIMS:-0}
JOB_TAG="${SLURM_JOB_ID:-local$$}"

module restore lux                              # openmpi etc. for srun-launched lux
repaint_preamble
repaint_lock

# The seed ladder must not depend on the job environment: GSL_RNG_TYPE would
# change the generator lux seeds per realization and silently break pairing.
unset GSL_RNG_TYPE GSL_RNG_SEED || true

(( N_REAL % CHUNK == 0 )) || { echo "ERROR: N_REAL=$N_REAL not divisible by CHUNK=$CHUNK" >&2; exit 1; }
NCHUNK=$(( N_REAL / CHUNK ))

# ── the plane directory: guarded HARD ─────────────────────────────────────────
# An inherited/exported LENSPLANE_DIR pointing at the RELEASED lensplanes would
# (a) hand lux the released tree as its declared LP_output_dir and (b) — far worse
# for the science — ray-trace the OLD wrong-cosmology planes into the NEW tree,
# passing every downstream gate silently.  So LP_DIR must live inside REPAINT_ROOT,
# must contain no symlink into a released tree, and is recorded so a second job
# cannot trace a different plane set into the same rt_output.
LP_DIR=${LENSPLANE_DIR:-$REPAINT_ROOT/lensplanes}
RT_DIR="$REPAINT_ROOT/rt_output"
CHUNK_STATE="$REPAINT_ROOT/.chunks"          # sentinels live OUTSIDE any scratch
repaint_assert_under_root "$LP_DIR" "lensplane dir" || exit 1
repaint_assert_no_released_symlinks "$LP_DIR" "lensplane dir" || exit 1
repaint_assert_under_root "$RT_DIR" "rt_output" || exit 1
mkdir -p "$RT_DIR" "$CHUNK_STATE"

LP_REC="$REPAINT_ROOT/.lp_dir"
LP_RESOLVED=$(realpath -m "$LP_DIR")
if [[ ! -f "$LP_REC" ]]; then
    echo "$LP_RESOLVED" > "$LP_REC.$$" && mv -n "$LP_REC.$$" "$LP_REC"; rm -f "$LP_REC.$$"
fi
if [[ "$(cat "$LP_REC")" != "$LP_RESOLVED" ]]; then
    echo "ERROR: this job would trace planes from '$LP_RESOLVED' but $REPAINT_ROOT" >&2
    echo "       was already traced from '$(cat "$LP_REC")'. Refusing to mix plane sets." >&2
    exit 1
fi

# ── planes must be complete (and exactly the right size) before lux starts ────
# NB the count MUST NOT be taken with `n=$(ls ... | wc -l)`: under `set -euo
# pipefail` a non-matching glob makes ls exit 2, pipefail propagates it to the
# assignment and set -e kills the job BEFORE the diagnostic prints (verified:
# exit 2, zero output).  A nullglob array has no such failure mode.
shopt -s nullglob
NEED=$(( N_SNAPS * 4 ))
for spec in "lenspot:$LENSPOT_BYTES" "yplane:$YTAU_BYTES" "tauplane:$YTAU_BYTES"; do
    pre=${spec%%:*}; want=${spec##*:}
    files=("$LP_DIR/${pre}"*.dat)
    n=${#files[@]}
    if [[ "$n" -ne "$NEED" ]]; then
        echo "ERROR: $n/$NEED ${pre}*.dat in $LP_DIR — run stage 3 first" >&2; exit 1
    fi
    for f in "${files[@]}"; do
        [[ -L "$f" ]] && { echo "ERROR: $f is a symlink (planes must be real files)" >&2; exit 1; }
        sz=$(stat -c%s "$f")
        [[ "$sz" -eq "$want" ]] || {
            echo "ERROR: $f is $sz B, expected $want B (truncated / wrong LP_GRID?)" >&2; exit 1; }
    done
done
shopt -u nullglob
[[ -f "$LP_DIR/config.dat" && ! -L "$LP_DIR/config.dat" ]] || {
    echo "ERROR: no real config.dat in $LP_DIR" >&2; exit 1; }
echo "=== planes complete: $NEED each of lenspot/yplane/tauplane in $LP_DIR (exact size) ==="

# ── completion marker, KEYED ON THE REALIZATION COUNT ─────────────────────────
# A bare-existence test made the documented 50 -> 550 extension a silent no-op
# that printed "already complete".  The marker now records what it completed.
HAVE=$(cat "$REPAINT_ROOT/.n_real" 2>/dev/null || echo 0)
[[ "$HAVE" =~ ^[0-9]+$ ]] || HAVE=0
if [[ -f "$REPAINT_ROOT/.trace_complete" && "$HAVE" -ge "$N_REAL" && "$FORCE" != "1" ]]; then
    echo "=== trace already complete for $REPAINT_ROOT at n_real=$HAVE >= $N_REAL — nothing to do ==="
    echo "    (raise N_REAL to extend, or FORCE=1 to re-collect the same chunks)"
    exit 0
fi
if [[ -f "$REPAINT_ROOT/.trace_complete" && "$HAVE" -lt "$N_REAL" ]]; then
    echo "=== extending trace: $HAVE -> $N_REAL realizations (finished chunks are kept) ==="
    rm -f "$REPAINT_ROOT/.trace_complete"
fi

if [[ "$RELEASE_STALE_CLAIMS" == "1" ]]; then
    for cl in "$CHUNK_STATE"/chunk_*.claim; do
        [[ -d "$cl" ]] || continue
        [[ -f "${cl%.claim}.done" ]] && continue
        echo "    releasing stale claim $(basename "$cl") (owner $(cat "$cl/owner" 2>/dev/null || echo '?'))"
        rm -rf "$cl"
    done
fi

# ── per-realization completeness (config + 5 planes per enabled field) ────────
expected_files() {
    local d=$1 p
    [[ -f "$d/config.dat" ]] || return 1
    for p in 26 45 59 70 78; do
        [[ -f "$d/kappa${p}.dat" ]] || return 1
        [[ "$COMPUTE_TSZ" == True && ! -f "$d/y${p}.dat" ]] && return 1
        [[ "$COMPUTE_TAU" == True && ! -f "$d/tau${p}.dat" ]] && return 1
    done
    return 0
}

# ── wall-clock budget: never start a chunk that cannot finish ─────────────────
CHUNK_SECONDS=${CHUNK_SECONDS:-9000}          # 2.5 h; measured 02:25:20 per chunk
seconds_left() {
    local now end
    if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
        now=$(date +%s); end=$SLURM_JOB_END_TIME
        echo $(( end - now ))
    else
        echo 999999                            # not under Slurm: no budget
    fi
}

MY_CHUNK_DIRS=()
DONE_THIS_JOB=0
for (( c = 0; c < NCHUNK; c++ )); do
    OFFSET=$(( c * CHUNK ))
    SENTINEL="$CHUNK_STATE/chunk_${CHUNK}_${c}.done"
    CLAIM="$CHUNK_STATE/chunk_${CHUNK}_${c}.claim"
    if [[ -f "$SENTINEL" && "$FORCE_CHUNKS" != "1" ]]; then
        echo "=== chunk $c (reals $((OFFSET + 1))-$((OFFSET + CHUNK))) already done ==="
        continue
    fi
    if (( MAX_CHUNKS_PER_JOB > 0 && DONE_THIS_JOB >= MAX_CHUNKS_PER_JOB )); then
        echo "=== MAX_CHUNKS_PER_JOB=$MAX_CHUNKS_PER_JOB reached — resubmit to continue ==="
        break
    fi
    LEFT=$(seconds_left)
    if (( LEFT < CHUNK_SECONDS )); then
        echo "=== only ${LEFT}s of wall left (< ${CHUNK_SECONDS}s per chunk) — stopping cleanly ==="
        echo "    resubmit the same command; nothing was discarded."
        break
    fi

    # atomic claim: two concurrent jobs (cca + preempt) divide the chunks instead
    # of both running the same one and deleting each other's scratch.
    [[ "$FORCE_CHUNKS" == "1" ]] && rm -rf "$CLAIM"
    if mkdir "$CLAIM" 2>/dev/null; then
        echo "$JOB_TAG" > "$CLAIM/owner"
    elif [[ ! -s "$CLAIM/owner" ]]; then
        echo "$JOB_TAG" > "$CLAIM/owner"; sleep 5
        [[ "$(cat "$CLAIM/owner" 2>/dev/null)" == "$JOB_TAG" ]] || {
            echo "=== chunk $c taken during claim self-heal — skipping ==="; continue; }
        echo "    reclaimed orphaned claim for chunk $c"
    elif [[ "$(cat "$CLAIM/owner" 2>/dev/null)" == "$JOB_TAG" ]]; then
        echo "    resuming own claim for chunk $c (requeue)"
    else
        echo "=== chunk $c claimed by job $(cat "$CLAIM/owner" 2>/dev/null || echo '?')" \
             "— skipping (rm -r $CLAIM to release)"
        continue
    fi

    CHUNK_DIR="$REPAINT_ROOT/_work/rt_chunk${c}_${JOB_TAG}"   # job-unique scratch
    MY_CHUNK_DIRS+=("$CHUNK_DIR")
    rm -rf "$CHUNK_DIR"; mkdir -p "$CHUNK_DIR"
    for i in $(seq 1 "$CHUNK"); do mkdir -p "$CHUNK_DIR/$(printf 'run%03d' "$i")"; done

    INI="$CHUNK_DIR/lux.ini"
    # SAFETY INVARIANT (same as n1000_body.sh:279-283): with
    # simulation_format=PreProjected the current lux build only READS
    # LP_output_dir / tsz_input_dir / tau_input_dir — every write goes to
    # RT_output_dir, because `lenspot Lenspot(...)` is commented out at
    # lux main.cpp:47 (lenspot::write_phi, lenspot.cpp:451, would otherwise
    # fopen("%s/lenspot%02d.dat", LPoutput_dir, "wb")).  RE-VERIFY IF LUX IS
    # EVER REBUILT — the n1000 seed gate doubles as that regression check.
    cat > "$INI" <<EOF
LP_output_dir = $LP_DIR
RT_output_dir = $CHUNK_DIR
tsz_input_dir = $LP_DIR
tau_input_dir = $LP_DIR
input_dir = $LP_DIR
LP_grid = $LP_GRID
RT_grid = $RT_GRID
planes_per_snapshot = 4
angle = $FOV_DEG
simulation_format = PreProjected
snapshot_list = 96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29
snapshot_stack = false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false
output_planes = 26, 45, 59, 70, 78
projection_direction = -1
translation_rotation = False
RT_random_seed = $(( RT_SEED + 7 * OFFSET ))
RT_randomization = True
compute_tsz = $COMPUTE_TSZ
compute_tau = $COMPUTE_TAU
n_realizations = $CHUNK
verbose = True
EOF
    echo "=== lux chunk $c: reals $((OFFSET + 1))-$((OFFSET + CHUNK)), base seed "\
"$(( RT_SEED + 7 * OFFSET )) | job=$JOB_TAG nodes=${SLURM_NNODES:-?} tasks=${SLURM_NTASKS:-?} | $(date) ==="
    srun -n "${SLURM_NTASKS}" "$LUX_BIN" "$INI"

    for i in $(seq 1 "$CHUNK"); do
        SRC="$CHUNK_DIR/$(printf 'run%03d' "$i")"
        DST="$RT_DIR/$(printf 'run%04d' $((OFFSET + i)))"
        expected_files "$SRC" || {
            echo "ERROR: chunk $c real $i incomplete — releasing claim" >&2
            rm -rf "$CLAIM"; exit 1; }
        rm -rf "$DST"          # leftover from an interrupted earlier attempt
        mv "$SRC" "$DST"
    done
    rm -rf "$CHUNK_DIR"
    touch "$SENTINEL"
    rm -rf "$CLAIM"
    DONE_THIS_JOB=$(( DONE_THIS_JOB + 1 ))
done

# ── all chunks present? ───────────────────────────────────────────────────────
MISSING=0
for (( c = 0; c < NCHUNK; c++ )); do
    [[ -f "$CHUNK_STATE/chunk_${CHUNK}_${c}.done" ]] || MISSING=$(( MISSING + 1 ))
done
if (( MISSING > 0 )); then
    echo "=== $MISSING/$NCHUNK chunks still to trace — resubmit the same command ==="
    for d in "${MY_CHUNK_DIRS[@]:-}"; do [[ -n "$d" ]] && rm -rf "$d"; done
    exit 0
fi

# ── collect -> {kappa,y,tau}_maps.npz ─────────────────────────────────────────
maps_complete() {
    python - "$REPAINT_ROOT" "$N_REAL" "$COMPUTE_TSZ" "$COMPUTE_TAU" <<'PY'
import sys
import numpy as np
out, n, tsz, tau = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "True", sys.argv[4] == "True"
want = [("kappa_maps.npz", "kappa")] + ([("y_maps.npz", "y")] if tsz else []) \
       + ([("tau_maps.npz", "tau")] if tau else [])
for fname, key in want:
    try:
        with np.load(f"{out}/{fname}") as f:
            got = f[key].shape[0]
    except Exception:
        sys.exit(1)
    if got != n:
        print(f"[verify] {fname}: {got} reals != {n}", file=sys.stderr); sys.exit(1)
    print(f"[verify] {fname} n_real={got} OK")
sys.exit(0)
PY
}

if maps_complete && [[ "$FORCE" != "1" ]]; then
    echo "=== maps already collected at n_real=$N_REAL — skipping collect ==="
else
    echo "=== collect $N_REAL realizations -> $REPAINT_ROOT | $(date) ==="
    COLLECT_OPT=()
    # KEEP_RAW=1 (default) keeps rt_output/*.dat (~115 GB at 550 reals) so a later
    # extension of N_REAL, or a re-collect, costs nothing.  Set KEEP_RAW=0 to
    # delete the raw planes as they are collected.
    [[ "$KEEP_RAW" == "0" ]] && COLLECT_OPT+=(--delete_raw)
    srun -N1 -n1 python -u -m bind.cli.lux_collect \
        --rt_root "$RT_DIR" --output_dir "$REPAINT_ROOT" \
        --n_real "$N_REAL" --fov_deg "$FOV_DEG" "${COLLECT_OPT[@]}"
    maps_complete || { echo "ERROR: collected maps failed verification" >&2; exit 1; }
fi

# remove only THIS job's scratch: another job may be mid-chunk in _work/
for d in "${MY_CHUNK_DIRS[@]:-}"; do [[ -n "$d" ]] && rm -rf "$d"; done
rmdir "$REPAINT_ROOT/_work" 2>/dev/null || true      # only if now empty
echo "$N_REAL" > "$REPAINT_ROOT/.n_real"
touch "$REPAINT_ROOT/.trace_complete"
echo "=== trace complete ($N_REAL reals) | $(date) ==="
ls -la "$REPAINT_ROOT"/{kappa,y,tau}_maps.npz
echo "    Next: sbatch repaint/run_repaint_stats.sh"
