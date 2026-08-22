#!/bin/bash
# ── Shared configuration + no-overwrite guard for the fiducial repaint ────────
# Sourced by every repaint/run_repaint_*.sh.  Nothing here submits anything.
#
# THE CAMPAIGN IN ONE LINE: repaint the fiducial lightcone with the CORRECT
# TNG300 cosmology (the released tree was conditioned on the CAMELS SB35
# cosmology: Ob/Om 1.038141x too high -> +7.7% gas/tau plane power), writing
# every product into a FRESH tree.  The released trees are read-only inputs.
#
# Env overrides (all optional; defaults shown):
#   REPAINT_ROOT   /mnt/home/mlee1/ceph/bind_lightcone_tng_fixed   new tree
#   LC_OLD         /mnt/home/mlee1/ceph/bind_lightcone_tng          released tree (RO)
#   N_REAL          50     realizations to trace  (50 = gate run AND the count the
#                          released 50-real sibling runs/bind/run_0000 uses;
#                          550 = release parity — an explicit, expensive choice)
#   CHUNK          50      realizations per lux chunk (N_REAL must be divisible by it)
#   RT_SEED        1992    base of the canonical seed ladder — NEVER change
#   LP_GRID 4096 | RT_GRID 1024 | FOV_DEG 5.0 | N_SNAPS 20
#   RUN_DIR        weights/fm_redshift_thermo    checkpoint used for the released paint
#   N_STEPS 50 | BATCH_SIZE 16                   sampler settings of the released paint
#   KEEP_RAW       1       keep rt_output/*.dat (needed to extend 50 -> 550 cheaply)
#   FORCE          0       1 = redo a stage's FINAL product even if its marker exists
#   FORCE_CHUNKS   0       1 = ALSO discard completed lux chunk sentinels (expensive:
#                          re-traces ~2.4 h x 4 nodes per chunk).  FORCE alone never
#                          throws a finished chunk away.
#   NO_AMP         0       1 = paint in fp32 (--no_amp).  REQUIRED on the V100S
#                          fallback nodes: paint.py:389 hard-codes bf16 autocast and
#                          sm_70 has no hardware bf16.
#   MAX_CHUNKS_PER_JOB 0   0 = as many chunks as the wall clock allows (the lux stage
#                          also stops on its own before starting a chunk that will
#                          not fit in the remaining allocation)
#
# COST (measured, not estimated — see docs/fiducial_repaint_plan.md §4):
#   generate  ~4.0-4.7 GPU-hr total.  The gpu QOS caps the user at cpu=40,gres/gpu=4,
#             so with --cpus-per-task=8 only 4 array tasks run at once => ~1-2.5 h wall,
#             NOT the ~25 min a fully parallel 20-way array would take.
#   lux trace  one 50-realization chunk = 02:25:20 wall on 4 nodes / 192 ranks
#             (measured, sacct -j 2441237; lux step itself 02:09:46), i.e. ~465 core-hr.
#             N_REAL=50  -> 1 chunk  ~2.4 h  / ~465 core-hr
#             N_REAL=550 -> 11 chunks ~26 h  / ~5100 core-hr.  The released 550-real
#             trace ran as ONE job (sacct -j 2457493: Timelimit 3-00:00:00,
#             Elapsed 22:39:55, 4 nodes) — hence the 48 h wall on the repaint lux stage.

# ── paths ─────────────────────────────────────────────────────────────────────
REPAINT_ROOT=${REPAINT_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng_fixed}
LC_OLD=${LC_OLD:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
PARAMS_FILE=${PARAMS_FILE:-$REPAINT_ROOT/params.npy}
BIND_REPO=${BIND_REPO:-/mnt/home/mlee1/BIND}
VENV=${VENV:-/mnt/home/mlee1/venvs/BIND_env/bin/activate}
LOGDIR=${LOGDIR:-/mnt/home/mlee1/ceph/logs}

# ── campaign knobs ────────────────────────────────────────────────────────────
N_SNAPS=${N_SNAPS:-20}
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)
RUN_DIR=${RUN_DIR:-weights/fm_redshift_thermo}
N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}
R200_FACTOR=${R200_FACTOR:-4.0}      # circular paste, matches the released tree
TAPER_FRAC=${TAPER_FRAC:-0.15}
LP_GRID=${LP_GRID:-4096}
RT_GRID=${RT_GRID:-1024}
FOV_DEG=${FOV_DEG:-5.0}
RT_SEED=${RT_SEED:-1992}             # seed ladder 1992+7r: keeps the repaint
                                     # realization-by-realization PAIRED with the
                                     # released DMO / truth / twobound traces
N_REAL=${N_REAL:-50}                 # the runbook's recommended first (and often only)
                                     # value; 550 is release parity, ~11x the cost
CHUNK=${CHUNK:-50}
COMPUTE_TSZ=${COMPUTE_TSZ:-True}
COMPUTE_TAU=${COMPUTE_TAU:-True}
KEEP_RAW=${KEEP_RAW:-1}
LUX_BIN=${LUX_BIN:-/mnt/home/mlee1/lux/lux}
FORCE=${FORCE:-0}                    # redo a stage's final product
FORCE_CHUNKS=${FORCE_CHUNKS:-0}      # ALSO discard finished lux chunks (rarely wanted)
NO_AMP=${NO_AMP:-0}                  # 1 = fp32 paint (mandatory on V100S, see above)
MAX_CHUNKS_PER_JOB=${MAX_CHUNKS_PER_JOB:-0}

# Exact on-disk sizes of a finished lux plane at LP_GRID (int32 N + float64 payload
# + int32 N).  Used instead of `-s` so a plane truncated by a killed job is never
# mistaken for a finished one:  verified against the released tree
# (lenspot 671088648 B, yplane/tauplane 134217736 B at LP_GRID=4096).
LENSPOT_BYTES=$(( 8 + LP_GRID * LP_GRID * 5 * 8 ))
YTAU_BYTES=$(( 8 + LP_GRID * LP_GRID * 8 ))

# ── the no-overwrite policy, in code ──────────────────────────────────────────
# /mnt/home/mlee1/ceph is a symlink to /mnt/sdceph/users/mlee1, so both sides are
# resolved before comparison; the trailing-separator test keeps
# ".../bind_lightcone_tng_fixed" from matching ".../bind_lightcone_tng".
REPAINT_RELEASED_TREES=(
    /mnt/home/mlee1/ceph/bind_lightcone_tng
    /mnt/home/mlee1/ceph/bind_science
    /mnt/home/mlee1/ceph/bind_sb35
    /mnt/home/mlee1/ceph/bind_n1000
    /mnt/home/mlee1/ceph/tng_full_validation
)

repaint_guard() {   # repaint_guard <path> [<description>]
    local p="$1" what="${2:-output}" t rt form
    # A relative path is resolved against the CURRENT cwd, and repaint_preamble
    # cd's into $BIND_REPO partway through — so "validated here, used there" is a
    # real hazard.  Absolute paths only.
    if [[ "$p" != /* ]]; then
        echo "REFUSING TO RUN: $what path '$p' is not absolute." >&2
        echo "  Relative paths resolve against whatever cwd the job happens to have." >&2
        return 1
    fi
    # Resolve BOTH ways and refuse if EITHER form lands in a released tree:
    #   -m            follows symlinks (/mnt/home/mlee1/ceph -> /mnt/sdceph/users/mlee1)
    #   -m --no-symlinks  collapses ".." textually, so a path that traverses out of
    #                     and back into a released tree cannot hide behind the alias.
    for form in "$(realpath -m "$p")" "$(realpath -m --no-symlinks "$p")"; do
        for t in "${REPAINT_RELEASED_TREES[@]}"; do
            for rt in "$(realpath -m "$t")" "$(realpath -m --no-symlinks "$t")"; do
                if [[ "$form" == "$rt" || "$form" == "$rt"/* ]]; then
                    echo "REFUSING TO RUN: $what path '$p' resolves ($form) inside the released tree '$t'." >&2
                    echo "  The repaint is strictly additive. Set REPAINT_ROOT to a fresh tree." >&2
                    return 1
                fi
            done
        done
    done
    return 0
}

repaint_assert_under_root() {   # repaint_assert_under_root <path> <description>
    # Stronger than repaint_guard: the path must live INSIDE $REPAINT_ROOT.  Used
    # for every directory a stage writes to, so that an inherited/exported
    # LENSPLANE_DIR (say) cannot silently redirect a stage at another tree — the
    # failure mode that would otherwise ray-trace the OLD planes into the NEW tree
    # and pass every downstream gate.
    local p="$1" what="${2:-path}" rp rr
    repaint_guard "$p" "$what" || return 1
    rp=$(realpath -m "$p"); rr=$(realpath -m "$REPAINT_ROOT")
    if [[ "$rp" != "$rr" && "$rp" != "$rr"/* ]]; then
        echo "REFUSING TO RUN: $what '$p' ($rp) is not inside REPAINT_ROOT ($rr)." >&2
        echo "  Unset it, or point REPAINT_ROOT at the tree you meant." >&2
        return 1
    fi
    return 0
}

repaint_assert_no_released_symlinks() {   # <dir> [<description>]
    # Refuse if ANY entry in <dir> is a symlink whose target lives in a released
    # tree.  Writing through such a link truncates the RELEASED file:
    # bind.inference.lensplane writes with a plain open(path,"wb") (lines 193/214/401),
    # which follows symlinks, and lux's own writers use fopen(...,"wb").  The
    # directory itself passes repaint_guard, so only a per-file check catches this.
    local d="$1" what="${2:-directory}" f tgt t rt bad=0
    [[ -d "$d" ]] || return 0
    for f in "$d"/*; do
        [[ -e "$f" || -L "$f" ]] || continue          # unmatched glob
        [[ -L "$f" ]] || continue
        tgt=$(realpath -m "$f")
        for t in "${REPAINT_RELEASED_TREES[@]}"; do
            rt=$(realpath -m "$t")
            if [[ "$tgt" == "$rt" || "$tgt" == "$rt"/* ]]; then
                echo "REFUSING TO RUN: $what contains a symlink into a released tree:" >&2
                echo "    $f -> $tgt" >&2
                bad=1
            fi
        done
    done
    if (( bad )); then
        echo "  A later write through that link would DESTROY the released file." >&2
        echo "  Remove the links (they are never needed: every plane is repainted)." >&2
        return 1
    fi
    return 0
}

repaint_preamble() {   # guard + env + log dir; call at the top of every stage
    local v
    for v in REPAINT_ROOT PARAMS_FILE LC_OLD BIND_REPO; do
        [[ "${!v}" == /* ]] || {
            echo "ERROR: $v must be an ABSOLUTE path (got '${!v}')." >&2; exit 1; }
    done
    repaint_guard "$REPAINT_ROOT" "REPAINT_ROOT" || exit 1
    repaint_guard "$PARAMS_FILE"  "PARAMS_FILE"  || exit 1
    [[ -d "$LC_OLD" ]] || { echo "ERROR: released tree not found: $LC_OLD" >&2; exit 1; }
    # shellcheck disable=SC1090
    source "$VENV"
    cd "$BIND_REPO"
    mkdir -p "$LOGDIR" "$REPAINT_ROOT"
    echo "=== repaint | REPAINT_ROOT=$REPAINT_ROOT | LC_OLD=$LC_OLD (read-only) ==="
    echo "    N_REAL=$N_REAL CHUNK=$CHUNK RT_SEED=$RT_SEED RUN_DIR=$RUN_DIR"
    echo "    N_STEPS=$N_STEPS BATCH_SIZE=$BATCH_SIZE R200_FACTOR=$R200_FACTOR FORCE=$FORCE"
}

# ── campaign-parameter lock (same idea as n1000_body.sh) ──────────────────────
# The first stage to run records the resolved knobs; every later stage must
# match, so a half-repaint can never mix two seed ladders or two grids.
repaint_lock() {
    local line="RT_SEED=$RT_SEED LP_GRID=$LP_GRID RT_GRID=$RT_GRID FOV_DEG=$FOV_DEG \
RUN_DIR=$RUN_DIR N_STEPS=$N_STEPS BATCH_SIZE=$BATCH_SIZE R200_FACTOR=$R200_FACTOR \
TAPER_FRAC=$TAPER_FRAC"
    mkdir -p "$REPAINT_ROOT"
    if [[ ! -f "$REPAINT_ROOT/.campaign_params" ]]; then
        echo "$line" > "$REPAINT_ROOT/.campaign_params.$$" \
            && mv -n "$REPAINT_ROOT/.campaign_params.$$" "$REPAINT_ROOT/.campaign_params"
        rm -f "$REPAINT_ROOT/.campaign_params.$$"
    fi
    if [[ "$(cat "$REPAINT_ROOT/.campaign_params")" != "$line" ]]; then
        echo "ERROR: campaign parameter mismatch." >&2
        echo "  this job : $line" >&2
        echo "  recorded : $(cat "$REPAINT_ROOT/.campaign_params")" >&2
        exit 1
    fi
}
# NB N_REAL/CHUNK are deliberately NOT in the lock: extending 50 -> 550 later is
# an intended workflow (the chunked trace reuses the first 50 as chunk 0).  The
# realization count is instead recorded in $REPAINT_ROOT/.n_real and READ BACK by
# both the lux and the stats stage, so a 550-real tree can never wear 50-real
# statistics (or vice versa).
#
# NB GENERATE_R200 is deliberately NOT in the lock either.  It is a stage-2-only
# knob with two legitimate settings (0.0 + recomposite, or 4.0 one-step) that must
# converge on the SAME product, and stages 3/4/5 never define it — putting it in
# the shared lock line would make every later stage's lock mismatch.  The invariant
# that actually matters (what taper the composites on disk were built with) is
# recorded in each snap_*/summary.json and is hard-asserted by run_repaint_planes.sh
# before a single plane is written.
