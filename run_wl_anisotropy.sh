#!/bin/bash
# ── WL-anisotropy: submit the full spherical-BCM decomposition ────────────────
# Orchestrator (run on the LOGIN node, NOT via sbatch). State-aware / idempotent:
# it inspects what is already on disk and submits only the missing stage, so it is
# safe to re-run after a partial/failed batch.  Per run it classifies:
#   COMPLETE    maps + stats present for bind/sph/mono/dmo   -> skip (unless FORCE=1)
#   NEEDS_MAPS  slab tree complete, maps missing/partial     -> submit maps only
#   NEEDS_ALL   slab tree incomplete                         -> slabs[array] -> maps
#
# Dependencies are wired so nothing starts before its inputs exist:
#   fid:  slabs --afterok--> maps  (builds bind/sph/mono/dmo + the shared fid/dmo
#                                   kappa every other run reuses)
#   tbNN: maps needs its own slabs AND the shared fid/dmo kappa.  If fid/dmo is
#         already on disk, tb maps run with no fid dependency; if fid is (re)built
#         in THIS invocation, tb maps wait on the fid maps job.
#
# Each maps job writes {bind,sph,mono,dmo}/{Cl_kappa,peak_counts_nufid,
# nongaussian_stats}.npz under $OUT_BASE/<run>/.  Decompose in
# examples/wl_anisotropy_paper.ipynb:
#   dX_aniso = X_bind - X_sph     (feedback anisotropy a radial BCM misses)
#   X_sph   - X_mono              (triaxiality-tracing a radial BCM reproduces)
#
# Usage:
#   ./run_wl_anisotropy.sh                 # fid + all 60 twobound, missing stages only
#   DO_TB=0 ./run_wl_anisotropy.sh         # fid only
#   DO_FID=0 ./run_wl_anisotropy.sh        # twobound only (reuse existing fid/dmo)
#   TB_RUNS="tb0000 tb0014 tb0042" ./run_wl_anisotropy.sh    # a subset
#   FORCE=1 ./run_wl_anisotropy.sh         # re-run even COMPLETE runs (full chain)
#   DRY_RUN=1 ./run_wl_anisotropy.sh       # print the plan, submit nothing
#
# Env overrides also forwarded to the jobs: OUT_BASE, STAGE1_ROOT, SCI, NPIX,
# N_REAL, FOV_DEG, SRCS_Z, SMOOTH_ARCMIN, NGAL, R200_FACTOR, TAPER_FRAC, SEED0.

set -euo pipefail
cd /mnt/home/mlee1/BIND

DO_FID=${DO_FID:-1}
DO_TB=${DO_TB:-1}
TB_RUNS=${TB_RUNS:-$(seq -f 'tb%04g' 0 59)}           # tb0000 .. tb0059
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}

SCI=${SCI:-/mnt/home/mlee1/ceph/bind_science}
OUT_BASE=${OUT_BASE:-$SCI/wl_anisotropy}
DMO_SHARED="$OUT_BASE/fid/dmo/kappa_maps.npz"         # built by the fid maps job

# classify <run> -> prints COMPLETE | NEEDS_MAPS | NEEDS_ALL  (reads only disk)
classify() {
    python - "$OUT_BASE" "$1" <<'PY'
import sys, numpy as np
from pathlib import Path
OB, run = Path(sys.argv[1]), sys.argv[2]
SNAPS=[96,90,85,80,76,71,67,63,59,56,52,49,46,43,41,38,35,33,31,29]
STATS=["Cl_kappa","peak_counts_nufid","nongaussian_stats"]
FIELDS=["bind","sph","mono","dmo"]
def maps_done():
    for fld in FIELDS:
        d=OB/run/fld
        if not (d/"kappa_maps.npz").exists(): return False
        if any(not (d/f"{s}.npz").exists() for s in STATS): return False
    return True
def slabs_ok():
    sr=OB/run/"slabs"
    if not sr.exists(): return False
    for sn in SNAPS:
        if not list((sr/f"snap_{sn:03d}").glob("composite_slab*.npz")): return False
    try:
        d=np.load(sorted((sr/"snap_096").glob("composite_slab*.npz"))[0])
        if "composite_mono" not in d.files: return False   # built by the OLD code
    except Exception:
        return False
    return True
print("COMPLETE" if maps_done() else "NEEDS_MAPS" if slabs_ok() else "NEEDS_ALL")
PY
}

# submit <RUN> <extra sbatch args...> <script> -> echoes job id (or DRY)
submit() {
    local run="$1"; shift
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry] RUN=$run sbatch --parsable $*" >&2
        echo "DRY"
    else
        RUN="$run" sbatch --parsable "$@"
    fi
}

echo "=== WL-anisotropy submission (DO_FID=$DO_FID DO_TB=$DO_TB FORCE=$FORCE DRY_RUN=$DRY_RUN) ==="
echo "    OUT_BASE=$OUT_BASE"

# ── fid ───────────────────────────────────────────────────────────────────────
FID_MAPS=""                          # set only if we (re)build fid in this run
if [[ "$DO_FID" == "1" ]]; then
    st=$(classify fid)
    [[ "$FORCE" == "1" ]] && st="NEEDS_ALL"
    echo "--- fid [$st] ---"
    case "$st" in
        COMPLETE)   echo "    skip (maps present)";;
        NEEDS_MAPS) FID_MAPS=$(submit fid run_spherical_maps.sh)
                    echo "    fid maps  : $FID_MAPS (slabs already on disk)";;
        NEEDS_ALL)  fid_slabs=$(submit fid run_spherical_slabs.sh)
                    FID_MAPS=$(submit fid --dependency="afterok:$fid_slabs" run_spherical_maps.sh)
                    echo "    fid slabs : $fid_slabs ; fid maps : $FID_MAPS";;
    esac
fi

# tb maps need the shared fid/dmo kappa: present on disk, or produced by FID_MAPS.
if [[ "$DO_TB" == "1" && -z "$FID_MAPS" && "$DRY_RUN" != "1" && ! -f "$DMO_SHARED" ]]; then
    echo "ERROR: twobound needs the shared DMO kappa but it is missing: $DMO_SHARED" >&2
    echo "       build fid first (DO_TB=0 ./run_wl_anisotropy.sh)." >&2
    exit 1
fi

# ── twobound ──────────────────────────────────────────────────────────────────
if [[ "$DO_TB" == "1" ]]; then
    n_skip=0 n_maps=0 n_all=0
    for tb in $TB_RUNS; do
        st=$(classify "$tb")
        [[ "$FORCE" == "1" ]] && st="NEEDS_ALL"
        case "$st" in
            COMPLETE)
                n_skip=$((n_skip+1));;
            NEEDS_MAPS)
                dep=""; [[ -n "$FID_MAPS" ]] && dep="--dependency=afterok:$FID_MAPS"
                m=$(submit "$tb" $dep run_spherical_maps.sh)
                echo "    $tb [maps]  : $m ${dep:+($dep)}"; n_maps=$((n_maps+1));;
            NEEDS_ALL)
                s=$(submit "$tb" run_spherical_slabs.sh)
                dep="afterok:$s"; [[ -n "$FID_MAPS" ]] && dep="$dep:$FID_MAPS"
                m=$(submit "$tb" --dependency="$dep" run_spherical_maps.sh)
                echo "    $tb [all]   : slabs=$s maps=$m (dep=$dep)"; n_all=$((n_all+1));;
        esac
    done
    echo "--- twobound: $n_all full, $n_maps maps-only, $n_skip already complete ---"
fi

echo "=== done. Track with:  squeue -u \$USER ==="
